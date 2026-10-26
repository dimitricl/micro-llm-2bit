"""G0 — construit le banc QA-passage gelé (test/dev) à partir de textes
propres, sans jamais modifier la sonde ni la val gelée.

- in-domain : `runs/abl_data/val_E_frozen_clean32.jsonl` (232 docs) ;
- OOD : `runs/abl_data/ood.jsonl` (340 docs, 3 romans) ;
- 150 items test in-domain + 150 test OOD + 50 dev in-domain + 50 dev OOD.
- La réponse est toujours une sous-chaîne exacte du passage.
- Templates test/dev distincts des templates d'entraînement (G1).
- Les docs test et dev sont disjoints.

Sorties gelées :
  data/qa/g_passage_test_v1.jsonl
  data/qa/g_passage_dev_v1.jsonl
  data/qa/g_passage_v1.manifest.json   (SHA256 + comptages + audit)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from data import encode_mapped  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "qa")
TEST = os.path.join(OUT_DIR, "g_passage_test_v1.jsonl")
DEV = os.path.join(OUT_DIR, "g_passage_dev_v1.jsonl")
MANIFEST = os.path.join(OUT_DIR, "g_passage_v1.manifest.json")
IN_DOMAIN = os.path.join(ROOT, "runs", "abl_data", "val_E_frozen_clean32.jsonl")
OOD = os.path.join(ROOT, "runs", "abl_data", "ood.jsonl")
VOCAB = os.path.join(ROOT, "runs", "E-t08", "model.pt.vocab.json")
TOKENIZER = "HuggingFaceTB/SmolLM-135M"
SEED = 0

# Templates d'entraînement (G1) : le banc ne les utilise jamais.
TRAIN_TEMPLATES = {
    "annee": [
        "Quelle année correspond à la phrase « {clue} » ?",
        "En quelle année la phrase « {clue} » situe-t-elle les faits ?",
        "Quelle est l'année citée dans « {clue} » ?",
    ],
    "nombre": [
        "Quel nombre correspond à la phrase « {clue} » ?",
        "Combien vaut le nombre cité dans « {clue} » ?",
        "Quel est le nombre présent dans « {clue} » ?",
    ],
    "entite": [
        "Quel nom propre complète « {clue} » ?",
        "Quel nom propre est associé à « {clue} » ?",
        "Quel est le nom propre cité dans « {clue} » ?",
    ],
}

# Test : formulations jamais vues à l'entraînement.
TEST_TEMPLATES = {
    "annee": [
        "Dans le passage, à quelle année renvoie la phrase « {clue} » ?",
        "D'après le texte, quelle année est associée à « {clue} » ?",
        "Le passage mentionne une année dans « {clue} » ; laquelle ?",
    ],
    "nombre": [
        "Dans le passage, quel nombre complète la phrase « {clue} » ?",
        "D'après le texte, quel nombre est associé à « {clue} » ?",
        "Le passage cite un nombre dans « {clue} » ; lequel ?",
    ],
    "entite": [
        "Dans le passage, quel nom propre correspond à « {clue} » ?",
        "D'après le texte, quel nom propre est associé à « {clue} » ?",
        "Le passage cite un nom propre dans « {clue} » ; lequel ?",
    ],
}

# Dev : troisième famille, distincte de test et d'entraînement.
DEV_TEMPLATES = {
    "annee": [
        "Quelle année le passage donne-t-il pour « {clue} » ?",
        "Repère l'année de « {clue} » dans le passage.",
        "À quelle année la phrase « {clue} » fait-elle référence ?",
    ],
    "nombre": [
        "Quel nombre le passage donne-t-il pour « {clue} » ?",
        "Repère le nombre de « {clue} » dans le passage.",
        "À quel nombre la phrase « {clue} » fait-elle référence ?",
    ],
    "entite": [
        "Quel nom propre le passage donne-t-il pour « {clue} » ?",
        "Repère le nom propre de « {clue} » dans le passage.",
        "À quel nom propre la phrase « {clue} » fait-elle référence ?",
    ],
}

MONTHS = ("janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
          "septembre|octobre|novembre|décembre|decembre")
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
DATE_RE = re.compile(rf"\b(\d{{1,2}}\s+(?:{MONTHS})\s+\d{{3,4}})\b", re.I)
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w.])")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
CAP = (r"(?:[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][a-zàâäéèêëîïôöùûüç'’]+"
       r"(?:-[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ][a-zàâäéèêëîïôöùûüç'’]+)*"
       r"|[IVXLCDM]{1,6}|[A-Z]{2,6})")
ENT_RE = re.compile(
    rf"\b({CAP}(?:\s+(?:de|du|des|d'|la|le|les)?\s*{CAP}){{0,2}})\b")
STOP_ENT = {
    "il", "elle", "le", "la", "les", "un", "une", "des", "dans", "mais",
    "puis", "quand", "comme", "cette", "ce", "cet", "j'", "nous", "vous",
    "ils", "elles", "en", "au", "aux", "sur", "sous", "par", "pour",
    "avec", "sans", "tout", "tous", "toute", "toutes", "cependant",
    "alors", "ainsi", "enfin", "pourtant", "c'est", "n'est", "l'", "d'",
    "à", "a", "et", "ou", "où", "si", "ne", "pas", "plus", "moins",
}
TITLES = {
    "m.", "mme", "mlle", "monsieur", "madame", "mademoiselle", "capitaine",
    "commandant", "père", "roi", "reine", "empereur", "saint", "sainte",
    "docteur", "professeur", "général", "colonel", "maître", "abbé", "pape",
    "président", "sénateur", "député", "comte", "comtesse", "baron",
    "prince", "princesse", "duc", "duchesse",
}
IMP_VERBS = {
    "lisez", "allons", "savez", "voyez", "faites", "prenez", "regardez",
    "écoutez", "venez", "allez", "songez", "croyez", "pensez", "dis",
    "dit", "répondit", "reprit", "poursuivit", "demanda", "s'écria",
}


def valid_entity(s: str, m: re.Match, ent: str) -> bool:
    """Haute précision : nom multi-mots, ou nom précédé d'un titre."""
    toks = ent.split()
    if toks[0].lower().strip("'’") in IMP_VERBS:
        return False
    before = s[:m.start(1)].strip().split()
    if (before and before[-1][:1].isupper()
            and before[-1].lower().strip("'’") not in TITLES):
        # Évite de couper « Frank Plumpton Ramsey » en « Plumpton Ramsey ».
        return False
    if "les" in [t.lower() for t in toks[1:]]:
        return False
    if len(toks) >= 2:
        return True
    if ent.isupper() and 2 <= len(ent) <= 8:
        return True
    before = s[:m.start(1)].strip().split()
    return bool(before) and before[-1].lower().strip("'’") in TITLES


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_answer(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", s, flags=re.U)
    return re.sub(r"\s+", " ", s)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_docs(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                out.append({"doc_id": i, "text": json.loads(line)["text"]})
    return out


def sentences(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out = []
    for p in paras:
        for s in SENT_RE.split(p):
            s = norm_text(s)
            if 40 <= len(s) <= 600:
                out.append(s)
    return out


def candidates_for_sentence(s: str) -> list[dict]:
    """Retourne 0–2 candidats (année, nombre, entité) non ambigus."""
    if len(s) < 50 or "|" in s or " : " in s or "/" in s or "=" in s:
        return []
    if any(k in s for k in ("Homonymie", "toponyme", "Connecteur logique",
                            "Vocabulaire des", "Logique mathématique",
                            "d'anciennes communes", "Propriété chimique")):
        return []
    found = []
    low = s.lower()
    coord = any(k in low for k in ("longitude", "latitude", "milles", "pieds",
                                   "lieues", "kilomètres", "kilometres",
                                   "mètres", "metres"))
    years = [m for m in DATE_RE.finditer(s)]
    if years:
        if len(years) == 1 and len(YEAR_RE.findall(s)) <= 1:
            m = years[0]
            # La question demande une année : on garde l'année, pas le jour.
            y = YEAR_RE.search(m.group(1))
            if y:
                found.append({"type": "annee", "answer": y.group(1),
                              "start": y.start(1), "end": y.end(1)})
    else:
        ys = list(YEAR_RE.finditer(s))
        if len(ys) == 1 and not coord:
            m = ys[0]
            found.append({"type": "annee", "answer": m.group(1),
                          "start": m.start(1), "end": m.end(1)})
    # Nombre : on évite les années, les dates, les coordonnées et les
    # appels de note entre crochets.
    nums = list(NUM_RE.finditer(s))
    nums = [m for m in nums
            if not (len(m.group(1)) == 4 and m.group(1).startswith(("1", "2")))
            and not (m.start(1) > 0 and s[m.start(1) - 1] in "[(")
            and not (m.end(1) < len(s) and s[m.end(1)] in "])")
            and not any(a["start"] <= m.start(1) < a["end"] for a in found)]
    if len(nums) == 1:
        m = nums[0]
        found.append({"type": "nombre", "answer": m.group(1),
                      "start": m.start(1), "end": m.end(1)})
    # Noms propres : secours pour les textes littéraires sans chiffres
    # (Verne écrit « vingt-cinq » en toutes lettres). On exige un seul
    # candidat non trivial dans la phrase, hors premier mot.
    ents = []
    for m in ENT_RE.finditer(s):
        ent = m.group(1).strip()
        if m.start(1) == 0 or len(ent) < 3:
            continue
        if ent.split()[0].lower().strip("'’") in STOP_ENT:
            continue
        if not valid_entity(s, m, ent):
            continue
        if any(a["answer"] in ent or ent in a["answer"] for a in found):
            continue
        ents.append({"type": "entite", "answer": ent,
                     "start": m.start(1), "end": m.end(1)})
    # Déduplique les répétitions exactes du même nom.
    uniq = {e["answer"]: e for e in ents}
    if len(uniq) == 1:
        found.append(next(iter(uniq.values())))
    return found


def build_passage(sents: list[str], idx: int, max_tokens: int, tok,
                  vocab: dict) -> str | None:
    """Contexte autour de la phrase cible, ≤ max_tokens tokens mappés."""
    if len(encode_mapped(tok, vocab, sents[idx], add_eos=False)) > max_tokens:
        return None
    chosen = [sents[idx]]
    left, right = idx - 1, idx + 1
    # Ajoute des voisins tant que le passage reste sous la limite.
    while left >= 0 or right < len(sents):
        added = False
        for side in (0, 1):
            if side == 0 and left >= 0:
                cand = [sents[left]] + chosen
                left -= 1
            elif side == 1 and right < len(sents):
                cand = chosen + [sents[right]]
                right += 1
            else:
                continue
            txt = " ".join(cand)
            if len(encode_mapped(tok, vocab, txt, add_eos=False)) <= max_tokens:
                chosen = cand
                added = True
        if not added:
            break
        if len(encode_mapped(tok, vocab, " ".join(chosen), add_eos=False)) >= max_tokens * 0.8:
            break
    return " ".join(chosen) if chosen else None


def make_item(doc: dict, source: str, sents: list[str], si: int,
              cand: dict, templates: dict, tok, vocab: dict,
              max_prompt_tokens: int) -> dict | None:
    passage = build_passage(sents, si, 130, tok, vocab)
    if not passage or cand["answer"] not in passage:
        return None
    # La phrase cible doit contenir la réponse une fois ; le passage peut
    # répéter une année, mais la question reste ancrée sur la phrase.
    target = sents[si]
    if target.count(cand["answer"]) != 1:
        return None
    repl = {"annee": "cette année", "nombre": "ce nombre",
            "entite": "cette entité"}[cand["type"]]
    marker = {"annee": "ZZANNEEZZ", "nombre": "ZZNOMBREZZ",
              "entite": "ZZENTITEZZ"}[cand["type"]]
    clue = target.replace(cand["answer"], marker, 1)
    if clue == target:
        return None
    # Clue courte : 6 mots avant/après le remplacement, pour que le prompt
    # (passage + question) tienne dans 230 tokens.
    words = clue.split()
    pos = next((i for i, w in enumerate(words) if marker in w),
               len(words) // 2)
    clue = " ".join(words[max(0, pos - 6):pos + 7]).replace(marker, repl)
    if not clue:
        return None
    tpl = templates[cand["type"]]
    # Choisit un template déterministe à partir du doc et de la phrase.
    # doc_id peut être un entier (banc) ou une chaîne (tranche-doc G1).
    if isinstance(doc["doc_id"], int):
        doc_key = doc["doc_id"]
    else:
        doc_key = int(hashlib.sha1(str(doc["doc_id"]).encode("utf-8"))
                      .hexdigest()[:8], 16)
    k = (doc_key * 131 + si * 17 + len(cand["answer"])) % len(tpl)
    question = tpl[k].format(clue=clue)
    prompt = f"Passage :\n{passage}\n\nQuestion : {question}\nRéponse :"
    pids = encode_mapped(tok, vocab, prompt, add_eos=False)
    aids = encode_mapped(tok, vocab, cand["answer"], add_eos=False)
    if len(pids) + len(aids) + 1 > max_prompt_tokens:
        return None
    return {
        "source": source,
        "doc_id": doc["doc_id"],
        "type": cand["type"],
        "template": tpl[k],
        "passage": passage,
        "question": question,
        "answer": cand["answer"],
        "prompt": prompt,
        "n_prompt_tokens": len(pids),
        "n_answer_tokens": len(aids),
        "answer_in_passage": cand["answer"] in passage,
        "answer_unique_in_target": target.count(cand["answer"]) == 1,
        "answer_unique_in_passage": passage.count(cand["answer"]) == 1,
    }


def collect(docs: list[dict], source: str, start: int, need: int,
            templates: dict, tok, vocab: dict, max_prompt_tokens: int,
            used_docs: set[int], max_per_doc: int = 1) -> list[dict]:
    items = []
    for doc in docs:
        if doc["doc_id"] in used_docs:
            continue
        if len(items) >= need:
            break
        sents = sentences(doc["text"])
        n_doc = 0
        seen_sent: set[int] = set()
        for si, s in enumerate(sents):
            if si in seen_sent:
                continue
            for cand in candidates_for_sentence(s):
                it = make_item(doc, source, sents, si, cand, templates,
                               tok, vocab, max_prompt_tokens)
                if it:
                    it["id"] = f"{source}-{doc['doc_id']}-{si}-{cand['type']}"
                    items.append(it)
                    used_docs.add(doc["doc_id"])
                    seen_sent.add(si)
                    n_doc += 1
                    break
            if n_doc >= max_per_doc or len(items) >= need:
                break
        if len(items) >= need:
            break
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    with open(VOCAB, encoding="utf-8") as f:
        vocab = json.load(f)
    ind = load_docs(IN_DOMAIN)
    ood = load_docs(OOD)
    rng = random.Random(a.seed)
    # Mélange déterministe des docs, puis test/dev disjoints.
    rng.shuffle(ind)
    rng.shuffle(ood)
    used_ind: set[int] = set()
    used_ood: set[int] = set()
    test_ind = collect(ind, "in_domain", 0, 150, TEST_TEMPLATES, tok, vocab,
                       230, used_ind)
    test_ood = collect(ood, "ood", 0, 150, TEST_TEMPLATES, tok, vocab,
                       230, used_ood)
    dev_ind = collect(ind, "in_domain", 0, 50, DEV_TEMPLATES, tok, vocab,
                      230, used_ind, max_per_doc=3)
    dev_ood = collect(ood, "ood", 0, 50, DEV_TEMPLATES, tok, vocab,
                      230, used_ood, max_per_doc=3)
    test = test_ind + test_ood
    dev = dev_ind + dev_ood
    print(f"[qa] brut : test_ind={len(test_ind)} test_ood={len(test_ood)} "
          f"dev_ind={len(dev_ind)} dev_ood={len(dev_ood)}")
    if len(test) < 300 or len(dev) < 100:
        print(f"[qa] insuffisant : test={len(test)} (300), dev={len(dev)} (100)")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    for path, rows in ((TEST, test), (DEV, dev)):
        with open(path, "w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    manifest = {
        "version": "g_passage_v1",
        "seed": a.seed,
        "tokenizer": TOKENIZER,
        "vocab": VOCAB,
        "sources": {"in_domain": IN_DOMAIN, "ood": OOD},
        "n_test": len(test), "n_dev": len(dev),
        "composition": {
            "test_in_domain": len(test_ind), "test_ood": len(test_ood),
            "dev_in_domain": len(dev_ind), "dev_ood": len(dev_ood),
        },
        "types": {
            "test": {t: sum(1 for x in test if x["type"] == t)
                     for t in ("annee", "nombre", "entite")},
            "dev": {t: sum(1 for x in dev if x["type"] == t)
                    for t in ("annee", "nombre", "entite")},
        },
        "templates_train": TRAIN_TEMPLATES,
        "templates_test": TEST_TEMPLATES,
        "templates_dev": DEV_TEMPLATES,
        "files": {
            os.path.relpath(TEST, ROOT): sha256_file(TEST),
            os.path.relpath(DEV, ROOT): sha256_file(DEV),
        },
        "audit": {
            "answers_in_passage": sum(1 for x in test + dev if x["answer_in_passage"]),
            "answers_unique_in_target": sum(
                1 for x in test + dev if x["answer_unique_in_target"]),
            "answers_unique_in_passage": sum(
                1 for x in test + dev if x["answer_unique_in_passage"]),
            "n": len(test) + len(dev),
            "max_prompt_tokens": max(x["n_prompt_tokens"] for x in test + dev),
            "max_answer_tokens": max(x["n_answer_tokens"] for x in test + dev),
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[qa] test={len(test)} dev={len(dev)} -> {TEST}, {DEV}")
    print(f"[qa] audit={manifest['audit']}")
    print(f"[qa] SHA256={manifest['files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
