"""Sonde FR gelée : exact-match par alias, greedy de déploiement (CPU/NEON).

Mesure ce que la PPL ne mesure pas : le modèle répond-il au FAIT demandé,
et le fait-il aussi quand la formulation n'est pas celle du corpus
(`par_coeur` vs `reformule`) ?

Protocole
---------
- Modèle chargé via `engine.Engine` (.bin), noyau NEON, CPU pur.
- Décodage **greedy** avec les défauts de déploiement documentés
  (docs/ABLATIONS §P3) : `repetition_penalty=1.15`,
  `no_repeat_ngram_size=3`, `max_new=32`.
- Aucun entraînement, aucun GPU/MPS, aucun Ollama.

Règle de score (exacte, documentée)
-----------------------------------
1. Normalisation : `unicodedata.normalize("NFD")` puis suppression des
   diacritiques (accents), minuscules, toute ponctuation (au sens
   `unicodedata.category(ch).startswith("P")`) remplacée par une espace,
   espaces multiples compactés, strip. Ex. `"Pékin."` -> `"pekin"`,
   `"9,8"` -> `"9 8"`.
2. HIT si un alias accepté, normalisé de la même façon, apparaît dans le
   texte généré normalisé **avec frontières de tokens** (début/fin de
   chaîne ou espaces alentour). Donc `100` ne matche pas `1000`, et
   `New Delhi` ne matche pas `NewDelhi`.
3. Un alias vide est ignoré (jamais un hit gratuit).
4. `eos_collapse` = génération terminée en **moins de 5 tokens** (< 5).

Usage
-----
  nice -n 10 .venv/bin/python tools/eval_probe.py \\
      --model runs/E-t08/model.bin \\
      --out eval/Et08-probe/samples.md \\
      --results-out eval/Et08-probe/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import data as D
import engine as E

PROBE_PATH = "data/probe/probe_fr_v1.jsonl"
MANIFEST_PATH = "data/probe/probe_fr_v1.manifest.json"

MAX_NEW = 32
REPETITION_PENALTY = 1.15   # défaut de déploiement (docs/ABLATIONS §P3)
NO_REPEAT_NGRAM = 3
EOS_COLLAPSE_SEUIL = 5      # < 5 tokens

CATEGORIES = ("capitales", "dates", "definitions", "unites_sciences")
SLICES = ("par_coeur", "reformule")
PERIMETRES = ("domaine_etroit", "wiki_general")


def normalize(text: str) -> str:
    """Minuscules, sans accents, ponctuation -> espace, espaces compactés."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed
                       if not unicodedata.combining(ch))
    out = []
    for ch in stripped.lower():
        if unicodedata.category(ch).startswith("P"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def is_hit(gen_text: str, accepted: list[str]) -> bool:
    """Un alias normalisé apparaît-il, avec frontières de tokens ?"""
    hay = normalize(gen_text)
    if not hay:
        return False
    padded = f" {hay} "
    for alias in accepted or []:
        needle = normalize(alias)
        if not needle:
            continue
        if f" {needle} " in padded:
            return True
    return False


def eos_collapse(n_tokens: int) -> bool:
    """Génération terminée en moins de 5 tokens."""
    return n_tokens < EOS_COLLAPSE_SEUIL


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_probe(path: str, manifest_path: str = "") -> list[dict]:
    """Charge la sonde ; refuse bruyamment si le hash ne correspond pas."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = path if os.path.isabs(path) else os.path.join(root, path)
    if manifest_path:
        mpath = (manifest_path if os.path.isabs(manifest_path)
                 else os.path.join(root, manifest_path))
        man = json.load(open(mpath, encoding="utf-8"))
        got = sha256_file(path)
        if got != man.get("sha256"):
            print(f"[probe] ERREUR : sonde modifiée après gel\n"
                  f"  sha256 fichier  = {got}\n"
                  f"  sha256 manifeste= {man.get('sha256')}\n"
                  f"  La sonde est GELÉE (anti-ajustement a posteriori) ; "
                  f"toute évolution passe par probe_fr_v2.", file=sys.stderr)
            raise SystemExit(2)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def score_one(item: dict, gen_text: str, n_tokens: int) -> dict:
    """Enregistrement plat : 1 item = 1 ligne de résultats."""
    hit = is_hit(gen_text, item.get("reponses_acceptees") or [])
    return {
        "id": item["id"],
        "categorie": item["categorie"],
        "slice": item["slice"],
        "perimetre": item["perimetre"],
        "prompt": item["prompt"],
        "gen": gen_text,
        "n_tokens": n_tokens,
        "hit": hit,
        "eos_collapse": eos_collapse(n_tokens),
    }


def _cell(recs: list[dict]) -> dict:
    n = len(recs)
    hits = sum(1 for r in recs if r["hit"])
    col = sum(1 for r in recs if r["eos_collapse"])
    return {
        "n": n,
        "hits": hits,
        "taux": (hits / n) if n else 0.0,
        "n_eos_collapse": col,
        "taux_eos_collapse": (col / n) if n else 0.0,
        "n_tokens_moy": (sum(r["n_tokens"] for r in recs) / n) if n else 0.0,
    }


def aggregate(recs: list[dict]) -> dict:
    """Agrégats global / slice / catégorie / périmètre / cat×slice."""
    agg = {"global": _cell(recs)}
    for key, values in (("par_slice", SLICES),
                        ("par_categorie", CATEGORIES),
                        ("par_perimetre", PERIMETRES)):
        agg[key] = {
            v: _cell([r for r in recs if r[key[4:]] == v]) for v in values
        }
    agg["par_categorie_x_slice"] = {
        c: {s: _cell([r for r in recs
                      if r["categorie"] == c and r["slice"] == s])
            for s in SLICES}
        for c in CATEGORIES
    }
    return agg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sonde FR gelée (exact-match).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-out", default="")
    ap.add_argument("--probe", default=PROBE_PATH)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--repetition-penalty", type=float,
                    default=REPETITION_PENALTY)
    ap.add_argument("--no-repeat-ngram", type=int, default=NO_REPEAT_NGRAM)
    a = ap.parse_args(argv)

    from main import load_tokenizer, decode_ids

    items = load_probe(a.probe, a.manifest)
    vocab = json.load(open(a.model + ".vocab.json", encoding="utf-8"))
    tok = load_tokenizer(vocab["parent"])
    eng = E.Engine(a.model)
    print(f"[probe] noyau={E.KERNEL_NAME} items={len(items)} "
          f"greedy pen={a.repetition_penalty} ngram={a.no_repeat_ngram} "
          f"max_new={a.max_new}", flush=True)

    recs = []
    lignes = [f"# Sonde FR v1 — {a.model}", "",
              f"décodage : greedy, repetition_penalty={a.repetition_penalty}, "
              f"no_repeat_ngram_size={a.no_repeat_ngram}, max_new={a.max_new}",
              ""]
    for it in items:
        ids = D.encode_mapped(tok, vocab, it["prompt"], add_eos=False)
        ids = ids[-eng.meta["seq_max"]:]
        out = eng.generate(
            ids, max_new=a.max_new, temperature=0.0,
            stop={vocab["eos_new"]},
            repetition_penalty=a.repetition_penalty,
            no_repeat_ngram_size=a.no_repeat_ngram,
        )
        txt = decode_ids(tok, vocab, out)
        rec = score_one(it, txt, len(out))
        recs.append(rec)
        mark = "HIT " if rec["hit"] else "miss"
        lignes += [f"## [{it['categorie']}/{it['slice']}] {it['id']}", "",
                   f"prompt : {it['prompt']}",
                   f"attendu : {', '.join(it['reponses_acceptees'])}",
                   f"généré  : {txt!r}",
                   f"{mark} ({rec['n_tokens']} tokens"
                   + (", EOS-collapse" if rec["eos_collapse"] else "") + ")",
                   ""]

    agg = aggregate(recs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"[probe] -> {a.out}")

    payload = {
        "version": "probe_fr_v1",
        "model": a.model,
        "decodage": {"mode": "greedy",
                     "repetition_penalty": a.repetition_penalty,
                     "no_repeat_ngram_size": a.no_repeat_ngram,
                     "max_new": a.max_new,
                     "eos_collapse_seuil": EOS_COLLAPSE_SEUIL},
        "probe_sha256": sha256_file(
            a.probe if os.path.isabs(a.probe)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", a.probe)),
        "agregats": agg,
        "items": recs,
    }
    if a.results_out:
        os.makedirs(os.path.dirname(a.results_out) or ".", exist_ok=True)
        json.dump(payload, open(a.results_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[probe] results -> {a.results_out}")

    g = agg["global"]
    print(f"[probe] GLOBAL {g['hits']}/{g['n']} = {g['taux']:.4f} ; "
          f"EOS-collapse {g['n_eos_collapse']}/{g['n']} = "
          f"{g['taux_eos_collapse']:.4f}", flush=True)
    for s in SLICES:
        c = agg["par_slice"][s]
        print(f"[probe]   slice {s:10s} {c['hits']:3d}/{c['n']:3d} = "
              f"{c['taux']:.4f}  (EOS {c['taux_eos_collapse']:.4f})")
    for p in PERIMETRES:
        c = agg["par_perimetre"][p]
        print(f"[probe]   perim {p:13s} {c['hits']:3d}/{c['n']:3d} = "
              f"{c['taux']:.4f}")
    for cname in CATEGORIES:
        c = agg["par_categorie"][cname]
        print(f"[probe]   cat   {cname:15s} {c['hits']:3d}/{c['n']:3d} = "
              f"{c['taux']:.4f}")


if __name__ == "__main__":
    main()
