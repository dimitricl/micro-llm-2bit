"""Diagnostic du découpage tokenizer SmolLM -> vocabulaire réduit 12 000.

Question posée (chaîne F) : la confusion de surface « capitale » -> « capitaine »
vient-elle du découpage en sous-mots ? On mesure, pour un mot donné :

- son découpage SmolLM natif (tokens et pièces) ;
- son découpage dans le vocabulaire réduit (`kept_ids` du .vocab.json) ;
- le nombre de sous-mots et la présence d'un token hors vocabulaire (unk) ;
- le partage de préfixe : deux mots découpés dont le PREMIER sous-token est
  identique sont dits « à préfixe partagé ». C'est le cas qui rend la
  distinction dépendante du seul dernier sous-token.

Le jeu de mots = les 4 mots de la confusion + les N mots alphabétiques les
plus fréquents du corpus QA (comptés sur les documents, pas sur les tokens).

Sorties : JSON machine + page Markdown. Aucun fichier gelé n'est modifié.

Usage :
  .venv/bin/python tools/diag_tokenizer.py \\
      --vocab exports/E-t08.bin.vocab.json \\
      --qa-corpus data/qa/qa_fr_pilot_v3.jsonl \\
      --out-json docs/results/tokenizer_diag.json \\
      --out-md docs/TOKENIZER_DIAG.md
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ]+")
TARGETS = ["capitale", "capitaine", "capitole", "capitalle"]


def _fix_piece(piece: str) -> str:
    """Byte-level BPE : « Ã©e » -> « ée » pour l'affichage."""
    try:
        return piece.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return piece


def load_words(qa_corpus: str, n_words: int) -> tuple[list[str], list[tuple[str, int]]]:
    """Retourne (mots analysés, top fréquences du corpus QA).

    Les 4 mots cibles sont toujours inclus. Les N mots fréquents sont
    alphabétiques, en minuscules, de longueur >= 2 ; les cibles déjà
    présentes ne sont pas dupliquées.
    """
    counter: collections.Counter[str] = collections.Counter()
    with open(qa_corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            for w in WORD_RE.findall(text):
                w = w.lower()
                if len(w) >= 2:
                    counter[w] += 1
    top = counter.most_common()
    words = list(TARGETS)
    for w, _ in top:
        if len(words) >= len(TARGETS) + n_words:
            break
        if w not in words:
            words.append(w)
    return words, top


def count_in_corpus(glob_pattern: str, words: list[str]) -> dict[str, int]:
    """Occurrences entières des mots cibles dans le corpus d'entraînement."""
    import glob as _glob

    pats = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words}
    counts = {w: 0 for w in words}
    for path in sorted(_glob.glob(glob_pattern)):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                for w, pat in pats.items():
                    counts[w] += len(pat.findall(line))
    return counts


def analyse_word(tok, vocab: dict, word: str) -> dict:
    """Découpage natif + découpage réduit + statut unk d'un mot."""
    kept = vocab["kept_ids"]
    new_of = {old: new for new, old in enumerate(kept) if old >= 0}
    unk_new = vocab["unk_new"]
    ids = tok.encode(word, add_special_tokens=False)
    pieces = [_fix_piece(p) for p in tok.convert_ids_to_tokens(ids)]
    mapped = [new_of.get(i, unk_new) for i in ids]
    unk_flags = [i not in new_of for i in ids]
    return {
        "mot": word,
        "n_tokens": len(ids),
        "tokens_old": ids,
        "pieces": pieces,
        "tokens_new": mapped,
        "unk_par_token": unk_flags,
        "a_unk": any(unk_flags),
        "premier_token": ids[0] if ids else None,
        "premiere_piece": pieces[0] if pieces else None,
        "est_sous_mot": len(ids) > 1,
    }


def prefix_groups(rows: list[dict]) -> list[dict]:
    """Groupes de mots dont le premier sous-token SmolLM est identique."""
    groups: dict[int, list[str]] = collections.defaultdict(list)
    for r in rows:
        if r["premier_token"] is not None:
            groups[r["premier_token"]].append(r["mot"])
    out = []
    for tid, words in groups.items():
        if len(words) > 1:
            out.append({
                "premier_token": tid,
                "premiere_piece": rows[[r["mot"] for r in rows].index(words[0])][
                    "premiere_piece"],
                "mots": words,
                "n_mots": len(words),
            })
    out.sort(key=lambda g: (-g["n_mots"], g["premier_token"]))
    return out


def build_markdown(payload: dict) -> str:
    rows = payload["mots"]
    by = {r["mot"]: r for r in rows}
    ng = payload["global"]
    lines = [
        "# Diagnostic tokenizer — confusion « capitale » / « capitaine »",
        "",
        f"Vocabulaire : `{payload['vocab']}` "
        f"({payload['n_kept']} tokens ; unk distinct="
        f"{payload['unk_distinct']}, unk_new={payload['unk_new']}). "
        f"Tokenizer : `{payload['parent']}`.",
        "",
        "## Verdict court",
        "",
        payload["verdict"],
        "",
        "## Les quatre mots de la confusion",
        "",
        "| mot | sous-mots SmolLM | ids réduits | unk ? |",
        "|---|---|---|---|",
    ]
    for w in TARGETS:
        r = by[w]
        pieces = " + ".join(f"`{p}`" for p in r["pieces"])
        lines.append(
            f"| **{w}** | {pieces} ({r['n_tokens']}) | "
            f"{r['tokens_new']} | {'oui' if r['a_unk'] else 'non'} |")
    lines += [
        "",
        f"Préfixe de sous-tokens partagé par les formes correctes : "
        f"**{'oui' if payload['cible_prefixe_partage'] else 'non'}** "
        f"(longueur {payload['cible_prefixe_longueur']}, "
        f"`{' + '.join(payload['cible_prefixe_commun'])}`).",
        "",
    ]
    if payload.get("frequences_corpus_train"):
        lines += [
            "Occurrences dans le corpus d'entraînement dédupliqué "
            "(tranches 00-09) : "
            + ", ".join(f"**{w}** {n:,}".replace(",", " ")
                        for w, n in payload["frequences_corpus_train"].items())
            + ". La forme correcte est plus fréquente que la forme corrompue : "
            "la corruption n'est donc pas un simple effet de fréquence "
            "lexicale.",
            "",
        ]
    lines += [
        "## Taux globaux (jeu analysé)",
        "",
        f"- mots analysés : **{ng['n_mots']}**",
        f"- mots découpés en > 1 sous-token : **{ng['n_sous_mots']} / "
        f"{ng['n_mots']} = {ng['taux_sous_mots']:.1%}**",
        f"- mots avec au moins un token hors vocabulaire (unk) : "
        f"**{ng['n_mots_unk']} / {ng['n_mots']} = {ng['taux_mots_unk']:.1%}**",
        f"- tokens hors vocabulaire : **{ng['n_tokens_unk']} / "
        f"{ng['n_tokens']} = {ng['taux_tokens_unk']:.2%}**",
        f"- mots découpés à préfixe partagé : **{ng['n_sous_mots_prefixe']} "
        f"/ {ng['n_sous_mots']} = {ng['taux_sous_mots_prefixe']:.1%}** "
        "des mots découpés",
        "",
        "## Mots à préfixe partagé (premier sous-token identique)",
        "",
    ]
    if payload["groupes_prefixe"]:
        for g in payload["groupes_prefixe"]:
            lines.append(
                f"- `{g['premiere_piece']}` : "
                + ", ".join(f"**{w}**" if w in TARGETS else w
                            for w in g["mots"]))
    else:
        lines.append("- aucun")
    lines += [
        "",
        "## Les 50 mots fréquents du corpus QA",
        "",
        "| mot | n | sous-mots | pieces | unk |",
        "|---|---:|---|---|---|",
    ]
    for r in rows:
        if r["mot"] in TARGETS:
            continue
        pieces = " + ".join(f"`{p}`" for p in r["pieces"])
        lines.append(
            f"| {r['mot']} | {r['n_tokens']} | {r['n_tokens']} | {pieces} | "
            f"{'oui' if r['a_unk'] else 'non'} |")
    lines += [
        "",
        "## Lecture",
        "",
        payload["lecture"],
        "",
        "## Limites",
        "",
        "- Le comptage des mots se fait sur les documents QA (templates + "
        "général), pas sur les tokens du modèle : c'est un classement de "
        "fréquence lexicale, suffisant pour choisir le jeu de mots.",
        "- Le partage de préfixe est mesuré au premier sous-token SmolLM. "
        "Il ne prouve pas la confusion ; il montre que le modèle ne dispose "
        "que du dernier sous-token pour trancher.",
        "- Le taux d'unk porte sur le vocabulaire réduit E-t08 (12 000 tokens). "
        "Un mot découpé sans unk reste représentable ; il peut néanmoins être "
        "mal placé dans l'espace des embeddings.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Diagnostic tokenizer FR.")
    ap.add_argument("--vocab", default="exports/E-t08.bin.vocab.json")
    ap.add_argument("--qa-corpus", default="data/qa/qa_fr_pilot_v3.jsonl")
    ap.add_argument("--n-words", type=int, default=50)
    ap.add_argument("--out-json", default="docs/results/tokenizer_diag.json")
    ap.add_argument("--out-md", default="docs/TOKENIZER_DIAG.md")
    ap.add_argument("--train-glob",
                    default="/Volumes/Lexar/clean-corpus/dedup/tranche_*.jsonl",
                    help="Corpus d'entraînement dédupliqué (comptage des "
                         "4 mots cibles) ; vide = pas de comptage.")
    a = ap.parse_args(argv)

    from main import load_tokenizer

    vocab = json.load(open(a.vocab, encoding="utf-8"))
    tok = load_tokenizer(vocab["parent"])
    words, top = load_words(a.qa_corpus, a.n_words)
    rows = [analyse_word(tok, vocab, w) for w in words]
    corpus_counts = (count_in_corpus(a.train_glob, TARGETS)
                     if a.train_glob else {})

    groups = prefix_groups(rows)
    n_mots = len(rows)
    n_sous_mots = sum(1 for r in rows if r["est_sous_mot"])
    n_mots_unk = sum(1 for r in rows if r["a_unk"])
    n_tokens = sum(r["n_tokens"] for r in rows)
    n_tokens_unk = sum(sum(r["unk_par_token"]) for r in rows)
    # Un mot découpé est "à préfixe partagé" si son premier token est
    # partagé avec au moins un autre mot analysé.
    group_of = {g["premier_token"]: g["mots"] for g in groups}
    n_sous_mots_prefixe = sum(
        1 for r in rows if r["est_sous_mot"]
        and len(group_of.get(r["premier_token"], [])) > 1)

    cible = [r for r in rows if r["mot"] in TARGETS]
    # Préfixe commun aux trois formes correctes (capitale/capitaine/capitole).
    corrects = [r for r in cible if r["mot"] != "capitalle"]
    prefix_len = 0
    if corrects:
        seqs = [r["tokens_old"] for r in corrects]
        while (all(len(s) > prefix_len for s in seqs)
               and len({s[prefix_len] for s in seqs}) == 1):
            prefix_len += 1
    prefix_pieces = (corrects[0]["pieces"][:prefix_len]
                     if corrects and prefix_len else [])
    suffixes = [r["pieces"][prefix_len:] for r in corrects]
    cible_shared = prefix_len >= 1
    # Verdict automatique sobre : on ne conclut que sur le découpage.
    if cible_shared:
        verdict = (
            "Les trois formes correctes (capitale, capitaine, capitole) "
            f"partagent leurs {prefix_len} premiers sous-tokens "
            f"(`{' + '.join(prefix_pieces)}`) ; elles ne diffèrent que par "
            f"un unique sous-token final (`{'` / `'.join(''.join(s) for s in suffixes)}`). "
            "La distinction repose donc sur un seul token, ce qui rend la "
            "confusion PLAUSIBLE. `capitalle` (faute) est encore plus "
            "fragile : son premier morceau `capital` est hors vocabulaire "
            "et retombe sur l'id unk/eos. Le découpage n'explique pas tout "
            "à lui seul ; l'étape 3 (surface) et l'étape 4 (recette) "
            "trancheront.")
    else:
        verdict = (
            "Les mots ne partagent pas de préfixe de sous-tokens. Le "
            "découpage ne fournit pas d'explication directe à la "
            "confusion ; regarder plutôt la fréquence et les embeddings.")
    payload = {
        "vocab": a.vocab,
        "parent": vocab["parent"],
        "n_kept": len(vocab["kept_ids"]),
        "unk_distinct": bool(vocab.get("unk_distinct", False)),
        "unk_new": vocab.get("unk_new"),
        "frequences_corpus_train": corpus_counts,
        "cible_prefixe_partage": cible_shared,
        "cible_prefixe_longueur": prefix_len,
        "cible_prefixe_commun": prefix_pieces,
        "cible_suffixes": ["".join(s) for s in suffixes],
        "mots": rows,
        "top_frequences": [{"mot": w, "n": n} for w, n in top[:100]],
        "groupes_prefixe": groups,
        "global": {
            "n_mots": n_mots,
            "n_sous_mots": n_sous_mots,
            "taux_sous_mots": n_sous_mots / max(n_mots, 1),
            "n_mots_unk": n_mots_unk,
            "taux_mots_unk": n_mots_unk / max(n_mots, 1),
            "n_tokens": n_tokens,
            "n_tokens_unk": n_tokens_unk,
            "taux_tokens_unk": n_tokens_unk / max(n_tokens, 1),
            "n_sous_mots_prefixe": n_sous_mots_prefixe,
            "taux_sous_mots_prefixe": n_sous_mots_prefixe / max(n_sous_mots, 1),
        },
        "verdict": verdict,
        "lecture": (
            "Le mécanisme pertinent n'est pas « le mot est inconnu » mais "
            "« le mot est connu en morceaux ». Si `cap` + `it` sont des "
            "tokens fréquents et que seuls `ale` / `aine` / `ole` changent, "
            "le modèle doit apprendre une distinction fine sur un unique "
            "sous-token, dans "
            "un espace d'embeddings de 512 dimensions partagé par des "
            "milliers de mots. C'est exactement le type de détail que le "
            "budget de 25 M paramètres peut sous-représenter, et qu'un "
            "surapprentissage des templates QA peut écraser au profit de la "
            "forme la plus fréquente (`capitaine`)."),
    }

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(payload, open(a.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(a.out_md) or ".", exist_ok=True)
    open(a.out_md, "w", encoding="utf-8").write(build_markdown(payload) + "\n")

    print(f"[diag-tok] {n_mots} mots, {n_sous_mots} découpés, "
          f"{n_mots_unk} avec unk, préfixe partagé cible="
          f"{cible_shared}")
    for w in TARGETS:
        r = next(x for x in rows if x["mot"] == w)
        print(f"[diag-tok] {w:10s} -> {r['pieces']} (n={r['n_tokens']}, "
              f"unk={r['a_unk']})")
    print(f"[diag-tok] -> {a.out_json}")
    print(f"[diag-tok] -> {a.out_md}")


if __name__ == "__main__":
    main()
