"""Découpe wiki_fr.jsonl en tranches ~100 Mo pour l'entraînement par tranches.

- Streaming : jamais tout en RAM (lecture ligne par ligne, N fichiers ouverts).
- Affectation déterministe par hash md5(salt + titre) % N : pas de
  recouvrement, reproductible, sans tout charger.
- Format .jsonl (1 doc/ligne) : préserve la granularité documents exigée
  par le split train/val (un .txt monolithe = 1 seul doc).
- Nettoyage basique : espaces normalisés, lignes vides supprimées,
  lignes purement marqueurs (*, #, ;, :) et headers de sections
  Wikipédia (Notes, Références, Voir aussi, ...) supprimées.

Usage : python make_slices.py [--slice-mb 100] [--salt v1]
"""

import hashlib
import json
import os
import re
import sys

IN = "/Volumes/Lexar/clean-corpus/wiki_fr.jsonl"
OUT_DIR = "/Volumes/Lexar/clean-corpus/slices"
SLICE_MB = float(sys.argv[sys.argv.index("--slice-mb") + 1]) if "--slice-mb" in sys.argv else 100.0
SALT = sys.argv[sys.argv.index("--salt") + 1] if "--salt" in sys.argv else "v1"

HEADER_RE = re.compile(
    r"^(notes?(\s+et\s+(références|cartes))?|références?|voir aussi|"
    r"liens externes|bibliographie|articles? connexes?)\s*$",
    re.IGNORECASE,
)
MARKER_RE = re.compile(r"^[*#;:]+$")


def nettoie(text: str) -> str:
    """Normalise espaces, supprime lignes vides/marqueurs/headers wiki."""
    lignes = []
    for ln in text.split("\n"):
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln or MARKER_RE.match(ln) or HEADER_RE.match(ln):
            continue
        lignes.append(ln)
    return "\n".join(lignes)


def tranche_de(titre: str, n: int) -> int:
    h = hashlib.md5((SALT + titre).encode("utf-8")).hexdigest()
    return int(h, 16) % n


def main() -> None:
    total = os.path.getsize(IN)
    n = max(1, round(total / (SLICE_MB * 1024 * 1024)))
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = [
        open(os.path.join(OUT_DIR, f"slice_{i:02d}.jsonl"), "w", encoding="utf-8")
        for i in range(n)
    ]
    tailles = [0] * n
    docs = [0] * n
    with open(IN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            titre = row.get("title", "")
            texte = nettoie((titre + "\n" + row.get("text", "")).strip())
            if len(texte) < 500:
                continue
            i = tranche_de(titre or texte[:64], n)
            l = json.dumps({"text": texte}, ensure_ascii=False)
            outs[i].write(l + "\n")
            tailles[i] += len(l.encode("utf-8")) + 1
            docs[i] += 1
    for fh in outs:
        fh.close()
    for i in range(n):
        print(f"[slices] slice_{i:02d}.jsonl : {docs[i]} docs, "
              f"{tailles[i] / 1e6:.1f} Mo")
    print(f"[slices] {n} tranches (salt={SALT}) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
