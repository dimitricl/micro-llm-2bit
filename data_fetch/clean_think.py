"""Valide les traces think existantes -> *.clean.jsonl (+ rejetées à part).

- think_pays.jsonl : attendu dérivé de data_clean/monde_fr.txt
  (capitale <-> pays selon le sens de la question).
- think_wiki.jsonl : format + français + longueurs (pas d'attendu connu).
- Écrit aussi q_wiki_retry.jsonl : questions de q_wiki.jsonl absentes
  de think_wiki.jsonl (comparaison sur le texte exact de la question).
Originaux jamais modifiés.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_fetch.gen_think import parse_strict, valide

CLEAN = "/Volumes/Lexar/clean-corpus"
MONDE = "/Volumes/Lexar/micro-llm-2bit/data_clean/monde_fr.txt"


def charge_monde() -> tuple[dict, dict]:
    """Retourne (pays->capitale, capitale->pays) depuis monde_fr.txt."""
    p2c, c2p = {}, {}
    pat = re.compile(r"(.+) est un pays .* dont la capitale est (.+)\.")
    with open(MONDE, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                p, c = m.group(1).strip(), m.group(2).strip()
                p2c[p] = c
                c2p[c] = p
    return p2c, c2p


def attendu_pays(q: str, p2c: dict, c2p: dict) -> str:
    m = re.match(r"Quelle est la capitale du (.+) \?", q)
    if m:
        return p2c.get(m.group(1).strip(), "")
    m = re.match(r"De quel pays (.+) est-elle la capitale \?", q)
    if m:
        return c2p.get(m.group(1).strip(), "")
    return ""


def nettoie(src: str, dst: str, rej: str, attendu_fn=None) -> tuple[int, int]:
    ok, ko = 0, 0
    with open(src, encoding="utf-8") as f_in, \
            open(dst, "w", encoding="utf-8") as f_out, \
            open(rej, "w", encoding="utf-8") as f_rej:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            txt = row.get("text", "")
            m = re.match(r"Question : (.+?)\n(.*)", txt, re.DOTALL)
            q, corps = (m.group(1), m.group(2)) if m else ("", txt)
            parsed = parse_strict(corps)
            if parsed is None:
                f_rej.write(json.dumps({"q": q, "raison": "pas_de_balises"}) + "\n")
                ko += 1
                continue
            think, rep = parsed
            att = attendu_fn(q) if attendu_fn else ""
            raison = valide(think, rep, att)
            if raison:
                f_rej.write(json.dumps({"q": q, "raison": raison},
                                       ensure_ascii=False) + "\n")
                ko += 1
                continue
            f_out.write(json.dumps(
                {"text": f"Question : {q}\n<think>{think}</think>\nReponse : {rep}"},
                ensure_ascii=False) + "\n")
            ok += 1
    print(f"[clean] {src} : {ok} OK, {ko} rejetées -> {dst}")
    return ok, ko


def questions_manquantes() -> int:
    """Questions de q_wiki.jsonl absentes de think_wiki.jsonl -> retry."""
    vues = set()
    with open(f"{CLEAN}/think_wiki.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"Question : (.+)", json.loads(line).get("text", ""))
            if m:
                vues.add(m.group(1))
    n = 0
    with open(f"{CLEAN}/q_wiki.jsonl", encoding="utf-8") as f_in, \
            open(f"{CLEAN}/q_wiki_retry.jsonl", "w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            if row.get("q") not in vues:
                f_out.write(line if line.endswith("\n") else line + "\n")
                n += 1
    print(f"[clean] {n} questions manquantes -> q_wiki_retry.jsonl")
    return n


if __name__ == "__main__":
    p2c, c2p = charge_monde()
    print(f"[clean] monde : {len(p2c)} couples pays/capitale")
    nettoie(f"{CLEAN}/think_pays.jsonl", f"{CLEAN}/think_pays.clean.jsonl",
            f"{CLEAN}/think_pays.rejected.jsonl",
            attendu_fn=lambda q: attendu_pays(q, p2c, c2p))
    nettoie(f"{CLEAN}/think_wiki.jsonl", f"{CLEAN}/think_wiki.clean.jsonl",
            f"{CLEAN}/think_wiki.rejected.jsonl")
    questions_manquantes()
