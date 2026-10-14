"""P2 : audit + déduplique clean-corpus/slices (CPU, sans dépendance).

Méthode documentée :
1. Exact : sha1 du texte normalisé (lower + espaces collapsés).
2. Near-dup : MinHash 125 signatures (8 bandes... cf. ci-dessous) sur
   shingles de 5 mots (hashés 64-bit, PRNG seed fixe), LSH b=25 bandes
   x r=5 (seuil effectif ~0.52, proba de capture à 0.85 ≈ 1.0),
   vérification Jaccard exacte, seuil gardé 0.85.
3. Heuristiques : FR/EN par stopwords (listes embarquées, pas de dep),
   pattern = doc dont >50% des lignes matchent un motif liste/fiche
   (titre court + lignes à deux-points/puces/quiz).
Sorties : docs/results/dedup_report.json (chiffres), pool dédupliqué
/Volumes/Lexar/clean-corpus/dedup/ (hors repo), tranches 100 Mo +
manifests (tokens comptés au tokenizer SmolLM).
Usage : nice -n 10 .venv/bin/python tools/dedup_corpus.py
"""
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SRC = "/Volumes/Lexar/clean-corpus/slices"
DST = "/Volumes/Lexar/clean-corpus/dedup"
NPERM = 125
BANDS = 25
ROWS = 5
THRESH = 0.85
_rng = random.Random(20261014)
_A = [_rng.randrange(1, 2**61 - 1) | 1 for _ in range(NPERM)]
_B = [_rng.randrange(0, 2**61 - 1) for _ in range(NPERM)]
_P = 2**61 - 1

FR_STOP = set("le la les de des du un une et est en dans que qui pour pas sur plus par il elle nous vous ils elles son sa ses ce cette ces au aux ce sont avec tout tous comme mais donc car ni oùdont".split())
EN_STOP = set("the a an and is in on of to for with that this it as at be are was were by from has have he she they we you his her its our their".split())
PAT_RE = re.compile(r"^(.{1,60}:|[-*•\d]+[.)\s]|Q\s*:|R\s*:|Fiche|Quiz|Vrai ou faux)", re.I)


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def shingles(t, k=5, cap=2000):
    w = norm(t).split()
    out = set()
    for i in range(min(len(w) - k + 1, cap)):
        h = int(hashlib.sha1(" ".join(w[i:i + k]).encode()).hexdigest()[:16], 16)
        out.add(h)
    return out


def minhash(s):
    sig = []
    for a, b in zip(_A, _B):
        m = _P
        for h in s:
            v = (a * (h % _P) + b) % _P
            if v < m:
                m = v
        sig.append(m)
    return sig


def lang_heu(t):
    w = norm(t).split()[:400]
    f = sum(1 for x in w if x in FR_STOP)
    e = sum(1 for x in w if x in EN_STOP)
    if f == 0 and e == 0:
        return "unk"
    return "fr" if f >= e else "en"


def is_pattern(t):
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    hit = sum(1 for l in lines if PAT_RE.match(l.strip()))
    return hit / len(lines) > 0.5


def main():
    t0 = time.time()
    docs = []  # (src, idx, text)
    for i in range(10):
        p = os.path.join(SRC, f"slice_{i:02d}.jsonl")
        with open(p, encoding="utf-8") as f:
            for j, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if t.strip():
                    docs.append((i, j, t))
    n0 = len(docs)
    chars0 = sum(len(t) for _, _, t in docs)
    print(f"[dedup] lus: {n0} docs, {chars0 / 1e6:.1f} M chars", flush=True)
    # Exact.
    seen, uniq = {}, []
    exact_dup = 0
    for rec in docs:
        h = hashlib.sha1(norm(rec[2]).encode()).hexdigest()
        if h in seen:
            exact_dup += 1
            continue
        seen[h] = True
        uniq.append(rec)
    print(f"[dedup] exact: -{exact_dup} doublons -> {len(uniq)} docs", flush=True)
    # Stats ratios sur uniq.
    langs = {"fr": 0, "en": 0, "unk": 0}
    pats = 0
    for _, _, t in uniq:
        langs[lang_heu(t)] += 1
        pats += is_pattern(t)
    print(f"[dedup] langues: {langs}, pattern: {pats} "
          f"({pats / max(len(uniq), 1) * 100:.1f} %)", flush=True)
    # MinHash + LSH.
    sigs, buckets = [], {}
    for n, (_, _, t) in enumerate(uniq):
        s = minhash(shingles(t))
        sigs.append(s)
        for b in range(BANDS):
            key = (b, tuple(s[b * ROWS:(b + 1) * ROWS]))
            buckets.setdefault(key, []).append(n)
    cand = set()
    for lst in buckets.values():
        if len(lst) > 1:
            lst = sorted(lst)
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    a, b = (lst[x], lst[y]) if lst[x] < lst[y] else (lst[y], lst[x])
                    cand.add((a, b))
    print(f"[dedup] paires candidates LSH: {len(cand)}", flush=True)
    drop = set()
    checked = 0
    for a, b in cand:
        if a in drop or b in drop:
            continue
        sa, sb = shingles(uniq[a][2]), shingles(uniq[b][2])
        if not sa or not sb:
            continue
        j = len(sa & sb) / len(sa | sb)
        checked += 1
        if j >= THRESH:
            drop.add(b)
    print(f"[dedup] near-dup (J>={THRESH}): -{len(drop)} docs "
          f"({checked} paires vérifiées)", flush=True)
    keep = [r for n, r in enumerate(uniq) if n not in drop]
    chars1 = sum(len(t) for _, _, t in keep)
    # Audit sample stratifié.
    rng = random.Random(7)
    sample_idx = sorted(rng.sample(range(len(keep)), min(200, len(keep))))
    audit = [{"n": n, "src": keep[n][0], "chars": len(keep[n][2]),
              "lang": lang_heu(keep[n][2]), "pattern": is_pattern(keep[n][2]),
              "head": keep[n][2][:160]} for n in sample_idx]
    # Tranches 100 Mo + manifests (tokens SmolLM).
    os.makedirs(DST, exist_ok=True)
    from main import load_tokenizer
    tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
    tranches, cur, cur_chars = [], [], 0
    for rec in keep:
        cur.append(rec)
        cur_chars += len(rec[2])
        if cur_chars >= 100 * 1024 * 1024:
            tranches.append(cur)
            cur, cur_chars = [], 0
    if cur:
        tranches.append(cur)
    manifests = []
    for ti, tr in enumerate(tranches):
        fp = os.path.join(DST, f" tranche_{ti:02d}.jsonl".strip())
        ntok = 0
        with open(fp, "w", encoding="utf-8") as f:
            for _, _, t in tr:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                ntok += len(tok.encode(t, add_special_tokens=False)) + 1
        ch = sum(len(t) for _, _, t in tr)
        manifests.append({"file": fp, "n_docs": len(tr), "chars": ch,
                          "tokens": ntok})
        print(f"[dedup] tranche {ti:02d}: {len(tr)} docs, "
              f"{ch / 1e6:.1f} M chars, {ntok / 1e6:.2f} M tokens", flush=True)
    report = {
        "n_avant": n0, "chars_avant": chars0,
        "doublons_exacts": exact_dup, "near_dup_085": len(drop),
        "n_apres": len(keep), "chars_apres": chars1,
        "langues": langs, "pattern_n": pats,
        "tranches": manifests, "audit_sample": audit,
        "methode": "sha1(normalise) + MinHash(125,shingle5)+LSH(25x5)+Jaccard>=0.85",
        "duree_s": round(time.time() - t0, 1),
    }
    with open("docs/results/dedup_report.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "audit_sample"},
                  f, indent=1)
    with open("docs/results/dedup_audit.json", "w") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    print(f"[dedup] {n0} -> {len(keep)} docs "
          f"({chars0 / 1e6:.1f} -> {chars1 / 1e6:.1f} M chars) "
          f"en {time.time() - t0:.0f}s", flush=True)
    print("[dedup] TERMINE", flush=True)


if __name__ == "__main__":
    main()
