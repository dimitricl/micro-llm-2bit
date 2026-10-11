"""Parité .bin vs .pt de tranche00 (S3) : NLL position par position.

Protocole : sur les 6 documents C2 (manifeste versionné
docs/results/c2_subset.json, SHA1 vérifiés), compare la NLL position
par position entre le .bin (moteur CPU) et le .pt (PyTorch, CPU fp32).

Fenêtrage UNIQUE (make_windows) pour les deux côtés : fenêtres S+1
avec pas S, par document (+eos/doc), dernière fenêtre incomplète
ignorée des deux côtés. Les deux listes DOIVENT avoir la même
longueur (assert, pas de min_len silencieux).

Sortie : distribution de |NLL_bin − NLL_pt| (p50, p99, max),
ΔBPC agrégé (bin − pt) en a/b/c (même dénominateur chars),
nb de positions. Tolérance fixée APRÈS mesure :
condition = |ΔBPC| < 0,002 sur a/b/c (deux décimales BPC stables).

Usage :
  uv run python tools/check_parity_bin_pt.py [--out docs/results/parity.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.eval_bpc import load_c2_texts, surfaces, variantes  # noqa: E402

VOCAB_PATH = "exports/tranche00.bin.vocab.json"
PT_PATH = "runs/2026-10-08-tranche00/tranche00.pt"
BIN_PATH = "exports/tranche00.bin"
S = 128
TOL_DBPC = 0.002


def make_windows(sids: list[int], S: int = S) -> list[list[int]]:
    """Fenêtres S+1, pas S. Incomplètes (<2) exclues (les deux côtés)."""
    return [sids[i:i + S + 1] for i in range(0, len(sids) - 1, S)
            if len(sids[i:i + S + 1]) >= 2]


def nll_bin(windows: list[list[int]], model_path: str) -> list[float]:
    import engine as E

    eng = E.Engine(model_path)
    out = []
    for win in windows:
        eng.reset()
        prev = eng.step(win[0])
        for i in range(1, len(win)):
            z = prev.astype(np.float64)
            z -= z.max()
            e = np.exp(z)
            s = e.sum()
            out.append(float(np.log(s) - np.log(e[win[i]])))
            if i + 1 < len(win):
                prev = eng.step(win[i])
    return out


def nll_pt(windows: list[list[int]], ckpt_path: str) -> list[float]:
    from model import TinyTransformer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.float().eval()
    out = []
    with torch.no_grad():
        for win in windows:
            b = torch.tensor([win], dtype=torch.long)
            logits = model(b[:, :-1]).float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                reduction="none").numpy()
            out.extend(nll[:len(win) - 1].tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Parité .bin vs .pt.")
    ap.add_argument("--out", default="docs/results/parity.json")
    a = ap.parse_args()
    t0 = __import__("time").time()
    texts = load_c2_texts()
    print(f"[parity] {len(texts)} docs", flush=True)
    vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
    kept = set(vocab["kept_ids"])
    new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
    from main import load_tokenizer

    tok = load_tokenizer(vocab["parent"])
    parent_ids = []
    for t in texts:
        parent_ids += tok.encode(t, add_special_tokens=False) + [
            tok.eos_token_id]
    surf = surfaces(tok, parent_ids)
    # Fenêtrage unique : construit une fois sur le stream réduit concaténé
    # avec frontières docs (add_eos/doc comme eval_bpc.collect_engine).
    ids, pids, off = [], [], 0
    for t in texts:
        pdoc = parent_ids[off:off + len(
            tok.encode(t, add_special_tokens=False)) + 1]
        off += len(pdoc)
        ids += [new_of.get(o, vocab["unk_new"]) for o in pdoc]
        pids += pdoc
    assert off == len(parent_ids) and pids == parent_ids, "streams désalignés"
    # Fenêtres par doc pour garder l'alignement positions <-> pids.
    windows, pwindows, start = [], [], 0
    for t in texts:
        n = len(tok.encode(t, add_special_tokens=False)) + 1
        doc_ids, doc_pids = ids[start:start + n], pids[start:start + n]
        start += n
        for w, pw in zip(make_windows(doc_ids), make_windows(doc_pids)):
            windows.append(w)
            pwindows.append(pw)
    assert len(windows) == len(pwindows) and all(
        len(w) == len(pw) for w, pw in zip(windows, pwindows))
    nb = nll_bin(windows, BIN_PATH)
    np_ = nll_pt(windows, PT_PATH)
    assert len(nb) == len(np_), (
        f"longueurs inégales bin={len(nb)} pt={len(np_)}")
    print(f"[parity] {len(nb)} positions", flush=True)
    d = np.abs(np.array(nb) - np.array(np_))
    p50, p99, mx = float(np.median(d)), float(np.quantile(d, 0.99)), float(d.max())
    print(f"[parity] |dNLL| p50={p50:.2e} p99={p99:.2e} max={mx:.2e}",
          flush=True)
    chars = sum(len(t) for t in texts)
    # Records alignés position par position (mêmes fenêtres, mêmes pids).
    flat_pids = [pid for pw in pwindows for pid in pw[1:]]
    recs_bin = [{"nll": v, "unk": pid not in kept,
                 "nbytes": surf[pid][2], "nchars": surf[pid][1]}
                for v, pid in zip(nb, flat_pids)]
    recs_pt = [{"nll": v, "unk": pid not in kept,
                "nbytes": surf[pid][2], "nchars": surf[pid][1]}
               for v, pid in zip(np_, flat_pids)]
    vb, vp = variantes(recs_bin, chars), variantes(recs_pt, chars)
    dabc = {v: vb[v] - vp[v] for v in ("a", "b", "c")}
    print(f"[parity] BPC bin a/b/c: {vb['a']:.4f}/{vb['b']:.4f}/{vb['c']:.4f}",
          flush=True)
    print(f"[parity] BPC pt  a/b/c: {vp['a']:.4f}/{vp['b']:.4f}/{vp['c']:.4f}",
          flush=True)
    print(f"[parity] dBPC(bin-pt) a={dabc['a']:+.6f} b={dabc['b']:+.6f} "
          f"c={dabc['c']:+.6f}", flush=True)
    ok = all(abs(dabc[v]) < TOL_DBPC for v in ("a", "b", "c"))
    print(f"[parity] condition |dBPC|<{TOL_DBPC} : "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    res = {"n": len(nb), "p50": p50, "p99": p99, "max": mx,
           "bpc_bin": {k: vb[k] for k in ("a", "b", "c")},
           "bpc_pt": {k: vp[k] for k in ("a", "b", "c")},
           "dbpc": dabc, "tol": TOL_DBPC, "verdict": "PASS" if ok else "FAIL"}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[parity] TERMINE en {__import__('time').time() - t0:.0f}s "
          f"-> {a.out}", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
