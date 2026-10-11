"""Eval val élargie (G2) : BPC/PPL + bootstrap par document (IC 95 %).

Protocole UNIFORME tous modèles : fenêtres S=128 par doc (+eos/doc),
mêmes docs, mêmes positions (tokenisation SmolLM 1-1 pour 135M/360M et
.pt ; Qwen exclu ici — voir eval_bpc.py). Bootstrap : 1000 tirages avec
remise sur les docs (numpy, seed fixée), IC 95 % (percentiles 2.5/97.5)
sur BPC et PPL, par variante G1 (a/b/c).
Choix .pt vs .bin : le .bin (moteur token-à-token, ~123 tok/s) coûterait
~6h sur val_big (7,7M chars) pour des chiffres à ±5 % du .pt (règle de
fidélité eval_ppl) ; le .pt (MPS par lots) prend ~10 min à protocole
identique. Le .bin n'est utilisé qu'en spot-check fidélité (G2) et pour
eval_fixed (boucles). Le critère tranche01 (G4) compare du .pt au .pt.

Usage :
  uv run python tools/eval_valbig.py --ckpt runs/.../tranche00.pt \\
      --docs runs/abl_data/val_big.jsonl --vocab exports/tranche00.bin.vocab.json \\
      --out /tmp/vb_tr.json
  uv run python tools/eval_valbig.py --parent HuggingFaceTB/SmolLM-135M \\
      --docs ... --vocab ... --out /tmp/vb_par.json
  uv run python tools/eval_valbig.py --bin exports/wiki_slice00_bestval.bin \\
      --docs ... --vocab ... --max-chars 350000 --out /tmp/vb_bin.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2)
S = 128


def load_docs(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line).get("text", "")
                if t.strip():
                    out.append(t)
    return out


def doc_windows(ids: list[int]) -> list[list[int]]:
    wins = [ids[i:i + S + 1] for i in range(0, len(ids) - 1, S)]
    return [w for w in wins if len(w) == S + 1]


def variant_sums(nll: np.ndarray, unk: np.ndarray, nbytes: np.ndarray,
                 nchars: np.ndarray) -> dict:
    """Sommes NLL par variante sur un vecteur de positions (un doc)."""
    pen = np.maximum(nll, 8.0 * nbytes * LN2)
    pen[~unk] = nll[~unk]
    return {"nll_a": float(nll.sum()), "nll_b": float(pen.sum()),
            "nll_c": float(nll[~unk].sum()),
            "chars_c": float(nchars[~unk].sum()), "n": len(nll),
            "n_unk": int(unk.sum())}


def collect_pt(ckpt_path: str, docs: list[str], vocab: dict,
               device: str) -> list[dict]:
    """Par doc : sommes NLL a/b/c (modèle .pt, MPS par lots)."""
    from main import load_tokenizer
    from model import TinyTransformer

    tok = load_tokenizer(vocab["parent"])
    kept = set(vocab["kept_ids"])
    new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
    surf: dict[int, tuple[int, int]] = {}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    dev = torch.device(device)
    model.to(dev).eval()
    per_doc = []
    with torch.no_grad():
        for t in docs:
            pids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
            for pid in set(pids):
                if pid not in surf:
                    s = tok.decode([pid], clean_up_tokenization_spaces=False)
                    surf[pid] = (len(s), len(s.encode("utf-8")))
            sids = [new_of.get(o, vocab["unk_new"]) for o in pids]
            nll_all, unk_all, nb_all, nc_all = [], [], [], []
            for i in range(0, len(sids) - 1, S * 16):
                chunk = [sids[j:j + S + 1]
                         for j in range(i, min(i + S * 16, len(sids) - 1), S)]
                chunk = [w for w in chunk if len(w) == S + 1]
                if not chunk:
                    continue
                b = torch.tensor(chunk, dtype=torch.long).to(dev)
                logits = model(b[:, :-1]).float()
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                    reduction="none").view(len(chunk), -1).cpu().numpy()
                for r, w in enumerate(chunk):
                    base = i + r * S
                    for j in range(S):
                        pid = pids[base + j + 1]
                        nll_all.append(float(nll[r, j]))
                        unk_all.append(pid not in kept)
                        nc, nb = surf[pid]
                        nc_all.append(nc)
                        nb_all.append(nb)
            per_doc.append(variant_sums(
                np.array(nll_all), np.array(unk_all),
                np.array(nb_all, dtype=float), np.array(nc_all, dtype=float)))
    return per_doc


def collect_parent(name: str, docs: list[str], kept: set[int],
                   device: str) -> list[dict]:
    """Par doc : sommes NLL a/b/c (parent HF fp32, S=128 comme les .pt)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32).to(device).eval()
    dev = torch.device(device)
    surf: dict[int, tuple[int, int]] = {}
    per_doc = []
    with torch.no_grad():
        for t in docs:
            pids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
            for pid in set(pids):
                if pid not in surf:
                    s = tok.decode([pid], clean_up_tokenization_spaces=False)
                    surf[pid] = (len(s), len(s.encode("utf-8")))
            nll_all, unk_all, nb_all, nc_all = [], [], [], []
            for i in range(0, len(pids) - 1, 16 * S):
                chunk = [pids[j:j + S + 1]
                         for j in range(i, min(i + 16 * S, len(pids) - 1), S)]
                chunk = [w for w in chunk if len(w) == S + 1]
                if not chunk:
                    continue
                L = S
                x = torch.tensor([w[:-1] for w in chunk],
                                 dtype=torch.long).to(dev)
                y = torch.tensor([w[1:] for w in chunk],
                                 dtype=torch.long).to(dev)
                logits = model(input_ids=x).logits.float()
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                    reduction="none").view(len(chunk), -1).cpu().numpy()
                for r, w in enumerate(chunk):
                    base = i + r * S
                    for j in range(S):
                        pid = w[j + 1]
                        nll_all.append(float(nll[r, j]))
                        unk_all.append(pid not in kept)
                        nc, nb = surf[pid]
                        nc_all.append(nc)
                        nb_all.append(nb)
            per_doc.append(variant_sums(
                np.array(nll_all), np.array(unk_all),
                np.array(nb_all, dtype=float), np.array(nc_all, dtype=float)))
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return per_doc


def collect_bin(model_path: str, vocab: dict, docs: list[str]) -> list[dict]:
    """Par doc : sommes NLL a/b/c (moteur .bin, lent, spot-check)."""
    import engine as E
    from main import load_tokenizer

    tok = load_tokenizer(vocab["parent"])
    kept = set(vocab["kept_ids"])
    new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
    surf: dict[int, tuple[int, int]] = {}
    eng = E.Engine(model_path)
    per_doc = []
    for t in docs:
        pids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
        for pid in set(pids):
            if pid not in surf:
                s = tok.decode([pid], clean_up_tokenization_spaces=False)
                surf[pid] = (len(s), len(s.encode("utf-8")))
        ids = [new_of.get(o, vocab["unk_new"]) for o in pids]
        nll_all, unk_all, nb_all, nc_all = [], [], [], []
        for b in range(0, len(ids) - 1, S):
            win = ids[b:b + S + 1]
            pw = pids[b:b + S + 1]
            if len(win) < 2:
                continue
            eng.reset()
            prev = eng.step(win[0])
            for i in range(1, len(win)):
                z = prev.astype(np.float64)
                z -= z.max()
                e = np.exp(z)
                s_ = e.sum()
                nll_all.append(float(np.log(s_) - np.log(e[win[i]])))
                pid = pw[i]
                unk_all.append(pid not in kept)
                nc, nb = surf[pid]
                nc_all.append(nc)
                nb_all.append(nb)
                if i + 1 < len(win):
                    prev = eng.step(win[i])
        per_doc.append(variant_sums(
            np.array(nll_all), np.array(unk_all),
            np.array(nb_all, dtype=float), np.array(nc_all, dtype=float)))
    return per_doc


def bootstrap(per_doc: list[dict], chars_total: int, n_draws: int = 1000,
              seed: int = 0) -> dict:
    """IC 95 % sur BPC/PPL par variante (tirages avec remise sur les docs)."""
    rng = np.random.default_rng(seed)
    n = len(per_doc)
    A = np.array([d["nll_a"] for d in per_doc])
    B = np.array([d["nll_b"] for d in per_doc])
    C = np.array([d["nll_c"] for d in per_doc])
    CC = np.array([d["chars_c"] for d in per_doc])
    N = np.array([d["n"] for d in per_doc], dtype=float)
    out = {}
    # Note : le dénominateur chars est fixe (docs fixés) ; seul le
    # numérateur est bootstrapé (incertitude d'échantillonnage des docs).
    for tag, arr, denom in (("a", A, chars_total), ("b", B, chars_total),
                            ("c", C, None)):
        samp = rng.integers(0, n, size=(n_draws, n))
        if tag == "c":
            num = C[samp].sum(axis=1)
            den = CC[samp].sum(axis=1)
            bpc = num / LN2 / np.maximum(den, 1e-9)
            ppl = np.exp(num / np.maximum(N[samp].sum(axis=1), 1))
        else:
            num = arr[samp].sum(axis=1)
            bpc = num / LN2 / denom
            ppl = np.exp(num / N[samp].sum(axis=1))
        out[tag] = {
            "bpc": float(np.mean(bpc)),
            "bpc_ci": [float(np.quantile(bpc, 0.025)),
                       float(np.quantile(bpc, 0.975))],
            "ppl": float(np.mean(ppl)),
            "ppl_ci": [float(np.quantile(ppl, 0.025)),
                       float(np.quantile(ppl, 0.975))],
        }
    out["n_docs"] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval val élargie + bootstrap.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", default="")
    g.add_argument("--parent", default="")
    g.add_argument("--bin", default="")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="0 = tous les docs, sinon tronque à N chars.")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    docs = load_docs(a.docs)
    if a.max_chars > 0:
        sub, tot = [], 0
        for t in docs:
            sub.append(t)
            tot += len(t)
            if tot >= a.max_chars:
                break
        docs = sub
    chars = sum(len(t) for t in docs)
    print(f"[valbig] {len(docs)} docs, {chars} chars", flush=True)
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    kept = set(vocab["kept_ids"])
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if a.ckpt:
        per_doc = collect_pt(a.ckpt, docs, vocab, dev)
    elif a.parent:
        per_doc = collect_parent(a.parent, docs, kept, dev)
    else:
        per_doc = collect_bin(a.bin, vocab, docs)
    res = bootstrap(per_doc, chars, a.draws, a.seed)
    res["chars"] = chars
    res["n_unk"] = sum(d["n_unk"] for d in per_doc)
    res["n"] = sum(d["n"] for d in per_doc)
    for v in ("a", "b", "c"):
        r = res[v]
        print(f"[valbig] {v}: BPC={r['bpc']:.4f} "
              f"[{r['bpc_ci'][0]:.4f}, {r['bpc_ci'][1]:.4f}] "
              f"PPL={r['ppl']:.2f} "
              f"[{r['ppl_ci'][0]:.2f}, {r['ppl_ci'][1]:.2f}]", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[valbig] TERMINE en {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
