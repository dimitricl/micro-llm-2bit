"""BPC équitable (G1) : 3 variantes sur les MÊMES documents et positions.

Variantes (NLL en nats, positions = tokens cibles) :
 a) BPC actuel : sum(NLL) / ln2 / chars (référence, doit retrouver les
    valeurs C2 : 1,5676 / 1,6566 / 1,5357 / 1,3362 / 1,1138).
 b) BPC "unk pénalisé" : à chaque position unk, NLL <- max(NLL, 8*octets*ln2)
    (plancher byte-uniforme : 8 bits par octet du token d'origine).
 c) BPC "in-vocab only" : positions unk exclues du numérateur ET du
    dénominateur (chars des surfaces unk retirés).

"unk" = token cible hors vocab réduit 12000. Pour SmolLM-135M/360M et les
.bin (même tokenizer, positions 1-1 avec le stream réduit) : parent_id
hors kept_ids. Pour Qwen3 (autre tokenizer) : token dont la surface,
re-tokenisée SmolLM, contient un id hors kept_ids.
Les surfaces BPE (byte-level) partitionnent le texte : chars bruts moins
surfaces unk = chars in-vocab exacts (hors eos, compté comme en (a)).

Usage : uv run python tools/eval_bpc.py [--subset JSON] [--out JSON]
Le subset par défaut = les 6 docs val C2 (354 234 chars).
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

import data as D
import engine as E

SUBSET_DEFAULT = "/tmp/c2_subset.json"
VOCAB_PATH = "exports/tranche00.bin.vocab.json"
LN2 = math.log(2)


def surfaces(tokenizer, ids: list[int]) -> dict[int, tuple[str, int, int]]:
    """Surface (texte, nb chars, nb octets) par id, décodé une fois chacun."""
    out = {}
    for i in set(ids):
        t = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        out[i] = (t, len(t), len(t.encode("utf-8")))
    return out


def variantes(records: list[dict], chars: int) -> dict:
    """records : [{nll, unk, nbytes, nchars}]. Retourne {a, b, c, ...}."""
    nll = np.array([r["nll"] for r in records], dtype=np.float64)
    unk = np.array([r["unk"] for r in records])
    nbytes = np.array([r["nbytes"] for r in records], dtype=np.float64)
    nchars = np.array([r["nchars"] for r in records], dtype=np.float64)
    a = float(nll.sum() / LN2 / chars)
    pen = np.maximum(nll, 8.0 * nbytes * LN2)
    pen[~unk] = nll[~unk]
    b = float(pen.sum() / LN2 / chars)
    denom_c = chars - float(nchars[unk].sum())
    c = float(nll[~unk].sum() / LN2 / denom_c) if denom_c > 0 else float("nan")
    return {"a": a, "b": b, "c": c, "n": len(records),
            "n_unk": int(unk.sum()),
            "chars_unk": float(nchars[unk].sum())}


def collect_parent(name: str, texts: list[str], kept: set[int],
                   smol_tok=None, S: int = 512, bs: int = 8,
                   device: str = "mps") -> tuple[list[dict], int]:
    """NLL par position (parent HF, fp32). Retourne (records, chars)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32).to(device).eval()
    dev = torch.device(device)
    own_tok = (smol_tok is None) or (
        tok.vocab_size == smol_tok.vocab_size)
    all_ids = []
    for t in texts:
        all_ids += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    surf = surfaces(tok, all_ids)
    if not own_tok:
        # Qwen : unk si la surface re-tokenisée SmolLM sort du vocab réduit.
        smol_hits = {}
        for i, (txt, _, _) in surf.items():
            back = smol_tok.encode(txt, add_special_tokens=False)
            smol_hits[i] = any(o not in kept for o in back)
    records = []
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
            wins = [ids[i:i + S + 1] for i in range(0, len(ids) - 1, S)]
            for i in range(0, len(wins), bs):
                chunk = wins[i:i + bs]
                L = max(len(w) for w in chunk)
                pad = tok.eos_token_id
                x = torch.tensor(
                    [w[:-1] + [pad] * (L - len(w)) for w in chunk],
                    dtype=torch.long).to(dev)
                y = torch.tensor(
                    [w[1:] + [-100] * (L - len(w)) for w in chunk],
                    dtype=torch.long).to(dev)
                logits = model(input_ids=x).logits.float()
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                    ignore_index=-100, reduction="none")
                nll = nll.view(len(chunk), -1)
                for r, w in enumerate(chunk):
                    for j in range(len(w) - 1):
                        pid = w[j + 1]
                        if own_tok:
                            is_unk = pid not in kept
                        else:
                            is_unk = smol_hits[pid]
                        _, nc, nb = surf[pid]
                        records.append({"nll": float(nll[r, j]),
                                        "unk": bool(is_unk),
                                        "nbytes": nb, "nchars": nc})
    chars = sum(len(t) for t in texts)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return records, chars


def collect_engine(model_path: str, vocab: dict, texts: list[str],
                   parent_ids: list[int]) -> tuple[list[dict], int]:
    """NLL par position (moteur .bin, procédure ppl_moteur à l'identique).

    parent_ids : stream parent aligné 1-1 avec le stream réduit (même
    tokenisation SmolLM) : donne l'id d'origine aux positions unk.
    """
    from main import load_tokenizer

    tok = load_tokenizer(vocab["parent"])
    kept = set(vocab["kept_ids"])
    surf = surfaces(tok, parent_ids)
    new_of = {o: n for n, o in enumerate(vocab["kept_ids"]) if o >= 0}
    # Réduit 1-1 depuis les ids parents fournis (même tokenisation SmolLM).
    # Découpe par doc : on re-marche les docs pour retrouver les frontières
    # (add_eos par doc, comme ppl_moteur via encode_mapped).
    ids, pids, off = [], [], 0
    for t in texts:
        pdoc = parent_ids[off:off + len(
            tok.encode(t, add_special_tokens=False)) + 1]
        off += len(pdoc)
        ids += [new_of.get(o, vocab["unk_new"]) for o in pdoc]
        pids += pdoc
    assert off == len(parent_ids) and pids == parent_ids, "streams désalignés"
    eng = E.Engine(model_path)
    S = eng.meta["seq_max"]
    records = []
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
            s = e.sum()
            nll = float(np.log(e[win[i]] / s))
            pid = pw[i]
            _, nc, nb = surf[pid]
            records.append({"nll": nll, "unk": pid not in kept,
                            "nbytes": nb, "nchars": nc})
            if i + 1 < len(win):
                prev = eng.step(win[i])
    chars = sum(len(t) for t in texts)
    return records, chars


def main() -> None:
    ap = argparse.ArgumentParser(description="BPC équitable a/b/c.")
    ap.add_argument("--subset", default=SUBSET_DEFAULT)
    ap.add_argument("--out", default="/tmp/bpc.json")
    ap.add_argument("--skip-parents", action="store_true")
    ap.add_argument("--skip-bins", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    texts = json.load(open(a.subset, encoding="utf-8"))
    print(f"[bpc] {len(texts)} docs", flush=True)
    vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
    kept = set(vocab["kept_ids"])
    from main import load_tokenizer
    smol_tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")
    res = {}
    if not a.skip_bins:
        parent_ids = []
        for t in texts:
            parent_ids += smol_tok.encode(t, add_special_tokens=False) + [
                smol_tok.eos_token_id]
        for name, path in (("tranche00", "exports/tranche00.bin"),
                           ("wiki", "exports/wiki_slice00_bestval.bin")):
            rec, chars = collect_engine(path, vocab, texts, parent_ids)
            res[name] = variantes(rec, chars)
            print(f"[bpc] {name}: a={res[name]['a']:.4f} "
                  f"b={res[name]['b']:.4f} c={res[name]['c']:.4f} "
                  f"(unk {res[name]['n_unk']}/{res[name]['n']})", flush=True)
    if not a.skip_parents:
        dev = ("mps" if torch.backends.mps.is_available() else "cpu")
        for name in ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M",
                     "Qwen/Qwen3-0.6B"):
            tag = {"HuggingFaceTB/SmolLM-135M": "smol135",
                   "HuggingFaceTB/SmolLM-360M": "smol360",
                   "Qwen/Qwen3-0.6B": "qwen" }[name]
            rec, chars = collect_parent(name, texts, kept, smol_tok,
                                        device=dev)
            res[tag] = variantes(rec, chars)
            print(f"[bpc] {tag}: a={res[tag]['a']:.4f} "
                  f"b={res[tag]['b']:.4f} c={res[tag]['c']:.4f} "
                  f"(unk {res[tag]['n_unk']}/{res[tag]['n']})", flush=True)
    if "tranche00" in res and "smol135" in res:
        for v in ("a", "b", "c"):
            gap = ((res["tranche00"][v] - res["smol135"][v])
                   / res["smol135"][v])
            print(f"[bpc] gap tranche00 vs 135M ({v}) : {gap*100:+.2f} %",
                  flush=True)
            res[f"gap_{v}"] = gap
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[bpc] TERMINE en {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
