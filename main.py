"""CLI : infer / train / export / bench / distill-loop.

Exemples :
  python main.py infer --model exports/enfant.bin --prompt "Bonjour"
  python main.py train --data ./data --parent HuggingFaceTB/SmolLM-360M --out checkpoints/enfant.pt
  python main.py export --ckpt checkpoints/enfant.pt --out exports/enfant.bin
  python main.py bench --model exports/enfant.bin
  python main.py distill-loop --data ./data --parent HuggingFaceTB/SmolLM-360M --ollama-model smollm2:360m
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import data as D
import distill as dist
import engine as E
from model import CONFIGS, TinyTransformer
from quant import pack_ternary


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def load_tokenizer(parent: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(parent, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def decode_ids(tokenizer, vocab: dict, ids: list[int]) -> str:
    kept = vocab["kept_ids"]
    old = [kept[i] if 0 <= i < len(kept) else kept[vocab["unk_new"]] for i in ids]
    return tokenizer.decode(old, skip_special_tokens=True)


def rss_mb() -> float:
    """RSS réel du processus en Mo (psutil si dispo, sinon getrusage).

    Attention : ru_maxrss est en OCTETS sur macOS, en KILO-octets sur Linux.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import sys as _sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 / 1024 if _sys.platform == "darwin" else rss / 1024


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


def cmd_infer(a: argparse.Namespace) -> None:
    vocab_path = a.vocab or (a.model + ".vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    tok = load_tokenizer(vocab["parent"] if not a.parent else a.parent)
    eng = E.Engine(a.model)
    print(
        f"[infer] noyau={E.KERNEL_NAME} vocab={len(vocab['kept_ids'])}", file=sys.stderr
    )
    ids = D.encode_mapped(tok, vocab, a.prompt, add_eos=False)
    ids = ids[-eng.meta["seq_max"] :]
    t0 = time.time()
    out = eng.generate(
        ids,
        max_new=a.max_new,
        temperature=a.temperature,
        top_k=a.top_k,
        top_p=a.top_p,
        stop={vocab["eos_new"]},
    )
    dt = time.time() - t0
    text = decode_ids(tok, vocab, out)
    print(text)
    print(
        f"[infer] {len(out)} tokens en {dt:.2f}s = {len(out) / max(dt, 1e-6):.1f} tok/s",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(a: argparse.Namespace) -> None:
    tok = load_tokenizer(a.parent)
    texts = D.load_texts(a.data)
    print(f"[train] {len(texts)} documents, {sum(len(t) for t in texts)} caractères")
    vocab = D.build_reduced_vocab(tok, texts, a.vocab_size)
    with open(a.vocab_out or (a.out + ".vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    # Deux flux parallèles alignés (mapping 1 token -> 1 token) : ids parent
    # pour le teacher forcing du parent, ids réduits pour l'enfant.
    parent_stream: list[int] = []
    for t in texts:
        parent_stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
    unk_old = tok.unk_token_id
    new_of = {old: new for new, old in enumerate(vocab["kept_ids"])}
    student_stream = [new_of.get(t, vocab["unk_new"]) for t in parent_stream]
    # Même seed -> même permutation -> batchs alignés parent/enfant.
    pb = D.make_batches(
        parent_stream, a.seq_len, a.save_batch, shuffle=True, seed=a.seed
    )
    sb = D.make_batches(
        student_stream, a.seq_len, a.save_batch, shuffle=True, seed=a.seed
    )
    assert len(pb) == len(sb), "Flux désalignés (ne devrait pas arriver)."
    cfg = CONFIGS[a.config]
    import dataclasses

    cfg = dataclasses.replace(cfg, vocab_size=len(vocab["kept_ids"]), seq_max=a.seq_len)
    student = TinyTransformer(cfg)
    cache = a.cache or (a.out + ".teacherk.npz")
    dist.train(
        student,
        pb,
        sb,
        vocab,
        a.parent,
        a.out,
        alpha=a.alpha,
        temperature=a.temperature,
        hidden_weight=a.hidden_weight,
        lr=a.lr,
        epochs=a.epochs,
        batch_size=a.batch_size,
        accum=a.accum,
        top_k_cache=a.top_k_cache,
        cache_path=cache if a.top_k_cache > 0 else "",
        resume=a.resume,
        device=None,
    )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def cmd_export(a: argparse.Namespace) -> None:
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, vocab = ckpt["cfg"], ckpt["vocab"]
    model = TinyTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tensors: list[np.ndarray] = []
    with torch.no_grad():
        # 1) Embeddings int8 + scales fp32 par ligne.
        emb = model.embed_latent.float()
        amax = emb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        esc = amax / 127.0
        eq = (emb / esc).round().clamp(-128, 127).to(torch.int8)
        tensors += [eq.numpy(), esc.squeeze(1).numpy().astype(np.float32)]
        # 2) Linéaires : packés + scales, dans l'ordre de l'Engine.
        order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"]
        for li, layer in enumerate(model.layers):
            mods = {
                "q_proj": layer.attn.q_proj,
                "k_proj": layer.attn.k_proj,
                "v_proj": layer.attn.v_proj,
                "o_proj": layer.attn.o_proj,
                "gate": layer.mlp.gate,
                "up": layer.mlp.up,
                "down": layer.mlp.down,
            }
            for name in order:
                q, scales = mods[name].ternary_weights()  # (out,in) int8, (out,G)
                packed, n = pack_ternary(q)
                assert n == q.numel()
                tensors += [packed.numpy(), scales.numpy().astype(np.float32)]
            tensors += [
                layer.attn_norm.weight.float().numpy().astype(np.float32),
                layer.mlp_norm.weight.float().numpy().astype(np.float32),
            ]
        tensors += [model.final_norm.weight.float().numpy().astype(np.float32)]
    meta = {
        "vocab": cfg.vocab_size,
        "d": cfg.d_model,
        "L": cfg.n_layers,
        "Hq": cfg.n_heads_q,
        "Hkv": cfg.n_heads_kv,
        "ffn": cfg.ffn_dim,
        "seq_max": cfg.seq_max,
        "group": cfg.group_size,
        "head_dim": cfg.d_model // cfg.n_heads_q,
    }
    E.write_bin(a.out, meta, tensors)
    vpath = a.out + ".vocab.json"
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    size = os.path.getsize(a.out)
    print(f"[export] {a.out} : {size / 1024 / 1024:.2f} Mo (+ {vpath})")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def cmd_bench(a: argparse.Namespace) -> None:
    before = rss_mb()
    eng = E.Engine(a.model)
    after_load = rss_mb()
    m = eng.meta
    # Recalcule le budget théorique depuis les métas du .bin.
    V, Dd, L = m["vocab"], m["d"], m["L"]
    Dh, F, G, S = m["head_dim"], m["ffn"], m["group"], m["seq_max"]
    lin = (2 * (m["Hq"] * Dh * Dd) + 2 * (m["Hkv"] * Dh * Dd) + 3 * (F * Dd)) * L
    weights = lin // 4 + (lin // G) * 4 + V * Dd + V * 4 + (2 * L + 1) * Dd * 4
    kv = 2 * L * S * m["Hkv"] * Dh * 1
    act = (Dd + 2 * F + 2 * m["Hq"] * Dh + V) * 4
    total = weights + kv + act
    # Génération de test : tokens/s.
    ids = list(range(min(16, V)))
    t0 = time.time()
    eng.generate(ids, max_new=a.bench_tokens, temperature=0.0)
    dt = time.time() - t0
    after_gen = rss_mb()
    ok = total < 100 * 1024 * 1024
    print(f"[bench] noyau     : {E.KERNEL_NAME}")
    print(
        f"[bench] config      : V={V} d={Dd} L={L} Hq={m['Hq']} Hkv={m['Hkv']} ffn={F} seq={S}"
    )
    print(
        f"[bench] poids packés: {weights / 1e6:.2f} Mo (fichier: {os.path.getsize(a.model) / 1e6:.2f} Mo)"
    )
    print(f"[bench] KV cache    : {kv / 1e6:.2f} Mo (int8 pré-alloué)")
    print(f"[bench] activations : {act / 1e6:.2f} Mo")
    print(
        f"[bench] TOTAL       : {total / 1024 / 1024:.2f} Mo / 100 Mo -> {'OK' if ok else 'DÉPASSEMENT'}"
    )
    print(
        f"[bench] RSS réel    : avant={before:.1f} Mo, après chargement={after_load:.1f} Mo, après génération={after_gen:.1f} Mo"
    )
    print(f"[bench] vitesse     : {a.bench_tokens / max(dt, 1e-6):.1f} tok/s")
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# distill-loop
# ---------------------------------------------------------------------------


def cmd_loop(a: argparse.Namespace) -> None:
    if not dist.ollama_available():
        print(
            "[loop] Ollama inaccessible (ollama list a échoué). Lancez `ollama serve`.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    texts = D.load_texts(a.data)
    prompts = texts[: a.num_prompts]
    for cycle in range(a.cycles):
        print(f"[loop] === cycle {cycle + 1}/{a.cycles} ===")
        # 1) Entraînement (1 epoch courte) sur les données courantes.
        b = argparse.Namespace(
            **{**vars(a), "epochs": 1, "out": f"{a.out}_c{cycle}.pt"}
        )
        cmd_train(b)
        # 2) Export + génération enfant sur les prompts.
        bin_path = f"{a.out}_c{cycle}.bin"
        cmd_export(argparse.Namespace(ckpt=f"{a.out}_c{cycle}.pt", out=bin_path))
        vocab = json.load(open(bin_path + ".vocab.json", encoding="utf-8"))
        tok = load_tokenizer(vocab["parent"])
        eng = E.Engine(bin_path)
        improved: list[str] = []
        for p in prompts:
            ids = D.encode_mapped(tok, vocab, p[:500], add_eos=False)
            gen = eng.generate(
                ids, max_new=64, temperature=0.0, stop={vocab["eos_new"]}
            )
            answer = decode_ids(tok, vocab, gen)
            better = dist.ollama_judge(a.ollama_model, p[:500], answer)
            improved.append(p + "\n" + better)
        # 3) Les réponses améliorées rejoignent le corpus du cycle suivant.
        texts = texts + improved
        out_dir = a.data[0] if isinstance(a.data, list) else a.data
        os.makedirs(out_dir, exist_ok=True)
        with open(
            os.path.join(out_dir, f"loop_c{cycle}.jsonl"), "w", encoding="utf-8"
        ) as f:
            for t in improved:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"[loop] {len(improved)} réponses améliorées ajoutées au corpus.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Micro-LLM 2 bits : inférence CPU + distillation."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("infer", help="Inférence sur un prompt.")
    pi.add_argument("--model", required=True, help="Chemin .bin (export).")
    pi.add_argument("--prompt", required=True)
    pi.add_argument(
        "--parent", default="", help="Tokenizer HF (défaut: celui de l'export)."
    )
    pi.add_argument("--vocab", default="")
    pi.add_argument("--max-new", type=int, default=64)
    pi.add_argument("--temperature", type=float, default=0.0)
    pi.add_argument("--top-k", type=int, default=0)
    pi.add_argument("--top-p", type=float, default=1.0)

    pt = sub.add_parser("train", help="Distillation parent -> enfant.")
    pt.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Dossier(s) ou fichier(s) .txt/.jsonl/.xml.",
    )
    pt.add_argument("--parent", required=True, help="Nom HF du parent.")
    pt.add_argument("--out", required=True, help="Checkpoint .pt de sortie.")
    pt.add_argument("--config", default="tiny", choices=list(CONFIGS))
    pt.add_argument("--vocab-size", type=int, default=8000)
    pt.add_argument("--vocab-out", default="")
    pt.add_argument("--seq-len", type=int, default=128)
    pt.add_argument(
        "--save-batch", type=int, default=8, help="Batch stocké (découpé ensuite)."
    )
    pt.add_argument(
        "--batch-size", type=int, default=2, help="Micro-batch (RAM unifiée !)."
    )
    pt.add_argument("--accum", type=int, default=8)
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--alpha", type=float, default=0.7)
    pt.add_argument("--temperature", type=float, default=2.0)
    pt.add_argument("--hidden-weight", type=float, default=0.0)
    pt.add_argument("--top-k-cache", type=int, default=0)
    pt.add_argument("--cache", default="")
    pt.add_argument("--resume", default="")
    pt.add_argument("--seed", type=int, default=0)

    pe = sub.add_parser("export", help="Quantifie et packe en .bin.")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--out", required=True)

    pb = sub.add_parser("bench", help="RAM réelle, tok/s, budget 100 Mo.")
    pb.add_argument("--model", required=True)
    pb.add_argument("--bench-tokens", type=int, default=32)

    pl = sub.add_parser(
        "distill-loop", help="Cycles enfant -> parent(Ollama) -> réentraînement."
    )
    pl.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Dossier(s) ou fichier(s) .txt/.jsonl/.xml.",
    )
    pl.add_argument("--parent", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--ollama-model", default="smollm2:360m")
    pl.add_argument("--cycles", type=int, default=2)
    pl.add_argument("--num-prompts", type=int, default=4)
    for name in (
        "config",
        "vocab_size",
        "seq_len",
        "save_batch",
        "batch_size",
        "accum",
        "lr",
        "alpha",
        "temperature",
        "hidden_weight",
        "top_k_cache",
        "cache",
        "resume",
        "seed",
        "vocab_out",
    ):
        d = {
            "vocab_size": 8000,
            "seq_len": 128,
            "save_batch": 8,
            "batch_size": 2,
            "accum": 8,
            "lr": 3e-4,
            "alpha": 0.7,
            "temperature": 2.0,
            "hidden_weight": 0.0,
            "top_k_cache": 0,
            "cache": "",
            "resume": "",
            "seed": 0,
            "vocab_out": "",
            "config": "tiny",
        }[name]
        t = int if isinstance(d, int) else (float if isinstance(d, float) else str)
        pl.add_argument(f"--{name.replace('_', '-')}", default=d, type=t)

    return p


def main() -> None:
    a = build_parser().parse_args()
    {
        "infer": cmd_infer,
        "train": cmd_train,
        "export": cmd_export,
        "bench": cmd_bench,
        "distill-loop": cmd_loop,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
