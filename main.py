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
    if a.think:
        out = eng.generate_think(
            ids,
            max_new=a.max_new,
            temperature=a.temperature if a.temperature > 0 else 0.7,
            top_k=a.top_k if a.top_k > 0 else 40,
            top_p=a.top_p if a.top_p < 1.0 else 0.9,
        )
        dt = time.time() - t0
        text = decode_ids(tok, vocab, out)
        parsed = E.parse_think(text)
        if parsed:
            think, response = parsed
            print(f"=== RAISONNEMENT ===\n{think}\n")
            print(f"=== RÉPONSE ===\n{response}")
        else:
            print("[infer] Format think non détecté ; texte brut :")
            print(text)
        print(
            f"[infer] mode think : {len(out)} tokens en {dt:.2f}s = "
            f"{len(out) / max(dt, 1e-6):.1f} tok/s",
            file=sys.stderr,
        )
        return
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


def prepare_corpus(tok, a: argparse.Namespace, save_vocab_to: str = ""):
    """Charge textes, split val, vocab, batchs parent/enfant alignés.

    Retourne (pb, sb, vpb, vsb, vocab). Même code pour `train` et
    `cache-logits` : le cache pré-calculé correspond exactement aux batchs
    d'entraînement (même seed, même découpage).
    """
    texts = D.load_texts(a.data)
    print(f"[corpus] {len(texts)} documents, {sum(len(t) for t in texts)} caractères")
    val_ratio = getattr(a, "val_ratio", 0.0) or 0.0
    val_texts: list[str] = []
    if val_ratio > 0 and len(texts) > 1:
        texts, val_texts = D.train_val_split(texts, val_ratio, getattr(a, "seed", 0))
        print(f"[corpus] split : {len(texts)} docs train, {len(val_texts)} docs val")
    if getattr(a, "vocab_from", ""):
        with open(a.vocab_from, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"[corpus] vocab réutilisé : {a.vocab_from} "
              f"({len(vocab['kept_ids'])} tokens)")
    else:
        vocab = D.build_reduced_vocab(tok, texts, a.vocab_size)
    if save_vocab_to:
        with open(save_vocab_to, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
    # Deux flux parallèles alignés (mapping 1 token -> 1 token) : ids parent
    # pour le teacher forcing du parent, ids réduits pour l'enfant.
    parent_stream: list[int] = []
    for t in texts:
        parent_stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
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
    # Batchs val alignés (même seed, sans mélange) pour l'éval périodique.
    vpb, vsb = [], []
    if val_texts:
        val_parent_stream: list[int] = []
        for t in val_texts:
            val_parent_stream += tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
        val_student_stream = [new_of.get(t, vocab["unk_new"]) for t in val_parent_stream]
        vpb = D.make_batches(
            val_parent_stream, a.seq_len, a.save_batch, shuffle=False, seed=a.seed
        )
        vsb = D.make_batches(
            val_student_stream, a.seq_len, a.save_batch, shuffle=False, seed=a.seed
        )
        assert len(vpb) == len(vsb), "Flux val désalignés."
    return pb, sb, vpb, vsb, vocab


def cmd_train(a: argparse.Namespace) -> None:
    tok = load_tokenizer(a.parent)
    pb, sb, vpb, vsb, vocab = prepare_corpus(
        tok, a, save_vocab_to=a.vocab_out or (a.out + ".vocab.json"))
    cfg = CONFIGS[a.config]
    import dataclasses

    cfg = dataclasses.replace(cfg, vocab_size=len(vocab["kept_ids"]), seq_max=a.seq_len)
    student = TinyTransformer(cfg)
    cache = a.cache or (a.out + ".teacherk")
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
        cache_k=getattr(a, "cache_k", 32),
        resume=a.resume,
        seed=a.seed,
        ckpt_every=getattr(a, "ckpt_every", 500),
        device=None,
        val_parent_batches=vpb,
        val_student_batches=vsb,
        eval_every=getattr(a, "eval_every", 0),
        patience=getattr(a, "patience", 0),
        lr_resume=getattr(a, "lr_resume", 0.0),
        rewarmup_ratio=getattr(a, "rewarmup_ratio", 0.0),
    )


# ---------------------------------------------------------------------------
# cache-logits
# ---------------------------------------------------------------------------


def cmd_cache_logits(a: argparse.Namespace) -> None:
    """Pré-calcule le cache top-k parent (train + val) sans entraîner.

    Utilise exactement le même prepare_corpus que `train` (même seed,
    même découpage) : le cache correspond aux batchs d'entraînement.
    Le train suivant utilisera --top-k-cache 1 --cache <out-dir>/train
    avec le MÊME --batch-size (vérifié au chargement).
    """
    tok = load_tokenizer(a.parent)
    os.makedirs(a.out_dir, exist_ok=True)
    pb, sb, vpb, vsb, vocab = prepare_corpus(
        tok, a, save_vocab_to=os.path.join(a.out_dir, "vocab.json"))
    device = dist.pick_device()
    print(f"[cache-logits] device={device} parent={a.parent}")
    from transformers import AutoModelForCausalLM

    dtype = dist.parent_dtype(device)
    parent = AutoModelForCausalLM.from_pretrained(a.parent, dtype=dtype)
    parent.to(device).eval()
    for p in parent.parameters():
        p.requires_grad_(False)
    kept = vocab["kept_ids"]
    dist.build_teacher_cache(parent, pb, sb, kept, a.cache_k, device,
                             a.batch_size, os.path.join(a.out_dir, "train"),
                             parent_name=a.parent)
    if vpb:
        dist.build_teacher_cache(parent, vpb, vsb, kept, a.cache_k, device,
                                 a.batch_size, os.path.join(a.out_dir, "val"),
                                 parent_name=a.parent)
    else:
        print("[cache-logits] pas de split val (--val-ratio 0) : cache train seul.")


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
    # Baseline : Python + imports déjà chargés, avant tout chargement modèle.
    baseline = rss_mb()
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
    delta_load = after_load - baseline
    delta_gen = after_gen - baseline
    ok_theory = total < 100 * 1024 * 1024
    # Critère de réussite : le DELTA RSS réel après génération < 100 Mo
    # (pas la RSS absolue, qui inclut Python + torch déjà chargés).
    ok_delta = delta_gen < 100.0
    ok = ok_theory and ok_delta
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
        f"[bench] RSS réel    : baseline={baseline:.1f} Mo, après chargement={after_load:.1f} Mo (+{delta_load:.1f}), après génération={after_gen:.1f} Mo (+{delta_gen:.1f})"
    )
    print(
        f"[bench] DELTA vs baseline : +{delta_gen:.1f} Mo / 100 Mo -> {'OK' if ok_delta else 'DÉPASSEMENT'}"
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
    pi.add_argument(
        "--think",
        action="store_true",
        help="Parse une trace <think>...</think> et une réponse finale.",
    )

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
    pt.add_argument("--cache-k", type=int, default=32,
                    help="Top-k stocké dans le cache mmap (défaut 32).")
    pt.add_argument("--cache", default="")
    pt.add_argument("--resume", default="")
    pt.add_argument("--ckpt-every", type=int, default=500,
                    help="Checkpoint périodique complet tous les N steps "
                    "(0 = désactivé, 2 derniers gardés).")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--val-ratio", type=float, default=0.0,
                    help="Part des docs en validation (0 = désactivé).")
    pt.add_argument("--eval-every", type=int, default=0,
                    help="Évalue la val tous les N steps (0 = désactivé).")
    pt.add_argument("--patience", type=int, default=0,
                    help="Early stopping après N evals sans progrès (0 = désactivé).")
    pt.add_argument("--vocab-from", default="",
                    help="Réutilise un .vocab.json existant au lieu de recalculer "
                    "(requis pour enchaîner les tranches à vocab constant).")
    pt.add_argument("--lr-resume", type=float, default=0.0,
                    help="Remplace le LR à la reprise (0 = garde celui du checkpoint).")
    pt.add_argument("--rewarmup-ratio", type=float, default=0.0,
                    help="Reconstruit le scheduler sur les steps restants à la "
                    "reprise (0 = garde celui du checkpoint).")

    pe = sub.add_parser("export", help="Quantifie et packe en .bin.")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--out", required=True)

    pc = sub.add_parser("cache-logits",
                        help="Pré-calcule le cache top-k parent (train+val).")
    pc.add_argument("--data", required=True, nargs="+")
    pc.add_argument("--parent", required=True)
    pc.add_argument("--out-dir", required=True,
                    help="Dossier : vocab.json + train.* + val.* (mmap).")
    pc.add_argument("--vocab-size", type=int, default=12000)
    pc.add_argument("--vocab-out", default="")
    pc.add_argument("--seq-len", type=int, default=128)
    pc.add_argument("--save-batch", type=int, default=8)
    pc.add_argument("--batch-size", type=int, default=2)
    pc.add_argument("--val-ratio", type=float, default=0.05)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--vocab-from", default="")
    pc.add_argument("--cache-k", type=int, default=32)

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
        "cache_k",
        "cache",
        "resume",
        "ckpt_every",
        "seed",
        "vocab_out",
        "val_ratio",
        "eval_every",
        "patience",
        "vocab_from",
        "lr_resume",
        "rewarmup_ratio",
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
            "cache_k": 32,
            "cache": "",
            "resume": "",
            "ckpt_every": 500,
            "seed": 0,
            "vocab_out": "",
            "val_ratio": 0.0,
            "eval_every": 0,
            "patience": 0,
            "vocab_from": "",
            "lr_resume": 0.0,
            "rewarmup_ratio": 0.0,
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
        "cache-logits": cmd_cache_logits,
        "export": cmd_export,
        "bench": cmd_bench,
        "distill-loop": cmd_loop,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
