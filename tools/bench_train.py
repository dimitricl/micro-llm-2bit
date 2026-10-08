"""Benchmark d'entraînement : où part le temps, et quels leviers l'accélèrent.

NE PAS LANCER pendant un entraînement ou un job GPU/Ollama : les chiffres
seraient faux. Le script vérifie d'abord (precheck) et écrit un
avertissement dans le rapport si la machine est occupée.

Mesures (moyenne sur ~30 steps après 5 de warmup, deadline ~2 min/config) :
1. Profil par étape : forward parent / forward enfant / loss / backward /
   optimiseur (torch.mps.synchronize() avant chaque mesure).
2. Matrice : enfant {tiny, base} x parent {135M, 360M} x micro-batch
   {2, 4, 8, 16} (OOM sautées proprement).
3. Autocast bf16/fp16 sur le parent (avec contrôle de finitude).
4. Cache top-k vs parent direct : écart de loss + vitesse.

Sortie : tools/bench_train.md (tableau tok/s + RAM max + recommandation).

Usage : .venv/bin/python tools/bench_train.py [--out-md ...] [--seq 128]
"""

from __future__ import annotations

import dataclasses
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

OUT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_train.md")

PHRASES = [
    "Paris est la capitale de la France.",
    "L'eau bout à cent degrés au niveau de la mer.",
    "Le chat dort sur le canapé du salon.",
    "La photosynthèse convertit la lumière en énergie.",
    "Le Japon est un archipel d'Asie de l'Est.",
    "L'ordinateur calcule avec des transistors.",
    "La Tour Eiffel mesure trois cent trente mètres.",
    "Le pain lève grâce à la fermentation.",
    "Les abeilles butinent les fleurs au printemps.",
    "La lune tourne autour de la Terre en vingt-sept jours.",
]


# ---------------------------------------------------------------------------
# Pré-vérification : ne pas benchmarker sur une machine occupée
# ---------------------------------------------------------------------------


def precheck() -> list[str]:
    """Retourne la liste des avertissements (vide = machine libre)."""
    warns: list[str] = []
    try:
        ps = subprocess.run(
            ["pgrep", "-af", "main.py train|gen_think|fetch_wiki"],
            capture_output=True, text=True, timeout=10,
        )
        for line in ps.stdout.strip().split("\n"):
            if line.strip():
                warns.append(f"process concurrent : {line.strip()[:100]}")
    except Exception:
        pass
    try:
        import torch as _t

        if hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
            alloc = _t.mps.current_allocated_memory() / 1024 / 1024
            if alloc > 200:
                warns.append(f"MPS déjà allouée : {alloc:.0f} Mo (GPU occupé ?)")
    except Exception:
        pass
    try:
        ps2 = subprocess.run(
            ["pgrep", "-af", "llama-server.*--model"],
            capture_output=True, text=True, timeout=10,
        )
        if ps2.stdout.strip():
            warns.append("Ollama (llama-server) actif : ne pas lancer de job Ollama")
    except Exception:
        pass
    return warns


def sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def is_oom(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return ("out of memory" in msg or "insufficient memory" in msg
            or "outofmemory" in msg)


# ---------------------------------------------------------------------------
# Une mesure profilée (timing uniquement : ids aléatoires valides)
# ---------------------------------------------------------------------------


def mesure_config(child_name: str, parent_name: str, micro_batch: int,
                  seq: int = 128, steps: int = 30, warmup: int = 5,
                  autocast: str = "", budget_s: float = 120.0) -> dict:
    """Retourne {tok_s, ms_parent, ms_enfant, ms_loss, ms_backward, ms_optim,
    ram_max, oom, finis, note}. Ne lève pas sur OOM (marque oom=True)."""
    from model import CONFIGS, TinyTransformer
    from distill import DistillationLoss

    t_start = time.time()
    res: dict = {"oom": False, "finis": False, "note": ""}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = (torch.device("mps") if hasattr(torch.backends, "mps")
               and torch.backends.mps.is_available() else torch.device("cpu"))
        tok = AutoTokenizer.from_pretrained(parent_name, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        P = tok.vocab_size
        kept = list(range(min(4000, P)))
        kept_t = torch.tensor(kept, device=dev)
        parent = AutoModelForCausalLM.from_pretrained(
            parent_name, dtype=torch.float16 if dev.type == "mps" else torch.float32)
        parent.to(dev).eval()
        for p in parent.parameters():
            p.requires_grad_(False)
        cfg = dataclasses.replace(CONFIGS[child_name], vocab_size=len(kept),
                                  seq_max=seq)
        student = TinyTransformer(cfg).to(dev).train()
        opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
        crit = DistillationLoss(0.7, 2.0)
        B, S, V = micro_batch, seq, len(kept)
        acc = {"p": 0.0, "e": 0.0, "l": 0.0, "b": 0.0, "o": 0.0}
        n_tok, n_steps = 0, 0
        ram_max = rss_mb()

        def ac():
            if autocast in ("bf16", "fp16"):
                dt = torch.bfloat16 if autocast == "bf16" else torch.float16
                return torch.autocast(device_type="mps", dtype=dt)
            import contextlib
            return contextlib.nullcontext()

        for it in range(warmup + steps):
            if time.time() - t_start > budget_s:
                res["note"] = f"deadline {budget_s:.0f}s, {n_steps} steps mesurés"
                break
            px = torch.randint(0, P, (B, S), device=dev)
            y = torch.randint(0, V, (B, S), device=dev)
            opt.zero_grad(set_to_none=True)
            sync(dev); t0 = time.perf_counter()
            with torch.no_grad(), ac():
                t_full = parent(input_ids=px).logits.float()
            sync(dev); t1 = time.perf_counter()
            if it >= warmup and not torch.isfinite(t_full).all():
                res["finis"] = False
                res["note"] = f"logits parent non finis ({autocast})"
                break
            t_logits = t_full.index_select(-1, kept_t)
            x = (px % V)
            s_logits = student(x)
            sync(dev); t2 = time.perf_counter()
            loss, _ = crit(s_logits, t_logits, y)
            sync(dev); t3 = time.perf_counter()
            loss.backward()
            sync(dev); t4 = time.perf_counter()
            opt.step()
            sync(dev); t5 = time.perf_counter()
            if it >= warmup:
                acc["p"] += t1 - t0; acc["e"] += t2 - t1; acc["l"] += t3 - t2
                acc["b"] += t4 - t3; acc["o"] += t5 - t4
                n_tok += B * S; n_steps += 1
            ram_max = max(ram_max, rss_mb())
        dt = time.time() - t_start
        if n_steps == 0 and not res["note"]:
            res["note"] = "0 step mesuré"
        res.update({
            "tok_s": n_tok / max(dt, 1e-9), "ram_max": ram_max,
            "ms_parent": acc["p"] / max(n_steps, 1) * 1000,
            "ms_enfant": acc["e"] / max(n_steps, 1) * 1000,
            "ms_loss": acc["l"] / max(n_steps, 1) * 1000,
            "ms_backward": acc["b"] / max(n_steps, 1) * 1000,
            "ms_optim": acc["o"] / max(n_steps, 1) * 1000,
            "n_steps": n_steps, "finis": n_steps > 0,
        })
    except Exception as e:
        if is_oom(e):
            res.update({"oom": True, "note": f"OOM : {str(e)[:120]}"})
        else:
            res.update({"note": f"ERREUR : {type(e).__name__}: {str(e)[:120]}"})
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return res


# ---------------------------------------------------------------------------
# Comparaison cache top-k vs parent direct (données réelles, écart de loss)
# ---------------------------------------------------------------------------


def compare_cache(top_k: int = 32, steps: int = 30) -> dict:
    """Construit le cache sur des phrases réelles, compare loss et vitesse.

    Compare à micro-chunk identique (batch découpé comme en train) :
    loss directe vs loss cache (KL tronquée renormalisée) + temps du
    forward parent seul vs lecture cache + temps d'un step complet caché.
    """
    import data as D
    from distill import build_teacher_cache, teacher_logits_from_cache
    from model import CONFIGS, TinyTransformer
    from distill import DistillationLoss

    out: dict = {}
    parent_name = "HuggingFaceTB/SmolLM-360M"
    MB = 2
    dev = (torch.device("mps") if hasattr(torch.backends, "mps")
           and torch.backends.mps.is_available() else torch.device("cpu"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(parent_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = PHRASES * 40
    vocab = D.build_reduced_vocab(tok, texts, 2000)
    kept = vocab["kept_ids"]
    stream_p, stream_s = [], []
    new_of = {o: n for n, o in enumerate(kept)}
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
        stream_p += ids
        stream_s += [new_of.get(i, vocab["unk_new"]) for i in ids]
    pb = D.make_batches(stream_p, 64, 4, shuffle=False, seed=0)
    sb = D.make_batches(stream_s, 64, 4, shuffle=False, seed=0)
    parent = AutoModelForCausalLM.from_pretrained(
        parent_name, dtype=torch.float16 if dev.type == "mps" else torch.float32)
    parent.to(dev).eval()
    for p in parent.parameters():
        p.requires_grad_(False)
    cache = build_teacher_cache(parent, pb, sb, kept, top_k, dev,
                                micro_batch=MB, prefix="/tmp/cachetest")
    cfg = dataclasses.replace(CONFIGS["tiny"], vocab_size=len(kept), seq_max=64)
    student = TinyTransformer(cfg).to(dev).train()
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    crit = DistillationLoss(0.7, 2.0)
    kept_t = torch.tensor(kept, device=dev)
    chunks = [(bi, k) for bi in range(len(pb))
              for k in range((pb[bi].shape[0] + MB - 1) // MB)][:steps]
    # 1) Écart de loss à entrées identiques.
    ld, lc = [], []
    with torch.no_grad():
        for bi, k in chunks:
            off = k * MB
            xb = sb[bi][off:off + MB][:, :-1].to(dev)
            yb = sb[bi][off:off + MB][:, 1:].to(dev)
            t_full = parent(
                input_ids=pb[bi][off:off + MB][:, :-1].to(dev)).logits.float()
            ref = crit(student(xb), t_full.index_select(-1, kept_t), yb)[0].item()
            got = crit(student(xb),
                       teacher_logits_from_cache(cache, bi, k,
                                                 len(kept), dev),
                       yb)[0].item()
            ld.append(ref); lc.append(got)
    import numpy as np
    out["loss_direct"] = float(np.mean(ld))
    out["loss_cache"] = float(np.mean(lc))
    out["ecart_relatif"] = (abs(out["loss_direct"] - out["loss_cache"])
                            / max(abs(out["loss_direct"]), 1e-9))
    # 2) Vitesse : parent seul vs lecture cache (mêmes chunks).
    sync(dev); t0 = time.perf_counter()
    with torch.no_grad():
        for bi, k in chunks:
            off = k * MB
            parent(input_ids=pb[bi][off:off + MB][:, :-1].to(dev)).logits
    sync(dev)
    out["parent_s"] = time.perf_counter() - t0
    sync(dev); t0 = time.perf_counter()
    for bi, k in chunks:
        teacher_logits_from_cache(cache, bi, k, len(kept), dev)
    sync(dev)
    out["cache_s"] = time.perf_counter() - t0
    # 3) Step complet caché (enfant + loss + backward + optim).
    sync(dev); t0 = time.perf_counter()
    n_tok = 0
    for bi, k in chunks:
        off = k * MB
        xb = sb[bi][off:off + MB][:, :-1].to(dev)
        yb = sb[bi][off:off + MB][:, 1:].to(dev)
        opt.zero_grad(set_to_none=True)
        tl = teacher_logits_from_cache(cache, bi, k, len(kept), dev)
        loss, _ = crit(student(xb), tl, yb)
        loss.backward()
        opt.step()
        n_tok += xb.numel()
    sync(dev)
    out["step_cache_s"] = time.perf_counter() - t0
    out["tok_s_cache"] = n_tok / max(out["step_cache_s"], 1e-9)
    out["n"] = len(chunks)
    return out


# ---------------------------------------------------------------------------
# Matrice complète + rapport markdown
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Bench d'entraînement (court).")
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--quick", action="store_true",
                    help="Matrice réduite (tiny/base x 360M x bs 2/8).")
    a = ap.parse_args()

    warns = precheck()
    lignes = ["# Bench entraînement", ""]
    if warns:
        lignes += ["## AVERTISSEMENT : machine occupée (chiffres non fiables)",
                   *[f"- {w}" for w in warns], ""]
    enfants = ["tiny", "base"]
    parents = (["HuggingFaceTB/SmolLM-360M"] if a.quick
               else ["HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M"])
    bss = [2, 8] if a.quick else [2, 4, 8, 16]
    lignes += ["## Matrice tok/s (30 steps après 5 warmup, ~2 min max/config)",
               "", "| enfant | parent | bs | tok/s | parent ms | enfant ms | "
               "loss ms | backward ms | optim ms | RAM max Mo | note |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    best = None
    for ce in enfants:
        for pa in parents:
            for bs in bss:
                r = mesure_config(ce, pa, bs, seq=a.seq, steps=a.steps)
                tag = "OOM" if r.get("oom") else f"{r.get('tok_s', 0):.0f}"
                if not r.get("oom") and r.get("n_steps", 0) > 0:
                    key = (ce, pa, bs)
                    if best is None or r["tok_s"] > best[1]["tok_s"]:
                        best = (key, r)
                lignes.append(
                    f"| {ce} | {pa.split('/')[-1]} | {bs} | {tag} | "
                    f"{r.get('ms_parent', 0):.0f} | {r.get('ms_enfant', 0):.0f} | "
                    f"{r.get('ms_loss', 0):.1f} | {r.get('ms_backward', 0):.0f} | "
                    f"{r.get('ms_optim', 0):.1f} | {r.get('ram_max', 0):.0f} | "
                    f"{r.get('note', '')} |")
                print(lignes[-1], flush=True)
    lignes += [""]
    # Autocast sur tiny/360M/bs2.
    lignes += ["## Autocast parent (tiny, 360M, bs 2)",
               "", "| mode | tok/s | logits finis | note |",
               "|---|---|---|---|"]
    for mode in ["", "fp16", "bf16"]:
        r = mesure_config("tiny", "HuggingFaceTB/SmolLM-360M", 2,
                          seq=a.seq, steps=a.steps, autocast=mode)
        lignes.append(f"| {mode or 'fp32'} | {r.get('tok_s', 0):.0f} | "
                      f"{'non' if 'non finis' in r.get('note', '') else 'oui'} | "
                      f"{r.get('note', '')} |")
        print(lignes[-1], flush=True)
    lignes += [""]
    # Cache vs direct.
    lignes += ["## Cache top-32 vs parent direct (tiny, phrases réelles)"]
    try:
        c = compare_cache()
        lignes += ["", f"- loss directe : {c['loss_direct']:.4f}",
                   f"- loss cache top-32 (renormalisée) : {c['loss_cache']:.4f}",
                   f"- écart relatif : {c['ecart_relatif'] * 100:.2f} %",
                   f"- temps parent seul ({c['n']} chunks) : {c['parent_s']:.1f}s",
                   f"- temps lecture cache ({c['n']} chunks) : {c['cache_s']:.1f}s",
                   f"- ratio parent/cache : "
                   f"{c['parent_s'] / max(c['cache_s'], 1e-9):.1f}x",
                   f"- step complet caché : {c['tok_s_cache']:.0f} tok/s", ""]
    except Exception as e:
        lignes += ["", f"- comparaison impossible : {type(e).__name__}: {e}", ""]
    if best:
        (ce, pa, bs), r = best
        lignes += [f"## Recommandation : enfant {ce}, parent {pa.split('/')[-1]}, "
                   f"micro-batch {bs} ({r['tok_s']:.0f} tok/s)",
                   "",
                   f"Tranche 100 Mo (~25M tokens) : "
                   f"{25e6 / max(r['tok_s'], 1) / 3600:.1f}h/epoch.", ""]
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes) + "\n")
    print(f"[bench-train] rapport -> {a.out_md}")


if __name__ == "__main__":
    main()
