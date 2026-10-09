"""Distillation parent -> enfant : loss KL + CE (+ option MSE cachée).

- Parent : modèle HF (ex. HuggingFaceTB/SmolLM-360M) en inférence seule, sur
  MPS si dispo sinon CPU. Poids figés (no_grad), éventuellement pré-calculés
  en top-k sur disque pour éviter de recharger le parent à chaque run.
- Enfant : TinyTransformer (poids latents fp32, QAT/STE au forward).
- Loss = alpha * KL(parent/T || enfant/T) * T^2 + (1-alpha) * CE.
- Bonus : MSE entre états cachés (via projection linéaire apprise).
- Optimiseur AdamW, warmup + cosine decay, gradient clipping, accumulation,
  checkpoints reprenables, logs (loss, perplexité, tokens/s).
- La RAM est unifiée sur Mac : petit batch + accumulation par défaut.
"""

from __future__ import annotations

import math
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CONFIGS, TinyTransformer

# Les opérateurs MPS manquants retombent sur CPU (requis sur certains Macs).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> torch.device:
    """ cuda > MPS > CPU (B5 : aucun changement sur Mac mps/cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parent_dtype(device: torch.device) -> torch.dtype:
    """Précision du parent en inférence : fp16 sur accélérateur, fp32 sur CPU."""
    return torch.float16 if device.type in ("mps", "cuda") else torch.float32


class DistillationLoss(nn.Module):
    """Loss de distillation (logits) + option MSE cachée."""

    def __init__(
        self, alpha: float = 0.7, temperature: float = 2.0, hidden_weight: float = 0.0
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha doit être dans [0, 1].")
        self.alpha = alpha
        self.temperature = temperature
        self.hidden_weight = hidden_weight

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, S, V)
        teacher_logits: torch.Tensor,  # (B, S, V) déjà restreints au vocab enfant
        targets: torch.Tensor,  # (B, S) ids réduits
    ) -> tuple[torch.Tensor, dict]:
        t = self.temperature
        # KL avec température, moyennée sur les TOKENS (B*S) : somme sur le
        # vocab puis moyenne. (F.kl_div batchmean diviserait seulement par B,
        # ce qui gonflerait la KL d'un facteur seq_len.)
        s_logp = F.log_softmax(student_logits / t, dim=-1)
        t_prob = F.softmax(teacher_logits / t, dim=-1)
        # Garde 0*log(0) : hors top-k (cache), t_prob vaut exactement 0 et
        # 0 * -inf = NaN. Analytiquement la contribution est 0 (limite).
        term = t_prob * (t_prob.log() - s_logp)
        kl = torch.where(t_prob > 0, term, torch.zeros_like(term)).sum(
            dim=-1).mean() * (t * t)
        ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            targets.reshape(-1),
        )
        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return loss, {"kl": kl.detach(), "ce": ce.detach()}


def hidden_mse(
    student_h: torch.Tensor, teacher_h: torch.Tensor, proj: nn.Linear
) -> torch.Tensor:
    """MSE entre caché parent et caché enfant projeté (même longueur S)."""
    s = proj(student_h.float())
    t = teacher_h.float().detach()
    n = min(s.shape[1], t.shape[1])
    return F.mse_loss(s[:, :n], t[:, :n])


# ---------------------------------------------------------------------------
# Cache logits parent top-k sur disque (format mmap + manifeste)
# ---------------------------------------------------------------------------
#
# L'ancien format .npz dense était à la fois lourd ET faux : en entraînement
# les batchs sont mélangés (randperm) mais le cache était relu en ordre
# séquentiel (compteur micro), donc les logits ne correspondaient plus aux
# entrées. Le nouveau format indexe chaque micro-chunk par (batch, chunk).
# Fichiers : {prefix}.manifest.json + {prefix}.vals.fp16 (lignes float16) +
# {prefix}.idx.u16 (uint16) + {prefix}.tgt.i64 (ids réduits).
# Chaque chunk = (B*S, K) lignes aplaties, ordre explicite dans le manifeste.

import hashlib
import json


def _sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def build_teacher_cache(
    parent,
    parent_batches: list,
    student_batches: list,
    kept_ids: list[int],
    top_k: int,
    device,
    micro_batch: int,
    prefix: str,
    parent_name: str = "",
) -> dict:
    """Pré-calcule le top-k parent par micro-chunk (même découpage que train).

    parent_batches / student_batches : mêmes listes que train() (alignées).
    Retourne le descripteur (manifest + memmaps). Le parent n'est plus
    nécessaire ensuite.
    """
    import numpy as np

    kept = torch.tensor(kept_ids, device=device)
    V = len(kept_ids)
    if V > 65535:
        raise ValueError("Cache uint16 : vocab réduit > 65535 non supporté.")
    if not parent_batches:
        raise ValueError("Cache vide : aucun batch parent.")
    S = parent_batches[0].shape[1] - 1
    K = min(top_k, V)
    order, rows_v, rows_i, rows_t = [], [], [], []
    parent.eval()
    with torch.no_grad():
        for bi, (pb, sb) in enumerate(zip(parent_batches, student_batches)):
            for ci, off in enumerate(range(0, pb.shape[0], micro_batch)):
                px = pb[off : off + micro_batch][:, :-1].to(device)
                y = sb[off : off + micro_batch][:, 1:]
                logits = parent(input_ids=px).logits.float()
                small = logits.index_select(-1, kept)
                v, i = small.topk(K, dim=-1)
                B = v.shape[0]
                order.append([bi, ci, B])
                rows_v.append(v.cpu().half().numpy())
                rows_i.append(i.cpu().numpy().astype("<u2"))
                rows_t.append(y.numpy().astype("<i8"))
    tot_rows = sum(r.shape[0] * r.shape[1] for r in rows_v)
    vals = np.memmap(prefix + ".vals.fp16", dtype="<f2", mode="w+",
                     shape=(tot_rows, K))
    idx = np.memmap(prefix + ".idx.u16", dtype="<u2", mode="w+",
                    shape=(tot_rows, K))
    tgt = np.memmap(prefix + ".tgt.i64", dtype="<i8", mode="w+",
                    shape=(tot_rows,))
    pos = 0
    for v, i, t in zip(rows_v, rows_i, rows_t):
        n = v.shape[0] * v.shape[1]
        vals[pos : pos + n] = v.reshape(-1, K)
        idx[pos : pos + n] = i.reshape(-1, K)
        tgt[pos : pos + n] = t.reshape(-1)
        pos += n
    vals.flush(); idx.flush(); tgt.flush()
    first = parent_batches[0].numpy().tobytes()
    last = parent_batches[-1].numpy().tobytes()
    manifest = {
        "version": 1,
        "parent": parent_name or getattr(parent, "name_or_path", None) or "?",
        "vocab_hash": _sha(list(kept_ids)),
        "seq_len": S,
        "k": K,
        "micro_batch": micro_batch,
        "n_chunks": len(order),
        "order": order,
        "slice_hash": _sha([len(parent_batches), first[:64].hex(), last[-64:].hex()]),
    }
    with open(prefix + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[cache] écrit : {prefix}.* ({len(order)} chunks, top-{K})")
    return load_teacher_cache(prefix, manifest["parent"], kept_ids, S, K)


def load_teacher_cache(prefix: str, parent_name: str, kept_ids: list[int],
                       seq_len: int, top_k: int) -> dict:
    """Recharge le cache mmap. Refuse tout cache incompatible (ValueError)."""
    import numpy as np

    try:
        with open(prefix + ".manifest.json", encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Cache introuvable : {prefix}.manifest.json")
    if m.get("version") != 1:
        raise ValueError("Version de cache non supportée.")
    if m.get("parent") != parent_name:
        raise ValueError(
            f"Cache d'un autre parent ({m.get('parent')} != {parent_name}).")
    if m.get("vocab_hash") != _sha(list(kept_ids)):
        raise ValueError("Cache incompatible : vocabulaire réduit différent "
                         "(--vocab-from / tranche différente ?).")
    if m.get("seq_len") != seq_len:
        raise ValueError("Cache incompatible : seq_len différente.")
    if m.get("k") != min(top_k, len(kept_ids)):
        raise ValueError("Cache incompatible : top-k différent.")
    tot = sum(B * m["seq_len"] for _, _, B in m["order"])
    K = m["k"]
    vals = np.memmap(prefix + ".vals.fp16", dtype="<f2", mode="r",
                     shape=(tot, K))
    idx = np.memmap(prefix + ".idx.u16", dtype="<u2", mode="r",
                    shape=(tot, K))
    tgt = np.memmap(prefix + ".tgt.i64", dtype="<i8", mode="r", shape=(tot,))
    print(f"[cache] chargé : {prefix}.* ({m['n_chunks']} chunks, top-{K})")
    return {"manifest": m, "vals": vals, "idx": idx, "tgt": tgt}


def teacher_logits_from_cache(cache: dict, bi: int, ci: int, vocab: int,
                              device) -> torch.Tensor:
    """Reconstruit les logits denses (B, S, V) d'un chunk : -inf hors top-k.

    Le softmax renormalise donc sur le top-k (KL tronquée renormalisée).
    """
    import numpy as np

    m = cache["manifest"]
    S = m["seq_len"]
    pos = 0
    for obi, oci, B in m["order"]:
        if obi == bi and oci == ci:
            break
        pos += B * S
    else:
        raise KeyError(f"Chunk ({bi}, {ci}) absent du cache.")
    v = torch.from_numpy(
        np.ascontiguousarray(cache["vals"][pos : pos + B * S]).copy()).float()
    i = torch.from_numpy(
        np.ascontiguousarray(cache["idx"][pos : pos + B * S]).copy()).long()
    dense = torch.full((B * S, vocab), float("-inf"))
    dense.scatter_(-1, i, v)
    return dense.view(B, S, vocab).to(device)


# ---------------------------------------------------------------------------
# Plan d'epoch : décompte exact des micros et steps (B1)
# ---------------------------------------------------------------------------


def plan_epoch(student_batches: list, batch_size: int, accum: int) -> dict:
    """Plan exact d'une epoch : micros, steps optimiseur, tokens.

    - micros = somme sur les batchs stockés de ceil(lignes / batch_size)
      (le dernier batch stocké peut être partiel : il compte pour
      ceil(lignes_restantes / batch_size) micros).
    - steps = ceil(micros / accum) : le reste de fin d'epoch est APPLIQUÉ
      (pas d'optimiseur jeté), contrairement à l'ancien calcul qui perdait
      les micro-batchs restants (ex. 6 chez tranche00).
    - tokens = fenêtres totales x seq_len.
    - Refuse (ValueError) si batch_size dépasse le plus gros batch stocké :
      le micro-batch réel serait alors save_batch (< demandé). Dans ce cas,
      utilisez --batch-size <= lignes stockées (ou reconstruisez les batchs
      avec --save-batch >= --batch-size).
    """
    if batch_size < 1 or accum < 1:
        raise ValueError("batch_size et accum doivent être >= 1.")
    if not student_batches:
        raise ValueError("Aucun batch d'entraînement.")
    rows = [b.shape[0] for b in student_batches]
    seq_len = student_batches[0].shape[1] - 1
    if batch_size > max(rows):
        raise ValueError(
            f"--batch-size ({batch_size}) > batch stocké ({max(rows)} lignes, "
            f"--save-batch ?) : le micro-batch réel serait {max(rows)}. "
            f"Utilisez --batch-size <= {max(rows)} ou reconstruisez les "
            f"batchs avec --save-batch >= {batch_size}."
        )
    micros = sum((r + batch_size - 1) // batch_size for r in rows)
    steps = max((micros + accum - 1) // accum, 1)
    return {
        "micros": micros,
        "steps": steps,
        "tokens": sum(rows) * seq_len,
        "seq_len": seq_len,
        "eff_micro": min(batch_size, max(rows)),
    }


# ---------------------------------------------------------------------------
# Reprise robuste : snapshots RNG (B4)
# ---------------------------------------------------------------------------


def _rng_snapshot(device) -> dict:
    """Capture les états RNG (reprise bit-reproductible)."""
    import numpy as _np

    snap = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": _np.random.get_state(),
    }
    if device.type == "mps":
        try:
            snap["mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            pass
    return snap


def _rng_restore(snap: dict | None, device) -> None:
    """Restaure les états RNG (best-effort : clés manquantes ignorées)."""
    import numpy as _np

    if not snap:
        return
    if snap.get("torch") is not None:
        torch.set_rng_state(snap["torch"].cpu())
    if snap.get("python") is not None:
        random.setstate(snap["python"])
    if snap.get("numpy") is not None:
        _np.random.set_state(snap["numpy"])
    if snap.get("mps") is not None and device.type == "mps":
        try:
            torch.mps.set_rng_state(snap["mps"])
        except (AttributeError, RuntimeError):
            pass


def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Permutation des batchs déterministe en (seed, epoch).

    Remplace le randperm non seedé : une reprise repart exactement au même
    endroit, et deux runs avec le même seed voient les données dans le même
    ordre (bruit inter-runs mesurable en variant le seed).
    """
    g = torch.Generator().manual_seed(
        (seed * 1000003 + epoch * 7919 + 1) & 0xFFFFFFFFFFFFFFFF)
    return torch.randperm(n, generator=g).tolist()


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Scheduler LambdaLR : warmup linéaire puis cosine decay vers 0.1x."""

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


@torch.no_grad()
def eval_loss(student, batches, device, kept_ids=None) -> float:
    """Perte CE moyenne (diagnostic rapide, sans parent)."""
    student.eval()
    tot, n = 0.0, 0
    for b in batches:
        x, y = b[:, :-1].to(device), b[:, 1:].to(device)
        logits = student(x)
        tot += (
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
            * x.numel()
        )
        n += x.numel()
    student.train()
    return tot / max(n, 1)


def train(
    student: TinyTransformer,
    parent_batches: list[torch.Tensor],  # ids parent (B, S+1)
    student_batches: list[torch.Tensor],  # ids réduits alignés (B, S+1)
    vocab: dict,
    parent_name: str,
    out: str,
    alpha: float = 0.7,
    temperature: float = 2.0,
    hidden_weight: float = 0.0,
    lr: float = 3e-4,
    epochs: int = 3,
    batch_size: int = 2,  # micro-batch (accumulation)
    accum: int = 8,  # steps d'accumulation
    warmup_ratio: float = 0.05,
    max_grad_norm: float = 1.0,
    top_k_cache: int = 0,  # >0 : utilise/crée le cache top-k
    cache_path: str = "",
    cache_k: int = 32,  # top-k stocké (uint16 + float16, mmap)
    resume: str = "",
    log_every: int = 20,
    seed: int = 0,  # B4 : ordre des batchs déterministe par (seed, epoch)
    ckpt_every: int = 500,  # B4 : >0 : checkpoint périodique tous les N steps
    device: torch.device | None = None,
    val_parent_batches: list | None = None,  # batchs val parent (même format)
    val_student_batches: list | None = None,  # batchs val enfant alignés
    eval_every: int = 0,  # >0 : évalue la val tous les N steps (0 = désactivé)
    patience: int = 0,  # >0 : early stopping après N evals sans progrès
    lr_resume: float = 0.0,  # >0 : remplace le LR à la reprise
    rewarmup_ratio: float = 0.0,  # >0 : reconstruit le scheduler sur les steps restants
) -> str:
    """Entraîne l'enfant par distillation. Retourne le chemin du checkpoint."""
    device = device or pick_device()
    print(f"[train] device={device} parent={parent_name}")
    kept = vocab["kept_ids"]
    student.to(device).train()

    # --- Plan d'epoch AVANT tout chargement parent (échec rapide, sans
    # télécharger 135M de poids si --batch-size est incohérent). ---
    plan = plan_epoch(student_batches, batch_size, accum)

    # --- Parent (inférence seule) ou cache mmap (parent non chargé). ---
    cache = None
    parent = None
    # La projection cachée est créée ici (None par défaut) AVANT toute
    # construction du cache : le test `if proj is None` ci-dessous la lit
    # (B3 : UnboundLocalError si créée après).
    proj = None
    if top_k_cache > 0 and cache_path:
        try:
            cache = load_teacher_cache(
                cache_path, parent_name, kept,
                parent_batches[0].shape[1] - 1, cache_k)
            if cache["manifest"]["micro_batch"] != batch_size:
                raise ValueError(
                    "Cache incompatible : micro_batch "
                    f"({cache['manifest']['micro_batch']}) != --batch-size "
                    f"({batch_size}). Reconstruisez le cache avec le même "
                    "--batch-size que l'entraînement.")
            print(f"[train] cache parent réutilisé (parent non chargé).")
        except ValueError as e:
            if os.path.exists(cache_path + ".manifest.json"):
                raise  # cache existant mais incompatible : on refuse
            print(f"[train] pas de cache : {e}")
    if cache is None:
        from transformers import AutoModelForCausalLM

        dtype = parent_dtype(device)
        parent = AutoModelForCausalLM.from_pretrained(parent_name, dtype=dtype)
        parent.to(device).eval()
        for p in parent.parameters():
            p.requires_grad_(False)
        if top_k_cache > 0 and cache_path:
            cache = build_teacher_cache(
                parent, parent_batches, student_batches, kept, cache_k,
                device, batch_size, cache_path, parent_name=parent_name)
            if proj is None:
                # Sans hidden : le parent ne sert plus, on libère la mémoire.
                print(f"[train] parent déchargé, suite sur cache.")
                del parent
                parent = None
                if device.type == "mps":
                    torch.mps.empty_cache()

    kept_t = torch.tensor(kept, device=device)
    # La projection cachée doit exister AVANT l'optimiseur et le scheduler
    # (le scheduler fige le nombre de groupes de paramètres).
    if hidden_weight > 0:
        if parent is None:
            raise ValueError(
                "hidden_weight exige le parent en mémoire (pas de cache seul)."
            )
        proj = nn.Linear(student.cfg.d_model, parent.config.hidden_size).to(device)
    params = list(student.parameters()) + (
        list(proj.parameters()) if proj is not None else []
    )
    opt = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = plan["steps"]
    total_steps = steps_per_epoch * epochs
    sched = cosine_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)
    criterion = DistillationLoss(alpha, temperature, hidden_weight)

    start_step, global_step = 0, 0
    start_epoch, resume_micro, resume_order = 0, 0, None
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        # Enchaînement de tranches : le vocab ne doit pas avoir bougé.
        if ckpt.get("vocab") is not None:
            from data import check_vocab_compatible

            check_vocab_compatible(vocab, ckpt["vocab"])
        student.load_state_dict(ckpt["model"])
        if proj is not None and ckpt.get("proj") is not None:
            proj.load_state_dict(ckpt["proj"])
        if ckpt.get("opt") is not None:
            # Reprise classique : on garde l'optimiseur.
            opt.load_state_dict(ckpt["opt"])
        else:
            # Best-val (poids seuls) : nouvel optimiseur. --lr-resume
            # devient alors le LR de départ (défaut : lr).
            print("[train] pas d'état optimiseur dans le checkpoint "
                  "(best-val) : nouvel optimiseur.")
        if lr_resume and lr_resume > 0:
            for g in opt.param_groups:
                g["lr"] = lr_resume
            print(f"[train] LR reprise forcé : {lr_resume}")
        if rewarmup_ratio and rewarmup_ratio > 0:
            # Scheduler reconstruit sur les steps restants (warmup frais),
            # adapté à l'enchaînement sur nouvelles données.
            remaining = max(total_steps - ckpt["step"], 1)
            sched = cosine_with_warmup(opt, int(remaining * rewarmup_ratio), remaining)
            print(f"[train] scheduler reconstruit : {remaining} steps restants")
        elif ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        else:
            # Best-val sans scheduler : reconstruction standard.
            sched = cosine_with_warmup(opt, int(total_steps * warmup_ratio),
                                       total_steps)
            print("[train] scheduler reconstruit (défaut, pas d'état).")
        start_step = ckpt["step"]
        global_step = start_step
        print(f"[train] reprise depuis {resume} (step {start_step})")
        # B4 : reprise positionnelle (epoch + micro + ordre + RNG). Les
        # anciens checkpoints (sans ces clés) repartent à l'epoch 0 comme avant.
        start_epoch = ckpt.get("epoch", 0)
        resume_micro = ckpt.get("micro", 0)
        resume_order = ckpt.get("order")
        if ckpt.get("rng") is not None:
            _rng_restore(ckpt["rng"], device)
            print(f"[train] états RNG restaurés (epoch {start_epoch}, "
                  f"micro {resume_micro}).")

    n_params = sum(p.numel() for p in student.parameters())
    print(
        f"[train] enfant: {n_params / 1e6:.1f}M params latents, "
        f"{len(student_batches)} batchs, {plan['micros']} micros/epoch, "
        f"{steps_per_epoch} steps/epoch x {epochs} epochs = {total_steps} steps, "
        f"{plan['tokens'] / 1e6:.1f}M tokens/epoch, "
        f"micro-batch effectif {plan['eff_micro']}"
    )
    use_val = (
        eval_every > 0
        and val_parent_batches
        and val_student_batches
        and len(val_student_batches) > 0
    )
    if eval_every > 0 and not use_val:
        print("[train] attention : --eval-every sans batchs val -> pas d'éval.")
    best_val, best_step, no_improve = float("inf"), start_step, 0
    best_path = out.replace(".pt", "_bestval.pt") if out.endswith(".pt") else out + "_bestval.pt"
    stopped_early = False

    t0 = time.time()
    tok_total = 0

    def _prune_periodic() -> None:
        # Ne garde que les 2 derniers checkpoints périodiques (B4).
        base = out[:-3] if out.endswith(".pt") else out
        cands = sorted(glob.glob(base + "_ckpt*.pt"))
        for old in cands[:-2]:
            try:
                os.remove(old)
            except OSError:
                pass

    def _save_periodic(epoch: int, micro: int, order: list) -> str:
        base = out[:-3] if out.endswith(".pt") else out
        path = f"{base}_ckpt{global_step:07d}.pt"
        torch.save(
            {
                "model": student.state_dict(),
                "proj": proj.state_dict() if proj is not None else None,
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": global_step,
                "epoch": epoch,
                "micro": micro,
                "order": list(order),
                "rng": _rng_snapshot(device),
                "cfg": student.cfg,
                "vocab": vocab,
                "parent": parent_name,
                "hidden_weight": hidden_weight,
            },
            path,
        )
        _prune_periodic()
        print(f"[train] checkpoint périodique : {path} "
              f"(epoch {epoch}, micro {micro}).")
        return path

    def _opt_step(loss_val: float, last_parts: dict, epoch: int, micro: int,
                  order: list) -> bool:
        """Applique un pas optimiseur (+ scheduler). Retourne True = arrêter.

        Factorise le pas "tous les `accum` micros" ET le pas de fin d'epoch
        sur les micros restants (B1 : plus aucun gradient jeté).
        """
        nonlocal global_step, best_val, best_step, no_improve, stopped_early
        nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        global_step += 1
        if global_step % log_every == 0 and global_step > start_step:
            dt = time.time() - t0
            ppl = math.exp(min(last_parts["ce"].item(), 20))
            print(
                f"[train] step {global_step}/{total_steps} "
                f"loss={loss_val:.4f} kl={last_parts['kl'].item():.4f} "
                f"ce={last_parts['ce'].item():.4f} ppl~{ppl:.1f} "
                f"{tok_total / dt:.0f} tok/s",
                flush=True,
            )
        if use_val and global_step % eval_every == 0:
            # Éval val (CE enfant seule, sans parent) + best + patience.
            val_ce = eval_loss(student, val_student_batches, device)
            val_ppl = math.exp(min(val_ce, 20))
            tag = ""
            if val_ce < best_val:
                best_val, best_step, no_improve = val_ce, global_step, 0
                torch.save(
                    {
                        "model": student.state_dict(),
                        "cfg": student.cfg,
                        "vocab": vocab,
                        "parent": parent_name,
                        "val_ce": val_ce,
                        "step": global_step,
                    },
                    best_path,
                )
                tag = " [best]"
            else:
                no_improve += 1
            print(
                f"[val] step {global_step}/{total_steps} "
                f"val_ce={val_ce:.4f} val_ppl~{val_ppl:.1f}{tag}",
                flush=True,
            )
            if patience > 0 and no_improve >= patience:
                print(
                    f"[val] early stopping : {patience} evals sans "
                    f"progrès (best step {best_step})."
                )
                stopped_early = True
        if (ckpt_every > 0 and micro % accum == 0
                and global_step % ckpt_every == 0
                and global_step < total_steps):
            # Checkpoint périodique UNIQUEMENT sur frontière d'accumulation
            # (micro % accum == 0) : les gradients sont à zéro, la reprise
            # retrouve exactement la même trajectoire (B4).
            _save_periodic(epoch, micro, order)
        return stopped_early or global_step >= total_steps

    for epoch in range(start_epoch, epochs):
        if epoch == start_epoch and resume_order is not None:
            order = list(resume_order)
            skip = resume_micro
            print(f"[train] reprise mid-epoch {epoch} : {skip} micros sautés.")
        else:
            order = _epoch_order(len(student_batches), seed, epoch)
            skip = 0
        micro = 0
        skip0 = skip  # micros déjà consommés avant reprise (B4)
        opt.zero_grad(set_to_none=True)
        for bi in order:
            sb = student_batches[bi]
            pb = parent_batches[bi]
            # Micro-batch : découpe le batch stocké si besoin.
            for off in range(0, sb.shape[0], batch_size):
                if skip > 0:
                    # Micros déjà vus avant l'interruption : on les saute
                    # (leurs gradients ont été appliqués avant le checkpoint).
                    skip -= 1
                    continue
                x = sb[off : off + batch_size][:, :-1].to(device)
                y = sb[off : off + batch_size][:, 1:].to(device)
                px = pb[off : off + batch_size][:, :-1].to(device)
                ci = off // batch_size
                if cache is not None and proj is None:
                    # Logits parent depuis le cache (parent non chargé).
                    t_logits = teacher_logits_from_cache(
                        cache, bi, ci, len(kept), device)
                    t_h = None
                else:
                    with torch.no_grad():
                        t_full = parent(input_ids=px).logits.float()
                        t_logits = t_full.index_select(-1, kept_t)
                    t_h = None
                if proj is not None:
                    # Distillation des états cachés : dernier caché enfant
                    # (projeté) vs dernier caché parent, MSE.
                    s_logits, s_h = student.forward_with_hidden(x)
                    loss, parts = criterion(s_logits, t_logits, y)
                    with torch.no_grad():
                        t_h = parent(
                            input_ids=px, output_hidden_states=True
                        ).hidden_states[-1]
                    loss = loss + criterion.hidden_weight * hidden_mse(s_h, t_h, proj)
                else:
                    s_logits = student(x)
                    loss, parts = criterion(s_logits, t_logits, y)
                if not torch.isfinite(loss):
                    # Garde-fou (ex. OOM GPU silencieuse -> NaN) : arrêt
                    # propre avec checkpoint de sauvegarde, pas de backward.
                    bad_path = (out.replace(".pt", "_nonfinite.pt")
                                if out.endswith(".pt") else out + "_nonfinite.pt")
                    torch.save(
                        {
                            "model": student.state_dict(),
                            "cfg": student.cfg,
                            "vocab": vocab,
                            "parent": parent_name,
                            "step": global_step,
                            "reason": "loss non finie",
                        },
                        bad_path,
                    )
                    print(
                        f"[train] ARRÊT : loss non finie à step {global_step} "
                        f"(OOM ?). Poids sauvegardés dans {bad_path}."
                    )
                    stopped_early = True
                    break
                (loss / accum).backward()
                tok_total += x.numel()
                micro += 1
                if micro % accum == 0:
                    if _opt_step(loss.item(), parts, epoch, skip0 + micro, order):
                        break
                if stopped_early or global_step >= total_steps:
                    break
            if stopped_early or global_step >= total_steps:
                break
        if micro % accum != 0 and not stopped_early and global_step < total_steps:
            # Reste de fin d'epoch : on applique l'optimiseur au lieu de
            # jeter les gradients accumulés (B1). L'arrêt éventuel
            # (patience / total_steps) est géré par le test sous le checkpoint.
            _opt_step(loss.item(), parts, epoch, skip0 + micro, order)
        ckpt_path = out.replace(".pt", f"_e{epoch}.pt")
        # B4 : position exacte. Epoch complète -> reprise à epoch+1 ; arrêt
        # précoce mid-epoch -> reprise au même (epoch, micro, ordre).
        epoch_done = (skip0 + micro) >= plan["micros"]
        torch.save(
            {
                "model": student.state_dict(),
                "proj": proj.state_dict() if proj is not None else None,
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": global_step,
                "epoch": epoch + 1 if epoch_done else epoch,
                "micro": 0 if epoch_done else skip0 + micro,
                "order": None if epoch_done else list(order),
                "rng": _rng_snapshot(device),
                "cfg": student.cfg,
                "vocab": vocab,
                "parent": parent_name,
                "hidden_weight": hidden_weight,
            },
            ckpt_path,
        )
        print(f"[train] checkpoint : {ckpt_path}")
        if stopped_early or global_step >= total_steps:
            break
    if stopped_early and os.path.exists(best_path):
        # Restaure les meilleurs poids avant la sauvegarde finale.
        best = torch.load(best_path, map_location=device, weights_only=False)
        student.load_state_dict(best["model"])
        print(f"[val] poids restaurés depuis {best_path} (step {best['step']}).")
    if use_val:
        # Tableau final : PPL train (échantillon) vs PPL val (complète).
        train_ce = eval_loss(student, student_batches[:50], device)
        val_ce = eval_loss(student, val_student_batches, device)
        train_ppl, val_ppl = math.exp(min(train_ce, 20)), math.exp(min(val_ce, 20))
        print(
            f"[eval] train-PPL ~{train_ppl:.1f} (50 batchs) vs "
            f"val-PPL ~{val_ppl:.1f} ({len(val_student_batches)} batchs)"
        )
        if train_ppl > 0 and val_ppl / train_ppl > 1.5:
            print(
                "[eval] AVERTISSEMENT : écart train/val > 1.5x, "
                "surapprentissage probable."
            )
    final = out if out.endswith(".pt") else out + ".pt"
    torch.save(
        {
            "model": student.state_dict(),
            "proj": proj.state_dict() if proj is not None else None,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": global_step,
            "cfg": student.cfg,
            "vocab": vocab,
            "parent": parent_name,
            "hidden_weight": hidden_weight,
        },
        final,
    )
    print(f"[train] terminé en {time.time() - t0:.0f}s -> {final}")
    return final


# ---------------------------------------------------------------------------
# Pont Ollama (distill-loop : génération + évaluation sans logits)
# ---------------------------------------------------------------------------


def ollama_available() -> bool:
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


def ollama_generate(model: str, prompt: str, **opts) -> str:
    """Génère un texte via Ollama (données synthétiques / réponses parent)."""
    import ollama

    res = ollama.generate(
        model=model, prompt=prompt, options=opts or {"temperature": 0.7}
    )
    return res.get("response", "")


def ollama_judge(model: str, instruction: str, answer: str) -> str:
    """Demande au modèle Ollama de corriger/améliorer une réponse."""
    import ollama

    prompt = (
        f"Instruction : {instruction}\nRéponse proposée : {answer}\n"
        "Donne une version améliorée et concise de la réponse."
    )
    res = ollama.generate(model=model, prompt=prompt)
    return res.get("response", "")
