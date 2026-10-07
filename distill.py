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
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CONFIGS, TinyTransformer

# Les opérateurs MPS manquants retombent sur CPU (requis sur certains Macs).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> torch.device:
    """MPS si dispo, sinon CPU. Pas de CUDA sur Mac."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        kl = (t_prob * (t_prob.log() - s_logp)).sum(dim=-1).mean() * (t * t)
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
# Cache logits parent top-k sur disque
# ---------------------------------------------------------------------------


def precompute_teacher_cache(
    parent,
    batches: list[torch.Tensor],
    kept_ids: list[int],
    top_k: int,
    device: torch.device,
    path: str,
) -> None:
    """Calcule les top-k logits parent (vocab réduit) et les sauve en .npz."""
    kept = torch.tensor(kept_ids, device=device)
    vals_all, idx_all, tgt_all = [], [], []
    parent.eval()
    with torch.no_grad():
        for b in batches:
            inp = b[:, :-1].to(device)  # ids parent
            logits = parent(input_ids=inp).logits.float()  # (B, S, P)
            small = logits.index_select(-1, kept)  # (B, S, V)
            v, i = small.topk(min(top_k, small.shape[-1]), dim=-1)
            vals_all.append(v.cpu())
            idx_all.append(i.cpu())
            tgt_all.append(b[:, 1:])  # cibles réduites
    np.savez_compressed(
        path,
        vals=torch.cat(vals_all).numpy(),
        idx=torch.cat(idx_all).numpy(),
        tgt=torch.cat(tgt_all).numpy(),
    )


def load_teacher_cache(path: str, top_k: int, vocab: int):
    """Recharge le cache -> tenseurs denses (B*batch, S, V) scatterés."""
    z = np.load(path)
    vals = torch.from_numpy(z["vals"])
    idx = torch.from_numpy(z["idx"])
    tgt = torch.from_numpy(z["tgt"]).long()
    # Reconstruction dense : -inf partout sauf top-k (approximation standard).
    dense = torch.full((*vals.shape[:2], vocab), float("-inf"))
    dense.scatter_(-1, idx, vals.float())
    return dense, tgt


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
    resume: str = "",
    log_every: int = 20,
    device: torch.device | None = None,
) -> str:
    """Entraîne l'enfant par distillation. Retourne le chemin du checkpoint."""
    device = device or pick_device()
    print(f"[train] device={device} parent={parent_name}")
    kept = vocab["kept_ids"]
    student.to(device).train()

    # --- Parent (inférence seule). ---
    teacher_logits_all = None
    parent = None
    if top_k_cache > 0 and cache_path and os.path.exists(cache_path):
        print(f"[train] cache parent réutilisé : {cache_path}")
        teacher_logits_all, _ = load_teacher_cache(cache_path, top_k_cache, len(kept))
    else:
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if device.type == "mps" else torch.float32
        parent = AutoModelForCausalLM.from_pretrained(parent_name, dtype=dtype)
        parent.to(device).eval()
        for p in parent.parameters():
            p.requires_grad_(False)
        if top_k_cache > 0 and cache_path:
            precompute_teacher_cache(
                parent, parent_batches, kept, top_k_cache, device, cache_path
            )
            print(f"[train] cache parent écrit : {cache_path}")
            teacher_logits_all, _ = load_teacher_cache(
                cache_path, top_k_cache, len(kept)
            )

    kept_t = torch.tensor(kept, device=device)
    # La projection cachée doit exister AVANT l'optimiseur et le scheduler
    # (le scheduler fige le nombre de groupes de paramètres).
    proj = None
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
    steps_per_epoch = max(len(student_batches) // batch_size, 1)
    total_steps = steps_per_epoch * epochs
    sched = cosine_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)
    criterion = DistillationLoss(alpha, temperature, hidden_weight)

    start_step, global_step = 0, 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        if proj is not None and ckpt.get("proj") is not None:
            proj.load_state_dict(ckpt["proj"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"]
        global_step = start_step
        print(f"[train] reprise depuis {resume} (step {start_step})")

    n_params = sum(p.numel() for p in student.parameters())
    print(
        f"[train] enfant: {n_params / 1e6:.1f}M params latents, "
        f"{len(student_batches)} batchs, {total_steps} steps"
    )

    t0 = time.time()
    tok_total = 0
    for epoch in range(epochs):
        order = torch.randperm(len(student_batches)).tolist()
        micro = 0
        opt.zero_grad(set_to_none=True)
        for bi in order:
            sb = student_batches[bi]
            pb = parent_batches[bi]
            # Micro-batch : découpe le batch stocké si besoin.
            for off in range(0, sb.shape[0], batch_size):
                x = sb[off : off + batch_size][:, :-1].to(device)
                y = sb[off : off + batch_size][:, 1:].to(device)
                px = pb[off : off + batch_size][:, :-1].to(device)
                if teacher_logits_all is not None:
                    t_logits = teacher_logits_all[micro].to(device)
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
                (loss / accum).backward()
                tok_total += x.numel()
                micro += 1
                if micro % accum == 0:
                    nn.utils.clip_grad_norm_(params, max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % log_every == 0 and global_step > start_step:
                        dt = time.time() - t0
                        ppl = math.exp(min(parts["ce"].item(), 20))
                        print(
                            f"[train] step {global_step}/{total_steps} "
                            f"loss={loss.item():.4f} kl={parts['kl'].item():.4f} "
                            f"ce={parts['ce'].item():.4f} ppl~{ppl:.1f} "
                            f"{tok_total / dt:.0f} tok/s",
                            flush=True,
                        )
                    if global_step >= total_steps:
                        break
            if global_step >= total_steps:
                break
        ckpt_path = out.replace(".pt", f"_e{epoch}.pt")
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
            ckpt_path,
        )
        print(f"[train] checkpoint : {ckpt_path}")
        if global_step >= total_steps:
            break
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
