"""Quantification 2 bits ternaire façon BitNet b1.58 + packing.

Contenu :
- quantification ternaire {-1, 0, +1} avec scale absmean par groupe ;
- quantification des activations en int8 (absmax par token / par ligne) ;
- Straight-Through Estimator (STE) pour l'entraînement QAT ;
- packing 4 poids 2 bits par octet (uint8) + unpack, aller-retour exact ;
- version de référence PyTorch du matmul int8 x ternaire (sans vraie
  multiplication, additions/soustractions) pour valider le noyau C.

Conventions de codage 2 bits (choix documenté, utilisé partout) :
    0  -> 0b00
    +1 -> 0b01
    -1 -> 0b11  (0b10 réservé, décodé comme 0 pour robustesse)
Le premier poids du groupe occupe les bits de poids faible de l'octet.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Ternaire + STE
# ---------------------------------------------------------------------------


class _TernarySTE(Function):
    """Forward : quantifie en {-1, 0, +1}. Backward : identité (STE)."""

    @staticmethod
    def forward(
        ctx, w: torch.Tensor, scale: torch.Tensor, group_size: int
    ) -> torch.Tensor:
        # w : (..., cols), scale : (..., num_groups)
        num_groups = scale.shape[-1]
        # Répète chaque scale sur son groupe pour normaliser.
        scale_expanded = scale.repeat_interleave(group_size, dim=-1)[..., : w.shape[-1]]
        # Évite la division par zéro (ligne/groupe mort).
        scale_safe = torch.clamp(scale_expanded, min=1e-8)
        q = torch.round(w / scale_safe).clamp_(-1, 1)
        return q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE : le gradient traverse la quantification tel quel.
        return grad_output, None, None


def absmean_scales(w: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Calcule les scales absmean par groupe sur la dernière dimension.

    scale_g = mean(|w_g|) avec w_g le groupe de `group_size` colonnes.
    Retour : (..., num_groups), fp32.
    """
    if w.shape[-1] % group_size != 0:
        raise ValueError(
            f"group_size={group_size} doit diviser la dim {w.shape[-1]} "
            "(paddez la matrice si besoin)."
        )
    groups = w.view(*w.shape[:-1], -1, group_size)
    return groups.abs().mean(dim=-1).to(torch.float32)


def quantize_ternary_ste(
    w: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantifie `w` (poids latents fp32) en ternaire avec STE.

    Retour : (q, scales) avec q dans {-1, 0, +1} (float32 pour le matmul
    de référence, valeurs entières) et scales fp32 (..., num_groups).
    Au backward, le gradient de q est routé vers w (STE).
    """
    scales = absmean_scales(w, group_size)
    q = _TernarySTE.apply(w, scales, group_size)
    return q, scales


def quantize_ternary_infer(
    w: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Version inférence (sans graphe) : q int8 + scales fp32."""
    with torch.no_grad():
        scales = absmean_scales(w, group_size)
        num_groups = scales.shape[-1]
        scale_expanded = scales.repeat_interleave(group_size, dim=-1)[
            ..., : w.shape[-1]
        ]
        q = torch.round(w / scale_expanded.clamp(min=1e-8)).clamp_(-1, 1).to(torch.int8)
    return q, scales


# ---------------------------------------------------------------------------
# Activations int8 (absmax par ligne / par token)
# ---------------------------------------------------------------------------


def quantize_activation_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantifie `x` (..., cols) en int8, scale absmax par ligne.

    scale = max(|x_ligne|) / 127. Retour : (q int8, scales fp32 [..., 1]).
    Une ligne nulle donne scale 1e-8 et q nul (pas de NaN).
    """
    with torch.no_grad():
        amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = (amax / 127.0).clamp(min=1e-8)
        q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


# ---------------------------------------------------------------------------
# Packing 2 bits : 4 poids par octet
# ---------------------------------------------------------------------------

# Codage 2 bits (cf. docstring pack/unpack) : 0b00 = 0, 0b01 = +1,
# 0b11 = -1, 0b10 réservé (décodé comme 0).


def pack_ternary(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Packe un tenseur ternaire (valeurs -1/0/+1) en uint8, 4 poids/octet.

    Entrée : tenseur int (n'importe quelle shape, aplati en C-order).
    Retour : (packed uint8 [..., ceil(n/4)], n) où n sert à l'unpack exact
    quand la taille n'est pas multiple de 4 (bourrage à 0).
    Vectorisé : instantané même sur 20M de poids.
    """
    flat = q.to(torch.int64).reshape(-1)
    n = flat.numel()
    if not torch.all((flat == -1) | (flat == 0) | (flat == 1)):
        raise ValueError("pack_ternary n'accepte que des valeurs -1, 0, +1.")
    # Mappe vers les codes 2 bits via table : {-1: 0b11, 0: 0b00, +1: 0b01}.
    lut = torch.tensor([0b11, 0b00, 0b01], dtype=torch.uint8, device=q.device)
    codes = lut[(flat + 1).to(torch.long)]
    # Bourre à un multiple de 4 avec des zéros (0b00).
    pad = (-n) % 4
    if pad:
        codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8, device=q.device)])
    codes = codes.view(-1, 4).to(torch.int64)
    packed = (
        codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    ).to(torch.uint8)
    return packed, n


def unpack_ternary(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse exact de pack_ternary : uint8 -> int8 1D de longueur n."""
    flat_p = packed.to(torch.int64).reshape(-1)
    codes = torch.stack([(flat_p >> s) & 0x3 for s in (0, 2, 4, 6)], dim=1).reshape(-1)[
        :n
    ]
    # Décodage : 0b00 -> 0, 0b01 -> +1, 0b11 -> -1, 0b10 -> 0 (réservé).
    out = torch.zeros(n, dtype=torch.int8)
    out[codes == 0b01] = 1
    out[codes == 0b11] = -1
    return out


# ---------------------------------------------------------------------------
# Matmul de référence PyTorch : int8 x ternaire, sans multiplication
# ---------------------------------------------------------------------------


def ternary_matmul_ref(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scales: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Référence pour valider le noyau C.

    x_q : (..., K) int8, x_scale : (..., 1) fp32,
    w_q : (K, N) ternaire {-1,0,1} (float ou int),
    w_scales : (N, num_groups) fp32.
    Calcule en n'utilisant que des additions/soustractions sur les entiers :
    pour chaque poids +1 on ajoute l'activation, -1 on la retranche, 0 on ignore.
    Retour : (..., N) fp32.
    """
    if w_q.shape[0] != x_q.shape[-1]:
        raise ValueError("Dimensions incompatibles : K différent.")
    k, n = w_q.shape
    if k % group_size != 0:
        raise ValueError("K doit être multiple de group_size.")
    w = w_q.to(torch.float32)
    x = x_q.to(torch.float32)
    # Masques : pos = (w == +1), neg = (w == -1). La somme signée
    # x @ (pos - neg) n'utilise aucune multiplication poids*activation :
    # +1 -> addition, -1 -> soustraction, 0 -> ignoré.
    sign = (w == 1).to(torch.float32) - (w == -1).to(torch.float32)  # (K, N)
    num_groups = k // group_size
    x_g = x.view(*x.shape[:-1], num_groups, group_size)  # (..., G, gs)
    sign_g = sign.view(num_groups, group_size, n)  # (G, gs, N)
    # Contribution entière par groupe, pondérée par la scale du groupe,
    # puis par la scale d'activation.
    contrib = torch.einsum("...gk,gkn->...gn", x_g, sign_g)  # (..., G, N)
    out = (contrib * w_scales.to(torch.float32).t()).sum(dim=-2)
    return (out * x_scale).to(torch.float32)


def quantized_linear_forward(
    x: torch.Tensor, w_latent: torch.Tensor, group_size: int = 64
) -> torch.Tensor:
    """Couche linéaire QAT complète (STE sur poids ET activations).

    x : (..., K) fp32, w_latent : (N, K) fp32 (poids latents entraînables).
    Forward exact avec valeurs quantifiées ; backward STE vers w_latent et x :
        w_eff = w + (quant(w) - w).detach()
        x_eff = x + (dequant(quant(x)) - x).detach()
    Retour : (..., N) fp32.
    """
    # --- Poids : ternaire + scales absmean par groupe sur K, STE. ---
    # NOTE : le groupement découpe la dim d'entrée K ; on quantifie donc
    # w_latent (N, K) directement, pas sa transposée.
    q_nk, _ = quantize_ternary_ste(w_latent, group_size)  # STE, -1/0/+1
    q_t = q_nk.t()  # (K, N)
    scales_n_g = _scales_for_ref(w_latent, group_size)  # (N, G)
    w_t = w_latent.t()  # (K, N)
    scale_expanded = scales_n_g.t().repeat_interleave(group_size, dim=0)
    scale_expanded = scale_expanded[: w_t.shape[0]]  # (K, N)
    w_quant = q_t * scale_expanded
    w_eff = w_t + (w_quant - w_t).detach()  # STE
    # --- Activations : int8 absmax par ligne + STE vers x. ---
    x_q, x_scale = quantize_activation_int8(x)
    x_dequant = x_q.to(torch.float32) * x_scale
    x_eff = x + (x_dequant - x).detach()  # STE
    return x_eff @ w_eff


def _scales_for_ref(w_latent: torch.Tensor, group_size: int) -> torch.Tensor:
    """Scales (K_groupes_rec... ) au format attendu par ternary_matmul_ref.

    ternary_matmul_ref attend w_scales de shape (N, num_groups) où le groupe
    découpe la dimension K. Comme w_latent est (N, K), c'est direct.
    """
    n, k = w_latent.shape
    return w_latent.view(n, k // group_size, group_size).abs().mean(dim=-1)
