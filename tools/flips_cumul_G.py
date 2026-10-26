"""Phase 0 G — flips ternaires CUMULÉS depuis t08.

Les logs de distill.py comptent les changements entre deux évaluations
(« flips » par intervalle). Ici on compare directement les poids ternaires
quantifiés de chaque checkpoint à ceux de t08 : c'est le churn cumulé, celui
qui décide si le WSD « éteint » réellement les retournements de signe.

Mesures par matrice :
- fraction de poids ternaires différents de t08 (q_ref != q) ;
- matrice de transition {-1,0,+1} -> {-1,0,+1} agrégée ;
- mouvement latent relatif ||w - w_ref|| / ||w_ref||.

Contrôles : bras A et C (mêmes données/budget), et intervalles B
(23250 -> 23500 -> final).

Sorties : docs/results/flips_cumul_G.json.
Usage : .venv/bin/python tools/flips_cumul_G.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from quant import quantize_ternary_infer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

REF = "runs/E-t08/model.pt"
MODELS = [
    ("B@23250", "runs/F-B/model_ckpt0023250.pt"),
    ("B@23500", "runs/F-B/model_ckpt0023500.pt"),
    ("B@final", "runs/F-B/model.pt"),
    ("B@e0", "runs/F-B/model_e0.pt"),
    ("B@bestval", "runs/F-B/model_bestval.pt"),
    ("A@final", "runs/F-A/model.pt"),
    ("C@final", "runs/F-C/model.pt"),
]
OUT = "docs/results/flips_cumul_G.json"


def is_ternary(k: str, v: torch.Tensor) -> bool:
    return (k.startswith("layers.") and k.endswith(".weight")
            and v.ndim == 2 and "norm" not in k)


def ternaries(path: str):
    c = torch.load(path, map_location="cpu", weights_only=False)
    sd = c["model"]
    q, lat = {}, {}
    for k, v in sd.items():
        if is_ternary(k, v):
            q[k] = quantize_ternary_infer(v.to(torch.float32))[0]
            lat[k] = v.to(torch.float32).clone()
    return q, lat, int(c.get("step", -1))


def compare(q_ref, lat_ref, q, lat) -> dict:
    per = {}
    tot = changed = 0
    trans = {f"{a}->{b}": 0 for a in (-1, 0, 1) for b in (-1, 0, 1)}
    lat_num = lat_den = 0.0
    for k in q:
        a, b = q_ref[k].to(torch.int8), q[k].to(torch.int8)
        diff = int((a != b).sum())
        n = a.numel()
        tot += n
        changed += diff
        for av in (-1, 0, 1):
            for bv in (-1, 0, 1):
                trans[f"{av}->{bv}"] += int(((a == av) & (b == bv)).sum())
        num = float((lat[k] - lat_ref[k]).pow(2).sum().sqrt())
        den = float(lat_ref[k].pow(2).sum().sqrt())
        lat_num += num
        lat_den += den
        per[k] = {"flips": diff / n if n else 0.0, "n": n,
                  "latent_rel": num / den if den else 0.0}
    return {
        "step": None,
        "flips_global": changed / tot if tot else 0.0,
        "n_poids_ternaires": tot,
        "transitions": trans,
        "latent_rel_global": lat_num / lat_den if lat_den else 0.0,
        "par_matrice": per,
    }


def main() -> int:
    q_ref, lat_ref, step_ref = ternaries(REF)
    out = {"reference": REF, "step_reference": step_ref, "modeles": {}}
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"[flips] absent : {path}")
            continue
        q, lat, step = ternaries(path)
        r = compare(q_ref, lat_ref, q, lat)
        r["step"] = step
        r["path"] = path
        out["modeles"][name] = r
        t = r["transitions"]
        micro = (t["1->-1"] + t["-1->1"]) / max(r["n_poids_ternaires"], 1)
        print(f"[flips] {name:10s} step {step:5d}  cumulé "
              f"{100*r['flips_global']:6.2f}%  retournements ±1 "
              f"{100*micro:5.2f}%  latent {100*r['latent_rel_global']:5.2f}%",
              flush=True)

    # Churn par intervalle B (indépendant de t08).
    for a, b in [("B@23250", "B@23500"), ("B@23500", "B@final")]:
        if a in out["modeles"] and b in out["modeles"]:
            qa, _, _ = ternaries(dict(MODELS)[a])
            qb, _, _ = ternaries(dict(MODELS)[b])
            tot = ch = 0
            for k in qa:
                ch += int((qa[k] != qb[k]).sum())
                tot += qa[k].numel()
            out["modeles"][b][f"intervalle_depuis_{a}"] = ch / tot
            print(f"[flips] intervalle {a} -> {b}: {100*ch/tot:.2f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[flips] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
