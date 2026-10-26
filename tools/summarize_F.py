"""Tableau comparatif de la chaîne F (t08, t09, bras A/B/C).

Lit uniquement des sorties existantes (aucun recalcul) :
- val-PPL gelée : dernière ligne ``[eval] ... val-PPL ~X`` du journal train ;
- val_big / OOD : résumés ``docs/results/{valbig,ood}_LBL.json`` ;
- gate apparié : ``logs/F/gate_{valbig,ood}_FARM.txt`` (PASS/FAIL) ;
- sonde gelée : ``agregats.global.taux`` ;
- surface : ``docs/results/surface_F.json`` (OOV N>=5 combiné, « capitain »,
  classement log-proba) ;
- flips ternaires : dernier ``[val] ... flips=`` du journal train.

Écrit ``docs/RESULTS_F.md`` et ``docs/results/F_summary.json``.

Usage : .venv/bin/python tools/summarize_F.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

REF_C = {"valbig": -0.1949, "ood": -0.1708}  # t08 (référence de l'hypothèse)
REF_PPL = 19.8
MODELS = [
    ("t08", "runs/E-t08/train.log", "Et08", "eval/Et08-probe/results.json"),
    ("t09", "runs/E-t09/train.log", "Et09", ""),
    ("F-A", "runs/F-A/train.log", "F-A", "eval/F-A-probe/results.json"),
    ("F-B", "runs/F-B/train.log", "F-B", "eval/F-B-probe/results.json"),
    ("F-C", "runs/F-C/train.log", "F-C", "eval/F-C-probe/results.json"),
]


def read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_train_log(path: str) -> dict:
    txt = read_text(path)
    out: dict = {}
    m = re.findall(r"val-PPL ~([0-9.]+)", txt)
    if m:
        out["val_ppl"] = float(m[-1])
    m = re.findall(r"terminé en ([0-9]+)s", txt)
    if m:
        out["duree_s"] = int(m[-1])
    flips = re.findall(r"flips=[0-9]+/[0-9]+ \(([0-9.]+)%\)", txt)
    if flips:
        out["flips_pct"] = float(flips[-1])
    return out


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bpc_delta(label: str, ref: str) -> dict | None:
    """Δ BPC (modèle − tranche00) sur a/b/c, depuis les résumés."""
    new = load_json(f"docs/results/{label}.json")
    base = load_json(f"docs/results/{ref}.summary.json")
    if not new or not base:
        return None
    return {k: new[k]["bpc"] - base[k]["bpc"] for k in "abc"}


def parse_gate(path: str) -> dict:
    """Δ et IC appariés lus dans la sortie de gate_paire.py."""
    txt = read_text(path)
    out: dict = {}
    for k, d, lo, hi in re.findall(
            r"^\[([abc])\] Δ=(-?[0-9.]+) IC95=\[(-?[0-9.]+), (-?[0-9.]+)\]",
            txt, flags=re.M):
        out[k] = {"delta": float(d), "lo": float(lo), "hi": float(hi)}
    m = re.findall(r"^(PASS|FAIL)$", txt, flags=re.M)
    out["verdict"] = m[-1] if m else "—"
    return out


def gate_verdict(path: str) -> str:
    return parse_gate(path).get("verdict", "—")


def surface_row(surface: dict | None, label: str) -> dict:
    if not surface or label not in surface.get("runs", {}):
        return {}
    r = surface["runs"][label]
    comb = r["surface"]["N>=5"]["combine"]
    lp = surface["classement_logproba"].get(label, {}).get("global", {})
    return {
        "oov_combine": comb["taux_oov"],
        "capitain": comb["capitain"],
        "logproba_top1": lp.get("top1_mean"),
        "logproba_mrr": lp.get("mrr_mean"),
    }


def fixed_repetition(path: str) -> float | None:
    d = load_json(path)
    if not d:
        return None
    vals = []
    for v in d.get("metrics", {}).values():
        g = v.get("greedy", {})
        if "taux_repetition_4g" in g:
            vals.append(g["taux_repetition_4g"])
    return sum(vals) / len(vals) if vals else None


def probe_taux(path: str) -> float | None:
    d = load_json(path)
    if not d:
        return None
    return d.get("agregats", {}).get("global", {}).get("taux")


def main() -> int:
    surface = load_json("docs/results/surface_F.json")
    rows = []
    for name, log, tag, probe in MODELS:
        row: dict = {"modele": name, "tag": tag}
        gv: dict = {}
        go: dict = {}
        row.update(parse_train_log(log))
        # Les fichiers de mesure utilisent le tag compact (FA, FB, FC).
        compact = tag.replace("-", "")
        vb = bpc_delta(f"valbig_{compact}", "valbig_tranche00")
        od = bpc_delta(f"ood_{compact}", "ood_tranche00")
        row["valbig"] = vb
        row["ood"] = od
        if tag.startswith("F-"):
            gv = parse_gate(f"logs/F/gate_valbig_{compact}.txt")
            go = parse_gate(f"logs/F/gate_ood_{compact}.txt")
            row["gate_valbig"] = gv.get("verdict", "—")
            row["gate_ood"] = go.get("verdict", "—")
            row["sonde"] = probe_taux(probe)
            row["repetition_4g"] = fixed_repetition(f"eval/{tag}-fixed/metrics.json")
        elif tag == "Et08":
            row["sonde"] = probe_taux(probe)
            row["repetition_4g"] = fixed_repetition("eval/F-Et08-fixed/metrics.json")
        row.update(surface_row(surface, "t08" if tag == "Et08" else tag))
        # Critère étape 4 : val-PPL < 19,8 ET Δc apparié (gate) meilleur que
        # t08 sur val_big et OOD. Réfs : -0,1949 / -0,1708.
        gc_vb = (gv or {}).get("c", {}).get("delta")
        gc_od = (go or {}).get("c", {}).get("delta")
        row["gate_dc_valbig"] = gc_vb
        row["gate_dc_ood"] = gc_od
        if tag.startswith("F-") and "val_ppl" in row:
            ok = (row["val_ppl"] < REF_PPL and gc_vb is not None
                  and gc_od is not None
                  and gc_vb < REF_C["valbig"] and gc_od < REF_C["ood"])
            row["critere_ok"] = bool(ok)
        rows.append(row)

    os.makedirs("docs/results", exist_ok=True)
    with open("docs/results/F_summary.json", "w", encoding="utf-8") as f:
        json.dump({"ref_ppl": REF_PPL, "ref_dc": REF_C, "rows": rows},
                  f, ensure_ascii=False, indent=1)

    def pct(x):
        return "—" if x is None else f"{100 * x:.2f} %"

    lines = [
        "# Chaîne F — tableau comparatif",
        "",
        "Référence de l'hypothèse : t08, val-PPL gelée 19,8, "
        "Δc val_big −0,1949, Δc OOD −0,1708.",
        "Δ BPC = modèle − tranche00 (négatif = meilleur). "
        "Gate = verdict apparié `gate_paire.py` (seuil ε = 0,03).",
        "Critère étape 4 : val-PPL < 19,8 **et** Δc val_big et OOD meilleurs "
        "que t08.",
        "",
        "| modèle | val-PPL | Δb val_big | Δc val_big | gate | Δb OOD | Δc OOD "
        "| gate | sonde | OOV N≥5 | « capitain » | logproba top-1 | flips "
        "| durée |",
        "|---|---:|---:|---:|:--:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        vb, od = r.get("valbig") or {}, r.get("ood") or {}

        def f(x, nd=4):
            return "—" if x is None else f"{x:.{nd}f}"

        lines.append(
            "| {m} | {ppl} | {vb} | {vc} | {gv} | {ob} | {oc} | {go} "
            "| {sonde} | {oov} | {cap} | {lp} | {fl} | {dur} |".format(
                m=r["modele"],
                ppl=f(r.get("val_ppl"), 1),
                vb=f(vb.get("b")), vc=f(vb.get("c")),
                gv=r.get("gate_valbig", "—"),
                ob=f(od.get("b")), oc=f(od.get("c")),
                go=r.get("gate_ood", "—"),
                sonde=pct(r.get("sonde")),
                oov=pct(r.get("oov_combine")),
                cap="—" if r.get("capitain") is None else r["capitain"],
                lp=pct(r.get("logproba_top1")),
                fl="—" if r.get("flips_pct") is None else f"{r['flips_pct']:.2f} %",
                dur="—" if r.get("duree_s") is None else r["duree_s"] // 60,
            )
        )
    lines += [
        "",
        "Durée en minutes (journal d'entraînement, hors mesures). "
        "Les cases vides correspondent à des mesures non faites pour t09 "
        "(pas de sonde ni de surface).",
        "",
    ]
    for r in rows:
        if "critere_ok" in r:
            lines.append(f"- {r['modele']} : critère étape 4 "
                         f"{'ATTEINT' if r['critere_ok'] else 'NON ATTEINT'}.")
    with open("docs/RESULTS_F.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
