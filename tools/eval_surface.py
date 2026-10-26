"""Métrique de surface F : taux hors lexique + classement log-proba.

Deux mesures indépendantes de la PPL et de la sonde exact-match :

1. TAUX HORS LEXIQUE
   Lexique construit sur le corpus d'entraînement dédupliqué (tranches 00-09),
   mots normalisés (minuscules, sans accents) de fréquence >= N. On compte la
   proportion de mots générés absents de ce lexique, séparément pour :
   - les 32 prompts de `eval_fixed` (greedy et sampling) ;
   - les 200 générations de la sonde gelée (`eval_probe.py`).
   Un taux élevé = mots inventés / morphologie cassée, ce que la PPL moyenne
   ne voit pas. On compte aussi les occurrences de la chaîne « capitain »
   (capitaine, capitaines, capitainerie...).

2. CLASSEMENT LOG-PROBA (1 bonne réponse vs 3 distracteurs)
   Pour chaque item de `data/probe/probe_fr_v1.jsonl` (fichier GELÉ, seulement
   lu, SHA256 vérifié), on tire 3 distracteurs déterministes dans la même
   catégorie (autres réponses acceptées de la sonde) et on classe les 4
   candidats par log-proba moyen conditionnel au prompt. Hasard = 25 %.
   Aucun fichier de la sonde n'est modifié : les distracteurs sont construits
   à la volée, seedés par (seed, id d'item).

Le moteur est le `.bin` CPU/NEON : mêmes conditions que la sonde.

Usage :
  .venv/bin/python tools/eval_surface.py \\
    --spec 't08|eval/F-Et08-fixed/samples.md|eval/Et08-probe/results.json|exports/E-t08.bin' \\
    --spec 'pilot-qa3|eval/F-Epilotqa3-fixed/samples.md|eval/Epilotqa3-probe/results.json|runs/E-pilot-qa3/model.bin' \\
    --spec 'tranche00|||exports/tranche00.bin' \\
    --lexicon-cache runs/F-diag/lexique_fr.tsv.gz \\
    --min-freq 5 20 \\
    --out-json docs/results/surface_F.json \\
    --out-md docs/SURFACE.md
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

PROBE_PATH = "data/probe/probe_fr_v1.jsonl"
PROBE_MANIFEST = "data/probe/probe_fr_v1.manifest.json"
WORD_RE = re.compile(r"[0-9a-z]+")
CAPITAIN_RE = re.compile(r"capitain")


# ---------------------------------------------------------------------------
# Lexique
# ---------------------------------------------------------------------------


def norm_text(text: str) -> str:
    """Minuscules, sans accents, œ/æ décomposés (pour le découpage en mots)."""
    text = text.replace("œ", "oe").replace("æ", "ae").replace("Œ", "OE").replace("Æ", "AE")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def words_of(text: str) -> list[str]:
    """Mots = suites [0-9a-z] après normalisation. Ponctuation = séparateur."""
    return WORD_RE.findall(norm_text(text))


def build_lexicon(glob_pattern: str, cache: str, min_keep: int = 2) -> dict[str, int]:
    """Compte les mots du corpus d'entraînement, cache TSV gzip trié."""
    import glob as _glob

    counter: collections.Counter[str] = collections.Counter()
    paths = sorted(_glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"Aucun fichier pour {glob_pattern}")
    t0 = time.time()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    text = line
                counter.update(words_of(text))
        print(f"[lexique] {os.path.basename(path)} : "
              f"{sum(counter.values()) / 1e6:.1f} M mots cumulés "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        for w, c in counter.most_common():
            if c >= min_keep:
                f.write(f"{w}\t{c}\n")
    print(f"[lexique] {len(counter)} types, cache -> {cache} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return dict(counter)


def load_lexicon(path: str) -> dict[str, int]:
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            out[w] = int(c)
    return out


def lexicon_at(counts: dict[str, int], min_freq: int) -> set[str]:
    return {w for w, c in counts.items() if c >= min_freq}


# ---------------------------------------------------------------------------
# Sources de texte
# ---------------------------------------------------------------------------


def parse_samples_md(path: str) -> dict[str, dict[str, str]]:
    """Parse un samples.md de eval_fixed -> {prompt: {greedy, sampling}}."""
    out: dict[str, dict[str, str]] = {}
    prompt = None
    mode = None
    buf: list[str] = []
    started = False

    def flush() -> None:
        nonlocal buf, started
        if prompt is not None and mode is not None and started:
            out.setdefault(prompt, {})[mode] = "\n".join(buf).strip()
        buf, started = [], False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("## ["):
                flush()
                prompt = line[4:].strip()
                mode, buf, started = None, [], False
            elif line.startswith("greedy ("):
                flush()
                mode, buf, started = "greedy", [], False
            elif line.startswith("sampling T=0.7 ("):
                flush()
                mode, buf, started = "sampling", [], False
            elif mode is not None:
                if not line.strip():
                    if started:
                        flush()
                    continue
                started = True
                buf.append(line)
    flush()
    return out


def load_probe_results(path: str) -> list[dict]:
    """Résultats de tools/eval_probe.py (JSON)."""
    d = json.load(open(path, encoding="utf-8"))
    return d["items"]


def load_probe_items(path: str) -> list[dict]:
    """Sonde gelée JSONL (une ligne = un item)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Statistiques de surface
# ---------------------------------------------------------------------------


def surface_stats(texts: list[str], lexicon: set[str]) -> dict:
    all_words: list[str] = []
    for t in texts:
        all_words += words_of(t)
    n = len(all_words)
    oov = [w for w in all_words if w not in lexicon]
    types = set(all_words)
    oov_types = [w for w in types if w not in lexicon]
    cap = sum(len(CAPITAIN_RE.findall(t)) for t in texts)
    top = collections.Counter(oov).most_common(12)
    return {
        "n_textes": len(texts),
        "n_mots": n,
        "n_oov": len(oov),
        "taux_oov": (len(oov) / n) if n else 0.0,
        "n_types": len(types),
        "n_types_oov": len(oov_types),
        "taux_types_oov": (len(oov_types) / len(types)) if types else 0.0,
        "capitain": cap,
        "top_oov": top,
    }


def collect_texts(samples_md: str, probe_results: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if samples_md:
        parsed = parse_samples_md(samples_md)
        out["eval_fixed_greedy"] = [v.get("greedy", "")
                                    for v in parsed.values()]
        out["eval_fixed_sampling"] = [v.get("sampling", "")
                                      for v in parsed.values()]
        assert len(parsed) >= 30, f"{samples_md}: {len(parsed)} prompts (<30)"
    if probe_results:
        items = load_probe_results(probe_results)
        out["sonde"] = [it.get("gen", "") for it in items]
        assert len(items) == 200, f"{probe_results}: {len(items)} items != 200"
    if out:
        out["combine"] = [t for group in out.values() for t in group]
    return out


# ---------------------------------------------------------------------------
# Classement log-proba
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_probe_frozen(probe: str, manifest: str) -> str:
    want = json.load(open(manifest, encoding="utf-8"))["sha256"]
    got = sha256_file(probe)
    if want != got:
        print(f"[surface] ERREUR sonde modifiée : {got} != {want}",
              file=sys.stderr)
        raise SystemExit(2)
    return got


def snapshot_engine(eng, last_logits) -> dict:
    """Copie l'état du KV cache + position + derniers logits après préfill."""
    return {
        "pos": eng.pos,
        "k_cache": eng.k_cache.copy(),
        "v_cache": eng.v_cache.copy(),
        "k_scales": eng.k_scales.copy(),
        "v_scales": eng.v_scales.copy(),
        "logits": last_logits.copy(),
    }


def restore_engine(eng, snap: dict) -> np.ndarray:
    """Restaure l'état et retourne une COPIE des logits du préfill."""
    eng.pos = snap["pos"]
    eng.k_cache[:] = snap["k_cache"]
    eng.v_cache[:] = snap["v_cache"]
    eng.k_scales[:] = snap["k_scales"]
    eng.v_scales[:] = snap["v_scales"]
    return snap["logits"].copy()


def score_candidate(eng, prompt_ids: list[int], cand_ids: list[int],
                    last=None) -> dict:
    """Log-proba conditionnel du candidat après le prompt (moteur .bin).

    `last` : logits à la position courante si le prompt a déjà été préfillé
    (mode snapshot). Sinon on préfille `prompt_ids`.
    """
    if last is None:
        for t in prompt_ids:
            last = eng.step(t)
    if last is None:
        raise ValueError("score_candidate : aucun logit disponible.")
    total = 0.0
    for j, t in enumerate(cand_ids):
        z = last.astype(np.float64)
        z -= z.max()
        lse = math.log(float(np.exp(z).sum()))
        total += lse - float(z[t])
        if j + 1 < len(cand_ids):
            last = eng.step(t)
    n = max(len(cand_ids), 1)
    return {"sum_logp": -total, "n_tokens": len(cand_ids),
            "mean_logp": -total / n}


def pick_distractors(item: dict, pool: list[str], correct: set[str],
                     rng: random.Random, k: int = 3) -> list[str]:
    cands = [x for x in pool if norm_text(x) not in correct]
    # Dédoublonne par forme normalisée, ordre stable.
    seen, uniq = set(), []
    for x in cands:
        key = norm_text(x)
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    if len(uniq) < k:
        raise ValueError(f"pool trop petit pour {item['id']}: {len(uniq)}")
    return rng.sample(uniq, k)


def rank_probe(eng, vocab: dict, tok, items: list[dict],
               seed: int = 0) -> dict:
    """Classe 1 bonne réponse (max alias) vs 3 distracteurs par log-proba."""
    import data as D

    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for it in items:
        by_cat[it["categorie"]].extend(it.get("reponses_acceptees") or [])

    recs = []
    for it in items:
        correct_aliases = it.get("reponses_acceptees") or []
        correct_norm = {norm_text(a) for a in correct_aliases if a.strip()}
        h = hashlib.sha256(f"{seed}:{it['id']}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        distractors = pick_distractors(it, by_cat[it["categorie"]],
                                       correct_norm, rng, 3)
        candidates = [("correct", a) for a in correct_aliases] + \
                     [("distracteur", d) for d in distractors]
        prompt_ids = D.encode_mapped(tok, vocab, it["prompt"], add_eos=False)
        # Marge : le plus long candidat + marge, pour ne pas dépasser seq_max.
        max_ans = max(len(D.encode_mapped(tok, vocab, text, add_eos=False))
                      for _, text in candidates)
        keep = max(eng.meta["seq_max"] - max_ans - 2, 1)
        prompt_ids = prompt_ids[-keep:]
        # Préfill une fois, snapshot ; chaque candidat repart de ce snapshot.
        eng.reset()
        last = None
        for t in prompt_ids:
            last = eng.step(t)
        snap = snapshot_engine(eng, last)
        scored = []
        for role, text in candidates:
            logits = restore_engine(eng, snap)
            ids = D.encode_mapped(tok, vocab, text, add_eos=False)
            s = score_candidate(eng, [], ids, last=logits)
            s.update({"role": role, "texte": text, "ids": ids})
            scored.append(s)
        # Le correct = meilleur de ses alias ; les distracteurs restent
        # des candidats séparés (un alias correct peut battre les 3 autres).
        correct = [s for s in scored if s["role"] == "correct"]
        distract = [s for s in scored if s["role"] == "distracteur"]
        best_correct = max(correct, key=lambda s: s["mean_logp"])
        allc = [best_correct] + distract
        allc.sort(key=lambda s: s["mean_logp"], reverse=True)
        rank_mean = 1 + sum(1 for s in distract
                            if s["mean_logp"] > best_correct["mean_logp"])
        allc_sum = sorted([best_correct] + distract,
                          key=lambda s: s["sum_logp"], reverse=True)
        rank_sum = 1 + sum(1 for s in distract
                           if s["sum_logp"] > best_correct["sum_logp"])
        recs.append({
            "id": it["id"], "categorie": it["categorie"],
            "slice": it["slice"], "perimetre": it["perimetre"],
            "correct": best_correct["texte"],
            "distracteurs": distractors,
            "rang_mean": rank_mean, "rang_sum": rank_sum,
            "hit_mean": rank_mean == 1, "hit_sum": rank_sum == 1,
            "scores": [{"role": s["role"], "texte": s["texte"],
                        "mean_logp": round(s["mean_logp"], 4),
                        "sum_logp": round(s["sum_logp"], 4),
                        "n_tokens": s["n_tokens"]} for s in scored],
        })

    def cell(rs: list[dict]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "top1_mean": sum(r["hit_mean"] for r in rs) / n if n else 0.0,
            "top1_sum": sum(r["hit_sum"] for r in rs) / n if n else 0.0,
            "mrr_mean": sum(1.0 / r["rang_mean"] for r in rs) / n if n else 0.0,
            "rang_moyen_mean": sum(r["rang_mean"] for r in rs) / n if n else 0.0,
        }

    cats = sorted({r["categorie"] for r in recs})
    slices = sorted({r["slice"] for r in recs})
    return {
        "global": cell(recs),
        "par_categorie": {c: cell([r for r in recs if r["categorie"] == c])
                          for c in cats},
        "par_slice": {s: cell([r for r in recs if r["slice"] == s])
                      for s in slices},
        "items": recs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_spec(spec: str) -> dict:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("--spec attend label|samples.md|probe_results.json|model.bin")
    return {"label": parts[0], "samples": parts[1], "probe_results": parts[2],
            "model": parts[3]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Surface F : OOL + log-proba.")
    ap.add_argument("--spec", action="append", required=True,
                    help="label|samples.md|probe_results.json|model.bin "
                         "(répétable ; chemins vides autorisés).")
    ap.add_argument("--lexicon-glob",
                    default="/Volumes/Lexar/clean-corpus/dedup/tranche_*.jsonl")
    ap.add_argument("--lexicon-cache",
                    default="runs/F-diag/lexique_fr.tsv.gz")
    ap.add_argument("--rebuild-lexicon", action="store_true")
    ap.add_argument("--min-freq", type=int, nargs="+", default=[5, 20],
                    help="Seuils de fréquence du lexique (le premier = primaire).")
    ap.add_argument("--probe", default=PROBE_PATH)
    ap.add_argument("--probe-manifest", default=PROBE_MANIFEST)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="docs/results/surface_F.json")
    ap.add_argument("--out-md", default="docs/SURFACE.md")
    a = ap.parse_args(argv)

    specs = [parse_spec(s) for s in a.spec]
    probe_sha = check_probe_frozen(a.probe, a.probe_manifest)
    print(f"[surface] sonde gelée OK sha256={probe_sha[:12]}…", flush=True)

    if os.path.exists(a.lexicon_cache) and not a.rebuild_lexicon:
        counts = load_lexicon(a.lexicon_cache)
        print(f"[surface] lexique chargé : {len(counts)} types "
              f"({a.lexicon_cache})", flush=True)
    else:
        counts = build_lexicon(a.lexicon_glob, a.lexicon_cache)
    lexicons = {n: lexicon_at(counts, n) for n in a.min_freq}
    primary_n = a.min_freq[0]
    print("[surface] lexique " + ", ".join(
        f"N>={n}: {len(lex)} types" for n, lex in lexicons.items()), flush=True)

    payload = {
        "sonde_sha256": probe_sha,
        "lexique": {"source": a.lexicon_glob, "cache": a.lexicon_cache,
                    "n_types": len(counts),
                    "seuils": {str(n): len(lex)
                               for n, lex in lexicons.items()}},
        "runs": {},
        "classement_logproba": {},
    }

    for sp in specs:
        label = sp["label"]
        print(f"[surface] === {label} ===", flush=True)
        run = {"samples_md": sp["samples"],
               "probe_results": sp["probe_results"],
               "model": sp["model"]}
        texts = collect_texts(sp["samples"], sp["probe_results"])
        run["surface"] = {}
        for n, lex in lexicons.items():
            run["surface"][f"N>={n}"] = {
                src: surface_stats(ts, lex) for src, ts in texts.items()
            }
        payload["runs"][label] = run

        if sp["model"]:
            import data as D
            import engine as E
            from main import load_tokenizer, decode_ids  # noqa: F401

            vocab = json.load(open(sp["model"] + ".vocab.json",
                                   encoding="utf-8"))
            tok = load_tokenizer(vocab["parent"])
            eng = E.Engine(sp["model"])
            t0 = time.time()
            rank = rank_probe(eng, vocab, tok,
                              load_probe_items(a.probe), a.seed)
            rank["duree_s"] = time.time() - t0
            payload["classement_logproba"][label] = rank
            g = rank["global"]
            print(f"[surface] {label} classement : "
                  f"top1={g['top1_mean']:.3f} MRR={g['mrr_mean']:.3f} "
                  f"rang={g['rang_moyen_mean']:.2f} "
                  f"({rank['duree_s']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(payload, open(a.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[surface] -> {a.out_json}")

    # --- Markdown ---
    lines = [
        "# Métrique de surface F — hors lexique et classement log-proba",
        "",
        f"Sonde gelée : `{a.probe}` (SHA256 `{probe_sha[:12]}…`, vérifié). "
        f"Lexique : corpus d'entraînement dédupliqué `{a.lexicon_glob}` ; "
        f"seuil primaire N>={primary_n}.",
        "",
        "## Taux de mots hors lexique",
        "",
        "| run | source | seuil | mots | hors lexique | taux | types OOV | « capitain » |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in payload["runs"].items():
        for n in a.min_freq:
            key = f"N>={n}"
            for src, s in run["surface"][key].items():
                lines.append(
                    f"| {label} | {src} | {n} | {s['n_mots']} | "
                    f"{s['n_oov']} | **{s['taux_oov']:.2%}** | "
                    f"{s['n_types_oov']}/{s['n_types']} "
                    f"({s['taux_types_oov']:.1%}) | {s['capitain']} |")
    lines += [
        "",
        "Le taux est un taux de tokens-mots (pas de types). `combine` = "
        "eval_fixed greedy + sampling + sonde. « capitain » compte les "
        "occurrences de la chaîne, pas seulement le mot exact.",
        "",
        "## Classement log-proba : 1 bonne réponse vs 3 distracteurs",
        "",
        "Hasard = 25 %. Score = log-proba moyen du candidat conditionné au "
        "prompt (moteur `.bin`, mêmes prompts gelés). Les distracteurs sont "
        "tirés dans la même catégorie, de façon déterministe (seed + id).",
        "",
        "| modèle | top-1 (moy.) | top-1 (somme) | MRR | rang moyen | par_coeur | reformule |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in payload["classement_logproba"].items():
        g = r["global"]
        pc = r["par_slice"].get("par_coeur", {}).get("top1_mean", float("nan"))
        rf = r["par_slice"].get("reformule", {}).get("top1_mean", float("nan"))
        lines.append(
            f"| {label} | **{g['top1_mean']:.1%}** | {g['top1_sum']:.1%} | "
            f"{g['mrr_mean']:.3f} | {g['rang_moyen_mean']:.2f} | "
            f"{pc:.1%} | {rf:.1%} |")
    lines += ["", "### Par catégorie", "",
              "| modèle | capitales | dates | definitions | unites_sciences |",
              "|---|---:|---:|---:|---:|"]
    for label, r in payload["classement_logproba"].items():
        cells = []
        for c in ("capitales", "dates", "definitions", "unites_sciences"):
            cells.append(f"{r['par_categorie'].get(c, {}).get('top1_mean', float('nan')):.1%}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Lecture",
        "",
        "- Le taux hors lexique mesure la propreté de surface : un modèle qui "
        "invente des mots peut avoir une PPL correcte sur les tokens fréquents.",
        "- Le classement log-proba teste une compétence plus fine que la "
        "génération libre : la bonne réponse est-elle mieux notée que trois "
        "réponses de la même catégorie ? Un score proche de 25 % signifie que "
        "le modèle ne sait pas discriminer, même sans produire la réponse.",
        "- Les deux mesures sont descriptives : elles ne remplacent ni la val "
        "gelée ni la sonde exact-match.",
        "",
        "## Limites",
        "",
        "- Le lexique vient de Wikipédia FR : il contient des noms propres et "
        "du vocabulaire technique. Il est donc permissif (sous-estime l'OOV).",
        "- Le découpage en mots ignore les apostrophes et la ponctuation : "
        "« l'eau » compte deux mots (`l`, `eau`).",
        "- Le classement log-proba dépend de la tokenisation du candidat et de "
        "la formulation du prompt ; il est comparable entre modèles, pas "
        "comme une probabilité absolue.",
        "- Les distracteurs viennent de la sonde elle-même (pas d'un jeu "
        "externe) : ils sont plausibles mais peuvent être trop faciles pour "
        "certaines catégories (dates, unités).",
        "",
    ]
    os.makedirs(os.path.dirname(a.out_md) or ".", exist_ok=True)
    open(a.out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"[surface] -> {a.out_md}")


if __name__ == "__main__":
    main()
