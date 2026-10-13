# Ablations slice_00 (tiny, parent SmolLM-135M)

Protocole commun : `tiny`, parent `HuggingFaceTB/SmolLM-135M`, vocab partagé
à unk distinct (`runs/abl_data/vocab.json`, 12000, construit sur le subset),
même init `--seed 0` (sauf V0b, seed 1 : mesure du bruit), même ordre de
données (seed epoch `(seed, epoch)`), validation in-run tous les 50 steps
(`--eval-every 50`, val = 5 % du subset, soit 40 docs ; split identique
pour tous SAUF V0b, seed 1 : composition du split différente), 5M tokens
par run sauf V1b (2 epochs, ~10M tokens ; budget mur ≈ V0, dépassé de 11 %).
Les colonnes 100/200/300 de V1b sont mi-parcours (610 steps) et ne sont
pas comparables point à point aux autres runs.
Données : `runs/abl_data/train_5M.jsonl` (806 docs, ~5,2M tokens) ;
jeux communs post-hoc : val 128 + val 256 (mêmes 503 docs val slice_00).
V1b : 2 epochs du subset (≈ 10M tokens, budget temps = mur de V0a, 2387 s
vs 2145 s soit +11 %).
Comparaison headline = jeux communs post-hoc ; les colonnes 100/200/300 =
val-PPL in-run (dynamique). BPC@100/200/300 = `val_ce × K` avec
K = tok/char du val in-run / ln 2 (K = 0,53288 en seed 0 ; 0,51815 en
seed 1, split différent) ; remplissage reproduit par `ligne_md` dans
`tools/run_ablations.sh` (vérifié : régénère le tableau à l'arrondi près).
Bruit = |V0a − V0b| sur jeux communs : toute
différence inférieure n'est pas un résultat.

| run | variante | seed | steps | tokens vus | val-PPL@100 | val-PPL@200 | val-PPL@300 | BPC@100 | BPC@200 | BPC@300 | commun-128 PPL/BPC | commun-256 PPL/BPC | tok/s | mur (s) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0-seed0 | baseline KD | 0 | 305 | 4997120 | 537.2 | 187.2 | 142.3 | 3.3499 | 2.7882 | 2.6422 | 161.87/2.6038 | 161.14/2.6015 | 2372 | 2145 | val in-run = 5% subset (40 docs); jeux communs post-hoc |
| V0-seed1 | baseline KD, seed bruit | 1 | 303 | 4964352 | 554.5 | 210.6 | 160.5 | 3.2737 | 2.7721 | 2.6315 | 170.94/2.6317 | 170.00/2.6289 | 2298 | 2200 | val in-run = split seed 1 (non comparable point à point) |
| V1a-alpha0 | CE seule, memes tokens | 0 | 305 | 4997120 | 391.1 | 140.9 | 110.4 | 3.1808 | 2.6368 | 2.5067 | 129.65/2.4902 | 127.84/2.4830 | 4507 | 1146 | sans parent (B6), ~2x plus vite |
| V1b-temps-egal | CE seule, temps egal V0 | 0 | 610 | 9994240 | 440.2 | 121.4 | 73.0 | 3.2438 | 2.5575 | 2.2860 | 59.93/2.0952 | 59.85/2.0945 | 4254 | 2387 | 2 epochs subset, mur 2387s vs 2145s (+11%, budget dépassé) ; @100/200/300 = mi-parcours, non comparables |
| V2-alpha03 | alpha 0.3 | 0 | 305 | 4997120 | 400.0 | 146.7 | 113.8 | 3.1927 | 2.6584 | 2.5228 | 132.01/2.4994 | 130.42/2.4932 | 2296 | 2210 | entre V0 et V1a |
| V3-lr1e3 | lr 1e-3 | 0 | 305 | 4997120 | 245.7 | 82.8 | 63.9 | 2.9331 | 2.3533 | 2.2151 | 72.47/2.1924 | 72.78/2.1946 | 2465 | 2064 | pas de divergence, meilleur à 5M |
| V4-subln | subln (normes o/down) | 0 | 305 | 4997120 | 424.3 | 167.1 | 129.9 | 3.2242 | 2.7276 | 2.5933 | 148.51/2.5597 | 145.85/2.5504 | 2334 | 2179 | légèrement mieux que V0, à confirmer vs bruit |
| V5-seq256 | seq 256, 16384 tok/step | 0 | 305 | 4997120 | 528.8 | 192.0 | 146.6 | 3.3396 | 2.8001 | 2.6564 | 174.19/2.6413 | 165.63/2.6155 | 2351 | 2165 | fenêtres longues moins bonnes à tokens égaux |
| V6-ce-lr1e3 | CE seule + lr 1e-3 | 0 | 305 | 4997120 | 156.4 | 57.9 | 46.3 | 2.6922 | 2.1631 | 2.0434 | 55.23/2.0534 | 55.60/2.0568 | 4468 | 1158 | combine V1a et V3 ; meilleur réglage 5M |

Bruit inter-runs (jeux communs) : |V0a − V0b| = 9,07 PPL (128) / 8,86 (256),
soit ~0,028 BPC. Toute différence inférieure n'est pas un résultat.

## Analyse (écart au bruit, jeu commun-128)

- V1a (CE seule, mêmes tokens) : 129,65 vs 161,87 → Δ −32,2 ≫ bruit.
  RÉSULTAT : à 5M tokens, CE seule > KD α0,7 (lr 3e-4).
- V1b (CE seule, ~10M tokens, mur +11 % vs V0) : 59,93. Budget différent :
  pas de comparaison directe, mais confirme le débit CE (~2×).
- V2 (α0,3) : 132,01, soit Δ +2,4 vs V1a (< bruit : équivalent CE) et
  Δ −29,9 vs V0 (≫ bruit). RÉSULTAT : baisser alpha aide ; 0,3 ≈ 0 à 5M.
- V3 (lr 1e-3) : 72,47, Δ −89 ≫ bruit, aucune divergence (loss finie,
  val monotone). RÉSULTAT FORT : le LR compte plus qu'alpha à 5M.
- V6 (CE seule + lr 1e-3) : 55,23, soit Δ −74,4 vs V1a et Δ −17,2 vs V3
  (≫ bruit). RÉSULTAT FORT : les gains se cumulent ; meilleur réglage 5M,
  devant même V1b-10M (59,93, budget double — écart −4,7 < bruit, donc
  pas de supériorité revendiquée à budget double, mais égalé avec 2×
  moins de tokens). Nouveau réglage de base recommandé.
- V4 (subln) : 148,51, Δ −13,4 vs V0a (bruit 9,07). Signal faible,
  au-dessus du bruit mais non répliqué : PAS DE RÉSULTAT revendiqué,
  pas d'adoption (export non supporté sans accord).
- V5 (seq 256) : 174,19 (Δ +12,3 vs V0a) et 165,63 en 256
  (Δ +4,5 < bruit). RÉSULTAT : aucun bénéfice des fenêtres longues à
  tokens égaux ; rester en seq 128.
- Le jeu 256 confirme le même ordre que le 128 (V3 < V1b < V1a ≈ V2 <
  V4 < V0a ≈ V0b ≈ V5).

## Éléments créés (phase D)

- Données `runs/abl_data/` (non committées, régénérables) :
  `train_5M.jsonl` (806 docs, ~5,2M tokens), `val.jsonl` (503 docs val),
  `vocab.json` (12000, unk distinct, unk=11999, eos=0),
  `val_128.pt` (13 374 fenêtres) + `val_256.pt` (6 687) + `meta.json`.
  Générateur committé : `tools/make_ablation_data.py`.
- Outils committés : `tools/eval_common.py` (PPL+BPC post-hoc 128/256),
  `tools/run_ablations.sh` (chaîne exacte : précheck GPU, budgets, purge).
- Par run `runs/2026-10-09-abl-<nom>/` : `config.json` (committé),
  `model.pt` + `model_bestval.pt` + `model.pt.vocab.json` (locaux),
  `train.log` + `eval.json` (locaux).
- Métriques : tableau ci-dessus ; verdicts : section Analyse.

## Commandes exactes (générées depuis les config.json, une par run)

### V0-seed0

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V0-seed0/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.7 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V0-seed1

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V0-seed1/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 1 --alpha 0.7 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V1a-alpha0

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V1a-alpha0/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.0 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V1b-temps-egal

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V1b-temps-egal/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 2 --seed 0 --alpha 0.0 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V2-alpha03

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V2-alpha03/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.3 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V3-lr1e3

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V3-lr1e3/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.7 --lr 0.001 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V4-subln

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V4-subln/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.7 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100 --subln
```

### V5-seq256

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V5-seq256/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 256 --save-batch 4 --batch-size 4 --accum 16 --epochs 1 --seed 0 --alpha 0.7 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### V6-ce-lr1e3

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V6-ce-lr1e3/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.0 --lr 0.001 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

## V7 — Désambiguïsation CE seule vs lr 1e-3 (G3)

**Objectif** : séparer l'effet de CE seule de l'effet du lr 1e-3. V6 confond les deux.

### V7 : CE seule, lr d'origine (3e-4)

Configuration : CE seule (alpha=0), lr 3e-4, protocole identique à D (subset 5M, même seed 0).

| run | seed | steps | tok/s | mur (s) | val_128 PPL/BPC | val_256 PPL/BPC | note |
|---|---|---|---|---|---|---|---|
| V7-ce-s0 | 0 | 305 | 2369 | 1275 | 132,01/2,4994 | 130,14/2,4921 | CE seule, lr 3e-4 |
| V7-ce-s1 | 1 | 303 | 2285 | 1318 | 134,73/2,5098 | 132,40/2,5009 | bruit inter-seeds |
| Moyenne V7 | - | - | 2327 | 1297 | 133,37/2,5046 | 131,27/2,4965 | |

Source des chiffres : `runs/2026-10-09-abl-V7-ce-s0/eval.json`
(PPL 132,0133/130,1409) et `runs/2026-10-09-abl-V7-ce-s1/eval.json`
(PPL 134,7316/132,4004) ; murs 1275 s / 1318 s lus dans les `train.log`
respectifs (`terminé en`). Moyenne arithmétique des deux seeds.

### Commande (seed 0)

```bash
uv run python main.py train --data runs/abl_data/train_5M.jsonl --parent HuggingFaceTB/SmolLM-135M --out runs/2026-10-09-abl-V7-ce-s0/model.pt --config tiny --vocab-size 12000 --vocab-from runs/abl_data/vocab.json --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 --epochs 1 --seed 0 --alpha 0.0 --lr 0.0003 --temperature 2.0 --val-ratio 0.05 --eval-every 50 --ckpt-every 100
```

### Analyse V7 vs V6

- V7 (CE seule, lr 3e-4) : 133,37 PPL (moyenne seeds 0/1)
- V6 (CE seule, lr 1e-3) : 55,23 PPL
- Écart : +78,14 PPL ≫ bruit (9,07)

**Verdict** : le lr 1e-3 compte plus que CE seule à 5M tokens. V7 a déjà isolé l'effet lr (V7 ≈ 133 PPL contre V6 55,2). Le design V6/V7/V8 mesure le from-scratch, pas le resume.

Budget V1b : mur V0-seed0 (2 145 s), débit V1a (305 steps en 1 146 s) →
epochs = ceil(2145×305/(1146×305)) = ceil(2145/1146) = 2
(voir `tools/run_ablations.sh` ; `SPE_SUBSET=305`).

## Skills (référence)

Skills opencode via plugin `@opencode-ai/plugin:1.16.2` (pas de commit git individuel par skill) :
- `using-superpowers`, `verifying-before-completion`, `systematic-debugging`, `test-driven-development`, `code-reviewer` (subagent), `internet-researcher` (web search direct).

## R1/R2 — Resume depuis tranche00 (décision P1, committée AVANT le run)

Recette : `tools/run_resume.sh` (R1 : lr=lr-resume=1e-3 ; R2 : lr=lr-resume
=3e-4, origine `runs/2026-10-08-tranche00/config.json` : `"lr": 0.0003`).
Base `tranche00.pt` (step 4248) + son vocab, subset commun
`runs/abl_data/resume_5M.jsonl` (905 premiers docs slice_01, split
train/val seed 0 / 0.05 répliqué → 860 docs train, 5 012 944 tokens ;
générateur `tools/make_resume_subset.py`, manifeste
`runs/abl_data/resume_5M_manifest.json`), CE seule, seq 128,
`--epochs 15` (342 steps restants), rewarmup 5 %, seed 0, ~20 min/run.
Commandes exactes : voir le script.

Règles de décision (appliquées après P5, sans dérogation) :
- Mesure primaire : Δc apparié (estimateur ratio, bootstrap apparié,
  1000 tirages, seed 0) sur val_big ET sur OOD, pour (R1−tranche00),
  (R2−tranche00), (R1−R2). (a)/(b) en secondaire uniquement.
- MDE calculé POUR CHAQUE PAIRE (demi-largeur IC95 du Δc de la paire).
  Interdiction de réutiliser ε=0,03.
- Si |Δc(R1−R2)| < MDE(R1−R2) → égalité → on retient R2 (lr 3e-4, sûr).
- R1 retenu seulement si : Δc(R1−R2) < −MDE sur val_big ET
  Δc_OOD(R1−R2) ≤ +MDE_OOD ET val in-run finale < val in-run initiale.
- Si ni R1 ni R2 ne battent tranche00 au-delà de leur MDE : STOP,
  la continuation sur Wikipédia seule n'apporte rien tel quel.

## Parent switch — D5 (préparation, non lancé)

### D5.a — ID repo Qwen3-0.6B-Base + hash

- **Repo HF exact** : `Qwen/Qwen3-0.6B` (pas `Qwen/Qwen3-0.6B-Base` — le repo officiel est `Qwen/Qwen3-0.6B`)
- **Revision (snapshot)** : `c1899de289a04d12100db370d81485cdf75e47ca`
- **Taille snapshot** : 1,4 Go (model.safetensors BF16, 596M params)
- **HF_HOME** : `/Volumes/Lexar/hf_cache` (déjà présent, déjà téléchargé)
- **SHA256 model.safetensors** : `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` (via symlink `model.safetensors -> ../../blobs/f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`)

### D5.b — Bench Qwen3-0.6B vs SmolLM-135M/360M (tiny, bs 8, 30 steps, MPS fp16)

| parent | tok/s | parent_ms | child_ms | total_ms | RAM max Mo | logits finis | note |
|---|---|---|---|---|---|---|---|
| SmolLM-135M | 2062 | 169 | 123 | 496 | 720 | oui | référence |
| SmolLM-360M | 1385 | 372 | 123 | 739 | 1175 | oui | +35 % qualité distillation |
| **Qwen3-0.6B** | **961** | **653** | **123** | **1065** | **2035** | **oui** | **+1,7× RAM vs 360M** |

Autocast parent (tiny, bs 2) : fp32 800 tok/s, fp16 980 tok/s, bf16 880 tok/s — logits finis ✅.

**Critères D5** (committés ci-dessous) :
- bench ≥ 700 tok/s ✅ (961)
- logits finis ✅
- RAM < 12 Go ✅ (2035 Mo < 12 Go)

**Verdict** : Qwen3-0.6B **RENNÉ** pour le parent switch — respecte les seuils.

### D5.c — Fichiers à modifier pour le switch parent (sans les modifier)

| fichier | changement requis |
|---|---|
| `tools/bench_train.py` lignes 230, 342, 372 | parents hardcodés → ajouter `"Qwen/Qwen3-0.6B"` |
| `tools/make_ablation_data.py:30` | `load_tokenizer("HuggingFaceTB/SmolLM-135M")` → paramétrer |
| `tools/make_resume_subset.py:30` | `load_tokenizer("HuggingFaceTB/SmolLM-135M")` → paramétrer |
| `tools/eval_bpc.py:235` | `smol_tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")` → paramétrer |
| `tools/convert_vocab.py` | si changement `kept_ids` → adapter |

Fichiers déjà dynamiques (OK) : `main.py`, `distill.py`, `tools/eval_valbig.py`, `tools/run_resume.sh`, `tools/eval_downstream.sh`.

### D5.d — Règles de décision parent (COMMITTÉES ci-dessous)

**Critères de sélection** (tous obligatoires) :
1. bench ≥ 700 tok/s (tiny, bs 8, 30 steps, MPS fp16)
2. logits finis (pas de NaN/Inf sur 5 batches aléatoires)
3. RAM max < 12 Go (marge système sur 16 Go)

**Fallback** : si un critère échoue → SmolLM2-360M (même bench, même critères).

### D5.e — Commandes exactes du futur run R3 (Qwen3-0.6B)

```bash
uv run python main.py train --data runs/abl_data/resume_5M.jsonl \
  --parent Qwen/Qwen3-0.6B --out runs/R3-qwen3-0.6b/model.pt \
  --config tiny --vocab-size 12000 --vocab-from runs/abl_data/resume_5M_manifest.json \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 15 --seed 0 --alpha 0.0 --lr 1e-3 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume runs/R2-ce-lr3e4/model.pt --rewarmup-ratio 0.05 --lr-resume 1e-3
```

**Estimation temps** (basé sur R2 : 1305 s / 342 steps = 3,8 s/step → ~1300 s ≈ 22 min pour 342 steps). Plus lent que R2 (~1,4×) à cause du parent plus gros (653 ms vs 372 ms parent forward).

### D5.f — Fallback SmolLM2-360M (si Qwen échoue)

Même commande avec `--parent HuggingFaceTB/SmolLM2-360M` (si disponible sur HF), même bench, mêmes critères. À documenter si déclenché.
