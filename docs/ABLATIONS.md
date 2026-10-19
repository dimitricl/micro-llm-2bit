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

Convention signes (F0c) : Δ(X−Y) = X−Y en BPC, négatif = X meilleur,
δ = |Δ|. Exemple : gate R2→R2b val_big Δc = −0,0054 (R2b meilleur
de 0,0054), OOD Δc = −0,0072. Tout tableau antérieur affichant +0,0054
pour R2b−R2 est corrigé en −0,0054.

## Parent switch — D5 (préparation, non lancé)

### D5.a — ID repo Qwen3-0.6B-Base + hash

- **Repo HF exact** : `Qwen/Qwen3-0.6B-Base` (modèle de base non-chat, logits propres pour distillation)
- **Revision (snapshot)** : `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- **Taille snapshot** : 1,1 Go (model.safetensors BF16, 596M params)
- **HF_HOME** : `/Volumes/Lexar/hf_cache` (déjà présent, déjà téléchargé)
- **SHA256 model.safetensors** : `2c465b10ceca99084a7d3d8451bd593ac1ea835f3516b7ccc7279b177351021f` (via symlink `model.safetensors -> ../../blobs/2c/2c465b10ceca99084a7d3d8451bd593ac1ea835f3516b7ccc7279b177351021f`)

### D5.b — Bench Qwen3-0.6B-Base vs SmolLM-135M/360M (tiny, bs 8, 30 steps, MPS fp16)

| parent | tok/s | parent_ms | child_ms | total_ms | RAM max Mo | logits finis | note |
|---|---|---|---|---|---|---|---|
| SmolLM-135M | 2062 | 169 | 123 | 496 | 720 | oui | référence |
| SmolLM-360M | 1385 | 372 | 123 | 739 | 1175 | oui | +35 % qualité distillation |
| **Qwen3-0.6B-Base** | **897** | **699** | **132** | **1142** | **2023** | **oui** | **base non-chat, logits propres** |

Autocast parent (tiny, bs 2) : fp32 800 tok/s, fp16 980 tok/s, bf16 880 tok/s — logits finis ✅.

**Critères D5** (committés ci-dessous) :
- bench ≥ 700 tok/s ✅ (897)
- logits finis ✅
- RAM < 12 Go ✅ (2023 Mo < 12 Go)

**Verdict** : Qwen3-0.6B-Base **RETENU** pour le parent switch — respecte les seuils.

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
### D5.e — Commande exacte du futur run R3 (Qwen3-0.6B-Base, KD D3)

```bash
uv run python main.py train --data runs/abl_data/resume_5M.jsonl \
  --parent Qwen/Qwen3-0.6B-Base --out runs/R3-qwen3-0.6b-base/model.pt \
  --config tiny --vocab-size 12000 --vocab-from runs/abl_data/resume_5M_manifest.json \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 15 --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume runs/2026-10-08-tranche00/tranche00.pt --rewarmup-ratio 0.05 --lr-resume 3e-4
```

**Règle D3 respectée** : KD (alpha 0.7), lr 3e-4, resume depuis tranche00.pt (step 4248), parent = Qwen3-0.6B-Base, schedule 342 steps, seed 0, subset 905 docs. Pas de CE (alpha > 0 charge le parent). R2b inchangé (copie exacte R2, seed 1).

**Estimation temps** (basée sur bench Qwen3-0.6B-Base KD) :
- total_step_ms = 1142 ms (bench C1, KD alpha=0.7, bs=8)
- Facteur backward+optim mesuré sur R1 : (145.5+34.9)/1142 ≈ 0,16 (source: logs/bench_qwen_base.log)
- Facteur observé sur R1 CE : total_step_ms / parent_ms = 496/169 ≈ 2,93
- Estimation KD Qwen : 1142 × 2,93 / 1,0 ≈ 3346 ms/step (facteur KD≈2,9 vs CE)
- 342 steps × 3,35 s/step ≈ **1145 s ≈ 19 min**

### D5.f — Fallback SmolLM2-360M (si Qwen échoue)

Même commande avec `--parent HuggingFaceTB/SmolLM2-360M` (si disponible sur HF), même bench, mêmes critères. À documenter si déclenché.

## R3 — Amendement post-mortem (3 déviations S2a + arrêt KD-Qwen)

### Déviation 1 : bs 4 vs bs 8 — pas d'OOM déterministe
Le passage bs 8 → bs 4 suivait un OOM MPS (Insufficient Memory) observé
sur le premier lancement R3, sous contention mémoire (modèle Ollama
résident ~7 Go + run). Log d'origine écrasé par les relances (non sourcé).
Re-test le 14/10 (`logs/probe_bs8.log`, flags K3, bs 8/accum 16, GPU
exclusif) : 981 tok/s, step 4260, **aucun OOM**. Hypothèse « bs 8 OOM de
façon déterministe » **infirmée** dans ces conditions. bs 4 reste le
standard conservateur (R3 a tenu 3591 steps sans OOM) ; bs 8 non retesté
à complétion. Statut : déviation documentée, sans preuve OOM reproductible.

### Déviation 2 : --vocab-from tranche00 vs manifest 5M
`runs/abl_data/resume_5M_manifest.json` ne contient QUE des stats
(K, n_train, plan, sha1) : `vocab['kept_ids']` → `KeyError` (reproduit :
`python -c "json.load(...)['kept_ids']"`). Le manifest n'a jamais été un
vocab valide. `--vocab-from tranche00.pt.vocab.json` (SmolLM kept_ids)
était le seul fichier valide — **mais invalide sémantiquement avec un
parent Qwen** (voir cause racine ci-dessous). Le refus B2
(`check_vocab_compatible`, `data.py:156`) ne claque que sur kept_ids
différents entre deux vocabs ; ici le vocab est identique à lui-même,
donc aucun refus : le trou est l'absence de contrôle
tokenizer-vs-vocab (Qwen tok 151643 vs kept SmolLM).

### Déviation 3 : schedule 7695 — arithmétique complète
À bs 4 (save-batch 4, micro-batch 4) : 8197 micros/epoch
(`logs/r3.log` : « 8197 batchs, 8197 micros/epoch, 513 steps/epoch
x 15 epochs = 7695 steps »). 513 = ceil(8197/16).
R2 (bs 8/accum 16) : 342 steps × 16 = 5472 micros, 5,6M tokens
(342×16×8×128).
R3 projeté : (3591 faits + 7695 reconstruits) = 11 286 steps × 16 × 4
× 128 = **92,4M tokens ≈ 22 époques** vs R2 5,6M (**×16,5**).
Comparaison D3 invalide : confond « KD » et « 16× plus de tokens ».
De plus le resume remettait le compteur à 0 par-dessus 3591 steps
(« nouveau run step 0 », `logs/r3_resume.log`). Sémantique B4 restaurée
(`distill.py`, commit ci-dessous) : compteurs cumulatifs, epochs
cumulatives.

### Cause racine : mismatch tokenizer/vocab (yinvalide R3)
R3 = tokenizer Qwen3 (`--parent`, 151643 tokens) + kept_ids SmolLM
(`--vocab-from` tranche00). Mesure CPU (sans GPU) : **46,45 %** des ids
Qwen du subset couverts par les kept SmolLM (211627/455571), **0 %
d'alignement sémantique** (Qwen id i ≠ SmolLM token i) ; référence SmolLM :
99,10 %. Conséquence : flux élève 54 % unk + tokens faux ; CE (30 %) +
KL (70 % vers logits Qwen indexés SmolLM) sur du garbage → collapse sur
id 0 (unk==eos ancien format) : e6 teacher-forced NLL **11,25**
(PPL ~77k), argmax==0 à **97,9 %** (`runs/R3-qwen3-0.6b-base/model_e6.pt`).
La val-PPL in-run 15,3 était mesurée dans l'espace d'indices Qwen
(auto-cohérente, non comparable à R2 22,9 en espace SmolLM).
Génération e6 : EOS immédiat (n_tokens=1, 32/32 prompts) —
`eval/R3-e6/metrics.json`, hors comparaison D3.
Corollaire B2 : `check_vocab_compatible` (`data.py:177`) **refuse**
tout kept_ids différent au resume → resume tranche00 + vocab Qwen
cohérent est impossible. La prémisse D3 (resume tranche00 + parent
d'une autre famille de tokenizer) est inimplémentable : KD-Qwen exigerait
vocab Qwen + départ from scratch (poids tranche00 = autre espace).
**Aucun run K3 relancé** : 1h de GPU pour un résultat prouvé invalide.

### Décision : STOP KD-Qwen sous prémisse D3
Options laissées à la décision : (i) KD-Qwen from scratch + vocab Qwen
(nouveau run ~1h, ne teste plus le resume) ; (ii) rester en CE (R1/R2) ;
(iii) parent switch repoussé à E avec vocab frais. En attente de « go ».

## R4 — KD-Qwen from-scratch (option i, exécuté)

Config miroir de R2 sauf : parent Qwen/Qwen3-0.6B-Base, vocab frais
`runs/abl_data/vocab_qwen.json` (12000, parent Qwen, unk_distinct v2,
couverture train 98,37 %, manifeste `docs/results/vocab_qwen_manifest.json`),
from-scratch (PAS de resume, PAS de lr-resume), alpha 0.7, lr 3e-4,
warmup 5 % from-scratch, bs 4/accum 16, epochs 1, seed 0, T 2.0.
Subset 905 docs identique. Schedule loggé : 8197 micros/epoch,
513 steps/epoch × 1 = **513 steps, 4,2M tokens** (`logs/R4.log`).
Temps réel **3884 s** (~1h05, ~1080 tok/s). Checkpoint intègre
(step=513, `runs/R4-qwen-kd/model.pt` + bestval ; intermédiaires purgés).

Confounders restants vs R2 : parent (Qwen vs SmolLM), vocab (Qwen 12k vs
SmolLM 12k), init (scratch vs resume tranche00), micro-batch (4 vs 8),
objectif (KD 0.7 vs CE). D'où la condition d'attribution ci-dessous.

Résultats (sources `docs/results/valbig_R4*.json`, `ood_R4*.json`) :
- val-PPL in-run finale **201,0** (best, monotone 1538→201).
- val_big vs R2 (réf c=1,6495) : R4 c=2,3341 →
  **Δc = +0,6847 [+0,6745, +0,6959], MDE 0,0107, FAIL**.
- OOD vs R2 (réf c=2,3641) : R4 c=2,9365 →
  **Δc = +0,5724 [+0,5674, +0,5769], MDE 0,0048, FAIL**.
- eval_fixed greedy (`eval/R4/metrics.md`) : repet4g **0,4168**,
  boucle moy **50,5** (max 60), distincts 2,2 — dégénéré (répétitions).

Décision D3 appliquée : OOD(R4−R2) = +0,5724 ≮ −δ_bruit (−0,0072) →
**KD NON retenue. CE (V6/R2) tranchée, pas de run contrôle**
(conformément à la règle : contrôle requis seulement si R4 gagne).

## R3-ORIGINAL — KD SmolLM-135M resume tranche00 (test D3 jamais lancé, exécuté)

Commande = R2 exacte, seules différences : --parent SmolLM-135M,
--alpha 0.7, --seed 0, --out runs/R3-smol135-kd/ (outil
`tools/run_r3_original.sh`). Même tokenizer des deux côtés, garde-fou
F0a silencieux vérifié. Schedule B4 : « 342 steps restants », total
4590, reprise step 4248 (`logs/R3.log`). Temps réel **2269 s** (~38 min,
~2500 tok/s) vs ETA ~40 min. Checkpoint intègre (step=4590).
Veilleur log-only : GO (val 29,4 → 28,0, jamais > 2× initiale).

Résultats (sources `docs/results/valbig_R3*.json`, `ood_R3*.json`) :
- val-PPL in-run finale **28,0** (best ; R2 : 22,9).
- val_big vs R2 (réf c=1,6495) : R3 c=1,7290 →
  **Δc = +0,0795 [+0,0747, +0,0844], MDE 0,0049**
  (perte +4,8 % ≤ 50 % ✓).
- OOD vs R2 (réf c=2,3641) : R3 c=2,2846 →
  **Δc = −0,0795 [−0,0827, −0,0757], MDE 0,0035**
  (< −δ_bruit −0,0072 ✓, ~10× le seuil).
- eval_fixed greedy (`eval/R3/metrics.md`) : repet4g **0,3255**,
  boucle moy **2,1** (max 26, un prompt outlier), distincts 6,7 —
  vs R2 0,3078/1,3/3/15,2 : légèrement plus boucleur, même ordre.

Décision D3 (règles hash 13aa7f1) : OOD(R3−R2) < −δ ✓ ET perte val_big
≤ 50 % ✓ → **KD VALIDÉE (espace SmolLM)**. La KD paie +4,8 % in-domain
pour −0,08 BPC OOD : effet régularisant du teacher, invisible aux
ablations V (from-scratch) et inversé chez R4-Qwen (confondue init).

Amendement : la conclusion « CE tranchée » du livrable R4 est **retirée**.
Elle reposait sur R4-Qwen (from-scratch, confondue init + parent) :
le vrai test KD-vs-CE à budget et espace constants (R3 vs R2) donne
l'inverse sur OOD. Chapitre KD ROUVERT, pas clos.

Suite : paire from-scratch (CE-vocab-Qwen vs KD-Qwen, même init, même
budget) pour tester « meilleur parent » avant E — spec à part, pas ce run.

## E — Spec du run principal (contrat du go, E gelé)

### P1 — Parent : SmolLM-135M par défaut, sans test 360M
Le test subset KD-360M (~1h GPU) est un run long : interdit par
l'en-tête E-PREP (« AUCUN RUN LONG »). **Renoncement explicite** au
test, donc **parent E = SmolLM-135M, sans discussion**.
Contexte chiffré existant (`tools/bench_train.md`) : 135M 2577 tok/s
bs16, 360M 1748 tok/s ; KD-135M validée D3 (R3 : Δc OOD −0,0795).
Fourche Qwen PARKÉE (R4 confondue, e6 poubelle).

### P2 — Données (mesures `docs/results/dedup_report.json`, audit
`docs/results/dedup_audit.json`, tranches `/Volumes/Lexar/clean-corpus/dedup/`)
- Corpus slices 00-09 : **99965 docs, 1012,2 M chars** avant.
- Doublons exacts (sha1 normalisé) : **0**. Near-dup MinHash
  (125 sigs shingle-5, LSH 25×5, Jaccard vérifié ≥ 0,85) : **3 docs**.
  Après : **99962 docs, 1012,2 M chars** (corpus déjà propre).
- Langues (stopwords) : **fr 99924 / en 41 / unk 0**. Pattern
  (fiches/listes/quiz) : **5329 docs (5,3 %)**. Audit stratifié 200 docs
  committé (`docs/results/dedup_audit.json`).
- 10 tranches ~105 Mo : t00-t08 ≈ **37,2M tokens** chacune, t09 24,2M
  (total 359,2M). Manifests avec tokens (tokenizer SmolLM) dans le rapport.
- **E utilise les tranches dédupliquées 01-09** (la 00 ≈ slice_00 déjà
  consommée par tranche00 est sautée : pas de réentraînement).
- Vocab : **RESTE 12000 espace SmolLM**. L'idée « 16k » est **enterrée** :
  E reprend `tranche00.pt` dont l'embedding est 12000 lignes indexées
  SmolLM ; changer de taille/kept_ids casserait le mapping poids↔tokens
  (B2 `check_vocab_compatible` refuserait, `data.py:177`). Pas de 16k.

### P4 — Recette E
- Objectif KD alpha 0.7 (R3), parent SmolLM-135M, lr 3e-4, warmup 5 %,
  rewarmup 5 % par tranche, seed 0, T 2.0, seq 128, val 5 %, eval/50.
- Micro-batch : **bs 8 / accum 16** (R3 : 2500 tok/s mesurés, zéro OOM
  en GPU exclusif). bs 4 seulement avec preuve OOM-négative committée
  (à ce jour : aucune — probe bs 8 passe).
- Schedule B4 cumulative depuis step 4248 (tranche00) :
  total_N = 4248 + Σ steps/tranche. À 16384 tok/step (8×16×128),
  tranche ~37,2M tok ≈ 2270 steps → total_9 ≈ 4248 + 8×2270 + 1479
  (t09 24,2M) ≈ **23887**. Tokens E totaux ≈ 321,9M (+34,8M tranche00).
- ETA au débit MESURÉ R3 (2500 tok/s) : 37,2M/2500 ≈ 14880 s ≈ 4,1 h
  par tranche → **~37 h les 9 tranches** + ~10 min d'évals par tranche.
  **Amendement t01 (run en cours)** : débit réel ~1500 tok/s, pas 2500.
  Cause mesurée : eval-every-50 sur val gelée 500 docs (~4 min/eval)
  vs 45 docs chez R3 — l'overhead eval dilue la moyenne (entraînement
  lui-même inchangé : GPU exclusif, 65 % mem libre, disque idle).
  ETA t01 révisée ≈ **7,5 h**. Run conforme à la spec (flags intacts),
  pas de STOP : seule l'estimation était fausse.
- Checkpoints par tranche : best-val (100 Mo) + final (300 Mo) gardés,
  `_ckpt*` (garde 2 en cours) et `_e*.pt` purgés après validation.
  Budget disque : 9 × 400 Mo ≈ **3,6 Go** (+2 _ckpt courants 600 Mo).
- Évals par tranche : val-PPL in-run, Δc val_big + OOD vs tranche00
  (gate apparié), eval_fixed 32 prompts + gate répétition P3
  (seuil sampling 0,15, câblé dans `verdict()`).
- STOP : **2 régressions OOD > δ** (δ = max(MDE, δ_bruit)) → STOP ;
  **gate répétition FAIL → tranche rejetée** (chaîne stoppée, diagnostic).
- Garde-fous actifs : staging F0b sur les 5 sites `torch.save`,
  règle stall S1, lock `logs/gpu.lock`, `ollama ps` vide, F0a silencieux
  attendu (même tokenizer).

## E-PILOTE t01 — pre-flight (committé avant le run)
- Val gelée : `runs/abl_data/val_E_frozen.jsonl` (500 docs, 5 149 388 chars,
  seed 0, pool dedup t00), manifeste+hash `docs/results/val_E_frozen_manifest.json`.
  Disjonction SHA1 vérifiée : t01-t09, val_big, OOD.
- Arithmétique t01 (plan exact `logs/plan_t01.log`) : 10236 docs,
  36355 micros, **SPE=2273**, E_CUMUL=3, total=6819, **restants=2571 steps**
  (≈42,1M tokens, ~1,13 époque). ETA 16849 s ≈ **4,7 h** à 2500 tok/s.
  Garde : total ≈ 6819 et « restants » ≈ 2571 dans le log, sinon kill < 50 steps.
- Disque : 650 Gi libres (`df`, budget 3,6 Go).
- F0a : silencieux attendu (tokenizer SmolLM des deux côtés).

## E-PILOTE t01 — run + décision (t02-t09 NON autorisés)
- Run : schedule 2273×3=6819, restants 2571, F0a silencieux, 1 kill
  précoce (bug `python3` système → epochs=1, garde STOP < 50 steps OK),
  fix + relance. Durée **25220 s (~7,0h)**, tok/s réel ~1670 (< 2000 :
  eval-overhead val gelée 500 docs, investigué, amendement ETA committé).
  Zéro incident (pas de NaN, pas d'EINTR — staging F0b OK), veilleur GO.
- Val gelée : 29,2 → **25,8** (best=final, monotone). Refs : tranche00 29,7,
  R2 22,9 (autre val : sens et ordre de grandeur OK).
- Gate vs tranche00 (`docs/results/valbig_Et01*.json`, `ood_Et01*.json`) :
  val_big **Δc −0,0711** [−0,0746,−0,0678] ; OOD **Δc −0,0557**
  [−0,0568,−0,0544] ; loops 6/32 pires (Wilson 0,089) ;
  répétition sampling 0,0216 < 0,15 → **PASS global** (exit 0).
  Compteur STOP OOD : **0 strike** (amélioration, pas régression).
- eval_fixed t01 : greedy 0,2824/1,4 — sampling 0,0216/1,5
  (vs R2 0,3078/1,3 — R3 0,3255/2,1) : le moins boucleur mesuré.

## E-SUITE t02-t09 — amendements Q0 (committés avant t02)

### Q0a — eval-every 100 (val gelée intouchée)
Courbe t01 monotone 29,2→25,8, best=final (`logs/Et01.log`, 51 evals).
Coût mesuré d'une eval gelée : **239 s** (`logs/measure_eval.log`,
1761 batchs, E-t01 sur MPS). 51 evals ≈ 12 200 s ≈ **3,4 h de t01
sur 7 h = la moitié du run en eval**.
eval-every 100 → ~26 evals/tranche : économie ≈ 25×270 s ≈ **6750 s
≈ 1,9 h/tranche ≈ 15 h sur t02-t09**. La courbe garde 26 points
(surveillance intacte) ; divergence précoce aussi visible via loss
train (toutes les 20 steps) + veilleur. La val gelée elle-même est
inchangée (comparabilité préservée).

### Q0b — resume depuis final (continuité B4)
t01 : best=final (6800, monotone) — les deux coïncident ici, mais la
règle générale retient **final (model.pt)** : seul lui porte l'état
complet (opt/sched/epoch/micro/order/RNG) pour continuité B4 exacte ;
bestval = poids seuls (nouvel optimiseur). bestval gardé comme fallback
documenté si tail suspecte (val remonte ≥2 evals consécutives en fin
de tranche — trancherait alors contre final, avec justification).
Continuité scheduler t02 : prev step 6819 (final t01) ; SPE t02 ≈ 2268
(manifest dedup t02, 37,15M tok). ceil() donnerait 5 epochs (2 époques
de données neuves — surestimation dès que PREV n'est pas multiple de
SPE) : l'outil utilise round() avec garde (restants ≥ SPE/2) →
E_CUMUL = 4, restants ≈ 2253 steps (~0,99 époque, parité t01).
(L'outil calcule l'exact au lancement depuis le plan réel ; garde
schedule-guard identique.)

### Q0c — plancher perf : 1300 tok/s reportés post-warmup
t01 : min 1429 post step 4350 (n=123 lignes, `logs/Et01.log`),
moyenne ~1550-1680. Train-pur estimé ~2500 (cohérent R3 ; l'écart aux
reportés = overhead eval, pas contention). Règle : moyenne reportée
< 1300 soutenue (hors 100 premiers steps) → **pas de tranche suivante
avant cause trouvée** (STOP t-suivante, run en cours non tué). Avec
eval-every 100 la moyenne remontera vers ~2000 ; le plancher 1300 reste
un garde contention/disque valide.

## P3 — Leviers de décodage (t08) : balayage CPU, amendement post-chaîne

Défaut ouvert mesuré : `runs/E-t08/model.bin` (best E) boucle en **greedy**
(`taux_repetition_4g` 0,5057 ; `plus_longue_boucle` 1,3) alors qu'il est propre
en **sampling** (0,0484 / 1,4). Le harnais ne testait pas les leviers : il
appelait `eng.generate` sans `repetition_penalty` ni `no_repeat_ngram_size`
(déjà présents dans `engine.generate`, défauts OFF). Amendement : deux options
rétro-compatibles (`--repetition-penalty`, `--no-repeat-ngram`) propagées aux
**deux** appels ; aux défauts, **aucun kwarg** n'est ajouté → sortie
**bit-à-bit identique** (test `tests/test_eval_fixed_decode.py`, dont un test
`slow` comparant à l'octet à `eval/Et08/metrics.json`).

**Protocole** : CPU pur, noyau NEON, mêmes 32 prompts, mêmes seeds, sampling
inchangé ; `nice -n 10` ; aucun GPU/MPS, aucun train, aucun lock GPU, aucun
Ollama. Sorties `eval/Et08-decode/<cfg>/`. Contrôles d'identité :
**C0 == `eval/Et08/metrics.json`** et **C5 == `eval/Et08/metrics.json`**, tous
deux **identiques à l'octet** (`cmp`) ; C4 rejoué → identique bit-à-bit
(déterminisme).

| cfg | réglage | repet4g moy | boucle moy / max | mots distincts moy / min | prompts repet>0,15 | prompts repet>0 | n_tok moy |
|---|---|---|---|---|---|---|---|
| C0 | 1.0 / 0 (défaut) | **0,5057** | 1,31 / 3 | 10,8 / 5 | 30/32 | 32/32 | 60,0 |
| C1 | pén. 1,15 | 0,3275 | 1,28 / 3 | 13,6 / 7 | 25/32 | 31/32 | 58,8 |
| C2 | pén. 1,30 | 0,2083 | 1,22 / 3 | 16,5 / 6 | 19/32 | 29/32 | 56,5 |
| C3 | no-repeat-3g | **0,0000** | 1,62 / 3 | 19,8 / 7 | 0/32 | 0/32 | 58,4 |
| **C4** | **pén. 1,15 + 3g** | **0,0000** | **1,47 / 2** | **20,0 / 8** | **0/32** | **0/32** | 57,5 |
| C5 | sampling réf. | 0,0484 | 1,38 / 3 | 20,6 / 5 | 2/32 | 22/32 | 56,7 |

Contrôle de diversité hors métrique 4-grammes (répétition de 5/6/8-grammes de
mots, moyenne sur 32 prompts) : C0 0,353/0,332/0,300 → C1 0,183/0,164/0,119 →
C2 0,088/0,069/0,036 → **C3 et C4 0,000/0,000/0,000** → C5 0,011/0,009/0,008.
Ratio de mots distincts (types/tokens) : 0,389 (C0) → 0,511 → 0,648 →
0,649 (C3) → **0,698 (C4)** → 0,786 (C5).

**Verdict — oui, un réglage tue la boucle greedy sans détruire la diversité :
C4 (`repetition_penalty=1.15`, `no_repeat_ngram_size=3`).**
- Répétition : 0,5057 → **0,0000** ; **0 prompt sur 32** conserve un seul
  4-gramme répété (C0 : 32/32), et 0/32 dépasse le seuil de gate 0,15.
- Diversité : mots distincts 10,8 → **20,0** (+85 %), soit le niveau du
  sampling (20,6) ; minimum par prompt 5 → 8 ; ratio types/tokens 0,389 →
  0,698. Aucune « destruction » mesurable.
- Boucles : max 3 → **2** (seul réglage à faire baisser le pire cas) ; moyenne
  1,31 → 1,47 (+0,16), effet mécanique du ban de 3-grammes qui ne peut pas
  interdire un unigramme — c'est précisément ce que la pénalité 1,15 corrige.
- La pénalité seule **ne suffit pas** : C1 (0,3275) et C2 (0,2083) laissent
  respectivement 25/32 et 19/32 prompts au-dessus de 0,15 ; le greedy
  re-choisit les mêmes 4-grammes, il faut le ban dur. C3 seul laisse la boucle
  moyenne monter à 1,62 : **C4 est le seul réglage qui gagne sur les deux axes
  à la fois**.
- Effet de bord mesuré : C4 provoque 3/32 arrêts EOS précoces (<60 tokens)
  contre 0/32 en C0 — mais le sampling en a 4/32, et les longueurs moyennes
  restent comparables (57,5 vs 56,7). 3 prompts restent sous 70 % de la
  diversité du sampling (`La Révolution française`, `Demain nous irons`,
  `Le Moyen Âge`) : la limite est le **modèle**, pas le décodeur.
- Sur le sampling, C4 est aussi ≥ au défaut (0,0000 vs 0,0484 ; 21,4 vs 20,6
  mots ; 1,31 vs 1,38) — mais le sampling est déjà conforme : on ne change pas
  la référence historique.

**Défaut de déploiement documenté** : `runs/E-t08/model.bin` doit être servi en
**greedy avec `repetition_penalty=1.15` + `no_repeat_ngram_size=3`**, ou en
sampling T=0,7 inchangé. Le greedy « nu » (1.0/0) est **interdit au
déploiement** : il boucle sur 32/32 prompts. Le gate de chaîne, qui ne mesure
que le sampling, était **aveugle** à ce défaut (cf. « ce qui contredit » §6.2) ;
il est désormais mesurable par `eval_fixed --repetition-penalty/--no-repeat-ngram`
sans réentraînement.

## P4 — Sonde FR v1 : la métrique « répond-il au fait ? » (baseline t08)

La PPL ne mesure ni le suivi d'instruction ni le format Q/A : t08 produit une
continuation FR plausible même quand la question exige un fait précis. Sonde
dédiée, **gelée avant toute mesure**.

### P4.a — Jeu de sondes (gelé)

`data/probe/probe_fr_v1.jsonl` — **200 items**, SHA256
`d9817afb15d85248c3f062d8b4b994e8b6c9ac3411de12f01923701f8815ba7c`
(manifeste `data/probe/probe_fr_v1.manifest.json`, générateur déterministe
`tools/make_probe.py`). Chaque item : `{id, categorie, slice, perimetre,
prompt, reponses_acceptees, note}`.

- 4 catégories × 2 slices, **50/50 par catégorie** : `capitales` 80 (40/40),
  `dates` 40 (20/20), `definitions` 40 (20/20), `unites_sciences` 40 (20/20).
- `slice` = cœur du diagnostic : **`par_coeur`** = formulation proche des
  templates du corpus (`La capitale de l'Italie est`, `{pays} est un pays dont
  la capitale est`, `la Révolution française date de`, `L'eau bout à`) ;
  **`reformule`** = même fait, formulation jamais vue (`Quelle est la capitale
  de l'Italie ?`, `Quand a eu lieu … ?`, `À quelle température … ?`). Chaque
  fait produit **deux** items appariés (même paire pc/rf, même fait).
- `perimetre` (décision A/B) : `domaine_etroit` = `capitales` (80) ;
  `wiki_general` = `dates` + `definitions` + `unites_sciences` (120).
- `reponses_acceptees` = alias multiples ; 76 items à alias numérique.
- **Gel** : hash SHA256 vérifié à chaque exécution ; hash divergent →
  `SystemExit(2)` (anti-ajustement a posteriori). Aucun item modifié après
  la première mesure ; toute évolution = `probe_fr_v2`.

### P4.b — Règle de score (exacte)

`tools/eval_probe.py`, CPU/NEON, greedy **avec les défauts de déploiement
P3** (`repetition_penalty=1.15`, `no_repeat_ngram_size=3`), `max_new=32`.
1. Normalisation : NFD + suppression des diacritiques, minuscules,
   ponctuation (`unicodedata` catégorie `P*`) → espace, espaces compactés.
2. **HIT** si un alias normalisé apparaît dans le texte généré normalisé
   **avec frontières de tokens** (donc `100` ≠ `1000`, `New Delhi` ≠
   `NewDelhi`). Alias vide ignoré.
3. **EOS-collapse** = génération terminée en **< 5 tokens**.
4. Agrégats : global, par `slice`, par `categorie`, par `perimetre`,
   `categorie × slice`, avec `taux_eos_collapse` par cellule.

### P4.c — Baseline t08 (`docs/results/probe_Et08_v1.json`, source unique)

Modèle `runs/E-t08/model.bin`, sonde hash `d9817afb…`, 200 items.

| axe | valeur | hits | taux | IC95 (Wilson) | EOS-collapse |
|---|---|---|---|---|---|
| **global** | — | **4/200** | **0,0200** | [0,0078 ; 0,0503] | 1/200 = 0,0050 |
| slice | `par_coeur` | 3/100 | 0,0300 | [0,0103 ; 0,0845] | 0,0000 |
| slice | `reformule` | 1/100 | 0,0100 | [0,0018 ; 0,0545] | 0,0100 |
| périmètre | `domaine_etroit` | 1/80 | 0,0125 | [0,0022 ; 0,0675] | 0,0000 |
| périmètre | `wiki_general` | 3/120 | 0,0250 | [0,0085 ; 0,0709] | 0,0083 |
| catégorie | `capitales` | 1/80 | 0,0125 | — | 0,0000 |
| catégorie | `dates` | 0/40 | 0,0000 | [0 ; 0,0876] | 0,0000 |
| catégorie | `definitions` | 0/40 | 0,0000 | [0 ; 0,0876] | 0,0250 |
| catégorie | `unites_sciences` | 3/40 | 0,0750 | [0,0258 ; 0,1994] | 0,0000 |

Détail `categorie × slice` (hits/n) : capitales pc 0/40, rf **1**/40 ;
dates pc 0/20, rf 0/20 ; definitions pc 0/20, rf 0/20 ;
unites_sciences pc **3**/20, rf 0/20.

**Contrôle pré-E / post-E** : `exports/tranche00.bin` (modèle pré-E, même
sonde, même décodeur) = **0/200** (IC95 [0 ; 0,0183]), EOS-collapse 4/200
= 0,0200. Delta E : **+4 hits** — au niveau du bruit d'un échantillon de 200
(IC qui se recouvrent largement).

### P4.d — Verdict factuel, sans enjolivement

1. **Le modèle ne répond pas.** 4/200 = **2,0 %**, IC95 [0,8 % ; 5,0 %] : la
   borne haute exclut tout taux utile. 196/200 réponses sont des continuations
   de corpus sans rapport avec le fait demandé (« La capitale de l'Italie est »
   → « une commune française, située dans le département de la Manche »).
2. **Le gap `par_coeur` vs `reformule` existe mais il est minuscule :
   +2,0 points (3,0 % vs 1,0 %), et il n'est pas significatif** — les IC95 se
   recouvrent presque entièrement ([0,0103 ; 0,0845] vs [0,0018 ; 0,0545]).
   Sur 200 items, un écart de 2 hits ne prouve rien. **Le modèle ne récite même
   pas ses templates** : 0/40 sur `capitales` en `par_coeur`, la formulation
   la plus proche du corpus. C'est le résultat le plus dur de cette mesure.
3. **Le seul axe non nul est `unites_sciences` en `par_coeur` (3/20, 15 %)**,
   et il est **fragile par construction** : ce sont des alias numériques
   (`1000`, `100`, `1000`) retrouvés dans une énumération ou une mesure
   (« 1000 m de longueur », « un mètre de 100 à 150 mèter »), pas une réponse
   à la question. Sur les 4 hits, **2 seulement ont l'alias dans les 3
   premiers mots** ; aucun n'est un format Q/A. Un exact-match numérique sur
   texte libre est **permissif** : ce chiffre est un **majorant**, pas une
   performance.
4. **Le périmètre le moins désespéré est `wiki_general` (2,5 %)**, pas
   `domaine_etroit` (1,25 %) — et uniquement via `unites_sciences` en
   `par_coeur`. `capitales` (le cœur du périmètre étroit, et la catégorie la
   plus représentée du corpus) plafonne à 1/80, sur un hit `reformule`
   fortuit (« La ville de Paris est une ville du pays de l'Ouest »).
5. **`dates` et `definitions` sont à 0/40** : aucune date, aucun mot-clé
   définitionnel. Les dates produites sont fausses et hors sujet (Révolution
   française → 1870 ; Bastille → 1944).
6. **L'EOS-collapse n'est pas le problème** (1/200 = 0,5 %) : le modèle ne
   s'arrête pas, il **dévie**. 183/200 générations atteignent la borne de 32
   tokens. Le défaut n'est donc pas « il refuse de répondre » mais « il ne
   comprend pas qu'on lui demande un fait ».
7. **Conséquence pour la décision A/B** : la sonde est maintenant
   l'instrument de mesure, et elle place la barre très bas (2 %). Tout ce qui
   suit (post-tune, D5 Qwen3-0.6B, format Q/A dans les données) se jugera sur
   **ce même JSONL gelé**, avec le même décodeur, et devra dépasser
   nettement l'IC95 haut de la baseline (5,0 %) pour compter.

## P5 — Pilote QA étroit : format + répétition (t08 → E-pilot-qa)

Hypothèse testée : une recette QA étroite (templates `par_coeur` + répétition
massive) **déplace-t-elle** les chiffres de la sonde gelée P4, ou bute-t-elle
sur un mur de capacité ? Périmètre entraîné : `capitales` + `unites_sciences`
uniquement. CPU/NEON pour la mesure, GPU exclusif pour le seul run.

### P5.a — Corpus gelé (`data/qa/qa_fr_pilot_v1.jsonl`)

- **60 faits** extraits de la sonde (source de vérité) : 40 capitales +
  20 unités/sciences. **Formulations `par_coeur` uniquement** ; toute
  formulation `reformule` (et tout « ? ») est **interdite et vérifiée à la
  génération** (`assert_no_leak` → AssertionError). Les questions de la
  sonde restent donc **jamais-vues** : c'est le test de généralisation.
- Structure : chaque fait → **40 paragraphes distincts** (rotations/ordres des
  mêmes phrases déclaratives), chacun dupliqué **×19** (facteur fixe,
  documenté) = 45 600 paragraphes QA.
- **7,0 % de texte FR général** (192 docs de `tranche_05.jsonl`, permutation
  déterministe seed 0, docs plafonnés à 4 000 tokens) pour éviter
  l'effondrement et garder l'EOS.
- **6 209 536 tokens** (48 512 fenêtres de 128) → **SPE = 379 steps/époque**.
  Depuis le step 22 740 de t08 : **E_CUMUL = 61 → 379 steps, exactement une
  époque neuve** (garde schedule-guard : restants 379 ∈ [SPE/2 ; 1,5·SPE]).
- SHA256 JSONL : `eb5bfd6470e26a98a97e374d169bcdbc833a6fe19af82485ebc5092b1a0e814e`
  (manifeste `data/qa/qa_fr_pilot_v1.manifest.json`, générateur déterministe
  `tools/make_qa_corpus.py`).
- Aucune fuite train/val : la val gelée (`val_E_frozen`) vient de t00,
  disjointe ; contrôle explicite dans `prepare_corpus`.

### P5.b — Run (GPU exclusif, un seul job)

`tools/run_pilot_qa.sh` : recette E identique (parent SmolLM-135M, KD α 0.7,
T 2.0, lr 3e-4, resume `runs/E-t08/model.pt`, rewarmup 0.05, bs 8/accum 16,
seq 128, seed 0, val gelée, eval-every 100). Lock `logs/gpu.lock` pris/libéré.
Durée 3 181 s. **Poids livrés : `runs/E-pilot-qa/model.bin`**.

| val gelée (best) | t08 (réf.) | E-pilot-qa | Δ |
|---|---|---|---|
| val_ce | 2,9851 | **3,1409** | **+5,2 %** |
| val_ppl | 19,8 | 23,1 | +16,7 % |

Train-PPL ~1,7 vs val-PPL ~23,1 : **surapprentissage assumé** sur 60 faits
répétés (avertissement du trainer). La garde anti-oubli **PASS** (Δ val_ce
+5,2 % ≪ +50 %), mais sur les **60 faits** le modèle est massivement surappris
— c'est le prix de la recette, pas un défaut du run.

### P5.c — Mesure (sonde P4 inchangée, décodeur de déploiement inchangé)

`docs/results/probe_Epilotqa_v1.json` (source unique). Sonde SHA256 inchangé
`d9817afb…` ; greedy pen 1,15 + no-repeat-3g ; max_new 32.

| axe | t08 (réf.) | **E-pilot-qa** | IC95 Wilson | statut |
|---|---|---|---|---|
| **global** | 4/200 = 0,0200 | **81/200 = 0,4050** | [0,3394 ; 0,4742] | — |
| slice `par_coeur` | 3/100 = 0,0300 | **48/100 = 0,4800** | [0,3846 ; 0,5768] | mémorisation |
| slice `reformule` | 1/100 = 0,0100 | **33/100 = 0,3300** | [0,2456 ; 0,4269] | généralisation |
| `capitales` (**entraîné**) | 1/80 = 0,0125 | **56/80 = 0,7000** | [0,5923 ; 0,7894] | — |
| `unites_sciences` (**entraîné**) | 3/40 = 0,0750 | **25/40 = 0,6250** | [0,4703 ; 0,7578] | — |
| `dates` (non entraîné) | 0/40 | **0/40 = 0,0000** | [0 ; 0,0876] | contrôle |
| `definitions` (non entraîné) | 0/40 | **0/40 = 0,0000** | [0 ; 0,0876] | contrôle |
| périmètre `domaine_etroit` | 1/80 | 56/80 = 0,7000 | [0,5923 ; 0,7894] | — |
| périmètre `wiki_general` | 3/120 | 25/120 = 0,2083 | [0,1453 ; 0,2895] | — |
| EOS-collapse global | 1/200 | **7/200 = 0,0350** | — | — |

Détail `categorie × slice` (hits/n, IC95) :

| catégorie (entraînée) | `par_coeur` | `reformule` |
|---|---|---|
| `capitales` | **33/40 = 0,825** [0,680 ; 0,913] | **23/40 = 0,575** [0,422 ; 0,715] |
| `unites_sciences` | **15/20 = 0,750** [0,531 ; 0,888] | **10/20 = 0,500** [0,299 ; 0,701] |
| `dates` / `definitions` | 0/20, 0/20 | 0/20, 0/20 |

Les **catégories non entraînées sont restées à 0/40** : la progression ne
vient donc pas d'un artefact du décodeur ou du harnais, mais bien de la
donnée QA. Contrôle négatif propre.

### P5.d — Verdict : **VERT** (règle pré-enregistrée)

Règle : sur les catégories **entraînées**, `reformule` ≥ 15-20 % **et** borne
basse IC95 Wilson > borne haute baseline t08 (5,03 %).

- `capitales` `reformule` = **57,5 %**, IC95 basse **42,2 %** > 5,03 %.
- `unites_sciences` `reformule` = **50,0 %**, IC95 basse **29,9 %** > 5,03 %.
- Les deux sont **≥ 15-20 %** avec une marge considérable.

→ **La recette QA marche** : +38,5 points de réussite globale, et surtout
+32 points sur des formulations **jamais vues à l'entraînement**. Un tiny de
25 M params **peut** mémoriser et restituer des faits sous question tournée.

**Mécanisme observé (honnête, sans enjolivement)** : le modèle ne produit pas
un format Q/A ; il **latche l'entité de la question** puis déroule une phrase
`par_coeur` mémorisée portant le bon fait — souvent déformée en surface
(« dont la **capitaine** est Paris », « la ville de Mosc**u** »). La
**généralisation est donc de la récupération de fait robuste à la
formulation**, pas du suivi d'instruction de format. Correct quant au fait,
dégradé quant à la langue. Sur les hits `reformule`, l'alias apparaît
rarement en tête (pos ≥ 7 mots) : le bon fait émerge au fil d'une phrase
récitée, il ne répond pas sèchement.

**Signaux secondaires** :
- EOS-collapse monte à **7/200**, tous en `capitales` `par_coeur` (17,5 % de
  cette cellule, 0 en `reformule`) : le modèle a appris à s'arrêter après
  certains prompts mémorisés, mais **parfois trop tôt** (EOS après le
  prompt) — effet de bord de la répétition.
- `dates`/`definitions` **inchangés à 0/40** : rien n'a fuité hors périmètre ;
  la baseline P4 est bien un plancher « pas entraîné ».
- val-PPL +5,2 % seulement malgré le surapprentissage massif des 60 faits :
  le texte général (7 %) suffit à amortir l'oubli.

### P5.e — Décision

Le verrou « le modèle ne sait aucun fait » (**2 %**) est **levé** : la recette
QA étroite est une **brique efficace**. Elle justifie le run cher (base 768 /
from-scratch), qui doit maintenant porter **plus de faits** et **un format
Q/A explicite** (le modèle restitue le bon fait mais ne répond pas *dans les
formes*). Points d'attention hérités : (i) la langue se dégrade (surappris),
(ii) l'EOS-collapse sur prompts mémorisés, (iii) le `reformule` est de la
récupération, pas encore du suivi d'instruction.

## P6 — Pilote QA #2 : dose rééquilibrée (t08 → E-pilot-qa2)

Hypothèse testée : **la dose** de la recette QA. Même recette que P5, mais
~500 faits (au lieu de 60), répétition **×4** (au lieu de ×19) et **35 % de
FR général** (au lieu de 7 %), sur les **4 catégories** de la sonde. Question :
les faits tiennent-ils **sans casser la langue** ? CPU/NEON pour la mesure,
GPU exclusif pour le seul run.

### P6.a — Corpus gelé (`data/qa/qa_fr_pilot_v2.jsonl`)

- **534 faits** : `capitales` 160, `dates` 168, `definitions` 105,
  `unites_sciences` 101 — dont **les 100 items `par_coeur` de la sonde**
  (40 capitales + 20 dates + 20 définitions + 20 unités) pour garder la
  mesure mémorisation/généralisation sur les mêmes items.
- **13 paragraphes distincts par fait × 4 répétitions = 27 768 paragraphes
  QA** (4 répétitions **pleines** : `n_docs_qa = 534×13×4`, aucune troncature).
- **35,73 % de FR général** (1 329 docs de `tranche_05.jsonl`, plafond 3 000
  tokens, permutation déterministe seed 0).
- **6 766 592 tokens** (52 864 fenêtres) → **SPE = 413 steps/époque**. Depuis
  le step 22 740 de t08 : **E_CUMUL = 56 → 388 steps** (0,94 époque ; garde
  schedule-guard OK).
- Formulations `par_coeur` **uniquement** ; garde `assert_no_leak` **élargie
  aux 4 catégories** (100 formulations `reformule` interdites) et « ? »
  interdit dans le QA. Les 100 prompts pc de la sonde sont des **préfixes
  exacts** d'une phrase d'entraînement (test dédié).
- SHA256 JSONL : `596d2eec50cfe7fdc5db2e1419f63664525f511d5fd3a2ead3dde77bfab44475`
  (manifeste `data/qa/qa_fr_pilot_v2.manifest.json`).
- **v1 reste reproductible à l'identique** : `tools/make_qa_corpus.py
  --version v1` régénère `eb5bfd64…` (vérifié). Les tables de faits v2 sont
  dans `tools/qa_facts_v2.py` (déterministes, vraies, mots-clés vérifiés).

### P6.b — Run (GPU exclusif, un seul job)

`tools/run_pilot_qa2.sh` : recette E identique (parent SmolLM-135M, KD α 0.7,
T 2.0, lr 3e-4, resume `runs/E-t08/model.pt`, rewarmup 0.05, bs 8/accum 16,
seq 128, seed 0, val gelée, eval-every 100). Lock `logs/gpu.lock` pris/libéré ;
`ollama stop gemma3:1b` avant et après. Durée **3 183 s**, exit 0.
**Poids livrés : `runs/E-pilot-qa2/model.bin`** (11,56 Mo).

| val gelée (best, step 23 100) | t08 (réf.) | E-pilot-qa2 | Δ |
|---|---|---|---|
| val_ce | 2,9851 | **3,0553** | **+2,35 %** |
| val_ppl | 19,8 | **21,2** | **+7,07 %** |

Le trainer signale toujours un écart train/val (`train-PPL` ~2,7 vs `val-PPL`
~21,2) : surapprentissage des faits, mais **le doublement du général et la
baisse de la répétition divisent par ~2,4 la casse de la val** (P5 : +16,7 %
de PPL ; ici +7,1 %).

### P6.c — Mesure (sonde P4 inchangée, décodeur de déploiement inchangé)

`docs/results/probe_Epilotqa2_v1.json` (source unique). Sonde SHA256 inchangé
`d9817afb…` ; greedy pen 1,15 + no-repeat-3g ; max_new 32.

| axe | t08 (réf.) | pilote #1 (×19, 7 %) | **pilote #2 (×4, 35 %)** |
|---|---|---|---|
| **global** | 4/200 = 2,00 % | 81/200 = 40,50 % | **21/200 = 10,50 %** [6,97 ; 15,52] |
| `capitales` `reformule` | 1/40 = 2,5 % | 23/40 = 57,5 % | **0/40 = 0,0 %** [0 ; 8,76] |
| `unites_sciences` `reformule` | 0/20 = 0 % | 10/20 = 50,0 % | **3/20 = 15,0 %** [5,24 ; 36,04] |
| `dates` (total) | 0/40 | 0/40 | **3/40 = 7,5 %** [2,58 ; 19,86] |
| `definitions` (total) | 0/40 | 0/40 | **7/40 = 17,5 %** [8,75 ; 31,95] |
| val-PPL | 19,8 | 23,1 (**+16,7 %**) | **21,2 (+7,07 %)** |
| EOS-collapse | 1/200 | 7/200 (3,5 %) | **15/200 (7,5 %)** |
| surface | référence | « capitaine », déformé | **corruption comparable** (« capitaine », « l'Allemandie », « l'Escrive ») |

Détail `categorie × slice` (hits/n, IC95 Wilson) :

| catégorie | `par_coeur` | `reformule` |
|---|---|---|
| `capitales` | 5/40 = 12,5 % [5,46 ; 26,11] | **0/40 = 0,0 %** [0 ; 8,76] |
| `dates` | 2/20 = 10,0 % [2,79 ; 30,10] | 1/20 = 5,0 % [0,89 ; 23,61] |
| `definitions` | 4/20 = 20,0 % [8,07 ; 41,60] | 3/20 = 15,0 % [5,24 ; 36,04] |
| `unites_sciences` | 3/20 = 15,0 % [5,24 ; 36,04] | 3/20 = 15,0 % [5,24 ; 36,04] |

Sanité langue (`tools/eval_fixed.py`, 32 prompts, `eval/Epilotqa2-fixed/`) :
- **déploiement** (greedy + pen 1,15 + no-repeat-3g) : `repet4g` **0,0000**,
  boucle max **3**, distincts moy 19,1 → **propre** ;
- greedy **nu** (1,0 / 0) : `repet4g` **0,3322**, boucle max **59** → dégénère
  comme tous les checkpoints de la chaîne : le décodeur de déploiement reste
  **obligatoire** (P3), ce n'est pas un défaut du run.

### P6.d — Verdict : **MITIGÉ**

Règle pré-enregistrée : `reformule` ≥ 15-20 % sur les catégories entraînées
**et** borne basse IC95 > 5,03 % (borne haute t08) **et** langue tenue
(val-PPL ≤ +15 %) **et** corruption de surface nettement réduite.

- **Faits : critère NON atteint.** `reformule` global **7,0 %** (IC95
  [3,43 ; 13,75]) < 15 % ; `capitales` `reformule` **0/40** ; seules
  `definitions` et `unites_sciences` atteignent 15 % (3/20, borne basse
  5,24 %, à la limite). Le global (10,5 %) reste > baseline t08 (2 %), mais
  **très en dessous de pilote #1** (40,5 %).
- **Langue : tenue sur la val** (+2,35 % CE, +7,07 % PPL ≤ 15 %) et
  **nettement moins abîmée que P5** en PPL. En revanche la **corruption de
  surface n'est pas nettement réduite** (« capitaine », « l'Allemandie ») et
  l'**EOS-collapse double** (15/200 vs 7/200).
- **Lecture franche** : avec ×4, **les faits ne tiennent pas**. La répétition
  massive (×19) était **l'ingrédient essentiel** de la mémorisation de P5 ;
  le général à 35 % achète la langue mais **dilue les faits**. Le périmètre
  élargi transmet un **signal faible** (definitions/dates passent de 0 à
  non-nul), pas une compétence.
- **Ce que ça arbitre** : pour le run final, il faut **garder une répétition
  élevée** (≥ 10-19) sur un **ensemble de faits modéré**, et traiter le
  texte général comme une **minorité** (≤ ~10-15 %). La « dose » n'est pas
  un curseur où l'on peut tout avoir : à budget de tokens fixé, répétition
  et général se **substituent**.

### P6.e — Décision

Pas de run supplémentaire dans cette étape (consigne : STOP après verdict).
Le couple (faits × langue) n'est pas résolu par la dose ; il faudra soit
**séparer** les phases (mémorisation QA forte puis reprise générale courte),
soit **accepter** la casse de langue au profit des faits. En attente de
décision (run base 768 / D5 / clôture).
