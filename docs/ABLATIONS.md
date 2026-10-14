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
