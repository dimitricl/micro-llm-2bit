# Rapport complet — micro-llm-2bit (8→9 oct 2026)

Généré le 9 oct 2026. Sources : `checkpoints/*.log`, `tools/bench_train.md`,
`runs/`, `eval/`, `exports/` (bench frais du jour), `docs/RUNS.md`.

## 1. Projet
LLM ternaire 2 bits (BitNet b1.58), decoder-only, distillation KL depuis
SmolLM-135M/360M. Inférence CPU Apple Silicon (NEON + ctypes), budget
**delta RSS < 100 Mo**. Configs : tiny (~23M) / base (~73M).
Toolchain : mise (Python 3.11.15) + uv + make (clang, GCD). Repo
`dimitricl/micro-llm-2bit`, 17 commits, arbre propre, 1 stash WIP.
Tests : **33/33 verts**. Code ~4300 lignes (quant, model, engine memmap,
distill KL+CE+early-stopping+cache mmap, data split val, main CLI, tools,
data_fetch, data_clean).

## 2. Bench inférence — tous les exports (mesuré le 9 oct, `--bench-tokens 32`)

| export | config | .bin | budget struct. | +RSS charg. | +RSS génér. | verdict | vit. |
|---|---|---|---|---|---|---|---|
| enfant.bin | tiny V8000 seq128 | 10.06 Mo | 10.14 Mo | +0.3 Mo | +25.5 Mo | OK | 95.0 tok/s |
| mini_test.bin | tiny V2000 seq64 | 6.97 Mo | 6.91 Mo | +0.2 Mo | +18.3 Mo | OK | 101.7 tok/s |
| tranche00.bin | tiny V12000 seq128 | 12.13 Mo | 12.12 Mo | +0.1 Mo | +27.4 Mo | OK | 76.2 tok/s |
| wiki_slice00_bestval.bin | tiny V12000 seq128 | 12.13 Mo | 12.12 Mo | +0.2 Mo | +27.4 Mo | OK | 77.8 tok/s |
| enfant_think.bin | tiny V12000 seq128 | 12.13 Mo | 12.12 Mo | +0.2 Mo | +27.5 Mo | OK | 76.1 tok/s |
| enfant_base.bin | base V12000 seq256 | 29.25 Mo | 30.21 Mo | +0.5 Mo | +50.6 Mo | OK | 36.6 tok/s |
| enfant_base_fr.bin | base V12000 seq256 | 29.25 Mo | 30.21 Mo | +0.5 Mo | +50.7 Mo | OK | 41.6 tok/s |

Logs d'époque : bench_base 42.3 tok/s (RSS 195.6→280.3→354.9),
bench_fr 41.3 tok/s (195.6→280.3→355.0). T1 memmap : chargement
+85→+0.5 Mo (base), +0.2 Mo (tiny) ; génération +159→+51 Mo (base),
+26 Mo (tiny). Greedy bit-identique avant/après.

## 3. Bench entraînement (`tools/bench_train.py`, mini M4, seq 128, 30 steps)

| enfant | parent | bs2 | bs4 | bs8 | bs16 |
|---|---|---|---|---|---|
| tiny | 135M | 555 | 1888 | 2311 | **2577** |
| tiny | 360M | 900 | 1420 | 1641 | 1748 |
| base | 135M | 666 | 990 | 1239 | 1407 |
| base | 360M | 503 | 828 | 1008 | 1086 |

Détail step (ms) : parent ~60 % du step (ex. tiny/135M/bs16 : parent 255,
enfant 172, loss 8.3, backward 205, optim 18.3). RAM max 849–1396 Mo.
Autocast parent (tiny/360M/bs2) : fp32 869, **fp16 1093 (+26 %, finis)**,
bf16 934. Retenu : tiny + 135M + bs16 + fp16 + direct.
Tranche 100 Mo (~26M tokens) ≈ 2.9 h/epoch (réel tranche00 ~2400 tok/s).

### Cache top-k vs direct (120 steps jumeaux, même init)
- direct : valCE **0.799** (PPL 2.2), 50 s
- k32 : valCE 1.138 (PPL 3.1), 27 s (+86 % écart valeurs)
- k64 : valCE 1.064 (PPL 2.9), 27 s (+17 % même à k=512)
Lecture cache 135x plus rapide, step caché 1843 tok/s, mais val 35-40 %
moins bonne, k64 ≈ k32 (pas de convergence). **DÉCISION : parent direct.**
Ancien .npz abandonné (faux sous mélange).

## 4. Runs et métriques train/val

### 4.1 Premiers runs (corpus mix 297 docs, 1.7M chars, parent 360M)
- tiny 3 epochs (639 steps, 859 s, ~1520 tok/s) : step20 loss 8.96/CE 8.58,
  step200 loss 3.74/CE 3.72/PPL 41, fin loss ~3.3. Infer dégénéré (attendu).
- base EN 12 epochs (636+636 steps, ~795 tok/s) : step20 loss 9.81/CE 7.78,
  fin CE ~2.5-3.7. Français salade (corpus EN).
- **base FR 40 epochs** (34 docs, 406k chars, 1200 steps, ~795 tok/s) :
  step20 loss 10.53 → step860 loss 1.22/CE 0.98 → **step1200 loss 0.94,
  KL 1.04, CE 0.59, PPL 1.8**. Pattern pays parfait, libre fragmentaire.
  Surapprentissage prouvé (TRAIN seul, pas de val à l'époque).

### 4.2 mini_test (tiny, 3 Mo, val 10 %, parent 360M, 927 steps)
val 2564→144 en 230 steps, train-PPL **55.7** vs val-PPL **60.9**
(ratio 1.09, sain). Export 6.6 Mo, bench +18 Mo OK.

### 4.3 think V1 (base+hidden+seq256) : OOM M4, loss NaN, run tué.
### 4.4 think V2 (tiny, 278 traces 219 Ko, 135M, seq128, mb1/acc16, 8 epochs)
- step20 loss 8.63/CE 9.46/PPL 12796 → step240 loss 2.99/CE 3.33/PPL 27.8.
- Val : 1155 → 570 → 202 → 119.6 (best step200). Final : train-PPL **35.3**
  vs val-PPL **94.1** (ratio 2.67) + alerte >1.5x = **surapprentissage**.
- 420 s, ~1250 tok/s. Export 12.13 Mo, +27.5 Mo, 76 tok/s.
- Qualité (live 9 oct) : `<think>` produit mais charabia en boucle.

### 4.5 tranche00 (tiny, 135M, bs16/acc8, slice_00 107 Mo 10067 docs, val 5 %)
720 steps (interrompu/patience) : val-PPL **63.7**, ~2474 tok/s.
### 4.6 wiki_slice00 (même tranche, 1 epoch, 2124 steps, 7927 s, ~2200 tok/s)
- step1640 loss 2.44/CE 3.97 → step2120 loss 1.92/CE 3.43.
- Val best step2000 : valCE 3.626/PPL 37.6. Final : train-PPL **39.2**
  vs val-PPL **37.3** (ratio 0.95, sain). Export bestval 12.13 Mo,
  +27.4 Mo, 77.8 tok/s. Génération libre incohérente malgré PPL
  (`commune de la commune`, faits inventés).

Tableau `docs/RUNS.md` : tranche00 63.7 / think 94.1 (2.67) / wiki-slice00 37.3 (0.95).

## 5. Échantillons d'inférence (logs)
- base EN : boucles « fréquence/échantillonnage » (60 tok, ~61-62 tok/s).
- base FR pays : « X est un pays … dont la capitale est Y » parfait
  (60 tok, ~60 tok/s) ; wiki libre fragmentaire.
- think : raisonnement/réponse dégénérés en boucle (115 tok, ~103 tok/s).
- `eval/mini_test/samples.md` : mini_test répond « de la » partout (20 prompts).

## 6. Fidélité export
`eval_ppl_mini.log` : PPL `.bin` quantifié **60.6** sur 711k tokens (4101 s)
vs val-PPL 60.9 → **écart 0.5 %** (seuil 5 %). 9862 candidats, 10 exclus
(fuite), 102 docs gardés. Export fidèle.

## 7. Données
- `wiki_fr.jsonl` 1.07 Go / 100208 docs (MIN_LEN 500), 10 slices ~107 Mo.
- `monde_fr.txt` 245 phrases + `monde_missing.txt` (5 sans capitale).
- think : 134 pays + 144 wiki clean (278), + 120 pays2 + 39 retry brutes.
- Ancien mix EN/RFC abandonné (69 % EN).

## 8. Coûts et état
`checkpoints/` 4.9 Go (intermédiaires non purgés : think_e0-7, wiki e0…),
corpus 2.1 Go, Lexar 31 % plein. Aucun run actif. Règles : pas de
GPU/Ollama pendant mesures, pas de train long sans "go", pas de fichier
>1 Mo commité, jamais `--force`.

## 9. Cap (RAPPORT_AMELIORATIONS, 9 oct)
PPL seule insuffisante. Plan : A. éval fixe + fidélité ckpt/bin ;
B. tranche 01 même vocab ; C. 5000+ traces seq256 ; D. anti-répétition ;
E. base après. Critères : RSS<100, val stable sur jamais-vu, répétitions
réduites, majorité questions OK, think complet, tenu après export.
