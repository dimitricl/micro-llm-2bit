# Micro-LLM 2 bits — LLM ternaire sur CPU en moins de 100 Mo

Petit Transformer decoder-only dont les linéaires sont **ternaires 2 bits**
`{-1, 0, +1}` (façon BitNet b1.58), qui tourne sur CPU Apple Silicon dans
**moins de 100 Mo de delta RSS**, entraîné par **distillation** depuis un
parent Hugging Face (SmolLM). Deux tailles : `tiny` (~23M, 9.6 Mo) et
`base` (~73M, 28 Mo).

## Installation (macOS Apple Silicon)

```bash
brew install mise uv
cd micro-llm-2bit
mise trust && mise install   # Python 3.11 épinglé
uv sync                      # venv + dépendances
make                         # noyau C NEON -> build/libmicro2bit.dylib
```

## Usage CLI

```bash
# Distillation parent -> enfant (logits HF en local)
python main.py train --data ./corpus --parent HuggingFaceTB/SmolLM-360M \
    --out checkpoints/enfant.pt --config base --epochs 3

# Avec split validation + early stopping
python main.py train --data ... --val-ratio 0.05 --eval-every 200 \
    --patience 3 --out checkpoints/m.pt

# Pré-calcul des logits parent (mmap + manifeste, refus si incompatible)
python main.py cache-logits --data ... --parent ... --out-dir caches/s0 \
    --batch-size 16
python main.py train --data ... --top-k-cache 1 --cache caches/s0/train \
    --batch-size 16 ...

# Export packé, bench mémoire, inférence
python main.py export --ckpt checkpoints/m.pt --out exports/m.bin
python main.py bench --model exports/m.bin --bench-tokens 64
python main.py infer --model exports/m.bin --prompt "Paris est la capitale" \
    --max-new 60 --temperature 0.7 --top-k 40 --top-p 0.9

# Modèle expérimental entraîné sur des traces de raisonnement
python main.py infer --model exports/enfant_think.bin \
    --prompt "Question : Quelle est la capitale du Japon ?" \
    --think --max-new 120 --temperature 0.2 --top-k 20 --top-p 0.9
```

## Architecture

- **Quantification** : groupes de 64 poids, scale absmean, STE à
  l'entraînement ; activations int8 ; embeddings int8 + tête liée.
- **Inférence** : noyau C NEON (additions/soustractions, GCD), `.bin`
  lu en **memmap zéro-copie**, logits par blocs de 2048 lignes,
  KV cache int8 pré-alloué. Fallbacks : `avx2.c`, `portable.c`, NumPy.
- **Distillation** : `Loss = α·KL·T² + (1-α)·CE` moyennée sur les tokens
  (pas de `batchmean`), garde `0·log(0)=0`, arrêt propre si loss non finie.
- **Données** : split train/val reproductible (seed fixe), vocabulaire
  réduit (12000) reconstruit par tranche, refus si incompatibilité.
- **Mode think** : option `--think` et extraction de
  `<think>...</think>` suivie de `Reponse :`; ce mode dépend d'un modèle
  entraîné sur ce format et ne transforme pas un modèle général en moteur de
  raisonnement.

## Chiffres mesurés (Mac mini M4, 8 oct 2026)

Sources : `main.py bench`, `tools/bench_train.md`, `checkpoints/*.log`.
Les chiffres du modèle think expérimental sont détaillés dans la section
qui lui est consacrée ci-dessous.

| Mesure | tiny | base | Source |
|---|---|---|---|
| Poids packés (.bin) | 9.6 Mo | 28 Mo | bench |
| Budget structurel total / 100 Mo | 10.14 Mo OK | 30.21 Mo OK | bench |
| Delta RSS chargement | +0.2 Mo | +0.5 Mo | bench |
| Delta RSS après génération | +26 Mo | +51 Mo | bench |
| Inférence (greedy) | ~110 tok/s | ~50 tok/s | bench/infer |

Débit d'entraînement mesuré (`tools/bench_train.py`, seq 128, 30 steps) :

| Enfant | Parent | bs 2 | bs 8 | bs 16 |
|---|---|---|---|---|
| tiny | 135M | 555 | 2311 | **2577 tok/s** |
| tiny | 360M | 900 | 1641 | 1748 tok/s |
| base | 135M | 666 | 1239 | 1407 tok/s |
| base | 360M | 503 | 828 | 1086 tok/s |

- Autocast fp16 sur le parent : +26 % (1093 vs 869 tok/s), logits finis.
- Généralisation (mini-run tiny, 3 Mo) : train-PPL 55.7 vs val-PPL 60.9
  (ratio 1.09, sain) — `checkpoints/train_mini.log`.

## Modèle think expérimental

Un premier modèle `tiny` a été entraîné le 8 octobre 2026 sur 278 traces
françaises nettoyées (134 pays + 144 Wikipedia), avec
`HuggingFaceTB/SmolLM-135M` comme parent, 8 epochs maximum, `seq_len=128`,
micro-batch 1 et accumulation 16. Le run a duré 420 s et s'est terminé sans
NaN ni OOM. La dernière mesure donne train-PPL 35.3 contre val-PPL 94.1 :
le surapprentissage est probable.

L'export `exports/enfant_think.bin` fait 12.13 Mo. Le benchmark NEON mesure
un delta RSS de +27.6 Mo après génération et 68.5 tok/s, donc reste sous le
budget mémoire de 100 Mo. En revanche, les deux prompts de contrôle testés
ont produit des sorties répétitives et tronquées avant `</think>`. Le modèle
est donc un prototype du format think, pas encore un modèle de raisonnement
fiable. Il faut augmenter et diversifier les données, puis augmenter le
contexte avant de revendiquer une amélioration qualitative.

## Évaluation

```bash
# 20 prompts FR fixes (greedy + sampling seedée), métriques sans jugement
python tools/eval_fixed.py --model exports/m.bin --out eval/m1/samples.md
# PPL du .bin quantifié sur jeu mis de côté (anti-fuite), vs val-PPL
python tools/eval_ppl.py --model exports/m.bin \
    --heldout slices/slice_09.jsonl --train-docs corpus_train.jsonl \
    --val-ppl 60.9
# Courbe PNG d'un run + tableau des runs
python tools/plot_run.py --log runs/DATE/train.log --run DATE
python tools/runs_table.py  # -> docs/RUNS.md
```

`tools/eval_ppl.py` signale tout écart > 5 % avec la val-PPL comme problème
de fidélité d'export. Les sorties `eval/` ne sont commitées que sous 200 Ko.
Limite moteur : le KV cache est de taille fixe — `generate()` s'arrête à
capacité (préfill + génération bornés par seq_max), sans fenêtre glissante.

## Limites honnêtes

- Modèle minuscule : le motif appris par cœur est parfait (fiches pays),
  mais le français libre reste fragmentaire sans grand corpus (400k chars
  × 40 epochs : CE 0.59 d'entraînement = surapprentissage prouvé).
- La PPL d'entraînement seule ne veut rien dire : toujours regarder la
  val-PPL (`--val-ratio`, `--eval-every`).
- 1 Go de Wikipédia ≈ 250M tokens : à ~750-2500 tok/s selon config, on
  entraîne par tranches de ~100 Mo (≈ 3-10h/epoch), pas en une fois.
- Mac mini peu doté en RAM unifiée : base + hidden + seq 256 fait OOM GPU
  (loss NaN) — réduire (`--batch-size 1 --seq-len 128`, sans hidden).
- **Anomalie connue** : le cache top-k des logits entraîne moins bien que
  le parent en direct (valCE 1.14/1.06 vs 0.80 à steps égaux, k=32/64).
  Cause en cours d'investigation (voir `tools/bench_train.md`).
