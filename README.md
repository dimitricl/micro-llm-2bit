# Micro-LLM 2 bits — LLM ternaire sur CPU en moins de 100 Mo

Petit Transformer decoder-only dont les linéaires sont **ternaires 2 bits**
`{-1, 0, +1}` (façon BitNet b1.58), qui tourne sur CPU Apple Silicon dans
**moins de 100 Mo de delta RSS**, entraîné par **distillation** depuis un
parent Hugging Face (SmolLM). Deux tailles : `tiny` (**25,0 M params latents**,
`.bin` **12,13 Mo**) et `base` (**72,9 M**, `.bin` **29,25 Mo**, soit ~28 MiB).
La chaîne de référence actuelle est `tiny` ; `base` n'a pas été entraînée dans
la chaîne E.

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
python main.py train --data ./corpus --parent HuggingFaceTB/SmolLM-135M \
    --out checkpoints/enfant.pt --config tiny --epochs 3

# Avec split validation + early stopping
python main.py train --data ... --val-ratio 0.05 --eval-every 200 \
    --patience 3 --out checkpoints/m.pt

# Pré-calcul des logits parent (mmap + manifeste, refus si incompatible)
# NB : --save-batch doit égaler --batch-size (B1 refuse bs > save).
python main.py cache-logits --data ... --parent ... --out-dir caches/s0 \
    --batch-size 16 --save-batch 16
python main.py train --data ... --top-k-cache 1 --cache caches/s0/train \
    --batch-size 16 --save-batch 16 ...

# Export packé, bench mémoire, inférence
python main.py export --ckpt runs/E-t08/model.pt --out runs/E-t08/model.bin
python main.py bench --model runs/E-t08/model.bin
python main.py infer --model runs/E-t08/model.bin \
    --prompt "La capitale de l'Italie est" --max-new 60 \
    --repetition-penalty 1.15 --no-repeat-ngram 3

# Sampling (référence historique, T=0,7)
python main.py infer --model runs/E-t08/model.bin \
    --prompt "La capitale de l'Italie est" --max-new 60 --temperature 0.7
```

**Défaut de déploiement obligatoire** : en **greedy**, servir avec
`--repetition-penalty 1.15 --no-repeat-ngram 3` (ou en sampling T=0,7).
Le greedy « nu » (1,0 / 0) boucle sur 32/32 prompts de contrôle : il est
**interdit au déploiement**. Détail et balayage C0-C5 : `docs/ABLATIONS.md`
§P3.

## État actuel (2026-10-20)

- **Chaîne E close** (t01 → t09, 15-17 oct 2026) : KD SmolLM-135M, 9 tranches
  Wikipédia, val gelée de 500 docs, un seul job GPU à la fois. Le meilleur
  checkpoint est **`runs/E-t08`** : val-PPL gelée **19,8**, **Δc val_big
  −0,1949**, **Δc OOD −0,1708**. `t09` a la même val (19,8) mais un Δc
  légèrement moins bon (−0,1945 / −0,1695) : **t08 reste la référence**.
  Export livré : `runs/E-t08/model.bin`, **12,13 Mo**.
- **La PPL plafonne** : 25,8 (t01) → 19,8 (t08), puis 19,8 (t09). Les gains
  OOD s'arrêtent aussi à t08. Détail des 9 tranches : `docs/RUNS.md`.
- **Sonde factuelle** : `data/probe/probe_fr_v1.jsonl` (200 items,
  4 catégories × 2 slices, **gelée**, SHA256 `d9817afb…`), harnais
  `tools/eval_probe.py`. Baseline t08 : **4/200 = 2,0 %** (IC95
  [0,8 % ; 5,0 %]). Le modèle ne répond pas aux questions ; il continue du
  corpus. Voir §P4.
- **Pilotes QA** (mémoire factuelle sur 25 M) : #1 **81/200** (reformulé
  33/100) avec ×19 et 60 faits ; #2 **21/200** (7/100) avec ×4 et 534 faits ;
  #3 **72/200** (20/100) avec ×10, 24,8 M tokens et 534 faits. Verdict #3 :
  **PARTIEL** — les faits montent, la corruption de surface (« capitaine »)
  ne recule pas (59 occurrences vs 44 en #1). Voir §P5-P7.
- **Décision en attente** : lancer **base 768 from-scratch** (seul levier de
  capacité dans l'enveloppe < 100 Mo) ou **clôturer/livrer t08**. Aucun run
  n'est lancé dans l'état actuel.

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

## Chiffres mesurés

| Mesure | tiny (E-t08) | base | Source |
|---|---|---|---|
| Params latents | **25,0 M** | **72,9 M** | `runs/E-t08/train.log`, `checkpoints/train_base2.log` |
| Poids packés (.bin) | **12,13 Mo** | **29,25 Mo** | fichiers ; `checkpoints/bench_base.log` |
| Structurel (poids + KV + activations) | n.d. | 30,21 Mo | `checkpoints/bench_base.log` |
| Débit CPU | n.d. | 42,3 tok/s | `checkpoints/bench_base.log` |
| Delta RSS | n.d. | +84,7 Mo au chargement, +159,3 Mo après génération | `checkpoints/bench_base.log` |

- Le critère projet reste **delta RSS < 100 Mo**. Le dernier bench committed
  porte sur `base` : +84,7 Mo au chargement mais **+159,3 Mo après génération
  depuis un process à vide** → la marge n'est pas démontrée pour `base`.
  **Aucun bench RSS/débit n'est committed pour l'export E-t08** ; les anciens
  chiffres « tiny 9,6 Mo / ~110 tok/s / +26 Mo » ne s'appliquent plus.
- Entraînement (seq 128, 30 steps + 5 warmup, `tools/bench_train.md`) :

| Enfant | Parent | bs 2 | bs 8 | bs 16 |
|---|---|---|---|---|
| tiny | 135M | 555 | 2311 | **2577 tok/s** |
| tiny | 360M | 900 | 1641 | 1748 tok/s |
| base | 135M | 666 | 1239 | 1407 tok/s |
| base | 360M | 503 | 828 | 1086 tok/s |

- Autocast fp16 sur le parent : +26 % (1093 vs 869 tok/s), logits finis.
- Généralisation (mini-run tiny, 3 Mo) : train-PPL 55,7 vs val-PPL 60,9
  (ratio 1,09, sain) — `checkpoints/train_mini.log`.

## Modèle think expérimental (8 oct 2026)

Un premier modèle `tiny` a été entraîné le 8 octobre 2026 sur 278 traces
françaises nettoyées (134 pays + 144 Wikipedia), avec
`HuggingFaceTB/SmolLM-135M` comme parent, 8 epochs maximum, `seq_len=128`,
micro-batch 1 et accumulation 16. Le run a duré 420 s et s'est terminé sans
NaN ni OOM. La dernière mesure donne train-PPL 35,3 contre val-PPL 94,1 :
le surapprentissage est probable.

L'export `exports/enfant_think.bin` fait 12,13 Mo. En revanche, les deux
prompts de contrôle testés ont produit des sorties répétitives et tronquées
avant `</think>`. Le modèle est donc un prototype du format think, pas un
moteur de raisonnement fiable.

## Évaluation

```bash
# 32 prompts FR fixes (greedy + sampling seedée), métriques sans jugement
python tools/eval_fixed.py --model runs/E-t08/model.bin \
    --out eval/Et08/samples.md --metrics-out eval/Et08/metrics.json \
    --repetition-penalty 1.15 --no-repeat-ngram 3

# Sonde factuelle gelée (200 items, hash vérifié à chaque exécution)
python tools/eval_probe.py --model runs/E-t08/model.bin \
    --out eval/Et08-probe/samples.md \
    --results-out eval/Et08-probe/results.json

# PPL du .bin quantifié sur jeu mis de côté (anti-fuite), vs val-PPL
python tools/eval_ppl.py --model runs/E-t08/model.bin \
    --heldout runs/abl_data/val_E_frozen.jsonl \
    --train-docs runs/abl_data/train_5M.jsonl --val-ppl 19.8

# Courbe PNG d'un run + tableau des runs
python tools/plot_run.py --log runs/E-t08/train.log --run E-t08
python tools/runs_table.py  # -> tableau brut des runs
```

`tools/eval_probe.py` applique la règle exacte de §P4 : normalisation NFD,
alias avec frontières de tokens, EOS-collapse < 5 tokens. La sonde est
**gelée** : un hash divergent provoque `SystemExit(2)`. Les sorties `eval/`
ne sont commitées que sous 200 Ko. Limite moteur : le KV cache est de taille
fixe — `generate()` s'arrête à capacité (préfill + génération bornés par
seq_max), sans fenêtre glissante.

## Tests

```bash
pytest -q                  # suite rapide par défaut (slow désélectionnés)
pytest -m slow -q          # test BPC à valeurs fixes (5 modèles, ~10 min, GPU)
pytest -q -m "not slow" --durations=20   # diagnostic lenteurs
```

État au 2026-10-20 : **171 passed, 1 skipped, 2 deselected**.
Marqueur `slow` déclaré dans `pyproject.toml` (`addopts = -m 'not slow'`,
`timeout = 120` via pytest-timeout).

## Limites honnêtes

- **Pas d'arithmétique** : aucune compétence de calcul n'est entraînée ni
  mesurée ; la sonde ne contient que des faits ponctuels (capitales, dates,
  définitions, unités).
- **Pas de suivi d'instruction** : sur 200 questions, t08 ne contient le
  fait attendu que 4 fois (2,0 %) ; les 196 autres réponses ne le contiennent
  pas. Le modèle ne « comprend » pas qu'on lui demande un fait.
- **Morphologie et surface fragiles** : « la capitaine est Rome » pour
  « la capitale est Rome », « Londres » comme capitale par défaut. Les
  pilotes QA montrent que le budget achète des faits, **pas** la propreté de
  la trame (§P7).
- **Contexte 128 maximum** : tous les runs E et QA sont en `seq_len=128` ;
  le KV cache est de taille fixe, sans fenêtre glissante.
- **Domaine FR-wiki** : entraînement sur Wikipédia FR dédupliqué + 245
  phrases pays. Hors de ce domaine, la PPL ne garantit rien.
- **La PPL seule ne veut rien dire** : toujours regarder la val gelée et la
  sonde factuelle. Le train-PPL très bas des pilotes QA (~1,7 à ~2,7 selon
  le run) cache un surapprentissage massif.
- **Base 768 from-scratch non lancée** : la décision base 768 / clôture est
  ouverte ; aucune mesure de qualité ni de RSS n'existe pour ce futur run.
- **Débit/RSS de l'export E-t08 non mesurés** : ne pas réutiliser les
  chiffres des exports 9,6 Mo.

## Historique phases C/D (9 oct 2026)

- Diagnostics (`docs/DIAGNOSTICS.md`) : BPC 135M 1,5357 / tranche00 1,5676,
  vocab 12000 à 1,0 % unk, débit 2 221 tok/s effectifs.
- Ablations (`docs/ABLATIONS.md`, 9 runs × 5M tokens, bruit ±9 PPL) :
  CE seule + lr 1e-3 gagne (V6 : 55,2 ; Δ −74,4 vs V1a, −17,2 vs V3),
  seq 128 gardée. La chaîne E a ensuite tranché la recette KD
  (SmolLM-135M) et la décision D3 (KD validée en espace SmolLM).
