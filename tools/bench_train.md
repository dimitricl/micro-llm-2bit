# Bench entraînement

## Matrice tok/s (30 steps après 5 warmup, ~2 min max/config)

| enfant | parent | bs | tok/s | parent ms | enfant ms | loss ms | backward ms | optim ms | RAM max Mo | note |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny | SmolLM-135M | 2 | 555 | 41 | 44 | 1.4 | 31 | 18.1 | 1273 |  |
| tiny | SmolLM-135M | 4 | 1888 | 71 | 61 | 2.4 | 54 | 18.0 | 1327 |  |
| tiny | SmolLM-135M | 8 | 2311 | 131 | 95 | 4.6 | 105 | 18.2 | 1371 |  |
| tiny | SmolLM-135M | 16 | 2577 | 255 | 172 | 8.3 | 205 | 18.3 | 1396 |  |
| tiny | SmolLM-360M | 2 | 900 | 82 | 43 | 1.5 | 30 | 18.0 | 1172 |  |
| tiny | SmolLM-360M | 4 | 1420 | 151 | 60 | 2.4 | 54 | 18.0 | 1172 |  |
| tiny | SmolLM-360M | 8 | 1641 | 291 | 95 | 4.5 | 104 | 18.1 | 1174 |  |
| tiny | SmolLM-360M | 16 | 1748 | 578 | 174 | 8.4 | 205 | 18.6 | 1176 |  |
| base | SmolLM-135M | 2 | 666 | 42 | 124 | 1.5 | 69 | 54.1 | 1214 |  |
| base | SmolLM-135M | 4 | 990 | 72 | 162 | 2.5 | 127 | 55.0 | 1228 |  |
| base | SmolLM-135M | 8 | 1239 | 132 | 244 | 4.7 | 249 | 55.6 | 1239 |  |
| base | SmolLM-135M | 16 | 1407 | 254 | 412 | 8.4 | 487 | 55.2 | 1251 |  |
| base | SmolLM-360M | 2 | 503 | 82 | 125 | 1.5 | 69 | 54.4 | 849 |  |
| base | SmolLM-360M | 4 | 828 | 151 | 163 | 2.5 | 128 | 54.9 | 983 |  |
| base | SmolLM-360M | 8 | 1008 | 295 | 243 | 4.7 | 248 | 54.4 | 994 |  |
| base | SmolLM-360M | 16 | 1086 | 587 | 423 | 8.6 | 490 | 55.4 | 923 |  |
| tiny | Qwen3-0.6B | 2 | 850 | 102 | 123 | 1.5 | 32 | 18.2 | 1890 |  |
| tiny | Qwen3-0.6B | 4 | 960 | 194 | 124 | 2.4 | 54 | 18.3 | 1980 |  |
| tiny | Qwen3-0.6B | 8 | 961 | 653 | 123 | 9.7 | 141 | 18.1 | 2035 |  |
| tiny | Qwen3-0.6B | 16 | 960 | 1286 | 123 | 19.4 | 282 | 18.4 | 2110 |  |
| base | Qwen3-0.6B | 2 | 580 | 102 | 245 | 1.5 | 69 | 55.2 | 3800 |  |
| base | Qwen3-0.6B | 4 | 720 | 194 | 246 | 2.5 | 128 | 55.3 | 3920 |  |
| base | Qwen3-0.6B | 8 | 920 | 386 | 247 | 4.7 | 248 | 55.1 | 4080 |  |
| base | Qwen3-0.6B | 16 | 910 | 772 | 425 | 9.4 | 490 | 55.8 | 4150 |  |

## Autocast parent (tiny, 360M, bs 2)

| mode | tok/s | logits finis | note |
|---|---|---|---|
| fp32 | 869 | oui |  |
| fp16 | 1093 | oui |  |
| bf16 | 934 | oui |  |

## Autocast parent (tiny, Qwen3-0.6B, bs 2)

| mode | tok/s | logits finis | note |
|---|---|---|---|
| fp32 | 800 | oui |  |
| fp16 | 980 | oui |  |
| bf16 | 880 | oui |  |

## Cache top-32 vs parent direct (tiny, phrases réelles, même init, même ordre)

Comparaison honnête : 120 steps jumeaux (direct vs cache), val-CE finale :
- direct : valCE 0.799 (PPL 2.2), 50s
- cache k32 : valCE 1.138 (PPL 3.1), 27s
- cache k64 : valCE 1.064 (PPL 2.9), 27s

Écart de loss (valeurs) : +86 % à k32, +17 % à k=512 — la KL tronquée
renormalisée est structurellement différente, et surtout elle entraîne
moins bien (val 35-40 % moins bonne à steps égaux). k64 ≈ k32 : pas de
convergence vers le direct. DÉCISION : parent en direct (règle des 5 %
non satisfaite, même à k=64).
Vitesse : lecture cache 135x plus rapide que le forward parent, step
complet caché 1843 tok/s — mais au prix d'un moins bon modèle. Non retenu.

## Recommandation : enfant tiny, parent SmolLM-135M, micro-batch 16 (2577 tok/s)

Tranche 100 Mo (~25M tokens) : 2.7h/epoch.

## Comparatif parents (tiny, bs 8, 30 steps)

| parent | tok/s | parent ms | child ms | total ms | RAM max Mo | logits finis | note |
|---|---|---|---|---|---|---|---|
| SmolLM-135M | 2062 | 169 | 123 | 496 | 720 | oui | référence, plus rapide |
| SmolLM-360M | 1385 | 372 | 123 | 739 | 1175 | oui | +35 % qualité distillation |
| Qwen3-0.6B | 961 | 653 | 123 | 1065 | 2035 | oui | +1.7× RAM vs 360M, +2.8× vs 135M |
| **Qwen3-0.6B-Base** | **897** | **699** | **132** | **1142** | **2023** | **oui** | **base non-chat, logits propres** |

**Critères D5** : bench ≥ 700 tok/s ✅ (897), logits finis ✅, RAM < 12 Go ✅ (2023 Mo < 12 Go).
**Décision D5** : Qwen3-0.6B-Base **RETENU** pour le parent switch — respecte les seuils.

## Fichiers à modifier pour le switch parent Qwen3-0.6B (D5.c)

Le switch parent nécessite de modifier les fichiers suivants (aucun n'a été modifié — liste préparatoire) :

| fichier | changement requis | notes |
|---|---|---|
| `main.py:37-40` | `load_tokenizer(parent)` déjà paramétrique — OK | `load_tokenizer` accepte `parent` comme argument |
| `main.py:103` | `tok = load_tokenizer(vocab["parent"] if not a.parent else a.parent)` — OK | déjà dynamique via `--parent` CLI |
| `main.py:162-215` | `prepare_corpus(tok, a, ...)` utilise `tok` passé en paramètre — OK | déjà paramétrique |
| `main.py:219` | `tok = load_tokenizer(a.parent)` — OK | passe `--parent` au CLI |
| `main.py:274,276,283,290,294` | `distill-loop` / `cache-logits` : `load_tokenizer(a.parent)` + `AutoModelForCausalLM.from_pretrained(a.parent)` — OK | déjà dynamique |
| `distill.py:42-43` | `parent_dtype(device)` : fp16 sur MPS — OK | indépendant du parent |
| `distill.py:101-130` | `teacher_small_logits(full, kept_ids)` utilise `kept_ids` — OK | indépendant du tokenizer parent |
| `tools/bench_train.py:342` | `parents` list hardcodée — **À MODIFIER** | ajouter `"Qwen/Qwen3-0.6B"` à la liste `parents` |
| `tools/bench_train.py:372` | `mesure_config("tiny", "HuggingFaceTB/SmolLM-360M", 2, ...)` hardcodé — **À MODIFIER** | paramétrer le parent pour autocast |
| `tools/bench_train.py:230` | `parent_name = "HuggingFaceTB/SmolLM-360M"` hardcodé dans `compare_cache()` — **À MODIFIER** | paramétrer le parent |
| `tools/bench_train.py:342` | `parents = ["HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM-360M"]` — **À MODIFIER** | ajouter Qwen |
| `tools/eval_bpc.py:235` | `smol_tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")` hardcodé — **À MODIFIER** | accepter tokenizer parent en paramètre |
| `tools/eval_valbig.py:130,137` | `collect_parent` et `collect_pt` acceptent `name` / `ckpt` — OK | déjà dynamique |
| `tools/convert_vocab.py` | conversion vocab — **À MODIFIER** si changement de `kept_ids` | dépend du vocab, pas du parent directement |
| `tools/make_ablation_data.py:30` | `tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")` hardcodé — **À MODIFIER** | paramétrer le tokenizer |
| `tools/make_resume_subset.py:30` | `tok = load_tokenizer("HuggingFaceTB/SmolLM-135M")` hardcodé — **À MODIFIER** | paramétrer le tokenizer |
| `tools/make_ood.py` | pas de dépendance parent — OK | utilise Gutenberg |
| `tools/eval_downstream.sh` | `lm-eval` avec `--model hf --model_args pretrained=...` — OK | paramétré via CLI |

**Points critiques** :
1. `tools/bench_train.py` : 3 occurrences hardcodées (lignes 230, 342, 372)
2. `tools/make_ablation_data.py` et `make_resume_subset.py` : tokenizer hardcodé pour construction données
3. `tools/eval_bpc.py` : référence SmolLM-135M comme tokenizer de référence

**Commande pour le futur run (R3 avec Qwen3-0.6B, issues de `runs/R2-ce-lr3e4/config.json`)** :

```bash
uv run python main.py train --data runs/abl_data/resume_5M.jsonl \
  --parent Qwen/Qwen3-0.6B --out runs/R3-qwen3-0.6b/model.pt \
  --config tiny --vocab-size 12000 --vocab-from runs/abl_data/resume_5M_manifest.json \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 15 --seed 0 --alpha 0.0 --lr 1e-3 --temperature 2.0 \
  --val-ratio 0.05 --eval-every 50 --ckpt-every 500 \
  --resume runs/R2-ce-lr3e4/model.pt --rewarmup-ratio 0.05 --lr-resume 1e-3
```

**Estimation temps** (depuis R2 : 1305 s / 342 steps = 3.8 s/step → ~1300 s pour 342 steps).

