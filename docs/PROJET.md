# micro-llm-2bit — dossier projet complet (8 oct 2026)

## 1. C'est quoi

LLM ternaire 2 bits (BitNet b1.58), decoder-only, distillation KL depuis
HuggingFaceTB/SmolLM (135M/360M). Inférence CPU (noyau C NEON + ctypes),
poids packés 4/octet. Repo : dimitricl/micro-llm-2bit (PUBLIC).
Projet : /Volumes/Lexar/micro-llm-2bit (ex-interne, session basculée).

| Config | V | d | L | Hq/Hkv | ffn | Params | .bin | Budget |
|---|---|---|---|---|---|---|---|---|
| tiny | 8000 | 512 | 8 | 8/4 | 1024 | ~23M | 9.6 Mo | 10.14 Mo OK |
| base | 12000 | 768 | 12 | 8/4 | 1536 | ~73M | 28 Mo | 30.21 Mo OK |

Fichiers : quant.py (ternaire/packing), model.py (Transformer+budget),
engine.py (infer memmap zéro-copie, logits par blocs), data.py (corpus,
vocab, split val), distill.py (KL+CE, hidden MSE, cache mmap, early
stopping), main.py (infer/train/export/bench/distill-loop/cache-logits),
tools/ (bench_train, eval_fixed, eval_ppl, plot_run, runs_table),
data_fetch/ (wiki, think, slices), data_clean/make_monde_fr.py.

## 2. Toolchain et recette

mise (Python 3.11) + uv (venv) + make (clang, GCD, .dylib). Entraînement
MPS. Ollama : distill-loop et traces think UNIQUEMENT (jamais pendant
les mesures/runs). Checkpoints ~843 Mo (base) : garder 1-2 derniers.
Bug KL à ne pas réintroduire : moyenne sur tokens (pas batchmean) +
garde 0*log(0)=0. Repère step 20 : loss ~8, CE ~7.

```bash
uv run python main.py train --data ... --parent ... --out ... --epochs N
uv run python main.py export --ckpt ... --out ....bin
uv run python main.py bench --model ....bin
uv run python main.py infer --model ....bin --prompt "..." --max-new 60
uv run python main.py cache-logits --data ... --parent ... --out-dir caches/x
```

## 3. Corpus

- /Volumes/Lexar/clean-corpus/wiki_fr.jsonl : 1.07 Go, 100208 docs
  (Wikipedia FR, MIN_LEN 500).
- slices/slice_00..09.jsonl : ~105 Mo chacune (hash md5 déterministe).
- data_clean/monde_fr.txt : 245 phrases pays FR (5 territoires sans
  capitale dans monde_missing.txt, jamais inventés).
- think_pays.clean.jsonl 134 traces, think_wiki.clean.jsonl 144 traces
  retenues après validation stricte du format (278 au total).
- Ancien mix EN/RFC abandonné (69 % anglais -> boucles "fréquence").

## 4. Runs (historique)

- tiny KL 3 epochs : loss ~3.3, infer dégénéré (attendu).
- base EN 12 epochs : CE ~2.5-3.7, français salade (corpus EN).
- base FR 40 epochs (400k chars) : CE 0.59, PPL 1.8 TRAIN SEULE =
  surapprentissage prouvé. Pattern pays parfait, libre fragmentaire.
- mini_test (tiny, 3 Mo, val 10 %) : train-PPL 55.7 vs val-PPL 60.9
  (ratio 1.09, sain). Export 6.6 Mo, bench delta +18 Mo OK.
- tranche00 EN COURS : tiny + 135M + bs16 + fp16 + direct, slice_00
  (107 Mo), 2 epochs max, val 5 %, eval/200, patience 3.
  runs/2026-10-08-tranche00/config.json. (Chiffres à la fin seulement.)
- think tiny TERMINÉ : 278 traces, parent SmolLM-135M, seq 128,
  micro-batch 1, accumulation 16, 2048 steps maximum, 420 s.
  Train-PPL 35.3, val-PPL 94.1 (surapprentissage probable). Export
  12.13 Mo, delta RSS +27.6 Mo, 68.5 tok/s. Les sorties de contrôle sont
  répétitives et parfois tronquées avant `</think>`.
- tranche00 Wikipedia TERMINÉ : tiny, parent SmolLM-135M, 2 124 steps,
  1 epoch en 7 927 s, train-PPL 39.2 et val-PPL 37.3. Export bestval
  12.13 Mo, delta RSS +27.3 Mo, 76.0 tok/s. La génération libre de contrôle
  reste incohérente malgré la PPL.

## 5. Benchmarks mesurés (mini M4 16 Go)

Sources : main.py bench, tools/bench_train.md, logs checkpoints/.

Inférence (NEON) : tiny ~110 tok/s (delta RSS +26 Mo), base ~50 tok/s
(delta +51 Mo). Chargement +0.2/+0.5 Mo (memmap). Critère : delta < 100.

Entraînement tok/s (seq 128, 30 steps + 5 warmup) :
- tiny/135M : bs2 555, bs4 1888, bs8 2311, bs16 2577 (retenu).
- tiny/360M : 900 / 1420 / 1641 / 1748.
- base/135M : 666 / 990 / 1239 / 1407 (< 60 % de tiny -> rejeté).
- base/360M : 503 / 828 / 1008 / 1086.
- Parent = ~60 % du step. Autocast fp16 : +26 % (1093 vs 869), finis.
- Vocab 12000 : tête < 1 % du step (gardé).
- Cache top-k vs direct (même init, 120 steps) : direct valCE 0.80
  (PPL 2.2), k32 1.14 (3.1), k64 1.06 (2.9). Écart valeurs +86 %/+17 %.
  Lecture cache 135x plus rapide mais modèle moins bon -> DIRECT retenu.
  Ancien cache .npz : jamais utilisé (tous runs à --top-k-cache 0),
  abandonné (faux sous mélange des batchs).
- Tranche 100 Mo : ~26M tokens / 2577 tok/s ~= 2.9 h/epoch (mesuré
  tranche00 : ~2400 tok/s réel).

Éval : 20 prompts FR (eval/), PPL .bin sur 2 Mo mis de côté (en cours),
deux prompts think de contrôle testés ; le format n'est pas encore fiable,
courbes docs/img/, tableau docs/RUNS.md.

## 6. État et règles

- En cours : aucune longue tâche ; les runs think et Wikipedia sont terminés
  et exportés.
- 33/33 tests verts. Le mode think est intégré dans `main`.
- Règles : pas de GPU/Ollama pendant mesures et runs ; pas de kill de
  process d'autrui ; pas de train long sans "go" ; pas de fichier > 1 Mo
  commité (eval < 200 Ko, PNG petits) ; jamais de --force.
