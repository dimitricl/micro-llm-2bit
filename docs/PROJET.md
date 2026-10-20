# micro-llm-2bit — dossier projet (2026-10-20)

## 1. C'est quoi

LLM ternaire 2 bits (BitNet b1.58), decoder-only, distillation KL depuis
HuggingFaceTB/SmolLM (135M/360M). Inférence CPU (noyau C NEON + ctypes),
poids packés 4/octet. Repo : dimitricl/micro-llm-2bit (PUBLIC).
Projet : /Volumes/Lexar/micro-llm-2bit.

| Config | V | d | L | Hq/Hkv | ffn | Params latents | .bin | Rôle |
|---|---|---|---|---|---|---|---|---|
| tiny | 8000 | 512 | 8 | 8/4 | 1024 | **25,0 M** | **12,13 Mo** | chaîne E + pilotes QA |
| base | 12000 | 768 | 12 | 8/4 | 1536 | **72,9 M** | **29,25 Mo** (~28 MiB) | non entraînée dans E ; décision ouverte |

Sources : `runs/E-t08/train.log` (25,0 M), `checkpoints/train_base2.log`
(72,9 M), tailles de fichiers ; `checkpoints/bench_base.log` (29,25 Mo,
structurel 30,21 Mo).

Fichiers : quant.py (ternaire/packing), model.py (Transformer+budget),
engine.py (infer memmap zéro-copie, logits par blocs), data.py (corpus,
vocab, split val), distill.py (KL+CE, hidden MSE, cache mmap, early
stopping), main.py (infer/train/export/bench/distill-loop/cache-logits),
tools/ (bench_train, eval_fixed, eval_ppl, eval_probe, plot_run,
runs_table, make_probe, make_qa_corpus), data_fetch/ (wiki, think, slices),
data_clean/make_monde_fr.py.

## 2. Toolchain et recette

mise (Python 3.11) + uv (venv) + make (clang, GCD, .dylib). Entraînement
MPS. Ollama : distill-loop et traces think UNIQUEMENT (jamais pendant les
mesures/runs). Checkpoints lourds : garder 1-2 derniers. Bug KL à ne pas
réintroduire : moyenne sur tokens (pas batchmean) + garde `0*log(0)=0`.

Recette de la chaîne E (figée) : parent SmolLM-135M, KD α 0.7, T 2.0,
lr 3e-4, bs 8/accum 16, seq 128, seed 0, reprise du run précédent,
rewarmup 5 %, val gelée 500 docs, eval-every 50 (t01) puis 100 (t02-t09).
Launchers : `tools/run_E_t01.sh` puis `tools/run_E_tranche.sh NN PREVDIR`.

```bash
uv run python main.py train --data ... --parent ... --out ... --epochs N
uv run python main.py export --ckpt ... --out ....bin
uv run python main.py infer --model ....bin --prompt "..." --max-new 60 \
  --repetition-penalty 1.15 --no-repeat-ngram 3
uv run python main.py cache-logits --data ... --parent ... --out-dir caches/x
```

**Déploiement** : greedy avec `repetition_penalty=1.15` +
`no_repeat_ngram_size=3`, ou sampling T=0,7. Le greedy nu boucle sur 32/32
prompts (§P3) et ne doit pas être servi.

## 3. Corpus

- `/Volumes/Lexar/clean-corpus/wiki_fr.jsonl` : 1,07 Go, 100 208 docs
  (Wikipédia FR, MIN_LEN 500).
- `/Volumes/Lexar/clean-corpus/dedup/tranche_00..09.jsonl` : ~104 Mo chacune
  (68 Mo pour la 09, hash md5 déterministe) ; la chaîne E utilise 9 tranches
  dédupliquées (tranches 01-09 ; tranche00 sert de reprise). Les tranches
  00-08 font ≈ 37,2 M tokens chacune, la 09 24,2 M (§E).
- `data_clean/monde_fr.txt` : 245 phrases pays FR (5 territoires sans
  capitale dans monde_missing.txt, jamais inventés).
- `think_pays.clean.jsonl` 134 traces + `think_wiki.clean.jsonl` 144 traces
  (278 au total), format `<think>...</think>` + `Reponse :`.
- **Pilotes QA** : `data/qa/qa_fr_pilot_v1.jsonl` (60 faits ×19, 7 % de
  général), `v2` (534 faits ×13 variantes ×4, 35 % de général),
  `v3` (534 faits ×40 variantes ×10, 22,09 % de général, 24 838 144 tokens,
  SHA256 `6a1ab5b0…`). Les 100 prompts `par_coeur` de la sonde sont des
  préfixes exacts ; les formulations `reformule` sont interdites à
  l'entraînement (§P4-P7).
- Ancien mix EN/RFC abandonné (69 % anglais → boucles « fréquence »).

## 4. Runs

### Chaîne E (close)

| run | val-PPL gelée | Δc val_big | Δc OOD |
|---|---|---|---|
| t01 | 25,8 | −0,0711 | −0,0557 |
| t05 | 21,0 | −0,1678 | −0,1490 |
| **t08** | **19,8** | **−0,1949** | **−0,1708** |
| t09 | 19,8 | −0,1945 | −0,1695 |

t01 → t08 : la PPL baisse de 25,8 à 19,8 et les gains OOD progressent ; t09
ne gagne plus (même val, Δc légèrement moins bon). **t08 est la référence**
(`runs/E-t08/model.bin`, 12,13 Mo). Détail des 9 tranches :
`docs/RUNS.md` ; protocole : §E.

### Ablations de recette

- **R2** (CE, lr 3e-4, reprise tranche00) : val 22,9, **référence CE**.
- **R3** (KD SmolLM-135M, reprise tranche00) : val 28,0, Δc val_big
  +0,0795, Δc OOD **−0,0795** → **KD validée** en espace SmolLM.
- **R3-Qwen** (KD Qwen3-0.6B, reprise tranche00) : **invalidé** par le
  mismatch tokenizer/vocab (46,45 % de couverture, 0 % d'alignement) ;
  val in-run 15,3 non comparable.
- **R4** (KD Qwen3-0.6B, from-scratch, vocab Qwen) : val 201,0, Δc OOD
  **+0,5724** → **rejeté** (D3).

### Pilotes QA (mémoire factuelle)

| pilote | global | reformulé | val-PPL Δ | « capitain » | verdict |
|---|---|---|---|---|---|
| #1 (60 faits ×19) | 81/200 = 40,5 % | 33/100 = 33,0 % | +16,7 % | 44 | VERT |
| #2 (534 faits ×4) | 21/200 = 10,5 % | 7/100 = 7,0 % | +7,07 % | 82 | MITIGÉ |
| #3 (534 faits ×10) | 72/200 = 36,0 % | 20/100 = 20,0 % | +11,11 % | 59 | PARTIEL |

## 5. Benchmarks mesurés

Entraînement tok/s (seq 128, 30 steps + 5 warmup, `tools/bench_train.md`) :

- tiny/135M : bs 2 555, bs 4 1888, bs 8 2311, bs 16 **2577**.
- tiny/360M : 900 / 1420 / 1641 / 1748.
- base/135M : 666 / 990 / 1239 / 1407.
- base/360M : 503 / 828 / 1008 / 1086.
- Autocast fp16 +26 % (1093 vs 869), logits finis.
- Vocab 12000 : tête < 1 % du step. Cache top-k k32 : valCE 1,14 vs 0,80 en
  direct à steps égaux → parent direct retenu.

Inférence (NEON) :

| mesure | tiny (E-t08) | base |
|---|---|---|
| poids packés | 12,13 Mo | 29,25 Mo |
| structurel (poids+KV+act) | n.d. | 30,21 Mo |
| débit CPU | n.d. | 42,3 tok/s |
| delta RSS | n.d. | +84,7 Mo chargement / +159,3 Mo après génération |

Le critère projet est **delta RSS < 100 Mo**. Le seul bench committed est
`base` (`checkpoints/bench_base.log`) : la marge est tenue au chargement,
pas après génération. **Aucun bench RSS/débit n'est committed pour l'export
E-t08** ; les chiffres tiny 9,6 Mo (~110 tok/s, +26 Mo) sont obsolètes.

## 6. État et règles

### Ce qui est prouvé

- **La KD à l'échelle améliore la PPL puis plafonne** : t01 25,8 → t08 19,8,
  t09 19,8 ; les gains OOD s'arrêtent à t08.
- **La recette QA installe de la mémoire factuelle à 25 M** : 40,5 % (#1)
  puis 36,0 % (#3) de hits globaux, contre 2,0 % pour t08. Les catégories
  non entraînées restent à 0 dans #1 : le signal vient bien de la donnée.
- **Le budget achète des faits, pas la propreté de surface** : #2 → #3 fait
  passer le reformulé de 7 % à 20 % (global 10,5 % → 36 %), mais
  « capitaine » reste à 59 occurrences (contre 44 en #1) et 37/40 générations
  `capitales` reformulées sont corrompues.
- **Le décodeur par défaut est C4** : `repet4g` 0,5057 → 0,0000, diversité
  10,8 → 20,0 mots distincts, boucle max 3 → 2.
- **La sonde factuelle est l'instrument de mesure** : 200 items gelés,
  baseline t08 4/200 (2,0 %), IC95 [0,8 % ; 5,0 %].

### Ce qui reste ouvert

- **Corruption de surface** (« capitaine », « Londres » par défaut) : non
  corrigée par le budget ni par le corpus naturel. C'est le point dur.
- **Unités et dates** : en reformulé, unités 15,0 % et dates 10,0 % (#3) ;
  en global, dates 27,5 % et unités 22,5 %. Les capitales reformulées
  restent à 25,0 % ; unités et dates reformulées restent sous 20 %.
- **Contexte 128** : aucun run au-delà ; KV cache fixe, pas de fenêtre
  glissante.
- **`base` 768** : un ancien entraînement `base` existe
  (`checkpoints/train_base2.log`) et un bench (`bench_base.log`), mais
  **aucun run base avec la recette E/QA** ; sa qualité factuelle n'est pas
  mesurée.
- **Hors domaine FR-wiki** : non évalué.

### Règles

- Pas de GPU/Ollama pendant les mesures et les runs ; un seul job GPU.
- Pas de kill de process d'autrui ; pas de train long sans « go ».
- Pas de fichier > 1 Mo commité (eval < 200 Ko, PNG petits) ; jamais de
  `--force`.
- Tests : `pytest -q` → **171 passed, 1 skipped, 2 deselected** (2026-10-20).

## 7. Décision en attente

Deux options, aucune n'est lancée :

1. **Base 768 from-scratch** — config `base` (d=768, 72,9 M), entraînement
   complet depuis zéro, ~60-100 h (estimation de cadrage, non mesurée).
   C'est le **seul levier de capacité** dans l'enveloppe < 100 Mo/CPU. Il
   attaque la confusion lexicale et la table de capitales, sans garantie sur
   la corruption de surface.
2. **Clôture / livraison t08** — servir `runs/E-t08/model.bin` (12,13 Mo)
   avec le décodeur C4, documenter les limites (4/200 sur la sonde, pas
   d'arithmétique, contexte 128, FR-wiki) et arrêter la chaîne.

La décision appartient au porteur du projet. Tant qu'elle n'est pas prise,
aucun run supplémentaire n'est lancé.
