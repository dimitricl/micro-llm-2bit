# Micro-LLM 2 bits — inférence CPU < 100 Mo + distillation parent → enfant

Petit Transformer decoder-only dont les linéaires sont **ternaires 2 bits**
`{-1, 0, +1}` (façon BitNet b1.58), qui tourne sur CPU Mac dans **moins de
100 Mo** (poids packés + KV cache + activations), entraîné par **distillation**
depuis un parent Hugging Face (ou assisté par Ollama en mode texte).

## Installation (macOS)

```bash
# 1. Outils Apple + Homebrew
xcode-select --install
brew install mise uv ollama

# 2. Projet (Python 3.11 épinglé via mise, venv + deps via uv)
cd micro-llm-2bit
mise trust && mise install
uv sync

# 3. Noyau C (clang Apple, détection arm64/x86_64, sortie .dylib)
make
# Variante portable si besoin : make portable

# 4. (Optionnel) modèle Ollama pour distill-loop
ollama pull smollm2:360m
```

> Clang Apple ne fournit pas OpenMP : le multithread passe par Grand Central
> Dispatch (`dispatch_apply`), sans dépendance. `brew install libomp` reste
> possible pour expérimenter, mais n'est pas requis.

## Usage

```bash
# Entraînement par distillation (texte/jsonl, parent HF en logits)
python main.py train --data ./data --parent HuggingFaceTB/SmolLM-360M \
    --out checkpoints/enfant.pt

# Quantifie + packe en .bin maison (header + poids + scales)
python main.py export --ckpt checkpoints/enfant.pt --out exports/enfant.bin

# Inférence sur vos données (noyau C, KV cache pré-alloué, sampling)
python main.py infer --model exports/enfant.bin --prompt "Explique la photosynthèse"

# Bench : RAM réelle (RSS), tok/s, vérification du budget 100 Mo
python main.py bench --model exports/enfant.bin

# Boucle : l'enfant génère, Ollama corrige, on réentraîne (cycles)
python main.py distill-loop --data ./data --parent HuggingFaceTB/SmolLM-360M \
    --ollama-model smollm2:360m --cycles 2
```

## Quantification 2 bits (ternaire)

Chaque groupe de 64 poids d'entrée partage une **scale absmean**
`scale = mean(|w|)`. Le poids quantifié vaut `clip(round(w/scale), -1, +1)`.
Les **activations** sont en **int8** (`absmax / 127` par token).
À l'entraînement, les poids latents restent fp32 et sont quantifiés au
forward, avec **Straight-Through Estimator** au backward (poids et
activations). Les embeddings sont int8 (1 scale fp32 par token), les normes
fp32, la tête de sortie est **liée** aux embeddings (0 paramètre en plus).

Packing : 4 poids par octet (`00=0, 01=+1, 11=-1`, premier poids en bits
faibles). Le matmul int8 × ternaire n'utilise **que des additions et
soustractions** (zéros ignorés), accumulation entière int32 exacte puis
pondération par `scale_poids × scale_act`. Trois noyaux C : `neon.c` (Apple
Silicon), `avx2.c` (Intel), `portable.c` (secours), + fallback NumPy si la
`.dylib` est absente.

## Distillation parent → enfant

`Loss = α·KL(logits_parent/T ‖ logits_enfant/T)·T² + (1-α)·CE`, avec `α` et
`T` configurables. Le parent est en **inférence seule** (`no_grad`), ses
logits sont restreints au vocabulaire réduit (`kept_ids`) et peuvent être
**pré-calculés en top-k sur disque** (`.npz`). Bonus : `--hidden-weight`
ajoute une MSE entre le dernier caché enfant (via projection apprise) et celui
du parent. Entraînement : AdamW, warmup + cosine decay, gradient clipping,
checkpoints reprenables, logs (loss, perplexité, tok/s), accumulation de
gradient (la RAM Mac est unifiée : micro-batch 2 × accum 8 par défaut).
Fonctionne sur **MPS** (avec fallback CPU auto) et CPU seul.

**Rôle d'Ollama** : Ollama n'expose pas les logits, il ne peut donc pas servir
à la distillation KL (qui exige un parent HF). Il sert dans `distill-loop` à
générer des données et à **corriger les réponses** de l'enfant en texte.

## Vocabulaire réduit (choix documenté)

Le parent fait 50–150k tokens : à 512 dims, 150k × 512 int8 = 77 Mo à lui
seul. On réduit à **8000** (tiny) : spéciaux conservés, puis tokens les plus
fréquents du corpus, le reste rabattu sur `unk`. Fichier `*.vocab.json`
(`kept_ids`, `unk`, `eos`) sauvegardé au `train`, réutilisé par
`infer`/`export`. Élargir le vocab rapproche du budget : chaque +1000 tokens
coûte ~0.5 Mo.

## Budget mémoire (tiny, exact)

| Poste | Calcul | Taille |
|---|---|---|
| Linéaires packés | 18 874 368 / 4 | 4.72 Mo |
| Scales fp32 | 294 912 × 4 | 1.18 Mo |
| Embeddings int8 + scales | 4 096 000 + 32 000 | 4.13 Mo |
| Normes fp32 | 8 704 × 4 | 0.03 Mo |
| KV cache int8 (2·8·2048·256) | 8 388 608 | 8.39 Mo |
| Activations (pire cas fp32) | — | ~0.04 Mo |
| **Total** | | **~18.5 Mo < 100 Mo** |

`python main.py bench` recalcule ce budget depuis le `.bin` et mesure la RSS
réelle (`psutil`, sinon `resource.getrusage` — octets sur macOS, Ko sur Linux).

## Limites honnêtes

- Modèle minuscule (~23M équivalents) + vocab 8k : français correct sur des
  textes simples, pas de raisonnement poussé, hallucinations possibles.
- Mots hors top-8k → `unk` : les textes techniques/specialisés se dégradent.
- Distillation KL exige le parent HF en local (~1 Go MPS pour 360M) ;
  `distill-loop` Ollama seul n'apprend que du texte (pas de KL).
- Noyau C : matvec optimisé, le reste (normes, softmax, attention) en NumPy.
- Entraînement QAT complet sur CPU seul : lent ; préférez MPS (Apple Silicon).
