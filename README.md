# micro-llm-2bit

Transformer **ternaire** (poids dans {−1, 0, +1}), entraîné avec **MLX** sur
Apple Silicon, exporté dans un format mmap-able (`LLM2`) et exécuté par un
runtime **C portable** (NEON, AVX2, scalaire) sans Python ni MLX.

**Objectif** : un assistant français compact, sous **100 Mo de RAM totale**,
capable de conversation simple, de complétion de code courte et d'un appel
d'outil contraint.

**Non-objectif** : un assistant généraliste fiable à 200 M paramètres. Le
function calling général n'entre pas dans le budget (voir `docs/ROADMAP.md`).

---

## État actuel

Base propre, phase 0 : squelette MLX + runtime C + tests. Aucun modèle entraîné
sur cette branche.

Les audits de l'état antérieur sont dans :

| document | contenu |
|---|---|
| `docs/AUDIT.md` | audit complet, classement GARDER / RÉÉCRIRE / ABANDONNER |
| `docs/BUDGET_100MB.md` | formules et tables de RAM totale |
| `docs/DATA_GAP.md` | inventaire mesuré des données |
| `docs/MLX_MIGRATION.md` | conception de la chaîne MLX + QAT ternaire |
| `docs/ROADMAP.md` | phases, portes, critères go/no-go |

---

## Structure

```
micro_llm/            # entraînement et export (MLX)
  config.py           # configurations et comptage de paramètres
  quant.py            # absmean, STE, packing 2 bits
  model.py            # Transformer ternaire (RMSNorm, GQA, RoPE, SwiGLU)
  data.py             # tokenisation, packing, masque de perte
  train.py            # boucle CE + QAT, validation
  export.py           # checkpoint MLX -> .bin LLM2 v2
  reference.py        # forward NumPy de référence (parité)

runtime/              # inférence C autonome
  include/llm2.h
  src/engine.c
  src/main.c
  src/kernels/{scalar,neon,avx2}.c

tests/                # pytest
data/probe/           # sonde FR gelée (seul jeu contractuel tracké)
```

---

## Démarrage rapide

```bash
# tests (sans entraînement)
.venv/bin/python -m pytest -q

# compilation du runtime
cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime -j

# inférence sur un export existant
./build/runtime/micro-llm --model exports/E-t08.bin --tokens 1 --gen 8 --stats
```

Le test RSS échoue si le runtime dépasse **100 Mo** de RSS après génération.

---

## Format `LLM2`

Header de 44 octets (`<4s10I`) :
`magic, version, vocab, d, L, Hq, Hkv, ffn, seq_max, group, head_dim`.

- `version = 1` : échelles de linéaires en **fp32** (compatibilité E-t08) ;
- `version = 2` : échelles de linéaires en **fp16** (format d'export cible).

Les embeddings sont en int8 avec une échelle fp32 par ligne. Les linéaires sont
packés 2 bits, 4 poids par octet, ordre `0b00 = 0`, `0b01 = +1`, `0b11 = −1`.

---

## Licence des données

Le corpus n'est pas distribué avec ce dépôt. Voir `docs/DATA_GAP.md` pour les
licences (Wikipédia CC BY-SA, Gutenberg domaine public US, code GitHub
hétérogène) et les filtres à appliquer avant toute distribution de poids.
