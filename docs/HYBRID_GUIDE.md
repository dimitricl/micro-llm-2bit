# Guide modèle hybride conversation + code

## Voie MLX révisée (2026-10-26)

Le pilote PyTorch a été arrêté au step 40/305. La voie `train_hybrid_mlx.py`
entraîne maintenant le modèle hybride 640/10 couches en flux depuis des
JSONL, construit un cache tokenisé sur disque, sauvegarde des checkpoints et
peut reprendre leurs poids. Cette voie entraîne en CE depuis zéro : elle
n'utilise pas SmolLM comme professeur.

L'export MLX ajouté fait une quantification **int4 après entraînement** et
fournit une inférence/mesure MLX. C'est une voie de déploiement sous 100 Mo à
mesurer ; elle ne produit pas les poids ternaires 2 bits de la chaîne
PyTorch. Le choix MLX/int4 favorise le débit, avec une quantification
différente du but historique du dépôt.

La chaîne séquentielle complète (préparation SFT, pré-entraînement sur les
corpus disponibles, SFT, export int4 et benchmark) est dans
`tools/run_hybrid_mlx_full.sh`. Le jeu SFT local vient d'être préparé : 1 583
exemples conservés, dont 56 exemples conversation/instruction et 1 527
complétions JavaScript. Après avoir vérifié que l'ancien run est arrêté,
lance-la ainsi :

```bash
cd /Volumes/Lexar/micro-llm-2bit
mkdir -p logs
nohup caffeinate -dims bash tools/run_hybrid_mlx_full.sh \
  > logs/hybrid-full.nohup 2>&1 < /dev/null &
echo $! > logs/hybrid-full.pid
```

Elle refuse d'écraser des sorties finales existantes. Les journaux de chaque
étape apparaissent dans `logs/hybrid-full.nohup`.

Pré-entraînement sur les dix tranches dédupliquées (≈1 Go) et les données
hybrides locales :

```bash
cd /Volumes/Lexar/micro-llm-2bit
mkdir -p logs runs/hybrid-mlx
nohup caffeinate -dims .venv/bin/python -u tools/train_hybrid_mlx.py \
  --data /Volumes/Lexar/clean-corpus/dedup data/hybrid \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-mlx \
  --batch-size 16 --seq-len 512 --epochs 1 --lr 3e-4 \
  --ckpt-every 500 --log-every 50 \
  > logs/hybrid-mlx.nohup 2>&1 < /dev/null &
echo $! > logs/hybrid-mlx.pid
```

Le premier lancement tokenise en deux passes et crée
`runs/hybrid-mlx/tokens.uint16.bin`. Les lancements suivants réutilisent ce
cache si les fichiers sources et le tokenizer n'ont pas changé. Les
checkpoints contiennent les poids ; une reprise réinitialise AdamW. Exemple :

```bash
.venv/bin/python tools/train_hybrid_mlx.py \
  --data /Volumes/Lexar/clean-corpus/dedup data/hybrid \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-mlx \
  --batch-size 16 --seq-len 512 --epochs 1 --lr 3e-4 \
  --resume runs/hybrid-mlx/checkpoint_0000500.safetensors \
  --resume-step 500
```

Après un entraînement complet, exporter et mesurer l'int4 :

```bash
.venv/bin/python tools/export_hybrid_mlx.py export \
  --checkpoint runs/hybrid-mlx/final.safetensors \
  --bits 4 --out exports/hybrid_mlx_int4.safetensors
.venv/bin/python tools/export_hybrid_mlx.py bench \
  --model exports/hybrid_mlx_int4.safetensors --bits 4
```

Le fine-tuning chat/code reste à préparer : le dépôt n'a que 100 réponses
synthétiques et dix dialogues de contrôle. Le préparateur SFT forme désormais
des cibles réponse uniquement à partir de ces dialogues, des réponses
synthétiques filtrées et de préfixes de code JavaScript. Il ajoute ensuite
10 % de replay du corpus de langue durant le SFT pour limiter l'oubli.

Préparer les exemples, puis lancer le SFT après la fin du pré-entraînement :

```bash
.venv/bin/python tools/prepare_hybrid_sft_mlx.py \
  --tokenizer models/tokenizer_fr_16k \
  --chat data/chat/chat_pilot_v1.jsonl \
  --synthetic data/hybrid/synthetic.jsonl \
  --code data/hybrid/javascript_code.jsonl \
  --seq-len 512 --out runs/hybrid-sft/sft_packed.npz

mkdir -p runs/hybrid-sft
nohup caffeinate -dims .venv/bin/python -u tools/train_hybrid_mlx.py \
  --stage sft \
  --data /Volumes/Lexar/clean-corpus/dedup data/hybrid \
  --sft-data runs/hybrid-sft/sft_packed.npz \
  --token-cache runs/hybrid-mlx/tokens.uint16.bin \
  --tokenizer models/tokenizer_fr_16k \
  --output runs/hybrid-sft \
  --resume runs/hybrid-mlx/final.safetensors \
  --batch-size 16 --seq-len 512 --epochs 3 --lr 5e-5 \
  --weight-decay 0.01 --replay-ratio 0.1 \
  --ckpt-every 100 --log-every 20 \
  > logs/hybrid-sft.nohup 2>&1 < /dev/null &
echo $! > logs/hybrid-sft.pid
```

Le fichier SFT et son manifeste affichent le nombre d'exemples conservés
après filtrage. Ce premier jeu reste petit ; il sert à valider la chaîne et
le comportement de base, pas à établir une compétence générale en chat.

L'export final utilise `runs/hybrid-sft/final.safetensors` à la place du
checkpoint de pré-entraînement montré ci-dessus.

## Vue d'ensemble

Ce guide décrit comment créer un LLM ultra-léger (<100 Mo) capable de :
- **Conversation** en français (questions/réponses, explications)
- **Complétion de code** (Python, JavaScript, TypeScript)
- **Détection automatique** du mode (code vs chat)

L'architecture repose sur votre base existante ternaire 2-bit avec une nouvelle configuration `hybrid`.

## Configuration hybride

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| d_model | 640 | Entre tiny (512) et base (768) |
| n_layers | 10 | Compromis capacité/vitesse |
| vocab_size | 16000 | Supporte tokens FR + code |
| seq_max | 512 | Conversations plus longues |
| ffn_dim | 1280 | Capacité hybride |
| Params estimés | ~47M | L'export int4 doit être mesuré pour confirmer la taille |

## Workflow complet

### Étape 1 : Télécharger les données

```bash
# Installer la dépendance
pip install datasets

# Télécharger datasets publics (conversation FR + code)
python tools/download_hybrid_data.py \
    --output-dir data/hybrid \
    --max-samples 50000
```

Cela télécharge :
- OASST1 (dialogues FR)
- no_robots (instruction-following FR)
- The Stack (code Python, JS, TS)
- CodeParrot (code Python)
- OpenOrca (instructions multilingues)

### Étape 2 : Générer des données synthétiques

```bash
# Générer des données hybrides avec qwen2.5-coder:1.5b
python tools/generate_hybrid_synthetic.py \
    --output data/hybrid/synthetic.jsonl \
    --num-samples 10000 \
    --model qwen2.5-coder:1.5b \
    --code-ratio 0.5
```

### Étape 3 : Pré-entraînement langage mixte

```bash
# Entraîner sur données mixtes (50% code + 50% conversation)
python tools/train_hybrid.py \
    --stage pretrain \
    --data-dir data/hybrid \
    --output runs/hybrid-pretrain \
    --config hybrid
```

Durée estimée : ~24-36h sur M4 avec ~100-200M tokens.

### Étape 4 : Fine-tuning instruction-following

```bash
# Fine-tuner sur format chat <user>...</user><assistant>...</assistant>
python tools/train_hybrid.py \
    --stage instruct \
    --checkpoint runs/hybrid-pretrain/model.pt \
    --data-dir data/hybrid \
    --output runs/hybrid-instruct \
    --config hybrid
```

Durée estimée : ~12-18h sur M4 avec ~30-50M tokens.

### Étape 5 : Fine-tuning QA-passage (optionnel)

```bash
# Améliorer la factualité avec votre technique G1 validée
python tools/train_hybrid.py \
    --stage qa \
    --checkpoint runs/hybrid-instruct/model.pt \
    --qa-data data/qa/g_passage_v1.jsonl \
    --output runs/hybrid-final \
    --config hybrid
```

Durée estimée : ~6-8h sur M4.

### Étape 6 : Export et inférence

```bash
# Exporter le modèle
python main.py export \
    --ckpt runs/hybrid-final/model.pt \
    --out runs/hybrid-final/model.bin

# Inférence avec détection automatique
python tools/infer_hybrid.py \
    --model runs/hybrid-final/model.bin \
    --prompt "comment trier une liste en Python ?" \
    --mode auto

# Mode interactif
python tools/infer_hybrid.py \
    --model runs/hybrid-final/model.bin \
    --interactive
```

## Détection automatique du mode

L'inférence hybride détecte automatiquement si l'input est du code ou du chat :

**Indicateurs de code :**
- Mots-clés : `function`, `def`, `class`, `import`, `return`, etc.
- Patterns de code : définitions de fonctions, imports, commentaires
- Contexte : mentions de langages de programmation

**Indicateurs de chat :**
- Mots-clés : `bonjour`, `explique`, `qu'est-ce que`, `comment`, etc.
- Questions générales, demandes d'opinion
- Histoires, exemples, définitions

**Paramètres par mode :**

| Mode | Température | Top-K | Repetition Penalty |
|------|-------------|-------|-------------------|
| code | 0.2 | 10 | 1.0 |
| chat | 0.7 | 40 | 1.15 |

## Optimisations possibles

### Si résultats insuffisants

1. **Augmenter la capacité**
   - Passer à config `base` (73M params)
   - Augmenter vocab_size à 20k

2. **Plus de données**
   - Augmenter `--max-samples` dans download_hybrid_data.py
   - Générer plus de données synthétiques
   - Ajouter d'autres datasets (The Pile, etc.)

3. **Recette d'entraînement**
   - Ajuster learning rate
   - Augmenter le nombre d'époques
   - Utiliser la distillation avec un parent plus gros

### Si trop lent

1. **Réduire seq_max**
   - Passer de 512 à 256 tokens

2. **Quantification**
   - Utiliser int4 au lieu de int8 (déjà supporté)

3. **Noyau C optimisé**
   - Adapter le noyau NEON pour seq_max 512

## Benchmarks suggérés

Après entraînement, évaluer avec :

```bash
# Benchmark PPL
python tools/eval_25m_mlx.py \
    --model runs/hybrid-final/model.bin \
    --data data/hybrid/pretrain_mixed.jsonl

# Benchmark code
python tools/infer_hybrid.py \
    --model runs/hybrid-final/model.bin \
    --prompt "def factorial(n):" \
    --mode code

# Benchmark chat
python tools/infer_hybrid.py \
    --model runs/hybrid-final/model.bin \
    --prompt "explique l'intelligence artificielle" \
    --mode chat
```

## Notes importantes

- **Tokenizer** : Utilisez votre tokenizer BPE FR 16k existant ou ré-entraînez avec des données de code
- **Budget mémoire** : La config hybrid est estimée à ~40M params, devrait tenir dans <100 Mo
- **Temps d'entraînement** : 2-3 jours sur M4 pour le pipeline complet
- **Ollama** : Assurez-vous que qwen2.5-coder:1.5b est disponible pour la génération synthétique

## Arbre de décision

```
Résultats après fine-tuning instruct ?
├─ Oui (bonnes réponses code + chat)
│  └─ Ajouter fine-tuning QA pour factualité
├─ Non (réponses pauvres)
│  ├─ Plus de données ?
│  │  └─ Oui → Télécharger + générer plus
│  └─ Non → Augmenter capacité (config base)
└─ Partiel (un seul mode fonctionne)
   └─ Ajuster ratio code/chat dans données
```

## Prochaines étapes

1. Tester le téléchargement de datasets
2. Générer un petit set de données synthétiques (1000 samples)
3. Lancer un pré-entraînement test (10M tokens)
4. Évaluer et ajuster si nécessaire
5. Lancer le pipeline complet
