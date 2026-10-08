# Mode Think - Guide d'utilisation

## Prérequis

Le mode think nécessite :
- La branche `feature/think-mode` (merge ou checkout)
- Un modèle entraîné sur les traces think (think_pays.clean.jsonl, think_wiki.clean.jsonl)
- Le modèle actuel (tranche00 Wikipedia) ne produira PAS de raisonnements structurés

## Commandes

### 1. Activer le mode think

```bash
# Option 1 : Checkout sur la branche (recommandé pour expérimentation)
git checkout feature/think-mode

# Option 2 : Merge la branche dans main (pour integration permanente)
git merge feature/think-mode
```

### 2. Entraîner un modèle sur les traces think

Les traces think nettoyées sont dans `data_clean/` :
- `think_pays.clean.jsonl` : 250 traces (questions sur les pays)
- `think_wiki.clean.jsonl` : 242 traces (questions générales Wikipedia)

```bash
# Entraînement avec validation (5% val, eval tous les 200 steps, patience 3)
python main.py train \
    --data data_clean/think_pays.clean.jsonl data_clean/think_wiki.clean.jsonl \
    --parent HuggingFaceTB/SmolLM-135M \
    --out checkpoints/think.pt \
    --config tiny \
    --vocab-size 12000 \
    --seq-len 128 \
    --batch-size 16 \
    --accum 8 \
    --epochs 3 \
    --val-ratio 0.05 \
    --eval-every 200 \
    --patience 3
```

**Paramètres importants :**
- `--data` : fichiers JSONL contenant les traces avec format think
- `--parent` : modèle HF pour la distillation (SmolLM-135M recommandé pour la vitesse)
- `--config tiny` : configuration légère (~23M params, 9.6 Mo exporté)
- `--epochs 3` : 3 passes sur les traces think (données limitées)
- `--val-ratio 0.05` : 5% validation pour éviter l'overfitting
- `--eval-every 200` : évaluation périodique pour early stopping

### 3. Exporter le modèle entraîné

```bash
python main.py export --ckpt checkpoints/think.pt --out exports/think.bin
```

### 4. Inférence en mode think

```bash
# Mode think avec génération plus longue (512 tokens par défaut)
python main.py infer \
    --model exports/think.bin \
    --prompt "Quelle est la capitale de la France ?" \
    --think \
    --max-new 512
```

**Paramètres think :**
- `--think` : active le mode think (parsing automatique des balises)
- `--max-new 512` : génération plus longue pour le raisonnement
- Température par défaut : 0.7 (si non spécifié)
- top-k par défaut : 40 (si non spécifié)
- top-p par défaut : 0.9 (si non spécifié)

**Sortie attendue :**
```
=== RAISONNEMENT ===
<le raisonnement étape par étape du modèle>

=== RÉPONSE ===
<la réponse finale>
```

Si le format think n'est pas détecté, le modèle affichera le texte brut avec un message d'avertissement.

## Notes importantes

1. **Données limitées** : ~500 traces think seulement → risque d'overfitting. Surveiller la val-PPL.

2. **Format attendu** : Le modèle doit avoir vu le format `<think>...</think>` et `Reponse : ...` pendant l'entraînement pour le reproduire.

3. **Run actuel** : Le run tranche00 (Wikipedia) NE peut PAS être utilisé pour le mode think. Il faut un entraînement séparé sur les traces think.

4. **Branche** : Le code du mode think est dans `feature/think-mode`. Le merge dans main est à faire quand vous serez prêt à l'intégrer.
