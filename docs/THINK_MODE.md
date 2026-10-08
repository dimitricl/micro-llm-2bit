# Mode Think - Guide d'utilisation

## Prérequis

Le mode think est intégré à `main`. Il nécessite un modèle entraîné sur des
traces contenant `<think>...</think>` et `Reponse :`. Le modèle Wikipedia
`tranche00` ne doit pas être utilisé pour cette fonction.

## Commandes

### 1. Entraîner un modèle sur les traces think

Les traces utilisées pour le run de référence sont dans le corpus local
`/Volumes/Lexar/clean-corpus/` :
- `think_pays.clean.jsonl` : 134 traces retenues
- `think_wiki.clean.jsonl` : 144 traces retenues

```bash
# Entraînement de référence avec validation (10% val, eval tous les 50 steps)
python main.py train \
    --data /Volumes/Lexar/clean-corpus/think_pays.clean.jsonl \
            /Volumes/Lexar/clean-corpus/think_wiki.clean.jsonl \
    --parent HuggingFaceTB/SmolLM-135M \
    --out checkpoints/enfant_think.pt \
    --config tiny \
    --vocab-size 12000 \
    --seq-len 128 \
    --save-batch 2 \
    --batch-size 1 \
    --accum 16 \
    --epochs 8 \
    --val-ratio 0.10 \
    --eval-every 50 \
    --patience 4
```

**Paramètres importants :**
- `--data` : fichiers JSONL contenant les traces avec format think
- `--parent` : modèle HF pour la distillation (SmolLM-135M recommandé pour la vitesse)
- `--config tiny` : configuration légère (~23M params, 9.6 Mo exporté)
- `--epochs 8` : plafond de passes, avec early stopping
- `--val-ratio 0.10` : 10% validation sur ce petit corpus
- `--eval-every 50` : évaluation périodique pour early stopping

### 3. Exporter le modèle entraîné

```bash
python main.py export --ckpt checkpoints/enfant_think.pt \
    --out exports/enfant_think.bin
```

### 4. Inférence en mode think

```bash
# Mode think ; la génération reste bornée par le seq_max exporté (128 ici)
python main.py infer \
    --model exports/enfant_think.bin \
    --prompt "Quelle est la capitale de la France ?" \
    --think \
    --max-new 512
```

**Paramètres think :**
- `--think` : active le mode think (parsing automatique des balises)
- `--max-new 512` : plafond demandé, limité en pratique par le contexte exporté
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

1. **Données limitées** : 278 traces think seulement → surapprentissage
   probable ; surveiller la val-PPL.

2. **Format attendu** : Le modèle doit avoir vu le format `<think>...</think>` et `Reponse : ...` pendant l'entraînement pour le reproduire.

3. **Résultat actuel** : `enfant_think.bin` sait produire le début du format
   think, mais les tests actuels montrent des répétitions et des sorties
   parfois tronquées avant `</think>`. Ce n'est pas encore un modèle fiable.

4. **Contexte** : avec `seq_len=128`, le prompt et la génération se
   partagent 128 positions ; `--max-new 512` ne permet donc pas 512 tokens
   effectifs.
