# Rapport d'améliorations prioritaires

Date : 9 octobre 2026

## 1. Résumé exécutif

Le projet respecte déjà sa contrainte système principale : un modèle
Transformer ternaire exporté, exécuté sur CPU Apple Silicon, avec un delta RSS
inférieur à 100 Mo. En revanche, les tests qualitatifs montrent que la qualité
de génération est insuffisante :

- le modèle Wikipedia répète des séquences comme `commune de la commune` ;
- il invente des noms et des relations malgré une val-PPL de 37.3 ;
- le modèle think produit parfois `<think>` mais pas `</think>`, puis répète
  des fragments ;
- la PPL mesurée sur une validation proche du corpus ne prédit pas encore la
  qualité sur des prompts libres.

La priorité est donc d'améliorer les données, l'évaluation et l'apprentissage
du comportement instructionnel avant d'augmenter la taille du modèle.

## 2. Résultats actuellement établis

### Modèle Wikipedia

- Configuration : `tiny`, 25.0M paramètres latents, vocabulaire 12 000,
  `seq_len=128`, parent `HuggingFaceTB/SmolLM-135M`.
- Données : une tranche Wikipedia française d'environ 107 Mo.
- Entraînement : 2 124 steps, 1 epoch, 7 927 secondes.
- Train-PPL : 39.2.
- Val-PPL : 37.3.
- Export : 12.13 Mo.
- Inférence : +27.3 Mo RSS et 76 tok/s au benchmark.
- Contrôle qualitatif : génération libre incohérente et sujette aux boucles.

### Modèle think

- Données : 278 traces validées, 134 pays et 144 Wikipedia.
- Configuration : `tiny`, `seq_len=128`, micro-batch 1, accumulation 16.
- Parent : `HuggingFaceTB/SmolLM-135M`.
- Train-PPL : 35.3.
- Val-PPL : 94.1.
- Export : 12.13 Mo.
- Contrôle qualitatif : traces répétitives, balises souvent incomplètes et
  réponse finale absente ou incorrecte.

## 3. Problèmes à corriger

### P0 — Évaluation insuffisante

La PPL évalue la prédiction de tokens sur des fenêtres de texte ; elle ne
mesure pas directement l'exactitude factuelle, la cohérence, la répétition ou
le respect d'un format de réponse. Il manque un jeu de test séparé, jamais
utilisé pour construire le vocabulaire ou entraîner le modèle.

À ajouter :

- un jeu de prompts fixes avec réponses attendues ;
- une mesure de répétition (`n`-grams répétés, longueur de boucle) ;
- une vérification stricte des balises think ;
- une comparaison checkpoint PyTorch contre export `.bin` ;
- une évaluation sur préfixes Wikipedia et sur questions libres séparément.

### P0 — Données think trop rares et trop homogènes

278 traces ne suffisent pas pour apprendre un comportement robuste. Les
questions pays sont très structurées et favorisent la mémorisation. Les
traces Wikipedia ne couvrent pas suffisamment les mathématiques, les sciences,
les comparaisons, les définitions et les problèmes multi-étapes.

Objectif initial : plusieurs milliers de traces validées, avec au moins 10 %
réservées au test final et aucun recouvrement de question entre train,
validation et test.

### P0 — Contexte trop court

Avec `seq_len=128`, le prompt et la réponse se partagent 128 positions. Une
trace think peut donc être tronquée avant `</think>`, même si
`--max-new=512` est demandé. Le prochain essai doit utiliser `seq_len=256`,
puis `512` si la mémoire le permet.

### P1 — Apprentissage du modèle général

Une seule tranche Wikipedia ne constitue pas encore une base générale. Les
neuf autres tranches doivent être utilisées progressivement, avec le même
vocabulaire et une validation stable. Il faut cependant mesurer le gain après
chaque tranche au lieu de lancer dix entraînements aveuglément.

La stratégie recommandée est :

1. conserver `wiki_slice00_bestval.pt` comme point de départ ;
2. réutiliser son vocabulaire ;
3. entraîner une tranche supplémentaire ;
4. évaluer PPL et génération ;
5. conserver le checkpoint seulement si la validation et la qualité
   qualitative progressent.

### P1 — Absence de garde-fous à l'inférence

Le sampling actuel peut amplifier les boucles. Il faut ajouter, sans modifier
le format du modèle :

- pénalité de répétition configurable ;
- blocage des répétitions de bigrammes ou trigrammes ;
- arrêt sur EOS et éventuellement sur `</think>` ;
- détection explicite d'une sortie think tronquée ;
- tests déterministes avec température zéro.

Ces garde-fous ne remplaceront pas de meilleures données, mais ils rendront le
comportement mesurable et éviteront les sorties manifestement dégénérées.

### P1 — Format des traces think

Le format d'entraînement doit être identique au format demandé en inférence.
Il faut normaliser les espaces, les marqueurs et la réponse finale, puis
rejeter les traces ambiguës. Les réponses doivent être contrôlées par une
source ou une règle lorsque c'est possible ; une trace fluent mais fausse
dégrade directement le modèle.

### P2 — Distillation et spécialisation

Le modèle think actuel est entraîné depuis zéro sur les traces think. Une
meilleure recette est :

1. pré-entraîner le modèle général sur un corpus français large ;
2. vérifier sa PPL et ses sorties ;
3. poursuivre l'entraînement sur des traces think diversifiées ;
4. réduire progressivement la part de texte général pour préserver la
   langue et les connaissances.

Le parent `SmolLM-135M` peut rester le professeur pour la première itération.
Passer à `SmolLM-360M` ne doit être envisagé qu'après stabilisation des
données et de l'évaluation.

## 4. Plan d'exécution recommandé

### Étape A — Mesure et fidélité

- ajouter une évaluation fixe indépendante ;
- mesurer la PPL du checkpoint et de l'export sur le même texte ;
- enregistrer les sorties avant/après export ;
- mesurer les répétitions et les troncatures.

Critère : aucun écart inexpliqué entre checkpoint et export, et un rapport
reproductible.

### Étape B — Données générales

- entraîner une deuxième tranche avec le vocabulaire de `slice_00` ;
- comparer la validation commune et les prompts fixes ;
- ne conserver que les améliorations vérifiées.

### Étape C — Données think

- générer au moins 5 000 traces diversifiées ;
- valider automatiquement le format et les réponses ;
- garder un test séparé de 10 % ;
- entraîner avec `seq_len=256`.

### Étape D — Qualité d'inférence

- ajouter les pénalités anti-répétition ;
- tester température 0, puis sampling ;
- vérifier les balises et la réponse finale ;
- documenter les échecs plutôt que les masquer.

### Étape E — Taille du modèle

Ne tester `base` qu'après les étapes précédentes. Le modèle `base` a déjà
montré un risque d'OOM MPS dans une configuration think ; augmenter la taille
avant d'avoir des données et des métriques solides rendrait le diagnostic plus
difficile.

## 5. Critères de réussite

Le projet pourra être considéré comme fonctionnel lorsque le modèle :

- conserve un delta RSS inférieur à 100 Mo ;
- garde une PPL de validation stable sur un jeu jamais vu ;
- réduit fortement les répétitions sur les prompts fixes ;
- répond correctement à une majorité des questions de contrôle ;
- produit systématiquement un format think complet lorsque ce mode est activé ;
- conserve ces propriétés après export et quantification.

Une PPL basse seule ne sera plus considérée comme suffisante.
