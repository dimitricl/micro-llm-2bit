# Plan phase 3 — chaîne G (pré-enregistré)

**Statut : pré-enregistré le 2026-10-21, aucune exécution lancée.**
Ce document fige les hypothèses, les critères, les coûts et les règles d'arrêt
avant le premier run. Il ne doit pas être modifié après le début d'une
expérience ; tout ajout se fait dans `docs/CHAINE_G.md`, avec la date.

Demande d'accord : **G0 + G1 + G2 sur M4, sans dépense externe** (≈ 2–3 jours).
La location GPU (G3/G4, plafond **30 $**) fera l'objet d'une validation
d'estimation séparée avant toute location, conformément à la règle « pas de
dépense sans accord ». **G3/G4 ne sont pas conditionnés à l'échec de G1/G2** :
ce sont des leviers indépendants, exécutés sur le lane GPU pendant que le
lane M4 poursuit G1/G2.

## 0. Règles communes

1. **Un seul job par lane.** Un lane M4 (G1 puis G2) et un lane GPU loué
   (G3 puis G4) peuvent tourner **en parallèle** ; jamais deux jobs sur le
   même lane. Locks séparés `logs/gpu-m4.lock` et `logs/gpu-rented.lock`,
   vérification `pgrep`, refus si Ollama > 2 Go de VRAM sur le M4. Les logs
   et les dossiers de run sont distincts ; aucun fichier d'écriture partagé.
2. **Aucun écrasement.** Nouveaux dossiers `runs/G0-*`, `runs/G1-*`,
   `runs/G2-*`, etc. Les fichiers gelés (`data/probe/probe_fr_v1.jsonl`,
   `runs/abl_data/val_E_frozen.jsonl`, `runs/abl_data/val_big.jsonl`,
   `runs/abl_data/ood.jsonl`) sont lus seuls ; SHA256 vérifié avant/après.
3. **Logs obligatoires.** `logs/G-heartbeat.md` (chronologie, commandes,
   résultats, y compris négatifs) et `logs/G/<exp>-<seed>.log` (sortie brute).
4. **Mesures systématiques** : OOD Δc apparié (IC95), clean32 PPL/BPC,
   val-PPL gelée, val_big Δc (descriptif), OOV, sonde + log-proba, flips,
   tok/s, durée, export, Δ RSS.
5. **Générations** : greedy avec `--repetition-penalty 1.15 --no-repeat-ngram 3`
   (défaut de déploiement) ; sampling T=0,7 en secondaire. Jamais de greedy nu
   pour comparer les modèles.
6. **Seeds** : 2 seeds (0,1) pour toute décision serrée. Un bras n'est promu
   que si la moyenne passe **et** si aucun seed ne régresse au-delà du seuil.
7. **Arrêt** : dès qu'un critère échoue, la branche s'arrête ; on ne poursuit
   pas une branche perdante avec des variantes de lr/α.
8. **Budget** : phase 3 ≤ 10 jours ouvrés, dépense externe ≤ **30 $** (hors
   électricité locale). Une **estimation chiffrée** (heures × prix, buffer,
   stockage) est envoyée avant la location ; un court run de calibration
   (< 15 min, < 1 $) mesure le débit réel avant d'engager G4. Tout
   dépassement projeté impose un arrêt et une nouvelle demande.
9. **Tests** : `pytest -q` doit rester vert (175 passed, 1 skipped,
   2 deselected au départ). Tout nouvel outil G a au moins un test unitaire.
10. **Calibrage des steps** : avant chaque run, vérifier dans le log le
    nombre de steps réellement effectués. G1 (seq 256) cible **924 steps**,
    soit 30,3 M tokens (8 × 16 × 256 = 32 768 tokens/step). G2 (seq 128)
    cible **924 steps** sur le fichier de 24,2 M tokens. Si le fichier
    s'écarte de plus de 0,1 %, le recouper avant de lancer ; sinon le budget
    n'est plus comparable.

## 1. Mesures, critères et puissance

### 1.1 Critères principaux (pré-enregistrés)

| axe | mesure | seuil de promotion | référence |
|---|---|---|---|
| Généralisation OOD | Δc OOD apparié (BPC candidat − BPC tranche00), IC95 10 000 tirages | **≤ −0,1808** pour G2 (0,010 mieux que t08) ; **≤ −0,2008** pour G4 (0,030 mieux) ; garde G1 : pas pire que t08 +0,005 ; IC95 excluant 0 | t08 = −0,1708 |
| In-domain propre | BPC/PPL sur `val_E_frozen_clean32` (232 docs) | BPC ≤ **1,5279** (t08 +0,005) et PPL ≤ **23,2** (t08 +5 %) | t08 = 1,5229 / 22,09 |
| Utilité | exact-match QA-passage test (300 items : 150 clean32 + 150 OOD) | **≥ 20 %** et ≥ contrôle +10 pp, IC95 de l'écart excluant 0 (G1 ; garde pour G4) | t08 ≈ 0–2 % (à mesurer G0) |
| Surface | « capitain » et OOV N≥5 | « capitain » ≤ 5 occurrences ; OOV ≤ 0,84 % (t08 +0,2 pp) | t08 = 0 ; 0,44 % |
| Déploiement | export et Δ RSS après génération | export ≤ 100 Mo ; Δ RSS < 100 Mo | t08 = 12,13 Mo / +28,4 Mo |

`clean32` = documents de `val_E_frozen` sans aucun 32-gramme commun avec
t01–t09 (232/500). C'est le seul in-domain non contaminé ; la val-PPL gelée
(19,8) reste reportée pour la continuité, mais ne suffit plus à promouvoir.

`val_big` est **descriptif** : 47,80 % de ses 32-grammes sont dans le train.
Il ne peut pas servir de critère principal.

### 1.2 Statistiques

- Δc OOD et Δc clean32 : bootstrap apparié par document, 10 000 tirages,
  seed 0, IC 95 % percentile ; effet = moyenne des différences par document.
- QA : bootstrap apparié par item, 10 000 tirages ; exact-match et F1 token ;
  McNemar en secondaire.
- PPL clean32 : bootstrap par document, 1000 tirages (protocole
  `eval_valbig.py`).
- OOV : bootstrap apparié sur les textes, 10 000 tirages (protocole phase 0).
- MDE observée en F : ~0,006 BPC sur OOD ; les seuils 0,010/0,030 sont
  au-dessus du bruit.
- Pas de « peeking » : le test OOD/QA final n'est calculé qu'à la fin d'un
  run. Les arrêts en cours de run se fondent sur la val-PPL et la loss.

### 1.3 Ce qui n'est pas un critère

- Une hausse de val-PPL in-domain seule.
- Une hausse de val_big seule.
- Une génération flatteuse choisie à la main.
- La sonde factuelle des anciens pilotes #1/#3 : elle monte à 36–40 % mais
  corrompt la surface (« capitain » 44–59 occurrences). Elle reste un
  diagnostic secondaire, pas un objectif.

## 2. G0 — geler le banc QA-passage et les références clean32

**But** : mesurer l'utilité réelle avant d'entraîner, et disposer d'une
référence appariée sur clean32.

### 2.1 Construction du banc QA-passage

- Sources de passages : `val_E_frozen_clean32.jsonl` (in-domain) et
  `ood.jsonl` (littérature), découpés en passages de 50–180 tokens SmolLM
  (total question + réponse + passage ≤ 230 tokens, pour tenir dans seq 256).
- Génération automatique de questions à réponse **extraite du passage** :
  dates, nombres, entités nommées, lieux. Templates séparés train/test
  (familles de formulations disjointes).
- 150 items test in-domain + 150 items test OOD + 100 items dev (templates
  différents). Réponse toujours sous-chaîne exacte du passage.
- Métriques : exact-match après normalisation (minuscules, accents conservés,
  ponctuation de bord retirée, espaces compactés) ; F1 token ; taux de copie
  (la réponse générée est-elle dans le passage ?) ; taux de refus. Le brut
  est reporté à côté du normalisé.
- Fichiers gelés : `data/qa/g_passage_test_v1.jsonl`,
  `data/qa/g_passage_dev_v1.jsonl`, manifeste SHA256.
- Audit humain de 50 items : réponse valide ≥ 95 %, question non ambiguë
  ≥ 90 %. Si l'audit échoue, le banc est corrigé **avant** tout run, puis
  regelé.

### 2.2 Références clean32

- `docs/results/clean32_tranche00.json` + per-doc : tranche00 sur les 232
  docs clean32.
- `docs/results/clean32_Et08_perdoc.json`, `clean32_FB_perdoc.json` :
  t08 et F-B, pour vérifier que le Δc clean32 est cohérent avec la PPL.
- `docs/results/qa_passage_baseline.json` : t08, F-B et base export sur le
  banc QA (greedy de déploiement).

### 2.3 Critère de viability du banc

- t08 exact-match ≤ 5 % (plancher) et borne haute IC95 < 15 %.
- Si t08 > 15 %, le banc est trop facile ou contaminé : on le révise une
  fois, puis on abandonne la branche utilité si le problème persiste.
- Coût : ~0,5 jour CPU + ~1 h MPS pour les références. Aucun entraînement.

## 3. G1 — QA-passage sur le tiny 25 M (priorité 1)

**Hypothèse** : un corpus extractif court (passage + question → réponse)
apprend à un 25 M à répondre depuis le passage, sans professeur ni capacité
supplémentaire.

### 3.1 Changement unique

- Point de départ : `runs/E-t08/model.pt` (poids seuls, optimiseur neuf).
- Données : QA-passage généré sur t01–t09, **30,3 M tokens (924 steps à
  seq 256)**, passages ≤ 180 tokens, question + réponse ≤ 30 tokens,
  total ≤ 230.
- Objectif : CE **uniquement sur les tokens de réponse** (nouveau flag
  `--mask-prompt answer`, testé) ; le prompt et le passage ne contribuent pas
  à la loss.
- Recette fixe : lr 3e-4, cosine, WD 0,01, warmup 5 %, **seq 256**,
  batch 8 × accum 16, 1 époque calibrée à 924 steps.
- Contrôle : mêmes passages, mêmes réponses, **question remplacée par un
  préfixe neutre** (passage + réponse sans question), même budget. Il isole
  l'effet de l'appariement question→réponse.
- Seeds 0 et 1 pour les deux bras (4 runs).

### 3.2 Critère

- QA test exact-match **≥ 20 %** et **≥ contrôle +10 pp**, IC95 de l'écart
  excluant 0, sur les 300 items test.
- clean32 PPL ≤ 23,2 (pas plus de 5 % de dégradation) et OOD Δc pas pire que
  t08 de plus de 0,005.
- « capitain » ≤ 5 ; OOV ≤ 0,84 %.
- Les deux seeds doivent montrer le même signe sur l'écart QA−contrôle.

### 3.3 Arrêt / coût

- Si QA < 10 % ou écart < 5 pp : arrêt de la branche QA à 25 M ; on ne
  multiplie pas les pilotes.
- Coût : génération de données ~2–4 h CPU ; 4 runs × ~2,3 h = ~9 h M4 ;
  mesures ~1 h. Aucune dépense. Le budget de tokens double par rapport à
  seq 128, mais le nombre de steps reste 924 et le bras contrôle a le même
  budget.

## 4. G2 — Data mix littérature sur le tiny 25 M (priorité 2)

**Hypothèse** : l'OOD est de la littérature ; ajouter 15 % de romans français
publics améliore l'OOD sans casser l'in-domain, à recette et budget égaux.

### 4.1 Changement unique

- Point de départ : `runs/E-t08/model.pt`, poids seuls, optimiseur neuf.
- Données : 24,2 M tokens au total = **85 % t09 + 15 % Gutenberg français**
  (remplacement, pas ajout), pour garder 1479 steps/époque et 924 steps.
- Nettoyage : bandeaux Gutenberg retirés, déduplication, filtre de langue,
  **exclusion des livres OOD par ebook ID (14155, 5097, 19657), des doublons
  5095/5096 et 2413, puis de tout livre avec ≥ 1 32-gramme commun avec
  OOD/val**. L'audit est figé dans
  `docs/results/gutenberg_ood_overlap_G.json` avant le run.
- Recette : identique à F-A (KD α 0,7, lr 3e-4, cosine, WD 0,01,
  warmup 5 %, seq 128, batch 8 × accum 16, 16 époques depuis step 22740).
- Contrôle : F-A (t09 seule, même recette). Seed 0 existe ; on ajoute F-A
  seed 1 pour avoir 2 seeds contrôle. Bras littérature : seeds 0 et 1.
  Total 3 nouveaux runs + F-A existant.

### 4.2 Critère

- OOD Δc **≤ −0,1808** (0,010 mieux que t08), IC95 excluant 0.
- clean32 BPC ≤ 1,5279 ; OOV ≤ 0,84 % ; « capitain » ≤ 5.
- Les deux seeds doivent être du même côté de t08 ; aucun seed ne doit
  régresser de plus de 0,005 sur OOD.

### 4.3 Arrêt / coût

- Si l'un des seeds régresse sur OOD de plus de 0,005, ou si clean32 se
  dégrade de plus de 0,010 BPC : arrêt de la branche littérature.
- Coût : téléchargement/nettoyage ~1–2 h CPU ; 3 runs × ~2,3 h = ~7 h M4 ;
  mesures ~1,5 h. Aucune dépense.

## 5. G3 — Tokenizer BPE FR 16k (priorité 3, conditionnel)

**Hypothèse** : à budget de texte fixe, un tokenizer français natif voit
~35 % plus de texte et produit moins de fragmentation ; il améliore le BPC
clean32/OOD.

### 5.1 Changement unique

- Deux bras : SmolLM réduit 12k (contrôle) vs BPE FR 16k (natif), même
  architecture tiny 25 M, mêmes **100 M premiers tokens** de t01–t04, même
  recette CE-only, seq 128, 1 époque, 2 seeds par bras (4 runs).
- Pourquoi CE-only : la KD SmolLM n'est pas alignable sans VocAgnoLM ; on
  isole le tokenizer. Le BPC (bits par caractère) est comparable entre
  tokenizers.
- Préalable d'ingénierie : adaptateur `tokenizers.Tokenizer` → interface
  `encode`/`eos`/`vocab_size`, `vocab.json` natif, export `.bin` et
  `main.py generate` ; tests unitaires. Si l'adaptateur n'est pas prêt,
  G3 est reporté, pas bricolé.

### 5.2 Critère

- BPE16k BPC **inférieur de ≥ 0,010** sur clean32 **et** OOD, IC95 apparié
  excluant 0.
- tokens/mot ≤ 1,80 (gain ≥ 18 %) et 0 OOV.
- Si le gain n'est que sur un seul jeu, pas de promotion.

### 5.3 Arrêt / coût

- Si BPE16k est pire sur clean32 ou OOD : arrêt tokenizer ; G4 reste en 12k.
- Coût : ~1 jour d'ingénierie + 4 runs × ~1,5 h sur 4090 = ~6 h,
  **2–6 $** ; ou ~40 h M4 (non retenu). Décision de location séparée.

### 5.4 G3 peut-il tourner sur le M4 plutôt que sur un GPU loué ?

Oui, mais c'est un mauvais échange temps/argent :

- G3 = 4 runs × 100 M tokens = **400 M tokens**.
- Débit CE-only observé en F-B sur M4 : **3 184 tok/s** → 400 M / 3 184
  ≈ **35 h** de calcul, hors évaluations. À 4 000 tok/s (optimiste), ~28 h.
- Le lane M4 est déjà occupé par G1 (~9 h) puis G2 (~7 h) : ajouter G3 sur
  le M4 sérialise la phase 3 à ~45–50 h, avec risque de swap/thermal.
- Sur 4090, G3 prend ~5–7 h pour **2–6 $**. Le M4 doit rester le lane des
  runs tiny de G1/G2 ; le GPU est le lane de l'A/B tokenizer et du base.

Décision : **G3 sur GPU loué par défaut**. Si la location est refusée, G3
peut tourner sur M4 après G1/G2, mais il faut alors accepter ~2 jours de plus
et ne pas lancer deux jobs M4 en parallèle. Un pilote de 5 M tokens sur M4
(≈ 30 min) peut mesurer le débit réel avant de choisir, sans entraîner le
modèle complet.

## 6. G4 — Base 768 from-scratch (priorité 4, location)

**Hypothèse** : le plafond OOD de 25 M est un plafond de capacité effective ;
un 72,9 M ternaire (~36 M équivalent fp16) entraîné sur un corpus plus gros
et plus divers le lève, tout en tenant dans 100 Mo.

### 6.1 Configuration (changement de configuration, pas ablation)

- Architecture `base` : d=768, L=12, ffn=1536, GQA 8/4, seq 256, V=12k
  (ou BPE16k si G3 passe), 72,9 M paramètres, export ~29,25 Mo (~30–31 Mo
  avec V=16k).
- Données : t01–t09 (322 M tokens) + 15 % littérature (si G2 passe) +
  QA-passage (si G1 passe), 2 époques ≈ 650–800 M tokens selon les ajouts,
  soit ~18–22 tokens/paramètre **effectif** (~36 M équivalent fp16).
- Recette : WSD (warmup 5 %, decay 20 %, min 0,1), lr 1e-3, WD 0,1→0 à
  50 %, seq 256, batch 32 × accum 4. KD α 0,7 si tokenizer 12k ; CE-only
  si BPE16k (ou VocAgnoLM seulement si l'adaptateur est validé).
- 1 seed (0) pour le go/no-go ; 2e seed seulement si le résultat est à
  ≤ 0,005 du seuil.

### 6.2 Critère

- OOD Δc **≤ −0,2008** (0,030 mieux que t08), IC95 excluant 0.
- clean32 BPC ≤ 1,5279 ; export ≤ 100 Mo ; Δ RSS < 100 Mo à seq 256.
- QA exact ≥ 20 % **seulement si G1 a passé et que la QA est incluse** ;
  sinon la QA reste descriptive, sans critère de promotion.
- Si un seul des deux critères OOD/clean32 passe, pas de promotion.

### 6.3 Arrêt / coût

- Si la loss diverge, ou si l'OOD final est pire que t08 : arrêt de la
  branche capacité ; pas de 3e seed, pas de variante lr.
- Coût : 10–20 h de 4090 (0,34–0,89 $/h) = **3–18 $** ; mesures ~2 h ;
  plafond global 30 $. Demande d'accord séparée avant location.

## 7. G5 — Optimiseur ternaire (conditionnel, priorité 5)

**Hypothèse** : le churn par zéro (20,1 % pour B, 0,06 % de retournements
directs) signale un deadzone trapping ; une réactivation type Tequila ou un
optimiseur orthogonalisé (Muon) récupère de la capacité à budget constant.

### 7.1 Déclencheur

- Seulement si G4 est un « near-miss » (OOD entre −0,190 et −0,200) **et**
  si les flips montrent > 30 % de poids passés par zéro, ou si la loss
  stagne.

### 7.2 Protocole

- Tiny 25 M, t09, 924 steps, 2 seeds ; un seul changement (optimiseur ou
  réactivation), recette G2/G4 sinon identique.
- Critère : ≥ 0,005 BPC mieux sur clean32 **et** OOD, IC95 excluant 0.
- Coût : 4–6 h M4 ou 2 h 4090 ; ≤ 5 $.

## 8. Classement (gain × probabilité) / coût

Score = gain attendu (0–1) × probabilité de passer (0–1) / coût relatif
(1 ≈ 3 h M4 ; 3,5 ≈ 15 h 4090). G1 est passé à seq 256 : 924 steps =
30,3 M tokens, donc ~9 h M4 pour les 4 runs.

| rang | exp | gain | prob. | coût | score | dépendance |
|---:|---|---:|---:|---:|---:|---|
| 1 | **G1 QA-passage** | 0,85 | 0,70 | 3,0 | **0,198** | G0 |
| 2 | **G2 data mix littérature** | 0,70 | 0,55 | 2,0 | **0,193** | G0, audit Gutenberg |
| 3 | **G4 base 768** | 0,90 | 0,65 | 3,5 | **0,167** | G3 (tokenizer), G1/G2 (données) |
| 4 | **G3 tokenizer BPE16k** | 0,50 | 0,60 | 2,5 | **0,120** | adaptateur natif |
| 5 | **G5 optimiseur** | 0,35 | 0,30 | 2,0 | **0,053** | G4 near-miss |

Le classement est un ordre de valeur attendue ; l'**ordre d'exécution** suit
les dépendances et les lanes :

```text
G0 (banc QA + références) ──┬─ lane M4  : G1 → G2
                            └─ lane GPU : G3 → G4   (G5 si near-miss)
```

G3 est un préalable conditionnel à G4 : son score bas ne reflète pas sa
valeur d'option. Les deux lanes sont indépendants ; G3/G4 ne sont **pas**
conditionnés à l'échec de G1/G2. Si G1/G2 passent, G3/G4 restent utiles pour
la vitesse, la capacité et la généralisation.

## 9. Arbre de décision

1. **G0** : banc utilisable ? Si non après une révision, stop utilité.
2. **G1** : QA ≥ 20 % et ≥ contrôle +10 pp ?
   - Oui : jalon utilité ; on garde le checkpoint QA comme variante.
   - Non : l'utilité à 25 M n'est pas acquise ; G4 (base, seq 256) reste
     lancé indépendamment, avec la QA si elle est disponible.
3. **G2** : OOD ≤ −0,1808 ?
   - Oui : la littérature est un levier ; on l'inclut dans G4.
   - Non : pas d'ajout de littérature dans G4 ; G3/G4 restent lancés.
4. **G3** : BPE16k gagne sur les deux jeux ?
   - Oui : tokenizer BPE16k pour G4.
   - Non : 12k + KD pour G4 ; pas de 3e tokenizer.
5. **G4** : OOD ≤ −0,2008 et clean32/RSS conformes ?
   - Oui : nouveau modèle de référence, documentation et export.
   - Non : la branche capacité n'est pas promue ; on conserve le meilleur
     des modèles G1/G2/G3 et t08.
6. **G5** : uniquement si G4 est un near-miss diagnostiqué ; sinon, non.

**Clôture sur t08** seulement si **G1, G2, G3 et G4 échouent tous**, ou si
le budget/temps est épuisé. G3/G4 ne sont jamais conditionnés à l'échec de
G1/G2. Dans tous les cas, t08 reste la référence livrable et les résultats
négatifs sont documentés.

## 10. Commandes prévues (à figer avant exécution)

### G1 — QA

```bash
# 1) générer le corpus QA (t01–t09), filtrer les passages proches du test
.venv/bin/python tools/build_qa_passage_G.py \
  --train /Volumes/Lexar/clean-corpus/dedup/tranche_0{1,2,3,4,5,6,7,8,9}.jsonl \
  --test data/qa/g_passage_test_v1.jsonl \
  --out runs/G1-data/qa_train.jsonl --manifest runs/G1-data/qa_train.manifest.json

# 2) un bras, seed 0 (répéter seed 1 et le contrôle)
caffeinate -dims .venv/bin/python main.py train \
  --data runs/G1-data/qa_train.jsonl \
  --parent HuggingFaceTB/SmolLM-135M \
  --out runs/G1-qa-s0/model.pt \
  --config tiny --vocab-from exports/E-t08.bin.vocab.json \
  --resume runs/E-t08/model.pt --reset-optimizer \
  --lr-resume 3e-4 --rewarmup-ratio 0.05 \
  --seq-len 256 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 1 --seed 0 --alpha 0 --lr 3e-4 --temperature 2.0 \
  --sched cosine --weight-decay 0.01 --mask-prompt answer \
  --val-data runs/abl_data/val_E_frozen.jsonl --eval-every 100 \
  --ckpt-every 250 --track-flips
```

### G2 — data mix

```bash
.venv/bin/python tools/build_gutenberg_G.py \
  --catalog /Volumes/Lexar/rag-data/livres/pg_catalog.csv \
  --ood-manifest runs/abl_data/ood_manifest.json \
  --out runs/G2-data/litterature_fr.jsonl \
  --exclude runs/abl_data/ood.jsonl runs/abl_data/val_E_frozen.jsonl \
  --target-tokens 3630000

.venv/bin/python tools/mix_data_G.py \
  --wiki /Volumes/Lexar/clean-corpus/dedup/tranche_09.jsonl \
  --extra runs/G2-data/litterature_fr.jsonl --extra-frac 0.15 \
  --target-tokens 24200000 --out runs/G2-data/mix_t09_lit15.jsonl

caffeinate -dims .venv/bin/python main.py train \
  --data runs/G2-data/mix_t09_lit15.jsonl \
  --parent HuggingFaceTB/SmolLM-135M \
  --out runs/G2-lit-s0/model.pt \
  --config tiny --vocab-from runs/2026-10-08-tranche00/tranche00.pt.vocab.json \
  --resume runs/E-t08/model.pt --reset-optimizer \
  --lr-resume 3e-4 --rewarmup-ratio 0.05 \
  --seq-len 128 --save-batch 8 --batch-size 8 --accum 16 \
  --epochs 16 --seed 0 --alpha 0.7 --lr 3e-4 --temperature 2.0 \
  --sched cosine --weight-decay 0.01 \
  --val-data runs/abl_data/val_E_frozen.jsonl --eval-every 100 \
  --ckpt-every 250 --track-flips
```

### G4 — base (après accord location)

```bash
caffeinate -dims .venv/bin/python main.py train \
  --data runs/G4-data/mix_wiki_lit_qa.jsonl \
  --parent HuggingFaceTB/SmolLM-135M \
  --out runs/G4-base768-s0/model.pt \
  --config base --vocab-from runs/2026-10-08-tranche00/tranche00.pt.vocab.json \
  --from-scratch --seq-len 256 \
  --save-batch 32 --batch-size 32 --accum 4 \
  --epochs 2 --seed 0 --alpha 0.7 --lr 1e-3 --temperature 2.0 \
  --sched wsd --wsd-warmup-frac 0.05 --wsd-decay-frac 0.2 \
  --wsd-min-ratio 0.1 --weight-decay 0.1 --wd-zero-frac 0.5 \
  --val-data runs/abl_data/val_E_frozen.jsonl --eval-every 500 \
  --ckpt-every 1000 --track-flips
```

Ces commandes sont des **gabarits** : les outils `build_qa_passage_G.py`,
`build_gutenberg_G.py`, `mix_data_G.py`, `eval_qa_passage_G.py` et le flag
`--mask-prompt` restent à écrire et à tester avant le run correspondant.
Pour G4, `main.py train` fixe déjà `seq_max = --seq-len` (256) dans la
configuration ; aucun flag `--seq-max` n'est nécessaire. Aucun run ne démarre
avec un outil non testé.

## 11. Mesures après chaque run

```bash
# export et budget
.venv/bin/python main.py export --ckpt runs/<exp>/model.pt --out exports/<exp>.bin
.venv/bin/python main.py bench --model exports/<exp>.bin --bench-tokens 128

# OOD + clean32 appariés
.venv/bin/python tools/eval_valbig.py --ckpt runs/<exp>/model.pt \
  --docs runs/abl_data/ood.jsonl --vocab exports/E-t08.bin.vocab.json \
  --out docs/results/ood_<exp>.json --dump-per-doc docs/results/ood_<exp>_perdoc.json
.venv/bin/python tools/gate_paire.py --ref docs/results/ood_tranche00.json \
  --new docs/results/ood_<exp>_perdoc.json

# QA, surface, sonde, flips
.venv/bin/python tools/eval_qa_passage_G.py --ckpt runs/<exp>/model.pt \
  --test data/qa/g_passage_test_v1.jsonl --out docs/results/qa_<exp>.json \
  --repetition-penalty 1.15 --no-repeat-ngram 3
.venv/bin/python tools/eval_surface.py --ckpt runs/<exp>/model.pt
.venv/bin/python tools/eval_probe.py --ckpt runs/<exp>/model.pt
```

## 12. Livrables si succès

- G1 : `runs/G1-qa-s*/model.pt`, export, `docs/results/qa_*.json` ; variante
  « QA passage » du tiny, sans prétendre à la conversation générale.
- G2 : `runs/G2-lit-s*/model.pt` ; preuve (ou réfutation) du data mix.
- G3 : tokenizer natif + adaptateur testé ; décision documentée.
- G4 : `runs/G4-base768-s0/model.pt`, export ≤ 100 Mo, mesures complètes ;
  nouveau modèle de référence si les critères passent.
- Dans tous les cas : `docs/CHAINE_G.md` mis à jour, `logs/G-heartbeat.md`,
  `AVANCEMENT.md` avec la décision et les commandes exécutées.

## 13. Risques et limites

| risque | mitigation |
|---|---|
| Le banc QA est appris par cœur | templates train/test disjoints, passages de clean32/OOD, audit humain, contrôle sans question |
| La littérature Gutenberg contient les livres OOD | exclusion par 32-grammes + SHA256 + contrôle n-grammes après mélange |
| Le tokenizer natif casse l'export/inférence | adaptateur testé, A/B BPC, pas de run G4 avant validation |
| Le base 768 dépasse 100 Mo au seq 256 | bench avant/après, KV int8, export 29,25 Mo, marge 48 Mo |
| Le GPU loué coûte plus que prévu | plafond 30 $, mesure de débit avant location, arrêt si projeté > 30 $ |
| Un seed suffit à conclure | 2 seeds pour G1/G2/G3 ; G4 2e seed si near-miss |
| La val_big trompe encore | critère principal OOD + clean32 ; val_big descriptif |

## 14. Décision demandée

**Accord demandé maintenant** : lancer **G0, G1 et G2** sur M4, sans dépense
externe (~2–3 jours, runs ≤ 2,3 h chacun, un seul job M4 à la fois).

**Accord séparé, avant location** : G3/G4 sont indépendants de l'échec de
G1/G2. Je présenterai l'estimation ci-dessous et le débit mesuré par un run
de calibration (< 15 min, < 1 $) avant d'engager la location, même si G1/G2
ont réussi.

### 14.1 Estimation de location (à valider avant toute location)

Hypothèses : RTX 4090 Community/Secure, **0,34–0,89 $/h** (page RunPod du
2026-10-08), prix de planification 0,50 $/h, buffer 20 %, stockage/egress
1–2 $. Les heures seront affinées par la calibration.

| poste | calcul | heures | coût bas | coût haut |
|---|---|---:|---:|---:|
| G3 tokenizer (4 × 100 M tokens) | 5–7 h | 5–7 | 1,7 $ | 6,2 $ |
| G4 base 768 (650–800 M tokens, KD) | 8–16 h | 8–16 | 2,7 $ | 14,2 $ |
| G5 conditionnel (2 seeds, 25 M) | 2–4 h | 2–4 | 0,7 $ | 3,6 $ |
| calibration + stockage/egress | forfait | — | 1,0 $ | 2,0 $ |
| **total** | | **15–27 h** | **6,1 $** | **26,0 $** |

Le plafond de **30 $** couvre le total haut. Si la calibration mesure un débit
qui projette G4 au-delà de 16 h ou le total au-delà de 30 $, **arrêt et
nouvelle demande** ; on ne lance pas G4 « pour voir ». Le run de calibration
est un court entraînement de 10–15 min sur le 4090, facturé < 1 $, qui mesure
tok/s et mémoire avant d'engager les heures complètes.
