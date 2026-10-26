# Chaîne G — phase 0 : diagnostic + phase 1 : QA-passage (G0–G1)

**Date : 2026-10-21.** Phase 0 terminée ; phase 1 (G0–G1) **terminée**. La
référence livrable reste `runs/E-t08`. Le plan phase 3 est pré-enregistré
dans **`docs/PLAN_G.md`** ; les critères G1 sont **validés** (voir §10).

**Mise à jour 2026-10-25** : le run from-scratch 46 M est terminé et évalué
(§11). OOD propre, BPC scoré 1,92 à S=128 et 1,79 à S=1024, mais QA 0-shot 0 % ;
G1 reste le livrable QA.

## Résumé en cinq points

1. **`val_big` n'est pas un jeu de généralisation propre.** 47,80 % de ses
   32-grammes existent dans t01–t09 (vs 0,00 % pour OOD). Les chiffres
   Δc val_big de la chaîne F restent des mesures descriptives, pas une preuve
   de transfert.
2. **Le gain in-domain de F-B est réel, mais ne transfère pas.** Sur le
   sous-ensemble `val_E_frozen_clean32` (232 docs sans aucun 32-gramme
   commun avec le train), F-B fait 18,11 de PPL vs 22,09 pour t08 (−18 %),
   alors qu'il reste moins bon que t08 sur OOD (−0,1471 vs −0,1708).
3. **Le churn ternaire est massif et passe par zéro.** B a 20,12 % de poids
   ternaires cumulés différents de t08, mais seulement 0,06 % de retournements
   directs ±1→∓1. Cela colle au « deadzone trapping » de Tequila.
4. **Le tokenizer est un levier réel.** BPE FR 16k : 1,629 token/mot vs 2,211
   (−26 %), 23,9 % de types fragmentés vs 53,5 %, et `capitale`/`capitaine`
   deviennent un seul token.
5. **Le budget 100 Mo tient pour le base 768.** Delta RSS après génération :
   +28,3–28,4 Mo (t08) et +52,0–52,1 Mo (base 768), pour 12,13 et 29,25 Mo
   de poids packés.

## 1. Phase 0.1 — flags d'évaluation et IC du taux OOV

### 1.1 Les chiffres de t08 ne sont pas un artefact de flags

`eval_fixed` a été rejoué sur t08 avec exactement les flags de A/B/C
(`--repetition-penalty 1.15 --no-repeat-ngram 3`) :

- `eval/G-Et08-fixed/metrics.json` est **bit-identique** à
  `eval/F-Et08-fixed/metrics.json` (sha1 `cd2394de2493b78214ef9c9cc520039881a4fe6f`,
  échantillons `ff487c70…`).
- Conclusion : l'écart de surface entre t08 et les bras F ne vient pas d'un
  oubli de flags. Les mesures publiées sont bien comparables.

### 1.2 IC bootstrap du taux hors lexique

Le taux OOV de `eval_surface.py` est un rapport de comptes ; sans IC, l'écart
0,44 % → 0,74 % n'est pas interprétable. Bootstrap apparié sur les **textes**
(32 greedy + 32 sampling + 200 sonde), 10 000 tirages, lexique N≥5,
`tools/oov_bootstrap_G.py` :

| modèle | OOV combiné | IC95 | écart vs t08 | IC95 de l'écart | p bootstrap |
|---|---:|---|---:|---|---:|
| t08 | **0,44 %** | [0,26 ; 0,64] | — | — | — |
| F-A | 0,67 % | [0,42 ; 0,93] | +0,22 pp | [−0,06 ; +0,52] | 0,121 |
| F-B | 0,74 % | [0,47 ; 1,03] | +0,30 pp | [−0,04 ; +0,65] | 0,088 |
| F-C | 0,72 % | [0,49 ; 0,98] | +0,28 pp | [−0,01 ; +0,58] | 0,057 |

Les quatre IC chevauchent 0 : **les écarts de surface entre t08 et les bras F
sont dans le bruit**. On ne peut pas dire que B ou C dégradent la propreté
lexicale, ni que A l'améliore.

## 2. Phase 0.2 — recouvrement de n-grammes : le point qui change l'analyse

### 2.1 Méthode

- Tokenisation **SmolLM native** (pas le vocab réduit, pour éviter que les
  tokens hors vocabulaire fusionnent en un `unk` et créent de faux matches).
- Pour n = 8, 10, 13, 32 : hachage polynomial 64 bits de chaque fenêtre,
  jamais à cheval sur deux documents ; déduplication exacte par tri tranche
  par tranche.
- L'union des tranches n'est jamais matérialisée : pour chaque fenêtre de
  validation, recherche dichotomique dans chaque tranche puis OU logique.
  Exact, pic mémoire ~une tranche.
- Collisions 64 bits négligeables (probabilité de collision ~0,3 % au total
  pour ~3,2 × 10⁸ hash ; faux positif par requête ~1,7 × 10⁻¹¹).

### 2.2 Taux de fenêtres présentes dans le train t01–t09

| jeu | n=8 | n=10 | n=13 | n=32 |
|---|---:|---:|---:|---:|
| `val_E_frozen` vs train | 24,72 % | 13,74 % | 7,93 % | **3,53 %** |
| `val_E_frozen` vs t00 | 100,00 % | 100,00 % | 100,00 % | 100,00 % |
| `val_big` vs train | 69,96 % | 65,26 % | 61,41 % | **47,80 %** |
| `val_big` vs t00 | 30,17 % | 25,30 % | 22,87 % | 18,05 % |
| `ood` vs train | 3,10 % | 0,56 % | 0,08 % | **0,00 %** |
| `ood` vs t00 | 1,14 % | 0,15 % | 0,01 % | 0,00 % |

`val_E_frozen` vs t00 = 100 % est le **contrôle positif** : les 500 docs
proviennent de t00, donc toutes leurs fenêtres y sont. Ce n'est pas une fuite
vers le train (t00 n'est jamais entraînée).

Au niveau **document**, le taux de fenêtres sous-estime la contamination :
un seul 32-gramme partagé suffit à rapprocher un document du train.

| jeu | n=8 docs avec ≥1 match | n=13 | n=32 | n=32 avec ≥5 matches |
|---|---:|---:|---:|---:|
| `val_E_frozen` (500) | 500 (100 %) | 495 (99,0 %) | **268 (53,6 %)** | 241 (48,2 %) |
| `val_big` (300) | 300 (100 %) | 298 (99,3 %) | **277 (92,3 %)** | 274 (91,3 %) |
| `ood` (340) | 340 (100 %) | 126 (37,1 %) | **0 (0,0 %)** | 0 (0,0 %) |

C'est pour cette raison que clean32 retire **268 docs** de `val_E_frozen`
(et non 3,5 % de docs) : il exige zéro 32-gramme partagé. La val gelée de
500 docs n'est donc pas un jeu propre au niveau document ; elle reste utile
comme mesure de continuité, pas comme preuve de généralisation.

### 2.3 Ventilation par tranche (n=32 / n=13)

| jeu | t00 | t01 | t02 | t03 | t04 | t05 | t06 | t07 | t08 | t09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_E_frozen n=32 | 100 | 2,27 | 2,37 | 2,40 | 2,40 | 2,28 | 2,34 | 2,24 | 2,28 | 2,14 |
| val_big n=32 | 18,05 | 6,39 | 3,64 | 4,51 | 8,69 | 5,04 | 8,92 | 7,47 | 4,99 | **0,13** |
| val_big n=13 | 22,87 | 9,24 | 5,56 | 6,55 | 11,55 | 7,12 | 11,93 | 10,07 | 7,15 | **0,74** |
| OOD n=32 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |

Lecture : `val_big` recoupe fortement t01–t08, mais très peu t09 (0,13 % à
n=32). Le bras B, entraîné sur t09, ne doit donc pas son gain val_big à une
fuite directe de t09. En revanche, `val_big` est **contaminé par le corpus
d'entraînement global** (near-duplicates Wikipédia), ce qui gonfle la PPL
absolue et interdit d'en faire le critère principal de généralisation.

### 2.4 Sous-ensembles « clean32 » et contrôle de F-B

Pour séparer le vrai transfert de la quasi-duplication, on a retiré les
documents ayant **au moins un 32-gramme** commun avec t01–t09 :

| jeu source | docs gardés | usage |
|---|---:|---|
| `val_E_frozen` | **232 / 500** | `runs/abl_data/val_E_frozen_clean32.jsonl` |
| `val_big` | **23 / 300** | trop petit pour remplacer val_big ; spot-check |
| `val_E_frozen` (n=13 strict) | 5 / 500 | non utilisé (trop petit) |
| `val_big` (n=13 strict) | 2 / 300 | non utilisé (trop petit) |

Évaluation sur `val_E_frozen_clean32` (232 docs, 1 496 048 chars, même
protocole `eval_valbig.py`, S=128, 1000 tirages) :

| modèle | BPC (a) | IC95 BPC | PPL (a) | IC95 PPL |
|---|---:|---|---:|---|
| t08 | 1,5229 | [1,2986 ; 1,7709] | **22,09** | [21,23 ; 23,03] |
| F-B | **1,4250** | [1,2155 ; 1,6605] | **18,11** | [17,33 ; 19,03] |

Le gain de B **survit** au nettoyage : −18 % de PPL, IC disjoints. Ce n'est
donc pas un artefact de 32-grammes. Mais B reste moins bon que t08 sur OOD
(−0,1471 vs −0,1708) : B améliore la distribution Wikipédia, pas le
transfert littéraire.

### 2.5 Composition de l'OOD et recouvrement Gutenberg

L'OOD n'est pas un échantillon de 340 textes indépendants : c'est **trois
romans** découpés en 340 docs de ~8 000 caractères, dans l'ordre des livres
(`runs/abl_data/ood_manifest.json`) :

| source | ebook Gutenberg | docs | chars nettoyés |
|---|---:|---:|---:|
| Madame Bovary, Flaubert | 14155 | 89 | 683 309 |
| Vingt mille lieues sous les mers, Verne | 5097 | 114 | 876 607 |
| Notre-Dame de Paris, Hugo | 19657 | 137 | 1 045 623 |
| **total** | — | **340** | **2 604 662** |

Conséquences pour G2 :

- Les trois livres sont dans le domaine public Gutenberg ; les inclure dans le
  data mix serait une **fuite directe** de l'OOD.
- Le catalogue local contient aussi les deux parties du Verne (ebooks 5095 et
  5096) et une édition anglaise de Madame Bovary (2413) : elles doivent être
  exclues par **titre/auteur**, pas seulement par ebook ID.
- `Notre-Dame de Paris` (19657) n'est pas dans le catalogue local actuel,
  mais il est dans l'OOD : l'exclusion par n-grammes est indispensable.
- Avant G2, un audit exact compare les 340 docs OOD aux textes Gutenberg
  téléchargés (32-grammes SmolLM) ; tout livre avec **≥ 1 match** est exclu.
  Le résultat et la liste d'exclusion seront figés dans
  `docs/results/gutenberg_ood_overlap_G.json` avant le run.

## 3. Phase 0.3 — tokenizer français

`tools/tokenizer_fr_g.py`, BPE byte-level entraîné sur 40 Mo de
`tranche_01`, évalué sur `val_E_frozen` (821 771 mots) :

| tokenizer | vocab | tokens/mot | types fragmentés (pondéré) | tokens unk | mots unk (pondéré) |
|---|---:|---:|---:|---:|---:|
| SmolLM natif | 49 152 | 2,211 | 53,5 % | 0,00 % | 0,00 % |
| SmolLM → réduit E-t08 | 12 000 | 2,211 | 53,5 % | 0,96 % | 3,47 % |
| **BPE FR 16k** | **16 000** | **1,629** | **23,9 %** | 0,00 % | 0,00 % |
| BPE FR 12k | 12 000 | 1,693 | 26,8 % | 0,00 % | 0,00 % |

Mots de la confusion :

| mot | SmolLM | réduit E-t08 | BPE 16k |
|---|---|---|---|
| **capitale** | `Ġcapit` + `ale` | `Ġcapit` + `ale` | `Ġcapitale` |
| **capitaine** | `Ġcapit` + `aine` | `Ġcapit` + `aine` | `Ġcapitaine` |
| **capitole** | `Ġcapit` + `ole` | `Ġcapit` + `ole` | `Ġcapit` + `ole` |

Le BPE 16k réduit le nombre de tokens de 26 % et corrige la collision
`capitale`/`capitaine`. Il reste à vérifier qu'un modèle entraîné avec ce
tokenizer gagne en BPC/OOD : c'est l'expérience G3 du plan, pas un résultat
acquis.

## 4. Phase 0.4 — débit et RSS

Banc `main.py bench --bench-tokens 128`, un seul job à la fois, machine au
repos :

| modèle | config | poids packés | KV cache | Δ RSS après génération | tok/s (2 mesures) |
|---|---|---:|---:|---:|---:|
| `exports/E-t08.bin` | V=12k, d=512, L=8, seq=128 | 12,13 Mo | 0,52 Mo | **+28,3–28,4 Mo** | **107,4–125,1** |
| `exports/enfant_base_fr.bin` | V=12k, d=768, L=12, seq=256 | 29,25 Mo | 2,36 Mo | **+52,0–52,1 Mo** | **46,6–54,9** |

Le budget est un **delta RSS** (après chargement/génération) < 100 Mo, pas la
RSS absolue du processus (≈ 237–262 Mo, dont Python/numpy). Les deux modèles
tiennent avec une marge de 48–72 Mo. Le débit varie de ~15 % entre deux
mesures (cache CPU, charge résiduelle) : on retient la valeur basse pour les
décisions de budget, la valeur haute n'étant pas un gain de modèle.

Débits d'entraînement observés (chaîne F, M4) : ~3 184 tokens/s en CE seule
(F-B), ~1 821 tokens/s avec KD α=0,7 (F-A).

## 5. Phase 0.5 — flips ternaires cumulés

`tools/flips_cumul_G.py` compare les poids ternaires quantifiés de chaque
checkpoint à ceux de t08 (pas les flips par intervalle des logs) :

| modèle | step | flips cumulés vs t08 | retournements directs ±1→∓1 | mouvement latent relatif |
|---|---:|---:|---:|---:|
| B@23250 | 23 250 | 17,37 % | 0,01 % | 23,50 % |
| B@23500 | 23 500 | 19,75 % | 0,05 % | 29,80 % |
| **B@final** | 23 664 | **20,12 %** | **0,06 %** | **31,16 %** |
| A@final | 23 664 | 6,87 % | 0,00 % | 5,78 % |
| C@final | 23 664 | 15,75 % | 0,01 % | 25,28 % |

Flips **par intervalle** (journal d'entraînement, pas différence de cumul) :
23250→23500 = 11,80 %, 23500→final = 6,38 %.

Le churn passe presque exclusivement par **0** : les retournements directs
±1→∓1 sont rarissimes. C'est la signature du deadzone trapping : les poids
oscillent autour de la frontière du zéro sans se réengager. B, avec 20 % de
poids différents de t08, a modifié beaucoup plus de capacité que A (6,9 %)
pour un gain qui ne transfère pas.

## 6. Tableau de référence

Aucun modèle G n'a été entraîné. Les colonnes G sont des mesures de phase 0.

| modèle | val-PPL gelée | clean32 PPL | Δc val_big | Δc OOD | sonde | OOV N≥5 | tok/s | Δ RSS | durée |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **t08 (réf.)** | **19,8** | **22,09** | −0,1949 | **−0,1708** | 2,0 % | 0,44 % | 107,4–125,1 | +28,3–28,4 Mo | 279 min |
| F-A (contrôle) | 19,8 | — | −0,1944 | −0,1714 | 3,5 % | 0,67 % | — | — | 138 min |
| F-B (CE, WSD) | **15,8** | **18,11** | **−0,2830** | −0,1471 | 4,0 % | 0,74 % | — | — | 79 min |
| F-C (KD, WSD) | 20,4 | — | −0,1788 | −0,1555 | 2,5 % | 0,72 % | — | — | 115 min |
| G-0.2 (n-grammes) | — | — | — | — | — | — | — | — | ~25 min CPU |
| G-0.3 (tokenizer) | — | — | — | — | — | — | — | — | ~1 min |
| G-0.4 (bench base) | — | — | — | — | — | — | 46,6–54,9 | +52,0–52,1 Mo | ~2 s |

Δc = BPC(candidat) − BPC(tranche00), apparié par document, 1000 tirages,
seed 0 (négatif = meilleur). `clean32` = 232 docs de `val_E_frozen` sans
32-gramme commun avec t01–t09. t08 : 25,03 M paramètres, dont 18,87 M
ternaires.

## 10. Phase 1 (G0–G1) — QA-passage sur le tiny 25 M

### 10.1 G0 : banc QA-passage gelé

Le banc a été construit par `tools/build_qa_passage_G.py` (seed 0,
deterministe) à partir de sources **jamais entraînées** :
- 150 items test in-domain : `val_E_frozen_clean32` (232 docs sans
  32-gramme commun avec t01–t09)
- 150 items test OOD : `ood.jsonl` (340 docs de 3 romans, 0 % de 32-gramme
  commun avec le train)
- 100 items dev : templates disjoints, mêmes sources

Chaque item : passage (50–180 tokens SmolLM) + question + réponse
**sous-chaîne exacte du passage** ; total ≤ 230 tokens mappés pour tenir dans
seq 256. Templates test disjoints des templates d'entraînement (G1).

Fichiers gelés (SHA256) :
- `data/qa/g_passage_test_v1.jsonl` : `60be2f1b2b2aa98d2b412cb6d2d9f670986841884189741819d195cb4cd45830`
- `data/qa/g_passage_dev_v1.jsonl` : `5ce2d55b76bee80bd773c5d19422b12dfaab0c6331077ce868c8312ba87c3dec`
- Manifeste : `data/qa/g_passage_v1.manifest.json`

Audit programmatique (400 items) : 100 % réponse dans passage, 100 % réponse
unique dans la phrase cible, 0 item sans placeholder. Revue humaine sur 50
items : 49 valides (98 %), 1 ambigu (`Ordonnanceur Programmation`).

### 10.2 Baselines QA sur le banc (greedy, rep-pen 1.15, no-repeat 3)

| modèle | exact-match | IC95 | contains | F1 | copy | refus |
|---|---:|---|---:|---:|---:|---:|
| E-t08 | **0,0 %** | [0,0 ; 0,0] | 0,0 % | 0,001 | 0,0 % | 94,0 % |
| F-B | **0,0 %** | [0,0 ; 0,0] | 0,0 % | 0,001 | 0,3 % | 94,0 % |
| base 768 | **0,0 %** | [0,0 ; 0,0] | 0,0 % | 0,008 | 0,0 % | 0,0 % |

Le banc est difficile (plancher ≤ 5 %) et discriminera les futurs modèles.
base 768 répond sans refus mais n'atteint pas la réponse exacte.

### 10.3 G1 : entraînement QA-passage (CE masquée, 30,3 M tokens, 924 steps, seq 256)

Données : `tools/build_qa_train_G.py` sur t01–t09, 181 979 exemples,
~30,3 M tokens de contenu (924 steps × 8 × 16 × 256). Recette : CE
**uniquement sur les tokens de réponse** (`--mask-prompt answer`), lr 3e-4,
cosine, WD 0,01, warmup 5 %, alpha 0, batch 8 × accum 16, 1 époque.

Deux bras, 2 seeds chacun (4 runs) :

| modèle | seed | exact-match | IC95 | contains | F1 | copy | refus |
|---|---|---:|---|---:|---:|---:|---:|
| **G1-qa** | 0 | **34,0 %** | [28,7 ; 39,3] | 34,3 % | 0,463 | 41,7 % | 3,7 % |
| **G1-qa** | 1 | **36,0 %** | [30,7 ; 41,3] | 36,7 % | 0,481 | 43,7 % | 4,3 % |
| **G1-ctrl** | 0 | **20,3 %** | [15,7 ; 25,0] | 20,3 % | 0,289 | 42,7 % | 9,3 % |
| **G1-ctrl** | 1 | **20,0 %** | [15,7 ; 24,7] | 20,3 % | 0,286 | 38,7 % | 7,0 % |

**Résultat** : G1-qa moyenne **35,0 %** exact-match (CI95 ~ [30 ; 41]),
G1-ctrl moyenne **20,2 %** (CI95 ~ [16 ; 25]). L'écart est de **~14,8 pp**,
**bien au-dessus du seuil de 10 pp**. Les deux IC95 sont disjoints (pas de
chevauchement), confirmant la significativité statistique. Les deux seeds QA
sont du même côté de l'écart.

Répartition par source (moyenne des 2 seeds) :
- QA : in-domain 13,5 %, OOD 53,5 %
- Ctrl : in-domain 8,0 %, OOD 32,3 %

Le modèle QA excelle sur OOD (littérature), moins bien sur in-domain (Wiki).
Le contrôle sans question apprend moins (20 % vs 35 %).

### 10.4 Déploiement (G1-qa-s0)

| métrique | valeur | seuil | OK |
|---|---:|---:|:---:|
| export | 12,13 Mo | ≤ 100 Mo | ✅ |
| Δ RSS | +28,7 Mo | < 100 Mo | ✅ |
| tok/s | 119,7 | — | — |

Tous les budgets respectés.

### 10.5 Limites de la PPL sur modèles G1

Les modèles G1 ont été entraînés avec **CE masquée sur la réponse seule** ;
les tokens de prompt/passage n'ont jamais contribué à la loss. La PPL
générale est donc très élevée (clean32 PPL ~166, OOD PPL ~500) et **ne
reflète pas la capacité QA**. Le critère principal pour G1 reste l'exact-match
QA (≥ 20 % et ≥ contrôle +10 pp). Les métriques PPL/BPC sur clean32 et OOD
ne sont pas des critères de promotion pour G1 ; elles le redeviennent pour
les phases suivantes (G2+).

### 7.2 La capacité reste l'hypothèse dominante pour l'OOD

F-B améliore fortement l'in-domain propre (clean32 18,11 vs 22,09) et reste
moins bon sur OOD. F-A, avec KD, préserve l'OOD mais ne l'améliore pas. Le
facteur qui manque pour l'OOD n'est donc ni le LR ni le WD : c'est la
capacité et/ou la diversité des données. Le base 768 (72,9 M, ~36 M
équivalent fp16) est le seul levier de capacité dans l'enveloppe 100 Mo ; le
data mix littérature est le levier de diversité le moins cher.

### 7.3 Le tokenizer est le levier « apprendre plus vite » le plus net

−26 % de tokens par mot, −55 % de types fragmentés, `capitale`/`capitaine`
réparés. Mais il exige un ré-entraînement des embeddings et casse la KD
SmolLM ; il doit être testé en A/B avant d'être intégré au base.

### 7.4 La surface est déjà propre, la factualité ne l'est pas

t08 a 0 occurrence de « capitain », 0,44 % d'OOV, et la sonde exact-match est
à 2,0 % (plancher). Les pilotes QA #1/#3 montent à 36–40 % sur la sonde mais
corrompent la surface (« capitain » 44–59 occurrences). Le prochain levier
d'utilité n'est donc pas « plus de QA factuelle » au sens des pilotes, mais
une **QA passage** propre, avec une évaluation exact-match dédiée.

## 8. Limites

- **Phase 0 sans entraînement** : aucun modèle G, aucun seed d'entraînement ;
  les conclusions sur la capacité sont des inférences, pas des mesures.
- **clean32** : 232 docs, 1,5 M chars ; le choix des docs dépend du train
  (retrait des quasi-doublons), donc la distribution est plus difficile que
  val_E_frozen. La PPL 22,09 n'est pas comparable à la val-PPL gelée 19,8.
- **val_big clean32** : 23 docs seulement ; utilisé en spot-check, pas comme
  critère.
- **OOD** : 340 docs issus de **3 romans** seulement. Un modèle peut
  progresser sur ces trois styles sans progresser sur d'autres OOD ; le
  résultat OOD est un signal fort mais étroit.
- **OOV** : bootstrap sur 264 textes ; les IC sont larges, les écarts F sont
  non significatifs.
- **Bench** : delta RSS vs baseline Python ; la RSS absolue (~237–262 Mo)
  inclut l'interpréteur et numpy. Le critère « < 100 Mo » doit être entendu
  comme delta de déploiement, comme dans le reste du projet.
- **N-grammes** : tokenisation SmolLM ; un tokenizer différent changerait les
  frontières, mais la conclusion à n=32 (val_big contaminé, OOD propre) est
  robuste.
- **Flips** : comparaison de signes quantifiés, pas de trajectoire continue ;
  le mouvement latent est une norme relative globale, pas par couche.

## 9. Artefacts

| contenu | chemin |
|---|---|
| flags t08 | `eval/G-Et08-fixed/metrics.json`, `logs/G/step0-fixed-Et08.log` |
| IC OOV | `docs/results/oov_bootstrap_G.json`, `logs/G/step0-oov-bootstrap.log` |
| n-grammes | `docs/results/ngram_overlap.json`, `docs/results/ngram_overlap_pertranche.json`, `logs/G/step0-ngram*.log` |
| clean32 | `runs/abl_data/val_E_frozen_clean32.jsonl`, `runs/abl_data/val_big_clean32.jsonl` |
| clean32 t08 / F-B | `docs/results/vb_clean32_Et08.json`, `docs/results/vb_clean32_FB.json`, `logs/G/step0-clean32-*.log` |
| tokenizer | `docs/results/tokenizer_fr_G.json`, `docs/TOKENIZER_FR_G.md`, `logs/G/step0-tokenizer.log` |
| flips | `docs/results/flips_cumul_G.json`, `logs/G/step0-flips.log` |
| bench | `logs/G/step0-bench.log` |
| revue | `docs/RECHERCHE_G.md` |
| plan | `docs/PLAN_G.md` |
| heartbeat | `logs/G-heartbeat.md` |

### 9.1 Intégrité (SHA256)

| fichier | SHA256 |
|---|---|
| `data/probe/probe_fr_v1.jsonl` | `d9817afb15d85248c3f062d8b4b994e8b6c9ac3411de12f01923701f8815ba7c` |
| `runs/abl_data/val_E_frozen.jsonl` | `2b73c6c4ec0b6b5a4956a6c6a56ac268cee0eb39b8927e371101f59e64a7ebd3` |
| `runs/abl_data/ood.jsonl` | `7090566dd3e78a295fda3241813b8869e39b0e0d27bf5cc9973fc400f6dfadb9` |
| `runs/abl_data/val_E_frozen_clean32.jsonl` | `4af51a62b551be3e7c3a0bd9fa3a002fc92affaf5bbdd2d5b97ab970e88fe0dd` |
| `runs/abl_data/val_big_clean32.jsonl` | `582af1c6e14b29d0113d5fec0edff5d4d77e4726ab3de65ca35869746196b5a2` |

Ces fichiers n'ont pas été modifiés pendant la phase 0 ; les clean32 sont de
nouveaux fichiers dérivés, jamais des remplacements des jeux gelés.

## 11. Phase 2 (G5) — pré-entraînement from-scratch 46 M

**Date : 2026-10-25.** Run M4 terminé, évaluation complète effectuée.
Le nom historique « 25M » est conservé pour les chemins, mais le modèle
compte **44,38 M de poids** (46,22 M de tenseurs avec le masque causal et
les tables RoPE, qui ne sont pas des paramètres entraînés).

### 11.1 Configuration et run

| paramètre | valeur |
|---|---|
| architecture | 12 blocs, d=512, 8 têtes / 2 KV (GQA 4:1), FFN 1536 SwiGLU |
| contexte | 1024, RoPE, RMSNorm, embeddings liés |
| vocab | 16 000 (BPE FR byte-level) |
| batch | 32 × 1024 = 8 192 tokens/step |
| steps | 100 000 |
| tokens vus | **819,2 M** (≈ 3,07 epochs de `data/corpus.bin`, 266,6 M tokens) |
| durée | **212 545 s = 2 j 11 h 02** (3 854 tok/s moyens) |
| loss finale | 2,46–2,75 (dernier log 2,4594) |
| LR | 3e-4, warmup 2 000, cosine jusqu'à 3e-5 |
| checkpoints | 20 × 176 MiB + `model.safetensors` |

Le checkpoint fait 184,9 Mo car il est en **F32** (44,4 M poids + 1,84 M
buffers recalculables), sans état Adam. L'export déployable est quantifié
(§11.4).

### 11.2 Audit de contamination — préalable indispensable

`train_25m.py` n'a **aucun split de validation** : il échantillonne des
fenêtres dans les 266,6 M tokens, donc tout PPL sur `corpus.bin` est un PPL
d'entraînement. Audit par snippets exacts de 80 caractères (`rg -F`) :

| jeu | snippets trouvés dans t00–t09 | conclusion |
|---|---:|---|
| `ood.jsonl` (340 docs) | **0 / 680** | propre au niveau verbatim |
| `val_E_frozen.jsonl` (500 docs) | **300 / 500 dans t00 seul** | **contaminé** (t00 a été entraîné) |
| `val_big.jsonl` (300 docs) | 8 / 12 en échantillon | contaminé (47,80 % à n=32, connu) |

Conséquence : **seul OOD est un jeu de généralisation crédible** pour ce
run. `valE32` et `val_big` sont reportés à titre descriptif/in-domain.

### 11.3 Résultats

Protocole `eval_25m_mlx.py` : fenêtres non recouvrantes, NLL sommé par doc,
BPC = NLL / ln2 / caractères, PPL = exp(NLL/tokens), bootstrap 1 000 tirages
sur les docs (seed 0).

| jeu | contexte | BPC protocole [IC95] | BPC scoré | PPL [IC95] |
|---|---:|---|---:|---|
| **OOD** | 128 | **1,8727** [1,8596 ; 1,8854] | 1,9240 | 72,21 [71,25 ; 73,18] |
| **OOD** | 1024 | 1,5069 [1,4951 ; 1,5182] | **1,7851** | 53,23 [52,47 ; 53,99] |
| valE32 (contaminé) | 128 | 1,1362 [1,1088 ; 1,1687] | 1,1733 | 22,23 [21,27 ; 23,18] |
| valE32 (contaminé) | 1024 | 0,9207 [0,8852 ; 0,9603] | 1,0700 | 17,60 [16,75 ; 18,49] |
| val_big (descriptif) | 128 | 1,2154 [1,1955 ; 1,2399] | 1,2256 | 26,66 [25,93 ; 27,43] |

Deux dénominateurs sont reportés :
- **BPC protocole** = NLL / ln2 / caractères du document entier, convention
  `eval_valbig.py` (celle des références t00/E-t08) ; les fins de document
  non couvertes par une fenêtre complète pèsent donc à tort dans le
  dénominateur (biais faible à S=128, fort à S=1024) ;
- **BPC scoré** = NLL / ln2 / caractères réellement couverts par les
  fenêtres ; c'est la valeur honnête pour comparer S=128 et S=1024.

À S=1024, les documents plus courts que 1 025 tokens sont écartés
(OOD 339/340 docs, valE32 114/232). Les colonnes S=1024 ne sont donc pas
strictement comparables aux colonnes S=128 (231/232 et 340/340). Le gain
de contexte natif reste réel (PPL 72,2 → 53,2 ; BPC scoré 1,924 → 1,785)
mais plus modeste que ne le suggérait le BPC protocole.

Δc apparié par document (modèle − référence, négatif = meilleur) :

| référence | jeu | Δc [IC95] |
|---|---|---:|
| t00 | OOD | **−0,3538** [−0,3597 ; −0,3481] |
| E-t08 | OOD | **−0,1880** [−0,1937 ; −0,1823] |
| E-t08 | val_big (descriptif) | −0,3239 [−0,3361 ; −0,3097] |

Le Δc OOD vs E-t08 est **meilleur que la référence de gate −0,1708**, avec
IC disjoint. Réserve : la tokenisation et la fenêtre diffèrent de la chaîne
E/F ; le BPC par caractère neutralise l'essentiel, mais l'écart reste
indicatif, pas un résultat apparié au token près.

### 11.4 Déploiement (quantization MLX)

| variante | fichier | taille | Δ RSS | tok/s (greedy) | BPC OOD S=128 |
|---|---|---:|---:|---:|---:|
| F32 | `model.safetensors` | 184,87 Mo | +202,8 Mo | 67,3 | 1,8727 |
| **int8** | `exports/llm_25m_fr_int8.safetensors` | **57,33 Mo** | **+76,5 Mo** | 68,4 | 1,8729 |
| **int4** | `exports/llm_25m_fr_int4.safetensors` | **35,15 Mo** | **+54,3 Mo** | 68,0 | 1,9161 (+2,3 %) |

Les deux variantes tiennent le budget **≤ 100 Mo / Δ RSS < 100 Mo**. À
batch 1 et contexte court, la quantization ne change pas le débit (calcul
lié aux projections, pas à la bande passante) ; elle réduit la mémoire.
L'int4 perd 2,3 % de BPC et 10 % de PPL — acceptable pour un gain de 22 Mo.

### 11.5 Utilité : QA et sonde

| mesure | résultat |
|---|---|
| QA-passage 0-shot (300 items, greedy 16 tokens) | **0,0 % EM** [0 ; 0], F1 0,012, copie 0,0 %, refus 0,7 % |
| sonde FR gelée (200 items, greedy 32 tokens) | **18,5 %** [13,5 ; 24,0] |

Sonde : capitales 37,5 %, dates 5,0 %, définitions 12,5 %, unités/sciences
0,0 % ; par cœur 21,0 %, reformulé 16,0 %.

Lecture : le modèle est un **bon modèle de langue** (BPC OOD scoré 1,79 en
contexte natif, 1,92 à S=128 ; Δc protocole −0,188 vs E-t08) mais **pas un assistant** : il ne suit
pas le format QA et dérive factuellement dès qu'il sort des schémas vus.
G1 (QA extractif, 35 % EM) reste le livrable QA ; G5 est une meilleure
base de langage, pas un remplacement de G1.

### 11.6 Générations

Fichiers : `docs/results/g5_25m/gen_greedy.md`, `gen_t07.md` (20 prompts,
pénalité 1,15, no-repeat 3). Le français est fluide et bien ponctué, mais
les faits dérivent : « Le système solaire comprend un grand nombre de
satellites, dont le plus gros est le satellite Hotu », « Le chat […] un des
premiers animaux à pouvoir se déplacer dans le temps ». Aucune répétition
catastrophique grâce aux flags.

### 11.7 Artefacts

| contenu | chemin |
|---|---|
| checkpoints | `models/llm_25m_fr/checkpoint_*.safetensors` |
| PPL/BPC S=128 | `docs/results/g5_25m/eval_s128.json` + `perdoc_s128/` |
| PPL/BPC S=1024 | `docs/results/g5_25m/eval_s1024.json` + `perdoc_s1024/` |
| Δc OOD | `docs/results/g5_25m/delta_ood_{t00,Et08}.json` |
| QA 0-shot | `docs/results/g5_25m/qa_0shot.json` |
| sonde | `docs/results/g5_25m/probe.json` |
| quantized | `exports/llm_25m_fr_int{8,4}.safetensors` |
| qualité int8/int4 | `docs/results/g5_25m/eval_int{8,4}_ood_s128.json` |
| générations | `docs/results/g5_25m/gen_{greedy,t07}.md` |
| outils | `tools/eval_25m_mlx.py`, `generate_25m.py`, `eval_qa_25m.py`, `eval_probe_25m.py`, `export_25m.py` |

Flips ternaires : **non applicables** ici (modèle F32, pas de quantification
ternaire 2 bits).
