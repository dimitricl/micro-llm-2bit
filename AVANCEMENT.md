# Avancement — micro-llm-2bit

**Date : 2026-10-25.** État après **clôture de la chaîne F** (test du « plafond de
recette » à 25 M), **phase 0 de G terminée** (diagnostic), **phase 1 (G0–G1)
terminée** (QA-passage validée), et **G5 terminé** (pré-entraînement
from-scratch 46 M sur M4, évalué). La référence livrable reste **`runs/E-t08`**.
La chaîne F montre qu'une recette peut fortement améliorer la PPL **dans la
distribution**, mais aucune des trois recettes testées ne lève le plafond
**hors distribution**. La phase 0 de G ajoute un résultat méthodologique majeur :
**`val_big` est contaminé** (47,80 % de ses 32-grammes sont dans le train),
alors que **OOD est propre** (0,00 %). La phase 1 valide le levier QA-passage :
**G1-qa 35,0 % EM vs 20,2 % contrôle (+14,8 pp, 2 seeds, IC95 disjoints)**.
Le plan phase 3 est pré-enregistré dans `docs/PLAN_G.md` ; G1 validé, G2
prochaine étape sur M4, G3/G4 en attente de validation d'estimation GPU.

## En une ligne

La référence livrable reste **`runs/E-t08`** (val-PPL gelée 19,8 ; Δc val_big
−0,1949 ; Δc OOD −0,1708 ; export 12,13 Mo). La chaîne F montre qu'une
recette peut fortement améliorer la PPL **dans la distribution**, mais aucune
des trois recettes testées ne lève le plafond **hors distribution**. La phase
0 de G ajoute un résultat méthodologique majeur : **`val_big` est contaminé**
(47,80 % de ses 32-grammes sont dans le train), alors que **OOD est propre**
(0,00 %). Le critère principal devient OOD + clean32 + QA-passage ; le plan
phase 3 est pré-enregistré dans `docs/PLAN_G.md` et attend l'accord.

## Chaîne E — close

| run | val-PPL | Δc val_big | Δc OOD |
|---|---:|---:|---:|
| t01 | 25,8 | −0,0711 | −0,0557 |
| t04 | 21,6 | −0,1544 | −0,1381 |
| t07 | 20,1 | −0,1904 | −0,1588 |
| **t08** | **19,8** | **−0,1949** | **−0,1708** |
| t09 | 19,8 | −0,1945 | −0,1695 |

La PPL et les gains OOD plafonnent à t08 ; t09 (même budget, tranche
supplémentaire) ne fait que maintenir. Détail : `docs/RUNS.md`.

## Chaîne F — close (résultat négatif documenté)

Question : le plafond de 25 M est-il un plafond de **recette** ? Trois bras,
mêmes données et même budget que t09 (924 steps, 15,1 M tokens), reprise des
poids de t08 avec optimiseur réinitialisé.

| bras | recette | val-PPL | Δc val_big | Δc OOD | verdict |
|---|---|---:|---:|---:|---|
| A | KD 0,7, lr 3e-4, cosine, WD 0,01 | 19,8 | −0,1944 | −0,1714 | contrôle : reproduit t09 |
| B | CE seule, lr 1e-3, WSD, WD 0,1→0 | **15,8** | **−0,2830** | −0,1471 | gain in-domain, **perte OOD** |
| C | KD 0,7, lr 1e-3, WSD, WD 0,1→0 | 20,4 | −0,1788 | −0,1555 | pire partout |

- **A** valide le protocole : le reset d'optimiseur ne change rien
  (−0,1944 vs −0,1947 pour t09), et 924 steps de plus n'améliorent rien.
- **B** est le seul gain réel : val-PPL 19,8 → 15,8 et Δc val_big −0,195 →
  −0,283. Mais OOD −0,147 contre −0,171 : le gain ne transfère pas. Surface
  et sonde restent dans le bruit (OOV 0,74 % vs 0,44 %, sonde 4,0 % vs 2,0 %,
  top-1 log-proba 30,0 % vs 32,5 %).
- **C** montre que le lr élevé ne se transpose pas à la KD : 20,4, flips
  ternaires 6,5-9,5 % (vs 1,9 % pour A).

Conclusion : le plafond est **partiellement** une affaire de recette
in-domain, mais le plafond OOD et la factualité ne bougent pas. Détail
complet, IC, limites et artefacts : **`docs/CHAINE_F.md`** ; tableau
automatique : `docs/RESULTS_F.md`.

**Décision :** pas d'étape 5, aucun bras promu, t08 reste la référence.

## Chaîne G — **phase 0 close, phase 1 (G0–G1) validée**

**G0–G1 terminés.** Phase 0 (diagnostic) close ; phase 1 (QA-passage) **critères
validés** : G1-qa 35,0 % EM vs 20,2 % contrôle (+14,8 pp, 2 seeds, IC95
disjoints, 2 seeds cohérents). Déploiement OK (12,13 Mo, Δ RSS +28,7 Mo).

| mesure | t08 | F-B | G1-qa | lecture |
|---|---:|---:|---:|---|
| val-PPL gelée | 19,8 | **15,8** | — | F-B gagne in-domain |
| PPL clean32 (232 docs) | 22,09 [21,23–23,03] | **18,11 [17,33–19,03]** | 166–177* | le gain B survit au nettoyage |
| Δc OOD | **−0,1708** | −0,1471 | — | B ne transfère pas |
| 32-grammes val_big dans le train | 47,80 % | — | — | **val_big non propre** |
| docs val_E_frozen avec ≥1 32-gramme | **268/500 (53,6 %)** | — | — | val gelée non propre au niveau doc |
| docs OOD avec ≥1 32-gramme | **0/340** | — | — | OOD propre à n=32 |
| tokenizer BPE16k | 1,629 tok/mot | 2,211 (SmolLM) | — | −26 % de tokens |
| flips ternaires cumulés vs t08 | — | B 20,12 % | — | churn via zéro |
| **QA exact-match (test 300 items)** | **0,0 %** | **0,0 %** | **35,0 %** | **G1 valide** |
| **Écart QA−Ctrl** | — | — | **+14,8 pp** | **> 10 pp ✅** |

*PPL G1 élevée car entraînement CE masquée sur réponse seule ; ne reflète pas la capacité QA.

- **Fuite** : `val_big` recoupe t01–t08 jusqu'à 8,9 % à n=32, et 47,8 % au
  total. Il mesure le fitting, pas la généralisation. `val_E_frozen` est
  meilleur en fenêtres (3,53 % à n=32) mais **268/500 docs (53,6 %)** ont au
  moins un 32-gramme commun avec le train : ce n'est pas un jeu propre au
  niveau document. `OOD` est le seul jeu propre à n=32 (0/340 docs).
- **Composition OOD** : 3 romans seulement (Madame Bovary, Vingt mille
  lieues, Notre-Dame de Paris), 340 docs / 2,60 M chars. Les trois sont
  Gutenberg ; le catalogue local contient aussi les deux parties du Verne
  (5095/5096) et une édition anglaise de Madame Bovary (2413). G2 doit
  exclure ces livres par ebook ID, titre/auteur **et** 32-grammes.
- **Contrôle clean32** : sur 232 docs de `val_E_frozen` sans aucun 32-gramme
  commun avec le train, F-B reste à 18,11 de PPL contre 22,09 pour t08. Le
  gain in-domain de B est donc réel, mais il ne se transfère pas à la
  littérature.
- **Surface** : les écarts OOV de F sont non significatifs (IC chevauchant 0,
  p = 0,057–0,121). t08 a 0 occurrence de « capitain ».
- **Vitesse** : t08 107–125 tok/s / +28,3–28,4 Mo de Δ RSS ; base 768
  46,6–54,9 tok/s / +52,0–52,1 Mo. Les deux tiennent le budget de déploiement.
- **Tokenizer** : BPE FR 16k à 1,629 token/mot, 23,9 % de types fragmentés
  (vs 2,211 / 53,5 %), `capitale` et `capitaine` réparés. Reste à prouver le
  gain de BPC/OOD par un A/B.
- **G1 QA-passage validé** : banc gelé (300 test + 100 dev, templates
  disjoints, réponse sous-chaîne exacte). Baselines : t08/F-B/base 0 % EM.
  G1-qa (2 seeds) : 34,0 % / 36,0 % EM ; G1-ctrl (2 seeds) : 20,3 % / 20,0 %.
  Écart = +14,8 pp (> 10 pp), IC95 disjoints, 2 seeds cohérents. **G1 promu**.
  Déploiement OK (12,13 Mo, Δ RSS +28,7 Mo, 119,7 tok/s). PPL élevée attendue
  (CE masquée sur réponse).

Détail, IC, limites et artefacts : **`docs/CHAINE_G.md`** ; revue de
littérature : **`docs/RECHERCHE_G.md`** ; plan pré-enregistré :
**`docs/PLAN_G.md`** ; chronologie : `logs/G-heartbeat.md`.

## Pilotes QA — état inchangé

| pilote | sonde exact-match | surface « capitain » |
|---|---:|---:|
| #1 | 81/200 (40,5 %) | 44 occurrences |
| #2 | 21/200 (10,5 %) | — |
| #3 | 72/200 (36,0 %) | 59 occurrences |

Les faits montent avec les données QA, mais la corruption de surface ne
recule pas. F ajoute un point : optimiser la PPL (bras B) ne fait **pas**
monter la sonde. Si l'objectif est factuel, le levier est la donnée, pas la
recette d'optimisation.

## G5 — from-scratch 46 M (hors plan initial, terminé le 2026-10-25)

Run M4 de 100 000 steps (819,2 M tokens, 3,07 epochs, 59 h), loss finale
2,46–2,75. Le « 25M » du nom est en réalité **44,38 M de poids** (46,22 M
avec buffers) ; le log « 8 192 512 » était un bug de `num_params()`, corrigé.
Checkpoint F32 184,9 Mo ; exports int8 57,33 Mo (Δ RSS +76,5 Mo) et int4
35,15 Mo (+54,3 Mo).

| mesure | valeur |
|---|---|
| OOD BPC scoré S=128 / S=1024 | **1,9240 / 1,7851** |
| Δc OOD vs E-t08 | **−0,1880** [−0,1937 ; −0,1823] |
| Δc OOD vs t00 | −0,3538 |
| QA-passage 0-shot | **0,0 % EM** (F1 0,012) |
| sonde FR | 18,5 % [13,5 ; 24,0] |

Audit : OOD 0/680 snippets, `val_E_frozen` **300/500 snippets dans t00**
(inutilisable comme validation), `val_big` contaminé. Seul OOD est propre.

Conclusion : **meilleur modèle de langue** que E-t08 sur OOD, mais **pas un
assistant** (0 % QA, dérive factuelle). G1 reste le livrable QA ; le 46 M
est une base pour un fine-tuning QA ultérieur. Détail : `docs/CHAINE_G.md` §11.

## Décision G (accord donné, **phase 1 terminée**, G2 prochaine)

**G1 validée** : QA-passage +14,8 pp vs contrôle, 2 seeds, déploiement OK.
La phase 0 montre que le problème n'est pas le schedule mais (a) la capacité
effective pour l'OOD et (b) l'absence d'un banc d'utilité propre (résolu par
G0). Décision mise à jour :

1. **G0 + G1 terminés sur M4, sans dépense** : banc QA-passage gelé, QA
   extractive seq 256 (2 seeds + contrôle, 30,3 M tokens) — **validé**.
2. **G2 prochaine sur M4** : data mix 85 % Wikipédia / 15 % littérature
   Gutenberg (2 seeds + contrôle F-A). Audit Gutenberg/OOD déjà figé
   (`docs/results/gutenberg_ood_overlap_G.json` : 5 exclus, 459 propres).
   Durée ~1,5 jour, runs ≤ 2,3 h.
3. **G3 + G4 en parallèle sur GPU loué**, sans condition d'échec de G1/G2 :
   A/B tokenizer BPE16k puis base 768 from-scratch (72,9 M, export 29,25 Mo).
   Estimation **6–26 $** (plafond 30 $), à valider après calibration de
   débit (< 15 min, < 1 $) **avant location**. Un lane M4 + un lane GPU
   tournent simultanément.

Critères pré-enregistrés inchangés : OOD Δc ≤ −0,1808 (G2) ou ≤ −0,2008 (base),
clean32 BPC ≤ 1,5279, QA exact-match ≥ 20 % (G1 ✅), export/Δ RSS < 100 Mo
(G1 ✅). Un bras promu que si les 2 seeds vont dans le même sens. Détail :
`docs/PLAN_G.md`.

G3/G4 ne sont **pas** conditionnés à l'échec de G1/G2. On clôture sur t08
seulement si G1–G4 échouent tous, ou si le budget/temps est épuisé.

## Prochaines étapes

1. **G5-suite** : décider entre (a) fine-tuning QA du backbone 46 M avec la
   recette G1 (seq 512, CE masquée sur la réponse) ou (b) poursuite du plan
   G2 data mix. Le 46 M a une meilleure modélisation de langue que t08 mais
   0 % QA en 0-shot : c'est le candidat naturel pour tester si la capacité
   débloque la QA extractive.
2. **G2** : data mix 85/15 sur M4 (2 seeds + contrôle F-A), après audit
   Gutenberg/OOD déjà figé. Lancer après rédaction estimation GPU.
3. **G3/G4** : préparer l'adaptateur tokenizer natif et l'estimation GPU ;
   demander la validation de l'estimation avant location, puis lancer les
   deux lanes en parallèle (GPU pour G3/G4, M4 libre pour autre chose).
4. Mettre à jour `docs/CHAINE_G.md` et `logs/G-heartbeat.md` à chaque run,
   résultat négatif compris.

## État technique

- Tests : **175 passed, 1 skipped, 2 deselected** (171 existants + 4 tests
  WSD/flips/WD).
- Nouveaux outils F : `tools/run_F_arm.sh`, `tools/measure_F_arm.sh`,
  `tools/run_F_step4.sh`, `tools/summarize_F.py`, `tools/eval_surface.py`,
  `tools/diag_tokenizer.py`.
- Nouveaux outils G : `tools/ngram_overlap.py` (réécrit après saturation
  mémoire), `tools/oov_bootstrap_G.py`, `tools/flips_cumul_G.py`,
  `tools/tokenizer_fr_g.py`.
- Intégrité : sonde gelée `data/probe/probe_fr_v1.jsonl` inchangée
  (SHA256 `d9817afb…`), val gelée inchangée, aucun run E écrasé. Les
  sous-ensembles `*_clean32.jsonl` sont de **nouveaux** fichiers.
- Déploiement : greedy **nu interdit** (boucle sur tous les modèles) ;
  servir avec `--repetition-penalty 1.15 --no-repeat-ngram 3` ou sampling
  T=0,7 (P3). Les mesures F publiées utilisent ce défaut.
- Bench phase 0 : t08 12,13 Mo, +28,3–28,4 Mo de Δ RSS après génération,
  107–125 tok/s (128 tokens) ; base 768 29,25 Mo, +52,0–52,1 Mo,
  46,6–54,9 tok/s. Marge 48–72 Mo sous le budget delta.

## Où regarder

| sujet | fichier |
|---|---|
| phase 0 G, fuite val_big, clean32 | `docs/CHAINE_G.md` |
| revue de littérature G | `docs/RECHERCHE_G.md` |
| plan phase 3 pré-enregistré | `docs/PLAN_G.md` |
| chronologie / commandes G | `logs/G-heartbeat.md` |
| résultats F détaillés | `docs/CHAINE_F.md` |
| tableau F régénéré | `docs/RESULTS_F.md` |
| surface / OOV / log-proba | `docs/SURFACE.md` |
| tokenizer | `docs/TOKENIZER_DIAG.md` |
| chaîne E | `docs/RUNS.md` |
| ablations, décodage, sonde | `docs/ABLATIONS.md` |
| journal F | `logs/F-heartbeat.md` |
| état du dépôt / commandes | `README.md` |
