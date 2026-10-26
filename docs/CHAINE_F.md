# Chaîne F — le plafond de 25 M est-il un plafond de recette ?

Hypothèse testée : la stagnation de la chaîne E (val-PPL gelée 19,8 de t01 à
t09, Δc OOD bloqué à −0,17) ne serait pas un plafond de **capacité** mais un
plafond de **recette** (KD α 0,7, lr 3e-4, cosine, WD 0,01 à chaque tranche).
Trois recettes ont été comparées depuis le **même point de départ**, sur les
**mêmes données** et le **même budget** que t09.

## Critère pré-enregistré

Un bras est retenu si :

1. val-PPL gelée **< 19,8** ; **et**
2. Δc apparié **meilleur que t08** sur val_big (−0,1949) **et** sur OOD
   (−0,1708).

Sans gain, l'étape 5 (pilote QA #4) n'est pas lancée.

## Protocole

| paramètre | valeur |
|---|---|
| point de départ | `runs/E-t08/model.pt`, step 22740 |
| reprise | **poids seuls**, optimiseur AdamW réinitialisé |
| données | `tranche_09.jsonl`, ordre seed 0 (identique à t09) |
| budget | 16 × 1479 = 23664 steps cumulés → **924 steps**, 15,1 M tokens |
| batch | 8 × accum 16, seq 128 |
| warmup | 5 % des 924 steps = **46 steps** |
| val | `runs/abl_data/val_E_frozen.jsonl` (gelée), éval tous les 100 steps |
| matériel | MPS, un seul job GPU, chien de garde val-PPL > 120 / NaN |

| bras | KD α | lr | scheduler | weight decay |
|---|---:|---:|---|---|
| **A** contrôle | 0,7 | 3e-4 | cosine + rewarmup 5 % | 0,01 constant |
| **B** CE seule | 0 | 1e-3 | WSD 5 % / 20 % / min 0,1 | 0,1 → 0 à 50 % |
| **C** KD | 0,7 | 1e-3 | WSD 5 % / 20 % / min 0,1 | 0,1 → 0 à 50 % |

A sert à mesurer l'effet du **reset d'optimiseur** : s'il ne change rien, A
doit reproduire t09 (19,8 ; Δc −0,1947 / −0,1695).

Commandes :

```bash
bash tools/run_F_step4.sh --go          # A -> mesures -> B -> mesures -> C
.venv/bin/python tools/summarize_F.py   # docs/RESULTS_F.md
```

## Résultats

Tableau complet (généré) : `docs/RESULTS_F.md`, données :
`docs/results/F_summary.json`.

| modèle | val-PPL | Δc val_big [IC95] | Δc OOD [IC95] | sonde | OOV N≥5 | top-1 logproba | flips | durée |
|---|---:|---|---|---:|---:|---:|---:|---:|
| t08 (réf.) | 19,8 | −0,1949 [−0,2013 ; −0,1890] | −0,1708 [−0,1736 ; −0,1680] | 2,0 % | 0,44 % | 32,5 % | — | 279 min |
| t09 | 19,8 | −0,1947 | −0,1695 | — | — | — | — | 125 min |
| **A** | **19,8** | **−0,1944** [−0,2009 ; −0,1883] | **−0,1714** [−0,1742 ; −0,1685] | 3,5 % | 0,67 % | 31,5 % | 1,87 % | 138 min |
| **B** | **15,8** | **−0,2830** [−0,2896 ; −0,2768] | **−0,1471** [−0,1501 ; −0,1444] | 4,0 % | 0,74 % | 30,0 % | 6,31 % | 79 min |
| **C** | **20,4** | **−0,1788** [−0,1849 ; −0,1728] | **−0,1555** [−0,1581 ; −0,1528] | 2,5 % | 0,72 % | 30,5 % | 6,51 % | 115 min |

Δc = BPC(candidat) − BPC(tranche00), apparié par document, 1000 tirages,
seed 0 (négatif = meilleur). Gates : **PASS** partout (aucune régression
> 0,03 BPC vs tranche00).

## Lecture

### A — le reset d'optimiseur n'est pas un facteur

A retombe exactement sur t09 : 19,8 ; −0,1944 vs −0,1947 sur val_big ;
−0,1714 vs −0,1695 sur OOD. L'écart est sous le MDE (0,006). Le protocole
« poids seuls + moments neufs » est donc un contrôle valide : les écarts de
B et C viennent des recettes, pas de la remise à zéro de l'optimiseur.

A confirme aussi que **924 steps de plus ne font pas progresser** : la
recette E, appliquée à t08, ne fait que maintenir la val et l'OOD.

### B — un vrai gain, mais local à la distribution d'entraînement

B (CE seule, lr 1e-3, WSD, WD 0,1 → 0) divise la val-PPL par 1,25 (19,8 →
**15,8**) et améliore le Δc val_big de **0,088 BPC** (−0,1949 → −0,2830).
C'est le seul bras qui déplace nettement un indicateur de capacité.

Mais le gain ne transfère pas :

- OOD : −0,1471 contre −0,1708 pour t08, soit **+0,024 BPC de régression**
  (IC disjoint) ;
- surface : OOV 0,74 % contre 0,44 % ; top-1 log-proba 30,0 % contre 32,5 %
  (écart non significatif à n = 200, mais pas d'amélioration) ;
- la sonde exact-match reste à 4,0 % (8/200) contre 2,0 % (4/200) : dans le
  bruit, aucune acquisition factuelle.

Interprétation : sans KD, le modèle consacre sa capacité à mieux prédire la
distribution de `tranche_09` (et val_big, qui en est proche). C'est un gain
d'**ajustement**, pas de **transfert**. La courbe val-PPL ne voyait pas la
différence ; l'OOD la voit.

### C — lr 1e-3 + KD casse la recette

C (même lr que B mais avec KD α 0,7) est **pire partout** : 20,4 de val-PPL,
−0,1788 sur val_big, −0,1555 sur OOD. Les flips ternaires montent à 6,5-9,5 %
contre 1,9 % pour A : à lr élevé, le gradient KL fait churner les poids
ternaires sans convergence. Le levier « lr plus haute » n'est donc pas
transposable tel quel à la distillation.

### Surface et factualité : rien ne bouge

Aucun bras ne produit « capitain » (0 occurrence), ni ne rapproche la sonde
du niveau du pilote QA #3 (36 %). Les OOV passent de 0,44 % à 0,67-0,74 %
(différences sur de petits compteurs), le top-1 log-proba reste à 30-32 %
(hasard 25 %). B n'achète pas de fait ; C ne dégrade pas la surface malgré sa
PPL.

En décodage de **déploiement** (pénalité 1,15 + no-repeat 3g, cf. P3), les
quatre modèles sont à 0,0000 de répétition 4-grammes : aucune régression de
boucles. En greedy nu, tous bouclent (t08 0,506 ; A 0,421 ; B 0,273 ;
C 0,521) : le défaut connu de t08 n'est ni créé ni corrigé par F, et le
défaut de déploiement reste obligatoire.

## Verdict

**Le critère pré-enregistré n'est atteint par aucun bras.**

- A : val-PPL affichée 19,8 (aucun gain mesurable ; 19,79 avant arrondi)
  et Δc val_big −0,1944, soit 0,0005 moins bon que −0,1949 — écart dans le
  bruit, mais pas un gain → échec sur le critère « meilleur que t08 ».
- B : val-PPL et val_big largement meilleurs, mais OOD moins bon
  (−0,1471 > −0,1708) → échec sur le transfert.
- C : échec sur les trois indicateurs.

**Réponse à l'hypothèse** : le plafond de 25 M est **partiellement** un
plafond de recette. Une recette (B) déplace fortement la PPL et le BPC
**dans la distribution**, donc la stagnation de la chaîne E n'était pas une
limite absolue de capacité in-domain. En revanche, aucun réglage testé ne
lève le plafond **hors distribution** ni la factualité : le plafond qui
compte pour le déploiement reste.

**Décision** : pas d'étape 5 (pilote QA #4). `runs/E-t08` reste la référence
livrée. Aucun des trois bras n'est promu.

## Suites possibles (non lancées)

1. **Clôturer et livrer t08** (recommandé dans l'état des preuves).
2. Tester une **recette mixte** (α entre 0 et 0,7, lr intermédiaire 5e-4,
   WSD, WD → 0) pour chercher le point où le gain in-domain de B est conservé
   sans la perte OOD. À pré-enregistrer avec le même critère ; un seul run
   de 924 steps (~2 h) suffirait à trancher.
3. Si l'objectif est factualité/mémoire, le levier reste les **données QA**
   (cf. pilotes #1-#3), pas la recette d'optimisation : B montre qu'optimiser
   la PPL ne fait pas monter la sonde.

## Limites

- Un seul seed par bras ; les IC appariés couvrent le bruit de mesure, pas la
  variance d'entraînement.
- val_big est proche de la distribution d'entraînement ; OOD (340 docs) est
  le seul contrôle de transfert.
- Sonde : n = 200, IC95 ≈ ±2,5 points ; les écarts 2,0 → 3,5-4,0 % ne sont
  pas significatifs.
- `eval_fixed` a d'abord été lancé en greedy nu pour les bras F (oubli des
  flags de déploiement). Les chiffres publiés ici et dans `SURFACE.md` ont
  été **remesurés** avec `--repetition-penalty 1.15 --no-repeat-ngram 3`,
  comme t08 et pilot-qa3. `tools/measure_F_arm.sh` porte désormais les flags.
- Le chien de garde (val-PPL > 120, NaN) n'a jamais déclenché ; C a divergé
  au sens statistique (20,4) sans exploser.

## Artefacts

| contenu | chemin |
|---|---|
| runs | `runs/F-A`, `runs/F-B`, `runs/F-C` (model.pt, train.log, measure.log) |
| exports | `exports/F-A.bin`, `F-B.bin`, `F-C.bin` |
| val_big / OOD | `docs/results/{valbig,ood}_F{A,B,C}[_perdoc].json` |
| gates | `logs/F/gate_{valbig,ood}_F{A,B,C}.txt` |
| surface | `docs/results/surface_F.json`, `docs/SURFACE.md` |
| échantillons | `eval/F-{A,B,C}-fixed/`, `eval/F-{A,B,C}-probe/` |
| tableau | `docs/RESULTS_F.md`, `docs/results/F_summary.json` |
| heartbeat | `logs/F-heartbeat.md` |
