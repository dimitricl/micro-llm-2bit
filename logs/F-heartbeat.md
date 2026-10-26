# Chaîne F — heartbeat

Hypothèse testée : le « plafond de capacité à 25 M » est peut-être un plafond
de **recette**. Référence : `runs/E-t08` (val-PPL gelée 19,8 ; Δc val_big
−0,1949 ; Δc OOD −0,1708 ; export `exports/E-t08.bin`).

Règles : un seul job GPU à la fois, sonde gelée
(`d9817afb…`) et val gelée intouchées, nouveaux dossiers `runs/F-*`,
aucun fichier de référence écrasé, tests 171 verts avant/après.

---

## 2026-10-20 17:31 — Étape 1 : bench de l'export E-t08

`exports/E-t08.bin` n'existait pas : créé par copie de
`runs/E-t08/model.bin` (+ vocab), sans écraser.

Commande :

```
.venv/bin/python main.py bench --model exports/E-t08.bin
.venv/bin/python main.py bench --model exports/E-t08.bin --bench-tokens 128
```

Résultats (`logs/F/step1-bench-Et08*.log`) :

| mesure | 32 tokens | 128 tokens |
|---|---:|---:|
| delta RSS au chargement | +0,2 Mo | +0,3 Mo |
| delta RSS après génération | +27,5 Mo | **+28,0 Mo** |
| débit | 67,9 tok/s | **118,9 tok/s** |
| structurel (poids + KV + act.) | 12,12 Mo | 12,12 Mo |

Verdict : large marge sous 100 Mo. Le débit de 118,9 tok/s est la valeur
stationnaire à retenir (le 32 tokens inclut le préfill).

---

## 2026-10-20 17:36 — Étape 2 : diagnostic tokenizer

Outil : `tools/diag_tokenizer.py` (nouveau). Sorties :
`docs/TOKENIZER_DIAG.md`, `docs/results/tokenizer_diag.json`,
`logs/F/step2-diag-tokenizer.log`.

Découpage SmolLM -> vocab réduit E-t08 :

| mot | sous-mots | unk |
|---|---|---|
| capitale | `cap` + `it` + `ale` | non |
| capitaine | `cap` + `it` + `aine` | non |
| capitole | `cap` + `it` + `ole` | non |
| capitalle | `capital` + `le` | **oui** (id 0 = eos) |

- Les 3 formes correctes partagent leurs **2 premiers sous-tokens** ; seul le
  dernier token diffère. La confusion est donc **plausible par le découpage**.
- `capitalle` n'est même pas représentable : `capital` est hors vocabulaire et
  retombe sur l'id 0 (unk = eos dans le vocab E).
- Sur les 54 mots analysés : 50,0 % sont découpés, 3,7 % ont un token unk,
  2,17 % des tokens sont unk ; 48,1 % des mots découpés partagent leur premier
  sous-token avec un autre mot du jeu.
- Fréquences dans le corpus d'entraînement : capitale 23 497, capitaine 12 699,
  capitole 1 073, capitalle 2. La forme correcte est **plus** fréquente : la
  corruption n'est pas un simple effet de fréquence lexicale.
- Conclusion : le découpage rend la confusion possible (un seul token pour
  trancher) mais ne suffit pas à l'expliquer. L'étape 3 mesure la surface.

---

## 2026-10-20 17:38 — Étape 3 : surface + classement log-proba

`tools/eval_surface.py` (nouveau). Lexique : 167,0 M mots du corpus
d'entraînement dédupliqué (tranches 00-09), 1 200 495 types ; seuils N≥5
(331 297 types) et N≥20 (145 508 types). Cache :
`runs/F-diag/lexique_fr.tsv.gz`.

### Taux hors lexique (N≥5, tokens-mots)

| run | source | mots | OOV | taux | « capitain » |
|---|---|---:|---:|---:|---:|
| t08 | eval_fixed greedy | 882 | 2 | 0,23 % | 0 |
| t08 | eval_fixed sampling | 858 | 5 | 0,58 % | 0 |
| t08 | sonde | 3 026 | 14 | **0,46 %** | 0 |
| t08 | combiné | 4 766 | 21 | **0,44 %** | 0 |
| pilot-qa3 | eval_fixed greedy | 752 | 18 | 2,39 % | 11 |
| pilot-qa3 | eval_fixed sampling | 747 | 13 | 1,74 % | 8 |
| pilot-qa3 | sonde | 2 671 | 77 | **2,88 %** | 59 |
| pilot-qa3 | combiné | 4 170 | 108 | **2,59 %** | 78 |

À N≥20 : t08 0,59 %, pilot-qa3 2,83 % (combiné). Le pilote QA #3 dégrade la
surface d'un facteur **5,9** sur la sonde et produit 78 occurrences
« capitain » contre 0 pour t08.

### Classement log-proba (1 bonne réponse vs 3 distracteurs, hasard 25 %)

| modèle | top-1 moyen | top-1 somme | MRR | rang moyen | par_coeur | reformulé |
|---|---:|---:|---:|---:|---:|---:|
| tranche00 | 27,5 % | 28,5 % | 0,532 | 2,48 | 25,0 % | 30,0 % |
| t08 | **32,5 %** | 35,5 % | 0,565 | 2,37 | 35,0 % | 30,0 % |
| pilot-qa3 | **40,0 %** | 46,5 % | 0,623 | 2,14 | 43,0 % | 37,0 % |

Par catégorie (top-1) : t08 31,2 % capitales / 40,0 % dates / 27,5 %
définitions / 32,5 % unités ; pilot-qa3 37,5 / 65,0 / 22,5 / 37,5 ;
tranche00 28,7 / 22,5 / 25,0 / 32,5.

Lecture : t08 est à peine au-dessus du hasard (2,45 σ), pilot-qa3 le dépasse
nettement (4,9 σ) — le pilote QA achète une discrimination factuelle, mais
au prix d'une surface nettement plus sale. La sonde exact-match (t08 2,0 %,
qa3 36,0 %) et le classement ne mesurent pas la même chose : qa3 sait mieux
*classer* les faits qu'il ne sait les produire, et il produit surtout la
mauvaise forme.

Sorties : `docs/SURFACE.md`, `docs/results/surface_F.json`,
`logs/F/step3-surface.log`.

---

## 2026-10-20 18:07 — étape 4 : campagne A/B/C

- ordre A B C ; 924 steps/bras ; reprise E-t08 poids seuls
- log global : `logs/F-step4.log`

## 2026-10-20 18:07 — étape 4, bras A : départ

- A contrôle (KD 0.7, lr 3e-4, cosine)
- données : `/Volumes/Lexar/clean-corpus/dedup/tranche_09.jsonl` (tranche_09, mêmes tokens que t09)
- reprise : `runs/E-t08/model.pt` ; 924 steps ; seed 0
- logs : `runs/F-A/train.log`

## 2026-10-20 19:07 — étape 4 : lancement de la campagne A/B/C

Accord utilisateur obtenu. `qwen2.5-coder:14b` déchargé (9,1 Go) ;
`gemma3:1b` (0,87 Go) laissé résident à la demande de l'utilisateur. Les
scripts tolèrent désormais < 2 Go de VRAM Ollama au lieu de refuser tout
modèle chargé.

Commande : `nohup bash tools/run_F_step4.sh --go > logs/F-step4.log 2>&1 &`

- ordre A → mesures → B → mesures → C → mesures → `summarize_F.py` ;
- 924 steps/bras, reprise `runs/E-t08/model.pt` (step 22740), poids seuls,
  optimiseur neuf, warmup 46 steps, seed 0 ;
- chien de garde : arrêt si val-PPL > 120 ou loss NaN (marqueur
  `runs/F-$ARM/.diverged`), puis arrêt de la campagne ;
- premier point de contrôle : bras A step 22760, loss 1,3736, val non encore
  évaluée, 2404 tok/s (≈ 8,4 s/step → ~2,15 h pour 924 steps).

Nouveaux outils : `tools/measure_F_arm.sh`, `tools/run_F_step4.sh`,
`tools/summarize_F.py` (tableau `docs/RESULTS_F.md`). Tests : 175 passed.
- fin bras A : exit=0, durée=8343s

## 2026-10-20 20:35 — étape 4, bras A : mesures

- durée mesures : 545s ; log `runs/F-A/measure.log`
- val_big : `docs/results/valbig_FA.json` ; gate `logs/F/gate_valbig_FA.txt`
- OOD : `docs/results/ood_FA.json` ; gate `logs/F/gate_ood_FA.txt`
- surface : `docs/results/surface_F.json` (cumulatif)
- val gelée : `runs/F-A/val_ppl.txt`

## 2026-10-20 20:35 — étape 4, bras B : départ

- B CE seule (lr 1e-3, WSD, WD 0.1->0)
- données : `/Volumes/Lexar/clean-corpus/dedup/tranche_09.jsonl` (tranche_09, mêmes tokens que t09)
- reprise : `runs/E-t08/model.pt` ; 924 steps ; seed 0
- logs : `runs/F-B/train.log`
- fin bras B : exit=0, durée=4777s

## 2026-10-20 22:03 — étape 4, bras B : mesures

- durée mesures : 507s ; log `runs/F-B/measure.log`
- val_big : `docs/results/valbig_FB.json` ; gate `logs/F/gate_valbig_FB.txt`
- OOD : `docs/results/ood_FB.json` ; gate `logs/F/gate_ood_FB.txt`
- surface : `docs/results/surface_F.json` (cumulatif)
- val gelée : `runs/F-B/val_ppl.txt`

## 2026-10-20 22:03 — étape 4, bras C : départ

- C KD 0.7 (lr 1e-3, WSD, WD 0.1->0)
- données : `/Volumes/Lexar/clean-corpus/dedup/tranche_09.jsonl` (tranche_09, mêmes tokens que t09)
- reprise : `runs/E-t08/model.pt` ; 924 steps ; seed 0
- logs : `runs/F-C/train.log`
- fin bras C : exit=0, durée=6924s

## 2026-10-21 00:07 — étape 4, bras C : mesures

- durée mesures : 546s ; log `runs/F-C/measure.log`
- val_big : `docs/results/valbig_FC.json` ; gate `logs/F/gate_valbig_FC.txt`
- OOD : `docs/results/ood_FC.json` ; gate `logs/F/gate_ood_FC.txt`
- surface : `docs/results/surface_F.json` (cumulatif)
- val gelée : `runs/F-C/val_ppl.txt`
- fin campagne : durée totale 21643s

## 2026-10-21 00:07 — étape 4 : campagne terminée (6 h 00 min)

`[step4] TERMINÉ en 21643s`, aucun chien de garde déclenché, aucun NaN,
aucun `.diverged`. Les trois bras et leurs mesures sont complets.

| bras | val-PPL | Δc val_big | Δc OOD | durée train | flips finaux |
|---|---:|---:|---:|---:|---:|
| A | 19,8 | −0,1944 | −0,1714 | 8 315 s | 1,87 % |
| B | **15,8** | **−0,2830** | −0,1471 | 4 754 s | 6,31 % |
| C | 20,4 | −0,1788 | −0,1555 | 6 901 s | 6,51 % |

- A reproduit t09 (−0,1947 / −0,1695) : le reset d'optimiseur n'est pas un
  facteur.
- B (CE seule, lr 1e-3, WSD) améliore fortement val-PPL et val_big, mais
  **régresse OOD** de +0,024 BPC contre t08 : gain in-domain, pas de
  transfert.
- C (KD à lr 1e-3) est pire partout ; flips 6,5-9,5 % contre 1,9 % pour A.
- Critère pré-enregistré : **aucun bras retenu** (A = 19,8 non strictement
  < 19,8 et Δc val_big marginalement moins bon ; B échoue OOD ; C échoue
  partout). **Pas d'étape 5.**

Correction de mesure : le premier `eval_fixed` des bras F avait été lancé en
greedy nu (oubli des flags de déploiement). Les trois bras ont été remesurés
avec `--repetition-penalty 1.15 --no-repeat-ngram 3`, comme t08/pilot-qa3,
puis `eval_surface.py` a été relancé. Les chiffres publiés (OOV 0,67 / 0,74 /
0,72 %, top-1 31,5 / 30,0 / 30,5 %) sont les chiffres de déploiement.
`tools/measure_F_arm.sh` porte désormais les flags.

Documentation : `docs/CHAINE_F.md` (analyse complète), `docs/RESULTS_F.md`
(tableau), `AVANCEMENT.md` (état et décision).
