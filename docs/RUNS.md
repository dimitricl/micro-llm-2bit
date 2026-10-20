# Runs

État au **2026-10-20**. Convention Δc : `Δ(X−Y) = X−Y` en BPC, **négatif =
X meilleur**. Les val-PPL de la chaîne E et des pilotes QA sont mesurées sur
la même val gelée (500 docs) ; les Δc E sont appariés sur `val_big` et `OOD`.
Sources : `logs/E-heartbeat.md`, `docs/results/`, `docs/ABLATIONS.md`
(§E, §R3-R4, §P3-P7).

## Chaîne E — t01 → t09 (15-17 oct 2026)

Recette : `tiny` (25,0 M), KD SmolLM-135M, α 0.7, T 2.0, lr 3e-4,
bs 8/accum 16, seq 128, seed 0, reprise du run précédent, val gelée,
eval-every 50 (t01) puis 100 (t02-t09).

| run | val-PPL gelée | Δc val_big | Δc OOD | durée | strikes | commit |
|---|---|---|---|---|---|---|
| t01 | 25,8 | −0,0711 | −0,0557 | 25 220 s | 0 | `eb839b2` |
| t02 | 23,8 | −0,1065 | −0,0978 | 17 672 s | 0 | `f967fd9` |
| t03 | 22,5 | −0,1351 | −0,1180 | 17 590 s | 0 | `fa7937b` |
| t04 | 21,6 | −0,1544 | −0,1381 | 12 465 s | 0 | `5b15aeb` |
| t05 | 21,0 | −0,1678 | −0,1490 | 19 596 s | 0 | `abef3bf` |
| t06 | 20,4 | −0,1837 | −0,1523 | 17 067 s | 0 | `3a6b148` |
| t07 | 20,1 | −0,1904 | −0,1588 | 17 913 s | 0 | `13bbbb1` |
| **t08** | **19,8** | **−0,1949** | **−0,1708** | 16 780 s | 0 | `20fde41` |
| t09 | 19,8 | −0,1945 | −0,1695 | 7 555 s | 0 | `481ca3e` |

- **Meilleur checkpoint : t08.** t09 a la même val mais un Δc légèrement
  moins bon ; la PPL et les gains OOD plafonnent à t08.
- Launchers : `tools/run_E_t01.sh` (t01), puis
  `bash tools/run_E_tranche.sh 02 runs/E-t01` (NN = 02…09, PREVDIR = run
  précédent). Commandes exactes : commit `08b3465`.
- Export : `runs/E-t08/model.bin`, **12,13 Mo** (12 125 100 octets).

## Ablations R2 / R3 / R4 — référence et statut

| run | config | val-PPL | Δc val_big | Δc OOD | durée | launcher | commit | statut |
|---|---|---|---|---|---|---|---|---|
| R2 | CE, lr 3e-4, seed 0, reprise tranche00 | **22,9** | réf. | réf. | 1 305 s | `tools/run_resume.sh` | `66547f5` | **référence CE** |
| R3 | KD SmolLM-135M, reprise tranche00 | 28,0 | **+0,0795** | **−0,0795** | 2 269 s | `tools/run_r3_original.sh` | `10ae6ad` | **KD validée** (espace SmolLM) |
| R3-Qwen | KD Qwen3-0.6B, reprise tranche00 | 15,3* | — | — | — | commande §D5.e | `5617507` | **invalidé** (tokenizer/vocab) |
| R4 | KD Qwen3-0.6B, from-scratch, vocab Qwen | 201,0 | **+0,6847** | **+0,5724** | 3 884 s | pas de script versionné (§R4) | `addd611` | **rejeté** (D3) |

\* R3-Qwen : val-PPL in-run **auto-cohérente dans l'espace d'indices Qwen**,
non comparable à R2 (22,9, espace SmolLM) ; mismatch tokenizer 46,45 % de
couverture, 0 % d'alignement sémantique. Le run est **hors comparaison D3**.

- Références Δc : R2 `c = 1,6495` (val_big) et `2,3641` (OOD). MDE :
  R3 0,0049 / 0,0035 ; R4 0,0107 / 0,0048 (`docs/ABLATIONS.md` §R3-R4).
- R3 valide la KD (OOD −0,0795 < −δ_bruit −0,0072 ; perte val_big +4,8 %
  ≤ 50 %). R4 est rejeté (OOD +0,5724 ≫ −δ_bruit).
- **R2 est la référence CE** ; R3 est la seule validation KD à espace et
  budget constants. R2b (seed 1) est un contrôle de bruit, non listé ici.

## Décodeur — balayage C0 → C5 sur t08

Protocole : 32 prompts fixes, CPU/NEON, mêmes seeds ; `repet4g` = taux de
répétition de 4-grammes ; diversité = mots distincts moyens ; boucle max =
plus longue boucle. Launcher : `logs/run_Et08_decode.sh` ; commit `800b9ea`.

| cfg | réglage | repet4g moy | mots distincts moy | boucle max | statut |
|---|---|---|---|---|---|
| C0 | 1,0 / 0 (greedy nu) | **0,5057** | 10,8 | 3 | interdit au déploiement |
| C1 | pén. 1,15 | 0,3275 | 13,6 | 3 | insuffisant |
| C2 | pén. 1,30 | 0,2083 | 16,5 | 3 | insuffisant |
| C3 | no-repeat-3g seul | 0,0000 | 19,8 | 3 | boucle moyenne 1,62 |
| **C4** | **pén. 1,15 + 3g** | **0,0000** | **20,0** | **2** | **défaut de déploiement** |
| C5 | sampling T=0,7 | 0,0484 | 20,6 | 3 | référence sampling |

C4 supprime la répétition sans détruire la diversité (10,8 → 20,0 mots
distincts) et fait baisser le pire cas (3 → 2). Voir §P3.

## Pilotes QA — mémoire factuelle sur 25 M

Même sonde gelée (`d9817afb…`), même décodeur (greedy pen 1,15 + 3g),
`max_new=32`. Δ val-PPL est relatif à t08 (19,8).

| pilote | corpus | global | reformulé | val-PPL Δ | « capitain » | EOS | durée | verdict | commit |
|---|---|---|---|---|---|---|---|---|---|
| #1 | 60 faits ×19, 7 % général | **81/200 = 40,5 %** | **33/100 = 33,0 %** | 23,1 (**+16,7 %**) | 44 | 7/200 | 3 181 s | **VERT** | `13057b6` |
| #2 | 534 faits ×4, 35 % général | 21/200 = 10,5 % | 7/100 = 7,0 % | 21,2 (+7,07 %) | 82 | 15/200 | 3 183 s | **MITIGÉ** | `0722254` |
| #3 | 534 faits ×10, 22 % général | **72/200 = 36,0 %** | **20/100 = 20,0 %** | 22,0 (**+11,11 %**) | 59 | 9/200 | 13 650 s | **PARTIEL** | `461622a` |

- Launchers : `tools/run_pilot_qa.sh`, `tools/run_pilot_qa2.sh`,
  `tools/run_pilot_qa3.sh`.
- #1 : 60 faits (capitales + unités), reformulé 33 %, mais PPL +16,7 % et
  corruption de surface. #2 : faits insuffisants (7 %), langue meilleure.
  #3 : faits au plancher (20,0 %, IC95 [13,34 ; 28,88]), PPL +11,11 %
  (> seuil +10 %), « capitaine » 59 > 22 → **PARTIEL** (§P7).
- Le corpus #3 est **gelé** : `data/qa/qa_fr_pilot_v3.jsonl`, SHA256
  `6a1ab5b0…`, 24 838 144 tokens, 22,09 % de général.

## Sonde — baseline t08

| modèle | global | par_coeur | reformule | EOS |
|---|---|---|---|---|
| t08 | **4/200 = 2,0 %** [0,78 ; 5,03] | 3/100 = 3,0 % | 1/100 = 1,0 % | 1/200 |

Contrôle pré-E `exports/tranche00.bin` : 0/200. Sonde gelée :
`data/probe/probe_fr_v1.jsonl`, SHA256 `d9817afb…` ; harnais
`tools/eval_probe.py` ; commit baseline `d0a98df`. Détail §P4.
