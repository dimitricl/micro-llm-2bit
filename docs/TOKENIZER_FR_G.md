# Tokenizer FR — fragmentation et comparaison BPE 16k

SmolLM-135M natif vs vocabulaire réduit E-t08 vs BPE byte-level entraîné sur 40 Mo de `tranche_01`  évalué sur `runs/abl_data/val_E_frozen.jsonl` (821 771 mots).

| tokenizer | vocab | tokens/mot | types fragmentés (pondéré) | tokens unk | mots unk (pondéré) |
|---|---:|---:|---:|---:|---:|
| SmolLM natif | 49 152 | 2.211 | 53.5 % | 0.00 % | 0.00 % |
| SmolLM -> réduit E-t08 | 12 000 | 2.211 | 53.5 % | 0.96 % | 3.47 % |
| BPE FR 16k | 16 000 | 1.629 | 23.9 % | 0.00 % | 0.00 % |
| BPE FR 12k | 12 000 | 1.693 | 26.8 % | 0.00 % | 0.00 % |

## Mots de la confusion

| mot | SmolLM | réduit E-t08 | BPE 16k |
|---|---|---|---|
| **capitale** | `Ġcapit` + `ale` | `Ġcapit` + `ale` | `Ġcapitale` |
| **capitaine** | `Ġcapit` + `aine` | `Ġcapit` + `aine` | `Ġcapitaine` |
| **capitole** | `Ġcapit` + `ole` | `Ġcapit` + `ole` | `Ġcapit` + `ole` |
| **capitalle** | `Ġcapital` + `le` | `Ġcapital` + `le` | `Ġcapital` + `le` |

## 50 mots FR les plus fréquents (extrait)

| mot | freq | SmolLM (n) | BPE 16k (n) |
|---|---:|---|---|
| de | 49756 | `Ġde` (1) | `Ġde` (1) |
| la | 28003 | `Ġla` (1) | `Ġla` (1) |
| le | 20605 | `Ġle` (1) | `Ġle` (1) |
| et | 19565 | `Ġet` (1) | `Ġet` (1) |
| en | 15649 | `Ġen` (1) | `Ġen` (1) |
| les | 13986 | `Ġles` (1) | `Ġles` (1) |
| des | 13392 | `Ġdes` (1) | `Ġdes` (1) |
| du | 12179 | `Ġdu` (1) | `Ġdu` (1) |
| est | 10098 | `Ġest` (1) | `Ġest` (1) |
| un | 8417 | `Ġun` (1) | `Ġun` (1) |
| une | 8024 | `Ġune` (1) | `Ġune` (1) |
| par | 7304 | `Ġpar` (1) | `Ġpar` (1) |
| dans | 7078 | `Ġdans` (1) | `Ġdans` (1) |
| il | 5897 | `Ġil` (1) | `Ġil` (1) |
| au | 5733 | `Ġau` (1) | `Ġau` (1) |
| pour | 5546 | `Ġpour` (1) | `Ġpour` (1) |
| qui | 4728 | `Ġqui` (1) | `Ġqui` (1) |
| sur | 4022 | `Ġsur` (1) | `Ġsur` (1) |
| que | 3710 | `Ġque` (1) | `Ġque` (1) |
| avec | 3247 | `Ġa` `vec` (2) | `Ġavec` (1) |
| plus | 3141 | `Ġplus` (1) | `Ġplus` (1) |
| son | 3019 | `Ġson` (1) | `Ġson` (1) |
| se | 2762 | `Ġse` (1) | `Ġse` (1) |
| sont | 2741 | `Ġs` `ont` (2) | `Ġsont` (1) |
| ou | 2570 | `Ġou` (1) | `Ġou` (1) |
| ce | 2104 | `Ġce` (1) | `Ġce` (1) |
| sa | 2033 | `Ġsa` (1) | `Ġsa` (1) |
| elle | 1890 | `Ġel` `le` (2) | `Ġelle` (1) |
| aux | 1884 | `Ġaux` (1) | `Ġaux` (1) |
| qu | 1876 | `Ġqu` (1) | `Ġqu` (1) |
| comme | 1836 | `Ġcomm` `e` (2) | `Ġcomme` (1) |
| cette | 1658 | `Ġc` `ette` (2) | `Ġcette` (1) |
| pas | 1572 | `Ġpas` (1) | `Ġpas` (1) |
| deux | 1563 | `Ġde` `ux` (2) | `Ġdeux` (1) |
| été | 1482 | `Ġ` `ét` `é` (3) | `ĠÃ©tÃ©` (1) |
| mais | 1463 | `Ġm` `ais` (2) | `Ġmais` (1) |
| ses | 1451 | `Ġs` `es` (2) | `Ġses` (1) |
| ne | 1357 | `Ġne` (1) | `Ġne` (1) |
| entre | 1291 | `Ġent` `re` (2) | `Ġentre` (1) |
| on | 1276 | `Ġon` (1) | `Ġon` (1) |
| après | 1205 | `Ġapr` `ès` (2) | `ĠaprÃ¨s` (1) |
| commune | 1177 | `Ġcommun` `e` (2) | `Ġcommune` (1) |
| fait | 1159 | `Ġf` `ait` (2) | `Ġfait` (1) |
| france | 1144 | `Ġf` `rance` (2) | `Ġfran` `ce` (2) |
| être | 1143 | `ĠÃ` `ª` `tre` (3) | `ĠÃªtre` (1) |
| ont | 1138 | `Ġont` (1) | `Ġont` (1) |
| saint | 1095 | `Ġsaint` (1) | `Ġsaint` (1) |
| même | 1088 | `Ġm` `ê` `me` (3) | `ĠmÃªme` (1) |
| sous | 1073 | `Ġs` `ous` (2) | `Ġsous` (1) |
| où | 1011 | `Ġo` `ù` (2) | `ĠoÃ¹` (1) |

## Lecture

Un BPE FR de 12-16k réduit le nombre de tokens par mot et la fragmentation ; l'écart chiffré indique le gain maximal attendu d'un tokenizer natif, à comparer au coût d'un changement de vocabulaire (embeddings + tête liée à réinit.)

## Limites

- Le BPE est entraîné sur 40 Mo seulement (sous-échantillon) : un entraînement sur les 1,06 Go ferait légèrement mieux.
- Les tokens/mot incluent ponctuation et espaces ; la colonne « types fragmentés » isole la fragmentation des mots.
- Aucun modèle de langage n'est entraîné ici : c'est un estimateur de coût de tokenisation, pas une preuve de gain de qualité.

