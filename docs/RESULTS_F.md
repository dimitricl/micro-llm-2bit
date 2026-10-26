# Chaîne F — tableau comparatif

Référence de l'hypothèse : t08, val-PPL gelée 19,8, Δc val_big −0,1949, Δc OOD −0,1708.
Δ BPC = modèle − tranche00 (négatif = meilleur). Gate = verdict apparié `gate_paire.py` (seuil ε = 0,03).
Critère étape 4 : val-PPL < 19,8 **et** Δc val_big et OOD meilleurs que t08.

| modèle | val-PPL | Δb val_big | Δc val_big | gate | Δb OOD | Δc OOD | gate | sonde | OOV N≥5 | « capitain » | logproba top-1 | flips | durée |
|---|---:|---:|---:|:--:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|
| t08 | 19.8 | -0.1893 | -0.1951 | — | -0.1639 | -0.1708 | — | 2.00 % | 0.44 % | 0 | 32.50 % | — | 279 |
| t09 | 19.8 | -0.1889 | -0.1947 | — | -0.1626 | -0.1695 | — | — | — | — | — | — | 125 |
| F-A | 19.8 | -0.1888 | -0.1946 | PASS | -0.1645 | -0.1714 | PASS | 3.50 % | 0.67 % | 0 | 31.50 % | 1.87 % | 138 |
| F-B | 15.8 | -0.2747 | -0.2831 | PASS | -0.1412 | -0.1472 | PASS | 4.00 % | 0.74 % | 0 | 30.00 % | 6.31 % | 79 |
| F-C | 20.4 | -0.1737 | -0.1789 | PASS | -0.1492 | -0.1555 | PASS | 2.50 % | 0.72 % | 0 | 30.50 % | 6.51 % | 115 |

Durée en minutes (journal d'entraînement, hors mesures). Les cases vides correspondent à des mesures non faites pour t09 (pas de sonde ni de surface).

- F-A : critère étape 4 NON ATTEINT.
- F-B : critère étape 4 NON ATTEINT.
- F-C : critère étape 4 NON ATTEINT.
