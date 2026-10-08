# Bench entraînement

## Matrice tok/s (30 steps après 5 warmup, ~2 min max/config)

| enfant | parent | bs | tok/s | parent ms | enfant ms | loss ms | backward ms | optim ms | RAM max Mo | note |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny | SmolLM-135M | 2 | 555 | 41 | 44 | 1.4 | 31 | 18.1 | 1273 |  |
| tiny | SmolLM-135M | 4 | 1888 | 71 | 61 | 2.4 | 54 | 18.0 | 1327 |  |
| tiny | SmolLM-135M | 8 | 2311 | 131 | 95 | 4.6 | 105 | 18.2 | 1371 |  |
| tiny | SmolLM-135M | 16 | 2577 | 255 | 172 | 8.3 | 205 | 18.3 | 1396 |  |
| tiny | SmolLM-360M | 2 | 900 | 82 | 43 | 1.5 | 30 | 18.0 | 1172 |  |
| tiny | SmolLM-360M | 4 | 1420 | 151 | 60 | 2.4 | 54 | 18.0 | 1172 |  |
| tiny | SmolLM-360M | 8 | 1641 | 291 | 95 | 4.5 | 104 | 18.1 | 1174 |  |
| tiny | SmolLM-360M | 16 | 1748 | 578 | 174 | 8.4 | 205 | 18.6 | 1176 |  |
| base | SmolLM-135M | 2 | 666 | 42 | 124 | 1.5 | 69 | 54.1 | 1214 |  |
| base | SmolLM-135M | 4 | 990 | 72 | 162 | 2.5 | 127 | 55.0 | 1228 |  |
| base | SmolLM-135M | 8 | 1239 | 132 | 244 | 4.7 | 249 | 55.6 | 1239 |  |
| base | SmolLM-135M | 16 | 1407 | 254 | 412 | 8.4 | 487 | 55.2 | 1251 |  |
| base | SmolLM-360M | 2 | 503 | 82 | 125 | 1.5 | 69 | 54.4 | 849 |  |
| base | SmolLM-360M | 4 | 828 | 151 | 163 | 2.5 | 128 | 54.9 | 983 |  |
| base | SmolLM-360M | 8 | 1008 | 295 | 243 | 4.7 | 248 | 54.4 | 994 |  |
| base | SmolLM-360M | 16 | 1086 | 587 | 423 | 8.6 | 490 | 55.4 | 923 |  |

## Autocast parent (tiny, 360M, bs 2)

| mode | tok/s | logits finis | note |
|---|---|---|---|
| fp32 | 869 | oui |  |
| fp16 | 1093 | oui |  |
| bf16 | 934 | oui |  |

## Cache top-32 vs parent direct (tiny, phrases réelles, même init, même ordre)

Comparaison honnête : 120 steps jumeaux (direct vs cache), val-CE finale :
- direct : valCE 0.799 (PPL 2.2), 50s
- cache k32 : valCE 1.138 (PPL 3.1), 27s
- cache k64 : valCE 1.064 (PPL 2.9), 27s

Écart de loss (valeurs) : +86 % à k32, +17 % à k=512 — la KL tronquée
renormalisée est structurellement différente, et surtout elle entraîne
moins bien (val 35-40 % moins bonne à steps égaux). k64 ≈ k32 : pas de
convergence vers le direct. DÉCISION : parent en direct (règle des 5 %
non satisfaite, même à k=64).
Vitesse : lecture cache 135x plus rapide que le forward parent, step
complet caché 1843 tok/s — mais au prix d'un moins bon modèle. Non retenu.

## Recommandation : enfant tiny, parent SmolLM-135M, micro-batch 16 (2577 tok/s)

Tranche 100 Mo (~25M tokens) : 2.7h/epoch.

