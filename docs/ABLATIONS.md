# Ablations slice_00 (tiny, parent SmolLM-135M)

Protocole commun : `tiny`, parent `HuggingFaceTB/SmolLM-135M`, vocab partagé
à unk distinct (`runs/abl_data/vocab.json`, 12000, construit sur le subset),
même init `--seed 0` (sauf V0b, seed 1 : mesure du bruit), même ordre de
données (seed epoch `(seed, epoch)`), validation in-run tous les 50 steps
(`--eval-every 50`, val = 5 % du subset, identique pour tous SAUF V0b et
V1b — voir notes), 5M tokens par run sauf V1b (temps égal V0).
Données : `runs/abl_data/train_5M.jsonl` (806 docs, ~5,2M tokens) ;
jeux communs post-hoc : val 128 + val 256 (mêmes 503 docs val slice_00).
V1b : 9 epochs du subset (≈ 47M tokens, budget temps = mur de V0a).
Comparaison headline = jeux communs post-hoc ; les colonnes 100/200/300 =
val-PPL in-run (dynamique). Bruit = |V0a − V0b| sur jeux communs : toute
différence inférieure n'est pas un résultat.

| run | variante | seed | steps | tokens vus | val-PPL@100 | val-PPL@200 | val-PPL@300 | BPC@100 | BPC@200 | BPC@300 | commun-128 PPL/BPC | commun-256 PPL/BPC | tok/s | mur (s) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0-seed0 | baseline KD | 0 | 305 | 4997120 | 537.2 | 187.2 | 142.3 | 3.3499 | 2.7882 | 2.6422 | 161.87/2.6038 | 161.14/2.6015 | 2372 | 2145 | val in-run = 5% subset (68 docs); jeux communs post-hoc |
| V0-seed1 | baseline KD, seed bruit | 1 | 303 | 4964352 | 554.5 | 210.6 | 160.5 | 3.2737 | 2.7721 | 2.6315 | 170.94/2.6317 | 170.00/2.6289 | 2298 | 2200 | val in-run = split seed 1 (non comparable point à point) |
| V1a-alpha0 | CE seule, memes tokens | 0 | 305 | 4997120 | 391.1 | 140.9 | 110.4 | 3.1808 | 2.6368 | 2.5067 | 129.65/2.4902 | 127.84/2.4830 | 4507 | 1146 | sans parent (B6), ~2x plus vite |
| V1b-temps-egal | CE seule, temps egal V0 | 0 | 610 | 9996160 | 440.2 | 121.4 | 73.0 | 3.2438 | 2.5575 | 2.2860 | 59.93/2.0952 | 59.85/2.0945 | 4254 | 2387 | 2 epochs subset, mur ≈ V0 (2145s) |
| V2-alpha03 | alpha 0.3 | 0 | 305 | 4997120 | 400.0 | 146.7 | 113.8 | 3.1927 | 2.6584 | 2.5228 | 132.01/2.4994 | 130.42/2.4932 | 2296 | 2210 | entre V0 et V1a |
| V3-lr1e3 | lr 1e-3 | 0 | 305 | 4997120 | 245.7 | 82.8 | 63.9 | 2.9331 | 2.3533 | 2.2151 | 72.47/2.1924 | 72.78/2.1946 | 2465 | 2064 | pas de divergence, meilleur à 5M |
| V4-subln | subln (normes o/down) | 0 | 305 | 4997120 | 424.3 | 167.1 | 129.9 | 3.2242 | 2.7276 | 2.5933 | 148.51/2.5597 | 145.85/2.5504 | 2334 | 2179 | légèrement mieux que V0, à confirmer vs bruit |
| V5-seq256 | seq 256, 16384 tok/step | 0 | 305 | 4997120 | 528.8 | 192.0 | 146.6 | 3.3396 | 2.8001 | 2.6564 | 174.19/2.6413 | 165.63/2.6155 | 2351 | 2165 | fenêtres longues moins bonnes à tokens égaux |

Bruit inter-runs (jeux communs) : |V0a − V0b| = 9,07 PPL (128) / 8,86 (256),
soit ~0,028 BPC. Toute différence inférieure n'est pas un résultat.

## Analyse (écart au bruit, jeu commun-128)

- V1a (CE seule, mêmes tokens) : 129,65 vs 161,87 → Δ −32,2 ≫ bruit.
  RÉSULTAT : à 5M tokens, CE seule > KD α0,7 (lr 3e-4).
- V1b (CE seule, temps égal, ~10M tokens) : 59,93. Budget différent :
  pas de comparaison directe, mais confirme le débit CE (~2×).
- V2 (α0,3) : 132,01, soit Δ +2,4 vs V1a (< bruit : équivalent CE) et
  Δ −29,9 vs V0 (≫ bruit). RÉSULTAT : baisser alpha aide ; 0,3 ≈ 0 à 5M.
- V3 (lr 1e-3) : 72,47, Δ −89 ≫ bruit, aucune divergence (loss finie,
  val monotone). RÉSULTAT FORT : le LR compte plus qu'alpha à 5M.
- V4 (subln) : 148,51, Δ −13,4 vs V0a (bruit 9,07). Signal faible,
  au-dessus du bruit mais non répliqué : PAS DE RÉSULTAT revendiqué,
  pas d'adoption (export non supporté sans accord).
- V5 (seq 256) : 174,19 (Δ +12,3 vs V0a) et 165,63 en 256
  (Δ +4,5 < bruit). RÉSULTAT : aucun bénéfice des fenêtres longues à
  tokens égaux ; rester en seq 128.
- Le jeu 256 confirme le même ordre que le 128 (V3 < V1b < V1a ≈ V2 <
  V4 < V0a ≈ V0b ≈ V5).
