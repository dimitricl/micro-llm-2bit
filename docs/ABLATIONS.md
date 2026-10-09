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

| run | variante | seed | steps | tokens vus | val-PPL@100 | val-PPL@200 | val-PPL@300 | commun-128 PPL/BPC | commun-256 PPL/BPC | tok/s | mur (s) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
