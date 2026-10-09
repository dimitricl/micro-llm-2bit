# Diagnostics (phase C)

Validation commune : 6 premiers docs val de `slice_00` (split seed 0,
`val_ratio` 0,05), 354 234 caractères, jamais vus à l'entraînement.
Limites communes : textes avec eos ajouté par doc (comme l'entraînement) ;
tokens unk du vocab réduit (1,0 %) inclus dans le calcul (masse réelle du
modèle, pas du parent) ; caractères = points de code Python.

## C1 — Couverture du vocabulaire (val slice_00, 1 711 982 tokens + eos)

| vocab | part unk (hors vrais eos) | embedding tiny int8+scales (Mo) |
|---|---|---|
| 8000 | 2,14 % (36 581) | 3,94 |
| 12000 | 1,00 % (17 115) | 5,91 |
| 16000 | 0,45 % (7 778) | 7,87 |

Coût embedding = V × (512 + 4) octets. Mimicry validée : sélection 12000
== vocab tranche00 à l'identique.

## C2 — Plafond du parent (BPC, même validation)

| modèle | BPC (fp32, même 354 Ko) |
|---|---|
| SmolLM-135M | 1,5357 |
| SmolLM-360M | 1,3362 |
| Qwen3-0.6B (multilingue, téléchargé sans effort : oui) | 1,1138 |
| tranche00 (.bin quantifié) | en cours (moteur CPU, ~25 min) |
| wiki_slice00 (.bin quantifié) | en cours |

Comparabilité tranche00 vs wiki_slice00 : vocabs `kept_ids` IDENTIQUES
(12000, ancien format unk==eos==0), même parent, même config, même split
val (mêmes comptes 10067 docs / 102 973 139 chars des deux logs) → même
validation. Comparaison autorisée.
Limites : BPC parents en fp32 (plafond optimiste ; l'entraînement utilise
fp16) ; .bin en fenêtres non recouvrantes avec reset (approximation
`eval_ppl.py`) ; unk inclus partout.

## C3 — Masse hors top-k (SmolLM-135M, 51 200 positions val)

| T | k=32 (moy / p95) | k=64 | k=128 | k=512 |
|---|---|---|---|---|
| 1 | 0,2488 / 0,6626 | 0,1926 / 0,5606 | 0,1448 / 0,4518 | 0,0701 / 0,2391 |
| 2 | 0,8164 / 0,9563 | 0,7753 / 0,9319 | 0,7256 / 0,8968 | 0,5941 / 0,7825 |

À T=2 (température de distillation), le top-32 ne couvre que ~18 % de la
masse en moyenne : le cache top-32 jette ~82 % de la masse, ce qui éclaire
l'anomalie cache (valCE k32 1,14 vs 0,80 en direct). Le top-512 garde ~93 %
à T=1 mais seulement ~41 % à T=2.

## C4 — Débit GPU réel (décompte corrigé)

Le tok/s des logs compte des tokens RÉELS (`tok_total`, `x.numel`), il
n'est pas affecté par le bug de décompte. Micro-batch effectif tranche00 :
8 (pas 16), soit 8 192 tokens/step (8×8×128) ; total 34,8M / 15 668 s =
2 221 tok/s effectifs (logs ~2 250). Matrice `bench_train.md` (vrais
micro-batch) : tiny/135M = 2 577 tok/s à bs 16.

## eval_fixed — tranche00 vs wiki_slice00 (sorties brutes côte à côte)

Fichiers : `eval/tranche00/samples.md`, `eval/wiki_slice00/samples.md`
(20 prompts, greedy + sampling, ni triés ni corrigés).
Agrégats (détail par prompt dans les fichiers) :

| modèle | greedy répétition-4g | greedy boucle max moy. | sampling répétition-4g |
|---|---|---|---|
| tranche00 (1 epoch) | 0,299 | 1,1 | 0,052 |
| wiki_slice00 (demi-epoch) | 0,454 | 4,9 | 0,010 |

Greedy : l'epoch complète boucle nettement moins. Sampling : équivalents.
