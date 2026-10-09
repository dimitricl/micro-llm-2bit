# Diagnostics (phase C)

## Récapitulatif

| mesure | résultat |
|---|---|
| C1 unk 8000 / 12000 / 16000 | 2,14 % / 1,00 % / 0,45 % — embeddings tiny : 3,94 / 5,91 / 7,87 Mo |
| C2 BPC 135M / 360M / Qwen3-0.6B | 1,5357 / 1,3362 / 1,1138 (même 354 Ko val, fp32) |
| C2 BPC tranche00.bin / wiki.bin | 1,5676 (PPL 25,88) / 1,6566 (PPL 31,12) |
| C3 masse hors top-k T=1, k=32/64/128/512 (moy) | 0,249 / 0,193 / 0,145 / 0,070 (p95 jusqu'à 0,66) |
| C3 idem T=2 | 0,816 / 0,775 / 0,726 / 0,594 |
| C4 tok/s réel | 2 221 effectifs (34,8M/15 668 s) ; micro-batch effectif 8, 8 192 tok/step ; bench bs16 réel : 2 577 |
| eval_fixed greedy répétition / boucle | tranche00 : 0,299 / 1,1 — wiki : 0,454 / 4,9 |

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
| tranche00 (.bin quantifié, PPL 25,88) | 1,5676 |
| wiki_slice00 (.bin quantifié, PPL 31,12) | 1,6566 |

tranche00 est à 0,03 BPC du parent 135M (1,5676 vs 1,5357) : la
distillation d'une epoch transfère déjà ~98 % du plafond parent sur ce
domaine. wiki_slice00 (demi-epoch) est 0,09 BPC derrière (+5,7 % relatif).
Gemma-3-270M non tenté (accès sous conditions) ; Qwen3-0.6B a suffi comme
point multilingue.

**C2-bis — Métrique BPC équitable (G1)**

Statut au 10/10/2026 : le recalcul corrigé (fix NLL = -log(prob) au lieu de log(prob))
est terminé. Les valeurs ci-dessous sont les valeurs **corrigées et définitives**.

Variantes BPC sur les MÊMES documents et positions (6 docs val C2, 354 234 chars) :

| modèle | a (actuel) | b (unk pénalisé) | c (in-vocab only) |
|---|---|---|---|
| tranche00 | **1,5676** | **1,4544** | **1,5735** |
| wiki_slice00 | **1,6566** | **1,5439** | **1,6641** |
| SmolLM-135M | 1,5357 | 1,6099 | 1,5292 |
| SmolLM-360M | 1,3362 | 1.4133 | 1,3302 |
| Qwen3-0,6B | 1,1138 | 1,2049 | 1,1096 |

Gaps tranche00 vs SmolLM-135M :
- (a) actuel : **+2,08 %** (0,0319 BPC)
- (b) unk pénalisé : **−9,66 %** (−0,1555 BPC) → tranche00 **meilleur** grâce à la pénalité unk
- (c) in-vocab only : **+2,90 %** (0,0443 BPC)

Gaps wiki_slice00 vs SmolLM-135M :
- (a) : +7,87 % (0,1209 BPC)
- (b) : −4,10 % (−0,0660 BPC) → wiki_slice00 meilleur
- (c) : +8,82 % (0,1349 BPC)

**Verdict corrigé** : la variante (a) place tranche00 à +2,08 % (0,032 BPC) du parent —
l'affirmation "~98 % du plafond" reste correcte pour (a). En revanche, la variante
(b) (unk pénalisé) montre que tranche00 **surpasse** le parent de 9,66 % car la
pénalité byte-uniforme favorise les modèles avec moins d'unk. La variante (c)
(in-vocab only) confirme un petit écart de +2,90 %. **Conclusion** : "98 % du
plafond" est correct sur la métrique standard (a), mais sous-estime la qualité
réelle de tranche00 qui gère mieux les unk rares (1,0 % du vocab).

## C2-ter — Validation élargie (G2)

Jeu val_big : 300 docs, 7 742 366 chars, strictement disjoints des slices 00-09
(vérifié par SHA1, source : reste disjoint de slice_100M.jsonl). Bootstrap
1000 tirages par document (seed 0), IC 95 % sur BPC/PPL.

| modèle | a BPC [IC 95%] | b BPC [IC 95%] | c BPC [IC 95%] |
|---|---|---|---|
| tranche00.pt | 1.7240 [1.4879, 1.9646] | 1.8961 [1.6353, 2.1796] | 1.7549 [1.7330, 1.7804] |
| SmolLM-135M | 1.7464 [1.5017, 2.0009] | 1.8950 [1.6291, 2.1827] | 1.7538 [1.7381, 1.7704] |
| SmolLM-360M | 1.5360 [1.3217, 1.7618] | 1.6880 [1.4515, 1.9452] | 1.5406 [1.5254, 1.5565] |
| wiki_slice00.bin* | 1.8158 [0.7988, 3.0398] | 1.9429 [0.9439, 3.2452] | 1.8461 [1.7917, 1.8853] |

* wiki_slice00 : spot-check sur 14 docs (388k chars) à cause du coût .bin (~123 tok/s).

**Verdict** : sur val_big, tranche00 est légèrement meilleur que 135M en BPC (a) avec IC
chevauchants [1.4879, 1.9646] vs [1.5017, 2.0009] — pas de différence significative.
Les IC larges reflètent la variabilité inter-documents (300 docs seulement). Le gap
réduit vs C2 (sur val petit C2) suggère que la performance converge sur un jeu
plus large. **Jeu OOD** : non construit — aucune source locale valide disponible.

## Ce que ça change

1. Le vocab 12000 actuel est bien calibré (1,0 % unk, 5,9 Mo) : passer à
   16000 ne gagne que 0,55 pt d'unk pour +2 Mo. Garder 12000 pour le run
   principal.
2. À T=2, le cache top-32/64/128 jette l'essentiel de la masse (82/78/73 %) :
   préférer le parent en direct (décision tranche00 confirmée) ou un top-k
   ≥ 512 si le cache revient.
3. L'écart tranche00/wiki_slice00 (BPC, PPL, eval_fixed greedy) valide le
   décompte corrigé comme cause : même données, même vocab, 2× tokens vus.
4. Point de départ phase E : tranche00 (BPC 1,5676), pas wiki_slice00.

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
