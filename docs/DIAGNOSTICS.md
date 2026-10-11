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

tranche00 est à 0,0319 BPC du parent 135M en variante (b)
et à 0,0443 BPC en variante (c) ; la métrique (a) est affichée
pour continuité mais n'est jamais utilisée comme métrique de
comparaison (elle favorise l'élève, cf. corrélation unk en C2-quater).
wiki_slice00 (demi-epoch) est derrière en (b) et (c)
(voir tableau C2-bis). Gemma-3-270M non tenté (accès
sous conditions) ; Qwen3-0.6B a suffi comme point multilingue.

**C2-bis — Métrique BPC équitable (G1)**

Statut au 10/10/2026 : le recalcul contrôlé avec `uv run` est terminé. Les
valeurs ci-dessous sont les valeurs **corrigées et définitives**.

Variantes BPC sur les MÊMES documents et positions (6 docs val C2, 354 234 chars) :

| modèle | a (actuel) | b (unk pénalisé) | c (in-vocab only) |
|---|---|---|---|
| tranche00 | **1,5676** | **1,6536** | **1,5735** |
| wiki_slice00 | **1,6566** | **1,7431** | **1,6641** |
| SmolLM-135M | 1,5357 | 1,6099 | 1,5292 |
| SmolLM-360M | 1,3362 | 1.4133 | 1,3302 |
| Qwen3-0,6B | 1,1138 | 1,2049 | 1,1096 |

Gaps tranche00 vs SmolLM-135M :
- (a) actuel : **+2,08 %** (0,0319 BPC)
- (b) unk pénalisé : **+2,71 %** (+0,0437 BPC)
- (c) in-vocab only : **+2,90 %** (0,0443 BPC)

Gaps wiki_slice00 vs SmolLM-135M :
- (a) : +7,87 % (0,1209 BPC)
- (b) : −4,10 % (−0,0660 BPC) → wiki_slice00 meilleur
- (c) : +8,82 % (0,1349 BPC)

**Verdict** : les variantes (b) et (c) sont les seules métriques de
comparaison (+2,71 % et +2,90 %, soit +0,0437 et +0,0443 BPC).
La variante (a) est affichée pour continuité mais n'est jamais
utilisée pour comparer (biais pro-élève, cf. C2-quater).

## C2-ter — Validation élargie (S5, gate apparié)

Jeu val_big : 300 docs, 7 742 366 chars, strictement disjoints des slices 00-09
(vérifié par SHA1, source : reste disjoint de slice_100M.jsonl).
Protocole : batch via `.pt` (S=128/doc, seed 0), per-doc persistés
(`docs/results/valbig_{tranche00,smol135,smol360}.json`), bootstrap
APPARIÉ par document (mêmes tirages, estimateur ratio Σbits/Σchars,
1000 tirages, seed 0), Δ = candidat − référence, IC 95 %.
Gate (`tools/gate_paire.py`, ε = 0,03 = 2× MDE mesuré) : FAIL si borne
haute IC95 du Δ > ε sur (b) OU (c) ; (a) affichée, sans veto.
MDE = demi-largeur IC95. Sources : `docs/results/valbig_*.json`.

| modèle | a BPC [IC 95% ind.] | b BPC [IC 95% ind.] | c BPC [IC 95% ind.] |
|---|---|---|---|
| tranche00.pt | 1,7240 [1,4879, 1,9646] | 1,8961 [1,6353, 2,1796] | 1,7549 [1,7330, 1,7804] |
| SmolLM-135M | 1,7464 [1,5017, 2,0009] | 1,8950 [1,6291, 2,1827] | 1,7538 [1,7381, 1,7704] |
| SmolLM-360M | 1,5360 [1,3217, 1,7618] | 1,6880 [1,4515, 1,9452] | 1,5406 [1,5254, 1,5565] |
| wiki_slice00.bin | NON COMPARABLE (spot-check 14 docs/388k chars, .bin token-à-token ; pas le même protocole — ligne conservée pour mémoire, exclue du gate) |

Δ appariés (candidat − référence) :

| paire | Δa [IC95] | Δb [IC95] | Δc [IC95] | MDE b/c | verdict (ε=0,03) |
|---|---|---|---|---|---|
| tranche00 − 135M | −0,0229 [−0,0341, −0,0091] | +0,0005 [−0,0127, +0,0164] | +0,0008 [−0,0127, +0,0173] | 0,0145/0,0150 | PASS + gate FAIBLE |
| tranche00 − 360M | +0,1883 [+0,1770, +0,2022] | +0,2073 [+0,1945, +0,2237] | +0,2140 [+0,2003, +0,2316] | 0,0146/0,0157 | FAIL (b, c) |
| 135M − 360M (contrôle +) | +0,2111 [+0,2084, +0,2138] | +0,2068 [+0,2039, +0,2096] | +0,2132 [+0,2103, +0,2159] | 0,0028/0,0028 | FAIL (b, c) |
| tranche00 − tranche00 (contrôle −) | 0 | 0 | 0 | 0 | PASS |
| tranche00_win32 − tranche00 (contrôle +) | +0,0870 [+0,0849, +0,0895] | +0,0895 [+0,0870, +0,0925] | +0,0788 [+0,0775, +0,0801] | 0,0028/0,0013 | FAIL (b, c) |

Contrôles : wiki_slice00.pt N'EXISTE PAS (seul le .bin existe ; un .pt
demanderait un entraînement, gelé) et tranche00_e0.pt est un doublon
bit-identique du final (même SHA1 des poids) — pas un init. Substitués
honnêtement par (i) 135M vs 360M (écart connu +0,21 → FAIL ✓) et
(ii) ctx32 vs ctx128 (dégradation connue +0,09 → FAIL ✓). Négatif
(tranche00 vs elle-même) → PASS ✓. Aucun seuil retouché après coup :
avec ε=0,01 (pré-mesure) le gate FAIL sur tranche00−135M (bruit,
hi=+0,0164) — c'est cette mesure qui a fixé ε=0,03, pas l'inverse.

**Verdict** : sur val_big (domaine wiki), tranche00 ≈ 135M en (b)/(c)
(Δ +0,0005/+0,0008, IC incluant 0) — MAIS le gate est FAIBLE
(MDE 0,015 > 0,01) : il ne résout que les régressions ≥ 0,03 BPC.
La phrase « le gap réduit suggère que la performance converge » des
versions précédentes est INFIRMÉE par l'OOD (C2-quater-bis) : c'est un
effet domaine (wiki→wiki), pas une convergence.

## C2-quater-bis — Jeu OOD littéraire (H4)

Construction (`tools/make_ood.py`, seed 0) : 3 textes du domaine public
(Gutenberg, licence « Public domain in the USA », auteurs morts > 70 ans),
en-tête/pied Gutenberg retirés (marqueurs START/END), découpage en docs
~8 000 chars sur frontières de paragraphes, ordre des livres conservé.
Disjonction vérifiée par SHA1 normé contre tranches 00-09, slice_100M,
val_big : 0 recoupement. Test : `tests/test_ood.py`.
Sources : `runs/abl_data/ood_manifest.json` (titres, URLs, SHA1
brut/nettoyé) ; docs : `runs/abl_data/ood.jsonl` (340 docs, 2 604 662
chars, ignoré par git, régénérable).

| source | ebook | chars nettoyés | docs |
|---|---|---|---|
| Madame Bovary, Flaubert (https://www.gutenberg.org/ebooks/14155) | 14155 | 683 309 | 89 |
| Vingt mille lieues (complet), Verne (https://www.gutenberg.org/ebooks/5097) | 5097 | 876 607 | 114 |
| Notre-Dame de Paris, Hugo (https://www.gutenberg.org/ebooks/19657) | 19657 | 1 045 623 | 137 |

Résultats (même protocole apparié, sources `docs/results/ood_*.json`) :

| paire OOD | Δa [IC95] | Δb [IC95] | Δc [IC95] | verdict (ε=0,03) |
|---|---|---|---|---|
| tranche00 − 135M | +0,1030 [+0,0967, +0,1094] | +0,1236 [+0,1172, +0,1299] | +0,1292 [+0,1225, +0,1359] | FAIL (b, c) |
| tranche00 − 360M | +0,3324 [+0,3264, +0,3387] | +0,3496 [+0,3433, +0,3559] | +0,3649 [+0,3581, +0,3717] | FAIL (b, c) |

**Verdict** : sur OOD littéraire, tranche00 est à +0,12 BPC (+5,5 % en b)
du parent 135M — contre +0,0005 sur val_big. La quasi-parité val_big ne
généralise PAS hors domaine wiki : tranche00 a sur-ajusté son domaine
d'entraînement. Le parent 135M généralise mieux (+2,23 en b vs 2,35).
Conséquence : la revalidation tranche01 devra inclure l'OOD, pas seulement
val_big.

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

## C2-quater — Parité .bin vs .pt de tranche00 (S3)

Commande : `uv run python tools/check_parity_bin_pt.py
--out docs/results/parity.json` (CPU, ~18 min : moteur token-à-token).
Fenêtrage UNIQUE des deux côtés (S=128 par doc, `make_windows`),
assert d'égalité des longueurs, `.pt` en CPU fp32.
Source : `docs/results/parity.json`.

Résultat (6 docs C2, 118 306 positions) : |NLL_bin − NLL_pt| p50 = 0,0123,
p99 = 0,0967, max = 0,377 (bruit de quantification 2 bits, PAS une
égalité bitwise — un `np.allclose` strict ÉCHOUERAIT : hypothèse
initiale H2.2 infirmée à ce niveau). En revanche l'agrégé compense :
ΔBPC(bin−pt) a = −0,000032, b = −0,000033, c = −0,000033,
soit |ΔBPC| < 0,002 (condition S3) → PASS.
Note : BPC parité (a=1,5661) vs G1 (a=1,5676) : −0,0015 d'écart de
fenêtrage (S=128 par doc avec chutes de queues vs S=512 du G1).

**Conclusion** : le véhicule de mesure (.bin quantifié) ne biaise pas
le BPC agrégé (3e-5 ≪ écarts G1 +0,03/+0,04). L'écart G1 (6 docs) vs
G2 (300 docs) est donc du bruit d'échantillonnage, pas un artefact
d'export. Aucune revendication de pourcentage du plafond (cf. S4).

## C2-quinquies — Biais pro-élève de la variante (a) (S4)

Hypothèse testée : « (a) favorise l'élève car elle ne pénalise pas les
unk ». Sur val_big, par document (modèle tranche00,
source `docs/results/valbig_tranche00.json`) : part de caractères unk
(moyenne 6,2 %, max 48,6 %) vs (a−b) par document (moyenne −0,20).
Pearson = −0,51, Spearman = −0,60 : plus un doc est riche en unk, plus
(b) dégrade (a). Le mécanisme est confirmé : (a) sous-pénalise les
docs à unk, donc flatte les modèles à mauvaise couverture unk.
Comparaisons sur (b)/(c) uniquement ; (a) affichée pour continuité.

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
