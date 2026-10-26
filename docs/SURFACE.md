# Métrique de surface F — hors lexique et classement log-proba

Sonde gelée : `data/probe/probe_fr_v1.jsonl` (SHA256 `d9817afb15d8…`, vérifié). Lexique : corpus d'entraînement dédupliqué `/Volumes/Lexar/clean-corpus/dedup/tranche_*.jsonl` ; seuil primaire N>=5.

## Taux de mots hors lexique

| run | source | seuil | mots | hors lexique | taux | types OOV | « capitain » |
|---|---|---|---:|---:|---:|---:|---:|
| t08 | eval_fixed_greedy | 5 | 882 | 2 | **0.23%** | 2/201 (1.0%) | 0 |
| t08 | eval_fixed_sampling | 5 | 858 | 5 | **0.58%** | 5/324 (1.5%) | 0 |
| t08 | sonde | 5 | 3026 | 14 | **0.46%** | 10/350 (2.9%) | 0 |
| t08 | combine | 5 | 4766 | 21 | **0.44%** | 16/622 (2.6%) | 0 |
| t08 | eval_fixed_greedy | 20 | 882 | 2 | **0.23%** | 2/201 (1.0%) | 0 |
| t08 | eval_fixed_sampling | 20 | 858 | 5 | **0.58%** | 5/324 (1.5%) | 0 |
| t08 | sonde | 20 | 3026 | 21 | **0.69%** | 15/350 (4.3%) | 0 |
| t08 | combine | 20 | 4766 | 28 | **0.59%** | 21/622 (3.4%) | 0 |
| pilot-qa3 | eval_fixed_greedy | 5 | 752 | 18 | **2.39%** | 15/227 (6.6%) | 11 |
| pilot-qa3 | eval_fixed_sampling | 5 | 747 | 13 | **1.74%** | 12/300 (4.0%) | 8 |
| pilot-qa3 | sonde | 5 | 2671 | 77 | **2.88%** | 63/537 (11.7%) | 59 |
| pilot-qa3 | combine | 5 | 4170 | 108 | **2.59%** | 84/794 (10.6%) | 78 |
| pilot-qa3 | eval_fixed_greedy | 20 | 752 | 19 | **2.53%** | 16/227 (7.0%) | 11 |
| pilot-qa3 | eval_fixed_sampling | 20 | 747 | 14 | **1.87%** | 13/300 (4.3%) | 8 |
| pilot-qa3 | sonde | 20 | 2671 | 85 | **3.18%** | 68/537 (12.7%) | 59 |
| pilot-qa3 | combine | 20 | 4170 | 118 | **2.83%** | 91/794 (11.5%) | 78 |
| F-A | eval_fixed_greedy | 5 | 855 | 4 | **0.47%** | 3/206 (1.5%) | 0 |
| F-A | eval_fixed_sampling | 5 | 807 | 3 | **0.37%** | 3/317 (0.9%) | 0 |
| F-A | sonde | 5 | 2696 | 22 | **0.82%** | 20/368 (5.4%) | 0 |
| F-A | combine | 5 | 4358 | 29 | **0.67%** | 26/646 (4.0%) | 0 |
| F-A | eval_fixed_greedy | 20 | 855 | 4 | **0.47%** | 3/206 (1.5%) | 0 |
| F-A | eval_fixed_sampling | 20 | 807 | 4 | **0.50%** | 4/317 (1.3%) | 0 |
| F-A | sonde | 20 | 2696 | 27 | **1.00%** | 21/368 (5.7%) | 0 |
| F-A | combine | 20 | 4358 | 35 | **0.80%** | 28/646 (4.3%) | 0 |
| F-B | eval_fixed_greedy | 5 | 732 | 5 | **0.68%** | 5/209 (2.4%) | 0 |
| F-B | eval_fixed_sampling | 5 | 840 | 4 | **0.48%** | 4/361 (1.1%) | 0 |
| F-B | sonde | 5 | 2624 | 22 | **0.84%** | 19/437 (4.3%) | 0 |
| F-B | combine | 5 | 4196 | 31 | **0.74%** | 28/773 (3.6%) | 0 |
| F-B | eval_fixed_greedy | 20 | 732 | 5 | **0.68%** | 5/209 (2.4%) | 0 |
| F-B | eval_fixed_sampling | 20 | 840 | 4 | **0.48%** | 4/361 (1.1%) | 0 |
| F-B | sonde | 20 | 2624 | 23 | **0.88%** | 20/437 (4.6%) | 0 |
| F-B | combine | 20 | 4196 | 32 | **0.76%** | 29/773 (3.8%) | 0 |
| F-C | eval_fixed_greedy | 5 | 874 | 7 | **0.80%** | 7/192 (3.6%) | 0 |
| F-C | eval_fixed_sampling | 5 | 772 | 6 | **0.78%** | 6/299 (2.0%) | 0 |
| F-C | sonde | 5 | 2910 | 20 | **0.69%** | 16/332 (4.8%) | 0 |
| F-C | combine | 5 | 4556 | 33 | **0.72%** | 27/584 (4.6%) | 0 |
| F-C | eval_fixed_greedy | 20 | 874 | 7 | **0.80%** | 7/192 (3.6%) | 0 |
| F-C | eval_fixed_sampling | 20 | 772 | 8 | **1.04%** | 8/299 (2.7%) | 0 |
| F-C | sonde | 20 | 2910 | 25 | **0.86%** | 21/332 (6.3%) | 0 |
| F-C | combine | 20 | 4556 | 40 | **0.88%** | 34/584 (5.8%) | 0 |

Le taux est un taux de tokens-mots (pas de types). `combine` = eval_fixed greedy + sampling + sonde. « capitain » compte les occurrences de la chaîne, pas seulement le mot exact.

## Classement log-proba : 1 bonne réponse vs 3 distracteurs

Hasard = 25 %. Score = log-proba moyen du candidat conditionné au prompt (moteur `.bin`, mêmes prompts gelés). Les distracteurs sont tirés dans la même catégorie, de façon déterministe (seed + id).

| modèle | top-1 (moy.) | top-1 (somme) | MRR | rang moyen | par_coeur | reformule |
|---|---:|---:|---:|---:|---:|---:|
| t08 | **32.5%** | 35.5% | 0.565 | 2.37 | 35.0% | 30.0% |
| pilot-qa3 | **40.0%** | 46.5% | 0.623 | 2.14 | 43.0% | 37.0% |
| F-A | **31.5%** | 35.0% | 0.566 | 2.35 | 36.0% | 27.0% |
| F-B | **30.0%** | 30.5% | 0.558 | 2.35 | 32.0% | 28.0% |
| F-C | **30.5%** | 31.5% | 0.552 | 2.40 | 33.0% | 28.0% |

### Par catégorie

| modèle | capitales | dates | definitions | unites_sciences |
|---|---:|---:|---:|---:|
| t08 | 31.2% | 40.0% | 27.5% | 32.5% |
| pilot-qa3 | 37.5% | 65.0% | 22.5% | 37.5% |
| F-A | 28.7% | 40.0% | 25.0% | 35.0% |
| F-B | 25.0% | 42.5% | 22.5% | 35.0% |
| F-C | 30.0% | 30.0% | 27.5% | 35.0% |

## Lecture

- Le taux hors lexique mesure la propreté de surface : un modèle qui invente des mots peut avoir une PPL correcte sur les tokens fréquents.
- Le classement log-proba teste une compétence plus fine que la génération libre : la bonne réponse est-elle mieux notée que trois réponses de la même catégorie ? Un score proche de 25 % signifie que le modèle ne sait pas discriminer, même sans produire la réponse.
- Les deux mesures sont descriptives : elles ne remplacent ni la val gelée ni la sonde exact-match.

## Limites

- Le lexique vient de Wikipédia FR : il contient des noms propres et du vocabulaire technique. Il est donc permissif (sous-estime l'OOV).
- Le découpage en mots ignore les apostrophes et la ponctuation : « l'eau » compte deux mots (`l`, `eau`).
- Le classement log-proba dépend de la tokenisation du candidat et de la formulation du prompt ; il est comparable entre modèles, pas comme une probabilité absolue.
- Les distracteurs viennent de la sonde elle-même (pas d'un jeu externe) : ils sont plausibles mais peuvent être trop faciles pour certaines catégories (dates, unités).

