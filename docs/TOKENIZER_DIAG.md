# Diagnostic tokenizer — confusion « capitale » / « capitaine »

Vocabulaire : `exports/E-t08.bin.vocab.json` (12000 tokens ; unk distinct=False, unk_new=0). Tokenizer : `HuggingFaceTB/SmolLM-135M`.

## Verdict court

Les trois formes correctes (capitale, capitaine, capitole) partagent leurs 2 premiers sous-tokens (`cap + it`) ; elles ne diffèrent que par un unique sous-token final (`ale` / `aine` / `ole`). La distinction repose donc sur un seul token, ce qui rend la confusion PLAUSIBLE. `capitalle` (faute) est encore plus fragile : son premier morceau `capital` est hors vocabulaire et retombe sur l'id unk/eos. Le découpage n'explique pas tout à lui seul ; l'étape 3 (surface) et l'étape 4 (recette) trancheront.

## Les quatre mots de la confusion

| mot | sous-mots SmolLM | ids réduits | unk ? |
|---|---|---|---|
| **capitale** | `cap` + `it` + `ale` (3) | [3676, 69, 101] | non |
| **capitaine** | `cap` + `it` + `aine` (3) | [3676, 69, 299] | non |
| **capitole** | `cap` + `it` + `ole` (3) | [3676, 69, 353] | non |
| **capitalle** | `capital` + `le` (2) | [0, 37] | oui |

Préfixe de sous-tokens partagé par les formes correctes : **oui** (longueur 2, `cap + it`).

Occurrences dans le corpus d'entraînement dédupliqué (tranches 00-09) : **capitale** 23 497, **capitaine** 12 699, **capitole** 1 073, **capitalle** 2. La forme correcte est plus fréquente que la forme corrompue : la corruption n'est donc pas un simple effet de fréquence lexicale.

## Taux globaux (jeu analysé)

- mots analysés : **54**
- mots découpés en > 1 sous-token : **27 / 54 = 50.0%**
- mots avec au moins un token hors vocabulaire (unk) : **2 / 54 = 3.7%**
- tokens hors vocabulaire : **2 / 92 = 2.17%**
- mots découpés à préfixe partagé : **13 / 27 = 48.1%** des mots découpés

## Mots à préfixe partagé (premier sous-token identique)

- `d` : dans, dont, début
- `cap` : **capitale**, **capitaine**, **capitole**
- `p` : pour, pays
- `v` : valeur, vaut
- `qu` : qui, qu
- `com` : comme, compte

## Les 50 mots fréquents du corpus QA

| mot | n | sous-mots | pieces | unk |
|---|---:|---|---|---|
| la | 1 | 1 | `la` | non |
| de | 1 | 1 | `de` | non |
| est | 1 | 1 | `est` | non |
| le | 1 | 1 | `le` | non |
| du | 1 | 1 | `du` | non |
| un | 1 | 1 | `un` | non |
| en | 1 | 1 | `en` | non |
| on | 1 | 1 | `on` | non |
| une | 1 | 1 | `une` | non |
| des | 1 | 1 | `des` | non |
| et | 1 | 1 | `et` | non |
| pour | 2 | 2 | `p` + `our` | non |
| les | 1 | 1 | `les` | non |
| par | 1 | 1 | `par` | non |
| lieu | 2 | 2 | `lie` + `u` | non |
| qui | 2 | 2 | `qu` + `i` | non |
| date | 1 | 1 | `date` | non |
| comme | 2 | 2 | `com` + `me` | non |
| sur | 1 | 1 | `sur` | non |
| eu | 1 | 1 | `eu` | non |
| retient | 2 | 2 | `ret` + `ient` | non |
| au | 1 | 1 | `au` | non |
| il | 1 | 1 | `il` | non |
| qu | 1 | 1 | `qu` | non |
| dans | 2 | 2 | `d` + `ans` | non |
| se | 1 | 1 | `se` | non |
| que | 1 | 1 | `que` | non |
| année | 2 | 2 | `ann` + `ée` | non |
| valeur | 3 | 3 | `v` + `ale` + `ur` | non |
| vaut | 2 | 2 | `v` + `aut` | non |
| correspond | 2 | 2 | `cor` + `respond` | non |
| compte | 3 | 3 | `com` + `pt` + `e` | non |
| situe | 2 | 2 | `sit` + `ue` | oui |
| sa | 1 | 1 | `sa` | non |
| pays | 2 | 2 | `p` + `ays` | non |
| ville | 1 | 1 | `ville` | non |
| dont | 2 | 2 | `d` + `ont` | non |
| porte | 2 | 2 | `port` + `e` | non |
| nom | 1 | 1 | `nom` | non |
| mot | 1 | 1 | `mot` | non |
| début | 3 | 3 | `d` + `é` + `but` | non |
| référence | 4 | 4 | `ré` + `f` + `é` + `rence` | non |
| retenir | 3 | 3 | `re` + `ten` + `ir` | non |
| plan | 1 | 1 | `plan` | non |
| chef | 2 | 2 | `che` + `f` | non |
| son | 1 | 1 | `son` | non |
| chimique | 3 | 3 | `ch` + `im` + `ique` | non |
| administratif | 3 | 3 | `administ` + `rat` + `if` | non |
| guerre | 2 | 2 | `gu` + `erre` | non |
| eau | 2 | 2 | `e` + `au` | non |

## Lecture

Le mécanisme pertinent n'est pas « le mot est inconnu » mais « le mot est connu en morceaux ». Si `cap` + `it` sont des tokens fréquents et que seuls `ale` / `aine` / `ole` changent, le modèle doit apprendre une distinction fine sur un unique sous-token, dans un espace d'embeddings de 512 dimensions partagé par des milliers de mots. C'est exactement le type de détail que le budget de 25 M paramètres peut sous-représenter, et qu'un surapprentissage des templates QA peut écraser au profit de la forme la plus fréquente (`capitaine`).

## Limites

- Le comptage des mots se fait sur les documents QA (templates + général), pas sur les tokens du modèle : c'est un classement de fréquence lexicale, suffisant pour choisir le jeu de mots.
- Le partage de préfixe est mesuré au premier sous-token SmolLM. Il ne prouve pas la confusion ; il montre que le modèle ne dispose que du dernier sous-token pour trancher.
- Le taux d'unk porte sur le vocabulaire réduit E-t08 (12 000 tokens). Un mot découpé sans unk reste représentable ; il peut néanmoins être mal placé dans l'espace des embeddings.

