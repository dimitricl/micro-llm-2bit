# Audit technique — `micro-llm-2bit`

**Date** : 2026-10-27
**Périmètre** : audit en lecture seule du dépôt, de ses mesures et de la voie MLX
non commitée. Aucune modification de code, aucun commit, aucun entraînement lancé
par cet audit.
**Objet** : établir ce qui, dans l'existant, peut servir à construire un LLM
ternaire capable de converser en français, de compléter du code et d'appeler des
outils, sous **100 Mo de RAM totale**, avec une inférence C portable (NEON, AVX2,
scalaire) sans Python ni MLX.

---

## 1. Résumé exécutif

Le dépôt contient une brique d'inférence ternaire sérieuse et déjà mesurée, mais
il ne contient **pas** de modèle conversationnel, **pas** de function calling, et
sa chaîne d'entraînement est encore PyTorch. La voie MLX en cours est une
réécriture parallèle qui ne respecte ni le ternaire ni la distillation : elle ne
peut pas être considérée comme une base validée.

Trois constats dominent :

1. **Le budget annoncé n'est pas le budget demandé.** Le dépôt mesure un
   *delta* RSS de +28,0 Mo pour E-t08, mais la RSS absolue du processus est de
   **225,9 Mo**. Le critère historique « < 100 Mo » porte sur le delta, pas sur
   la RAM totale. En l'état, l'artefact consomme **2,26× le budget total**.
2. **Les données conversationnelles sont quasi inexistantes.** Le corpus
   exploitable contient **892 tokens** de dialogues manuels, 37 k tokens de
   synthétique de mauvaise qualité et 226 k tokens de messages OASST1 isolés.
   Le reste est du Wikipédia français (264 M tokens), utile pour la langue, pas
   pour le dialogue ni les outils.
3. **Le cœur expérimental est honnête mais petit.** 25,0 M paramètres, val-PPL
   19,8, Δc OOD −0,1708, 119 tok/s, 0 % en QA 0-shot. C'est un bon modèle de
   langue français compact, pas un assistant.

La conclusion de faisabilité est détaillée dans `docs/ROADMAP.md` : un modèle
ternaire de ~200 M paramètres tient dans 100 Mo **à condition d'écrire un runtime
C natif**. En revanche, « converser + coder + appeler des outils de façon
fiable » à cette taille n'est pas crédible. La conversation FR basique et la
complétion de code sont atteignables ; le function calling ne peut être qu'une
démonstration contrainte, pas une capacité générale.

---

## 2. Méthode et limites

### 2.1 Ce qui a été mesuré dans cet audit

| Mesure | Méthode | Résultat |
|---|---|---|
| Tests | `pytest -q` | **176 passed, 1 skipped, 2 deselected** en 104,88 s |
| Paramètres E-t08 | calcul exact depuis le header `.bin` | **25 027 072** |
| Ventilation `.bin` | offsets du format `LLM2` | 12 125 100 octets (détail §4.3) |
| Tokens du corpus | tokenizer réel, échantillonnage 1 ligne / 97 | 264,2 M tokens Wikipédia FR |
| Tokens des fichiers hybrides | idem, échantillonnage | table dans `DATA_GAP.md` |
| Comportement du tokenizer | encodage/décodage aller-retour | round-trip exact, 12 tokens pour 39 caractères accentués |
| État git | `git status`, `git log`, `git branch -a` | 6 commits d'avance, 2 modifiés, 8 untracked |
| Avancement MLX | log `logs/hybrid-full.nohup` | step 35 000/36 980, loss 2,8708, 4 729 tok/s |

### 2.2 Ce qui n'a pas pu être mesuré

- **Débit et RSS de la voie MLX** : un entraînement MLX tourne déjà (PID 20115)
  et aucun « go » GPU n'a été donné. Toute mesure concurrente serait polluée.
- **Comportement AVX2 réel** : aucun PC Windows/Kali accessible depuis cette
  session. Le noyau `kernels/avx2.c` n'a jamais été exécuté sur un vrai x86 ici.
- **Qualité du futur modèle 200 M** : aucune hypothèse de qualité n'est
  vérifiable sans entraînement. Les chiffres de capacité sont des extrapolations
  explicites.
- **RSS d'un runtime C** : le runtime n'existe pas. Les chiffres de la section
  budget sont des formules, pas des mesures.

### 2.3 Convention de lecture

Tout chiffre est étiqueté :
- **[mesuré]** : produit par une commande exécutée dans cette session ou lu dans
  un log committed du dépôt (la source est citée) ;
- **[calculé]** : dérivé arithmétiquement d'un chiffre mesuré, formule visible ;
- **[hypothèse]** : extrapolation ou choix d'ingénierie non validé.

---

## 3. État du dépôt

### 3.1 Git

```
branche courante : feature/hybrid-chat-code
6 commits d'avance sur main
main = origin/main (0/0)
feature/think-mode : 1 commit propre, 105 commits de retard
```

Fichiers modifiés non commités : `docs/HYBRID_GUIDE.md`, `tools/train_hybrid.py`.
Fichiers untracked : `tools/train_hybrid_mlx.py`, `tools/export_hybrid_mlx.py`,
`tools/prepare_hybrid_sft_mlx.py`, `tools/run_hybrid_mlx_full.sh`,
`tools/watch_hybrid_mlx.py`, `runs/hybrid-mlx/` (942 Mo), trois logs
`logs/hybrid-*.nohup`.

Les 6 commits sont des commits de génération de code (infrastructure hybride,
correctifs de dataset, timeout Ollama, `--parent`). Ils ne contiennent ni modèle
entraîné ni résultats validés. **Ils ne doivent pas être la base de la nouvelle
branche** ; ils constituent un travail parallèle à archiver.

### 3.2 Hygiène

- Aucun secret détecté dans l'historique git lors de l'audit initial.
- Plus gros fichier tracké : `models/tokenizer_fr_16k/tokenizer.json`
  (1 122 623 octets, soit 1,07 Mo — juste au-dessus de la limite de 1 Mo
  évoquée dans les contraintes).
- `runs/hybrid-mlx/` (942 Mo) et les checkpoints (191 Mo pièce) ne doivent
  jamais être commités.

### 3.3 Tests

`176 passed, 1 skipped, 2 deselected` en 104,88 s. La couverture réelle est
inconnue (pas de `pytest-cov` configuré). Les tests portent sur le format,
l'export, la quantification, le cache KV et la parité PyTorch/NumPy ; **aucun
test ne couvre le function calling, le chat, le RSS absolu ou le runtime C
standalone**, pour la raison simple que ces briques n'existent pas.

---

## 4. Architecture existante

### 4.1 Modèle (`model.py`)

Transformer décodeur de type Llama :
- RMSNorm pré-norm ;
- GQA (8 têtes Q, 4 têtes KV sur `tiny`) ;
- RoPE ;
- SwiGLU ;
- embeddings liés à la tête de sortie (`lm_head` partagé).

Deux configurations documentées :

| config | d | L | Hq | Hkv | Dh | ffn | vocab | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tiny` (E-t08) | 512 | 8 | 8 | 4 | 64 | 1024 | 12 000 | **25,03 M** [calculé] |
| `base` | 768 | 12 | 8 | 4 | 96 | 1536 | 12 000 | **72,9 M** [mesuré] |

La config `base` est celle du banc `checkpoints/bench_base.log` :
`V=12000 d=768 L=12 Hq=8 Hkv=4 ffn=1536 seq=256`, soit `head_dim=96`.

Le calcul exact pour E-t08 :
```
embeddings       = 12 000 × 512            =  6 144 000
par couche       = q 512×512 + k 512×256 + v 512×256 + o 512×512
                 + gate 512×1024 + up 512×1024 + down 1024×512
                 + 2 normes 512
                 = 2 360 320
8 couches        = 18 882 560
norme finale     = 512
TOTAL            = 25 027 072
```

`memory_report()` annonce un budget « poids + KV + activations » mais **ignore
le runtime Python, les temporaires NumPy et le cache d'allocateur**. C'est
l'origine de l'écart entre 12,7 Mo théoriques et 28,0 Mo de delta mesuré.

### 4.2 Quantification (`quant.py`)

- Quantification ternaire par **absmean** : `w_q = round(w / (mean(|w|) + eps))`,
  bornée à {−1, 0, +1} ; les valeurs nulles sont conservées.
- Straight-through estimator (STE) pour le gradient.
- Packing 2 bits : 4 poids par octet, ordre déterministe.
- Référence `ternary_matmul` en PyTorch pour la parité.
- Échelle par groupe de 64 colonnes (`group_size=64`), stockée en fp32 dans le
  `.bin` actuel.

Le principe est bon et réutilisable. Le point faible est le **coût des échelles** :
4 octets par 64 poids = 0,5 bit/poids, soit 20 % du poids des linéaires. Passer
les échelles en fp16 (2 octets) ferait tomber ce coût à 0,25 bit/poids.

### 4.3 Format `.bin` (`LLM2`)

Header `struct "<4s10I"` = 44 octets :
`magic, version, vocab, d, L, Hq, Hkv, ffn, seq_max, group, head_dim`.

Pour E-t08 (`V=12000, d=512, L=8, Hq=8, Hkv=4, ffn=1024, seq=128, group=64,
head_dim=64`), ventilation exacte :

| bloc | octets | part |
|---|---:|---:|
| header | 44 | 0,0004 % |
| embeddings int8 | 6 144 000 | **50,67 %** |
| échelles d'embedding fp32 | 48 000 | 0,40 % |
| linéaires packés 2 bits | 4 718 592 | 38,92 % |
| échelles de linéaires fp32 | 1 179 648 | 9,73 % |
| normes fp32 (17 × 512) | 34 816 | 0,29 % |
| **total** | **12 125 100** | 100 % |

Deux enseignements :
- **Les embeddings pèsent la moitié du fichier.** À vocabulaire 12 k et d=512,
  ils sont en int8 (1 octet/param). C'est le premier levier de budget.
- Les linéaires sont à **2,5 bits/poids** (2 bits + échelles fp32 group 64),
  pas à 2 bits. Passer les échelles en fp16 donne 2,25 bits/poids [calculé].

### 4.4 Moteur (`engine.py`)

- Lecture mmap, vues zéro-copie sur le blob.
- KV cache pré-alloué **int8** `[L, S, Hkv*Dh]` + échelles fp32 `[L, S, Hkv]`.
- Logits calculés par blocs de 2048 lignes pour éviter une matrice fp32 complète.
- Fallback NumPy si la `.dylib` est absente.
- **Le moteur reste un programme Python.** Il charge NumPy, alloue des
  temporaires, et ne peut pas tenir dans 100 Mo de RSS total.

### 4.5 Noyaux C

| fichier | lignes | rôle | état |
|---|---:|---|---|
| `kernels/ternary.h` | 31 | interface commune | propre |
| `kernels/neon.c` | 98 | matvec int8 × ternaire, NEON | **exécuté** sur M4 |
| `kernels/avx2.c` | 83 | matvec int8 × ternaire, AVX2 | **jamais exécuté** ici |
| `kernels/portable.c` | 69 | fallback C99 | compilable, non mesuré |

Le `Makefile` ne produit qu'une `.dylib` macOS, avec un choix de source à la
compilation. Il n'y a **ni CMake, ni build Windows, ni binaire standalone, ni
dispatch runtime NEON/AVX2/scalaire**. Le noyau AVX2 est un actif à préserver,
mais il n'a aucune preuve d'exécution.

### 4.6 Distillation (`distill.py`, 1110 lignes)

Points solides :
- perte KL + CE avec température ;
- cache top-k des logits professeur ;
- reprise sur checkpoint ;
- écriture atomique des checkpoints ;
- garde NaN.

C'est la brique la plus mature du dépôt après le format. Elle est en PyTorch et
devra être portée en MLX, mais **l'algorithme et les garde-fous sont à garder**.

---

## 5. Résultats expérimentaux : ce qui est prouvé, ce qui ne l'est pas

### 5.1 Chaîne E/F — le socle

| modèle | val-PPL gelée | clean32 PPL | Δc OOD | sonde | QA 0-shot | tok/s | Δ RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E-t08** | **19,8** | 22,09 | **−0,1708** | 2,0 % | 0 % | 107–125 | +28,0 Mo |
| F-A (KD) | 19,8 | — | −0,1714 | 3,5 % | — | — | — |
| F-B (CE, WSD) | **15,8** | **18,11** | −0,1471 | 4,0 % | — | — | — |
| F-C (KD, WSD) | 20,4 | — | −0,1555 | 2,5 % | — | — | — |

Lecture honnête :
- E-t08 est un **bon modèle de langue français** pour sa taille.
- La distillation ne bat pas la CE seule sur la val-PPL (F-C 20,4 vs F-B 15,8) ;
  elle ne se justifie que si elle apporte du **format conversationnel**, ce qui
  n'a pas été mesuré.
- La sonde à 2,0 % signifie que le modèle ne restitue pas des faits simples.
- QA 0-shot à 0 % : ce n'est pas un assistant.

### 5.2 Chaîne G — QA extractive

G1 (QA-passage) atteint **35,0 % EM** contre 20,2 % pour le contrôle, soit
+14,8 pp. C'est réel, mais c'est un modèle **extractif** : il recopie un span du
passage. Cela ne prouve aucune capacité conversationnelle ni de raisonnement.

G5 (46 M) obtient un bon BPC OOD (1,9240 à S=128) et 0 % en QA 0-shot. Même
conclusion : bon modèle de langue, pas un assistant.

### 5.3 Contamination — le point critique

- `val_big` : **47,80 %** des 32-grammes présents dans le train. Les chiffres de
  val-PPL sur `val_big` sont donc optimistes.
- `val_E_frozen` : **53,6 %** des documents contaminés.
- `OOD` : **0/340** de contamination. C'est **la seule mesure propre**.
- Les décisions doivent être prises sur `OOD` et `clean32`, pas sur `val_big`.

Ce point est correctement documenté dans `docs/CHAINE_G.md`, ce qui est à
l'honneur du projet, mais il invalide une partie des comparaisons historiques.

---

## 6. La voie MLX en cours (travail parallèle non commité)

### 6.1 Ce qui tourne

```
tools/train_hybrid_mlx.py --stage pretrain
  data      : /Volumes/Lexar/clean-corpus/dedup + data/hybrid
  tokenizer : models/tokenizer_fr_16k
  batch 16, seq 512, 1 époque, lr 3e-4, wd 0.1
  modèle    : d=640, L=10, Hq=8, Hkv=4, ffn=1280, vocab 16000
  params    : 47,7 M
  tokens    : 302,9 M (cache uint16)
  steps     : 36 980
  débit     : ~4 730 tok/s
  état      : step 35 000, loss 2,8708, ppl 17,65
```

### 6.2 Écarts avec l'objectif

| objectif | état MLX | verdict |
|---|---|---|
| Poids ternaires | **fp32 dense**, export int4 post-training | **non conforme** |
| Distillation | CE seule, aucun professeur | **non conforme** |
| Split de validation | aucun | **non conforme** |
| Données FR propres | 30,5 M tokens OpenOrca à 97 % anglais + 5,2 M tokens de Python mal étiqueté | **non conforme** |
| Format chat | SFT 1 583 exemples, 259 593 tokens supervisés | **insuffisant** |
| EOS cible | absent | **bug** |
| Padding | id 0 | **risque** |
| Export | int4 post-training, pas de ternary QAT | **non conforme** |

### 6.3 Ce qui est réutilisable

- Le chargeur de données et le cache de tokens (302,9 M tokens déjà tokenisés).
- La boucle MLX `value_and_grad`, le scheduler, la reprise sur checkpoint.
- Le script d'export `.safetensors` → `.bin` (à adapter au ternaire).
- Le script de préparation SFT (à corriger : EOS, masque, format).

### 6.4 Ce qui doit être jeté

- L'idée qu'un entraînement CE from-scratch en fp32 puis une quantification int4
  post-training produise un modèle ternaire équivalent. Ce n'est pas le cas :
  le QAT ternaire change la dynamique d'entraînement et doit être fait pendant
  l'entraînement, pas après.
- Le mélange OpenOrca majoritairement anglais pour un modèle FR-first.
- Le fichier `javascript_code.jsonl` (voir `DATA_GAP.md`).

---

## 7. Données — synthèse

Le détail est dans `docs/DATA_GAP.md`. Résumé :

| bloc | tokens | qualité | usage |
|---|---:|---|---|
| Wikipédia FR dédupliqué | 264,2 M | bonne langue, 99 % FR | pré-entraînement |
| QA FR (`qa_fr_pilot_v3`) | 18,1 M | extractif | post-training QA |
| OpenOrca | 30,5 M | **97 % anglais** | à filtrer ou abandonner |
| Code « javascript » | 5,2 M | **Python mal étiqueté** | à réétiqueter |
| OASST1 FR | 0,23 M | messages isolés | pool d'instructions |
| Synthétique | 0,037 M | **mauvaise qualité** | abandonner |
| Chat manuel | **0,000892 M** | 10 dialogues | insuffisant |
| **Total exploitable dialogue** | **< 0,3 M** | | **le goulot d'étranglement** |

Le déséquilibre est le fait majeur : **264 M tokens de langue contre moins de
0,3 M tokens de dialogue**. Aucun modèle ne devient assistant avec ce ratio.

---

## 8. Function calling : absent

Recherche exhaustive dans le dépôt : aucun schéma d'outil, aucun token spécial
`<tool>`, aucune grammaire contrainte, aucun décodage JSON structuré, aucun
exemple de trace d'appel. Le seul format existant est
`Utilisateur :` / `Assistant :` dans le SFT MLX.

Conséquence : le function calling n'est pas « à améliorer », il est **à créer
entièrement** : format, tokens, données, décodage contraint, runtime, tests.
C'est le chantier le plus lourd du projet, et celui qui a le moins de marge dans
100 Mo.

---

## 9. Classement des composants

### 9.1 GARDER

| composant | raison |
|---|---|
| `quant.py` (absmean, STE, packing) | algorithme correct, à porter en MLX |
| `distill.py` (KL+CE, top-k, reprise, garde NaN) | brique mature, à porter en MLX |
| `model.py` (RMSNorm, GQA, RoPE, SwiGLU, tied embeddings) | spécification d'architecture validée |
| format `.bin` `LLM2` | compact, mmap-able, documenté |
| `kernels/ternary.h`, `neon.c`, `portable.c` | base de noyaux à étendre |
| `kernels/avx2.c` | actif à conserver, à tester sur x86 |
| `data/qa`, `data/probe`, Wikipédia FR | données propres et utiles |
| suite de tests | 176 tests verts, à étendre |
| documentation `CHAINE_G.md`, `PLAN_G.md` | rigueur méthodologique (contamination, critères pré-enregistrés) |

### 9.2 RÉÉCRIRE

| composant | raison |
|---|---|
| boucle d'entraînement | PyTorch → MLX, avec validation et QAT ternaire |
| export | MLX → `.bin`, ternaire, échelles fp16, métadonnées de chat |
| tokenizer | ajouter EOS/PAD/outils, vérifier le coût des accents, vocab 16k vs 12k |
| pipeline de données | filtrage langue, dédup, format chat, EOS, masque de perte |
| moteur d'inférence | C standalone, mmap, dispatch NEON/AVX2/scalaire |
| build | CMake multi-plateforme (macOS, Linux, Windows) |
| banc | RSS **absolue**, pas delta ; tok/s ; latence ; mémoire par étape |
| format de chat | rôles, EOS, séparateurs, tokens spéciaux |

### 9.3 ABANDONNER

| composant | raison |
|---|---|
| `feature/hybrid-chat-code` comme base | 6 commits non validés, données mal étiquetées, pas de ternaire |
| `data/hybrid/javascript_code.jsonl` | 500 fichiers **Python** étiquetés JavaScript |
| `data/hybrid/openorca_instructions.jsonl` | 97 % anglais, 30,5 M tokens, contraire au FR-first |
| `data/hybrid/synthetic.jsonl` | lignes coupées, répétitions, 100 exemples |
| `data/hybrid/oasst1_fr.jsonl` **comme dialogues** | messages isolés, pas des conversations |
| run MLX CE from-scratch | ne produit ni ternaire ni assistant |
| export int4 post-training | ne remplace pas le QAT ternaire |
| Python comme runtime final | incompatible avec 100 Mo de RSS total |
| `Makefile` macOS-only | ne couvre ni Windows ni Linux |
| `memory_report()` comme preuve de budget | ignore le runtime et les temporaires |

---

## 10. Risques et angles morts

1. **Risque de budget** : le projet croit respecter 100 Mo alors qu'il mesure un
   delta. C'est le risque n°1, car il peut invalider tout le reste.
2. **Risque de données** : 264 M tokens de langue pour 0,3 M de dialogue. Sans
   distillation massive ou génération de dialogues, le modèle restera un modèle
   de langue.
3. **Risque de capacité** : 200 M paramètres ternaires, c'est petit. Le function
   calling fiable demande typiquement ≥ 1 Md. À 200 M, il faut accepter une
   capacité étroite et contrainte.
4. **Risque de généralisation** : `val_big` est contaminé à 47,8 %. Toute
   décision prise sur cette métrique est suspecte.
5. **Risque x86** : le noyau AVX2 n'a jamais tourné ici. Le jour où il tourne, il
   peut révéler des bugs de packing ou d'alignement.
6. **Risque de reproductibilité** : la voie MLX n'a pas de split de validation,
   pas de seed de données documentée, pas de hash de corpus.
7. **Risque juridique** : les licences des données (Wikipédia CC BY-SA,
   Gutenberg domaine public US, codeparrot licences hétérogènes dont GPL) doivent
   être vérifiées avant toute distribution.

---

## 11. Verdict honnête

Le dépôt n'est pas un projet d'assistant : c'est un **excellent projet de modèle
de langue français compact avec un runtime ternaire original**. La partie
inférence (format, noyaux, KV int8, mmap) est le vrai actif. La partie
conversation et outils est un chantier neuf.

Sur les trois capacités demandées :

- **Converser en français** : atteignable à ~200 M paramètres, à condition
  d'investir massivement dans les données de dialogue et la distillation. Le
  résultat sera un assistant simple, pas un généraliste.
- **Compléter du code** : atteignable pour un ou deux langages, en complétion
  courte. Pas de génération de projet.
- **Appeler des outils de façon fiable** : **non crédible à 200 M paramètres**.
  Une démonstration contrainte (un outil, schéma fixe, décodage par grammaire)
  est possible ; un function calling général ne l'est pas.

Sur le budget : **100 Mo de RAM totale est atteignable** avec un runtime C natif
et un modèle de ~200 M paramètres ternaires (voir `BUDGET_100MB.md`). Ce n'est
**pas** atteignable avec le moteur Python actuel, qui consomme 225,9 Mo.

La recommandation est donc de **poursuivre**, mais en requalifiant l'objectif :
« assistant français compact, code court, un outil contraint », et en abandonnant
l'ambition d'un function calling général sous 100 Mo. Le plan de reprise est dans
`docs/ROADMAP.md`.
