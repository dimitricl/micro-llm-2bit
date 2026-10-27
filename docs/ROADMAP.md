# Plan de reprise et feuille de route

**Date** : 2026-10-27
**Objet** : proposer la reprise du projet après audit, sans exécuter aucune
opération git écrivante avant le « go » explicite.
**Rappel des contraintes** : entraînement MLX uniquement, inférence C portable
(NEON / AVX2 / scalaire), 100 Mo de RAM totale, conversation FR prioritaire,
code, appels d'outils macOS et Windows.

---

## 1. Décision d'architecture

L'audit conduit à trois choix structurants.

### 1.1 Le ternaire est entraîné, pas appliqué après

Le run MLX actuel entraîne un modèle fp32 de 47,7 M puis l'exporte en int4. Ce
n'est pas un modèle ternaire. La nouvelle chaîne fait du **QAT ternaire dès le
premier step** (absmean + STE), avec des poids latents fp32 et un forward
{−1, 0, +1}.

Conséquence : le run MLX actuel est **abandonné comme modèle**, mais son code et
son cache de tokens sont réutilisés.

### 1.2 Le runtime final est un binaire C

Le moteur Python consomme 225,9 Mo de RSS pour un modèle de 12,13 Mo. Aucun
réglage ne fera tenir Python sous 100 Mo. Le runtime final est un binaire C
autonome, avec :
- chargement mmap du `.bin` ;
- dispatch NEON (arm64), AVX2 (x86_64), scalaire (fallback) ;
- KV cache int8 ;
- tokenizer natif ;
- décodage contraint pour les outils ;
- build CMake multi-plateforme (macOS, Linux, Windows).

Le moteur Python reste comme **référence de parité**, pas comme livrable.

### 1.3 L'objectif est requalifié

« Converser + coder + appeler des outils de façon fiable » n'est pas crédible à
200 M paramètres. L'objectif retenu est :

> **Assistant français compact, conversation simple, complétion de code courte,
> un outil contraint à la fois (macOS et Windows), sous 100 Mo de RAM totale.**

Le function calling généraliste est explicitement **hors périmètre**. Il pourra
être revisité si un modèle ≥ 1 Md tient un jour dans le budget (il n'y tient pas :
~350 Mo).

---

## 2. Plan git (à exécuter seulement après « go »)

### 2.1 Archiver l'état actuel

**Avant tout**, copier les 5 livrables d'audit hors du dépôt (ils seront repris
dans le premier commit de `feat/mlx-100mb` et ne doivent pas être perdus lors du
changement de branche) :

```bash
mkdir -p /tmp/audit-2026-10-27
cp docs/AUDIT.md docs/MLX_MIGRATION.md docs/BUDGET_100MB.md \
   docs/DATA_GAP.md docs/ROADMAP.md /tmp/audit-2026-10-27/
```

Puis :

```bash
# 1. Vérifier l'état
git status
git diff --stat
git log --oneline main..HEAD

# 2. Créer une branche datée depuis la branche courante
git switch -c archive/2026-10-27-hybrid-chat-code

# 3. Commiter TOUT l'état, y compris les modifications non commitées
#    (pas de --force, pas de réécriture d'historique)
git add -A
git commit -m "archive: état du 2026-10-27 (voie hybride MLX non validée)"

# 4. Pousser la branche d'archive
git push -u origin archive/2026-10-27-hybrid-chat-code
```

Points d'attention :
- `runs/hybrid-mlx/` (942 Mo) et les `.bin` sont dans `.gitignore` : ils ne
  seront **pas** commités. C'est voulu. Si l'état du run doit être préservé, le
  copier hors du dépôt (disque externe) avant de nettoyer.
- Ne **jamais** utiliser `--force`, ne jamais réécrire `main`.
- Vérifier qu'aucun secret n'est ajouté (`git diff --cached | grep -i token`).

### 2.2 Créer la branche de travail

```bash
git switch main
git pull --ff-only
git switch -c feat/mlx-100mb

# restaurer les livrables d'audit
cp /tmp/audit-2026-10-27/*.md docs/
```

La nouvelle branche part de `main`, **pas** de l'archive. Elle ne reprend que les
composants classés **GARDER** dans `AUDIT.md` :

| repris | comment |
|---|---|
| `quant.py` | porté en MLX (`mlx/quant.py`), pas copié |
| `distill.py` | porté en MLX (`mlx/distill.py`) |
| `model.py` | spécification, réimplémenté en MLX |
| format `.bin` | conservé, version 2 |
| `kernels/*.c` | repris et étendus |
| `data/probe`, `data/qa`, Wikipédia | données conservées localement |
| tests | repris, étendus |
| docs | `AUDIT.md`, `BUDGET_100MB.md`, `DATA_GAP.md`, `MLX_MIGRATION.md`, `ROADMAP.md` |

Ne sont **pas** repris : `tools/train_hybrid_mlx.py` (réécrit), les données
hybrides mal étiquetées, l'export int4, le `Makefile` macOS-only.

### 2.3 Premier commit de `feat/mlx-100mb`

Contenu exact :

```
mlx/
  __init__.py
  model.py          # Transformer ternaire, config 200M
  quant.py          # absmean + STE + packing
  data.py           # packing, EOS, masque
  train.py          # boucle CE + QAT, split validation
  export.py         # .safetensors -> .bin v2
runtime/
  CMakeLists.txt
  src/main.c
  src/engine.c
  src/tokenizer.c
  src/kernels/neon.c
  src/kernels/avx2.c
  src/kernels/scalar.c
tests/
  test_quant.py
  test_export_parity.py
  test_rss.py        # échoue si RSS > 100 Mo
  test_format.py
docs/
  AUDIT.md
  MLX_MIGRATION.md
  BUDGET_100MB.md
  DATA_GAP.md
  ROADMAP.md
```

Critères du premier commit :
- `pytest -q` vert, < 60 s ;
- `test_rss.py` mesure le binaire C sur `exports/E-t08.bin` et **échoue si
  `VmRSS`/`WorkingSetSize` > 100 Mo** ;
- le binaire C génère 8 tokens identiques au moteur Python (parité) ;
- aucun fichier > 1 Mo, aucun secret.

C'est un commit volontairement petit : il prouve la chaîne (MLX → `.bin` → C →
RSS) avant d'investir dans l'entraînement.

---

## 3. Phases

### Phase 0 — Socle vérifiable (2–3 jours)

| tâche | livrable | critère |
|---|---|---|
| binaire C minimal | `runtime/` | charge E-t08, génère 8 tokens |
| parité Python/C | `test_export_parity.py` | ids identiques |
| mesure RSS totale | `test_rss.py` | < 100 Mo |
| build CMake | `CMakeLists.txt` | macOS + Linux |
| squelette MLX | `mlx/` | forward passe, tests verts |

**Porte 0** : RSS totale < 100 Mo sur E-t08 **et** parité exacte. Sinon, on
s'arrête et on corrige le runtime avant tout entraînement.

### Phase 1 — QAT ternaire MLX (1 semaine)

| tâche | critère |
|---|---|
| modèle 130 M et 200 M en MLX | forward/backward OK |
| QAT absmean + STE | sorties ∈ {−s, 0, +s} |
| split par document | 0 % de 32-grammes train/val |
| run court 50 M tokens | PPL valide, pas de NaN |
| comparaison fp32 vs ternaire | écart PPL < 5 % [seuil] |

**Porte 1** : le QAT converge et l'écart fp32/ternaire reste < 5 % sur 50 M
tokens. Sinon, réduir à 130 M ou revoir le groupe d'échelle.

### Phase 2 — Données (1–2 semaines, en parallèle)

| tâche | cible |
|---|---|
| reconstruction OASST1 par arbre | ≥ 1 M tokens de dialogues |
| génération de dialogues FR par professeur | 20–50 M tokens |
| filtrage + dédup | < 1 % de doublons |
| code instruct multi-langage | 10–20 M tokens |
| trajectoires d'outils | 50 000–200 000 |
| splits propres + manifeste | hash SHA-256 |

**Porte 2** : ≥ 5 M tokens de dialogues FR propres **avant** de lancer le SFT.
En dessous, le SFT est reporté.

### Phase 3 — Pré-entraînement + distillation (2–3 semaines)

| étape | données | durée estimée |
|---|---|---|
| pré-entraînement CE | 300 M tokens × 3 époques | 9–12 jours (200 M) |
| KD top-k=32 | 5–10 M tokens chat/code | 2–4 jours |
| SFT | 5–20 M tokens | 2–3 jours |

Estimation [hypothèse] : 200 M à ~1 130 tok/s → 3,1 j/époque sur 300 M tokens.
Tenir un journal de bord avec la mesure réelle dès le premier jour.

**Porte 3** : sur `OOD`, le modèle ternaire doit être à moins de 10 % de PPL du
meilleur modèle fp32 de taille comparable, et le chat doit battre E-t08 en
préférence humaine sur 100 prompts.

### Phase 4 — Export et runtime C (1 semaine)

| tâche | critère |
|---|---|
| export ternaire → `.bin` v2 | échelles fp16, header documenté |
| noyau NEON | ≥ 40 tok/s à 200 M |
| noyau AVX2 | ≥ 30 tok/s sur x86 (à mesurer réellement) |
| fallback scalaire | correct, lent |
| build Windows | `.exe` testé |
| tokenizer natif | round-trip exact |
| décodage contraint JSON | schéma respecté à 100 % sur 200 cas |

**Porte 4** : RSS < 100 Mo **et** ≥ 30 tok/s sur les deux plateformes.

### Phase 5 — Évaluation et outils (1 semaine)

| évaluation | seuil |
|---|---|
| conversation FR (100 prompts, juge humain) | ≥ 60 % de réponses acceptables |
| code (30 tâches courtes) | ≥ 40 % pass@1 |
| outil macOS (100 cas) | ≥ 90 % de JSON valide, ≥ 80 % d'action correcte |
| outil Windows (100 cas) | idem |
| sonde FR | ≥ 20 % |
| OOD PPL | ≤ 25 |

**Porte 5** : les seuils conversation et outil sont atteints. Sinon, on
requalifie à la baisse (un seul outil, un seul OS) plutôt que de mentir sur la
capacité.

### Phase 6 — Décision (2 jours)

Rapport final avec :
- mesures RSS, tok/s, PPL, chat, code, outils ;
- ce qui marche, ce qui ne marche pas ;
- recommandation go/no-go pour une v1 distribuable.

---

## 4. Calendrier réaliste

| phase | durée | cumul |
|---|---:|---:|
| 0 — socle | 2–3 j | 3 j |
| 1 — QAT | 7 j | 10 j |
| 2 — données | 10 j (parallèle) | 10 j |
| 3 — entraînement | 15–20 j | 30 j |
| 4 — runtime | 7 j | 37 j |
| 5 — évaluation | 7 j | 44 j |
| 6 — décision | 2 j | 46 j |

**~6 à 7 semaines** à temps plein, dominées par l'entraînement et la collecte de
dialogues. Ce calendrier suppose que le professeur et les données sont
disponibles ; tout téléchargement volumineux requiert un « go ».

---

## 5. Budget matériel

| poste | besoin | remarque |
|---|---|---|
| Mac M4 16 Go | entraînement MLX | existant |
| disque | ~50 Go libres | corpus + cache + checkpoints |
| professeur 1,5B | ~3 Go | téléchargement soumis à « go » |
| PC Windows | test AVX2 | **non disponible aujourd'hui** |
| PC Linux/Kali | test AVX2 | **non disponible aujourd'hui** |

L'absence de machine x86 est un risque de planning : le noyau AVX2 ne peut pas
être validé sur cette session. Il faut soit une VM x86, soit un accès distant,
soit accepter que la phase 4 glisse.

---

## 6. Critères go / no-go

### Go si

- RSS totale < 100 Mo mesurée sur le binaire C ;
- parité C/Python exacte ;
- QAT ternaire à moins de 5 % de la PPL fp32 ;
- ≥ 5 M tokens de dialogues FR propres ;
- ≥ 90 % de JSON valide sur les outils contraints ;
- ≥ 60 % de réponses conversation acceptables.

### No-go si

- la RSS dépasse 100 Mo après trois itérations d'optimisation ;
- le QAT s'effondre (PPL > 2× fp32) ;
- les données de dialogue restent < 1 M tokens ;
- le débit tombe sous 20 tok/s à 200 M ;
- le noyau AVX2 ne peut pas être testé.

### Plan B

Si le 200 M ne tient pas la qualité, descendre à **130 M** (47 Mo de poids,
64 Mo au total). On perd en capacité mais on garde la marge, le débit et la
simplicité. C'est la configuration la plus probable pour une v1.

Si les outils restent sous 80 %, **descendre à un seul outil** (`open_app` sur
macOS, `start_process` sur Windows) avec un schéma fixe et un décodage par
grammaire. C'est une démonstration honnête, pas un function calling général.

---

## 7. Risques majeurs

| risque | probabilité | impact | mitigation |
|---|---|---|---|
| données de dialogue insuffisantes | **élevée** | **critique** | génération + filtrage dès la phase 2 |
| QAT instable | moyenne | élevé | run court, courbes, fallback 130 M |
| RSS > 100 Mo | moyenne | élevé | runtime C, KV int8, contexte 1024 |
| AVX2 non testable | élevée | moyen | VM x86 ou test différé |
| temps d'entraînement > 3 semaines | moyenne | moyen | réduire à 130 M |
| contamination des splits | moyenne | élevé | split par document, hash |
| licences du code | moyenne | moyen | filtrage permissif |

---

## 8. Ce qui est explicitement abandonné

- Le run MLX fp32 47,7 M comme modèle livrable.
- L'export int4 post-training.
- Le moteur Python comme runtime final.
- OpenOrca majoritairement anglais.
- Le fichier « javascript » (Python mal étiqueté).
- Le synthétique de 100 exemples.
- Le function calling généraliste sous 100 Mo.
- Le `Makefile` macOS-only.

---

## 9. Verdict honnête

Le projet peut produire, en 6 à 7 semaines, un **assistant français compact
original** : ~130–200 M paramètres ternaires, 60–95 Mo de RAM totale, 30–60 tok/s,
conversation simple, complétion de code courte, un outil contraint par OS.

Il ne peut pas produire, dans ce budget, un assistant généraliste qui converse,
code et appelle des outils de façon fiable. La capacité du modèle et le volume de
données de dialogue ne le permettent pas. Promettre cela serait malhonnête.

La valeur réelle du projet est ailleurs : **un runtime ternaire portable, mmap,
sous 100 Mo, avec un format de poids original et des noyaux NEON/AVX2**. C'est un
actif rare et défendable. Le modèle est la partie interchangeable ; le runtime est
la partie précieuse.

La recommandation est donc :

1. **go** pour la phase 0 (socle C + RSS + parité), qui est peu risquée et
   immédiatement utile ;
2. **go conditionnel** pour les phases 1–3, suspendu à l'atteinte de 5 M tokens
   de dialogues propres ;
3. **no-go** pour toute promesse de function calling généraliste sous 100 Mo.

Le premier commit de `feat/mlx-100mb` doit être petit et prouver la chaîne
complète. C'est le meilleur moyen d'éviter de reproduire l'erreur actuelle :
beaucoup de code, beaucoup de logs, et un budget mesuré au mauvais endroit.
