# G — heartbeat

Journal chronologique de la chaîne G. Les résultats complets sont dans
`docs/CHAINE_G.md` ; ce fichier garde les commandes, l'ordre des opérations et
les échecs, y compris ceux qui ont été corrigés.

Machine : Mac mini M4, 16 Go, 10 cœurs, disque Lexar. Un seul job GPU/MPS à la
fois. Phase 0 (diagnostic) terminée ; phase 1 (G0–G1) **terminée** le 2026-10-21.

## 2026-10-21 — phase 0

### 09:35 — 0.1 flags d'évaluation

But : vérifier que les mesures de t08 ne dépendent pas des flags de décodage
utilisés par les bras F.

```bash
cd /Volumes/Lexar/micro-llm-2bit
# même protocole que tools/measure_F_arm.sh, sortie dédiée G
.venv/bin/python tools/eval_fixed.py \
  --model exports/E-t08.bin \
  --out eval/G-Et08-fixed \
  --repetition-penalty 1.15 --no-repeat-ngram 3
```

Résultat : `eval/G-Et08-fixed/metrics.json` bit-identique à
`eval/F-Et08-fixed/metrics.json` (sha1 `cd2394de…`, échantillons `ff487c70…`).
Log : `logs/G/step0-fixed-Et08.log`.

### 09:40 — 0.5 flips cumulés

```bash
.venv/bin/python tools/flips_cumul_G.py
```

Résultat : B 20,12 % de poids ternaires différents de t08, 0,06 % de
retournements directs ±1→∓1 ; A 6,87 %/0,00 % ; C 15,75 %/0,01 %.
Log : `logs/G/step0-flips.log`.

### 09:41 — 0.3 tokenizer

```bash
.venv/bin/python tools/tokenizer_fr_g.py
```

Résultat : BPE16k 1,629 token/mot (vs SmolLM 2,211), 23,9 % de types
fragmentés (vs 53,5 %) ; `capitale` et `capitaine` deviennent un seul token.
Log : `logs/G/step0-tokenizer.log` ; synthèse : `docs/TOKENIZER_FR_G.md`.

### 09:42 — 0.1 IC bootstrap OOV

```bash
.venv/bin/python tools/oov_bootstrap_G.py
```

Résultat : t08 0,44 % [0,26–0,64] ; F-A +0,22 pp (p=0,121), F-B +0,30 pp
(p=0,088), F-C +0,28 pp (p=0,057). Aucun écart significatif.
Log : `logs/G/step0-oov-bootstrap.log`.

### 09:46–10:08 — 0.2 n-grammes : échec puis réécriture

Première version : fusion `np.unique(np.concatenate(parts))` des 9 tranches.
Pour n=8, le tableau concaténé fait ~2,3 Go et `np.unique` en garde plusieurs
copies ; la machine est passée à **8,4 Go de swap**, le processus n'a rien
affiché pendant plus de 15 min. Processus arrêté.

Seconde version : ne jamais matérialiser l'union. Pour chaque fenêtre de
validation, recherche dichotomique dans chaque tranche (une tranche chargée à
la fois), puis OU logique. Exact, pic mémoire ~300 Mo.

```bash
.venv/bin/python tools/ngram_overlap.py 2>&1 | tee logs/G/step0-ngram.log
```

Résultat principal :

- `val_big` vs t01–t09 : **47,80 %** des 32-grammes ; `val_E_frozen` : 3,53 % ;
  `ood` : **0,00 %**.
- `val_E_frozen` vs t00 = 100 % : contrôle positif attendu (les 500 docs
  viennent de t00, jamais entraînée).
- Ventilation : `val_big` recoupe t04/t06 jusqu'à 8,7–8,9 % à n=32, mais t09
  seulement 0,13 %. `ood` est à 0,00 % sur toutes les tranches à n=32.

Ventilation par tranche : script ad hoc réutilisant les mêmes caches
(`load_docs`, `hashes`, `searchsorted`), sortie conservée dans
`logs/G/step0-ngram-pertranche.log` ; le détail de la commande est dans le
document de résultats.

Logs : `logs/G/step0-ngram.log`, `logs/G/step0-ngram-pertranche.log` ;
résultats : `docs/results/ngram_overlap.json`,
`docs/results/ngram_overlap_pertranche.json`.

### 10:17–10:20 — sous-ensembles clean32 et contrôle de F-B

Filtrage ad hoc avec les caches n=32 de `tools/ngram_overlap.py` : un document
est retenu si **aucune** de ses fenêtres n=32 n'est présente dans l'union
t01–t09. Sorties : `val_E_frozen_clean32.jsonl` (232/500) et
`val_big_clean32.jsonl` (23/300). Le script de génération n'a pas été
conservé ; les fichiers et leurs SHA256 font foi (voir `docs/CHAINE_G.md`).

```bash
.venv/bin/python tools/eval_valbig.py \
  --ckpt runs/E-t08/model.pt \
  --docs runs/abl_data/val_E_frozen_clean32.jsonl \
  --vocab exports/E-t08.bin.vocab.json \
  --out docs/results/vb_clean32_Et08.json

.venv/bin/python tools/eval_valbig.py \
  --ckpt runs/F-B/model.pt \
  --docs runs/abl_data/val_E_frozen_clean32.jsonl \
  --vocab exports/E-t08.bin.vocab.json \
  --out docs/results/vb_clean32_FB.json
```

Résultat : t08 PPL 22,09 [21,23–23,03] ; F-B 18,11 [17,33–19,03]. Le gain
in-domain de B survit au nettoyage (−18 % de PPL, IC disjoints). `val_big`
clean32 ne garde que 23 docs : trop peu pour remplacer val_big.
Logs : `logs/G/step0-clean-val.log`, `logs/G/step0-clean32-Et08.log`,
`logs/G/step0-clean32-FB.log`.

### 10:22 — 0.4 bench

```bash
for f in exports/E-t08.bin exports/enfant_base_fr.bin; do
  .venv/bin/python main.py bench --model "$f" --bench-tokens 128
done
```

Résultat (deux mesures, même commande, machine au repos) :

| modèle | tok/s | Δ RSS après génération | poids |
|---|---:|---:|---:|
| E-t08 | 107,4–125,1 | +28,3–28,4 Mo | 12,13 Mo |
| base 768 | 46,6–54,9 | +52,0–52,1 Mo | 29,25 Mo |

La RSS absolue est 237–262 Mo (Python/numpy) ; le budget est le **delta**
après chargement/génération. Aucun entraînement, un seul modèle chargé à la
fois. Log de la seconde mesure : `logs/G/step0-bench.log`.

### 10:25–10:40 — rédaction

```bash
# documents complets, pas de diffs
docs/RECHERCHE_G.md
docs/CHAINE_G.md
docs/PLAN_G.md
logs/G-heartbeat.md
```

### Tests

```bash
.venv/bin/python -m pytest -q
```

Résultat : **175 passed, 1 skipped, 2 deselected** en 86,61 s
(`logs/G/pytest-G.log`). Aucun test de la chaîne G n'ajoute de dépendance ;
`tools/ngram_overlap.py` a été réécrit après l'échec mémoire.

## État à la fin de la phase 0

- Aucun modèle G entraîné ; `runs/E-t08` reste la référence.
- Phase 0 close : flags, OOV, n-grammes, clean32, tokenizer, bench, flips.
- Découverte majeure : `val_big` est contaminé à 47,8 % (32-grammes) ;
  l'OOD est propre à n=32. Le critère principal de la phase 3 devient
  OOD + clean32 + QA, pas val_big.
- Plan phase 3 pré-enregistré dans `docs/PLAN_G.md`.
- **Aucun run phase 3 lancé** : accord utilisateur requis.

---

## 2026-10-21 — phase 1 (G0–G1)

### 11:00 — G0 : construction du banc QA-passage

```bash
.venv/bin/python tools/build_qa_passage_G.py
```

Résultat : 300 items test (150 clean32 + 150 OOD) + 100 items dev, templates
disjoints, réponse toujours sous-chaîne du passage. SHA256 :
`g_passage_test_v1` = `60be2f1b…`, `g_passage_dev_v1` = `5ce2d55b…`.
Audit 50 items : 49 valides (98 %), 1 ambigu.

### 11:15 — baselines QA (greedy, rep-pen 1.15, no-repeat 3)

```bash
.venv/bin/python tools/eval_qa_passage_G.py \
  --items data/qa/g_passage_test_v1.jsonl \
  --model E-t08=exports/E-t08.bin \
  --model F-B=exports/F-B.bin \
  --model base768=exports/enfant_base_fr.bin
```

Résultat : E-t08 0,0 % EM, F-B 0,0 % EM, base 768 0,0 % EM (base répond sans
refus mais n'atteint pas la réponse exacte). Plancher ≤ 5 % respecté.

### 11:30 — G1 : génération corpus QA train (t01–t09)

```bash
.venv/bin/python tools/build_qa_train_G.py --target-tokens 30300000
```

Résultat : 181 979 exemples, 30,30 M tokens de contenu, 180 703 fenêtres
packées = 924 steps calibrés.

### 11:45–13:30 — G1-qa-s0 (seed 0, QA avec question)

```bash
.venv/bin/python main.py train \
  --qa-data runs/G1-data/qa_train.jsonl --mask-prompt answer --qa-steps 924 \
  --parent HuggingFaceTB/SmolLM-135M --config tiny \
  --out runs/G1-qa-s0/model.pt --resume runs/G1-init-step0.pt \
  --vocab-from runs/E-t08/model.pt.vocab.json \
  --seq-len 256 --epochs 1 --batch-size 8 --accum 16 --alpha 0 \
  --top-k-cache 0 --ckpt-every 0 --lr 3e-4 --weight-decay 0.01 --seed 0
```

Durée : 6 302 s (~1 h 45 min). Export : 12,13 Mo, Δ RSS +28,7 Mo, 119,7 tok/s.
QA test : **34,0 % EM** [28,7 ; 39,3], F1 0,463, copy 41,7 %, refus 3,7 %.

### 13:45–15:20 — G1-ctrl-s0 (seed 0, contrôle sans question)

Même commande avec `--control`. Durée : 5 843 s (~1 h 37 min).
QA test : **20,3 % EM** [15,7 ; 25,0], F1 0,289, copy 42,7 %, refus 9,3 %.
Écart QA−Ctrl = **13,7 pp** (> seuil 10 pp).

### 15:30–17:15 — G1-qa-s1 (seed 1, QA)

Durée : 5 943 s (~1 h 39 min). QA test : **36,0 % EM** [30,7 ; 41,3].
Écart vs contrôle seed 1 (lancé après) = 16,0 pp.

### 17:30–19:15 — G1-ctrl-s1 (seed 1, contrôle)

Durée : 6 815 s (~1 h 53 min). QA test : **20,0 % EM** [15,7 ; 24,7].

### 19:30 — évaluations finales G1

| modèle | EM | IC95 | écart vs contrôle |
|---|---:|---|---:|
| G1-qa (moy. 2 seeds) | **35,0 %** | [30 ; 41] | **+14,8 pp** |
| G1-ctrl (moy. 2 seeds) | **20,2 %** | [16 ; 25] | — |

**Critères G1 validés** :
- QA ≥ 20 % : ✅ (35,0 %)
- QA ≥ contrôle + 10 pp : ✅ (+14,8 pp)
- IC95 disjoints : ✅ (pas de chevauchement)
- 2 seeds cohérents : ✅ (34,0 % / 36,0 % vs 20,3 % / 20,0 %)
- Déploiement ≤ 100 Mo / Δ RSS < 100 Mo : ✅ (12,13 Mo / +28,7 Mo)

Export, bench, artefacts : `runs/G1-qa-s*/`, `runs/G1-ctrl-s*/`,
`docs/results/qa_passage_G1-*.json`.

---

## État à la fin de la phase 1

- G0 : banc QA gelé (300 test + 100 dev), baselines mesurées.
- G1 : **critères validés** — QA-passage +14,8 pp vs contrôle, 2 seeds cohérents,
  déploiement OK. La branche QA est promue ; la branche contrôle confirme
  l'effet spécifique de la question.
- Prochaine étape : G2 (data mix littérature 15 %) sur le lane M4, en parallèle
  de la location GPU pour G3/G4 (soumise à validation d'estimation).

---

## 2026-10-25 — G5 : run from-scratch 46 M terminé + évaluation

**Run** : 100 000 steps, 819,2 M tokens (3,07 epochs), 2 j 11 h 02,
loss finale 2,46–2,75, 20 checkpoints de 176 MiB.
Modèle réel : 44,38 M poids (46,22 M de tenseurs avec buffers) — le log
« 8 192 512 » était un bug de `num_params()`, corrigé.

**Contamination** : `val_E_frozen` est dans t00 (300/500 snippets de 80
caractères retrouvés) → inutilisable comme validation pour ce run.
OOD : 0/680 snippets → propre. Seul OOD est un jeu de généralisation.

**Résultats** :

| mesure | valeur |
|---|---|
| OOD BPC S=128 | **1,8727** [1,8596 ; 1,8854] |
| OOD BPC S=1024 | **1,7851** scoré (1,5069 protocole) |
| Δc OOD vs t00 | **−0,3538** [−0,3597 ; −0,3481] |
| Δc OOD vs E-t08 | **−0,1880** [−0,1937 ; −0,1823] |
| QA 0-shot | **0,0 % EM**, F1 0,012 |
| sonde FR | **18,5 %** [13,5 ; 24,0] |
| export int8 | 57,33 Mo, Δ RSS +76,5 Mo, BPC 1,8729 |
| export int4 | 35,15 Mo, Δ RSS +54,3 Mo, BPC 1,9161 |

**Lecture** : meilleur modèle de langue que E-t08 sur OOD, mais aucune
capacité QA en 0-shot et dérive factuelle. G1 reste le livrable QA ;
G5 est une base de langage (backbone) pour la suite, pas un assistant.

Artefacts : `docs/results/g5_25m/`, `exports/llm_25m_fr_int{8,4}.safetensors`,
`docs/CHAINE_G.md` §11.

---

## 2026-10-26 — G5-QA : fine-tuning QA du backbone 46 M

**Décision** : appliquer la recette G1 au backbone from-scratch (run de
100 000 steps, 2 j 11 h) plutôt que lancer G2 data mix. Hypothèse testée :
la capacité (44,38 M vs 25 M) débloque la QA extractive là où l'entraînement
brut donnait 0 % EM.

**Données** : `runs/G5-qa-data/qa_prompt_packed.npz` — 129 596 fenêtres ×
257 tokens, 867 175 tokens de réponse utiles, séparateur `\n\n` (le
tokenizer 16k n'a pas d'EOS). Contrôle : `qa_ctrl_packed.npz` (85 294
fenêtres, sans question).

**Calibration** : deux échecs avant de trouver la cause.
- 1re tentative (batch 32 × accum 4) : `exit 137` (tué par l'OS) — les
  4 graphes d'accumulation restaient vivants en mémoire.
- Correction : `mx.eval(acc, loss)` à chaque micro-batch.
- Résultat : **4 777 tok/s, 6,8–7,1 s/step** avec batch 32 × accum 4
  (32 768 tokens/step), swap redescendu sous 1,3 Go.
- 924 steps → **~1 h 49**.

**Run seed 0** : lancé le 26/10 à 10:07, sortie `runs/G5-qa-s0`,
checkpoints tous les 200 steps. Loss 5,07 (step 0) → **0,52 (step 25)**.

**Évaluateur adapté** : `eval_qa_25m.py` coupe désormais la sortie au
premier saut de ligne (borne `\n\n` apprise), équivalent du `stop=eos`
utilisé pour G1 ; le taux `stopped_nl` est reporté.

**Bug corrigé** : `eval_qa_passage_G.py --peritem` avait un défaut fixe qui
écrasait `docs/results/qa_passage_baseline_peritem.json` ; il dérive
maintenant de `--out`.

### Seed 0 — résultats (2026-10-26, 11:52)

Run : 924 steps, 5 972 s (**1 h 39**), 30,3 M tokens, 5 070 tok/s,
loss finale 0,011. Sortie `runs/G5-qa-s0/model.safetensors` (184,9 Mo).

| mesure | seed 0 | référence |
|---|---:|---:|
| **EM test (300)** | **57,3 %** [51,7 ; 63,3] | G1 35,0 % |
| F1 | 0,708 | G1 0,472 |
| copie | 59,7 % | G1 42,7 % |
| refus | 0,0 % | G1 4,0 % |
| arrêt sur `\n` | 100 % | — |
| EM in-domain (150) | 64,0 % [56,7 ; 71,3] | **gonflé : t00 entraîné** |
| **EM OOD (150)** | **50,7 %** [42,7 ; 58,7] | G1 OOD 53,5 % |
| EM années (67) | 100,0 % | — |
| EM nombres (33) | 93,9 % | — |
| EM entités (200) | 37,0 % | — |

**Oubli du langage** : OOD BPC 1,8727 → **3,8998** (PPL 72 → 7 422) ;
sonde FR 18,5 % → **0,0 %**. En génération libre, le modèle produit du
charabia répétitif (« La France est un pays pays pays Loire Loire… »).

Ce n'est **pas** une régression de la recette : G1 avait déjà clean32
PPL ~166 et OOD PPL ~500, documenté en §10.5 comme conséquence normale de la
CE masquée sur la réponse seule. Le critère « OOD +10 % » annoncé avant le
run était mal choisi pour un modèle QA ; le critère valide reste l'EM.

**Fuite à signaler** : le banc in-domain vient de `val_E_frozen_clean32`
⊂ t00, **entraîné** dans le run 46 M (audit : 300/500 snippets). Le 64 %
in-domain mesure donc en partie la mémorisation du passage par le backbone,
pas la QA. Le chiffre honnête de généralisation est **OOD 50,7 %**. Les
passages OOD sont propres (0/680).

**Déploiement** : `exports/g5qa_s0_int8.safetensors` 57,33 Mo,
Δ RSS +76,6 Mo, 84,3 tok/s, **EM identique 57,3 %** (F1 0,707).
Budget respecté (≤ 100 Mo / < 100 Mo).

**Suite lancée** : seed 1 (`runs/G5-qa-s1`) puis contrôle sans question
(`runs/G5-qa-ctrl-s0`), 924 steps chacun, ~3 h 20 au total.
