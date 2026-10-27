# Écart de données (data gap)

**Date** : 2026-10-27
**Objet** : inventorier ce que le dépôt possède réellement, mesurer la couverture
des trois capacités visées (conversation FR, code, appels d'outils) et chiffrer
ce qui manque.
**Méthode** : comptage par tokenizer réel (`models/tokenizer_fr_16k`), un
échantillon d'une ligne sur 97 pour les gros fichiers, passage complet pour les
petits. Les tokens des fichiers non échantillonnés sont comptés exhaustivement.
Tous les chiffres sont **[mesurés]** sauf mention contraire.

---

## 1. Résumé

Le corpus est **très déséquilibré** :

- **264,2 M tokens** de Wikipédia français, propres et utiles pour la langue ;
- **18,1 M tokens** de QA extractive française ;
- **5,2 M tokens** de code, dont 92 % de Python **mal étiqueté JavaScript** ;
- **30,5 M tokens** d'OpenOrca à **97 % anglais** ;
- **0,26 M tokens** de dialogue, dont **892 tokens** seulement de vraies
  conversations multi-tours.

Autrement dit : **99,9 % du corpus est de la langue ou de l'instruction, 0,1 %
est de la conversation.**

Pour un modèle de 200 M paramètres, la règle de Chinchilla (20 tokens/param)
demanderait 4 Md tokens. Le dépôt en fournit ~288 M, soit **7,2 tokens/param**,
2,8× sous le régime compute-optimal. Et pour le dialogue seul, l'écart est de
**deux ordres de grandeur**.

---

## 2. Inventaire mesuré

### 2.1 Corpus de langue

| fichier | docs | caractères | tokens | taille | FR | remarque |
|---|---:|---:|---:|---:|---:|---|
| `tranche_00` | 10 134 | 104 918 140 | 27 006 380 | 109 Mo | 99,4 % | 0 % ASCII (accents) |
| `tranche_01` | 10 236 | 104 860 561 | 27 417 156 | 109 Mo | 99,2 % | |
| `tranche_02` | 10 240 | 104 865 455 | 27 237 908 | 109 Mo | 99,2 % | |
| `tranche_03` | 10 344 | 104 876 207 | 27 241 292 | 109 Mo | 99,4 % | |
| `tranche_04` | 10 450 | 104 918 222 | 27 411 399 | 109 Mo | 99,2 % | |
| `tranche_05` | 10 074 | 104 875 979 | 27 379 231 | 109 Mo | 99,1 % | |
| `tranche_06` | 10 151 | 104 866 684 | 27 837 083 | 109 Mo | 98,7 % | |
| `tranche_07` | 10 304 | 104 873 517 | 27 852 437 | 109 Mo | 99,0 % | |
| `tranche_08` | 10 422 | 104 896 908 | 27 374 637 | 109 Mo | 99,4 % | |
| `tranche_09` | 7 607 | 68 236 379 | 17 479 180 | 71 Mo | 99,0 % | tranche incomplète |
| **total** | **99 962** | **1 012 188 052** | **264 236 703** | **1,05 Go** | **~99 %** | |

C'est un corpus **propre, dédupliqué, monolingue**, de bonne qualité
encyclopédique. `tok/char = 0,253` : le tokenizer est efficace sur du texte long.

### 2.2 Données conversationnelles et d'instruction

| fichier | docs | caractères | tokens | FR | format | qualité |
|---|---:|---:|---:|---:|---|---|
| `data/chat/chat_pilot_v1.jsonl` | 10 | 1 965 | **892** | 100 % | multi-tours | propre mais minuscule |
| `data/hybrid/oasst1_fr.jsonl` | 2 474 | 852 966 | 225 812 | 99,3 % | **messages isolés** | pas des dialogues |
| `data/hybrid/synthetic.jsonl` | 100 | — | 37 199 | 100 % | Q/R | **mauvaise qualité** |
| `data/probe/probe_fr_v1.jsonl` | 200 | — | 1 724 (prompts) | 100 % | sonde | **gelé, contractuel** |

Le seul vrai corpus conversationnel fait **892 tokens**, soit environ **6
secondes de parole**. Les 225 812 tokens OASST1 sont des messages séparés
(questions, réponses isolées), pas des paires tour/user/assistant : ils ne
peuvent pas enseigner le format multi-tours sans reconstruction.

Exemples de défauts du synthétique [mesuré] :
```
Q: Corrige ce bug: def foo(x): return x + 1
A: ... Le retour de la fonction `foo` n'est pas correct. Vous
   Vous devez retourner la valeur de `x + 1`. ...
   (bloc de code non fermé, texte tronqué)
Q: Explique ce code: x = [1, 2, 3]; x.sort()
A: Cet code est une instruction ... (faute, ligne coupée)
```
100 exemples, tous générés par un petit modèle, avec troncatures et répétitions.
Ce fichier n'apporte rien et doit être abandonné.

### 2.3 Code

| fichier | docs | caractères | tokens | ASCII | langue réelle |
|---|---:|---:|---:|---:|---|
| `data/hybrid/javascript_code.jsonl` | 500 | 8 499 313 | **5 233 367** | 100 % | **462 Python, 11 JS, 27 autre** |

Le nom du fichier est faux : 92 % des fichiers sont du **Python** (Django,
scikit-learn, Ansible). Il n'y a quasiment pas de JavaScript. Le contenu est du
code source brut, **sans paire instruction/réponse**, donc inutilisable tel quel
pour du code instruct.

### 2.4 Instructions anglaises

| fichier | docs | caractères | tokens | FR | remarque |
|---|---:|---:|---:|---:|---|
| `data/hybrid/openorca_instructions.jsonl` | 50 000 | 83 205 512 | **30 548 268** | **2,9 %** | 97 % anglais |

30,5 M tokens, soit **10 % du corpus total**, dans une langue que le modèle ne
doit pas prioriser. Le sous-ensemble français représente ~0,9 M tokens. Le reste
doit être exclu ou lourdement sous-échantillonné.

### 2.5 QA extractive

| fichier | docs | caractères | tokens | remarque |
|---|---:|---:|---:|---|
| `data/qa/qa_fr_pilot_v1.jsonl` | 34 557 | 18 073 144 | 4 656 281 | dupliqué par `_repro_v1` |
| `data/qa/qa_fr_pilot_v2.jsonl` | 29 097 | 18 756 377 | 5 061 300 | dupliqué par `_repro_v2` |
| `data/qa/qa_fr_pilot_v3.jsonl` | 216 614 | 68 434 983 | **18 133 312** | référence |
| `data/qa/g_passage_dev_v1.jsonl` | 100 | 11 771 | 3 961 | dev |
| `data/qa/g_passage_test_v1.jsonl` | 300 | 36 438 | 9 860 | test |

`_repro_v1` et `_repro_v2` sont des **copies octet pour octet** de v1 et v2
(même taille : 18 872 933 et 19 693 057 octets). Il y a donc redondance disque,
pas de données supplémentaires.

La QA est **extractive** : le modèle apprend à recopier un span. C'est utile
pour une tâche de lecture, pas pour converser.

---

## 3. Couverture des capacités

| capacité | données disponibles | besoin estimé [hypothèse] | écart |
|---|---:|---:|---:|
| Langue FR | 264,2 M tokens | 1–4 Md | **4–15×** |
| Dialogue FR multi-tours | **0,0009 M** | 20–50 M | **22 000–55 000×** |
| Instruction FR | 0,26 M | 5–10 M | **19–38×** |
| Code (instruct) | 5,2 M bruts, 0 paire | 10–50 M | **≥ 2×** et mauvais format |
| Code multi-langage | ~0,6 M non-Python | 10 M | **17×** |
| Appels d'outils | **0** | 50 000–200 000 trajectoires | **infini** |
| QA extractive | 18,1 M | 5 M | ✅ excédent |

Le tableau dit l'essentiel : **le dialogue et les outils ne sont pas
« sous-représentés », ils sont absents.**

---

## 4. Contamination et splits

| split | contamination 32-grammes | statut |
|---|---:|---|
| `val_big` | **47,80 %** | **inutilisable pour décider** |
| `val_E_frozen` | **53,6 %** des documents | inutilisable |
| `OOD` | **0 / 340** | **seul split propre** |
| `clean32` | conçu pour être propre | à vérifier à chaque run |

Cause probable : le split a été fait **par chunk** et non **par document**, et le
corpus Wikipédia dédupliqué conserve des paragraphes quasi identiques entre
articles. Tant que le split n'est pas refait par document (hash de l'article,
groupement, puis découpage), les métriques de validation resteront optimistes.

**Action** : reconstruire les splits avec `group_id = hash(article)`, split
80/10/10 par groupe, et vérifier 0 % de 32-grammes communs avant tout
entraînement.

---

## 5. Tokenizer

Mesures [mesuré] :

| texte | caractères | tokens | tok/char |
|---|---:|---:|---:|
| Wikipédia (texte long) | 1 012 M | 264,2 M | **0,253** |
| « Bonjour, comment allez-vous ? » | 29 | 11 | 0,379 |
| « Le président s'est rendu à l'Élysée. » | 39 | 12 | 0,308 |
| « Café déjà vu, où est l'été ? » | 28 | 12 | 0,429 |

Le round-trip `decode(encode(x)) == x` est **exact** sur les tests accentués.
Les tokens s'affichent en mojibake (`Ã©`) dans les diagnostics, mais le
décodage est correct : c'est un artefact d'affichage, pas un bug fonctionnel.

Points à corriger :
- **vocabulaire 16 000** : correct pour le FR, un peu juste pour le code
  (identifiants longs découpés).
- **tokens spéciaux existants** : `<|endoftext|>` (id 0) et `<|pad|>` (id 1).
- **tokens manquants** : `<|system|>`, `<|user|>`, `<|assistant|>`,
  `<|tool_call|>`, `<|tool_result|>`, `<|end|>`.
- **padding** : le SFT actuel padde à l'id **0**, qui est `<|endoftext|>`, avec un
  masque de perte à 0. Le padding n'est donc pas supervisé, mais l'attention
  voit des EOS artificiels en fin de séquence. Il faut padder à l'id 1
  (`<|pad|>`) et garder le masque.
- **EOS cible absent** : aucune séquence ne se termine par un EOS supervisé.
  Le modèle n'apprend donc pas à s'arrêter.

Le coût des accents est réel mais acceptable : ~0,31 tok/char sur une phrase
conversationnelle, soit ~1 600 caractères pour 512 tokens. Un contexte de 2 048
tokens couvre ~6 400 caractères, suffisant pour un tour de conversation.

---

## 6. Licences

| source | licence | risque |
|---|---|---|
| Wikipédia FR | **CC BY-SA 4.0** | attribution + partage à l'identique ; statut des poids à clarifier |
| Gutenberg | domaine public **US** | pas nécessairement public en FR/EU |
| OASST1 | Apache 2.0 | faible |
| OpenOrca | MIT (dataset) | données sources hétérogènes |
| SmolLM2 | Apache 2.0 | faible |
| Qwen2.5 / Qwen3 | Apache 2.0 | faible |
| `codeparrot-clean` | licences GitHub **hétérogènes, dont GPL** | **contamination copyleft** |
| The Stack v2 | filtrage par licence possible | à filtrer explicitement |

Le point noir est le code : un corpus GitHub non filtré contient des fichiers
GPL/AGPL. Pour un modèle distribué, il faut **filtrer par licence** (MIT, Apache,
BSD, ISC) et conserver la provenance par fichier.

---

## 7. Ce qu'il faut construire

### 7.1 Priorité 1 — dialogues FR (le goulot)

Cible : **20–50 M tokens** de dialogues multi-tours, format
`system/user/assistant`, avec EOS.

Sources possibles :
1. **Génération par professeur** (Qwen2.5-1.5B/3B-Instruct) sur 2 000–5 000
   amorces FR, puis filtrage. C'est la seule voie rapide.
2. **OASST1 reconstruit** : regrouper les messages par arbre de conversation
   (message_id / parent_id) pour retrouver les dialogues complets.
3. **ShareGPT / WildChat filtrés FR** : vérifier les licences et le consentement.
4. **Traductions de dialogues anglais** (UltraChat, OpenHermes) : qualité
   variable, à filtrer.
5. **Corpus français authentiques** : forums, StackExchange FR (CC BY-SA),
   transcriptions (attention au droit).

### 7.2 Priorité 2 — code instruct

Cible : **10–20 M tokens** de paires instruction → code, multi-langage
(Python, JS/TS, Rust, Go, SQL, shell).

Sources :
- The Stack v2 filtré permissif ;
- paires existantes (CodeAlpaca, Magicoder, OSS-Instruct) ;
- génération de paires à partir de fichiers réels (docstring → code, bug →
  correctif).

### 7.3 Priorité 3 — outils

Cible : **50 000–200 000 trajectoires** couvrant :
- macOS : `open_app`, `run_applescript`, `read_file`, `write_file`,
  `list_dir`, `http_get` ;
- Windows : `start_process`, `powershell`, `read_file`, `write_file`,
  `list_dir`, `http_get` ;
- format JSON Schema, arguments typés, erreurs, résultats.

Sources : ToolBench, API-Bank, Glaive Function Calling, Hermes Function Calling,
puis génération synthétique à partir de schémas réels.

### 7.4 Priorité 4 — splits propres

- split par document, pas par chunk ;
- OOD gelé et hashé ;
- vérification 0 % de 32-grammes train/test ;
- manifeste avec hash SHA-256 du corpus.

---

## 8. Ce qui est propre et à garder

- Wikipédia FR dédupliqué (264,2 M tokens) : **le meilleur actif du dépôt**.
- QA extractive v3 (18,1 M tokens) : utile pour une tâche de lecture, à garder
  comme tâche auxiliaire.
- Sonde `probe_fr_v1` (200 prompts gelés) : **à préserver tel quel**, c'est le
  seul test non contaminé avec OOD.
- Manifests `data/qa/*.manifest.json` : bonne pratique, à généraliser.
- Scripts de collecte (`download_hybrid_data.py`, `prepare_*`) : à corriger, pas
  à jeter.

---

## 9. Ce qui doit être jeté

- `data/hybrid/javascript_code.jsonl` : Python mal étiqueté, sans paires.
- `data/hybrid/synthetic.jsonl` : 100 exemples tronqués.
- `data/hybrid/openorca_instructions.jsonl` : 97 % anglais, sauf le sous-ensemble
  FR (~0,9 M tokens) à extraire.
- `data/hybrid/oasst1_fr.jsonl` **comme dialogues** : messages isolés ; à
  reconstruire par arbre.
- `data/hybrid/no_robots_fr.jsonl` et `octopack_code.jsonl` : **0 octet**,
  téléchargements échoués.
- Les copies `_repro_v1` / `_repro_v2` : redondance.

---

## 10. Verdict honnête

Le dépôt a **un très bon corpus de langue française** et **presque rien pour
converser**. Ce n'est pas un détail : c'est le facteur limitant du projet.

Avec 264 M tokens de Wikipédia, on peut entraîner un excellent modèle de langue
de 100–200 M paramètres. On ne peut pas entraîner un assistant : il manque
**20 à 50 M tokens de dialogues**, **10 à 20 M tokens de code instruct** et
**50 000 à 200 000 trajectoires d'outils**. Ces volumes ne se trouvent pas dans
le dépôt ; ils doivent être collectés, générés et filtrés.

Tant que ce chantier données n'est pas fait, la migration MLX et le runtime C
produiront un modèle techniquement élégant qui ne conversera pas. C'est le
risque principal du projet, et il est plus difficile que la technique.

**Recommandation** : bloquer la phase d'entraînement tant que le corpus de
dialogue n'atteint pas 5 M tokens propres. En dessous, le résultat sera un
modèle de langue, pas un assistant.
