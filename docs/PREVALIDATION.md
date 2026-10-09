# Prévalidation — session du 09/10/2026 (phases A à F)

Document de synthèse : tout ce qui a été fait, vérifié et livré ce jour,
avant les runs longs (tranche01 / E) qui attendent un go explicite.
Détails : `docs/DIAGNOSTICS.md` (C), `docs/ABLATIONS.md` (D + commandes),
`docs/RUNS.md`, `README.md`, `MEMO.local.md` (local, non committé).

## Règles de la session — tenue

- Toolchain uv/mise inchangée ; fichiers modifiés complets et cohérents.
- Tests : 33 → 70 (69 verts + 1 skip cuda, matériel absent). Chaque
  correctif a ajouté ses tests. Bug KL du MEMO intact (formule inchangée).
- Commits atomiques (24 ce jour), jamais de --force, aucun fichier > 1 Mo,
  tout poussé sur `main`.
- GPU exclusif : précheck avant chaque mesure/run (ps + Ollama) ; runs
  > 5 min en détaché (nohup + log + veilleur fin/NaN/val).
- Aucun entraînement long lancé sans go (E en attente). Stash WIP montré
  puis appliqué sur ordre (conflit résolu : HEAD déjà à jour, matplotlib
  bien en dev, stash droppé après vérification).

## Phase A — vérifications (4 défauts confirmés, 1 infirmé)

- A1 CONFIRMÉ : `steps_per_epoch` ignorait `accum` et la découpe ; wiki_slice00
  (bs16/accum8/save8) n'a vu que 50,0 % de la tranche (17,4/34,8M tokens).
  Tranche réelle slice_00 : 36 517 844 tokens, 2,82 chars/token. tranche00 :
  1 epoch complète en 15 668 s (pas 2).
- A2 CONFIRMÉ : bos=eos=unk=0 (`<|endoftext|>`), pad None ; unk==eos==0
  dans wiki_slice00 ; 1,00 % de tokens val rabattus sur l'id 0 hors vrais eos.
- A3 CONFIRMÉ : `UnboundLocalError` sur `proj` (test minimal reproduit).
- A4 CONFIRMÉ : `pick_device()` sans cuda, fp16 que sur mps.
- INFIRMÉ (1) : « 720 steps / PPL 63,7 » de RUNS.md = docs périmées
  (vrai : 4248 steps, PPL 29,7), pas un bug de code.

## Phase B — corrections (ordre demandé : B1, B3, B4, B5, B6, B2 en dernier)

- B1 : `plan_epoch()` (steps = ceil(micros/accum), dernier batch partiel
  inclus), refus clair si bs > save AVANT chargement parent, reste de fin
  d'epoch appliqué (plus de gradients jetés), log avec tokens/steps/epoch
  et micro-batch effectif, vérif cache `micro_batch` inchangée. 10 tests.
  Consigne appliquée : `--save-batch` == `--batch-size` partout (README,
  THINK_MODE, `run_slices.sh` + formules B1, bench sentinelle).
- B3 : `proj = None` avant construction du cache. Test dédié.
- B4 : `--ckpt-every` (défaut 500, 2 derniers gardés), checkpoints complets
  (modèle/opt/sched/epoch/micro/ordre/RNG), ordre déterministe (seed,epoch),
  reprise mid-epoch bit-identique (testée). Garantie à hyperparamètres
  identiques (scheduler dépend du total) — documenté.
- B5 : `cuda > mps > cpu`, fp16 parent sur cuda comme mps, mps/cpu inchangés,
  tests cuda skippés sans GPU.
- B6 : alpha=0 (+hidden 0) = CE seule, parent ni chargé ni appelé (ni cache).
  Tests : loss==CE, `from_pretrained` jamais appelé.
- B2 (dernier, impact 1 %) : ligne unk dédiée (sentinelle −1, taille
  inchangée, eos intact, `vocab_version` 2) ; colonne unk parent = logsumexp
  fp32 hors-vocab (direct ET cache, format `.bin`/cache inchangé) ; stop sur
  vrai eos seul, unk masqué par défaut (`--allow-unk`), marqueur `�` au
  décodage par tronçons ; anciens vocabs inchangés, mélange refusé ;
  `tools/convert_vocab.py` testé (ligne évincée = moyenne). 12 tests.
  Conversion NON utilisée pour les ablations (runs depuis zéro, sur accord).

## Phase C — diagnostics (`docs/DIAGNOSTICS.md`)

Validation commune : 6 docs val slice_00, 354 234 chars, jamais vus.
Comparaison tranche00 vs wiki autorisée (vocabs `kept_ids` identiques vérifiés).

- C1 : unk 8000/12000/16000 = 2,14 % / 1,00 % / 0,45 % (embeddings tiny
  3,94 / 5,91 / 7,87 Mo). Verdict : garder 12000.
- C2 BPC : 135M 1,5357 / 360M 1,3362 / Qwen3-0.6B 1,1138 / tranche00.bin
  1,5676 (PPL 25,88) / wiki.bin 1,6566 (PPL 31,12). tranche00 à 0,03 BPC
  du parent (~98 % du plafond en 1 epoch). Gemma non tenté (accès sous
  conditions). Limites notées (fp32 optimiste, fenêtres avec reset, unk inclus).
- C3 : masse hors top-k T=1 : k32/64/128/512 = 0,249/0,193/0,145/0,070 ;
  à T=2 : 0,816/0,775/0,726/0,594. Le cache top-32 jette ~82 % de la masse
  à T=2 : anomalie cache expliquée, parent direct confirmé.
- C4 : 2 221 tok/s effectifs (tokens réels, bug-independent), micro-batch
  réel 8 (8 192 tok/step), bench bs16 : 2 577.
- eval_fixed : sorties brutes côte à côte (`eval/tranche00/`,
  `eval/wiki_slice00/`) ; greedy tranche00 boucle ~3× moins (0,299 vs 0,454).

## Phase D — ablations (`docs/ABLATIONS.md`, 9 runs, ~6h GPU, bruit ±9,07 PPL)

Protocole : tiny, SmolLM-135M, vocab partagé unk-distinct, subset 5M,
val tous les 50 steps, jeux communs 128/256 post-hoc, un run à la fois
(`tools/run_ablations.sh`), précheck avant chacun, ligne par run terminé.
PPL-128 : V0a 161,9 · V0b 170,9 · V1a 129,7 · V1b 59,9 · V2 132,0 ·
V3 72,5 · V4 148,5 · V5 174,2 · **V6 55,2**.
Verdicts : V6 (CE+lr1e-3, gains cumulés, 1 158 s) meilleur 5M ; V3 résultat
fort (−89) ; V1a>V0 et V2≈V1a (baisser alpha aide) ; V4/V5 écartés
(bruit / aucun gain seq256). V6 lancé par la session parallèle, vérifié
(config, log, ckpt) et intégré ici. Incidents notés : extraction auto
steps/tokens vide sur les lignes (réparé via ckpt, cause transitoire non
élucidée) ; V1b mur +11 % (noté) ; tokens V1b corrigés (9 994 240).
Commandes exactes des 9 runs dans le doc (générées des config.json).

## Phase E — run principal (PRÊT, NON LANCÉ)

Recette V6 : CE seule (SANS parent, ~4 400 tok/s), lr 1e-3, seq 128,
reprise `runs/2026-10-08-tranche00/tranche00.pt` + son vocab (ancien format,
supporté). Tranches slice_01→slice_09, ~34,8M tokens/tranche (décompte
corrigé), ~310M au total, ~20 h. Garde-fou : revalidation après tranche01
(BPC cible < 1,55), fallback CE/KD documenté. Commande tranche01 dans le
rapport de session (adaptée de V6 + `--resume` + `--rewarmup-ratio 0.05`
+ `--lr-resume 1e-3`).

## Phase F — profil inférence (CPU, tranche00.bin, 7,68 ms/token)

56 matvecs C 47,3 % · tête logits 28,7 % · attention+normes+quant+overhead
24,0 % (mesures cProfile + froid concordantes). Tête < 30 % : PAS de noyau
C (condition du protocole non remplie). KV non retouchée (même condition).
Préfill par lots écarté (forward par lots = non local). Moteur inchangé,
aucun test à mettre à jour.

## Livraison et reste

- Livré : 70 tests, README (résultats C/D/F + recette V6), MEMO.local.md
  (local, ignoré par design), `docs/NOTES.md` vérifié propre (sans IP ni
  chemins), 25+ commits poussés, 4 rapports Telegram (2160–2163, API
  directe car outil intégré sans token, `.env.local` hors dépôt).
- En attente de go : `go tranche01` (recommandé : 1 tranche + revalidation)
  ou `go E` (chaîne 9 tranches) ; accord conversion unk-distinct (oui/non).

## Tableau de correspondance (demandé → livré → où)

| Demandé | Livré | Où |
|---|---|---|
| Règles (uv, tests+fix, pas de KL-bug, atomiques, GPU exclusif, détaché+veilleur, pas de --force, <1 Mo) | Toutes tenues, E non lancé | commits |
| A1–A4 + mini-rapport, défaut faux signalé | ✓ (RUNS périmé = le seul point infirmé) | chat + logs |
| B1→B6 dans l'ordre, B2 dernier, rapport B | ✓ +10/1/2/6/3/12 tests | `distill.py`, `tests/` |
| Stash pop, matplotlib dev, uv.lock | ✓ (conflit résolu : HEAD déjà à jour) | — |
| RUNS.md + RAPPORT_AMELIORATIONS corrigés | ✓ (tranche00 référence) | `docs/` |
| save==bs partout | ✓ (+ `run_slices.sh`, THINK_MODE, bench sentinelle) | README/docs/tools |
| C1–C4 + eval_fixed côte à côte + rapport C | ✓, comparaison autorisée (vocabs identiques vérifiés) | `docs/DIAGNOSTICS.md` |
| D : protocole, 1 à la fois, ABLATIONS.md progressif, tokens corrigés au log | ✓ 9 runs (V0×2→V6), veilleur, préchecks | `docs/ABLATIONS.md` |
| E : config, tokens, temps, commande, rien lancé | ✓ (V6 : CE+lr1e-3, ~20 h) | chat + README |
| F : profil + tableau, noyau si >30 %, préfill si local | ✓ mesuré 28,7 % → **pas de noyau** (condition non remplie), préfill écarté (non local) | chat + Telegram |
| Livraison : tests, README/MEMO FR, NOTES, push, rapport | ✓ 69+1, NOTES déjà propre vérifié | GitHub |
| Telegram | Réparé (API directe), 4 rapports (2160–2163) | — |
