# MÉMO — micro-llm-2bit (note de passage pour les sessions futures)

## C'est quoi
Petit LLM ternaire 2 bits (BitNet b1.58), decoder-only tiny
(V=8000, d=512, L=8, Hq=8, Hkv=4, ffn=1024, ~23M params, ~18 Mo < 100 Mo),
inférence CPU via noyau C NEON (`.dylib` + ctypes), entraîné par distillation
KL depuis `HuggingFaceTB/SmolLM-360M`. CLI : `infer / train / export / bench /
distill-loop` (`python main.py <cmd> --help`).

## Toolchain (ne pas changer)
- `mise` = Python 3.11 uniquement (`.mise.toml`), `uv` = venv + deps
  (`pyproject.toml` + `uv.lock`). Poetry écarté, cargo hors sujet.
- `make` compile `kernels/neon.c` (arm64) vers `build/libmicro2bit.dylib`
  (clang Apple, GCD, pas d'OpenMP). Fallbacks : `avx2.c`, `portable.c`, NumPy.
- Fichiers : `quant.py` (ternaire + packing), `model.py` (Transformer + budget),
  `engine.py` (infer C/NumPy + `.bin`), `data.py` (corpus + vocab réduit),
  `distill.py` (KL + boucle train + pont Ollama), `main.py` (CLI).

## Recette qui marche (ordre exact)
```bash
cd ~/micro-llm-2bit && uv run python main.py train --data <dirs...> \
  --parent HuggingFaceTB/SmolLM-360M --out checkpoints/enfant.pt --epochs 3
uv run python main.py export --ckpt checkpoints/enfant.pt --out exports/enfant.bin
uv run python main.py bench --model exports/enfant.bin --bench-tokens 32
uv run python main.py infer --model exports/enfant.bin --prompt "..." --max-new 60
```

## Corpus du disque externe
`/Volumes/dique secours/rag-data` (47 Mo). Tri effectué :
- GARDÉS : `wiki/` (fr), `monde/` (fiches pays), `internet/` (stackexchange en),
  `actu/` (RSS Le Monde/BBC via support `.xml` ajouté à `data.py`), `rfc/` (en).
  = 297 docs, 1.7M caractères.
- EXCLUS : `cuisine/` (pages d'erreur HTML), `france/`+`livres/` (métadonnées),
  `monde_src/` (code), `pdf/` (datasheets en).
- `--data` accepte plusieurs chemins (`nargs="+"`, cf. `main.py` + `data.py`).

## Bug déjà corrigé (ne pas réintroduire)
`DistillationLoss` utilisait `F.kl_div(..., reduction="batchmean")` qui ne
divise que par B (pas B×S) → KL ~128× trop grande (loss ~789). Fix :
`(t_prob * (t_prob.log() - s_logp)).sum(-1).mean() * T²`. Repère : à step 20,
loss ~8, CE ~7 (< hasard ln(8000)=8.99). Test numérique inline + 15 tests pytest.

## Ollama / Mac mini distant
- Serveur : `http://100.101.108.111:11434` (mini, Tailscale). Modèle juge :
  `smollm2:360m` (`ollama pull` **sur le mini**).
- Le client Python lit `OLLAMA_HOST` (vérifié dans `ollama/_client.py`) :
  `export OLLAMA_HOST=http://100.101.108.111:11434` avant `distill-loop`.
- IMPORTANT : Ollama ne sert QUE pour `distill-loop` (texte). Le `train` KL
  exige le parent HF en local (logits) et tourne à 100 % sur le Mac principal
  (MPS). ~639 steps ≈ 20-40 min pour le corpus 1.7M.

## GitHub
Repo : `dimitricl/micro-llm-2bit`. Ne jamais committer : `.venv/`,
`*.pt`, `*.bin`, `~/.cache/huggingface` (déjà dans `.gitignore`).
Pour runner sur le mini : `git clone`, `mise install`, `uv sync`, `make`,
mêmes commandes (adapter `--data` au chemin local du corpus).
