# Notes de travail (version publique)

Orientation rapide pour les contributeurs. Les notes détaillées avec
chemins locaux restent hors dépôt.

## Recette qui marche (ordre exact)

```bash
cd micro-llm-2bit && uv run python main.py train --data <dirs...> \
  --parent HuggingFaceTB/SmolLM-360M --out checkpoints/enfant.pt --epochs 3
uv run python main.py export --ckpt checkpoints/enfant.pt --out exports/enfant.bin
uv run python main.py bench --model exports/enfant.bin --bench-tokens 32
uv run python main.py infer --model exports/enfant.bin --prompt "..." --max-new 60
```

`--data` accepte plusieurs chemins (dossiers ou fichiers
`.txt`/`.jsonl`/`.xml`).

## Toolchain (ne pas changer)

- `mise` = Python 3.11 uniquement, `uv` = venv + dépendances.
- `make` compile `kernels/neon.c` (arm64) vers `build/libmicro2bit.dylib`
  (clang Apple, GCD, pas d'OpenMP). Fallbacks : `avx2.c`, `portable.c`, NumPy.
- Fichiers : `quant.py`, `model.py`, `engine.py`, `data.py`, `distill.py`,
  `main.py`, `tools/bench_train.py`, `data_fetch/`, `data_clean/`.

## Bug déjà corrigé (ne pas réintroduire)

`DistillationLoss` utilisait `F.kl_div(..., reduction="batchmean")` qui ne
divise que par B (pas B×S) → KL ~128× trop grande (loss ~789). Fix :
`(t_prob * (t_prob.log() - s_logp)).sum(-1).mean() * T²`, avec garde
`0*log(0)=0` pour le cache top-k. Repère : à step 20, loss ~8, CE ~7
(< hasard ln(8000)=8.99).

## Ollama

Le client Python lit `OLLAMA_HOST`. Ollama ne sert QUE pour `distill-loop`
(texte) et les traces think : la distillation KL exige le parent HF en
local (logits). Le `train` tourne sur le Mac principal (MPS).

## Checkpoints et reproductibilité

- Chaque checkpoint fait ~843 Mo (base) : ne garder que les 1-2 derniers.
- Les runs versionnés gardent `runs/<date>/config.json` (reproductibilité).
- Ne jamais committer : `.venv/`, `*.pt`, `*.bin`, corpus, caches, logs
  (voir `.gitignore`).
