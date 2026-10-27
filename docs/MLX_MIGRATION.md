# Migration de l'entraînement vers MLX

**Date** : 2026-10-27
**Objet** : décrire comment passer d'une chaîne PyTorch/MPS à une chaîne
**MLX uniquement**, avec quantification ternaire pendant l'entraînement (QAT) et
distillation, sans jamais utiliser PyTorch ni MPS pour l'entraînement.
**Statut** : document de conception. Aucun code de cette migration n'est encore
écrit ni exécuté (hors le prototype MLX non commité décrit §2).

---

## 1. Principes

1. **Un seul runtime d'entraînement : MLX.** Pas de PyTorch, pas de MPS, pas de
   fallback CUDA. Les scripts PyTorch existants restent comme référence
   historique, pas comme chemin d'exécution.
2. **Le ternaire est appris, pas appliqué après coup.** Le forward utilise des
   poids quantifiés {−1, 0, +1} avec STE ; le backward met à jour des poids
   latents fp32. Un export int4 post-training ne produit pas un modèle ternaire
   équivalent.
3. **La distillation est optionnelle mais ciblée.** Elle sert au format
   conversationnel et au code, pas au pré-entraînement langue de masse (où la CE
   suffit et où le cache top-k coûterait trop cher).
4. **Tout est mesuré sur `OOD` et `clean32`.** `val_big` est contaminé à 47,8 % ;
   il ne peut pas servir de critère de décision.
5. **Le budget se mesure en RSS absolue**, pas en delta. Voir
   `BUDGET_100MB.md`.

---

## 2. État actuel de la voie MLX (non commitée)

Le dépôt contient déjà un prototype fonctionnel :

| fichier | rôle | état |
|---|---|---|
| `tools/train_hybrid_mlx.py` | pré-entraînement CE, modèle 47,7 M | **tourne** (step 35 000/36 980) |
| `tools/export_hybrid_mlx.py` | export `.safetensors` → `.bin` int4 | fonctionnel, non conforme ternaire |
| `tools/prepare_hybrid_sft_mlx.py` | préparation SFT 1 583 exemples | fonctionnel, bugs de format |
| `tools/run_hybrid_mlx_full.sh` | orchestration | fonctionnel |
| `tools/watch_hybrid_mlx.py` | suivi | fonctionnel |

Mesures du run en cours [mesuré] :
```
modèle      : 47,7 M paramètres, d=640, L=10, Hq=8, Hkv=4, ffn=1280, V=16000
données     : 302,9 M tokens (cache uint16, 605 Mo)
débit       : 4 730 tok/s
durée       : ~18 h pour 1 époque
loss        : 2,8708 (ppl 17,65) au step 35 000
```

Ce prototype prouve que MLX tient la charge sur M4 16 Go. Il ne prouve rien sur
le ternaire, la distillation ou la conversation.

---

## 3. Architecture cible

Contrainte : ~200 M paramètres ternaires, 100 Mo de RAM totale à l'inférence.

| paramètre | valeur cible | justification |
|---|---:|---|
| vocabulaire | 16 384 | compromis FR/code, embeddings 1 octet |
| d_model | 1 024 | capacité |
| couches | 16 | profondeur |
| têtes Q | 16 | head_dim 64 |
| têtes KV | 4 | GQA 4:1, KV cache divisé par 4 |
| ffn | 2 816 | ratio 2,75 |
| seq_max | 2 048 | context utile |
| embeddings | int8, liés | 25,2 Mo |
| linéaires | ternaires, échelles fp16 group 64 | 50,7 Mo |
| **poids totaux** | **~76 Mo** | voir `BUDGET_100MB.md` |
| **total inférence** | **~94,5 Mo** | KV 17 Mo + act 2,2 + tok 1,2 + runtime 1,5 |

Alternative plus sûre : `d=896, L=14, ffn=2304, V=16384` → 130 M paramètres,
45 Mo de poids, 61 Mo au total. Plus de marge pour le KV et le runtime, moins de
capacité. Le choix se fait après la phase 2 (voir `ROADMAP.md`).

---

## 4. Quantification ternaire en MLX (QAT)

### 4.1 Forward ternaire

MLX 0.32.4 ne fournit pas de matmul ternaire. On quantifie **dans le forward**,
en fp32, avec STE :

```python
import mlx.core as mx

def ternary_dequant(w, group_size=64):
    """w: (out, in) fp32 -> w_q dequantifié (out, in) fp32, valeurs dans {-s,0,+s}."""
    out_dim, in_dim = w.shape
    wg = w.reshape(out_dim, in_dim // group_size, group_size)
    scale = mx.abs(wg).mean(axis=-1, keepdims=True) + 1e-8
    q = mx.clip(mx.round(wg / scale), -1.0, 1.0)
    return (q * scale).reshape(out_dim, in_dim)

def ternary_ste(w, group_size=64):
    """Poids utilisé en forward ; le gradient passe vers w."""
    wq = ternary_dequant(w, group_size)
    return w + mx.stop_gradient(wq - w)
```

Le coût est un reshape + mean + round par matmul. Sur M4, cela reste dominé par
le matmul fp32 ; le surcoût est estimé à 10–20 % [hypothèse], à mesurer.

### 4.2 Initialisation

Deux options :

- **Depuis zéro** (recommandé pour le ternaire) : initialisation normale
  `std=0.02`, QAT dès le premier step. C'est la voie BitNet.
- **Depuis un parent fp32** : charger un checkpoint dense, puis activer le QAT
  avec un warmup de 500 steps à LR réduit. Plus rapide, mais le parent 47,7 M
  n'a pas la bonne taille ; il faudrait un parent 200 M, qui n'existe pas.

Recommandation : from-scratch ternaire, avec QAT immédiat.

### 4.3 Ce qu'il faut mesurer

- Perte de qualité QAT vs fp32 à iso-paramètres (gap attendu 1–3 % de PPL)
  [hypothèse].
- Stabilité : les échelles absmean peuvent s'effondrer si un groupe est nul.
  Garde : `scale = max(scale, 1e-4)` et clipping des poids latents.
- Distribution des poids : fraction de zéros attendue 30–50 % [hypothèse] ; si
  elle dépasse 80 %, le modèle s'effondre.

### 4.4 Optimiseur

AdamW MLX, `lr=3e-4`, `wd=0.1`, warmup 5 %, décroissance cosinus ou WSD. Les
poids latents restent fp32 ; seuls les forwards sont ternaires. Pas de
quantification des activations (le KV cache int8 est un choix d'inférence, pas
d'entraînement).

---

## 5. Distillation

### 5.1 Pourquoi offline

Un cache top-k des logits professeur coûte, par token, `k × (2 + 2)` octets
(indices uint16 + valeurs fp16). Pour k=64 : 256 octets/token.

| tokens distillés | k=64 | k=32 |
|---:|---:|---:|
| 10 M | 2,56 Go | 1,28 Go |
| 50 M | 12,8 Go | 6,4 Go |
| 300 M | 76,8 Go | 38,4 Go |

Distiller les 300 M tokens du pré-entraînement est donc **exclu** sur un disque
de travail raisonnable. La stratégie est :

- **CE seule** sur le corpus langue (Wikipédia, QA).
- **KD offline top-k=32** sur un sous-ensemble de 5–10 M tokens (chat, code,
  instructions), là où le format compte.

### 5.2 Professeur

Le professeur doit être :
- fort en français ;
- fort en code ;
- capable de produire des traces d'appels d'outils ;
- exécutable en MLX sur 16 Go.

Candidats [hypothèse, à valider] : Qwen2.5-Coder-1.5B/3B-Instruct (Apache 2.0),
SmolLM2-1.7B-Instruct (Apache 2.0). Un professeur 3B en bf16 pèse ~6 Go, ce qui
laisse peu de place au student 200 M + optimiseur (~4 Go). Un professeur 1,5B
(~3 Go) est plus sûr.

**Décision requise avant tout téléchargement** : le téléchargement d'un
professeur est un téléchargement volumineux, donc soumis à un « go » explicite.

### 5.3 Perte

```python
# logits_student: (B, S, V), logits_teacher_topk: (B, S, K)
# indices: (B, S, K) uint16, valeurs: (B, S, K) fp16
def kd_loss(logits_s, idx_t, val_t, T=2.0, alpha=0.7):
    ls = logits_s / T
    logp_s = ls - mx.logsumexp(ls, axis=-1, keepdims=True)
    # rassembler les logits student aux indices du professeur
    logp_s_k = mx.take_along_axis(logp_s, idx_t, axis=-1)
    p_t = mx.softmax(val_t / T, axis=-1)
    kl = mx.sum(p_t * (mx.log(p_t + 1e-9) - logp_s_k), axis=-1).mean()
    return alpha * (T * T) * kl
```

La CE reste calculée sur les tokens supervisés (masque de perte), pas sur le
padding.

### 5.4 Génération de dialogues synthétiques

Le vrai goulot est le dialogue (voir `DATA_GAP.md` : < 0,3 M tokens). La
distillation ne crée pas de dialogues ; il faut les **générer** avec le
professeur, puis filtrer :

1. amorces FR variées (1 000–5 000 situations) ;
2. génération multi-tours avec le professeur ;
3. filtrage : longueur, langue, absence de répétition, cohérence ;
4. déduplication MinHash ;
5. relecture manuelle d'un échantillon de 200 dialogues.

Cible : 20–50 M tokens de dialogues FR propres. C'est un chantier de plusieurs
jours de génération et de filtrage, pas une après-midi.

---

## 6. Pipeline de données

### 6.1 Ce qui existe

- Wikipédia FR dédupliqué : 99 962 documents, 1,01 Md caractères, **264,2 M
  tokens** [mesuré].
- QA FR : 18,1 M tokens.
- OpenOrca : 30,5 M tokens, 97 % anglais → à exclure du FR-first.
- Code : 5,2 M tokens de Python mal étiqueté → à réétiqueter.
- Chat : 892 tokens → insuffisant.

### 6.2 Ce qu'il faut construire

```
raw → filtre langue (fastText ou heuristique) → dédup MinHash
    → tokenisation → packing seq 2048 → cache uint16
    → split train/val/OOD par document (pas par chunk)
```

Points durs :
- **Split par document**, pas par chunk : sinon les mêmes paragraphes fuient
  entre train et val (c'est exactement le problème de `val_big`).
- **EOS explicite** en fin de document et en fin de tour assistant.
- **Masque de perte** : −100 sur le prompt utilisateur, sur le padding et sur
  les tokens système.
- **Packing** : concaténer les documents avec EOS, pas de padding à l'id 0.
- **Dédup** : MinHash 5-grammes, seuil 0,8, sur le corpus entier.

### 6.3 Format conversationnel cible

```
<|system|>Tu es un assistant...<|end|>
<|user|>...<|end|>
<|assistant|>...<|end|>
```

Tokens spéciaux à ajouter au tokenizer : `<|system|>`, `<|user|>`,
`<|assistant|>`, `<|end|>`, `<|tool_call|>`, `<|tool_result|>`, `<|pad|>`.
Sans eux, le modèle ne peut pas distinguer les rôles de façon fiable.

Pour les outils :

```
<|assistant|><|tool_call|>{"name":"open_app","arguments":{"app":"Safari"}}<|end|>
<|tool|>{"ok":true}<|end|>
<|assistant|>J'ai ouvert Safari.<|end|>
```

Le décodage contraint (JSON Schema → grammaire) est un sujet d'inférence, pas
d'entraînement ; il est traité dans `ROADMAP.md`.

---

## 7. Migration brique par brique

| brique PyTorch | équivalent MLX | difficulté | remarque |
|---|---|---|---|
| `model.py` | `nn.Module` MLX | faible | mêmes opérations |
| `quant.py` absmean | fonctions `mx` | faible | §4 |
| `quant.py` packing | NumPy à l'export | faible | packing hors graphe |
| `distill.py` KL+CE | `mx` + cache memmap | moyenne | §5 |
| boucle d'entraînement | `mx.value_and_grad` | faible | prototype existant |
| optimiseur AdamW | `mlx.optimizers.AdamW` | faible | — |
| scheduler | `mx` ou Python | faible | — |
| checkpoints | `mx.save_safetensors` | faible | atomique à ajouter |
| export `.bin` | NumPy + `write_bin` | moyenne | ajouter le ternaire |
| évaluation PPL | `mx` | faible | garder les mêmes splits |
| génération | `mx` | faible | température, top-p |
| tests de parité | NumPy de référence | moyenne | seuil 1e-4 |

Le prototype `tools/train_hybrid_mlx.py` couvre déjà 60 % de cette table. Le
travail restant est le ternaire, la distillation, le format et l'export.

---

## 8. Performance et temps d'entraînement

Mesure de référence [mesuré] : **4 730 tok/s** pour 47,7 M paramètres, seq 512,
batch 16, sur M4 16 Go.

Extrapolation linéaire en paramètres [hypothèse] :

| modèle | tok/s estimé | 300 M tokens | 1 Md tokens | 4 Md tokens |
|---|---:|---:|---:|---:|
| 47,7 M | 4 730 | 17,6 h | 2,4 j | 9,8 j |
| 130 M | 1 735 | 2,0 j | 6,7 j | 26,7 j |
| 200 M | 1 130 | 3,1 j | 10,3 j | **41 j** |

Conséquence majeure : **4 Md tokens (Chinchilla pour 200 M) est hors de portée
sur un seul M4**. Le plan réaliste est :

- 300 M tokens × 3–4 époques = 0,9–1,2 Md tokens vus, soit **9–12 jours** pour
  200 M ;
- ou 130 M paramètres × 1 Md tokens = **6,7 jours**, plus sûr ;
- ou 100 M paramètres × 1,5 Md tokens = **~7 jours**, avec une capacité moindre.

Ce calcul doit être refait dès la première mesure MLX du modèle cible : la
linéarité en paramètres est une approximation grossière (l'attention et le KV
coûtent aussi).

Mémoire d'entraînement [calculé] pour 200 M :
```
poids fp32        800 Mo
gradients         800 Mo
Adam m + v      1 600 Mo
activations      1 000–2 000 Mo (batch 16 × 512)
total            4,2–5,2 Go   → tient dans 16 Go
```

---

## 9. Export et parité

### 9.1 Chaîne d'export

```
checkpoint MLX (.safetensors, poids latents fp32)
  → ternaire absmean group 64
  → packing 2 bits (4 poids/octet)
  → échelles fp16 (2 octets/poids de groupe)
  → embeddings int8 + échelles fp32
  → header LLM2 (ajouter un champ « quant = ternary »)
  → .bin mmap-able
```

Le format `LLM2` actuel a un champ `version` et `group`. Il faut :
- passer `version` à 2 ;
- ajouter un flag de quantification ;
- documenter l'ordre des tenseurs (déjà implicite dans `engine.py`).

### 9.2 Parité

Test obligatoire : pour un même prompt et une même seed, la sortie MLX
(avant export) et la sortie du moteur C doivent produire les mêmes ids sur 32
tokens, à tolérance près. Le test existant `tools/check_parity_bin_pt.py` fait
cela pour PyTorch ; il faut le porter en MLX.

Seuil : divergence max des logits < 1e-3 en fp32, et ids identiques en greedy.

### 9.3 Ce qui peut casser

- L'ordre des tenseurs (q, k, v, o, gate, up, down) ;
- le layout des têtes (Hq vs Hkv, head_dim) ;
- l'orientation des matrices (`out × in` vs `in × out`) ;
- les échelles par ligne d'embedding vs par colonne ;
- le RoPE (fréquences, ordre pair/impair).

Chacun de ces points doit avoir un test unitaire.

---

## 10. Tests à écrire

| test | critère |
|---|---|
| QAT forward | sortie dans {−s, 0, +s} par groupe |
| STE | gradient non nul sur les poids latents |
| packing | 4 poids → 1 octet, round-trip exact |
| export | `off == len(blob)` |
| parité MLX/C | ids identiques sur 32 tokens |
| PPL | écart < 1 % entre eval MLX et eval NumPy |
| RSS | < 100 Mo en runtime C (voir `BUDGET_100MB.md`) |
| données | aucun 32-gramme train dans OOD |
| format chat | EOS présent, masque correct |
| distillation | KL finie, pas de NaN sur 1 000 steps |

---

## 11. Risques

1. **Le ternaire QAT peut ne pas converger** aussi bien que prévu. Mitigation :
   courbe de PPL fp32 vs ternaire sur 50 M tokens avant d'engager 10 jours.
2. **La distillation offline peut ne pas valoir son coût** (F-C ne bat pas F-B).
   Mitigation : ne distiller que le chat/code, mesurer sur OOD.
3. **Le professeur 3B ne tient pas** avec le student sur 16 Go. Mitigation :
   professeur 1,5B ou distillation en deux passes.
4. **Le temps d'entraînement** peut dépasser 2 semaines. Mitigation : réduire à
   130 M paramètres.
5. **Les données de dialogue** peuvent rester insuffisantes même après
   génération. Mitigation : accepter un assistant à domaine étroit.

---

## 12. Verdict honnête

La migration MLX est **la partie la plus simple du projet**. Le prototype tourne
déjà, MLX couvre toutes les opérations nécessaires, et le portage des briques
(quantification, distillation, export) est mécanique.

Les deux vrais obstacles ne sont pas MLX :

- **le temps** : 200 M paramètres × 1 Md tokens = ~10 jours sur un seul M4 ;
- **les données** : moins de 0,3 M tokens de dialogue pour 264 M tokens de
  langue.

La migration est donc **faisable et recommandée**, mais elle ne résout ni le
budget de données ni la capacité du modèle. Elle doit être menée en parallèle
d'un chantier données, sinon elle produira un excellent modèle de langue
ternaire qui ne conversera toujours pas.
