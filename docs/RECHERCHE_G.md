# Recherche G — leviers généraux pour un petit modèle ternaire français

**Date de consultation : 2026-10-21.** Sources primaires ouvertes pendant la
phase 1 (arXiv, cartes de datasets, blogs techniques, page de prix). Ce
document distingue explicitement ce qui est **ÉTABLI** dans la source, ce qui
est **HYPOTHÈSE** pour notre cas, et ce qui relève du **CONTEXTE** (chiffre
isolé, non reproduit).

Objet : répondre à « comment faire apprendre plus vite et mieux, à ≤ 100 Mo,
avec une qualité hors distribution, une langue propre et une utilité réelle »
sans se limiter à des réglages `lr`/`α`.

## 0. Méthode et limites de cette revue

- Une source n'est citée que si sa page a été ouverte et lue ; les affirmations
  sont reformulées, pas recopiées.
- Les résultats de BitNet sont majoritairement obtenus sur des modèles de
  1–3 B paramètres et/ou des corpus anglais. Leur transfert à un modèle
  français de 25–73 M paramètres est une **hypothèse**, pas un résultat.
- Les chiffres de vitesse (MLX/MPS, GPU loués) dépendent du matériel, de la
  version de PyTorch et du batch ; ils servent d'ordre de grandeur.
- Les prix GPU sont datés ; ils changent.

## 1. Ternaire, BitNet et capacité effective

### 1.1 Établi

| fait | source | ce que ça dit exactement |
|---|---|---|
| BitNet entraîne des poids 1-bit dès le départ avec BitLinear, et suit une loi d'échelle proche du fp16 | Wang et al., *BitNet*, arXiv:2310.11453 | Architecture et stabilité ; pas de garantie pour 25 M. |
| BitNet b1.58 utilise des poids ternaires {-1, 0, 1} et égale le fp16 à taille et tokens égaux, dans leur régime | Ma et al., *The Era of 1-bit LLMs*, arXiv:2402.17764 | Le résultat phare est à grande échelle ; il ne dit pas que 25 M ternaire = 25 M fp16. |
| BitNet b1.58 2B4T (2 B paramètres, 4 T tokens) atteint le niveau de modèles fp16 de taille similaire | Ma et al., *BitNet b1.58 2B4T*, arXiv:2504.12285 | Recette : LR en deux phases (cosine haute puis cooldown abrupt), weight decay en deux phases (pic 0,1 puis 0), données DCLM + FineWeb-Edu + synthétique, données de haute qualité concentrées dans le cooldown. |
| À petite échelle (100 K–48 M), le 1.58-bit a une capacité effective d'environ **la moitié** du 16-bit : il faut ~**2× la taille cachée** pour égaler un modèle 16-bit | Nielsen & Schneider-Kamp, *BitNet b1.58 Reloaded*, arXiv:2407.09527 | C'est le résultat le plus directement pertinent pour nous. Les auteurs balaient aussi `lr` 1e-3–1e-2 et plusieurs weight decays ; la variante « median » converge plus lentement que « mean ». |
| Un pré-entraînement 16-bit suivi d'un passage 1.58-bit (QAT) est préférable à un entraînement 1.58-bit intégral ; garder l'optimiseur et phaser la quantification réduit les pics de loss | Nielsen et al., *Continual Quantization-Aware Pre-Training*, arXiv:2502.11895 | Résultat sur 11 tâches aval ; les pics restent compensables par plus d'entraînement. |
| Une partie des poids ternaires reste piégée à la frontière du zéro (« deadzone trapping ») et reçoit des gradients peu informatifs ; Tequila les réactive comme biais dynamiques | *Tequila*, arXiv:2509.23809 | Explique un churn qui passe par 0 plutôt que par des retournements directs ±1→∓1. |

### 1.2 Hypothèses pour notre modèle

- **H1.1** — Le plateau OOD de t08 (25 M, 18,9 M de poids ternaires) est
  d'abord un plateau de **capacité effective** : ~12 M équivalent fp16 selon
  la règle des 2×. Un base 768 (72,9 M paramètres, ~36 M équivalent fp16)
  devrait déplacer l'OOD, même à recette égale.
- **H1.2** — Le churn mesuré (20,1 % de poids ternaires cumulés différents de
  t08 pour B, seulement 0,06 % de retournements directs ±1→∓1) est cohérent
  avec le deadzone trapping. Une optimisation ternaire-aware (Tequila, biais
  continus, ou quantification progressive) pourrait récupérer de la capacité
  sans changer le budget.
- **H1.3** — La recette 2B4T (deux phases LR, WD 0,1→0, cooldown sur données
  propres) est conçue pour un 2 B ; à 25 M, elle risque de sur-ajuster. Elle
  est plus prometteuse sur le base 768 que sur le tiny.

### 1.3 Implication

Le levier « capacité » (base 768) est le seul qui attaque directement le
plafond OOD dans l'enveloppe < 100 Mo. Les leviers « recette ternaire » sont
réels mais d'effet attendu plus faible à cette taille ; ils ne doivent pas
être testés avant d'avoir une mesure OOD propre.

## 2. Optimisation et schedules

### 2.1 Établi

| fait | source | conséquence |
|---|---|---|
| WSD (warmup–stable–decay) garde un LR constant, puis décroît rapidement ; la loss reste haute pendant la phase stable et chute au decay | Wen et al., *WSD*, arXiv:2410.05192 | Le gain de B (WSD, WD→0) est cohérent avec ce mécanisme ; WSD permet de brancher à n'importe quel budget. |
| Muon (orthogonalisation matricielle) + weight decay + ajustement d'échelle par paramètre atteint ~**2× l'efficacité de calcul** d'AdamW en régime compute-optimal | Liu et al., *Muon is Scalable*, arXiv:2502.16982 | Levier de vitesse/convergence, mais validé à grande échelle ; l'implémentation MPS et l'interaction avec la quantification ternaire ne sont pas documentées. |
| μP/μTransfer : les hyperparamètres optimaux restent stables quand la taille change ; on peut tuner sur 13 M et transférer à 350 M, ou 40 M → 6,7 B | Yang et al., *Tensor Programs V*, arXiv:2203.03466 | Permet de tuner le base 768 sur le tiny 25 M, à condition que la paramétrisation et la quantification ne cassent pas les hypothèses μP. |
| Chinchilla : à budget de calcul fixe, doubler les paramètres doit doubler les tokens ; règle pratique ~20 tokens/param | Hoffmann et al., arXiv:2203.15556 | t08 a vu ~372 M token-passes (22 740 steps × 16 384) pour 25 M paramètres, soit ~15 tokens/param brut ; rapporté à ~12 M équivalent fp16, c'est ~31 tokens/param effectif. Il n'est donc pas massivement sous-entraîné au sens Chinchilla. |
| Avec une demande d'inférence, il est optimal d'entraîner **plus petit et plus longtemps** ; la qualité continue de progresser jusqu'à ~10 000 tokens/param | Sardana et al., *Beyond Chinchilla-Optimal*, arXiv:2401.00448 | Justifie d'entraîner le base 768 sur plusieurs centaines de millions de tokens, pas seulement 1×Chinchilla. |

### 2.2 Hypothèses

- **H2.1** — À 25 M, le facteur limitant n'est pas le schedule (F-A vs F-B le
  montrent) mais la capacité et la distribution. Muon ne changera pas le
  plafond OOD s'il ne change pas la capacité. Le base 768 (72,9 M, ~36 M
  équivalent fp16) à 700–800 M token-passes reste ~20 tokens/param effectif :
  c'est le minimum Chinchilla, pas un sur-entraînement.
- **H2.2** — Sur le base 768, un schedule WSD + WD→0 + cooldown sur données
  propres (littérature + QA) donnera plus qu'un cosine constant.
- **H2.3** — μP peut réduire le coût de tuning du base, mais son interaction
  avec l'absmean par groupe de 64 et le STE doit être validée sur un petit
  modèle avant tout run long.

## 3. Données

### 3.1 Établi

| fait | source | conséquence |
|---|---|---|
| FineWeb2 : pipeline de curation adaptable à toute langue, 20 To, 5 Md documents, 1 000+ langues, avec filtrage qualité, déduplication et rééquilibrage | Penedo et al., arXiv:2506.20920 ; carte `HuggingFaceFW/fineweb-2` | Source possible pour élargir le français au-delà de Wikipédia. La configuration `fra_Latn` fait **360 103 218 lignes** (~540 Go de parquet), donc un sous-ensemble seulement est réaliste. |
| TinyStories : des modèles de **moins de 10 M** paramètres produisent des histoires cohérentes et grammaticales quand le domaine est étroit et synthétique | Eldan & Li, arXiv:2305.07759 | La distribution des données peut compenser une capacité faible pour une tâche étroite ; argument fort pour un corpus QA-passage synthétique. |
| phi-1 : 1,3 B paramètres, 6 B tokens web filtrés + 1 B tokens de « manuels » synthétiques ; phi-1-small (350 M) atteint 45 % HumanEval avec le même pipeline | Gunasekar et al., *Textbooks Are all You Need*, arXiv:2306.11644 | La qualité/structuration des données compte autant que la taille ; un petit modèle peut apprendre une compétence étroite. |
| Cosmopedia : génération de manuels/scénarios synthétiques à grande échelle pour le pré-entraînement | blog Hugging Face, *Cosmopedia* | Le synthétique est une option, mais doit être filtré et évalué ; pas de garantie de transfert. |

### 3.2 Données locales et OOD

- Le corpus d'entraînement E est du **Wikipédia français** dédupliqué
  (`/Volumes/Lexar/clean-corpus/dedup/tranche_00..09.jsonl`, ~322 M tokens
  SmolLM pour t01–t09).
- Le jeu OOD (`runs/abl_data/ood.jsonl`, 340 docs) est de la **littérature
  française** (Madame Bovary, etc.). C'est le seul jeu de transfert propre
  mesuré en phase 0 : 0 % de 32-grammes communs avec le train.
- Le catalogue Gutenberg local (`/Volumes/Lexar/rag-data/livres/pg_catalog.csv`)
  recense **465 livres français**. C'est une source publique, gratuite, qui
  recoupe directement la distribution OOD (romans), avec un risque de fuite à
  contrôler (Madame Bovary est dans Gutenberg).

### 3.3 Hypothèses

- **H3.1** — Ajouter 10–15 % de littérature française au train devrait
  améliorer l'OOD sans dégrader la val in-domain, si le mélange est
  suffisamment entraîné et si les livres OOD sont exclus par recouvrement
  n-grammes.
- **H3.2** — Un corpus QA-passage synthétique construit sur le train
  (passage + question + réponse extraite) peut apprendre la copie/extraction
  à 25 M ; c'est le levier le plus direct pour l'« utilité réelle ».
- **H3.3** — FineWeb2 français apporterait de la diversité, mais 540 Go et un
  filtrage à refaire rendent le coût élevé pour ce projet ; la littérature
  Gutenberg est un meilleur premier pas, ciblé sur l'OOD mesuré.

## 4. Tokenizer

### 4.1 Établi

| fait | source | conséquence |
|---|---|---|
| La tokenisation multilinguale impose des « taxes » : fertilité élevée, fragmentation, dépendance aux diacritiques, mots composés | blog HF, *Tokenization is Killing our Multilingual LLM Dream* (2026) | Un tokenizer anglais appliqué au français peut coûter cher en tokens et en morphologie. |
| La tokenisation est fragile aux typos et variations de format ; l'échelle atténue mais ne supprime pas le biais ; BPE-dropout peut aider | Chai et al., *Tokenization Falling Short*, EMNLP Findings 2024, arXiv:2406.11687 | À petite échelle, le tokenizer est un levier de qualité, pas un détail. |
| Notre BPE FR 16k : **1,629 token/mot** vs SmolLM **2,211** (−26 %), fragmentation 23,9 % vs 53,5 % ; `capitale` et `capitaine` deviennent un seul token | mesure G, `docs/TOKENIZER_FR_G.md` | Gain de vitesse et de morphologie ; nécessite un ré-entraînement des embeddings (ou une initialisation alignée). |

### 4.2 Hypothèses

- **H4.1** — À budget de tokens fixe, un BPE 16k voit ~35 % plus de texte
  que le vocabulaire SmolLM réduit, et produit moins de mots fragmentés.
- **H4.2** — Le gain de BPC d'un tokenizer natif est plus fort sur la
  morphologie et l'OOD que sur la PPL in-domain, parce qu'il réduit les
  séquences aberrantes et les `unk`.
- **H4.3** — Changer de tokenizer casse l'alignement avec le professeur
  SmolLM ; il faut soit entraîner sans KD, soit utiliser un alignement
  lexical (VocAgnoLM), soit initialiser les embeddings par correspondance de
  surfaces. C'est un coût d'ingénierie réel, pas un simple flag.

## 5. Distillation

### 5.1 Établi

| fait | source | conséquence |
|---|---|---|
| MiniLLM remplace la KL forward par une **KL reverse** et optimise on-policy ; meilleur que la KD forward pour la génération, de 120 M à 13 B | Gu et al., arXiv:2306.08543 | La KD forward α=0,7 de la chaîne E n'est pas la seule option ; la KD reverse préserve mieux les régions de faible probabilité. |
| GKD entraîne l'élève sur ses propres séquences, avec feedback du professeur ; réduit l'exposure bias | Agarwal et al., arXiv:2306.13649 | On-policy est plus coûteux (génération), mais peut améliorer l'OOD. |
| VocAgnoLM aligne les tokens élève/professeur de vocabulaire différent et utilise la loss du professeur ; +46 % vs continual pretraining naïf avec seulement ~6 % de vocabulaire commun | arXiv:2503.19123 | Solution si BPE 16k + professeur SmolLM. |
| La KD offline (top-k logits mis en cache) égale la KD online à loss quasi identique, ~29 % plus rapide par itération et jusqu'à 41 % de débit en plus ; la KL chunked permet 4× le contexte (32 768) | arXiv:2608.03796 | Réduit la mémoire et le coût de la KD sur M4 ; permet des contextes plus longs pour la QA. |

### 5.2 Hypothèses

- **H5.1** — La KD forward préserve l'OOD (F-A : −0,1714 vs t08 −0,1708)
  parce qu'elle maintient la distribution du professeur ; la CE seule (F-B)
  sur-ajuste Wikipédia et perd l'OOD. Toute recette qui abandonne la KD doit
  compenser par des données diverses.
- **H5.2** — Une KD reverse/on-policy à 25 M améliorerait la calibration et
  l'OOD, mais le coût de génération sur M4 est prohibitif ; la KD offline
  top-k est le compromis réaliste.
- **H5.3** — Avec un tokenizer natif, la KD devient plus complexe ; pour le
  base from-scratch, la CE sur données mélangées peut suffire si le corpus
  est assez gros (plusieurs centaines de millions de tokens).

## 6. Utilité réelle : QA extractive, passage, petit modèle

### 6.1 Établi

| fait | source | conséquence |
|---|---|---|
| Les « induction heads » implémentent [A][B] … [A] → [B] et sont liés à l'in-context learning ; preuve causale sur de petits modèles attention-only | Olsson et al., arXiv:2209.11895 | La copie/extraction depuis un passage est un mécanisme accessible aux petits transformeurs. |
| La performance dépend de la position de l'information : meilleure au début ou à la fin, dégradée au milieu | Liu et al., *Lost in the Middle*, arXiv:2307.03172 | Pour une QA passage+question, placer la question et la réponse près des bords ; ne pas enterrer le passage. |
| Le padding peut représenter jusqu'à 50 % des tokens dans des datasets NLP variable-length ; le packing donne ~2× sur la phase 2 de BERT | Krell et al., arXiv:2107.02027 | Le pré-entraînement actuel concatène déjà les documents ; pour la QA SFT, il faudra packer les exemples, pas padder. |
| Les SLM sont jugés suffisants, adaptés et économiques pour des tâches spécialisées répétitives dans les agents | Belcak et al., *Small Language Models are the Future of Agentic AI*, arXiv:2506.02153 | Position, pas preuve : justifie de viser une utilité étroite (QA passage) plutôt que la conversation générale. |

### 6.2 Hypothèses

- **H6.1** — Un 25 M peut apprendre à répondre à une question dont la réponse
  est **dans le passage**, si le format est stable et le passage court
  (≤ 100 tokens). C'est le test le plus direct de l'utilité.
- **H6.2** — Un base 768 avec `seq_max=256` peut traiter des passages plus
  longs et améliorer le exact-match, mais la capacité 72,9 M reste le facteur
  limitant ; il ne faut pas attendre un raisonnement multi-saut.
- **H6.3** — Un benchmark QA-passage généré par templates peut être appris
  par cœur si le test partage les mêmes templates. Il faut séparer
  templates/entités train et test, et inclure un test OOD (romans) pour
  mesurer le transfert.

## 7. Vitesse, matériel et coût

### 7.1 Établi / contexte

| fait | source | conséquence |
|---|---|---|
| Sur M2 Ultra, GPT-large batch 8 : MLX 2,710 s vs MPS 3,239 s par passe (~1,2× pour MLX) ; M2 Max ~1,25×, M2 Pro ~1,23× | Lin et al., *Profiling Apple Silicon Performance for ML Training*, arXiv:2501.14925 | MLX n'est pas un gain massif pour ce type de workload. |
| Sur un Mac 16 Go, Qwen 0.5B LoRA : PyTorch MPS **5,35 it/s** vs MLX 4-bit 0,94 et MLX 16-bit 1,30 | blog HF, *Fine-Tuning Coding Agents on a 16GB Mac* (2026) | Contradiction avec le papier : MLX n'est pas automatiquement plus rapide ; le résultat dépend du modèle/batch/précision. |
| `torch.compile` sur MPS est encore un chantier/prototype (issue de suivi #150121) | GitHub pytorch/pytorch#150121 | Ne pas compter dessus pour un run long sans validation locale. |
| RunPod, page du 8 octobre 2026 : A100 80 Go **1,19–1,79 $/h**, H100 PCIe **1,99–2,89 $/h**, RTX 4090 **0,34–0,89 $/h**, RTX 3090 **0,22–0,50 $/h** (Community/Secure) | runpod.io/pricing | Un base 768 sur 4090 reste dans une enveloppe de quelques dizaines de dollars. |

### 7.2 Mesures locales (phase 0.4)

| modèle | tokens/s (génération, 128 tokens) | Δ RSS après génération | poids packés |
|---|---:|---:|---:|
| E-t08 (25 M, d=512, L=8, seq 128) | **107,4** | **+28,4 Mo** | 12,13 Mo |
| base 768 (72,9 M, d=768, L=12, seq 256) | **46,6** | **+52,0 Mo** | 29,25 Mo |

Entraînement observé sur M4 : ~3 184 tokens/s en CE seule (F-B), ~1 821
tokens/s avec KD α=0,7 (F-A). Le base 768 devrait être plus lent par token,
mais plus efficace par unité de qualité si la capacité est le facteur
limitant.

### 7.3 Hypothèses

- **H7.1** — Le passage sur un 4090 loué donne un facteur 5–15× sur
  l'entraînement du base, mais le petit modèle est partiellement limité par
  le CPU/overhead ; il faut mesurer avant de louer.
- **H7.2** — Le budget de 100 Mo est tenu par le base 768 (delta RSS 52 Mo,
  export 29,25 Mo) ; le risque est le cache KV et les activations si on passe
  à `seq 512`.
- **H7.3** — MLX ne vaut pas le portage pour ce projet avant d'avoir un
  gain mesuré > 1,5× sur notre modèle ; le chemin GPU loué est plus simple.

## 8. Synthèse : ce que la littérature ne tranche pas

1. **Aucun papier ne mesure un ternaire français de 25–73 M** sur un corpus
   Wikipédia + littérature avec une contrainte de 100 Mo. Les résultats
   BitNet sont une extrapolation.
2. **Aucun papier ne dit** si la capacité effective (2× la largeur) ou la
   recette (WSD, KD) est le facteur dominant à 25 M ; F suggère la capacité
   pour l'OOD, mais avec un seul seed et un jeu val_big contaminé.
3. **Aucun papier ne donne** le gain d'un tokenizer français natif à budget
   de calcul fixe sur l'OOD ; notre mesure de fertilité est un préalable, pas
   une preuve de qualité.
4. **Aucun papier ne garantit** qu'un 25 M peut faire de la QA passage
   multi-phrases ; TinyStories et les induction heads le rendent plausible,
   mais le exact-match reste à mesurer.
5. **Les prix GPU et les débits MPS/MLX** sont datés et variables ; le plan
   doit inclure une mesure de débit avant tout engagement.

## 9. Références ouvertes

| # | référence | URL | usage dans ce document |
|---|---|---|---|
| 1 | Wang et al., *BitNet: Scaling 1-bit Transformers*, arXiv:2310.11453 | https://arxiv.org/abs/2310.11453 | BitLinear, 1-bit from scratch |
| 2 | Ma et al., *The Era of 1-bit LLMs*, arXiv:2402.17764 | https://arxiv.org/abs/2402.17764 | ternaire {-1,0,1}, égalité fp16 à grande échelle |
| 3 | Ma et al., *BitNet b1.58 2B4T*, arXiv:2504.12285 | https://arxiv.org/abs/2504.12285 | recette 2 phases LR/WD, données |
| 4 | Nielsen & Schneider-Kamp, *BitNet b1.58 Reloaded*, arXiv:2407.09527 | https://arxiv.org/abs/2407.09527 | 2× largeur à petite échelle, lr/WD |
| 5 | Nielsen et al., *Continual QAT 16→1.58*, arXiv:2502.11895 | https://arxiv.org/abs/2502.11895 | QAT progressif, pics de loss |
| 6 | *Tequila: Trapping-free Ternary Quantization*, arXiv:2509.23809 | https://arxiv.org/abs/2509.23809 | deadzone trapping |
| 7 | Wen et al., *WSD*, arXiv:2410.05192 | https://arxiv.org/abs/2410.05192 | warmup–stable–decay |
| 8 | Liu et al., *Muon is Scalable*, arXiv:2502.16982 | https://arxiv.org/abs/2502.16982 | ~2× efficacité vs AdamW |
| 9 | Yang et al., *Tensor Programs V*, arXiv:2203.03466 | https://arxiv.org/abs/2203.03466 | μP/μTransfer |
| 10 | Hoffmann et al., *Chinchilla*, arXiv:2203.15556 | https://arxiv.org/abs/2203.15556 | tokens/paramètres |
| 11 | Sardana et al., *Beyond Chinchilla-Optimal*, arXiv:2401.00448 | https://arxiv.org/abs/2401.00448 | petit et longtemps, inférence |
| 12 | Penedo et al., *FineWeb2*, arXiv:2506.20920 | https://arxiv.org/abs/2506.20920 | corpus multilingue, pipeline |
| 13 | carte `HuggingFaceFW/fineweb-2` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | 360 103 218 lignes `fra_Latn` |
| 14 | Eldan & Li, *TinyStories*, arXiv:2305.07759 | https://arxiv.org/abs/2305.07759 | <10 M cohérent sur domaine étroit |
| 15 | Gunasekar et al., *Textbooks Are All You Need*, arXiv:2306.11644 | https://arxiv.org/abs/2306.11644 | données filtrées + synthétiques |
| 16 | blog HF, *Cosmopedia* | https://huggingface.co/blog/cosmopedia | manuels synthétiques |
| 17 | blog HF, *Tokenization is Killing our Multilingual LLM Dream* (2026) | https://huggingface.co/blog/omarkamali/tokenization | taxes de tokenisation |
| 18 | Chai et al., *Tokenization Falling Short*, arXiv:2406.11687 | https://arxiv.org/abs/2406.11687 | robustesse, BPE-dropout |
| 19 | Gu et al., *MiniLLM*, arXiv:2306.08543 | https://arxiv.org/abs/2306.08543 | KL reverse, on-policy |
| 20 | Agarwal et al., *GKD*, arXiv:2306.13649 | https://arxiv.org/abs/2306.13649 | on-policy, exposure bias |
| 21 | *VocAgnoLM*, arXiv:2503.19123 | https://arxiv.org/abs/2503.19123 | alignement lexical cross-tokenizer |
| 22 | *Offline Top-K Logits + Fused Chunked KL*, arXiv:2608.03796 | https://arxiv.org/abs/2608.03796 | KD offline, mémoire, contexte |
| 23 | Olsson et al., *In-context Learning and Induction Heads*, arXiv:2209.11895 | https://arxiv.org/abs/2209.11895 | copie [A][B]…[A]→[B] |
| 24 | Liu et al., *Lost in the Middle*, arXiv:2307.03172 | https://arxiv.org/abs/2307.03172 | position de l'information |
| 25 | Krell et al., *Efficient Sequence Packing*, arXiv:2107.02027 | https://arxiv.org/abs/2107.02027 | padding/packing |
| 26 | Belcak et al., *Small Language Models are the Future of Agentic AI*, arXiv:2506.02153 | https://arxiv.org/abs/2506.02153 | utilité des SLM |
| 27 | Lin et al., *Profiling Apple Silicon Performance for ML Training*, arXiv:2501.14925 | https://arxiv.org/abs/2501.14925 | MLX vs MPS |
| 28 | blog HF, *Fine-Tuning Coding Agents on a 16GB Mac* (2026) | https://huggingface.co/blog/harshitkgupta/fine-tuning-coding-agents-on-mac-pytorch-mps-mlx | MPS vs MLX LoRA |
| 29 | GitHub, *torch.compile on MPS progress tracker* #150121 | https://github.com/pytorch/pytorch/issues/150121 | statut prototype |
| 30 | RunPod, *GPU Cloud Pricing* (8 oct. 2026) | https://www.runpod.io/pricing | prix A100/H100/4090/3090 |
