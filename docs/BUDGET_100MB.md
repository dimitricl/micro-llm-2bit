# Budget 100 Mo de RAM totale

**Date** : 2026-10-27
**Objet** : établir ce qui tient, et ce qui ne tient pas, dans **100 Mo de RAM
totale** (poids + KV cache + activations + tokenizer + runtime), pour une
inférence C portable sans Python ni MLX.
**Unité** : le Mo vaut 10⁶ octets, comme dans les mesures du dépôt
(`12 125 100` octets = `12,13 Mo`).

---

## 1. Le malentendu à lever d'abord

Le dépôt utilise depuis l'origine le critère **« delta RSS < 100 Mo »**, pas
**« RSS totale < 100 Mo »**. La différence est décisive.

Mesure de référence, `logs/F/step1-bench-Et08-128.log` [mesuré] :

```
[bench] RSS réel : baseline=197,9 Mo, après chargement=198,2 Mo (+0,3),
                   après génération=225,9 Mo (+28,0)
[bench] DELTA vs baseline : +28,0 Mo / 100 Mo -> OK
[bench] TOTAL (structurel) : 12,12 Mo / 100 Mo -> OK
```

Trois chiffres, trois sens différents :

| chiffre | valeur | ce qu'il mesure | verdict vs 100 Mo **total** |
|---|---:|---|---|
| structurel | 12,12 Mo | poids + KV + activations théoriques | ✅ |
| delta RSS | +28,0 Mo | ce que le processus gagne après génération | ✅ |
| **RSS absolue** | **225,9 Mo** | **tout le processus** | ❌ **2,26×** |

Le « TOTAL 12,12 Mo » du log est un budget **du modèle**, pas du processus. Il
ignore 197,9 Mo de Python + NumPy, et il ignore aussi les 15,9 Mo d'écart entre
le delta mesuré (28,0) et le structurel (12,1).

**Conclusion immédiate** : l'artefact actuel ne respecte pas la contrainte de
100 Mo de RAM totale. Il respecte une contrainte plus faible, le delta. Toute la
suite de ce document raisonne en **RSS absolue**.

---

## 2. Pourquoi +28 Mo et pas +12,1 Mo

Le fichier `.bin` fait 12,13 Mo et le KV 0,53 Mo. Le delta mesuré est de 28,0 Mo.
L'écart de ~15,4 Mo vient de :

1. **les pages du fichier mmap touchées** : 12,13 Mo, comptées en RSS ;
2. **les temporaires NumPy** : blocs de logits, copies contiguës, produits
   matriciels ; `LOGIT_BLOCK = 2048` lignes × 512 colonnes × 4 octets = 4,2 Mo
   par bloc temporaire ;
3. **l'allocateur glibc/macOS** : les arènes ne rendent pas la mémoire au système
   après génération ;
4. **les objets Python** : listes de tenseurs, `PackedLinear`, caches.

Autrement dit, **le runtime Python coûte à lui seul plus que le modèle**. C'est
la raison pour laquelle la cible 100 Mo totale impose un runtime natif.

---

## 3. Formules

### 3.1 Poids

Pour un modèle de `V` vocabulaire, `d` dimension, `L` couches, `Hq` têtes Q,
`Hkv` têtes KV, `F` FFN, `Dh = d / Hq` :

```
embeddings        = V × d                     (int8 : 1 octet/param)
échelles emb.     = V × 4                     (fp32, par ligne)

par couche :
  q, o            = d × d
  k, v            = Hkv × Dh × d
  gate, up        = F × d
  down            = d × F
  linéaires       = somme des 7 matrices

poids linéaires   = linéaires × (0,25 + s / G)  octets/param
                    avec s = 2 (fp16) ou 4 (fp32), G = groupe
normes            = (2L + 1) × d × 4
```

Le terme `0,25` est le packing 2 bits ; `s / G` est le coût des échelles.
Avec `G = 64` et `s = 2` : **0,28125 octet/param = 2,25 bits/poids**.
Avec `s = 4` (format actuel) : **0,3125 octet/param = 2,5 bits/poids**.

### 3.2 KV cache

```
KV = 2 × L × S × Hkv × Dh × b        (b = 1 octet en int8)
   + 2 × L × S × Hkv × 4            (échelles fp32)
```

Pour E-t08 (`L=8, Hkv=4, Dh=64`) :
```
S=128  : 2×8×128×256  + 2×8×128×4  =   524 288 + 32 768 =   557 056 o = 0,56 Mo
S=2048 : 2×8×2048×256 + 2×8×2048×4 = 8 388 608 + 524 288 = 8 912 896 o = 8,91 Mo
```

Le KV croît **linéairement en S et en L**, et proportionnellement à `Hkv`. C'est
le premier poste compressible via GQA.

### 3.3 Activations

Pour un batch de 1 :
```
résidu          = d × 4
FFN intermédiaire = F × 4
scores d'attention = S × Hq × 4
logits          = V × 4  (ou un bloc de 2048 × 4)
marge de travail  ≈ 2 Mo
```

À `d=1024, F=2816, V=24576, S=2048` : 4 Ko + 11 Ko + 131 Ko + 98 Ko + 2 Mo
≈ **2,2 Mo**. Les activations ne sont jamais le problème en batch 1.

### 3.4 Tokenizer et runtime

| poste | taille | commentaire |
|---|---:|---|
| `tokenizer.json` actuel | 1,12 Mo | JSON verbeux |
| tokenizer natif compact | 0,6–1,2 Mo | vocab + merges binaires |
| binaire C statique | 0,3–0,8 Mo | code + tables |
| libc / loader | 1,5–3 Mo | incompressible |
| pile + garde | 0,5–1 Mo | — |
| **total** | **2,9–6 Mo** | [hypothèse] |

Un runtime C réel ne descend pas sous ~3 Mo de RSS. C'est le plancher.

---

## 4. Ventilation du modèle actuel (E-t08)

Mesures et calculs exacts :

| poste | octets | Mo | part |
|---|---:|---:|---:|
| header | 44 | 0,00004 | 0,0 % |
| embeddings int8 | 6 144 000 | 6,14 | 50,7 % |
| échelles d'embedding fp32 | 48 000 | 0,05 | 0,4 % |
| linéaires packés | 4 718 592 | 4,72 | 38,9 % |
| échelles de linéaires fp32 | 1 179 648 | 1,18 | 9,7 % |
| normes | 34 816 | 0,03 | 0,3 % |
| **poids** | **12 125 100** | **12,13** | 100 % |
| KV S=128 | 557 056 | 0,56 | — |
| activations | ~60 000 | 0,06 | — |
| **structurel** | **12 742 156** | **12,74** | — |
| **RSS absolue mesurée** | **225 900 000** | **225,9** | — |

Le poids des embeddings (6,14 Mo) dépasse celui de tous les linéaires packés
(4,72 Mo). À petit vocabulaire et petite dimension, **le vocabulaire est le
premier poste**.

---

## 5. Combien de paramètres tiennent dans 100 Mo ?

Hypothèses de calcul :
- embeddings int8, liés à la sortie ;
- linéaires ternaires, échelles fp16, groupe 64 ;
- KV int8 + échelles fp32 ;
- activations + marge 2,2 Mo ;
- tokenizer 1,2 Mo ;
- runtime 1,5 Mo ;
- **aucune marge de sécurité** (à ajouter : 5–10 %).

| configuration | params | poids | KV | act. | tok. | runtime | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| E-t08, S=128 | 25,0 M | 12,1 | 0,6 | 0,1 | 1,2 | 1,5 | **15,5** |
| E-t08, S=2048 | 25,0 M | 12,1 | 8,9 | 2,1 | 1,2 | 1,5 | **25,8** |
| A : V16k, d768, L12, F2048, S1024 | 88,1 M | 34,0 | 6,7 | 2,1 | 1,2 | 1,5 | **45,5** |
| A, S=2048 | 88,1 M | 34,0 | 13,4 | 2,2 | 1,2 | 1,5 | **52,3** |
| C : V16k, d896, L14, F2304, S1536 | 130,3 M | 47,4 | 11,7 | 2,2 | 1,2 | 1,5 | **64,0** |
| B : V24k, d1024, L16, F2816, S1024 | 205,6 M | 76,1 | 8,9 | 2,2 | 1,2 | 1,5 | **89,9** |
| B, S=2048 | 205,6 M | 76,1 | 17,8 | 2,2 | 1,2 | 1,5 | **98,9** |
| B', V16k, d1024, L16, S2048 | 197,2 M | 67,7 | 17,8 | 2,2 | 1,2 | 1,5 | **90,4** |

**Plafond pratique** : ~200 M paramètres à contexte 2048, ou ~205 M à contexte
1024. Au-delà, il faut rogner le KV, le vocabulaire ou les embeddings.

### 5.1 La variante la plus sûre

`V=16 384, d=1024, L=16, Hq=16, Hkv=4, F=2816, S=2048` :
- 197,2 M paramètres ;
- 67,7 Mo de poids ;
- 17,8 Mo de KV ;
- **90,4 Mo au total**, soit 9,6 Mo de marge.

C'est la configuration recommandée pour la phase 2. Elle laisse de la place pour
un tokenizer plus gros, un buffer d'outils et les imprécisions du runtime.

### 5.2 La variante « capacité »

`V=24 576, d=1024, L=16, F=2816, S=2048` : 205,6 M paramètres, **98,9 Mo**.
Trop juste. Il faudrait descendre le KV en int4 (8,9 Mo au lieu de 17,8) pour
retrouver ~90 Mo, au prix d'une perte de qualité à mesurer.

---

## 6. Où sont les leviers

| levier | gain | coût |
|---|---:|---|
| échelles fp32 → fp16 | −0,6 Mo (E-t08), −2,1 Mo (200 M) | précision négligeable |
| groupe 64 → 128 | −1,2 Mo (200 M) | qualité à mesurer |
| embeddings int8 → 2 bits | −19 Mo (V16k, d1024) | forte perte sur les tokens rares |
| embeddings factorisés (`d_emb=256` puis projection) | −12 Mo | qualité à mesurer |
| vocab 24k → 16k | −8,4 Mo | moins de code/FR rare |
| GQA 4 → 2 têtes KV | −8,9 Mo à S=2048 | perte d'attention |
| KV int8 → int4 | −8,9 Mo à S=2048 | perte de qualité |
| contexte 2048 → 1024 | −8,9 Mo | context divisé par 2 |
| attention fenêtrée | −50 % du KV au-delà de la fenêtre | implémentation |
| runtime C au lieu de Python | **−198 Mo** | chantier |

Le levier dominant est évidemment le **runtime natif** : il fait passer de
225,9 Mo à ~15–20 Mo pour le même modèle.

---

## 7. Ce qui ne rentre pas

| configuration | RSS absolue | verdict |
|---|---:|---|
| E-t08 + Python | 225,9 Mo | ❌ 2,26× |
| base 72,9 M + Python | **354,9 Mo** (banc : +159,3 ; baseline 195,6) | ❌ |
| base 72,9 M + runtime C | ~34 Mo | ✅ |
| 200 M + runtime C | ~90–99 Mo | ✅ (juste) |
| 400 M + runtime C | ~180 Mo | ❌ |
| 1 Md + runtime C | ~350 Mo | ❌ |

Le `base` (72,9 M) tient largement dans 100 Mo **avec un runtime C**, alors
qu'il explose avec Python (+159,3 Mo de delta). C'est une preuve de plus que le
probleme est le runtime, pas le format.

---

## 8. Le piège du mmap et de la page cache

Un fichier mmapé n'est pas chargé entièrement : seules les pages touchées
deviennent résidentes. Sur un modèle de 76 Mo, la RSS peut donc afficher 40 Mo
après un prompt court, puis monter à 76 Mo après plusieurs générations.

Pour un budget **honnête**, il faut compter **le pire cas** : toutes les pages
résidentes. Sinon on mesure la générosité du page cache, pas la mémoire du
modèle.

Protocole de mesure recommandé :

1. lancer le binaire seul, sans shell intermédiaire ;
2. mesurer la RSS avant chargement ;
3. charger le modèle ;
4. générer 512 tokens (assez pour toucher tous les tenseurs) ;
5. mesurer la RSS après génération ;
6. forcer `madvise(MADV_DONTNEED)` n'est pas autorisé : on garde le pire cas ;
7. répéter 3 fois, retenir le maximum.

Outils :
- macOS : `/usr/bin/time -l`, `task_info(mach_task_self(), MACH_TASK_BASIC_INFO)`,
  `footprint` ;
- Linux : `/proc/self/status` (`VmRSS`), `getrusage` ;
- Windows : `GetProcessMemoryInfo` (`WorkingSetSize`).

Le critère doit être : **`WorkingSetSize` / `VmRSS` < 100 Mo après 512 tokens**,
pas le delta.

---

## 9. Débit : ce que le budget implique

Mesure [mesuré] : 118,9 tok/s pour 25 M paramètres, moteur Python + noyau NEON.

Le débit n'est pas limité par la bande passante mémoire :
```
25 M paramètres × 2 FLOP = 50 MFLOP/token
118,9 tok/s × 50 MFLOP = 5,9 GFLOP/s
```
Or un M4 peut faire plusieurs dizaines de GFLOP/s en fp32. Le facteur limitant
est le **coût par appel** : 56 appels `ctypes` par token (7 matvecs × 8 couches),
plus la glue Python et les allocations.

Extrapolations [hypothèse] :

| modèle | moteur Python (linéaire en params) | moteur C natif (3–6× plus rapide) |
|---|---:|---:|
| 25 M | 119 tok/s (mesuré) | 350–700 tok/s |
| 88 M | 34 tok/s | 100–200 tok/s |
| 130 M | 23 tok/s | 65–140 tok/s |
| 200 M | 15 tok/s | 40–90 tok/s |

Un runtime C natif est donc **indispensable pour le budget et pour l'usage** :
à 15 tok/s, un assistant est pénible ; à 40–90 tok/s, il est utilisable.

---

## 10. Protocole de validation du budget

La phase 0 de la nouvelle branche doit produire :

1. un binaire C `micro-llm` qui charge un `.bin` et génère N tokens ;
2. une mesure `VmRSS`/`WorkingSetSize` avant, après chargement, après 512
   tokens ;
3. un test automatisé qui échoue si la RSS dépasse 100 Mo ;
4. un rapport `docs/BENCH_RSS.md` avec la machine, l'OS, la commande exacte.

Tant que ce test n'existe pas, le budget n'est pas prouvé.

---

## 11. Verdict honnête

**100 Mo de RAM totale est atteignable**, mais pas avec l'existant :

- il faut un **runtime C natif** (le Python coûte 198 Mo à lui seul) ;
- il faut un modèle **≤ ~200 M paramètres** ternaires, embeddings int8,
  échelles fp16, GQA ;
- il faut un **contexte ≤ 2048** et un KV int8 ;
- il faut accepter **~90–99 Mo** selon la configuration, sans marge confortable.

Le budget n'est donc pas un problème de format ni de quantification : c'est un
problème de **runtime**. Le format `LLM2` est déjà bon ; c'est le programme qui
le lit qui coûte trop cher.

Corollaire : le chiffre « 12,13 Mo » communiqué jusqu'ici est le poids du
fichier, pas la mémoire du processus. Il faut cesser de l'appeler « budget
100 Mo » et distinguer systématiquement :

- **poids** : 12,13 Mo ;
- **structurel** : 12,74 Mo ;
- **delta RSS** : +28,0 Mo ;
- **RSS totale** : 225,9 Mo.

Tant que cette distinction n'est pas faite, toute décision de déploiement est
fausse.
