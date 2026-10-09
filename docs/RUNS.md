# Runs

| run | config | tranche | steps | epochs vues | tokens vus | val-PPL finale | ratio val/train | tok/s moyen | durée |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-08-tranche00 | tiny | slice_00.jsonl (107 Mo, 10067 docs) | 4248 | 1 complète (34,8M) | 34,8M | 29,7 | - | ~2250 | 15668 s |
| 2026-10-08-think | tiny | 278 traces think (219 Ko) | 2048 max | - | - | 94,1 | 2,67 | ~1250 | 420 s |
| 2026-10-09-wiki-slice00 | tiny | slice_00.jsonl (107 Mo) | 2124 | ~0,5 (DEMI-EPOCH, bug B1) | 17,4M | 37,3 | 0,95 | ~2200 | 7927 s |

Notes (correctif phase B) :
- wiki_slice00 n'a vu que la moitié de la tranche (bug de décompte B1 :
  `bs16 > save8` → micro-batch réel 8, `steps_per_epoch` sous-compté de
  moitié). Sa val-PPL (37,3) n'est pas comparable à tranche00 (29,7) sans
  vérifier d'abord vocabulaire et validation identiques (phase C).
- tranche00 : 1 epoch complète (33990 micros, 6 micros jetés en fin d'epoch
  par l'ancien code). Référence pour la suite (pas wiki_slice00_bestval).
