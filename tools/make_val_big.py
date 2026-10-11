"""Construit val_big (G2) : >=300 docs STRICTEMENT disjoints des slices 00-09.

Source : reste disjoint de slice_100M.jsonl (4950 docs disjoints trouvés
par l'enquête hash). Règle : ordre du fichier, exclusion hashée, stop à
300 docs. + jeu OOD : AUCUNE source locale valide (voir rapport) -> non
construit, besoin listé.
Écrit runs/abl_data/val_big.jsonl + val_big_manifest.json.
ÉCHOUE bruyamment si <300 docs disjoints.
"""
import hashlib, json, re, sys

sys.path.insert(0, "/Volumes/Lexar/micro-llm-2bit")

ROOT = "/Volumes/Lexar/micro-llm-2bit/runs/abl_data"
SEED = 0  # tirage : ordre du fichier (déterministe, pas d'aléa)


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def h(t):
    return hashlib.sha1(norm(t).encode("utf-8")).hexdigest()


def docs_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    continue
                if t.strip():
                    out.append(t)
    return out


print("[valbig] chargement interdits (slices 00-09)...", flush=True)
forbidden = set()
for i in range(10):
    for t in docs_jsonl(f"/Volumes/Lexar/clean-corpus/slices/slice_0{i}.jsonl"):
        forbidden.add(h(t))
print(f"[valbig] interdits: {len(forbidden)}", flush=True)
json.dump(sorted(forbidden),
          open(f"{ROOT}/forbidden_sha1.json", "w"))

print("[valbig] scan slice_100M...", flush=True)
pris, pris_h = [], []
for t in docs_jsonl("/Volumes/Lexar/clean-corpus/slice_100M.jsonl"):
    x = h(t)
    if x not in forbidden:
        pris.append(t)
        pris_h.append(x)
    if len(pris) >= 300:
        break
print(f"[valbig] disjoints retenus: {len(pris)}", flush=True)
if len(pris) < 300:
    raise SystemExit(f"[valbig] ÉCHEC : seulement {len(pris)} docs disjoints "
                     "(il en faut >= 300).")
with open(f"{ROOT}/val_big.jsonl", "w") as f:
    for t in pris:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
manifest = {
    "source": "reste disjoint de /Volumes/Lexar/clean-corpus/slice_100M.jsonl",
    "regle_tirage": "ordre du fichier, exclusion par sha1(normalise), stop a 300",
    "seed": SEED,
    "n_docs": len(pris),
    "chars": sum(len(t) for t in pris),
    "doc_hashes": pris_h,
    "forbidden_count": len(forbidden),
    "forbidden_file": "runs/abl_data/forbidden_sha1.json",
}
json.dump(manifest, open(f"{ROOT}/val_big_manifest.json", "w"), indent=1)
print(f"[valbig] {len(pris)} docs, {manifest['chars']} chars -> "
      f"{ROOT}/val_big.jsonl", flush=True)
print("[valbig] TERMINE", flush=True)
