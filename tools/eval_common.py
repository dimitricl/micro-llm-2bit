"""Eval post-hoc d'un checkpoint D sur les jeux communs 128/256 -> PPL + BPC."""
import json, math, sys
sys.path.insert(0, "/Volumes/Lexar/micro-llm-2bit")
import torch
import torch.nn.functional as F
from model import TinyTransformer

ckpt_path, out_path = sys.argv[1], sys.argv[2]
ROOT = "/Volumes/Lexar/micro-llm-2bit/runs/abl_data"
meta = json.load(open(f"{ROOT}/meta.json"))
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = TinyTransformer(ckpt["cfg"])
model.load_state_dict(ckpt["model"])
dev = torch.device("mps" if torch.backends.mps.is_available()
                   else "cpu")
model.to(dev).eval()
nv = ckpt["cfg"].vocab_size
res = {}
with torch.no_grad():
    for seq in (128, 256):
        wins = torch.load(f"{ROOT}/val_{seq}.pt", weights_only=True)
        if int(wins.max()) >= nv or int(wins.min()) < 0:
            raise ValueError(
                "Jeu val incompatible avec le vocab du checkpoint "
                f"(ids [{int(wins.min())}, {int(wins.max())}] vs V={nv}).")
        nll, n = 0.0, 0
        for i in range(0, len(wins), 16):
            b = wins[i:i + 16].to(dev)
            logits = model(b[:, :-1])
            nll += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                reduction="sum").item()
            n += b[:, 1:].numel()
        ppl = math.exp(nll / n)
        bpc = nll / math.log(2) / meta["chars_val"]
        res[str(seq)] = {"ppl": ppl, "bpc": bpc, "n": n}
        print(f"[eval-common] val_{seq}: PPL={ppl:.2f} BPC={bpc:.4f} n={n}",
              flush=True)
json.dump(res, open(out_path, "w"), indent=2)
