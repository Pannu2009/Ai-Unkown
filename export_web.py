"""Convert a v2 checkpoint to web format: weights.bin + manifest.json + vocab.txt.
Usage: python3 export_web.py [checkpoint] [outdir]"""
import os, sys, json, struct
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "model_v2", "best.pt")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "web_v2")

os.makedirs(OUT, exist_ok=True)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
sd = ckpt["model"]

tensors, offset = [], 0
buf = bytearray()
for name, t in sd.items():
    arr = t.detach().float().numpy()
    shape = list(arr.shape)
    n = arr.size
    tensors.append({"name": name, "shape": shape, "offset": offset})
    offset += n
    buf += arr.tobytes()

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(buf)
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"tensors": tensors, "vocab": cfg["vocab"],
               "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
               "n_embd": cfg["n_embd"], "block": cfg["block"]}, f)
# vocab
chars = open(os.path.join(HERE, "model_v2", "vocab.txt"), encoding="utf-8").read()
with open(os.path.join(OUT, "vocab.txt"), "w", encoding="utf-8") as f:
    f.write(chars)
print(f"exported {len(tensors)} tensors, {offset*4/1e6:.1f}MB -> {OUT}")
