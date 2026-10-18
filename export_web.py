"""Convert a checkpoint to web format: sharded weights + manifest.json + vocab.txt.
Shards stay under 20MB each (Cloudflare KV limit is 25MB/value).
Usage: python3 export_web.py [checkpoint] [outdir]"""
import os, sys, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "model_v2", "best.pt")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "web_v2")
SHARD_MAX = 20 * 1024 * 1024  # 20MB, safe margin under KV's 25MB

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

# shard the buffer (tensor offsets stay global; loader concatenates shards)
shards = []
for i in range(0, len(buf), SHARD_MAX):
    name = f"weights_{len(shards)}.bin"
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(buf[i:i + SHARD_MAX])
    shards.append(name)
# remove stale single weights.bin / extra shards from previous exports
for fn in os.listdir(OUT):
    if (fn == "weights.bin" or
            (fn.startswith("weights_") and fn.endswith(".bin") and fn not in shards)):
        os.remove(os.path.join(OUT, fn))

total_params = offset
# trainable params exclude the fixed causal-mask buffers ("tril")
trainable = 0
for t in tensors:
    if "tril" in t["name"]:
        continue
    n = 1
    for d in t["shape"]:
        n *= d
    trainable += n
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"tensors": tensors, "vocab": cfg["vocab"],
               "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
               "n_embd": cfg["n_embd"], "block": cfg["block"],
               "shards": shards, "params": trainable,
               "params_m": f"{trainable / 1e6:.2f}M"}, f)
# vocab lives alongside the checkpoint
ckpt_dir = os.path.dirname(CKPT)
chars = open(os.path.join(ckpt_dir, "vocab.txt"), encoding="utf-8").read()
with open(os.path.join(OUT, "vocab.txt"), "w", encoding="utf-8") as f:
    f.write(chars)
print(f"exported {len(tensors)} tensors, {trainable / 1e6:.2f}M params, "
      f"{len(buf) / 1e6:.1f}MB in {len(shards)} shard(s) -> {OUT}")
