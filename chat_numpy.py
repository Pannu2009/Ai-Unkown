"""NeuroBrain chat - pure numpy, zero dependencies beyond numpy.
Runs on Termux with just: pip install numpy
Usage: python3 chat_numpy.py [--temp 0.8] [--len 150]
Loads model/best.npz (converted from the torch checkpoint)."""
import os, math, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--len", type=int, default=150)
ap.add_argument("--weights", default=os.path.join(HERE, "model", "best.npz"))
a = ap.parse_args()

W = dict(np.load(a.weights))
N_LAYER, N_HEAD, N_EMBD, BLOCK = 4, 4, 128, 128
HS = N_EMBD // N_HEAD

chars = open(os.path.join(HERE, "model", "vocab.txt"), encoding="utf-8").read()
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
VOCAB = len(chars)

_erf = np.vectorize(math.erf)

def layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b

def gelu(x):
    return 0.5 * x * (1 + _erf(x / math.sqrt(2)))

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def forward(ids):
    T = len(ids)
    x = W["tok.weight"][ids] + W["pos.weight"][:T]
    for i in range(N_LAYER):
        p = f"blocks.{i}."
        h = layernorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
        outs = []
        for j in range(N_HEAD):
            qkv = h @ W[f"{p}heads.{j}.qkv.weight"].T
            q, k, v = np.split(qkv, 3, axis=-1)
            att = q @ k.T / math.sqrt(HS)
            mask = np.triu(np.ones((T, T)), 1).astype(bool)
            att = np.where(mask, -1e10, att)
            outs.append(softmax(att) @ v)
        x = x + np.concatenate(outs, -1) @ W[p + "proj.weight"].T
        h2 = layernorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        m = gelu(h2 @ W[p + "mlp.0.weight"].T + W[p + "mlp.0.bias"])
        x = x + m @ W[p + "mlp.2.weight"].T + W[p + "mlp.2.bias"]
    x = layernorm(x, W["ln.weight"], W["ln.bias"])
    return x @ W["head.weight"].T

def reply(prompt, n=a.len, temp=a.temp):
    ids = [stoi.get(c, 0) for c in prompt]
    for _ in range(n):
        ctx = ids[-BLOCK:]
        logits = forward(np.array(ctx))[-1] / temp
        probs = softmax(logits)
        ids.append(int(np.random.choice(VOCAB, p=probs)))
    return "".join(itos[i] for i in ids)[len(prompt):]

print("NeuroBrain online (numpy edition). Type 'quit' to exit.\n")
while True:
    try:
        p = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBrain shutting down.")
        break
    if p.lower() in ("quit", "exit"):
        print("Brain shutting down.")
        break
    if not p:
        continue
    print("Brain:", reply(p + "\n").strip(), "\n")
