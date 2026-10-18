"""NeuroBrain chat - talk to your trained neural network.
Usage: python3 chat.py [--temp 0.8] [--len 200]
No internet, no APIs, no keys. Just your brain."""
import os, argparse, torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--len", type=int, default=200)
ap.add_argument("--ckpt", default=os.path.join(HERE, "model", "best.pt"))
a = ap.parse_args()

ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
BLOCK, N_LAYER, N_HEAD, N_EMBD, VOCAB = (cfg["block"], cfg["n_layer"],
                                         cfg["n_head"], cfg["n_embd"], cfg["vocab"])

class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK, BLOCK)))
        self.hs = hs
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.hs, dim=2)
        att = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.proj(att @ v) if hasattr(self, "proj") else (F.softmax(att, -1) @ v)

class Block_(nn.Module):
    def __init__(self):
        super().__init__()
        hs = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(hs) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                 nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(self, x):
        x = x + self.proj(torch.cat([h(self.ln1(x)) for h in self.heads], dim=2))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block_() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
    def forward(self, x):
        B, T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T))
        return self.head(self.ln(self.blocks(x)))
    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            logits = self(idx[:, -BLOCK:])
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

chars = open(os.path.join(HERE, "model", "vocab.txt"), encoding="utf-8").read()
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

model = GPT()
model.load_state_dict(ckpt["model"])
model.eval()

def reply(prompt, n=a.len, temp=a.temp):
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, n, temp=temp)[0].tolist()
    return "".join(itos[i] for i in out)[len(prompt):]

print("NeuroBrain online. Trained from scratch, no APIs. Type 'quit' to exit.\n")
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
