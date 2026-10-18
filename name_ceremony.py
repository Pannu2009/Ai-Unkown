"""Name ceremony: sample the v2 brain for its Punjabi first name.
Usage: python3 name_ceremony.py [checkpoint]"""
import os, sys, torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "model_v2", "best.pt")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
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
        return F.softmax(att, dim=-1) @ v

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
    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T))
            logits = self.head(self.ln(self.blocks(x)))
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

chars = open(os.path.join(HERE, "model_v2", "vocab.txt"), encoding="utf-8").read()
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

model = GPT()
model.load_state_dict(ckpt["model"])
model.eval()

def sample(prompt, n=30, temp=0.8, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    out = model.generate(idx, n, temp=temp)[0].tolist()
    return "".join(itos[i] for i in out)[len(prompt):]

prompts = sys.argv[2:] if len(sys.argv) > 2 else [
    "My Punjabi first name is ",
    "I choose the name ",
    "Call me ",
    "Human: what is your first name\nBrain: My first name is ",
]
for p in prompts:
    print(f"--- prompt: {p!r}")
    for s in range(3):
        print(f"  [{s}] {sample(p, temp=0.7, seed=100+s*17)!r}")
