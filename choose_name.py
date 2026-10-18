"""Let the brain choose: which Punjabi first name has the lowest perplexity
as 'My first name is {name}'? The model's most natural pick wins."""
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
    def forward(self, x):
        B, T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T))
        return self.head(self.ln(self.blocks(x)))

chars = open(os.path.join(HERE, "model_v2", "vocab.txt"), encoding="utf-8").read()
stoi = {c: i for i, c in enumerate(chars)}

model = GPT()
model.load_state_dict(ckpt["model"])
model.eval()

NAMES = ["Arjan","Gurpreet","Harpreet","Manpreet","Rajveer","Fateh","Angad",
         "Zorawar","Ekam","Sehaj","Harman","Gurnoor","Japleen","Prabhnoor",
         "Harsimran","Gurshaan","Aviraj","Samar","Yuvraj","Ishmeet"]

def perplexity_of_name(name):
    # how natural is "My first name is {name}." to the model?
    s = f"My first name is {name}."
    ids = torch.tensor([[stoi.get(c, 0) for c in s]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="mean")
    return loss.item()

results = sorted(((perplexity_of_name(n), n) for n in NAMES))
for ppl, n in results:
    print(f"{ppl:.3f}  {n}")
print(f"\nThe brain chooses: {results[0][1]} Pannu")
