"""NeuroBrain - a from-scratch micro transformer. No APIs, no pretrained weights.
Trains a tiny GPT on raw text using only PyTorch ops. ~800k params."""
import os, time, math, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- config ----------------
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "model")
BLOCK       = 128     # context length
BATCH       = 32
N_LAYER     = 4
N_HEAD      = 4
N_EMBD      = 128
DROPOUT     = 0.0
LR          = 3e-4
MAX_ITERS   = 4000
EVAL_EVERY  = 250
SAVE_EVERY  = 1000
DEVICE      = "cpu"
SEED        = 1337
# ----------------------------------------

torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

text = open(DATA_PATH, encoding="utf-8").read()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
open(os.path.join(OUT_DIR, "vocab.txt"), "w", encoding="utf-8").write("".join(chars))
print(f"chars: {len(text)}, vocab: {VOCAB}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

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
        att = F.softmax(att, dim=-1)
        return att @ v

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
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block_() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
    def forward(self, x, y=None):
        B, T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T, device=DEVICE))
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            idx_c = idx[:, -BLOCK:]
            logits, _ = self(idx_c)
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = GPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=LR)

@torch.no_grad()
def estimate():
    model.eval()
    out = {}
    for s in ("train", "val"):
        ls = torch.zeros(20)
        for i in range(20):
            x, y = get_batch(s)
            _, l = model(x, y)
            ls[i] = l.item()
        out[s] = ls.mean().item()
    model.train()
    return out

t0 = time.time()
best = float("inf")
for it in range(1, MAX_ITERS + 1):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % EVAL_EVERY == 0 or it == 1:
        e = estimate()
        dt = time.time() - t0
        print(f"iter {it:4d}  train {e['train']:.3f}  val {e['val']:.3f}  "
              f"{dt/60:.1f}min  {it/dt:.1f}it/s", flush=True)
        if e["val"] < best:
            best = e["val"]
            torch.save({"model": model.state_dict(),
                        "cfg": dict(block=BLOCK, n_layer=N_LAYER, n_head=N_HEAD,
                                    n_embd=N_EMBD, vocab=VOCAB)},
                       os.path.join(OUT_DIR, "best.pt"))
    if it % SAVE_EVERY == 0:
        torch.save({"model": model.state_dict(),
                    "cfg": dict(block=BLOCK, n_layer=N_LAYER, n_head=N_HEAD,
                                n_embd=N_EMBD, vocab=VOCAB)},
                   os.path.join(OUT_DIR, f"ckpt_{it}.pt"))

torch.save({"model": model.state_dict(),
            "cfg": dict(block=BLOCK, n_layer=N_LAYER, n_head=N_HEAD,
                        n_embd=N_EMBD, vocab=VOCAB)},
           os.path.join(OUT_DIR, "final.pt"))
print("done. best val:", round(best, 3))

# sample
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt"),
                                 map_location=DEVICE)["model"])
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(ctx, 300, temp=0.8)[0].tolist()))
