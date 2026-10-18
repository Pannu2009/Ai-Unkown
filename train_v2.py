"""NeuroBrain v2 - overnight training. Bigger brain, mixed diet.
Usage: python3 train_v2.py [--resume]
Saves checkpoints to model_v2/. Writes DONE file when finished."""
import os, sys, time, glob, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

# ---------------- config ----------------
BLOCK, BATCH = 128, 32
N_LAYER, N_HEAD, N_EMBD = 6, 6, 192
LR, MAX_ITERS = 3e-4, 12000
EVAL_EVERY, SAVE_EVERY = 500, 1000
DEVICE, SEED = "cpu", 1337
OUT_DIR = os.path.join(HERE, "model_v2")
DONE_FILE = os.path.join(OUT_DIR, "DONE")
# ----------------------------------------

torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

def load_corpus():
    d = os.path.join(HERE, "data")
    def rd(f): return open(os.path.join(d, f), encoding="utf-8").read()
    lua = rd("lua_corpus.txt")
    chat = rd("chat_corpus.txt")
    sikh = rd("sikh_corpus.txt")
    engines = rd("engines_corpus.txt")
    gurmukhi = rd("punjabi_gurmukhi.txt")
    shahmukhi = rd("punjabi_shahmukhi.txt")
    pjhist = rd("punjabi_history.txt")
    pinglish = rd("pinglish.txt")
    english = rd("english_corpus.txt")
    dreams = rd("dreams_corpus.txt")
    opinions = rd("opinions_corpus.txt")
    knowledge = rd("knowledge_corpus.txt")
    chat2 = rd("chat_v2.txt")
    # curiosity: everything he researched himself
    curious = ""
    cdir = os.path.join(d, "curiosity")
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".txt"):
                curious += open(os.path.join(cdir, fn),
                                encoding="utf-8").read() + "\n"
    # repetition weights: facts get memorized, code patterns get variety
    text = (lua * 4 + chat * 25 + sikh * 12 + engines * 8 +
            gurmukhi * 20 + shahmukhi * 15 + pjhist * 10 +
            pinglish * 20 + english * 8 +
            dreams * 15 + opinions * 15 +
            knowledge * 12 + chat2 * 20 +
            curious * 10)
    return text

text = load_corpus()
chars = sorted(set(text))
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
print(f"corpus chars: {len(text)}, vocab: {VOCAB}", flush=True)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data))
train_data, val_data = data[:n], data[n:]
open(os.path.join(OUT_DIR, "vocab.txt"), "w", encoding="utf-8").write("".join(chars))

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
    def forward(self, x, y=None):
        B, T = x.shape
        x = self.tok(x) + self.pos(torch.arange(T, device=DEVICE))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        return logits, loss

model = GPT().to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

start_iter = 1
if a.resume:
    ckpts = sorted(glob.glob(os.path.join(OUT_DIR, "ckpt_*.pt")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter = ck["iter"] + 1
        print(f"resumed from iter {ck['iter']}", flush=True)

@torch.no_grad()
def estimate():
    model.eval()
    out = {}
    for s in ("train", "val"):
        ls = torch.zeros(10)
        for i in range(10):
            x, y = get_batch(s)
            _, l = model(x, y)
            ls[i] = l.item()
        out[s] = ls.mean().item()
    model.train()
    return out

def save(path, it):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "iter": it,
                "cfg": dict(block=BLOCK, n_layer=N_LAYER, n_head=N_HEAD,
                            n_embd=N_EMBD, vocab=VOCAB)}, path)

t0 = time.time()
best = float("inf")
for it in range(start_iter, MAX_ITERS + 1):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % EVAL_EVERY == 0 or it == start_iter:
        e = estimate()
        dt = time.time() - t0
        print(f"iter {it:5d} train {e['train']:.3f} val {e['val']:.3f} "
              f"{dt/3600:.1f}h {it/max(dt,1):.1f}it/s", flush=True)
        if e["val"] < best:
            best = e["val"]
            save(os.path.join(OUT_DIR, "best.pt"), it)
    if it % SAVE_EVERY == 0:
        save(os.path.join(OUT_DIR, f"ckpt_{it}.pt"), it)

save(os.path.join(OUT_DIR, "final.pt"), MAX_ITERS)
open(DONE_FILE, "w").write(f"done val {best:.3f}\n")
print(f"done. best val: {best:.3f}", flush=True)
