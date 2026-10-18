# NeuroBrain

A neural network built from scratch that replies to you. No APIs. No API keys.
No pretrained weights. No third-party AI services. Just math.

## What it is

A micro transformer (~0.82M parameters, 4 layers, 4 heads) written in pure
PyTorch, trained on raw text with a hand-rolled training loop. It learns to
predict the next character, over and over, until it can generate text that
sounds like what it trained on.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 train.py      # trains the brain (~1h on CPU), saves to model/
python3 chat.py       # talk to it
```

Chat commands: type anything, the brain replies. `quit` exits.
Flags: `--temp 0.8` (creativity), `--len 200` (reply length).

## Files

- `train.py` — model definition + training loop. Everything from scratch:
  token embedding, positional embedding, causal multi-head self-attention,
  MLP blocks, layer norm, AdamW.
- `chat.py` — interactive chat loop loading `model/best.pt`.
- `data/tinyshakespeare.txt` — training corpus (public domain).
- `model/` — `best.pt` (lowest validation loss), `final.pt`, `vocab.txt`.

## Honest specs

Trained in ~2 hours on 2 CPU cores. It will reply to you with
Shakespeare-flavored text that is locally coherent and globally unhinged.
It is a real neural network that really learned patterns from data — and a
toy next to production models, which train for weeks on GPUs over
terabytes of text. The code here is the real thing though: scale the model,
feed it more data and compute, and this exact architecture family is what
the big ones use.

## Retrain on your own text

Drop any `.txt` file next to the data folder, point `DATA_PATH` at it in
`train.py`, and run. The brain will learn to write like your corpus.
Convos, lyrics, code — whatever you feed it.
