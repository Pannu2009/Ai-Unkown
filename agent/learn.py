"""Arjan's continual learning: every conversation becomes training data.
Not real-time weight updates (unstable at this scale) — the honest version:
log everything, fine-tune nightly on what's new. He literally gets smarter
while Arzh sleeps.
"""
import os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONVO_FILE = os.path.join(ROOT, "data", "conversations.txt")
TRIGGER_FILE = os.path.join(HERE, "retrain.trigger")
# fine-tune when this many new exchanges pile up
RETRAIN_THRESHOLD = 200


def log_exchange(human, brain):
    """Call after every chat turn. Appends to the training corpus."""
    with open(CONVO_FILE, "a", encoding="utf-8") as f:
        f.write(f"Human: {human}\nBrain: {brain}\n")
    # count new exchanges since last fine-tune
    with open(CONVO_FILE, encoding="utf-8") as f:
        n = f.read().count("Human: ")
    try:
        with open(TRIGGER_FILE) as f:
            last = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        last = 0
    if n - last >= RETRAIN_THRESHOLD:
        with open(TRIGGER_FILE, "w") as f:
            f.write(str(n))
        return True  # signal: enough new data, run nightly fine-tune
    return False


def get_new_data():
    """Return conversation text added since last fine-tune (for the trainer)."""
    try:
        with open(TRIGGER_FILE) as f:
            last = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        last = 0
    with open(CONVO_FILE, encoding="utf-8") as f:
        lines = f.read().split("Human: ")
    # crude: return everything (fine-tune mixes it with base corpora anyway)
    return "Human: ".join(lines)
