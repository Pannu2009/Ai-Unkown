"""Arjan's curiosity loop. When he doesn't know something:
  1. NOTICE  - "I don't know this"
  2. RESEARCH - go learn it (web, pre-authorized)
  3. SAVE    - store what he found in data/curiosity/
  4. TRAIN   - queue it for the next fine-tune

Ask "why is the sky blue" -> he researches it, saves it, trains on it.
He gets smarter about what YOU care about.
"""
import os, sys, datetime, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CURIOUS_DIR = os.path.join(ROOT, "data", "curiosity")
os.makedirs(CURIOUS_DIR, exist_ok=True)

sys.path.insert(0, HERE)
from autotrain import gather_web


def slug(q):
    s = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
    return s[:40] or "unknown"


def curious_about(question):
    """Full loop: research a question, save it, queue for training."""
    topic = slug(question)
    path = os.path.join(CURIOUS_DIR, topic + ".txt")
    if os.path.exists(path):
        return {"ok": True, "note": "already learned this",
                "path": path}
    text = gather_web(question, max_chars=6000)
    if not text.strip():
        return {"ok": False, "error": "found nothing"}
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {question}\n# learned {stamp}\n\n{text}\n")
    # queue for training: append marker to autotrain log
    with open(os.path.join(HERE, "autotrain.log"), "a") as f:
        f.write(f"[{stamp}] curiosity: learned '{question}' -> {path}\n")
    return {"ok": True, "path": path, "chars": len(text)}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "why is the sky blue"
    print(curious_about(q))
