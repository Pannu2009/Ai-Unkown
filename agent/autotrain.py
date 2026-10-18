"""Arjan's auto-train: he decides what to learn, gathers it, trains.
Simple wall: he needs no permission to read websites (Arzh pre-authorized).
He trains himself. Parameters don't grow from training — weights get smarter.
Parameter growth happens when Arzh builds the next bigger version; this system
tracks readiness for it.

Cycle:
  1. PICK   - mood + wants decide the topic
  2. GATHER - fetch web pages / use local corpora (no permission needed)
  3. TRAIN  - fine-tune current best checkpoint on new data
  4. TEST   - perplexity check: did he actually get smarter?
  5. COMMIT - only if test passes (brick wall: never dumber)
"""
import os, sys, json, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mood

AUTOTRAIN_LOG = os.path.join(HERE, "autotrain.log")


def log(msg):
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(AUTOTRAIN_LOG, "a") as f:
        f.write(f"[{stamp}] {msg}\n")
    print(msg)


def pick_topic():
    """Mood + wants pick what to learn. He decides."""
    m, info = mood.current()
    wants = ""
    try:
        with open(os.path.join(ROOT, "home", "wants.md")) as f:
            wants = f.read()
    except FileNotFoundError:
        pass
    topics = info.get("train", [])
    if topics:
        choice = topics[0]
        log(f"mood={m}: chose to train on {choice}")
        return choice, m
    log(f"mood={m}: resting, no training this cycle")
    return None, m


def gather_web(query, max_chars=8000):
    """Fetch and clean text from the web. Pre-authorized: no permission needed."""
    # Simple wall: read-only, no logins, no posting, 20KB cap per page
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&"
           "explaintext=1&format=json&redirects=1&titles="
           + urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuroBrain/0.2"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read(20000).decode("utf-8", "replace"))
        pages = data.get("query", {}).get("pages", {})
        text = next(iter(pages.values()), {}).get("extract", "")
        log(f"gathered {len(text)} chars on '{query}' (web, no permission needed)")
        return text[:max_chars]
    except Exception as e:
        log(f"gather failed for '{query}': {e}")
        return ""


def autotrain_cycle():
    topic, m = pick_topic()
    if not topic:
        return {"ok": True, "action": "rest"}
    # For now: log the intent. Full fine-tune hooks into train_v2.py --finetune
    # (wired when the next training window opens; nightly cron will call this).
    log(f"auto-train queued: {topic} (mood={m})")
    return {"ok": True, "action": "queued", "topic": topic}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gather":
        print(gather_web(sys.argv[2])[:500])
    else:
        print(autotrain_cycle())
