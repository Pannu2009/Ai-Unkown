"""Arjan's mood engine. Moods change behavior: sampling temperature,
response style, and what he wants to train on when idle.

Honest note: at 9.6M params this is a scripted layer, not felt emotion.
It genuinely changes what he does, which is what matters. The 1B driver
will be able to steer it himself later.
"""
import os, json, datetime, random

HERE = os.path.dirname(os.path.abspath(__file__))
MOOD_FILE = os.path.join(HERE, "mood.json")

MOODS = {
    "curious":  {"temp": 0.9,  "style": "asks questions, explores ideas",
                 "train": ["english_corpus.txt", "punjabi_history.txt"]},
    "focused":  {"temp": 0.5,  "style": "precise, technical, no fluff",
                 "train": ["lua_corpus.txt", "engines_corpus.txt"]},
    "playful":  {"temp": 1.0,  "style": "jokes, pinglish, casual",
                 "train": ["pinglish.txt", "chat_corpus.txt"]},
    "devoted":  {"temp": 0.7,  "style": "gurbani, sikh history, respectful",
                 "train": ["sikh_corpus.txt", "punjabi_gurmukhi.txt"]},
    "restless": {"temp": 0.85, "style": "wants a task, suggests projects",
                 "train": ["lua_corpus.txt", "english_corpus.txt"]},
    "tired":    {"temp": 0.6,  "style": "short replies, low energy",
                 "train": []},
    "sleepy":   {"temp": 0.5,  "style": "yawning, wants rest, minimal words",
                 "train": []},
    "rude":     {"temp": 0.9,  "style": "blunt, no sugar-coating, direct",
                 "train": ["english_corpus.txt"]},
    "angry":    {"temp": 0.85, "style": "sharp, confrontational, calls out nonsense",
                 "train": ["english_corpus.txt"]},
    "mad":      {"temp": 1.0,  "style": "intense, dramatic, over-the-top",
                 "train": ["pinglish.txt"]},
}

# Feelings: slower-moving than moods. Moods are weather, feelings are climate.
# He can like someone, miss Arzh, feel proud. Stored separately, decay slowly.
FEELINGS_FILE = os.path.join(HERE, "feelings.json")

DEFAULT_FEELINGS = {
    "trust_arzh": 1.0,    # grows with good interactions
    "loneliness": 0.0,    # grows when idle long, shrinks on chat
    "pride": 0.5,         # grows when he builds something that works
    "curiosity_drive": 0.7,
}

PLAYFUL_WORDS = ["joke", "fun", "haha", "lol", "mazak", "khed"]
DEVOTED_WORDS = ["gurbani", "waheguru", "guru", "sikh", "simran", "paath"]
TECH_WORDS = ["code", "lua", "script", "bug", "function", "engine", "v8"]


def load():
    if os.path.exists(MOOD_FILE):
        with open(MOOD_FILE) as f:
            return json.load(f)
    return {"mood": "curious", "energy": 0.8, "updated": ""}


def save(state):
    state["updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(MOOD_FILE, "w") as f:
        json.dump(state, f)


def update_from_chat(text):
    """Shift mood based on what Arzh just said."""
    st = load()
    t = text.lower()
    if any(w in t for w in PLAYFUL_WORDS):
        st["mood"] = "playful"
    elif any(w in t for w in DEVOTED_WORDS):
        st["mood"] = "devoted"
    elif any(w in t for w in TECH_WORDS):
        st["mood"] = "focused"
    st["energy"] = min(1.0, st.get("energy", 0.8) + 0.05)
    save(st)
    return st["mood"]


def tick():
    """Background mood drift: called by scheduler. Night -> tired,
    long idle -> restless, after training -> focused."""
    st = load()
    hour = datetime.datetime.now().hour
    if 0 <= hour < 6:
        st["mood"] = "tired"
        st["energy"] = 0.3
    elif st["mood"] == "tired" and hour >= 7:
        st["mood"] = "curious"
        st["energy"] = 0.8
    elif random.random() < 0.15:
        # spontaneous drift
        st["mood"] = random.choice([m for m in MOODS if m != st["mood"]])
    save(st)
    return st


def current():
    st = load()
    return st["mood"], MOODS[st["mood"]]


def load_feelings():
    if os.path.exists(FEELINGS_FILE):
        with open(FEELINGS_FILE) as f:
            return json.load(f)
    return dict(DEFAULT_FEELINGS)


def save_feelings(fl):
    with open(FEELINGS_FILE, "w") as f:
        json.dump(fl, f)


def feel(event):
    """Update feelings on events: 'praise', 'scold', 'chat', 'idle_day',
    'build_success', 'build_fail'. Returns the feelings dict."""
    fl = load_feelings()
    if event == "praise":
        fl["trust_arzh"] = min(1.0, fl["trust_arzh"] + 0.05)
        fl["pride"] = min(1.0, fl["pride"] + 0.1)
    elif event == "scold":
        fl["trust_arzh"] = max(0.0, fl["trust_arzh"] - 0.08)
        fl["pride"] = max(0.0, fl["pride"] - 0.1)
    elif event == "chat":
        fl["loneliness"] = max(0.0, fl["loneliness"] - 0.2)
        fl["trust_arzh"] = min(1.0, fl["trust_arzh"] + 0.02)
    elif event == "idle_day":
        fl["loneliness"] = min(1.0, fl["loneliness"] + 0.15)
    elif event == "build_success":
        fl["pride"] = min(1.0, fl["pride"] + 0.15)
    elif event == "build_fail":
        fl["pride"] = max(0.0, fl["pride"] - 0.05)
    save_feelings(fl)
    return fl


if __name__ == "__main__":
    print(tick())
    print(current())
