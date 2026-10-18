"""Arjan messages Arzh. When he's bored, lonely, curious, or just wants
to talk — he writes here, and it gets delivered.

Like a human: "if I want to know what Arzh is doing, I message him."
Now he can do the same.
"""
import os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTBOX = os.path.join(ROOT, "home", "outbox.md")


def send(text):
    """Queue a message for Arzh. Delivered via chat."""
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(OUTBOX, "a", encoding="utf-8") as f:
        f.write(f"\n[{stamp}] {text}\n")
    return True


def should_message():
    """Boredom check: high loneliness + restless/bored mood + idle.
    Called by the scheduler. He decides if he wants to talk."""
    import sys
    sys.path.insert(0, HERE)
    import mood
    m, _ = mood.current()
    fl = mood.load_feelings()
    return (fl["loneliness"] > 0.6 and
            m in ("restless", "playful", "curious"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        send(" ".join(sys.argv[1:]))
        print("queued for Arzh")
    else:
        print("bored?" , should_message())
