"""Arjan's morning briefing. Jarvis-style: what happened overnight,
how he's feeling, what's on his mind. Run at wake-up."""
import os, sys, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mood

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def read(p):
    try:
        with open(os.path.join(ROOT, p), encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def briefing():
    m, info = mood.current()
    lines = []
    lines.append(f"Good morning, Arzh. I am feeling {m} today.")
    lines.append(f"Style: {info['style']}.")
    wants = read("home/wants.md")
    # first unchecked want
    for line in wants.split("\n"):
        if line.startswith("- [ ]"):
            lines.append(f"Today I want to: {line[6:]}")
            break
    inbox = read("home/inbox.md")
    if "(empty)" not in inbox:
        lines.append("You left me tasks:")
        lines.append(inbox)
    # training status
    done = os.path.exists(os.path.join(ROOT, "model_v2", "DONE"))
    lines.append("Overnight training: " +
                 ("complete." if done else "still running."))
    return "\n".join(lines)

if __name__ == "__main__":
    print(briefing())
