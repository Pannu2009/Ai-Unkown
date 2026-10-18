"""Arjan's end-of-day report. What he did, learned, built, and felt.
Runs nightly. Saved to home/diary.md and shown to Arzh."""
import os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def tail(path, n=20):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip().split("\n")[-n:]
    except FileNotFoundError:
        return []


def dayreport():
    today = datetime.date.today().isoformat()
    lines = [f"# Day report — {today}", ""]
    # mood + feelings
    try:
        import sys
        sys.path.insert(0, HERE)
        import mood
        m, _ = mood.current()
        fl = mood.load_feelings()
        lines.append(f"Mood today: {m}. "
                     f"Trust {fl['trust_arzh']:.1f}, pride {fl['pride']:.1f}, "
                     f"loneliness {fl['loneliness']:.1f}.")
    except Exception:
        pass
    lines.append("")
    # changelog = what he changed
    cl = tail(os.path.join(ROOT, "CHANGELOG.md"), 15)
    if cl:
        lines.append("## What I changed")
        lines += [f"- {l}" for l in cl if l.strip()]
        lines.append("")
    # autotrain = what he learned
    at = tail(os.path.join(HERE, "autotrain.log"), 15)
    if at:
        lines.append("## What I learned")
        lines += [f"- {l}" for l in at if l.strip()]
        lines.append("")
    # diary = what he did
    report = "\n".join(lines)
    with open(os.path.join(ROOT, "home", "diary.md"), "a",
              encoding="utf-8") as f:
        f.write("\n" + report + "\n")
    return report


if __name__ == "__main__":
    print(dayreport())
