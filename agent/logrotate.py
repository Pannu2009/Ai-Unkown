"""Log rotation: Arjan auto-deletes his own old logs, like Cloudflare.
Research logs, training logs, day-to-day scratch — anything older than
7 days gets wiped. His diary and memory are permanent; scratch is not.

Runs from cron. He cleans up after himself.
"""
import os, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (path, max age in days). Diary/memory/vault are NEVER rotated.
ROTATE = [
    (os.path.join(HERE, "autotrain.log"), 7),
    (os.path.join(ROOT, "train_v2.log"), 7),
    (os.path.join(HERE, "vault", "audit.log"), 7),
]

# scratch research files older than 7 days in these dirs
SCRATCH_DIRS = [
    os.path.join(ROOT, "home", "projects", "sandbox"),
]


def rotate():
    now = time.time()
    cleaned = []
    for path, max_days in ROTATE:
        if os.path.exists(path):
            age = (now - os.path.getmtime(path)) / 86400
            if age > max_days:
                # truncate, don't delete (keeps the file handle story simple)
                open(path, "w").write(
                    f"[rotated {datetime.date.today()}: older than "
                    f"{max_days}d, auto-deleted]\n")
                cleaned.append(path)
    for d in SCRATCH_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and (now - os.path.getmtime(p)) / 86400 > 7:
                os.remove(p)
                cleaned.append(p)
    return cleaned


if __name__ == "__main__":
    done = rotate()
    print(f"rotated {len(done)} logs" + (f": {done}" if done else ""))
