"""Parse training progress into JSON for the /trains dashboard.
Uploads to KV as trainstats.json (called every 5 min by cron)."""
import os, re, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def parse_log():
    path = os.path.join(ROOT, "train_v2.log")
    pts = []
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"iter\s+(\d+)\s+train\s+([\d.]+)\s+val\s+([\d.]+)", line)
                if m:
                    pts.append({"iter": int(m.group(1)),
                                "train": float(m.group(2)),
                                "val": float(m.group(3))})
    except FileNotFoundError:
        pass
    return pts[-60:]  # last 60 points for the graph

def topics():
    out = []
    cdir = os.path.join(ROOT, "data", "curiosity")
    if os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".txt"):
                p = os.path.join(cdir, fn)
                out.append({"topic": fn[:-4].replace("_", " "),
                            "chars": os.path.getsize(p),
                            "when": datetime.datetime.fromtimestamp(
                                os.path.getmtime(p)).isoformat(timespec="seconds")})
    return out

def autotrain_recent():
    try:
        with open(os.path.join(HERE, "autotrain.log")) as f:
            return f.read().strip().split("\n")[-10:]
    except FileNotFoundError:
        return []

def stats():
    pts = parse_log()
    done = os.path.exists(os.path.join(ROOT, "model_v2", "DONE"))
    return {
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "done": done,
        "points": pts,
        "topics": topics(),
        "recent": autotrain_recent(),
        "params": "2.76M",
    }

if __name__ == "__main__":
    print(json.dumps(stats(), indent=1)[:800])
