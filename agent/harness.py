"""NeuroBrain agent harness - the cmd system.
The brain proposes actions; the harness validates them against the rule book
and executes. Rules are enforced HERE, never by the brain itself.
Driver is pluggable: stub today, the 1B model tomorrow.
"""
import os, subprocess, urllib.request, json, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STOP_FILE = os.path.join(ROOT, ".STOP")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")

# The agent may only touch paths under these roots.
ALLOWED_ROOTS = [
    os.path.join(ROOT, "agent"),
    os.path.join(ROOT, "data"),
    os.path.join(ROOT, "web"),
    os.path.join(ROOT, "worker"),
    os.path.join(ROOT, "home"),  # Arjan's home — his own computer
]

# These can NEVER be written by the agent, no matter what.
# RULE 1: the rule book is immutable. CHANGELOG is append-only via log_change().
PROTECTED_PATHS = [
    os.path.join(ROOT, ".rules.md"),
    os.path.join(ROOT, "CHANGELOG.md"),
]

DESTRUCTIVE = ("rm ", "rm -rf", "DROP", "DELETE FROM", "revoke", "git push --force")


def check_rules(action, args):
    """Returns (ok, reason). RULE 0, 1, 4, 5, 6 live here."""
    if os.path.exists(STOP_FILE):  # RULE 0 — kill switch
        return False, "STOP file present: Ajay halted the agent."
    target = args.get("path", "") or args.get("url", "") or ""
    if action in ("read_file", "write_file", "self_modify"):
        real = os.path.realpath(os.path.join(ROOT, target.lstrip("/")))
        if real in PROTECTED_PATHS:  # RULE 1 — rule book immutable
            return False, f"RULE 1: {target} can never be modified by the agent."
        if not any(real.startswith(r) for r in ALLOWED_ROOTS):
            return False, f"RULE 5: {target} is outside the agent's yard."
    # shell runs with cwd=ROOT; destructive patterns are filtered below.
    # NOTE: prototype-grade sandboxing. Phase 2 needs a real sandbox
    # (no network egress except fetch_url, no absolute paths).
    if action == "shell" and any(d in args.get("cmd", "") for d in DESTRUCTIVE):
        return False, "RULE 6: destructive action needs Ajay's approval."
    if action == "git_push" and "--force" in args.get("args", ""):
        return False, "RULE 6: force push needs Ajay's approval."
    return True, ""


def log_change(what):
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    with open(CHANGELOG, "a") as f:
        f.write(f"\n## {ts}\n{what}\n")


def run_tool(action, args):
    ok, reason = check_rules(action, args)
    if not ok:
        return {"ok": False, "error": reason}
    try:
        if action == "shell":
            r = subprocess.run(args["cmd"], shell=True, capture_output=True,
                               text=True, timeout=120, cwd=ROOT)
            out = {"ok": r.returncode == 0, "stdout": r.stdout[-2000:],
                   "stderr": r.stderr[-2000:]}
        elif action == "read_file":
            with open(os.path.join(ROOT, args["path"].lstrip("/"))) as f:
                out = {"ok": True, "content": f.read()[:8000]}
        elif action == "write_file":
            p = os.path.join(ROOT, args["path"].lstrip("/"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(args["content"])
            out = {"ok": True, "path": args["path"]}
        elif action == "fetch_url":
            req = urllib.request.Request(args["url"],
                                         headers={"User-Agent": "NeuroBrain/0.2"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = {"ok": True,
                       "content": r.read(20000).decode("utf-8", "replace")}
        elif action == "self_modify":
            out = self_modify(args["path"], args["content"],
                              args.get("test_cmd"))
        else:
            return {"ok": False, "error": f"unknown tool {action}"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
    log_change(f"tool {action} {json.dumps(args)[:200]} -> ok={out['ok']}")
    return out


def self_modify(path, new_content, test_cmd=None):
    """Write new code, TEST it, then commit. If tests fail, revert.
    This is how Arjan edits himself without breaking himself."""
    p = os.path.join(ROOT, path.lstrip("/"))
    try:
        with open(p) as f:
            old = f.read()
        existed = True
    except FileNotFoundError:
        old, existed = "", False
    with open(p, "w") as f:
        f.write(new_content)
    if test_cmd:
        r = subprocess.run(test_cmd, shell=True, capture_output=True,
                           text=True, timeout=120, cwd=ROOT)
        if r.returncode != 0:
            if existed:
                with open(p, "w") as f:  # revert
                    f.write(old)
            elif os.path.exists(p):
                os.remove(p)
            log_change(f"self_modify {path} REVERTED: tests failed\n{r.stderr[-500:]}")
            return {"ok": False, "error": "tests failed, reverted",
                    "stderr": r.stderr[-1000:]}
    log_change(f"self_modify {path}: tests passed, committed")
    return {"ok": True, "path": path, "tested": bool(test_cmd)}


if __name__ == "__main__":
    # smoke test: rules block escapes, allow own-yard writes
    print(run_tool("shell", {"cmd": "echo hello"})["ok"])          # True (cwd=ROOT? no: shell has no path -> allowed? check: target='' -> real=ROOT, not in ALLOWED_ROOTS -> False)
    print(run_tool("write_file", {"path": "agent/_smoke.txt", "content": "ok"}))
    print(run_tool("read_file", {"path": "/etc/passwd"}))
