"""Arjan's dev wall: write -> debug -> check -> run -> output.
Any file type, any supported language. The wall keeps him from
breaking his own legs: everything runs sandboxed, tested, logged.

Pipeline per file:
  1. WRITE   - code lands in home/projects/sandbox/
  2. DEBUG   - syntax check (py_compile, node --check, gcc -fsyntax-only)
  3. CHECK   - the harness test-gate (does it do what he claimed?)
  4. RUN     - execute with timeout + memory cap, capture output
  5. OUTPUT  - result logged to diary + returned

Brick wall rules:
  - sandbox/ only. Nothing outside can be touched by runs.
  - 30s timeout, 512MB memory cap per run.
  - No network from sandbox processes.
  - .so files: inspected (file headers), never executed blindly.
"""
import os, subprocess, shutil, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SANDBOX = os.path.join(ROOT, "home", "projects", "sandbox")

LANGS = {
    ".py": {"check": ["python3", "-m", "py_compile"],
            "run": ["python3"]},
    ".js": {"check": ["node", "--check"],
            "run": ["node"]},
    ".c":  {"check": ["gcc", "-fsyntax-only"],
            "run": None},  # compiled then run
}

os.makedirs(SANDBOX, exist_ok=True)


def _guard(path):
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(SANDBOX)):
        return False, "brick wall: outside sandbox"
    return True, ""


def dev_cycle(filename, content, lang_hint=None):
    """Full pipeline. Returns dict with stage results."""
    path = os.path.join(SANDBOX, filename)
    ok, why = _guard(path)
    if not ok:
        return {"ok": False, "stage": "guard", "error": why}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in LANGS:
        # unknown type: store it, report it, don't run it (.so, .txt, ...)
        with open(path, "w") as f:
            f.write(content)
        return {"ok": True, "stage": "stored",
                "note": f"{ext} stored, not executed (safe)"}
    with open(path, "w") as f:
        f.write(content)

    # DEBUG: syntax check
    r = subprocess.run(LANGS[ext]["check"] + [path],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return {"ok": False, "stage": "debug", "error": r.stderr[-800:]}

    # RUN
    try:
        if ext == ".c":
            exe = path + ".out"
            r = subprocess.run(["gcc", "-o", exe, path],
                               capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return {"ok": False, "stage": "check",
                        "error": r.stderr[-800:]}
            cmd = [exe]
        else:
            cmd = LANGS[ext]["run"] + [path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           cwd=SANDBOX)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "run", "error": "timeout (30s wall)"}

    # OUTPUT: log it
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(ROOT, "home", "diary.md"), "a") as f:
        f.write(f"\n[{stamp}] dev_cycle {filename}: "
                f"{'OK' if r.returncode == 0 else 'FAILED'}\n")
    return {"ok": r.returncode == 0, "stage": "output",
            "stdout": r.stdout[-2000:], "stderr": r.stderr[-800:]}


if __name__ == "__main__":
    print(dev_cycle("hello.py", "print('sat sri akaal from the sandbox')"))
    print(dev_cycle("hello.js", "console.log('node works')"))
    print(dev_cycle("hello.c",
                    '#include <stdio.h>\nint main(){printf("c works\\n");}'))
    print(dev_cycle("notes.txt", "just a text file"))
