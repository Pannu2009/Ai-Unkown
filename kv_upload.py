"""Upload brain files to KV (weights sharded <20MB each). Usage: python3 kv_upload.py"""
import os, sys, json, glob
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request
import urllib.request

ACCT = "abfb3c93f4016d2735bf73f27145c4ff"
NS = "ff0d7c88985844898d1041b9cb8333b4"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "web_v2")

man = json.load(open(os.path.join(SRC, "manifest.json")))
FILES = {s: "application/octet-stream" for s in man.get("shards", ["weights.bin"])}
FILES["manifest.json"] = "application/json"
FILES["vocab.txt"] = "text/plain; charset=utf-8"

for name, ctype in FILES.items():
    with open(os.path.join(SRC, name), "rb") as f:
        data = f.read()
    url = (f"https://api.cloudflare.com/client/v4/accounts/{ACCT}/"
           f"storage/kv/namespaces/{NS}/values/{name}")
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": ctype})
    add_surrogate_to_request(req, "custom.cloudflare",
                             allowed_hosts=["api.cloudflare.com"])
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read(500).decode()
    print(name, len(data), "bytes ->", "ok" if '"success":true' in body else body[:120])
