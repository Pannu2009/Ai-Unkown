#!/usr/bin/env python3
"""Upload the neurobrain worker script to Cloudflare. Uses the stored credential."""
import sys, json, uuid
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_response_body
import urllib.request, urllib.error

ACCOUNT = "abfb3c93f4016d2735bf73f27145c4ff"
SCRIPT_NAME = "neurobrain"

with open("worker/worker.js", "rb") as f:
    script = f.read()

metadata = json.dumps({
    "main_module": "worker.js",
    "compatibility_date": "2024-06-01",
    "bindings": [{"type": "kv_namespace", "name": "BRAIN",
                  "namespace_id": "ff0d7c88985844898d1041b9cb8333b4"}],
}).encode()

boundary = uuid.uuid4().hex.encode()
body = b""
body += b"--" + boundary + b"\r\n"
body += b'Content-Disposition: form-data; name="metadata"\r\n'
body += b"Content-Type: application/json\r\n\r\n" + metadata + b"\r\n"
body += b"--" + boundary + b"\r\n"
body += b'Content-Disposition: form-data; name="worker.js"; filename="worker.js"\r\n'
body += b"Content-Type: application/javascript+module\r\n\r\n" + script + b"\r\n"
body += b"--" + boundary + b"--\r\n"

url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/workers/scripts/{SCRIPT_NAME}"
req = urllib.request.Request(url, data=body, method="PUT")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
add_surrogate_to_request(req, "custom.cloudflare", allowed_hosts=["api.cloudflare.com"])
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(read_response_body(resp).decode())
        print("success:", out.get("success"))
        if not out.get("success"):
            print(json.dumps(out.get("errors"), indent=2)[:1500])
        else:
            print("script:", out["result"]["id"] if "result" in out else "?")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:1500]}")
    sys.exit(1)
