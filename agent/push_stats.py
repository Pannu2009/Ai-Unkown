"""Push trainstats.json to KV every 5 min (cron)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainstats import stats
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request
import urllib.request

ACCT = "abfb3c93f4016d2735bf73f27145c4ff"
NS = "ff0d7c88985844898d1041b9cb8333b4"

data = json.dumps(stats()).encode()
url = (f"https://api.cloudflare.com/client/v4/accounts/{ACCT}/"
       f"storage/kv/namespaces/{NS}/values/trainstats.json")
req = urllib.request.Request(url, data=data, method="PUT",
                             headers={"Content-Type": "application/json"})
add_surrogate_to_request(req, "custom.cloudflare",
                         allowed_hosts=["api.cloudflare.com"])
with urllib.request.urlopen(req, timeout=30) as r:
    body = r.read(300).decode()
print("trainstats ->", "ok" if '"success":true' in body else body[:100])
