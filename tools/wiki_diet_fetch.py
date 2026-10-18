#!/usr/bin/env python3
"""Bulk-fetch Wikipedia extracts, ONE title per request (batch extracts come back empty)."""
import json, time, urllib.request, urllib.parse, os, re

STAGING = os.path.expanduser("~/workspace/neuro-brain/data/staging")
UA = {"User-Agent": "NeuroBrainDietBot/1.0 (private AI training corpus builder)"}
CAP = 40000
PER_FILE = 20

TOPICS = [
"Newton's laws of motion", "Thermodynamics", "Electromagnetism", "Quantum mechanics",
"Theory of relativity", "Gravity", "Friction", "Energy", "Wave", "Electricity",
"Atom", "Periodic table", "Chemical bond", "Chemical reaction", "Acid",
"Base (chemistry)", "Redox", "Catalysis", "Metal", "Polymer",
"Cell (biology)", "DNA", "Protein", "Photosynthesis", "Evolution",
"Human body", "Immune system", "Neuron", "Digestion", "Cellular respiration",
"Plate tectonics", "Solar System", "Black hole", "Big Bang", "Star",
"Moon", "Mars", "Water cycle", "Volcano", "Earthquake",
"Internal combustion engine", "Transmission (mechanical device)", "Manual transmission",
"Automatic transmission", "Differential (mechanical device)", "Suspension (vehicle)",
"Brake", "Anti-lock braking system", "Turbocharger", "Electric vehicle",
"Hybrid electric vehicle", "Automobile",
"Algebra", "Geometry", "Calculus", "Trigonometry", "Probability",
"Statistics", "Pythagorean theorem", "Prime number", "Fraction", "Exponentiation",
"Quadratic equation", "Mathematical proof",
"Ancient Egypt", "Roman Empire", "Maurya Empire", "Achaemenid Empire",
"Mongol Empire", "Ottoman Empire", "Mughal Empire", "British Raj",
"World War I", "World War II", "Industrial Revolution", "Partition of India",
"Sikh Empire", "Ranjit Singh", "Silk Road", "Cold War",
"Mohandas Gandhi", "Bhagat Singh",
"Sikhism", "Guru Nanak", "Guru Gobind Singh", "Khalsa", "Golden Temple",
"Punjabi language", "Gurmukhi", "Bhangra (dance)", "Heer Ranjha", "Bulleh Shah",
"Internet", "Computer", "Smartphone", "Electrical grid", "Refrigerator",
"Air conditioning", "Jet engine", "Global Positioning System", "Nuclear power",
"Solar cell", "Wind turbine", "Microwave oven", "Elevator", "Lock (security device)",
"Logic", "Critical thinking", "Cognitive bias", "Problem solving",
"Fermi problem", "Game theory", "Monty Hall problem", "Occam's razor",
"Lua (programming language)", "Roblox", "Coroutine", "Sorting algorithm",
"Binary search", "Debugging", "Data structure", "Algorithm",
]

# skip titles already in wiki_batch_01.txt
done = set()
p0 = os.path.join(STAGING, "wiki_batch_01.txt")
if os.path.exists(p0):
    with open(p0, encoding="utf-8") as f:
        done = set(re.findall(r"^Human: tell me about (.+)$", f.read(), re.M))
print("already have:", len(done))

def fetch_one(title):
    params = {"action": "query", "prop": "extracts", "explaintext": "1",
              "exsectionformat": "plain", "redirects": "1",
              "titles": title, "format": "json", "formatversion": "2"}
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    txt = (pages[0].get("extract") or "").strip()
    if len(txt) < 2000:
        return None
    return pages[0]["title"], txt[:CAP]

articles = []
for i, t in enumerate(TOPICS):
    if t in done:
        continue
    try:
        res = fetch_one(t)
    except Exception as e:
        print("FAIL %s: %s" % (t, e))
        res = None
    if res:
        articles.append(res)
        print("[%d] ok: %s (%d)" % (len(articles), res[0], len(res[1])))
    else:
        print("[--] skip: %s" % t)
    time.sleep(1.2)

print("new articles:", len(articles))
for fi in range(0, len(articles), PER_FILE):
    chunk = articles[fi:fi+PER_FILE]
    path = os.path.join(STAGING, "wiki_batch_%02d.txt" % (fi//PER_FILE + 2))
    with open(path, "w", encoding="utf-8") as f:
        f.write("Wikipedia extracts. Factual reference material for training.\n\n")
        for title, txt in chunk:
            f.write("Human: tell me about %s\n" % title)
            f.write("Brain: %s\n\n" % txt)
    print("wrote", path)
