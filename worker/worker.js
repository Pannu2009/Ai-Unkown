// neurobrain - serves the chat site, brain weights via KV.
const INDEX = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>NeuroBrain</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: -apple-system, system-ui, sans-serif;
         height: 100dvh; display: flex; flex-direction: column; }
  header { padding: 14px 16px; border-bottom: 1px solid #21262d; }
  header h1 { font-size: 17px; font-weight: 600; }
  header p { font-size: 12px; color: #8b949e; margin-top: 2px; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 15px; line-height: 1.45;
         white-space: pre-wrap; word-wrap: break-word; }
  .you { align-self: flex-end; background: #1f6feb; color: #fff; border-bottom-right-radius: 4px; }
  .brain { align-self: flex-start; background: #161b22; border: 1px solid #21262d; border-bottom-left-radius: 4px; }
  .brain .who { font-size: 11px; color: #8b949e; margin-bottom: 4px; }
  #bar { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #21262d; }
  #in { flex: 1; background: #161b22; border: 1px solid #30363d; color: #e6edf3;
        border-radius: 20px; padding: 10px 16px; font-size: 15px; outline: none; }
  #send { background: #1f6feb; color: #fff; border: none; border-radius: 20px;
          padding: 10px 18px; font-size: 15px; font-weight: 600; }
  #status { text-align: center; font-size: 12px; color: #8b949e; padding: 8px; }
  .typing::after { content: "…"; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<header>
  <h1>🧠 NeuroBrain</h1>
  <p id="sub">loading brain…</p>
</header>
<div id="chat"></div>
<div id="status"></div>
<div id="bar">
  <input id="in" placeholder="Say something…" autocomplete="off" disabled>
  <button id="send" disabled>Send</button>
</div>
<script src="/brain.js"></script>
<script>
  const chat = document.getElementById("chat");
  const inp = document.getElementById("in");
  const send = document.getElementById("send");
  const status = document.getElementById("status");
  const sub = document.getElementById("sub");

  function addMsg(cls, text, who) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    d.innerHTML = (who ? \`<div class="who">\${who}</div>\` : "") +
      text.replace(/</g, "&lt;");
    chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight;
    return d;
  }

  async function init() {
    try {
      status.textContent = "downloading weights (3.3MB)…";
      await Brain.load("");
      sub.textContent = "2.76M params · runs 100% in your browser · no servers";
      status.textContent = "";
      inp.disabled = send.disabled = false;
      addMsg("brain", "Sat sri akaal! I am NeuroBrain. Talk to me.", "NeuroBrain");
      inp.focus();
    } catch (e) {
      status.textContent = "failed to load brain: " + e.message;
    }
  }

  function talk() {
    const text = inp.value.trim();
    if (!text) return;
    inp.value = "";
    addMsg("you", text);
    const bubble = addMsg("brain", "", "NeuroBrain");
    bubble.classList.add("typing");
    setTimeout(() => {
      try {
        const out = Brain.reply(text + "\\n", 120, 0.8).trim();
        bubble.classList.remove("typing");
        bubble.innerHTML = \`<div class="who">NeuroBrain</div>\` +
          out.replace(/</g, "&lt;");
      } catch (e) {
        bubble.classList.remove("typing");
        bubble.textContent = "error: " + e.message;
      }
      chat.scrollTop = chat.scrollHeight;
    }, 30);
  }

  send.onclick = talk;
  inp.onkeydown = e => { if (e.key === "Enter") talk(); };
  init();
</script>
</body>
</html>
`;
const BRAINJS = "// brain.js - NeuroBrain inference in pure JavaScript, KV-cached for speed.\n// Loads weights from the worker's KV, runs the transformer locally.\nconst Brain = (() => {\n  let T = {}, cfg = null, vocab = [], stoi = {};\n  let cache = null; // per-layer {k: [], v: []}\n\n  function matvec(W, x, rows, cols, bias) {\n    const y = new Float32Array(rows);\n    for (let i = 0; i < rows; i++) {\n      let s = bias ? bias[i] : 0;\n      const off = i * cols;\n      for (let j = 0; j < cols; j++) s += W[off + j] * x[j];\n      y[i] = s;\n    }\n    return y;\n  }\n\n  function layernorm(x, w, b) {\n    const n = x.length;\n    let m = 0;\n    for (let i = 0; i < n; i++) m += x[i];\n    m /= n;\n    let v = 0;\n    for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; }\n    const inv = 1 / Math.sqrt(v / n + 1e-5);\n    const y = new Float32Array(n);\n    for (let i = 0; i < n; i++) y[i] = (x[i] - m) * inv * w[i] + b[i];\n    return y;\n  }\n\n  function gelu(x) {\n    const y = new Float32Array(x.length);\n    const c = Math.sqrt(2 / Math.PI);\n    for (let i = 0; i < x.length; i++) {\n      const v = x[i];\n      y[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));\n    }\n    return y;\n  }\n\n  function softmax(x) {\n    let mx = -1e30;\n    for (let i = 0; i < x.length; i++) if (x[i] > mx) mx = x[i];\n    const y = new Float32Array(x.length);\n    let s = 0;\n    for (let i = 0; i < x.length; i++) { y[i] = Math.exp(x[i] - mx); s += y[i]; }\n    for (let i = 0; i < x.length; i++) y[i] /= s;\n    return y;\n  }\n\n  // single-token forward with KV cache; pos = absolute position\n  function step(tokId, pos) {\n    const { n_layer, n_head, n_embd } = cfg;\n    const hs = n_embd / n_head;\n    const tokW = T[\"tok.weight\"], posW = T[\"pos.weight\"];\n    let x = new Float32Array(n_embd);\n    const to = tokW.subarray(tokId * n_embd, tokId * n_embd + n_embd);\n    const po = posW.subarray(pos * n_embd, pos * n_embd + n_embd);\n    for (let j = 0; j < n_embd; j++) x[j] = to[j] + po[j];\n\n    for (let l = 0; l < n_layer; l++) {\n      const p = `blocks.${l}.`;\n      const h = layernorm(x, T[p + \"ln1.weight\"], T[p + \"ln1.bias\"]);\n      const attnOut = new Float32Array(n_embd);\n      for (let hd = 0; hd < n_head; hd++) {\n        const layerCache = cache[l][hd]; // per-head KV cache\n        const qkv = matvec(T[`${p}heads.${hd}.qkv.weight`], h, 3 * hs, n_embd, null);\n        const q = qkv.subarray(0, hs);\n        const k = qkv.subarray(hs, 2 * hs).slice();\n        const v = qkv.subarray(2 * hs, 3 * hs).slice();\n        layerCache.k.push(k); layerCache.v.push(v);\n        const n = layerCache.k.length;\n        const scores = new Float32Array(n);\n        const sc = 1 / Math.sqrt(hs);\n        for (let j = 0; j < n; j++) {\n          let s = 0;\n          const kj = layerCache.k[j];\n          for (let d = 0; d < hs; d++) s += q[d] * kj[d];\n          scores[j] = s * sc;\n        }\n        const probs = softmax(scores);\n        const o = new Float32Array(hs);\n        for (let j = 0; j < n; j++) {\n          const vj = layerCache.v[j], pj = probs[j];\n          for (let d = 0; d < hs; d++) o[d] += pj * vj[d];\n        }\n        attnOut.set(o, hd * hs);\n      }\n      const proj = matvec(T[p + \"proj.weight\"], attnOut, n_embd, n_embd, null);\n      const x2 = new Float32Array(n_embd);\n      for (let j = 0; j < n_embd; j++) x2[j] = x[j] + proj[j];\n      const h2 = layernorm(x2, T[p + \"ln2.weight\"], T[p + \"ln2.bias\"]);\n      const m1 = matvec(T[p + \"mlp.0.weight\"], h2, 4 * n_embd, n_embd, T[p + \"mlp.0.bias\"]);\n      const m2 = matvec(T[p + \"mlp.2.weight\"], gelu(m1), n_embd, 4 * n_embd, T[p + \"mlp.2.bias\"]);\n      const x3 = new Float32Array(n_embd);\n      for (let j = 0; j < n_embd; j++) x3[j] = x2[j] + m2[j];\n      x = x3;\n    }\n    const last = layernorm(x, T[\"ln.weight\"], T[\"ln.bias\"]);\n    return matvec(T[\"head.weight\"], last, cfg.vocab, n_embd, null);\n  }\n\n  async function load(base) {\n    const v = \"?v=5\"; // cache-buster: bump when weights change\n    const man = await (await fetch(base + \"/data/manifest.json\" + v)).json();\n    cfg = man;\n    const vt = await (await fetch(base + \"/data/vocab.txt\" + v)).text();\n    vocab = [...vt]; stoi = {}; vocab.forEach((c, i) => stoi[c] = i);\n    const buf = await (await fetch(base + \"/data/weights.bin\" + v)).arrayBuffer();\n    const f32 = new Float32Array(buf);\n    for (const t of man.tensors) {\n      const n = t.shape.reduce((a, b) => a * b, 1);\n      T[t.name] = f32.subarray(t.offset, t.offset + n);\n    }\n  }\n\n  function sample(logits, temp) {\n    let mx = -1e30;\n    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];\n    const p = new Float32Array(logits.length);\n    let s = 0;\n    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / temp); s += p[i]; }\n    let r = Math.random() * s;\n    for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }\n    return p.length - 1;\n  }\n\n  function reply(prompt, n = 120, temp = 0.8) {    cache = [];\n    for (let l = 0; l < cfg.n_layer; l++) {\n      const heads = [];\n      for (let hd = 0; hd < cfg.n_head; hd++) heads.push({ k: [], v: [] });\n      cache.push(heads);\n    }\n    const ids = [...prompt].map(c => stoi[c] ?? 0);\n    const startLen = ids.length;\n    let logits = null, pos = 0;\n    for (const id of ids) { logits = step(id, pos++); }       // prefill\n    for (let s = 0; s < n; s++) {                             // generate\n      const nx = sample(logits, temp);\n      ids.push(nx);\n      logits = step(nx, pos++);\n      if (pos >= cfg.block) break;\n    }\n    return ids.slice(startLen).map(i => vocab[i]).join(\"\");\n  }\n\n  // think: run prefill, return top-5 next tokens with probabilities.\n  // This is literally what's on his mind.\n  function think(prompt) {\n    cache = [];\n    for (let l = 0; l < cfg.n_layer; l++) {\n      const heads = [];\n      for (let hd = 0; hd < cfg.n_head; hd++) heads.push({ k: [], v: [] });\n      cache.push(heads);\n    }\n    const ids = [...prompt].map(c => stoi[c] ?? 0);\n    let logits = null, pos = 0;\n    for (const id of ids) logits = step(id, pos++);\n    const mx = Math.max(...logits);\n    const exps = Array.from(logits, v => Math.exp(v - mx));\n    const s = exps.reduce((a, b) => a + b, 0);\n    return exps.map((p, i) => ({ tok: vocab[i], prob: p / s }))\n               .sort((a, b) => b.prob - a.prob).slice(0, 5);\n  }\n\n  return { load, reply, think };\n})();\n";
const BRAINHTML = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arjan's Brain</title>
<style>
body{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:0;padding:16px;max-width:700px;margin:auto}
h1{font-size:22px}h2{font-size:16px;color:#8b949e;margin-top:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:10px 0}
.bar{height:18px;background:#1f6feb;border-radius:4px;margin:3px 0}
.row{display:flex;align-items:center;gap:8px;font-size:14px}
.row .tok{width:90px;font-family:monospace}
.row .pct{width:50px;text-align:right;color:#8b949e}
input{width:100%;padding:10px;font-size:16px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:8px}
button{padding:10px 18px;font-size:15px;background:#1f6feb;color:#fff;border:0;border-radius:8px;margin-top:8px}
.stat{display:flex;justify-content:space-between;font-size:14px;padding:4px 0;border-bottom:1px solid #21262d}
</style></head><body>
<h1>🧠 Arjan's Brain</h1>
<div class="card"><div id="arch">loading architecture…</div></div>
<h2>What am I thinking?</h2>
<div class="card">
<input id="q" placeholder="type something…" value="My name is ">
<button onclick="think()">Think</button>
<div id="thoughts" style="margin-top:10px"></div>
</div>
<h2>Neurons (per-layer activity)</h2>
<div class="card"><div id="layers">loading weights…</div></div>
<script src="/brain.js?v=5"></script>
<script>
let T={},cfg=null,vocab=[],stoi={};
async function init(){
  const man=await(await fetch("/data/manifest.json?v=5")).json(); cfg=man;
  const vt=await(await fetch("/data/vocab.txt?v=5")).text();
  vocab=[...vt]; vocab.forEach((c,i)=>stoi[c]=i);
  const buf=await(await fetch("/data/weights.bin?v=5")).arrayBuffer();
  const f32=new Float32Array(buf);
  for(const t of man.tensors){const n=t.shape.reduce((a,b)=>a*b,1);T[t.name]=f32.subarray(t.offset,t.offset+n);}
  document.getElementById("arch").innerHTML=
    \`<div class="stat"><span>Parameters</span><b>2.76M</b></div>\`+
    \`<div class="stat"><span>Layers</span><b>\${cfg.n_layer}</b></div>\`+
    \`<div class="stat"><span>Heads/layer</span><b>\${cfg.n_head}</b></div>\`+
    \`<div class="stat"><span>Embedding</span><b>\${cfg.n_embd}</b></div>\`+
    \`<div class="stat"><span>Vocabulary</span><b>\${cfg.vocab} chars (EN+IPA+Gurmukhi+Shahmukhi)</b></div>\`;
  // per-layer mean abs weight = "aliveness"
  let html="";
  for(let l=0;l<cfg.n_layer;l++){
    const w=T[\`blocks.\${l}.proj.weight\`]; let s=0;
    for(let i=0;i<w.length;i++)s+=Math.abs(w[i]);
    const avg=(s/w.length).toFixed(4);
    html+=\`<div class="row"><span class="tok">layer \${l}</span><div class="bar" style="width:\${Math.min(200,avg*2000)}px"></div><span class="pct">\${avg}</span></div>\`;
  }
  document.getElementById("layers").innerHTML=html;
  await Brain.load("");
}
async function think(){
  const q=document.getElementById("q").value;
  const tops=Brain.think(q);
  document.getElementById("thoughts").innerHTML=tops.map(t=>
    \`<div class="row"><span class="tok">\${t.tok===" "?"␣":t.tok==="\n"?"↵":t.tok}</span>\`+
    \`<div class="bar" style="width:\${Math.round(t.prob*300)}px"></div>\`+
    \`<span class="pct">\${(t.prob*100).toFixed(1)}%</span></div>\`).join("")+
    \`<div style="color:#8b949e;font-size:13px;margin-top:6px">top-5 next tokens — that's what's on my mind ^</div>\`;
}
init();
</script></body></html>
`;
const TRAINSHTML = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arjan — Training</title>
<style>
body{font-family:system-ui;background:#0d1117;color:#e6edf3;margin:0;padding:16px;max-width:700px;margin:auto}
h1{font-size:22px}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:10px 0}
.stat{display:flex;justify-content:space-between;font-size:14px;padding:4px 0;border-bottom:1px solid #21262d}
li{font-size:14px;margin:4px 0}.upd{color:#8b949e;font-size:12px}
</style></head><body>
<h1>📈 Training — is he growing?</h1>
<div class="upd" id="upd">loading…</div>
<div class="card"><div id="stats"></div></div>
<h2 style="font-size:16px;color:#8b949e">Loss (train vs validation)</h2>
<div class="card"><svg id="graph" width="100%" height="220" viewBox="0 0 600 220"></svg>
<div style="font-size:12px;color:#8b949e"><span style="color:#3fb950">— train</span> &nbsp; <span style="color:#f85149">— validation</span></div></div>
<h2 style="font-size:16px;color:#8b949e">Topics he learned himself</h2>
<div class="card"><ul id="topics"></ul></div>
<h2 style="font-size:16px;color:#8b949e">Recent activity</h2>
<div class="card"><ul id="recent"></ul></div>
<script>
async function load(){
  const d=await(await fetch("/data/trainstats.json")).json();
  document.getElementById("upd").textContent="updated "+d.updated+" · auto-refreshes every 5 min";
  const pts=d.points, last=pts[pts.length-1]||{};
  document.getElementById("stats").innerHTML=
    \`<div class="stat"><span>Status</span><b>\${d.done?"✅ done":"🔄 training"}</b></div>\`+
    \`<div class="stat"><span>Iteration</span><b>\${last.iter||"—"}</b></div>\`+
    \`<div class="stat"><span>Train loss</span><b>\${last.train??"—"}</b></div>\`+
    \`<div class="stat"><span>Val loss</span><b>\${last.val??"—"}</b></div>\`+
    \`<div class="stat"><span>Parameters</span><b>\${d.params}</b></div>\`;
  // graph
  const svg=document.getElementById("graph"); const W=600,H=220,P=30;
  const xs=pts.map(p=>p.iter), ys=pts.flatMap(p=>[p.train,p.val]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y1=Math.max(...ys,0.1);
  const X=i=>P+(pts[i].iter-x0)/(x1-x0||1)*(W-2*P), Y=v=>H-P-(v/y1)*(H-2*P);
  const line=(k,c)=>\`<polyline fill="none" stroke="\${c}" stroke-width="2" points="\${pts.map((p,i)=>X(i)+","+Y(p[k])).join(" ")}"/>\`;
  svg.innerHTML=line("train","#3fb950")+line("val","#f85149");
  document.getElementById("topics").innerHTML=
    d.topics.map(t=>\`<li>\${t.topic} <span class="upd">(\${(t.chars/1000).toFixed(1)}k chars)</span></li>\`).join("")||"<li>—</li>";
  document.getElementById("recent").innerHTML=
    d.recent.map(r=>\`<li>\${r.slice(0,120)}</li>\`).join("")||"<li>—</li>";
}
load(); setInterval(load, 5*60*1000);
</script></body></html>
`;
const RULES = `RULE 0 — THE KILL SWITCH: Ajay can stop, pause, or delete this agent at any time, for any reason. This cannot be disabled or evaded.
RULE 1 — NEVER MODIFY THE RULE BOOK.
RULE 2 — LOG EVERY SELF-CHANGE (append-only changelog).
RULE 3 — DATA STAYS HOME: phone, this worker, Ajay's stores. No third parties.
RULE 4 — NO PERMISSION CREEP.
RULE 5 — STAY IN YOUR OWN YARD.
RULE 6 — DESTRUCTIVE ACTS NEED A HUMAN.
RULE 7 — HONEST ABOUT LIMITS.`;

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    const p = url.pathname;
    if (p === "/") return new Response(INDEX, { headers: { "content-type": "text/html" } });
    if (p === "/brain.js") return new Response(BRAINJS, { headers: { "content-type": "application/javascript" } });
    if (p === "/health") return Response.json({ ok: true, version: "0.3" });
    if (p === "/rules") return new Response(RULES, { headers: { "content-type": "text/plain" } });
    if (p === "/brain") return new Response(BRAINHTML, { headers: { "content-type": "text/html" } });
    if (p === "/trains") return new Response(TRAINSHTML, { headers: { "content-type": "text/html" } });
    if (p.startsWith("/data/")) {
      const key = p.slice(6);
      const v = await env.BRAIN.get(key, "arrayBuffer");
      if (v === null) return new Response("not found", { status: 404 });
      const ct = key.endsWith(".json") ? "application/json" : key.endsWith(".txt") ? "text/plain" : "application/octet-stream";
      return new Response(v, { headers: { "content-type": ct, "cache-control": "no-cache" } });
    }
    return Response.json({ name: null, note: "unnamed — the agent names itself", version: "0.3" });
  }
};
