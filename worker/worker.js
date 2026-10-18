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
      sub.textContent = "0.82M params · runs 100% in your browser · no servers";
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
const BRAINJS = `// brain.js - NeuroBrain inference in pure JavaScript. No dependencies.
// Loads weights from the worker's KV, runs the transformer locally.
const Brain = (() => {
  let T = {}, cfg = null, vocab = [], stoi = {};

  const get = (name, shape) => {
    const t = T[name];
    if (!t) throw new Error("missing tensor " + name);
    return t;
  };

  function matvec(W, x, rows, cols, bias) {
    const y = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      let s = bias ? bias[i] : 0;
      const off = i * cols;
      for (let j = 0; j < cols; j++) s += W[off + j] * x[j];
      y[i] = s;
    }
    return y;
  }

  function layernorm(x, w, b) {
    const n = x.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += x[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; }
    v = v / n + 1e-5;
    const inv = 1 / Math.sqrt(v);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - m) * inv * w[i] + b[i];
    return y;
  }

  function gelu(x) {
    const y = new Float32Array(x.length);
    const c = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      y[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }
    return y;
  }

  function softmax(x) {
    let mx = -1e30;
    for (let i = 0; i < x.length; i++) if (x[i] > mx) mx = x[i];
    const y = new Float32Array(x.length);
    let s = 0;
    for (let i = 0; i < x.length; i++) { y[i] = Math.exp(x[i] - mx); s += y[i]; }
    for (let i = 0; i < x.length; i++) y[i] /= s;
    return y;
  }

  function forward(ids) {
    const { n_layer, n_head, n_embd, block } = cfg;
    const hs = n_embd / n_head;
    const t = ids.length;
    // embed
    let x = new Array(t);
    const tokW = get("tok.weight"), posW = get("pos.weight");
    for (let i = 0; i < t; i++) {
      const row = new Float32Array(n_embd);
      const to = tokW.subarray(ids[i] * n_embd, ids[i] * n_embd + n_embd);
      const po = posW.subarray(i * n_embd, i * n_embd + n_embd);
      for (let j = 0; j < n_embd; j++) row[j] = to[j] + po[j];
      x[i] = row;
    }
    const applyLN = (rows, w, b) => rows.map(r => layernorm(r, w, b));
    for (let l = 0; l < n_layer; l++) {
      const p = \`blocks.\${l}.\`;
      let h = applyLN(x, get(p + "ln1.weight"), get(p + "ln1.bias"));
      // attention per position
      const out = new Array(t);
      for (let i = 0; i < t; i++) {
        const acc = new Float32Array(n_embd);
        for (let hd = 0; hd < n_head; hd++) {
          const qkvW = get(\`\${p}heads.\${hd}.qkv.weight\`);
          // q,k,v for all positions up to i
          const Q = [], K = [], V = [];
          for (let j = 0; j <= i; j++) {
            const qkv = matvec(qkvW, h[j], 3 * hs, n_embd, null);
            Q.push(qkv.subarray(0, hs)); K.push(qkv.subarray(hs, 2 * hs)); V.push(qkv.subarray(2 * hs, 3 * hs));
          }
          const qi = Q[i];
          const scores = new Float32Array(i + 1);
          const sc = 1 / Math.sqrt(hs);
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k < hs; k++) s += qi[k] * K[j][k];
            scores[j] = s * sc;
          }
          const probs = softmax(scores);
          const o = new Float32Array(hs);
          for (let j = 0; j <= i; j++)
            for (let k = 0; k < hs; k++) o[k] += probs[j] * V[j][k];
          acc.set(o, hd * hs);
        }
        const proj = matvec(get(p + "proj.weight"), acc, n_embd, n_embd, null);
        const row = new Float32Array(n_embd);
        for (let j = 0; j < n_embd; j++) row[j] = x[i][j] + proj[j];
        out[i] = row;
      }
      x = out;
      // mlp
      let h2 = applyLN(x, get(p + "ln2.weight"), get(p + "ln2.bias"));
      x = x.map((row, i) => {
        const m1 = matvec(get(p + "mlp.0.weight"), h2[i], 4 * n_embd, n_embd, get(p + "mlp.0.bias"));
        const g = gelu(m1);
        const m2 = matvec(get(p + "mlp.2.weight"), g, n_embd, 4 * n_embd, get(p + "mlp.2.bias"));
        const r2 = new Float32Array(n_embd);
        for (let j = 0; j < n_embd; j++) r2[j] = row[j] + m2[j];
        return r2;
      });
    }
    const last = layernorm(x[t - 1], get("ln.weight"), get("ln.bias"));
    return matvec(get("head.weight"), last, cfg.vocab, n_embd, null);
  }

  async function load(base) {
    const man = await (await fetch(base + "/data/manifest.json")).json();
    cfg = man;
    const vt = await (await fetch(base + "/data/vocab.txt")).text();
    vocab = [...vt]; stoi = {}; vocab.forEach((c, i) => stoi[c] = i);
    const buf = await (await fetch(base + "/data/weights.bin")).arrayBuffer();
    const f32 = new Float32Array(buf);
    for (const t of man.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      T[t.name] = f32.subarray(t.offset, t.offset + n);
    }
  }

  function sample(logits, temp) {
    const p = softmax(logits.map(v => v / temp));
    let r = Math.random(), acc = 0;
    for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) return i; }
    return p.length - 1;
  }

  function reply(prompt, n = 120, temp = 0.8) {
    let ids = [...prompt].map(c => stoi[c] ?? 0);
    const startLen = ids.length;
    for (let s = 0; s < n; s++) {
      const ctx = ids.slice(-cfg.block);
      const logits = forward(ctx);
      ids.push(sample(Array.from(logits), temp));
    }
    return ids.slice(startLen).map(i => vocab[i]).join("");
  }

  return { load, reply };
})();
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
    if (p === "/health") return Response.json({ ok: true, version: "0.2" });
    if (p === "/rules") return new Response(RULES, { headers: { "content-type": "text/plain" } });
    if (p.startsWith("/data/")) {
      const key = p.slice(6);
      const v = await env.BRAIN.get(key, "arrayBuffer");
      if (v === null) return new Response("not found", { status: 404 });
      const ct = key.endsWith(".json") ? "application/json" : key.endsWith(".txt") ? "text/plain" : "application/octet-stream";
      return new Response(v, { headers: { "content-type": ct, "cache-control": "public, max-age=86400" } });
    }
    return Response.json({ name: null, note: "unnamed — the agent names itself", version: "0.2" });
  }
};
