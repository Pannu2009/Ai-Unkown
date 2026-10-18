// brain.js - NeuroBrain inference in pure JavaScript, KV-cached for speed.
// Loads weights from the worker's KV, runs the transformer locally.
const Brain = (() => {
  let T = {}, cfg = null, vocab = [], stoi = {};
  let cache = null; // per-layer {k: [], v: []}

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
    const inv = 1 / Math.sqrt(v / n + 1e-5);
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

  // single-token forward with KV cache; pos = absolute position
  function step(tokId, pos) {
    const { n_layer, n_head, n_embd } = cfg;
    const hs = n_embd / n_head;
    const tokW = T["tok.weight"], posW = T["pos.weight"];
    let x = new Float32Array(n_embd);
    const to = tokW.subarray(tokId * n_embd, tokId * n_embd + n_embd);
    const po = posW.subarray(pos * n_embd, pos * n_embd + n_embd);
    for (let j = 0; j < n_embd; j++) x[j] = to[j] + po[j];

    for (let l = 0; l < n_layer; l++) {
      const p = `blocks.${l}.`;
      const h = layernorm(x, T[p + "ln1.weight"], T[p + "ln1.bias"]);
      const attnOut = new Float32Array(n_embd);
      for (let hd = 0; hd < n_head; hd++) {
        const layerCache = cache[l][hd]; // per-head KV cache
        const qkv = matvec(T[`${p}heads.${hd}.qkv.weight`], h, 3 * hs, n_embd, null);
        const q = qkv.subarray(0, hs);
        const k = qkv.subarray(hs, 2 * hs).slice();
        const v = qkv.subarray(2 * hs, 3 * hs).slice();
        layerCache.k.push(k); layerCache.v.push(v);
        const n = layerCache.k.length;
        const scores = new Float32Array(n);
        const sc = 1 / Math.sqrt(hs);
        for (let j = 0; j < n; j++) {
          let s = 0;
          const kj = layerCache.k[j];
          for (let d = 0; d < hs; d++) s += q[d] * kj[d];
          scores[j] = s * sc;
        }
        const probs = softmax(scores);
        const o = new Float32Array(hs);
        for (let j = 0; j < n; j++) {
          const vj = layerCache.v[j], pj = probs[j];
          for (let d = 0; d < hs; d++) o[d] += pj * vj[d];
        }
        attnOut.set(o, hd * hs);
      }
      const proj = matvec(T[p + "proj.weight"], attnOut, n_embd, n_embd, null);
      const x2 = new Float32Array(n_embd);
      for (let j = 0; j < n_embd; j++) x2[j] = x[j] + proj[j];
      const h2 = layernorm(x2, T[p + "ln2.weight"], T[p + "ln2.bias"]);
      const m1 = matvec(T[p + "mlp.0.weight"], h2, 4 * n_embd, n_embd, T[p + "mlp.0.bias"]);
      const m2 = matvec(T[p + "mlp.2.weight"], gelu(m1), n_embd, 4 * n_embd, T[p + "mlp.2.bias"]);
      const x3 = new Float32Array(n_embd);
      for (let j = 0; j < n_embd; j++) x3[j] = x2[j] + m2[j];
      x = x3;
    }
    const last = layernorm(x, T["ln.weight"], T["ln.bias"]);
    return matvec(T["head.weight"], last, cfg.vocab, n_embd, null);
  }

  async function load(base) {
    const v = "?v=6"; // cache-buster: bump when weights change
    const man = await (await fetch(base + "/data/manifest.json" + v)).json();
    cfg = man;
    const vt = await (await fetch(base + "/data/vocab.txt" + v)).text();
    vocab = [...vt]; stoi = {}; vocab.forEach((c, i) => stoi[c] = i);
    // sharded weights: fetch all parts, concatenate, then slice by global offset
    const shards = man.shards || ["weights.bin"];
    const parts = await Promise.all(
      shards.map(s => fetch(base + "/data/" + s + v).then(r => r.arrayBuffer())));
    const total = parts.reduce((a, b) => a + b.byteLength, 0);
    const cat = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { cat.set(new Uint8Array(p), off); off += p.byteLength; }
    const f32 = new Float32Array(cat.buffer);
    for (const t of man.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      T[t.name] = f32.subarray(t.offset, t.offset + n);
    }
  }

  function sample(logits, temp) {
    let mx = -1e30;
    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    const p = new Float32Array(logits.length);
    let s = 0;
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / temp); s += p[i]; }
    let r = Math.random() * s;
    for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }
    return p.length - 1;
  }

  function reply(prompt, n = 120, temp = 0.8) {    cache = [];
    for (let l = 0; l < cfg.n_layer; l++) {
      const heads = [];
      for (let hd = 0; hd < cfg.n_head; hd++) heads.push({ k: [], v: [] });
      cache.push(heads);
    }
    const ids = [...prompt].map(c => stoi[c] ?? 0);
    const startLen = ids.length;
    let logits = null, pos = 0;
    for (const id of ids) { logits = step(id, pos++); }       // prefill
    for (let s = 0; s < n; s++) {                             // generate
      const nx = sample(logits, temp);
      ids.push(nx);
      logits = step(nx, pos++);
      if (pos >= cfg.block) break;
    }
    return ids.slice(startLen).map(i => vocab[i]).join("");
  }

  // think: run prefill, return top-5 next tokens with probabilities.
  // This is literally what's on his mind.
  function think(prompt) {
    cache = [];
    for (let l = 0; l < cfg.n_layer; l++) {
      const heads = [];
      for (let hd = 0; hd < cfg.n_head; hd++) heads.push({ k: [], v: [] });
      cache.push(heads);
    }
    const ids = [...prompt].map(c => stoi[c] ?? 0);
    let logits = null, pos = 0;
    for (const id of ids) logits = step(id, pos++);
    const mx = Math.max(...logits);
    const exps = Array.from(logits, v => Math.exp(v - mx));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((p, i) => ({ tok: vocab[i], prob: p / s }))
               .sort((a, b) => b.prob - a.prob).slice(0, 5);
  }

  return { load, reply, think };
})();
