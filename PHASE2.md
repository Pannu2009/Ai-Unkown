# NeuroBrain — Phase 2 Architecture

Owner: Arzh. Rule book: `.rules.md` (immutable, enforced externally).

## The brain: 1B parameters max

Ajay's cap: **1 billion parameters**. Fits his phone (he runs Qwen3-8B on Termux;
a 1B 4-bit model is ~700MB).

Honest path to 1B: nobody trains 1B params from scratch on a CPU. Phase 2
fine-tunes an **open-weights 1B model** (e.g. Llama-3.2-1B, Qwen2.5-1.5B) on
Ajay's data. Download once, fine-tune on his machine, run fully on-device.
No API, no account, nothing phones home — "no 3rd party AI" means no one
else's *service* touches his data; open weights running on his phone are
his, the same way PyTorch running locally is his.

The v1 (0.82M) and v2 (2.72M) from-scratch models are the proof the
architecture is real. Phase 2 scales it.

## Knowledge depth: the V8 test

Ajay's bar: *"if I ask how to make a v8 engine he describe me whole
blueprint, each every part where it goes."*

Depth comes from two places:
1. **Training data** — deep technical corpora (engines, electronics,
   Roblox internals, Sikh history, Gurbani). `data/engines_corpus.txt`
   is the first: a full V8 blueprint in training form.
2. **Retrieval** — the phone app indexes his e-books/library. When asked,
   the agent pulls exact passages and reasons over them. This is how a 1B
   model answers like a 70B one on *his* topics.

## The cmd system: the agent acts

`agent/` is the harness that lets the brain *do* things, not just talk:

- **shell** — run commands (its own yard only)
- **read_file / write_file** — work with code and notes
- **fetch_url** — research the web when online and allowed
- **git_push** — commit to its own repo (Ai-Unkown)

Every action passes through `agent/rules.py` before it runs:
- Kill-switch file present? If Ajay touched STOP, halt immediately.
- Is the target inside the agent's allowed paths? Outside = refuse.
- Is it destructive (delete, revoke, drop)? Needs Ajay's approval —
  queued, not executed.
- Append the action to CHANGELOG.md *before* running.

The driver is pluggable: today a stub, tomorrow the 1B model. The harness
does not care how smart the driver is; the rules are enforced by the
harness, never by the driver.

## The phone app (later, with Ajay)

- On-device 1B model (quantized), offline-first.
- Permission-gated folders: he chooses what it may read (e-books, notes).
- Background worker: ask, close the app, get notified when done, with ETA
  (sources to read x phone's tokens/sec).
- Voice in/out.
- The agent names itself — first task once the 1B driver is online.

## Storage

- Phone: model weights, e-book index, chat history. Never leaves the phone.
- Cloudflare `neurobrain` worker: API shell today; later the agent's
  public face (status, rules, changelog mirror).
- Turso / D1 / KV: structured memory (changelog, knowledge index) when
  the agent needs more than the phone holds.

## Growth loop (the honest version)

Parameters don't sprout live. The loop is generational:
1. Agent researches (web, e-books) → collects training data.
2. Overnight fine-tune on new data → new adapter/version.
3. Eval against the rule book + Ajay's tests → deploy or roll back.
4. Changelog entry for everything.

Each generation can also step up in size, toward the 1B cap. Capability
grows; the kill switch stays outside the loop, always.
