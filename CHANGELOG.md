# NeuroBrain Changelog (append-only)

## 2026-10-18T09:15Z
v0.1: from-scratch micro transformer (0.82M params), trained on Shakespeare, val loss 1.687. chat.py (torch) + chat_numpy.py (numpy, Termux-friendly).

## 2026-10-18T09:30Z
v0.2: web chat live at https://neurobrain.ajay15deep.workers.dev — brain runs 100% in-browser (brain.js), weights (3.3MB) served from worker KV. Rule book published at /rules.

## 2026-10-18T09:45Z
Overnight: v2 training started (2.72M params, Lua + chat + Sikh history + Gurbani + Punjabi corpora). Phase 2 architecture (PHASE2.md), agent cmd harness with rule enforcement, V8 engine corpus.
