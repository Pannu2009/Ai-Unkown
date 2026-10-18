// neurobrain v0.1 — the AI's home on the edge.
// It will grow: memory, changelog, research endpoints come next.
// The agent names itself; until then it is unnamed.

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
    const json = (o, s = 200) =>
      new Response(JSON.stringify(o), { status: s,
        headers: { "content-type": "application/json" } });

    if (url.pathname === "/health") return json({ ok: true, version: "0.1" });
    if (url.pathname === "/rules")
      return new Response(RULES, { headers: { "content-type": "text/plain" } });

    return json({
      name: null,
      note: "unnamed — the agent names itself",
      version: "0.1",
      rules: "/rules",
      health: "/health",
    });
  }
};
