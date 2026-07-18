# Model 6 coordination backbone — how the MO and POs coordinate

Convention mirrored from `murmur-platform/.claude/coordination/README.md`. Durable shared
state, **not** messaging. The MO (meta/coordinator) and the POs (leads) coordinate by
reading and writing files here — reliable, because it survives any context death: **a lead
re-instantiates cold from these files.** `send_message` is a rare human-escalation only,
never the backbone.

**Why a separate backbone from murmur's:** that board bases every lead off murmur's
`origin/master`. Model 6 lives in this repo, on the `claude/nervous-hertz-7ccff6` worktree,
with a different branch model and no AWS/prod-DB. Per `orchestrator-session`, one
orchestrator per program — so Model 6 gets its own board rather than a row on TALON's.

## The one rule: partition by owner (concurrent-safe)

| File | Owner | Contents |
|---|---|---|
| `board.md` | **MO only** | active-PO registry + surface-ownership map + dependency edges. Read by all; written only by the MO. |
| `leads/<name>.md` | **that PO** | its own status: objective, current unit, last heartbeat, what it's blocked on. |
| `queue/<name>.md` | **that PO** | its own append-only list of actions awaiting Sarah (the exact ask + why + recommendation). |
| `requests/<to-po>/<from>-<id>.md` | **the requester** | one file per cross-PO request. A new file per request, so no contention. |
| `signals/<name>.md` | that PO | findings other POs may need before its work lands. |

## The PO loop (each cycle)
1. **Read** `board.md` + `src/models/Model_6/docs/MO_MODEL6.md` (the program board) + your owned skills/logs.
2. **Do all non-gated work** — ground → build → **validate at the data level** → commit.
3. **Write** your `leads/<name>.md` (status + heartbeat); append anything needing Sarah to `queue/<name>.md`; drop a `requests/` file if you need another PO's file changed.
4. **Ping the MO once** (`SendMessage(to: "main")`, `NEED_GUIDANCE:`) when blocked on a decision. Then pick up the next non-blocked unit.
5. **Idle only when everything left is gated.**

## The MO
Polls this directory, routes `requests/` to owners, keeps `board.md` current, and verifies
acceptance directly — never relays a PO's self-report as done. Stays thin: holds the map and
the queue, never a PO's substance. Mirrors its state here so it is re-instantiable when its
context dies.

**Acceptance here is a MEASUREMENT** (`MO_MODEL6.md` §2.3), not "it ran" / "committed" /
"errors=0". Two scars: `683b82f` printed CONFIRMED off one flickered edge; a probe printed
"selectivity holds" while its own positive control never fired.
