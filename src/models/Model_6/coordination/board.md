# Model 6 coordination board — MO-owned (read by all POs; written only by the MO)

**Updated:** 2026-07-18. **Worktree:** `/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6`
(branch `claude/nervous-hertz-7ccff6`) — NOT the master tree, which is ~20 commits behind.
All POs work in that worktree. Python: `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`.

**The program board is `src/models/Model_6/docs/MO_MODEL6.md`** — six POs with coordinates,
acceptance bars, dependency edges, surface ownership, the kickoff convention, the LOCKED
list. This file is the live registry; that file is the durable plan. Read both.

## Active POs
| PO | Objective | Status | Owner files | Last update |
|---|---|---|---|---|
| `model6-mo` | the MO itself (meta/coordinator) | BOOTING | `board.md`, `MO_MODEL6.md` | 2026-07-18 |
| `po1-b2` | B2 — retire the per-synapse pump site | NOT SPAWNED | `vibrational_cascade_module.py`, backbone params | — |
| `po3-einvasion` | E_invasion provenance + the ratchet test | NOT SPAWNED | `spine_plasticity_module.py` | — |

_(The MO adds a row here + the PO's `leads/`/`queue/` files when it spawns one.)_

## Surface-ownership map (collision spine — never edit another owner's files; drop a `requests/` file)
| Surface | Owner |
|---|---|
| `vibrational_cascade_module.py`, backbone params (`model6_parameters.py:759-805`) | **PO-1** |
| `atp_system.py`, the phosphate path in `model6_core.py` | **PO-2** |
| `spine_plasticity_module.py` actin / E_invasion block | **PO-3** |
| `analytical_gap` in the drivers, `run_theta_burst_45s.py` | **PO-4** |
| `multi_synapse_network.py` partition path, the T1' probe family | **PO-5** |
| the orphan modules, `quantum_dimensions.py`, `sweep_runner.py` | **PO-6** |
| the research logs | ALL — each PO writes its OWN entries; nobody rewrites another's |

**Shared-file hazard:** `model6_core.py` is touched by PO-1, PO-2 and PO-4. Only ONE may hold
uncommitted edits at a time. Sequence, or require commit-at-boundary.

## Dependency edges (sequence, route through the MO)
- **PO-2 → PO-6 (HARD).** The sweep tests self-organization; if the phosphate loop is not
  mass-conserving the SOC engine does not exist and the sweep measures nothing.
- **PO-3 → PO-5 (HARD).** At eta=0 there is no partition to be selective (L·ETA-3: zero
  cross-synapse edges all trial).
- **PO-1 ∥ everything** — file-disjoint.
- **PO-4 ∥ PO-1, PO-2** — but PO-4's `K_CLASSICAL` decision touches chemistry rates PO-2
  reasons about. **`K_CLASSICAL` routes through the MO, never through a PO** (50× spread
  across three sites: 0.05 / 0.005 / 0.001). Parked this round.

## Decided, do not re-open
- Dispatch **PO-1 and PO-3 first, in parallel**. PO-2 follows at PO-1's commit boundary.
- PO-3 compute cap: one backgrounded run, per-traversal progress output, `r`-vs-traversal
  persisted incrementally. Raise once with a stated reason; beyond that, escalate.
- **PO-3's negative branch is Sarah's call.** The PO measures and STOPS.

## Human-gated (escalate; do not attempt)
Any physics call · anything on the `MO_MODEL6.md` §7 LOCKED list · anything that would
re-validate a closed result (the T1' probe family is deliberately NMDAR-shut) · a PO moving
a constant to reach an outcome (bounce first, then report) · any acceptance the MO cannot
verify directly.
