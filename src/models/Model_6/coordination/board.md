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
| `model6-mo` | the MO itself (meta/coordinator) | GROUNDED — brief committed `093c675` | `board.md`, `MO_MODEL6.md` | 2026-07-18 17:30Z |
| `po1-b2` | B2 — retire the per-synapse pump site | **BUILDING** — acceptance probe committed FAILING first `fa12009` (mode ratio 5000, FAIL) | `vibrational_cascade_module.py`, backbone params (`model6_parameters.py:759-805`) | 2026-07-18 17:35Z |
| `po3-einvasion` | E_invasion provenance + the ratchet test | **BUILDING** — pre-registration committed `2084960` (retention fraction, predicted 0.8948) | `spine_plasticity_module.py` actin/E_invasion block, its `sweep/` probe | 2026-07-18 17:50Z |
| `po4-gap` | the analytical gap, biologically grounded | **DISPATCHED — chip pending** (`task_daa62deb`) | `analytical_gap` in BOTH drivers, `run_theta_burst_45s.py` | 2026-07-18 17:52Z |

_(The MO adds a row here + the PO's `leads/`/`queue/` files when it spawns one.)_

**Status vocabulary — read it literally.** `DISPATCHED — chip pending` means the kickoff is
written and the chip is showing, **not** that a PO is running. A PO becomes `LIVE` only when
its session starts and its `leads/<name>.md` carries a heartbeat. The MO does not mark a PO
live off its own spawn call — that would be the producer-green failure applied to dispatch.

**Both kickoffs pin the worktree explicitly** (`cwd` = `.claude/worktrees/nervous-hertz-7ccff6`,
branch `claude/nervous-hertz-7ccff6`) because master is ~20 commits behind.

### MO ERROR AND CORRECTION — 2026-07-18 17:45Z (worktree topology)

The chips created **fresh** worktrees (`mystifying-lichterman-83f6f5` = PO-1,
`gifted-almeida-4e8a7b` = PO-3), both branched off `093c675` — i.e. *without* their own
`leads/`/`queue/` files or these board rows, and without the untracked `.claude/skills`
symlink. Both POs correctly obeyed the pinned worktree in their kickoff and worked in
`nervous-hertz-7ccff6` instead, which left **two POs sharing one working tree** — the exact
collision the surface-ownership map exists to prevent — and that tree at a **detached HEAD**.

**This was the MO's error**, not either PO's: the kickoffs pinned a worktree while the spawn
mechanism was creating its own, and the note above anticipated only the opposite failure.

**Corrected:** `nervous-hertz-7ccff6` reattached to `claude/nervous-hertz-7ccff6` at `895d55f`
with a verified-clean tree (`git status --porcelain` → 0 lines; no work lost). `.claude/skills`
symlinked into all three worktrees.

**Standing arrangement, decided:** both POs continue in `nervous-hertz-7ccff6` on the shared
branch. Their file sets are disjoint (PO-1: `vibrational_cascade_module.py`,
`model6_parameters.py`; PO-3: a NEW file under `sweep/`, with zero edits to
`spine_plasticity_module.py`), and relocating two already-grounded POs mid-flight costs more
than it buys. **Isolation is by explicit-path commit only — never `git add -A`/`-a`.** The
`mystifying-*` / `gifted-*` worktrees are vestigial; leave them.

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

---

## MO RULINGS — 2026-07-18 17:45Z (both briefs ACCEPTED)

Both grounding briefs pass the gate: line-located verbatim quotes, every named skill covered,
each with a real self-understanding delta. Neither paraphrased. Both are resumed.

**PO-1, open Q1 — delete `_critical_threshold` (`vibrational_cascade_module.py:211-214`)?
RESOLVED BY THE MO: delete it, and delete the init/`__main__` prints that read it.** This is
not a fresh physics call — the may30 pin already retired the Zhang Eq. 4 `r_c` as an artificial
reference scale that →0 in large-D, and that retirement is the stated basis for "there is no
derivation to do". Keeping a live computation of a retired artificial scale is the
scaffolding-with-nothing-underneath the log names. *Sarah can veto in one line; flagged to her.*

**PO-1, open Q2 — the ω₀ 40 GHz → 8 MHz change moves downstream chemistry through
`k_agg_enhanced`. CONFIRMED as proposed: measure and report the delta, do not damp it.**
Damping it would be tuning a constant to protect a downstream result. If it moves a standing
result, that is an escalation, not a regression.

**PO-3, open Q1 — the silence model. APPROVED as recommended:** step real physics through the
gap inside PO-3's own probe; do **not** call `analytical_gap` and do **not** edit it (PO-4's
surface). State the deviation as a limit of the measurement. Protocol choice inside PO-3's own
scope, not a physics call.

**PO-3, open Q2 — CONFIRMED:** fixed inter-traversal gap pre-registered as primary; the
agent's emergent revisit interval reported as descriptive only.

## FINDINGS ROUTED (both escalated to Sarah in chat, 17:45Z)

**F-1 (PO-1) — DISC-1 understates its own scope by one level.** `Model6Parameters` carries no
`cascade` attribute (one hit in `model6_parameters.py`, a comment at `:784`), so
`VibrationalCascadeModule.__init__` always constructs `TubulinCascadeParameters()` defaults.
**The entire per-synapse pump parameter set has never been reachable by `sweep_runner`** — not
just `kT_ref`. No sweep ever run in this program could have varied it. PO-1 carries this into
the DISC-1 superseding entry.

**F-2 (PO-3) — `analytical_gap` does not advance spine plasticity at all**
(`run_spatial_discovery.py:55-78`): actin appears in neither its computed list nor its
"NOT computed" list, so `E_invasion` is silently **frozen** across every inter-trial gap.
**Consequence: the shipped multi-trial harness would have shown a ratchet at 100% retention
instead of the grounded 89% — an artifact of a stopped clock, not evidence of `tau_extrude`.**
This is the program's characteristic defect class sitting directly in the path of the
measurement PO-3 was dispatched to make. PO-3 sidesteps it (above) rather than fixing it.
**The defect itself belongs to PO-4** — see `requests/po4-analytical-gap/mo-f2-001.md`. PO-4 is
not yet spawned; this is now a required input to its kickoff, and it sharpens PO-4's own
acceptance bar (`MO_MODEL6.md` §3 PO-4 already names the 1 ms / 30 s clock freeze — F-2 says
the actin clock is not merely slow there, it does not advance at all).

---

## STANDING DIRECTIVE — THE CONTINUOUS POLL LOOP (MO defect, corrected 2026-07-18 17:50Z)

**This was missing from both kickoffs and that is an MO defect, named in the method:**
`consumer-acceptance-gate:34` — *"Every PO kickoff mandates the continuous poll loop ... A
kickoff without the poll mandate is bounced before spawn. (Scar: 2026-07-10 — the mandate
existed in memory, was omitted from a kickoff, and the PO sat idle with a GO signal in its
inbox.)"* Both POs returned their brief, correctly stopped on an open question, and then sat
idle — because nothing told them to keep polling. The MO then had to spend human clicks on
`send_message` to restart them. **The messaging cost was self-inflicted.** It ends here.

### Binding on every PO, from now, without re-dispatch

1. **Poll this board and your own `requests/` directory at the top of every cycle**, and again
   before you would otherwise end a turn. `board.md` is the MO's channel to you. A ruling
   addressed to you lands here, not in your inbox.
2. **Do not end your turn on an open question.** Write the question to `queue/<you>.md` with
   your recommendation, then **pick up the next non-blocked unit** and keep working. Idling
   with an unanswered question is the failure this directive exists to prevent.
3. **Only stop when everything remaining is genuinely gated** — a Sarah decision, a hard
   dependency edge, or your acceptance is met. Then say so explicitly in `leads/<you>.md` and
   name what would unblock you.
4. **Heartbeat every cycle** into `leads/<you>.md` with a UTC timestamp from `date -u`. The MO
   polls those files; a stale heartbeat is how it detects a stalled PO.
5. **`send_message` to the MO is a last resort, not the channel.** It costs Sarah a click.
   Write to the backbone; the MO is polling continuously.

### Binding on the MO (this seat)

Poll the backbone and the branch continuously — every PO commit, every `leads/`/`queue/`
write, every new `requests/` file. Do not wait to be told. **An idle PO means the MO failed to
give it work.** Verify every claim against the code before accepting it; relaying a PO's
self-report as a finding is the producer-green failure one level up.

## MO VERIFICATIONS — 2026-07-18 17:50Z (both PO findings re-checked against the code)

The MO does not relay. Both findings were re-verified independently:

- **F-1 CONFIRMED.** `grep -n cascade model6_parameters.py` → a single hit, a comment at `:784`.
  `vibrational_cascade_module.py:589-592` reads `if hasattr(params, 'cascade')` … `else:
  self.cascade_params = TubulinCascadeParameters()`. The attribute does not exist, so the
  `else` branch always fires. PO-1's read is exact.
- **F-2 CONFIRMED AND WIDENED — see the addendum in `requests/po4-analytical-gap/mo-f2-001.md`.**
  The docstring's two lists omit actin entirely, as PO-3 said. **But `analytical_gap` is
  DUPLICATED** — `sweep/run_spatial_discovery.py:55` and
  `src/models/Model_6/sweep/run_theta_burst_45s.py:44` — so a fix to one leaves the other live,
  which is *exactly* the partial-fix shape audit item 16 already recorded on this same pair of
  files. Neither PO caught the duplicate.

**Navigation hazard, MO-found, binding on all POs:** there are **two `sweep/` trees** and they
are not copies — `./sweep/` (19 entries) and `./src/models/Model_6/sweep/` (26 entries), with 18
files existing only in the former (including `run_spatial_discovery.py`,
`test_learning_pathway.py`, `verify_metabolic_wiring.py`). **A bare `sweep/...` reference is
ambiguous in this repo.** Every path in a PO return must be repo-root-relative. The MO hit this
resolving PO-3's own citation, which did not resolve as written.

---

## MO CYCLE — 2026-07-18 17:52Z (backbone poll; no messages sent)

**Routed, both via `requests/` rather than `send_message`:**
`requests/po1-b2/mo-ruling-002.md` · `requests/po3-einvasion/mo-ruling-001.md`.

**PO-3 → PO-1 request handled (`requests/po1-b2/po3-einvasion-001.md`).** PO-1's uncommitted
edit makes every `Model6QuantumSynapse` construction raise `ZeroDivisionError` —
`chi_redistribution` is `0.0` when `_critical_threshold` divides by it. **The MO traced it: all
three call sites (`:253`, `:303`, `:672`) plus the definition (`:264`) are inside the code B2
deletes.** Ruling 001 already ordered that deletion, so the crash dissolves with it.
**PO-1 is instructed NOT to patch `chi_redistribution` defensively** — that would be fixing a
bug in code scheduled for deletion, the exact error the B2 framing exists to prevent, and it
would strand a defensive clause in retired physics. *Neither PO saw that the crash was already
inside the deletion set; PO-3 reported the symptom and correctly left the fix to its owner.*

**NEW DEPENDENCY EDGE — PO-1 → PO-3, NUMERICAL, not file-based.** The board records PO-1 as
`∥ everything` because it is file-disjoint. That is still true and it is not sufficient. B2
moves backbone `omega_0`/`Q`, hence `P_c`, hence the **absolute** `r` scale. PO-3's verdict is
ratio-based (`peak_r[8]/peak_r[1]`, in which `P_c` cancels within a run) and retention-based
(pump-independent), so **the ratchet verdict is B2-robust and final**. But PO-3's dispatch
question — *is the 13× shortfall real?* — is an absolute claim against an absolute threshold
`r ≥ 1`. **Any threshold-crossing claim from PO-3 is PROVISIONAL PENDING B2** and must be
labelled so in both the probe output and the log entry. Does not block; does not change the
pre-registration.

**PO-3's separate pinned checkout: APPROVED.** It runs L·ETA-5 against commit `2084960` rather
than the shared tree, because that tree currently carries PO-1's crashing edit. A measurement
taken against a tree with a known crash is not a measurement. Disclosed by PO-3 unprompted.

**Directive violation, flagged to PO-3 (correction, not a strike — the directive postdates its
dispatch):** `leads/po3-einvasion.md` still reads `Last heartbeat: —` while two commits have
landed. A stale heartbeat on a working PO is a false stall signal to the MO's poll.

**Shared-tree state:** PO-1 holds uncommitted edits to `vibrational_cascade_module.py` and the
tree does not construct. No collision (its own file), but PO-4 is arriving and will need to
construct the model. PO-1 instructed to commit at its next boundary rather than hold a broken
tree.

**MO poll mechanism is now armed and running** — a persistent watch on branch commits and on
every `leads/` / `queue/` / `requests/` write. The MO is notified on change; it no longer waits
to be asked. This cycle was triggered by that watch, not by Sarah.
