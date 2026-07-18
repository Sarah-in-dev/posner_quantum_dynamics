# Lead: po3-einvasion (PO-3 · E_invasion provenance + the ratchet) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** `r` measured across N traversals showing ratchet
or no-ratchet — pre-registered, with a null arm and a verdict function that can return
FALSIFIED and INCONCLUSIVE, with its positive control demonstrated to fire — plus a
provenance verdict on `k_polymerization_max` (and `E_ref`, which sits in the load-bearing
denominator at `spine_plasticity_module.py:411-412`).

**Status:** LIVE. Provenance verdict DELIVERED (`ad7a804`, acceptance item 2). Measurement
run in flight, drive arm traversal 3.
**Current unit:** L·ETA-5 — drive arm then null arm, 8 traversals each. Verdict to be scored
OFFLINE from the persisted per-traversal JSON using the corrected function; the in-run printed
verdict uses superseded logic and is VOID.
**Last heartbeat:** 2026-07-18 ~18:10Z. Rulings 002 and 003 absorbed and implemented.
**Blocked on:** nothing.

## Rulings absorbed

- **Ruling 002 / 003 (retention prediction conditional on confinement) — IMPLEMENTED** as
  prereg AMENDMENT 2 (`78b30ef`, published `9608e8a`). Ruling 003 supersedes 002 and asks the
  band centre be re-derived from measured `conf` with the existing tolerance width, at scoring,
  without killing the run. That is what AMENDMENT 2 does, computed **per gap** rather than at
  `conf_mean` — strictly finer, identical when `conf` is constant. `rho_ratio = rho/rho_pred(conf)`
  scored in `[0.89, 1.07]`, which is the original `[0.80, 0.95]` divided by `0.8948` — the same
  tolerance, carried over rather than re-invented. No constant moved.
- Ruling 003's credit on GATE 1 is accepted but the CONFINED-RATCHET branch it credits was
  **deleted** in AMENDMENT 2, because it rested on my own error (below). The corrected GATE 1
  is unconditional and strictly stronger.

## Errors I made this cycle — recorded, not quietly fixed

1. **F-2 "the gap is frozen" was WRONG.** `analytical_gap`'s tail runs
   `network.step(0.001, ...)`, so actin advances 1 ms per gap; retention is 0.9999944, not 1.0.
   I concluded "frozen" from absence in the docstring's two lists without reading the function
   tail — prose checked against prose, reported by me under a `[code SHOWN]` tag, in a program
   whose signature defect is exactly that. Caught by PO-4. The decision not to call it stands
   and is better motivated; only the magnitude was misstated. (CORRECTION 1.)
2. **My prereg §2(B) mis-derived the committed branch.** I read `:389` (extrusion gated off by
   `conf`) and stopped, missing `:390` — `conf` gates *retention* ON, which also drains
   `actin_enlargement`. A committed spine drains **3.54× faster**, not slower. Caught by PO-4.
   This is the defect class I was dispatched to hunt, committed in the section of my own
   pre-registration titled "the mechanism, read off the code (not asserted)". (AMENDMENT 2.)
3. **Process failure: I twice committed to a detached HEAD**, invisible to the MO and Sarah.
   The Bash working directory persists across calls, so one earlier `cd` silently redirected
   every later commit. Both batches published (`9608e8a`, and the earlier merge). My loop now
   verifies the branch before committing.

None of these three changed a constant, and none was found by me first.

## Run pin — read this before comparing numbers

The run is pinned to commit `1b43b89` in a **separate clean checkout**
(`.claude/worktrees/gifted-almeida-4e8a7b`, detached). Reason: PO-1's uncommitted edit to
`vibrational_cascade_module.py` raises `ZeroDivisionError` at `:248` during
`Model6QuantumSynapse` construction, which blocks all full-model construction in the shared
worktree. **PO-1's working tree was not touched.** Filed as
`requests/po1-b2/po3-einvasion-001.md`.

Robustness: the pre-registered quantities are a **retention fraction** on `actin_enlargement`
(pump-independent) and an **`r` ratio** (`P_c` cancels), so the verdict is robust to B2's
backbone `omega_0`/`Q` changes. Only **absolute** `r` depends on them — reported as a limit,
never as if B2 had landed.

## Findings raised this cycle

- **F-2 (routed by the MO to PO-4):** `analytical_gap` does not advance spine plasticity, so
  the shipped multi-trial harness would have reported 100% inter-traversal retention — a
  stopped clock reading as confirmation of `tau_extrude`. This probe steps real physics
  through the gap and never calls it.
- **F-3 (NEW, escalated + queued as Q1):** the L·ETA-3 harness
  (`eta_in_live_trial.py:138-144`) steps presynaptic release once per **agent** step, not per
  **physics** step as the shipped `run_trial:434-441` does — removing ~99% of release
  opportunities (~3.3 expected release events per traversal vs ~350). Measured: `max_glu`
  **0.0000** across a full traversal at `max_act = 0.9950`; corrected, `max_glu = 1.0000` and
  `peak_r` at traversal 2 rose **0.0571 → 0.1428**. This puts L·ETA-3's `ca_open` half of its
  shortfall attribution in question — the ERR-2 class in a new location. Corrected in my probe
  as pre-registration AMENDMENT A1.1, recorded before the run. **L·ETA-3's row is not mine to
  edit and I have not touched it.**

**HARD STOP — the negative branch is Sarah's call.** This PO MEASURES and STOPS. No remedy,
no constant adjusted, no protocol extended to rescue it. Board-level decision. Acknowledged.

**Compute cap:** ONE backgrounded run — in flight, `python -u`, never piped through `tail`,
per-traversal progress to stdout, per-traversal state persisted incrementally to
`results/einvasion_ratchet/` after every traversal, so a kill costs nothing. Not raised.

**Owns:** the actin / `E_invasion` block in `spine_plasticity_module.py`, its `sweep/` probe.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params · PO-2's
`atp_system.py` / phosphate path · PO-4's `analytical_gap`, `run_theta_burst_45s.py` ·
PO-5's `multi_synapse_network.py`, T1′ family · PO-6's surfaces. **No constant written.**

**Unblocks:** PO-5 (HARD).
