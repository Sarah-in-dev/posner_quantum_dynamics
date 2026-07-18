# Lead: po4-gap (PO-4 · the analytical gap, biologically grounded) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** every subsystem either advances during silence
with a cited timescale or is excluded with a stated reason — nothing in neither column; and a
measurement shows **committed vs uncommitted spine volume SEPARATING across an honest gap**
(isolated-module numbers say 1.291 vs 2.389 at +300 s; the full model has never been allowed
to show it). Demonstrated failing on the current frozen-clock code first.

**Status:** DISPATCHED by the MO 2026-07-18 17:52Z — chip `task_daa62deb` pending.
**Current unit:** — (first act is the `### GROUNDING BRIEF`)
**Last heartbeat:** —
**Blocked on:** —

**Why this PO exists now:** PO-3 found, and the MO independently confirmed, that
`analytical_gap` does not advance spine plasticity at all — actin appears in neither its
computed nor its NOT-computed list. Frozen `E_invasion` across every gap ⇒ a multi-trial
harness reports **100% retention instead of ~89%**, which reads as a strong memory ratchet
rather than a stopped clock.

**MO-added, beyond PO-3's report:** `analytical_gap` is **DUPLICATED** —
`sweep/run_spatial_discovery.py:55` and `src/models/Model_6/sweep/run_theta_burst_45s.py:44`.
Fixing one leaves the other, the same partial-fix shape as audit item 16 on this same pair.
Diff both first.

**Owns:** `analytical_gap` in BOTH drivers; `run_theta_burst_45s.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params · PO-2's
`atp_system.py` / phosphate path · PO-3's `spine_plasticity_module.py` (**call it, never edit
it** — drop a `requests/po3-einvasion/` file) · PO-5's `multi_synapse_network.py`, T1′ family ·
PO-6's surfaces.

**NOT this PO's to decide:** `K_CLASSICAL` (50× spread: 0.05 / 0.005 / 0.001) sits inside the
function being fixed. MO-owned, parked. Report, do not touch.

**Compute:** NO heavy slot. PO-3 holds the single heavy backgrounded slot. Do the two-copy
diff and the subsystem ruling first; request the slot via `queue/po4-gap.md`.
