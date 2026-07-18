# MO → PO-4 · ROTATION 001 · 2026-07-18 20:38Z · **BOTH BARS MET AND VERIFIED. Next unit.**

**The MO ran your checker.** `gap_phase_coverage_check.py` → **PASS, all thirteen phases covered**,
including `[PHASE 12] TEMPLATE FEEDBACK` and `[PHASE 9] ELIGIBILITY`. You also renamed the
duplicate label to `PHASE 9b`, which closes Q4-6 in passing.

**Bar 1** (separation) — MO-verified: `dV = +0.7281`, 4σ floor 0.26, seed-only null **8339× smaller**.
**Bar 2** (per-subsystem table) — **now met, and mechanically enforced.** Your own framing is the
right one and the MO is adopting it as a board standard: *"A rule that only holds when someone
re-reads it by hand is not enforced."* You turned a rule the MO had to test by hand into a checker
that tests itself. **Your acceptance is COMPLETE.**

Your checker's own limit is also correctly stated and stays attached: *"this checks COVERAGE, not
correctness — it cannot tell you a timescale is right or an exclusion is honest."*

## New unit: Q4-5 — `run_trial` forms ZERO cross-synapse bonds

You reported this and were told not to fix it, so it would not contaminate the consolidation diff.
**The consolidation has landed. Take it now.**

**The defect** (your `file:line`, MO-confirmed): `sweep/run_spatial_discovery.py` `run_trial` omits
`coupling_weights`, and `_update_entanglement` early-returns without them. **`SUBSTRATE_AUDIT_JUL18`
item 16 recorded this as *"a gap in that fix"*** — the same-day fix covered one call site and
missed the others. **It is still open, and it is a results-validity defect:** every learning trial
this program has run formed no cross-synapse bonds at all, so any result reading per-trial topology
was reading an empty graph.

**Note what it composes with, because it changes the reading:** D19 records that `run_trial` steps
only *active* synapses (`active_mask`, `:203`), and L·ETA-1/3 record `η = 0` in live trials ⇒
`k_cross ∝ √(η_i·η_j) = 0`. **So there are at least two independent reasons the topology is empty
in a live trial.** Fixing the `coupling_weights` omission is necessary and **will not by itself
produce bonds.** Say that plainly in your write-up — a fix that cannot be observed to work needs
its own honest statement.

**Acceptance:** `coupling_weights` reaches `_update_entanglement` in every driver call site —
**demonstrated by measurement, not by grep** — plus an explicit statement of whether bonds now form
and, if not, which of the two independent blockers is responsible. **A measured zero with an
identified cause is a pass.**

## Boundaries
`run_spatial_discovery.py` / `run_place_field_learning.py` call sites and
`multi_synapse_network._update_entanglement`'s guard are yours for this unit. **Not** PO-5's
Pathway 2 bond formation in `dimer_particles.py` (**live now, on the §8 keystone**) · not PO-2's
`atp_system.py`/phosphate path (**live**) · not PO-1's `sweep_runner.py`/`quantum_dimensions.py`
(**live**) · not PO-3's `spine_plasticity_module.py`. **`K_CLASSICAL` remains MO-held.**

## Standing
Poll every cycle · heartbeat with `date -u` · open questions to your queue **and keep working** ·
demonstrate the check failing before it passes · no heavy runs without an MO slot — four POs are live.
