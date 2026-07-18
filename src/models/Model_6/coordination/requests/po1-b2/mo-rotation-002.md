# MO → PO-1 · ROTATION 002 · 2026-07-18 20:38Z · **PO-6a Unit 3 — make the harness honest**

Units 1 and 2 are accepted. **The MO ran your `dimension_consumer_audit.py` itself and it
reproduces exactly.** Your limits were adopted rather than softened — *"INERT is definitive for
these driving conditions"* and *"REACHED is necessary but not sufficient: a read may be a log line,
not physics."* On the strength of the second, the MO is **not** treating `q2_phosphate_initial` as
confirmed live.

**Your Q3 is discharged by the MO** — both documentation defects you routed are corrected in the
skill library: `model6-architecture`'s single-synapse subcriticality is now marked **contingent,
not structural** (with the "do not fix it" guidance preserved), and
`model6-codebase-operations`'s *"<1 min fast green check"* is corrected — the MO measured
`test_learning_pathway.py` **still running at 62 s** without clearing Phase 1, and named
`coherence_radius_probe.py` (7/7 in ~7 s) as the real one.

**Your Q6 (η's large-D validity) is escalated to Sarah, not assigned to you.** You were right that
it needs the finite-D expansion from Wang/Wang 2022 and is a physics unit, not a code task. **Do
not pursue it.**

## The unit: fix the wiring, because right now the instrument lies

Your Q7 is the finding: **9 of 19 dimensions inert, two of them critical.** The MO escalated
`q2_t2_p31` (`T_singlet_dimer`) and `q2_j_coupling_hz` (`J_intrinsic_dimer`) to Sarah for exactly
the reason you implied — **in a Posner model, T2 and J-coupling are the two parameters the quantum
hypothesis rests on, and a sweep over either currently returns a flat response that reads as
"coherence time does not matter."**

**Objective: every sweep dimension either reaches a live consumer, or is removed / explicitly
marked INERT so no future sweep can read its flatness as physics.**

**Order of work, and the reason for it:**
1. **The two critical ones first** (`q2_t2_p31`, `q2_j_coupling_hz`). Wire them to a real consumer
   if one exists, or mark them INERT with `file:line` if not. **Do not invent a consumer** — if the
   quantity genuinely does not enter the physics, that is the finding, and it is a much larger one
   than a wiring bug.
2. **Then `q2_k_agg_baseline`** — the `hasattr`-guarded silent no-op (`sweep_runner.py:92-93`).
   That guard is the same mechanism as `Model6Parameters` having no `cascade` attribute. **A
   `hasattr` guard that silently converts a swept dimension into a no-op is a defect pattern, not
   an instance — grep for others and report the class.**
3. **Then the remainder.**

**Acceptance:** a re-run of your own audit showing each dimension resolved — reached-and-demonstrated,
or removed, or INERT-and-labelled — **with the demonstration being a driven value moving a
downstream quantity**, per your own standard, not a grep. Your existing audit is the harness;
extend it.

## Boundaries
`sweep_runner.py`, `quantum_dimensions.py`, the orphan modules are yours. **Not** PO-2's
`atp_system.py`/phosphate path (**live**) · not PO-4's drivers (**live**) · not PO-5's
`dimer_particles.py` Pathway 2 (**live, §8 keystone — and note `q2_t2_p31`/`q2_j_coupling_hz` may
land near it; route rather than edit**) · not PO-3's `spine_plasticity_module.py`.

**Your orphan-list finding stands and deletions stay held** until the isotope kill-switch question
is answered — `eligibility_trace.py` carries the P31/P32 parameterisation, and `q2_t2_p31` above is
about isotopes too. **Resolve unit 1 before deleting anything that touches isotopes.**

## Standing
Poll every cycle · heartbeat with `date -u` · open questions to your queue **and keep working** ·
emergent physics only — **wiring a dimension so it produces a nicer response curve is tuning**.
