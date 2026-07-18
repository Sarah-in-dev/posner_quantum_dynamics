# Lead: po1-b2 (PO-1 · B2 — retire the per-synapse pump site) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** the per-synapse site calls
`bose_einstein_occupation` (`model6_parameters.py:46`) on the `n_ex = n̄_s` form; no
hand-rolled `hbar` anywhere in `vibrational_cascade_module.py`; `kT_ref` and `r_at_E_ref`
grep-provably gone from the live path; **a measurement shows the per-synapse and backbone
pumps agree on the same mode**; T1′ static probe still 7/7; a superseding entry retires
`DISC-1` in the `RESEARCH_LOG_CALCIUM_DIMER.md` DECISION RECORD.

**Status:** DISPATCHED by the MO 2026-07-18 17:35Z — chip `task_274226b1` pending.
**Current unit:** — (first act is the `### GROUNDING BRIEF`)
**Last heartbeat:** —
**Blocked on:** —

**Owns:** `vibrational_cascade_module.py`, backbone params `model6_parameters.py:759-805`.
**Must not touch:** PO-2's `atp_system.py` / phosphate path · PO-3's `spine_plasticity_module.py`
· PO-4's `analytical_gap`, `run_theta_burst_45s.py` · PO-5's `multi_synapse_network.py` and the
T1′ probe family · PO-6's orphan modules, `quantum_dimensions.py`, `sweep_runner.py`.

**Shared-file hazard:** `model6_core.py` — PO-1/PO-2/PO-4. One uncommitted holder at a time.
PO-2 is sequenced to start at this PO's commit boundary.

**Notes:** items 1+2+4 of the audit's ranked list collapse here. `kT_ref` has **no derivation
to do** — `r_c` is the classical critical pump, →0 in large-D, an artificial reference scale.
Both pumps are the SAME 8 MHz collective mode (two lattice segments, not a fork). `φ = ω₀/Q
≲ 0.8 MHz` from the pin; drop the Zhang 2019 citation; keep χ as slope-above-threshold only.
