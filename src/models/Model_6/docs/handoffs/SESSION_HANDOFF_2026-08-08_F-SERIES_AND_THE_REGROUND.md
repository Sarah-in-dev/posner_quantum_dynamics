# Session Handoff — 2026-08-08 · the F-series, the reground, and the real frontier

> **⚠ STATUS: prior-session handoff (Claude-authored, first-person), not canonical direction.** For the current
> read-order and frontier see **`docs/README.md`**. The F-series findings it summarizes are real and live in
> `RESEARCH_LOG_CALCIUM_DIMER.md`; the "covariance-matched design" it dispatches (Part 5) is SUPERSEDED / off-program
> (see that prereg's header). Verify against the logs + skills before acting on anything here.

**One line:** Built and grounded the F-series (F2 measurement, F3 reward-gated consolidation) with cited
literature; then Sarah corrected a fundamental misframing — **coherence is PROVEN, not the problem** — and the
session re-grounded onto the *real* open problem (the entanglement-topology sub-program: is the partition a
genuine *directed* computation, or "quantum constrains, classical computes"). First engagement with that
problem — the **covariance-route readout (COV-1)** — is a clean NEGATIVE that points at the next experiment:
a **covariance-matched design**. That is the frontier this handoff dispatches a PO to build.

## Read-order (ground first)
1. `session-discipline`, `agent-grounding-protocol` (in full).
2. **`RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`** — the "what the computation IS" sub-program (PO5–PO11 + COV-1).
   THIS is where the real problem lives. Read the DECISION RECORD (newest first): `COV-1`, `PO10-2`, `PO9-2`.
3. `PO10_ADVISOR_UPDATE_UNIT_C_2026-07-22.md` — the weight-level keystone + the sign problem + Addendum 2 (the
   covariance-route open question this thread tested).
4. `PO8_EXTERNAL_LIT_REVIEW_2026-07-20.md` — the timescale grounding; "τ≈2 s is the CLASSICAL baseline being
   BEATEN, not the target."  `MO_MODEL6.md` — the board/method (2026-07-18 snapshot).
5. `RESEARCH_LOG_CALCIUM_DIMER.md` rows F1, F2-a..e, F3-a..d — the F-series (the chemistry/reward half).
6. Grounding docs (cited, key numbers verified at source): `RESEARCH_DOPAMINE_REWARD_SIGNAL_2026-08-06`,
   `RESEARCH_TCA_MECHANISM_2026-08-08`, `RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08`.

## Part 1 — What was built this thread (the F-series; all committed local on master, NOT pushed)
- **F2 (measurement) — `posner_binding.py`**: DERIVED spin-selective Posner binding-melt rate (Fisher 2015;
  QDS Fisher & Radzihovsky 2018), `__main__` 4/4. Wired as the collapse trigger (Design B), reward decoupled.
  Full control ladder: **F2-c** — grounding the trigger made it fire early/local (~88 ms), which CORRECTED the
  "computation = network partition" claim (propagated to canonical skills `quantum-system-canonical §4.4/§5` +
  `talon-architectural-north-star #2`); **F2-e** — commitment collapsed ~7× and reward is inert (calcium-dominated).
- **F3 (reward-gated consolidation) — `reward_gating.py` + `model6_core` commitment gate (flag
  `_reward_gated_consolidation`, mode quantum|classical)**: eligibility trace = coherent P_S tag; the readout is
  the **dopamine-GATED binding-melt** (corrected — see Part 3), coherence-lifetime window. `__main__` 7/7.
  **F3-b** delayed-credit (n=6): quantum credits to 30 s, classical dies at 2 s, gap p=0.001. **F3-c** isotope
  (n=6): ⁶Li≈undoped credits all delays, ⁷Li short-only, contrast p=0.000. Probes: `sweep/f3_delayed_credit_probe.py`,
  `f3_isotope_credit_probe.py` (single-synapse; need `em_coupling_enabled=True` + `_network_controlled=True` +
  inject explicit tonic DA=20e-9 when no transient — the live DA field drifts >tonic and self-commits otherwise).
- **3 grounding passes** (cited, verified at source): Yagishita 0.3–2 s dopamine window; the eligibility trace's
  molecular identity is the field's #1 UNKNOWN and every measured trace is too short (Shindou ~2 s, self-declared
  incomplete); the dopamine "decoherence readout" framing is UNSUPPORTED (Part 3).

## Part 2 — THE REGROUND (Sarah's correction — the load-bearing part of this handoff)
**I mis-framed coherence as an open problem/falsifier. It is not.** `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md:304`:
the correlated graph dies at ~20–25 s from **population loss, not decoherence** — mean P_S is still 0.63 at 120 s,
intra F=1.000 throughout. Coherence is settled. **The real problem is the COMPUTATION** (the entanglement-topology
sub-program, PO5–PO11): *does the partition carry INPUT-DEPENDENT structure that becomes a USABLE (directed) weight
update a classical reservoir cannot reproduce* — or is it "scalar as computation"? Repeated findings: "topology is
a function of density alone" (PO5-11), input-dependence d≈0.02 flat (PO5-13/PO7-1); the weight-level keystone
(PO10-2) is DETECTED not robust, decode ~0.75 = readout NOISE, **sign UNRESOLVED**, verdict **"quantum CONSTRAINS,
classical COMPUTES."**  My F-series did not engage this; my F3 "credit" is a magnitude readout = exactly the
abundance-leak kind PO10 flagged; my initial synthesis (`SYNTHESIS_TEMPORAL_CREDIT_F1_F2_F3_2026-08-08.md`)
OVERCLAIMED (treated the substrate coherence as the falsifier) and should be read through this correction.

## Part 3 — the readout-physics correction (F3-d)
"Dopamine triggers decoherence of the ³¹P tag" is physically UNSUPPORTED (decoherence is passive leakage, not a
signal; no mechanism for a neuromodulator collapsing a nuclear spin). The GROUNDED readout: dopamine never touches
the spin — it **GATES/TIMES Fisher's Ca/pH-dependent spin-selective binding-melt** (D1/D2→cAMP/PKA→local Ca/pH).
This UNIFIES F2 (binding-melt = readout) + F3 (dopamine = timing). Corrected across code+docs; functional results
(F3-b/F3-c) unchanged. Substrate premise stays flagged (Agarwal: the DIMER holds ~100 s coherence — verified — but
dimer-vs-trimer and ~37 min–1 day lifetime are disputed; the model is in-silico, does not measure nature).

## Part 4 — THE FRONTIER + the COV-1 result (where we are NOW)
**COV-1 (`sweep/po10_covariance_readout.py`, post-hoc on ucB keystone data, NO model runs):** tested the advisor's
open Q — does a covariance-across-trials readout recover co-membership DIRECTIONALLY without the abundance leak?
Reproduced the baseline (sign-agreement `full` 0.750 DEC, controls chance). Covariance readout = per-trial products
of z-scored Δw. Naive → leaks via a whole-trial common mode. **Common-mode removed → fixes bindoff (0.938→0.562
chance) and lamshort, BUT scramble STILL decodes 0.958 (S=+1.96, p=0).** Diagnosis: sign-agreement gives scramble
**0.000** (binding destroyed) vs covariance **0.958** — the gap is a **co-activation-structured abundance
covariance** (co-DRIVEN clusters have correlated committed-dimer COUNTS trial-to-trial; survives membership-
scrambling; needs no binding). **CONCLUSION:** on this design, any magnitude-weighted/directional readout reopens
the leak; sign-INVARIANCE is what makes 0.75 binding-specific → **~0.75 is the honest ceiling here.**
**CONSTRUCTIVE:** the leak is that the design matches MARGINALS but not the trial-to-trial CO-ACTIVATION
COVARIANCE. A NEW design that also matches it is the next experiment.

## Part 5 — THE DISPATCHED PO (the frontier experiment)
See `PREREG_PO_COVARIANCE_MATCHED_DESIGN_2026-08-08.md`. Build a **covariance-matched** version of the PO10 Unit C
pairing task: co-driven and non-co-driven cluster pairs have EQUAL trial-to-trial abundance-covariance by
construction (e.g. partial/jittered co-activation so committed-dimer-count correlation is equalized across
paired/unpaired pairs). Pre-register: does a directional covariance readout (`po10_covariance_readout.py`) decode
`full` AND go to CHANCE on `scramble` under the covariance-matched design? PASS ⇒ the ~0.75 ceiling is REMOVABLE
and there is a genuine directed computation (dopamine's sign-resolution job, F3, becomes testable). NULL ⇒ ~0.75
sign-invariant is the honest output of this architecture (quantum constrains, classical computes) — a real result.

## Discipline + traps (LOCKED)
- **Emergent physics only** — no constant tuned to a downstream result. Score a STRUCTURAL invariant, never times.
  A null that can show FALSIFIED. Pre-register before running.
- **The CONSUMER is a MEASUREMENT** (MO §2.3) — "the probe ran / committed / printed CONFIRMED" is producer-green;
  done = a data-level measurement that distinguishes its outcomes, limits stated.
- Compute: `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  NUMEXPR_NUM_THREADS=1`; a Unit-C-class draw is minutes; long runs backgrounded/daemonized + caffeinate
  (`caffeinate -i -m -s`, double-fork to survive teardown) + per-draw checkpoint.
- **Data path trap:** the PO10 harness/scorer point at the OLD worktree
  `.claude/worktrees/trusting-heyrovsky-1338e9/results/po10_unitC` — the keystone data is `ucB_<arm>_<mode>fwd/rev`
  (48 full trials), NOT `val_` (a different run). Reproduce the registered scorer's `full`=0.750 before trusting any
  new decoder.
- Werner 0.5 is a theorem, not a knob. One synapse = one nanodomain.

## Commits this thread (posner_quantum_dynamics, local on master, NOT pushed — Sarah decides pushing)
F2: 202eb24, 00b1fd8, +ladder; F3: 4cb8c5e (A+B), 7056edc (b), e80202a (c), reframe ea7496a, F3-d 16b56cb;
groundings: fbec24c, 6a8da2a, 76d1e5f; synthesis (read-through-the-reground) 632ff88; **COV-1 312634f (the frontier
result)**. Canonical skill corrections are in the murmur-platform repo (bc6faa683 there), NOT this repo.
