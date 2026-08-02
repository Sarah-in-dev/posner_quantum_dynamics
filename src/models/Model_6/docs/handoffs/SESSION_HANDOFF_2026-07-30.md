# Session Handoff — 2026-07-30 (for a fresh thread)

**One line:** Thread #2 advanced hard — Step 1 (structured/fixable) and Step 2 (leak-immune consolidation
NOT demonstrated; real downstream washout) are DONE; then Sarah's isotope question exposed that the P31/P32
attribution lever **and** the dopamine-triggered collapse were BOTH non-physical, which **reordered the
program**: ground the two quantum pillars (F1 isotope lever, F2 measurement trigger) BEFORE Step 3.
**F1 is DONE** — the circular P31/P32 swap is replaced by a DERIVED ⁶Li/⁷Li scalar-relaxation lever,
validated against the literature and wired in. **F2 is next.** Governing discipline: **EMERGENT ONLY —
no tuned constants** (Sarah, ratified). Branch `claude/trusting-heyrovsky-1338e9`, **7 commits ahead of
origin, NOT pushed.** Nothing running.

## Read-order (ground first, always)
1. `session-discipline` + `agent-grounding-protocol` (in full).
2. **Research log `RESEARCH_LOG_CALCIUM_DIMER.md` rows (newest first): `ISO-1`, `STEP2`, `STEP1`** — these carry
   the full reasoning + evidence for everything below. Read them before the code.
3. Skills: `quantum-computation-and-attribution` (§5 the attribution lever, §6.1 emergent-physics/no-tuning),
   `model6-commitment-pathway` (DDSC — the consolidation chain), `coherence-gated-learning`,
   `quantum-system-canonical` (§4.3 eta gate, §4.4 commitment).
4. Prereg docs: `PREREG_F1_EMERGENT_ISOTOPE_LEVER.md` (done), `PREREG_STEP2_SELECTIVE_CONSOLIDATION.md` (carries a
   CORRECTION header — the "non-circular" claim is retracted), `PREREG_STEP3_VALENCE_INTO_CONSOLIDATION.md`.
5. Code: `nuclear_relaxation.py` (the emergent lever; its `__main__` IS the literature-reproduction acceptance test).

## What's DONE this session (with commits)
- **STEP 1 (`76ac3f6`):** cross-synapse partition is **STRUCTURED/FIXABLE** — input-dependent below ~2200 dimers,
  swamps above (5 seeds, pre-registered). The "blob" was the synchronous/high-dimer regime, not inherent.
- **STEP 2 (`7f749f7` harness+prereg, `2e9b323` result):** leak-immune selective consolidation **NOT demonstrated**
  (registered within-condition partial-corr at CHANCE for P31, null-p 0.725; only the abundance leak separates).
  **Diagnosis (collapse-timing probe):** the measurement fires at reward step 0 on the CLEAN {AB}|{CD} partition —
  so the credit is lost **DOWNSTREAM**: commitment is calcium-dominated (~7/8 fire regardless of the per-component
  coin). Not a timing artifact. Harness `sweep/step2_selective_consolidation_probe.py`, scorer
  `sweep/po11_valence_score.py`, data `results/step2_consolidation/` (48 draws), timing probe
  `sweep/step2_collapse_timing_probe.py`.
- **ISO-1 audit + REORDER (`35664c2`):** the isotope lever (`fraction_P31` → hardcoded `T_singlet` 216↔0.4) is
  CIRCULAR (magnitude assumed, no quadrupolar calc), the WRONG isotope (field uses ⁶Li/⁷Li or Ca/O; ³²P radioactive
  + no stable P alternative), and even 216 s is untraceable. Dopamine-collapse has ZERO Fisher basis (his readout is
  Ca/glutamate binding). Was a KNOWN "isotope hold" run past. → reorder: F1, F2 before Step 3.
- **F1 DONE (`5a0d5a9` module, `bfb25e6` wiring):** emergent ⁶Li/⁷Li lever. ³¹P stays the qubit; Li⁺ dopes the Ca
  site and DERIVES ³¹P T₂ via scalar relaxation of the 2nd kind (Abragam; arXiv 2310.13484 eq.9). The ⁶Li/⁷Li
  contrast is **parameter-free** (tabulated γ/I/Q); one calibrated input `J_LI7=18 Hz` (paper's anchor) sets the
  absolute scale. VALIDATED: reproduces the paper (5 orders, ⁷Li→~15 s scalar); wired via `environment.dopant ∈
  {None,'Li6','Li7'}`; **undoped bit-identical** (216 s, no regression); ⁷Li → 13.76 s and it PROPAGATES to live P_S
  (0.934→0.678 over 5 s, crosses the Werner floor). `fraction_P31` retained-but-DEPRECATED.

## The reorder + the discipline (do not relitigate)
Two quantum pillars were phenomenological; emergent-only makes grounding them mandatory before Step 3:
- **F1 (DONE):** coherence/isotope lever — now derived.
- **F2 (NEXT):** the MEASUREMENT trigger — still phenomenological.
- **F3:** Step 3 (valence into consolidation) — deferred behind F2.
**EMERGENT ONLY:** every constant on a causal path is cited/derived, never tuned to a downstream result. The
honest NEGATIVE (e.g. "consolidation is calcium-dominated / classical") is as publishable as a positive.

## F2 — the next build (ground the measurement trigger)
**Finding (research-log ISO-1 + audit):** the collapse fires on a raw boolean `dopamine_present =
stimulus.get('reward', False)` in `multi_synapse_network.py` `_evaluate_coordinated_gate` (and a per-dimer
collapse in `model6_core.py` ~:633 gated on `dopamine_read and calcium_elevated`). The biophysical
`dopamine_system.py` runs but never gates the collapse. **Fisher's real measurement:** two spin-correlated
Posners/dimers BIND — the binding *is* the projective measurement — → melt → release Ca²⁺ (the code ALREADY
models Ca-return-on-dissolution, 6 Ca/dimer, `model6_core.py:479-484,777-781`) → enhance glutamate.
**Grounded shape:** trigger the collapse on the **binding event** (which needs a Posner-Posner binding model —
does NOT currently exist), and demote reward/dopamine to a SEPARATE three-factor learning signal, decoupled
from the measurement. Task #9. This is a from-scratch physics build like F1 was.

## F1 follow-on (before any isotope experiment)
The derived ⁷Li kill is GENTLE (14 s vs 216 s, ~15×) vs the old instant 0.4 s — it operates on the COHERENCE
WINDOW. So the ⁶Li/⁷Li contrast lives in **cross-trial persistence / delayed readout**, NOT immediate partition
formation (a 3 s write still forms a ⁷Li partition). Any isotope discrimination experiment must span the window;
the old Step-2 `edges=[]`-on-contact design will NOT show it. Redesign needed.

## F3 (deferred — Step 3) — note before building
`PREREG_STEP3` is written. Emergent-only surfaced that the commitment path already carries FITTED params
(`field_threshold_kT=20` uncited, `mean_eligibility>0.3` a "chosen cut", `apply_reward_correlated`
`learning_rate=0.05`, circular "~50 dimer"). Step 3 "Task 0" must ground/replace these first, or a P31-recovers
result is unattributable. Do NOT tune.

## Operational state + GOTCHAS
- Branch `claude/trusting-heyrovsky-1338e9`, **7 commits ahead of origin, NOT pushed** (Sarah decides pushing).
  All work + data committed (results force-added).
- **COMPUTE — cap numpy threads (learned the hard way):** 4 uncapped Model-6 workers spawn BLAS threads across
  ALL cores; on 2026-07-30 this saturated the 14-core Mac for ~70 min and it **REBOOTED** mid-sweep (checkpointed
  jsonl saved 28/48 draws). ALWAYS `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1` (1 core/worker; N workers = N cores). Checkpoint per draw.
  See memory `long-batch-teardown-robust-launch` (updated). Daemonize gotcha: use raw fd `os.dup2(f.fileno(),1/2)`
  and wrap the stdin redirect in try/except (tool-launched stdin fd is invalid).
- **Compute reality:** Step-2-class draw ~47 min thread-capped (single core); the O(n²) tracker is the cost.
- Skill fix owed: `quantum-computation-and-attribution` §9 cites "Mulhall et al. 2018" which appears not to exist —
  likely Player & Hore 2018 (arXiv 1807.06339). Flagged in `35664c2`.

## First move for the fresh thread
Ground per read-order (ISO-1/STEP2/STEP1 rows carry the reasoning), then pick up **F2**: audit the measurement
trigger + the existing Ca-return machinery, and scope the emergent **binding-as-measurement** build (collapse on
the Posner-binding event; dopamine → separate learning signal) — pre-registered emergent-only, with a control
ladder. F1's `nuclear_relaxation.py` is the template for "derive it, reproduce the literature, then wire."
