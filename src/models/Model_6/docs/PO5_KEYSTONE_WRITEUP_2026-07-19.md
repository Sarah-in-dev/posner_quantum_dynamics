# §8 Keystone — complete write-up and research agenda
**PO-5 · 2026-07-19 · branch `claude/nervous-hertz-7ccff6` · worktree `.claude/worktrees/po5-keystone`**

---

## 1. THE QUESTION

`quantum-system-canonical` §8, Keystone #1:

> *"Topology is the computation" needs **pair-level** selectivity (which dimers bond depends on
> input), not just gate-level (which regions/timings are eligible). If formation is gate-selective
> but pair-flat, the partition carries no more than active-region density and "graph as computation"
> weakens to "scalar as computation."*

Never tested before this session.

## 2. THE ANSWER

**Not supported.** As constructed, the entanglement topology does not carry input-specific
information beyond active-region density. *"Topology is the computation"* is **not realized in this
implementation.**

**Not "disproven".** The decisive control is underpowered against the effect size in question
(scatter ≈9–10 components; detectable ≥18–20; the disputed effect was ≈8). Single synapse, 1 s,
3 seeds, one drive contrast. **This is not a verdict on Posner-based computation** — it is a verdict
on this build.

**Also NOT tested:** whether dimers have computational capability. We tested whether the graph
*construction* preserves input information. It does not. That is a statement about bond formation
and readout, not about the substrate.

---

## 3. WHAT WE DID — ten units, in order

| # | Question | Result |
|---|---|---|
| 1 | Is the 1/r³ factor `g` inert? | **LIVE.** `f_sat=0.176`, dynamic range `D=33.5`. **Both standing priors refuted** — the board's saturation prediction and PO-5's own vanishing prediction. |
| 2 | Which mechanism makes the bonds? | **83% birth-pairing / 0% Pathway-1 / 17% EM.** A **third, undocumented, deterministic site** (`dimer_particles.py:218-228`) dominates. Instrument gate failed first on real data, cause traced, fixed. |
| 3 | Is P0 an indifference graph on birth time? | **CONFIRMED 18/18** across seeds. `components = 1 + count(birth gaps > 100 ms)`, predicted from timestamps alone. Births are **bursty**: 12–18 events in 5 s, ~60–90 dimers each, gaps to 3.5 s. **Drive pattern modulates burst structure.** |
| 4 | Is the collective field a percolation control parameter? | Curve measured: components 60→1 as bus 0→22. **(Framing later corrected — see #7.)** |
| — | *Q-B, first attempt* | Ran 58 min, all gates passed, **no verdict** — the statistic indexed cells per-run and was never comparable across runs. Design flaw, not a bug. |
| 5 | Can a genuine (non-constant) sheaf be built? | **Yes, and it's classical.** Stalk = the dead ℝ⁶ `j_couplings_intra`; restriction = coordinate projection (real, non-identity, **no phase**). Constant-sheaf validation exact. **H⁰(engaged)=8 where component-count=1.** |
| 6 | Does J-mismatch dissolution fragment the graph? | **NOT SUPPORTED.** `(B−A)/spread = 0.00` everywhere. **Dissolution is inert**: `k ≈ 1e-4/s` ⇒ the graph is **write-once**. |
| 7 | Where is the percolation threshold? | **BOTH mechanisms percolate independently.** P0 threshold ≈2–10 ms vs native 100 ms; and at native field, P2 spans the graph **alone** even with P0 fragmented. **Doubly supercritical — no single lever can work.** |
| 8 | Is there an informative transient? | **No.** Field is `min=max=22.095` from the first sample. **No rise, no window.** Falsified a claim PO-5 had written into the log minutes earlier. |
| 9 | Does input modulate topology where structure survives? | Apparent effect, `d=2.92`, clearing the registered bar. |
| 10 | **Is that effect pair-level or density?** | **DENSITY.** Residual +0.74 / +0.67 SD — on the curve density alone produces. **And Unit 9 reverses sign on fresh seeds** (−8.3 → +5.3). The effect was seed noise. |

---

## 4. FINDINGS THAT STAND

1. **Bond provenance: 83% / 0% / 17%.** The dominant mechanism is a deterministic loop that bonds
   every template-bound dimer born within 100 ms — no rate, no RNG, no distance term. Every
   document, including PO-5's own charter, analysed the 17% mechanism.
2. **P0 is a unit-interval (indifference) graph on the birth-time axis.** Components = maximal runs
   of births with no >100 ms gap. Predicted from timestamps, exact, 18/18.
3. **The system is doubly supercritical.** P0 alone percolates (`largest_frac` 0.89 with zero EM
   bonds); P2 alone percolates (`largest_frac` 1.000 with P0 fragmented to 0.41).
4. **Bond dissolution is inert** (`k≈1e-4/s`). The graph is write-once; the observed saturation
   decline over 30 s was **dimer death**, not bond dissolution.
5. **No transient window.** The field is saturated from the first measurable instant.
6. **A non-trivial classical sheaf is constructible** on data the model already carries and discards,
   and its H⁰ is non-degenerate exactly where component-counting is dead.
7. **The keystone is not supported** at the power available.

---

## 5. WHAT IS MISSING — the research agenda

### 5.1 PHYSICS / LITERATURE — questions for you

**Q1. What is the correct inherited-entanglement timescale, and is it pairwise or all-to-all?**
This is the single highest-leverage unknown. Fisher 2015 gives entanglement between **two phosphates
from one pyrophosphate** — a *pair*. The code entangles **all** template-bound dimers born within
100 ms — a *clique*, which is what generates the 60–90-dimer groups that percolate the graph.
`model6-research-findings-may29:64,141` already recorded this as an **overclaim** ("actual gate is
spatial proximity to template field"); the docstring at `dimer_particles.py:234-236` is **still
uncorrected**. The 100 ms itself is grounded **only by an upper bound** (Fisher's ~1 s), recorded as
*"Tunable parameter — candidate for TALON sweep"*, and never swept.
*What would settle it:* the actual coincidence structure of ATP/pyrophosphate hydrolysis at a
synapse, and whether co-hydrolysis produces pairwise or collective correlation.

**Q2. Does the derived 20 kT apply as a global multiplier on every pair's bond rate?**
The 20 kT trp-superradiance coupling is derived two independent ways and marked *do not re-derive*
(`model6-network-layer-feasibility-may30`). But we measured that a field of that strength percolates
the graph **by itself**. Either the derivation describes a different quantity than the code's global
bond-rate multiplier, or the code's *use* of it is wrong, or graph-as-computation has a real problem
at this scale. **The derivation is not in question; its application is.**

**Q3. Is intra-dimer J-coupling relevant to inter-dimer entanglement?**
PO-5 assumed a resonance/detuning form (Hartmann–Hahn-like). **Unverified**, and the mechanism
failed — though it failed because it was applied to dissolution, which is inert, so the assumption
was never actually tested.

**Q4. What physically selects which spin pair mediates a bond?**
Needed for a genuine sheaf. PO-5 used bond direction as a stated modelling choice, not a derivation.

**Q5. What physically reads the partition?**
`entanglement-topology-measurement` A5 rejects spectral readout because *"no molecule diagonalises a
Laplacian"* — correct for λ₂. But **sheaf H⁰ is a ground-state degeneracy**: the dimension of the
space of locally-consistent global assignments, i.e. how many ways the system can settle. That is a
physical property, not a computed statistic. **This argument has not been adjudicated** and it
decides whether the richer readout is admissible.

**Q6. Should the system sit at criticality, and what holds it there?**
Both mechanisms sit above their thresholds. If the intended architecture is critical, the missing
piece is a **homeostat**, not a signal path — which routes into the SOC thread and PO-2's finding
that the phosphate loop is not mass-conserving so the reset never closes.

### 5.2 IMPLEMENTATION GAPS — measured, not speculative

- **The condensate cannot reach the field.** `model6_core.py:543` passes
  `backbone_eta * E_invasion`; **both factors measure 0.0000 in every live trial.** The modulator
  input is wired to a dead line; the field free-runs on metabolic UV instead.
- **The readout reads only component count** (`dim ker L₀`), which is pinned at 1 at the native
  operating point — the one channel provably empty there.
- **`j_couplings_intra` (ℝ⁶ per dimer, Agarwal DFT) is collapsed to `std`/`mean`** at
  `dimer_particles.py:310-311` and never reaches the entanglement layer.
- **Dissolution is inert**, so any structural gating must act on **formation**.
- **`_remove_dimer` (`:252-261`) never pops `_bond_lookup`** — currently dead code (zero calls
  measured across a full 9-run protocol), a latent landmine for any future caller.

### 5.3 WHAT WOULD ACTUALLY SETTLE THE KEYSTONE

A properly powered version of Unit 10: **≥10 seeds per condition** (3 was too few — it produced a
`d=2.92` that reversed sign), **matched or covaried density**, in a regime where the readout is not
saturated, ideally with **sheaf H⁰** alongside component count since the latter is coarse. Estimated
~2–3 h compute. **This is the experiment that would move "not supported" to a real verdict.**

---

## 6. METHODOLOGICAL LESSONS — for whoever continues

1. **Persist the scored intermediate; score offline.** A scoring bug destroyed 58 min of physics.
   `sweep/score_leta5.py` already solved this and was named as prior art in PO-5's own brief.
2. **Three seeds is not enough to report an effect.** Unit 9's `d=2.92` reversed sign on replication.
3. **Register criteria that test the actual bar.** PO-5 twice registered criteria that *passed*
   without answering: Unit 8's `largest_frac < 0.99` admitted "blob plus crumbs"; Unit 9's omitted
   the density control that §8 is explicitly about.
4. **Check a constant's provenance before building on it.** PO-5 argued the operating point was
   misconfigured before checking that 20 kT is derived and locked.
5. **Never read a min/max range as a trajectory.** PO-5 wrote "the field climbs every run" from a
   range; Unit 8 showed it is saturated from the first sample.
6. **Verify Tier-3 edits are bit-identical when off.** Both code changes here were verified
   `1034 / 369740 / 0.991922159684` against pre-change code.

---

## 7. ARTIFACTS

Probes `sweep/po5_unit{1..10}*.py` · pre-registrations `docs/PREREG_PO5_UNIT{1,2,3,6,7}*.md` ·
results JSON + run logs committed beside each probe · research-log entries `L·PO5-1` … `L·PO5-9`
with DECISION RECORD rows in `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`, including the withdrawals.

**Two open items:** the Unit 6 J-mismatch edit (dead mechanism, opt-in, recommended revert) and the
Unit 6 scrambled-control RNG bug (`np.random.permutation` draws from the global stream — arm C is
confounded and uncitable).
