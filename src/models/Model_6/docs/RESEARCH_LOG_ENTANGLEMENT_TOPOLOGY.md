---
name: research-log-entanglement-topology
description: >
  Append-only research log + decision record for the Model 6 ENTANGLEMENT-TOPOLOGY
  sub-program (the entanglement partition as the computational primitive: the Werner
  distance-graph, the coherence-set radius, the far-pairs-first fragmentation test T1',
  and the retired SOC-topology power-law T1). The PRIMARY provenance record the paper
  draws from for the topology claims: every load-bearing claim carries a source and an
  epistemic-status tag, every decision carries its reason and date. The
  entanglement-topology-measurement and model6-entanglement-partition-werner skills distil
  the LOCKED decisions; this log carries the granular "why" and the evidence behind them.
  Sibling to RESEARCH_LOG_CALCIUM_DIMER.md (the chemistry sub-program). Read when
  continuing the topology work, writing it up, or reconstructing why a geometry, threshold,
  or verdict rule was chosen. APPEND newest entry at the top of the LOG; never rewrite
  history — supersede with a dated note.
---

# Research Log — Entanglement Topology (the computational primitive)

## Purpose & how to use this

This is the **decision-provenance record** for the entanglement-topology sub-program — the
half of Model 6 that asks *what the computation IS* (the partition of the entanglement
graph) as opposed to *what forms the dimers* (the calcium→dimer chemistry, logged
separately in `RESEARCH_LOG_CALCIUM_DIMER.md`). It exists so that when we write this up,
every number and every modeling choice traces to its source and the reason it was chosen,
and a future session can see *why* a decision was made, not just what it was.

- **Append, don't rewrite.** New work goes in a dated entry at the top of *The Log*. If a
  later finding overturns an earlier one, add a new entry that **supersedes** it with a
  pointer — leave the original in place (the paper needs the trail, including the wrong
  turns; this sub-program has several instructive ones).
- **Companion layers:** `entanglement-topology-measurement` (Appendix A is the current
  authority) and `model6-entanglement-partition-werner` carry the **distilled LOCKED
  decisions**; this log carries the **granular evidence and rationale**. When a decision
  locks, summarize it in the skill and point back here for the "why."
- **Epistemic-status tags** (same legend as the chemistry log):
  `[PROVEN]` literature/algebra-established · `[GROUNDED]` tied to a named measurement ·
  `[MODELED]` defensible choice not forced by physics · `[INFERRED]` follows from the model ·
  `[CONTESTED]` an unsettled bet · `[LOCKED]` settled, not relitigated without new physics.
- **Discipline (LOCKED):** emergent physics only — no constant tuned to a downstream target.
  The Werner bound 0.5 is a separability THEOREM; d*, T_eff, k_classical are not knobs. If
  the physics doesn't give the result, the log records the gap; it is not a license to slide
  a knob. **Score the ORDER, never the times** (T1' §3 below is the scar behind this).

## The epistemic frame (Sarah, LOCKED 2026-07-17)

> "We will never be able to model quantum in a silicon system, but we can model how the
> system expects to work."

This log's results are **not** claims about nature. A simulation has no privileged access
to whether the biology is quantum (`quantum-computation-and-attribution` §5, the attribution
gap: no experiment measures the quantum state *in the living computational system* AND
attributes the computation to it — attribution always routes through theory). The model is
the **theory of how the system expects to operate**, made realistic and — the load-bearing
part — made **DISCRIMINATING**: built so it predicts *differently* from its classical rival
on a measurement we can actually make (§6.3). T1' is exactly such a discrimination test.
A confirmed far-pairs-first cascade does not prove the biology is quantum; it shows the
*model's* memory carries spatial structure a classical scalar trace cannot, which is what
makes the model falsifiable and worth believing by convergence rather than by fit.

---

## DECISION RECORD (running summary — newest first)

| # | Date | Decision / finding | Status | Entry |
|---|------|--------------------|--------|-------|
| ERR-2 | 2026-07-18 | **CORRECTION — ETA-1 overstates its conclusion; its measurements were taken with NMDARs SILENT.** `eta_probe.py:68` drives voltage-only (`{"voltage": v, "reward": False}`); `analytical_calcium_system` defaults `glutamate=0.0`, so only 25 of 50 channels conducted. `model6-input-engine` documents this as a KNOWN OPEN INTEGRATION GAP and predicted the symptom verbatim — *"will show reduced calcium and lower E_invasion ... NOT a regression. Do not read it as one"* — and states that `get_open_fraction()` IS the `ca_open_fraction` feeding `compute_metabolic_power`. So the term setting `r` was measured with its glutamate contingency unsatisfied. **SURVIVES:** the arithmetic, the `d*`/row-sum geometry, the duty-cycle analysis, and the identity that eta=0 ⇒ no cross-synapse bonds. **DOES NOT SURVIVE:** "the pump does not ignite" as a claim about the model — measured `r` is a **LOWER BOUND**, and ETA-1's three-way fork is premature because it presumes a complete input path. **Grounding failure:** `model6-input-engine` was not read before dispatching the measurement; it owns the input path and carries a "READ THIS before any full-model run" section naming this exact misreading. Supersede with L·ETA-2 after the integration is finished and eta re-measured. | [GROUNDED, code SHOWN] | L·ETA-1 (banner) |
| ETA-1 | 2026-07-18 | **[SEE ERR-2 — CONCLUSION CORRECTED]** ~~THE PUMP DOES NOT IGNITE — the cross-synapse partition exists only under an IMPOSED η, never a driven one.** Measured across 4 conditions (`sweep/loop_audit_2026_07_18/eta_probe.py`): `eta = 0` under rest, under the spatial-discovery −40 mV drive, and under the −10 mV theta burst **with the clamp removed**. `r = p_met_agg/P_c` floors at **0.0390** and peaks at **0.1409** after 30 s of sustained drive; extrapolating `E_invasion` to its ceiling gives `r ≈ 0.32`, still **3× below** the `r ≥ 1` condensation threshold. Cause: `ca_open` is the binding constraint and measures **0.063** duty-averaged over the real theta protocol, not the `≈1` assumed at `soc_pump_threshold_stage1.py:88` — a ~10× overstatement, and why that falsifier never fired. **Consequence:** `k_cross ∝ sqrt(eta_i·eta_j)`, so at eta=0 **zero cross-synapse bonds form** — the clamp in every topology probe is what creates the partition, not merely what holds it steady. **T1′ is untouched** (it clamps by declaration, scores order only, and the Werner algebra is independent) but its LICENSE is narrowed: "the topology IS the eligibility trace **in a running network**" is unsupported — in every end-to-end run the topology is empty. **NOT FIXED BY TUNING:** `P_c`(ω₀,Q) and `p_active_max_W` were left alone deliberately; moving them is the §6.1 emergent-physics failure mode. Open fork: wrong drive protocol vs condensation being intrinsically a population phenomenon (~10–20 synapses at ≤1 µm; unreachable at ≥2 µm at any N). **Blocks the §8 input-selectivity phase**, whose "vary only the drive" constraint is currently unsatisfiable. | [GROUNDED, 4 conditions] | L·ETA-1 |
| T1'-6 | 2026-07-18 | **CHANNEL SEPARATION — the population confound is measured INERT; §6's conclusion stands, §6's argument is replaced.** Adversarial review charged that dimer loss feeds the same extreme-value statistic that sets `d*_eff`, so it produces far-pairs-first too and replication separates nothing. Pre-registered 4 arms (+2 counterfactual) BEFORE running (`abdb549`). Result: **arm A ≡ arm B bit-identically** (`max|A−B| = 0.000e+00` on every `d*_eff`/`max_pair` sample, 4/4 seeds) and **arm C produces ZERO breaks** while the population falls 2223→~50, retaining **100.0000%** of `max_pair(P_S²)`. Cause, from code: `dimer_particles.py:230-241` removes **lowest-`P_S`-first** (`coherence` is an affine increasing map of `P_S`, :57-62), so attrition is rank-selective and the argmax is the LAST thing removed. The charge additionally fails **under its own assumption**: arm C_rand (uniform random removal) also gives zero breaks — `Δd*_eff = −0.002 µm` against a 1.35 µm span — because `P_S(0)` is packed at its ceiling (median 0.9986, max 1.0), leaving no room below the max. Null arm D clean. **Where it does bite:** arm A_rand (random removal + decay) confirms 4/4 with systematically earlier breaks — a model that removed randomly *would* have a live population channel. **SECONDARY, damaging:** order-recovery power measured **37/40 ≈ 92%** across noise draws (seed 0 = 7/10), not the **10/10** §4 cites — different estimators (this run applies the probe's `CONSECUTIVE_ABSENT=3` guard; the power probe used unguarded first-crossing). Headline unaffected: `p≈3.0×10⁻⁶` is against the classical null, which power does not move, and 4/4 at 92% power has probability 0.72. | [GROUNDED, 4 seeds × 6 arms] | L·T1'-6 |
| ERR-1 | 2026-07-17 | **CORRECTION — three P_S_crit annotations were arithmetically wrong when first committed.** The wide ladder's critical-coherence values were logged as 2.90µm→0.9327, 2.45→0.8801, 2.00→0.8305. Correct values (`P_S_crit = sqrt(0.5·exp(gap/λ))`, λ=5µm) are **0.9450 / 0.9034 / 0.8637**; 3.35µm→0.9885 was right. Verified against the Jul-16 ladder's values, which were correct (3.0→0.9545, 2.8→0.9356, 2.5→0.9079, 2.0→0.8637 all reproduce exactly). **The T1' result is UNAFFECTED**: these are descriptive annotations only — the probe never reads them, it computes d* at runtime from measured P_S and compares gaps directly, so the 4/4 cascade used the real geometry and the real Werner threshold. Caught when Sarah asked for a from-scratch explanation of the mathematics and the derivation was re-done by hand. Fixed in `sweep/coherence_fragmentation_probe.py` and in T1'-3/L·T1'-3 below. | [GROUNDED, corrected] | — |
| T1'-5 | 2026-07-17 | **T1' REPLICATION COMPLETE — far-pairs-first CONFIRMED across 4/4 independent seeds; CONCLUSIVE.** Seeds 0,1,2,3 (wide ladder, 90 s, dt=1e-3; seeds 1–3 run in parallel). ALL FOUR broke in the exact pre-registered order **3.35 > 2.90 > 2.45 > 2.00**. Under the classical null (no spatial structure ⇒ break order is a uniformly random permutation of 4 ⇒ 1/24 per seed), 4/4 in order ⇒ **p = (1/24)⁴ ≈ 3.0×10⁻⁶**. Break TIMES scatter seed-to-seed (gap 2.90 broke at 32.5/37.0/42.0/32.5 s; gap 2.45 at 61.5/55.0/54.5/64.5 s) while the ORDER is invariant — vindicating the score-order-not-times decision (L·T1'-2). Replication also DEFEATS the population-collapse confound on the late breaks: uniform dissolution lowers every pair's radius together (edges would die ~simultaneously), so it CANNOT manufacture a consistent gap-spaced order across independent seeds — only the coherence/distance mechanism does. Altitude unchanged: **(A)** — the model's partition carries spatial structure a classical scalar eligibility trace cannot; NOT a claim about quantumness (attribution gap stands; see the epistemic frame). Completes L·T1'-4. | [GROUNDED, 4 seeds] | L·T1'-5 |
| T1'-4 | 2026-07-17 | **T1' DYNAMIC — far-pairs-first fragmentation CONFIRMED, seed 0 (single seed).** → *completed by T1'-5 (4/4 replication); kept as the trail — the moment the result was one seed and only suggestive.* Wide ladder `[3.35,4.5,2.90,4.5,2.45,4.5,2.00]µm`, 90 s silence, dt=1e-3. All 4 live edges broke in the exact pre-registered gap order — 3.35µm@14.5s, 2.90@32.5s, 2.45@61.5s, 2.00@78.0s — verdict `far-pairs-first order CONFIRMED over 4 breaks`. This is the DISCRIMINATING result: a classical scalar eligibility trace decays uniformly and cannot produce a spacing-ordered cascade. **Single seed ⇒ p≈1/24≈0.042 vs the classical null — suggestive, NOT conclusive** (a 2nd independent seed in order → p≈0.0017). Breaks 1–2 landed at HEALTHY population (1843, 1043 dimers) — clean, coherence-driven; breaks 3–4 at CRATERED population (259, 98) — TIMING confounded by dimer-loss, ORDER preserved (uniform dissolution lowers every pair's radius equally). Guards worked live: 2 flickers (gap3.35, gap2.45) correctly rejected, not scored. Altitude: **(A)** — the partition carries SPATIAL structure; says nothing about **(B)**/quantumness. Ran 2.7 h not the projected ~8 h (O(n²) tracker cost collapsed with the population). `sweep/coherence_fragmentation_probe.py`; log session-scoped. | [GROUNDED, single seed] | L·T1'-4 |
| T1'-3 | 2026-07-17 | **The trustworthy rebuild — geometry chosen for POWER, not for "early breaks".** The Jul-16 redesign premise ("gaps just under d*(0) so the cascade lands early") was FALSIFIED, measured not argued: edge survival is governed by the P_S TAIL (max over bonded pairs), which decays ~3.6× slower than the median — an 8 s null showed the median radius below all 4 gaps while all 4 edges lived. Worse, each synapse's tail is set by its own frozen-at-creation `T_eff`, giving ~0.29µm between-synapse d*_eff scatter, so rungs finer than that are decided by luck. **Measured order-test power (10 seeds via the validated d*_eff replay): tight 0.10µm rungs 6/10, medium 0.25µm 5/10, WIDE 0.45µm 10/10.** Geometry set to the wide ladder — 0.45µm rungs clear the scatter. Cost: cascade lands later (~90 s), accepted for a resolvable order. | [GROUNDED, 10-seed replay] | L·T1'-3 |
| T1'-2 | 2026-07-17 | **d*(0) MEASURED = 3.4521µm (median), NOT assumed.** Rig = the confirmed static-probe rig, t=0.08 s, n≈2200. P_S median 0.9987, min 0.9922, max 1.0 ⇒ d* min 3.387 / median 3.452 / max 3.466 µm. The SPREAD is the load-bearing fact (two prior wrong claims came from assuming P_S). d*_eff replay (`sweep/dstar_eff_replay.py`) VALIDATED against the real rig (median P_S matches to <0.0005); its P_S→d*_eff mapping is UNVALIDATED by construction (no honest cascade datum — the only candidate, "gap 3.0 broke @34 s", is the retracted Jul-16 FLICKER) ⇒ it emits an UPPER BOUND on break times, never a prediction to score. | [GROUNDED] | L·T1'-2 |
| dt-1 | 2026-07-17 | **dt convergence — the order test is honest at production dt, and the operating point survives.** P_S and the Werner edge set are dt-CONVERGED (`d*_med=3.45`, `edges=5` at every dt from 1e-4 to 5e-3), so the ORDER test reads converged quantities at dt=1e-3. The dimer COUNT is NOT converged in the drive transient (~+38% at dt=1e-3 vs 1e-4; dt=1e-2 OVERFLOWS — do not use) — explicit-Euler error on stiff formation, NOT a new bug. BUT the saturated OPERATING POINT is converged to ~5% (plateau 156.7µM @dt=5e-4 vs 163.9 @dt=1e-3): at saturation formation balances the correctly-dt-scaled dissolution, so the transient inflation cancels. **The ~155µM operating point is NOT a dt artifact.** (Corrects an in-session over-extrapolation of the 38% to the operating point.) Does NOT reopen D8/D14/D16/D17 on dt grounds. | [GROUNDED] | L·dt-1 |
| T1'-1 | 2026-07-16 | **T1' STATIC HALF — the Werner cut IS a coherence-set distance rule.** Algebra from live code: `F = P_S_i·P_S_j·w`, `w=exp(-d/5µm)`, edge iff `F>0.5` ⇒ **bond iff `d < d* = 5·ln(P_product/0.5)`**. Two theorems fall out, neither tuned: (1) hard coherence floor `P_S>1/√2≈0.7071` (since `w≤1 ⇒ F≤P_S²`); (2) radius SHRINKS as coherence decays ⇒ partition must fragment far-pairs-first. CONFIRMED pre-registered (`sweep/coherence_radius_probe.py`, ~43 s): 8-synapse ladder called 7/7 gaps, exact edge list, betti0_cross=3, sizes=[3,3,2], betti1=0. **Retrodicts the Stage 3 chain/ring validation with NO free parameters.** This GENERATES the T1' dynamic prediction. | [PROVEN algebra + GROUNDED probe] | L·T1'-1 |
| T1-RET | 2026-07-16 | **T1 (SOC-topology power-law) RETIRED — structural, ~0 compute, NOT a negative result.** (1) 1D forbids it: the honest dendrite is 1D (`_generate_positions 'linear'`); in 1D clump size is exactly geometric `P(k)=p^(k-1)(1-p)` — exponential tail (geometric fit R²=0.97–0.996 at every density, beats power-law every time). (2) D18 forbids it: nucleation is all-or-none bistable with a quantal, drive-independent ON amplitude (~135) — a characteristic SCALE, where SOC needs scale-FREEDOM. (3) Two "SOC" claims were conflated: SOC-*chemistry* (phosphate depletion→S→1, D8/D14) STANDS; SOC-*topology* (power-law clumps) inherited the name and is what's retired. | [GROUNDED structural] | L·T1-RET |

---

## THE LOG (newest first)

### L·ETA-1 — The pump does not ignite: the cross-synapse partition exists only under an imposed η · 2026-07-18  `[GROUNDED, 4 measured conditions + computed thresholds]`

> **CORRECTED SAME DAY — ERR-2. The headline below overstates its conclusion; read this
> first.** Every measurement in this entry was taken with **NMDARs structurally silent**.
> `eta_probe.py:68` drives `net.step(dt, {"voltage": v, "reward": False})` with **no
> glutamate**, and `analytical_calcium_system` defaults `glutamate = 0.0`, so only the 25
> VGCC of the 50-channel population conducted. `model6-input-engine` (June 14) documents
> this as a KNOWN OPEN INTEGRATION GAP and predicted this exact symptom in advance:
> *"Any full-model run between now and that wiring will show reduced calcium and lower
> `E_invasion` — this is the expected consequence of the half-VGCC live path, **not a
> regression**. Do not read it as one."* It further states that
> `channels.get_open_fraction()` **is** the `ca_open_fraction` consumed by
> `compute_metabolic_power` — i.e. the term that sets `r` is glutamate-contingent, and was
> measured with its glutamate contingency unsatisfied.
>
> **What survives:** the arithmetic (`P_c = 21.51 fW`, the `d*` / row-sum geometry table,
> the duty-cycle analysis), and the structural consequence that at eta=0 no cross-synapse
> bonds form — that is an identity in `k_cross`, independent of why eta is 0. The T1′
> caveat also survives: the topology probes clamp, and they too pass voltage only.
>
> **What does NOT survive:** the conclusion *"the pump does not ignite"* as a statement
> about the model. It ignites or not under an input path that is **documented-incomplete in
> exactly the term that drives it**. The measured `r` values are a **LOWER BOUND**, not the
> model's capability. The three-way fork below (protocol / population / genuine negative) is
> **premature** — it presumes the input path was complete, and it was not.
>
> **Also now suspected, not confirmed:** the "unresolved ~4× discrepancy" noted below
> between isolated and in-network `ca_open` is NOT explained by glutamate — `ca_probe.py:13`
> calls `update_gating(0.001, v)` without it either, so NMDARs were closed in both. That
> discrepancy remains genuinely open.
>
> **Grounding failure recorded, per house discipline:** `model6-input-engine` was not read
> before the eta measurement was dispatched. It is the skill that owns the input path and it
> carries a "READ THIS before any full-model run" section naming this precise
> misinterpretation. The topology and learning skills were read; the input one was not.
> Superseded by **L·ETA-2** once the integration is finished and eta is re-measured.

**This entry does not overturn T1′. It states the regime T1′ lives in, which was never
written down, and which materially qualifies "the topology IS the eligibility trace."**

**The measurement.** `sweep/loop_audit_2026_07_18/eta_probe.py`. Under every drive the
learning drivers actually apply, `eta` is pinned at **0**:

| condition | ca_open peak | E_invasion peak | r | eta > 0 |
|---|---|---|---|---|
| −70 mV rest, N=6 | 0.020 | 0.0000 | 0.0390 | NO |
| −40 mV sustained 3 s, N=6 (spatial-discovery drive) | 0.120 | 0.0005 | 0.0391 | NO |
| theta burst −10 mV, N=7, **clamp removed** | 0.440 | 0.0000 | 0.0390 | NO |
| −40 mV sustained **30 s**, N=7 | 0.060 | **0.3596** | **0.1409** | NO |

`eta = (r−1)/(r+1) if r ≥ 1 else 0` (`multi_synapse_network.py:1130`), with
`r = p_met_agg / P_c`, `P_c = n̄ℏω₀²/Q = 21.51 fW` (ω₀=8e6 Hz, Q=10,
`model6_parameters.py:801-802`) and `P_BASAL_W = 0.84 fW`. So `r` has a hard floor of
**0.0390** at zero drive. Run F is the decisive one: `E_invasion` does climb (0 → 0.36 over
30 s) and `r` tracks it, but linear extrapolation to the ceiling `E_invasion = 1.0` gives
**r ≈ 0.32 — still 3× below threshold.** `[MODELED]` on the extrapolation; the run was
stopped at 30 s on CPU budget. The verdict does not turn on it (3× is not an extrapolation
artifact), but the number is not a measurement.

**Why: `ca_open` is the binding constraint, and a prior probe overstated it ~10×.**
In-network `ca_open` at −40 mV measures 0.04–0.06. `sweep/soc_pump_threshold_stage1.py:88`
sets `reachable = 0.74 * 1.0` with the comment `ca_open(burst)~1`. Measured
(`ca_probe.py`): analytic steady state is 0.726 at −10 mV and 0.179 at −40 mV, and
**duty-averaged over the actual theta protocol** (4 spikes × 2 ms per 125 ms = 6.4% duty,
`coherence_fragmentation_probe.py:196-198`) it is **0.063**. That is why the stage-1
falsifier did not fire. `[GROUNDED]`

**CONSEQUENCE FOR THE PARTITION — the part that matters.** `k_cross ∝ sqrt(eta_i·eta_j)`
(`multi_synapse_network.py:309, 321`). **At eta = 0, zero cross-synapse bonds form.** So the
clamp every topology probe applies is not a convenience and not merely a control that "holds
the pump fixed" — *without it there is no cross-synapse entanglement at all, hence no
partition to measure.* T1′, the static radius probe, and the two-cluster Werner validation
all live in a regime the model does not reach from its own drive.

**What this does and does not do to T1′** (state both, do not let the caveat inflate):
- **Does NOT touch it.** T1′ clamps eta at 0.26 as a *declared* control, says so in the
  write-up (§4 "Drive protocol"), scores order only, and the Werner/distance algebra it tests
  is independent of how eta got there. The 4/4 result stands exactly as recorded in T1'-5.
- **DOES qualify what it licenses.** T1′ shows the partition *of this model, under an imposed
  pump*, carries spatial structure a classical scalar trace cannot. It does not show the
  model reaches that partition on its own. Any claim of the form "the entanglement topology
  IS the eligibility trace **in a running network**" is currently **unsupported** — in every
  end-to-end run the topology is empty.

**Geometry dependence** `[computed]`, `dep_probe.py`. Critical drive
`d* = (P_c − P_BASAL)/(p_active_max · rowsum)`; `d*` **saturates** with N because coupling is
`exp(−d/λ)`, λ=5 µm. To cross r=1 at −40 mV with `E_inv` saturated (d≈0.05) needs
rowsum > 6.89: **~10–20 synapses at 1 µm, ~10 at 0.5 µm; unreachable at ≥2 µm spacing at any
N** (rowsum floors at ~5.1).

**THE DECISION THIS FORCES (Sarah's, explicitly NOT the thread's).** The shortfall is 3×, and
`P_c` is set by `omega_0` and `Q` while `p_active_max_W` is a free parameter. Moving any of
them would make eta ignite. **That is the named emergent-physics failure mode** ("what value
makes the result come out right", `quantum-computation-and-attribution` §6.1) and this entry
records that it was NOT done. The real fork is: (a) the drive protocol is wrong — the
subthreshold −40 mV was a deliberate choice (`run_spatial_discovery.py:360-365`, "was −10 mV,
which illegally merged the plateau into the synaptic knob"); or (b) the pump genuinely does
not ignite at few-synapse scale and requires the ~10–20 co-driven synapses at ≤1 µm the table
above implies — in which case condensation is intrinsically a *population* phenomenon and the
experiment must be designed at that scale. **Open.**

**Bearing on the designated next phase (input-selectivity).** The phase directive
(`SESSION_HANDOFF_JUL17_T1PRIME_REDESIGN.md` §8) inherits from T1′ the constraint "fixed
geometry across conditions; vary only the DRIVE." That constraint is currently
**unsatisfiable**: drive does not move eta, eta is the only input-dependent channel into the
partition, and at eta=0 there is no partition. The selectivity experiment cannot be designed
until the fork above is resolved. This is a prerequisite, not a scheduling detail.

### L·T1'-6 — Channel separation: the population channel does not order, and cannot · 2026-07-18  `[GROUNDED, 4 seeds × 6 arms + 40-draw stability]`

**Outcome under the pre-registered reading (L·T1'-6-PRE, committed before the run at
`abdb549`): "B orders, C does not ⇒ coherence-driven."** §6's *conclusion* survives. §6's
*argument* does not, and is replaced below by the property that actually does the work.

`sweep/population_channel_arms.py`; traces persisted to `results/T1prime6_arms/`.

| arm | P_S | population | rule | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|---|---|
| A | decays | decays | model | FALSIFIED | CONFIRMED | CONFIRMED | CONFIRMED |
| B | decays | held | — | FALSIFIED | CONFIRMED | CONFIRMED | CONFIRMED |
| C | frozen | decays | model | **INCONCLUSIVE (0 breaks)** | 0 breaks | 0 breaks | 0 breaks |
| D | frozen | held | — | 0 breaks | 0 breaks | 0 breaks | 0 breaks |
| A_rand | decays | decays | random | CONFIRMED | CONFIRMED | CONFIRMED | CONFIRMED |
| C_rand | frozen | decays | random | **0 breaks** | 0 breaks | 0 breaks | 0 breaks |

**1. The population channel is EXACTLY inert under the model's own removal rule.**
Arm A and arm B are **bit-identical**: `max|A−B| = 0.000e+00` across every sample of every
`d*_eff` and `max_pair(P_S²)` column, in all four seeds. Arm C retains **100.0000%** of
`max_pair(P_S²)` while the population falls 2223 → ~50. This is the measured consequence of
`dimer_particles.py:230-241` removing lowest-`P_S`-first: the argmax is the last thing
removed, so attrition cannot touch the extreme-value statistic that governs edge survival.
Not "small" — zero.

**2. The criticism is immaterial even under ITS OWN assumption.** Arm C_rand imposes uniform
random removal — the charge's implicit rule — and still produces **zero breaks**:
`max_pair(P_S²)` retained 99.92–99.96%, `Δd*_eff = −0.002 µm` against the **1.35 µm** span
the cascade must traverse (0.15%). The reason is measured, not argued: `P_S(0)` is packed
against its ceiling (median 0.9986, max 1.0000), so the max over ~50 draws ≈ the max over
~2200. The extreme-value intuition in step 3 of the charge needs a distribution with room
below the maximum; this one has none at t=0. **So the charge fails twice over** — once on
the model's actual removal rule, once on the shape of the `P_S` distribution.

**3. The null is clean.** Arm D produced no breaks in any seed. The rig is not manufacturing
orderings.

**4. Where population loss DOES bite — reported because it is real.** Arm A_rand (random
removal *plus* coherence decay) confirmed 4/4 with systematically **earlier** breaks
(e.g. seed 3: 16.0/33.5/52.5/65.5 s vs arm B's 16.0/37.5/66.5/83.5 s). Once `P_S` spreads
out under decay, the tail is carried by a few long-`T_eff` dimers, and random removal can
kill them. So a model that removed dimers randomly *would* have a live population channel —
it would still order correctly, but the times would be confounded. This model does not
remove randomly. The distinction belongs in §6 rather than being allowed to read as "the
criticism was wrong".

**5. SECONDARY, and it damages a different claim: the ladder's order-recovery power is
~92%, not 10/10.** Across 40 independent noise draws (arm B, 200 s horizon, 4 seeds × 10
draws): **37 CONFIRMED / 3 FALSIFIED / 0 INCONCLUSIVE**. Seed 0 alone is **7/10** — its
last two rungs (2.45, 2.00) break within ~1.5 s of each other and invert under some draws,
which is why arm A/B above show seed 0 FALSIFIED. `RESULTS §4` cites **10/10** for this
geometry from `order_power_probe.py`. The two numbers are different **estimators**, not a
contradiction: the power probe uses unguarded first-crossing detection and a different noise
stream (`seed+10000`), while this run applies the probe's own `CONSECUTIVE_ABSENT=3` guard,
which is stricter and converts marginal orderings into violations. **The 92% is the better
number** and §4 should carry it. `[MODELED]` — it is an estimate over noise draws, not a
property of the published run.

**Does this damage the headline? No, and the arithmetic should be stated rather than
asserted.** The p-value is computed against the classical null (uniform permutation, 1/24
per seed); power affects the experiment's *sensitivity*, not the null, so `p ≈ 3.0×10⁻⁶` is
untouched. At 92% per-seed power, observing 4/4 has probability `0.92⁴ ≈ 0.72` — entirely
unremarkable. The result is not weakened; the *power claim behind the geometry choice* was
overstated.

**Replay-vs-rig honesty.** Absolute break times here run LATER than the published rig
(seed 0: 16.0/41.5/76.5 s vs 14.5/32.5/61.5 s), exactly the documented upper-bound direction
of the replay (`dstar_eff_replay.py:24-41` — fixed population ⇒ higher max ⇒ later breaks).
No replay reproduces a published realisation: the rig interleaves its noise draws with
network stepping. **This instrument is valid for ORDER questions and invalid for times**,
which is why times are not scored in any arm.

**Carried limitation** `[MODELED]`, pre-declared: N(t) is log-linear through five transcribed
seed-0 anchors, applied to all four seeds, because the raw logs are gone. Findings 1–3 are
insensitive to it — arm C's erosion is *exactly zero* under any monotone trajectory, since
the rule preserves the argmax regardless of how fast the population falls. Finding 4 is
trajectory-sensitive and is reported as directional, not quantitative.

**Open item this does NOT close:** §6 defends only breaks 1–2 as unconfounded, which is two
breaks — below the experiment's own `MIN_BREAKS=3` bar. This run makes that defence
unnecessary (the confound is measured inert, so all four breaks are unconfounded by
population loss), but §6 must still stop claiming a two-break subset as its control.

### L·T1'-6-PRE — PRE-REGISTRATION: population vs coherence channel separation · 2026-07-18  `[PRE-REGISTERED — written and committed BEFORE the run]`

**This entry is committed before the arms are run. Nothing below has been observed.**

**The charge.** Adversarial review of `RESULTS_T1prime_far_pairs_first.md` §6. §6 defends
the ordering with *"Dissolution is spatially uniform — it lowers every pair's effective
radius equally... It cannot generate a consistent spacing-ordered cascade."* The refutation,
assembled from §4's own physics: (1) edge survival is governed by
`d*_eff = λ·ln(max_pair(P_S²)/0.5)`, an EXTREME-VALUE statistic; (2) bonded pairs scale
~N², and N falls 2200→98; (3) the max of fewer draws is smaller, so `d*_eff` contracts from
population loss ALONE; (4) any contracting radius crosses gaps in width order (§2).
⇒ dissolution is uniform in RATE but not order-neutral in EFFECT, and replication across
seeds does not separate the channels.

**Grounding finding that motivates the arms** `[GROUNDED, code]`: step (3) assumes RANDOM
removal. `dimer_particles.py:230-241` removes `sorted(self.dimers, key=lambda d: d.coherence)`
from index 0 — i.e. LOWEST-coherence first — and `coherence` is a strictly increasing affine
map of `P_S` (`dimer_particles.py:57-62`). Attrition is therefore **rank-selective**: the
survivor set is the top-n by current `P_S` and the argmax is the last thing removed. This is
also the only live attrition path (`step_population` is called every step at
`dimer_particles.py:602`; `_remove_dimer` at :252 has no callers). If that reading is right,
the population channel should be *unable* to erode `max_pair(P_S²)` at all. **The arms are
run anyway, because an argument from code is not a measurement.**

**The arms** (`sweep/population_channel_arms.py`, replay — not fresh 2.7 h/seed sims), seeds
0–3, wide ladder, 90 s, guards `CONSECUTIVE_ABSENT=3` / `MIN_BREAKS=3`, verdict can return
CONFIRMED / FALSIFIED / INCONCLUSIVE:

| arm | P_S decays | population decays | attrition rule | purpose |
|---|---|---|---|---|
| A | yes | yes | model (top-n) | reproduce the published result |
| B | yes | **held** | — | coherence-only channel |
| C | **frozen** | yes | model (top-n) | population-only channel |
| D | **frozen** | **held** | — | null — ordering here means the rig is broken |
| A_rand | yes | yes | **random** | the criticism's world, full |
| C_rand | **frozen** | yes | **random** | the criticism's world, isolated |

A_rand/C_rand are outside the pre-registered four; they exist so "is the N-dependence
material?" gets a NUMBER under the criticism's own assumption rather than a dismissal.

**PRE-REGISTERED READING OF OUTCOMES** (committed before looking):
- **B orders, C does not** ⇒ coherence-driven. §6's conclusion stands; its *argument* is
  rewritten on the rank-selectivity ground, which is the property that actually does the work.
- **C orders, B does not** ⇒ the result is population-driven. §6 is REFUTED and the
  coherence claim in §1/§5 must be retracted or restated. This entry commits us to reporting
  that outcome if it occurs.
- **BOTH order** ⇒ degenerate channels; T1′ cannot separate them. §6 must say so plainly and
  the claim narrows to "spatially structured", not "coherence-driven".
- **D orders** ⇒ rig artifact. Everything downstream is suspect; stop and report.
- **C_rand orders while C does not** ⇒ the criticism is valid *in general* and refuted *for
  this model specifically*, by the rank-selective removal rule. That distinction gets stated
  explicitly rather than being allowed to read as "the criticism was wrong".

**LIMITATION, pre-declared** `[MODELED]`: the population trajectory N(t) is NOT replayed
from physics — the raw run logs were session-scoped scratchpad and are GONE. N(t) is
log-linear interpolation through five transcribed seed-0 anchors from L·T1'-4 (2200 at t=0;
1843/1043/259/98 at 14.5/32.5/61.5/78.0 s), applied to **all four seeds** because seeds 1–3
trajectories were never persisted. So the C arms test whether the population channel orders
**at all**, not seed-specific timing. Times are not scored in any arm. From this run on,
traces are persisted to a tracked path (`src/models/Model_6/results/T1prime6_arms/`).

**Known live tension to address regardless of outcome:** §6 defends only breaks 1–2 as
unconfounded, which is TWO breaks — below the experiment's own `MIN_BREAKS=3` sufficiency
bar. The unconfounded subset does not score under the probe's own rule. This must be stated
in §6 explicitly, not left implicit.

### L·T1'-5 — T1' replication complete: 4/4 seeds far-pairs-first, CONCLUSIVE · 2026-07-17  `[GROUNDED, 4 seeds]`

Seeds 1–3 run in parallel (3 cores, ~2.7 h wall; parallelism cost ~nothing — each ran at
the same ~355 s/sim-s as seed 0 alone). All four seeds broke in the exact pre-registered
order. **The order is invariant; the times are not** — which is the whole methodology,
borne out:

| seed | gap 3.35 | gap 2.90 | gap 2.45 | gap 2.00 | order |
|---|---|---|---|---|---|
| 0 | 14.5 s | 32.5 s | 61.5 s | 78.0 s | ✓ |
| 1 | 14.0 s | 37.0 s | 55.0 s | 82.5 s | ✓ |
| 2 | 14.0 s | 42.0 s | 54.5 s | 71.0 s | ✓ |
| 3 | 11.0 s | 32.5 s | 64.5 s | 82.0 s | ✓ |

**Statistical standing.** Pre-registered discriminating claim: as coherence decays the
partition fragments in gap order, widest first. Classical null: a scalar eligibility trace
decays uniformly, carries no spatial structure, so the break order is uninformative — a
uniformly random permutation of the 4 rungs, P(exact order) = 1/24 per seed. Observed 4/4
⇒ **p = (1/24)⁴ ≈ 3.0×10⁻⁶**. Conclusive by any reasonable threshold.

**Why replication is more than 4× the confidence.** Two objections a single seed could not
answer, both closed by the replication:
1. *Luck.* At the measured 10/10 geometry power (L·T1'-3), one in-order seed is already
   unlikely by chance; four independent ones is decisive.
2. *The population-collapse confound.* Breaks 3–4 land in a cratered population, so their
   TIMES are confounded by dimer-loss. But dimer-loss is spatially UNIFORM — it lowers
   every pair's radius together, which drives edges toward dying *simultaneously*, not in
   gap-spaced order. A consistent gap-ordered cascade across four independent stochastic
   realizations is the signature of the coherence/distance mechanism specifically; uniform
   dissolution cannot fake it repeatably. So the ORDER result is clean even though the late
   TIMES are confounded (and the times were never scored anyway).

**What it establishes / does not.** Establishes: the model's entanglement partition
fragments with SPATIAL structure — far pairs decouple first — the discriminating behavior a
classical scalar trace cannot produce; this is now a conclusive result for the *model*.
Does NOT establish quantumness: this is the **(A)** reading (one shared coin per component,
classical common-cause), and per the epistemic frame the result is a discrimination win for
the theory of expected operation, not a claim about nature (the attribution gap is
untouched). Next candidate work: L·T1'-4's caveats are retired; open directions are the
small-N non-classicality witness (the **(B)** question, separate build) and folding this
into the coherence-gated-learning discrimination story.

### L·T1'-4 — T1' dynamic: far-pairs-first CONFIRMED, seed 0 · 2026-07-17  `[GROUNDED, single seed]`

*→ Completed by L·T1'-5 (4/4 replication, conclusive). Retained as the trail: the point at
which the result was a single seed (p≈0.042) and honestly only suggestive — we replicated
rather than declaring victory on one run.*

**The claim under test.** As coherence decays, d* shrinks, so the cross-synapse partition
must fragment in GAP ORDER — widest gap first. This is the discriminating claim for the
whole "topology-is-the-computation" thesis: a classical scalar eligibility trace decays
uniformly and carries no spatial structure, so it *cannot* produce a spacing-ordered
cascade (`coherence-gated-learning` primitive #1; `entanglement-topology-measurement` A7b).

**Result (seed 0).** Wide ladder, 90 s silence, dt=1e-3. All four live edges broke in the
exact pre-registered order:

| gap (µm) | broke at | population at break | regime |
|---|---|---|---|
| 3.35 | 14.5 s | 1843 dimers | healthy — clean |
| 2.90 | 32.5 s | 1043 dimers | healthy — clean |
| 2.45 | 61.5 s |  259 dimers | cratered — timing confounded |
| 2.00 | 78.0 s |   98 dimers | cratered — timing confounded |

Verdict function (guarded, CAN return INCONCLUSIVE): `far-pairs-first order CONFIRMED over
4 breaks`. Two flickers (gap3.35, gap2.45) were caught and NOT scored — the guard that the
Jul-16 false positive lacked.

**What it establishes, and what it does not.**
- Establishes: the *model's* partition fragments with spatial structure (far pairs first) —
  the thing a classical scalar trace cannot do. First non-vacuous T1' result.
- Does NOT establish: (a) conclusiveness — ONE seed, p≈1/24≈0.042 vs the classical null;
  needs a 2nd independent seed (→ p≈0.0017) to be defensible. (b) quantumness — this is the
  **(A)** reading (one shared coin per component, classical common-cause), unrelated to the
  attribution gap. Per the epistemic frame: a discrimination win for the model, not a claim
  about nature.

**The confound, handled.** Breaks 3–4 landed in a cratered population (259, 98 dimers), so
their break TIMES are confounded by dimer-loss. Their ORDER is not: dissolution is spatially
uniform, lowering every pair's radius equally, so closer pairs outlast farther ones
regardless of mechanism. Only a total collapse to zero would scramble the order. `n_dimers`
is logged as the control; **times are recorded, never scored** (see L·T1'-2 for why times
are not analytically predictable here).

**Cost note.** Ran 2.7 h, not the projected ~8 h. The entanglement tracker is O(n²) in
dimer count; the population collapsed 2223→92 over the run, so cost/step fell ~500× by the
end. The ~8 h projection extrapolated the high-population front-regime cost flat — a milder
instance of the "cost from a microbenchmark, not a profile" scar. Seed 1 will be ~2.7 h too.

**Next:** run seed 1 (independent) for p≈0.0017; then decide on 2–3 more seeds for a solid
per-seed distribution. Score the ORDER per seed.

### L·T1'-3 — The trustworthy rebuild: geometry for POWER · 2026-07-17  `[GROUNDED, 10-seed replay]`

The Jul-16 handoff §3 said put the four live gaps just under d*(0)=3.45µm (3.4/3.3/3.2/3.1)
so the cascade "lands early, while the population is >90% alive." **Both halves false —
measured, and I built to it before catching it (the grounding brief did not flag it).**

1. **Tight gaps do NOT break early.** An edge survives while ANY bonded pair clears F>0.5,
   so the governing radius is `d*_eff = 5·ln(max_pair(P_S²)/0.5)` — the extreme TAIL, not
   the median. The tail decays ~3.6× slower than the median. An 8 s sanity run: median
   radius fell below ALL four gaps (3.04µm @7.5s) while ALL four edges were still alive.
   Zero breaks in 8 s. The null killed the plan.
2. **Tight gaps destroy the order signal.** Each synapse's tail is set by its own luckiest
   dimer, whose `T_eff` is frozen at creation (`dimer_particles.py:47-50`). Between-synapse
   scatter in d*_eff ≈ 0.29µm. Rungs finer than that are decided by luck.

**Measured order-test power** (`sweep/order_power_probe.py`, 10 seeds via the validated
d*_eff replay):

| geometry | live gaps (µm) | power | vs chance (1/24) |
|---|---|---|---|
| tight  | 3.35 3.25 3.15 3.05 | 6/10 | 0.10µm rungs < 0.29µm scatter |
| medium | 3.35 3.10 2.85 2.60 | 5/10 | still under-resolved |
| WIDE   | 3.35 2.90 2.45 2.00 | **10/10** | 0.45µm rungs clear the scatter |

Counterintuitively WIDER is better (it costs more wall-clock — breaks land later — but the
order becomes resolvable). The Jul-16 2.0–3.0µm gaps were directionally closer to right;
their real flaw was the 35 s window, not the spacing. Power is what lets the verdict
FALSIFY: at 6/10 a WRONG result means nothing; at 10/10 it is real evidence against.

### L·T1'-2 — d*(0) measured; the d*_eff replay instrument · 2026-07-17  `[GROUNDED]`

**d*(0) MEASURED** (`sweep/measure_dstar0.py`), not assumed — two prior wrong claims came
from assuming P_S (the "knife-edge" off an assumed 0.90 when it is 0.998; §6.3 of the
Jul-16 handoff). Rig = confirmed static-probe rig, t=0.08 s, n≈2200: P_S median 0.9987
(min 0.9922, max 1.0) ⇒ d* min 3.387 / median 3.4521 / max 3.4657 µm. The *distribution*
(not a single number) is the fact that drives the geometry choice in L·T1'-3.

**The d*_eff replay** (`sweep/dstar_eff_replay.py`) — the reusable sizing instrument.
P_S dynamics are intra-synapse only (`step_coherence` reads local J-field + template
binding; `T_eff` fixed per dimer for life; the network never feeds back into P_S), so the
tail can be replayed in vectorised numpy with NO network — 200 s of sim in ~5 s of compute.
**VALIDATED:** median P_S matches the real 8 s rig to <0.0005. Its P_S→d*_eff mapping is
**UNVALIDATED by construction** — there is no honest cascade datum to check it against (the
only candidate, "gap 3.0 broke @34 s" from Jul-16, is the retracted FLICKER = fabricated
data). Because the replay holds the population fixed (higher max ⇒ later breaks), it emits
an **UPPER BOUND on break times**, correct for SIZING a window, never a prediction to score.

**Why the times are never scored:** they are an extreme-value statistic over a noisy random
walk (`step_coherence` multiplicative noise, ~±5.8% on P_excess over 34k steps) across
hundreds of pairs. Mis-derived THREE times on Jul-16 (median 9.5s → p95 ~13s → "ceiling"
19.3s vs observed 34.0s). **Pre-register and score the ORDER only.**

### L·dt-1 — dt convergence · 2026-07-17  `[GROUNDED]`

See the DECISION RECORD row. Instruments: `sweep/dt_convergence_drive.py`,
`sweep/dt_convergence_operating_point.py`, `sweep/dt_independence_tail.py`. Bottom line:
ORDER test valid at dt=1e-3 (P_S + edges converged); operating point ~155µM is not a dt
artifact (converged to ~5% at saturation); dt=1e-2 overflows the count — do not use.

### L·T1'-1 — T1' static half: the Werner cut IS a distance rule · 2026-07-16  `[PROVEN algebra + GROUNDED probe]`

See the DECISION RECORD row. `bond iff d < d* = 5·ln(P_product/0.5)`, confirmed 7/7 on a
pre-registered ladder (`sweep/coherence_radius_probe.py`), retrodicts the Stage 3 chain/ring
validation with no free parameters. This is what GENERATES the T1' dynamic order prediction.
Detail and the two theorems (hard floor P_S>1/√2; radius shrinks) in
`entanglement-topology-measurement` Appendix A7b.

### L·T1-RET — T1 (SOC-topology power-law) retired · 2026-07-16  `[GROUNDED structural]`

See the DECISION RECORD row. Killed on structure (1D geometric clump-size; D18 quantal
scale), not on a negative run. SOC-*chemistry* (D8/D14) is untouched and stands.

---

## Cross-references

- **Skills:** `entanglement-topology-measurement` (Appendix A = current authority),
  `model6-entanglement-partition-werner` (the LOCKED partition + Werner bound),
  `quantum-computation-and-attribution` (the A/B fork, the attribution gap, the
  discrimination discipline this log's epistemic frame rests on),
  `coherence-gated-learning` (why the topology-as-trace claim is discriminating).
- **Sibling log:** `RESEARCH_LOG_CALCIUM_DIMER.md` (the chemistry sub-program, D1–D18).
- **Handoffs:** `docs/handoffs/SESSION_HANDOFF_JUL17_T1PRIME_REDESIGN.md` (this session's
  thread baton), `SESSION_HANDOFF_JUL16_TOPOLOGY_DT_FIX.md` (the failed run + dt fix).
- **Code:** `sweep/coherence_fragmentation_probe.py` (T1' dynamic),
  `sweep/coherence_radius_probe.py` (T1' static, CONFIRMED 7/7),
  `sweep/{measure_dstar0,dstar_eff_replay,order_power_probe,dt_*}.py` (instruments).
