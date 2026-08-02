# Session Handoff — 2026-07-27 (for a fresh thread)

**One line:** PO-11 valence is **DONE and positive** (the "to what end" bridge, weight-level); the **maze
loop / #2** is the live open thread (does the loop learn *via the structured quantum partition* or
classically); the **cross-domain EFE spec** for TALON is written and persisted. Branch
`claude/trusting-heyrovsky-1338e9` is pushed to origin — everything below is backed up. Nothing is running.

## Read-order (ground first, always)
1. `session-discipline` + `agent-grounding-protocol` (in full).
2. Quantum threads: `quantum-computation-and-attribution` (the A/B fork, attribution gap), `quantum-system-canonical`
   (§5 what-the-computation-is, §4.4 commitment), `coherence-gated-learning` (the four primitives).
3. Loop / #2: `spatial-discovery-experiment`, and research-log entries **D19/D20/D21** (`RESEARCH_LOG_CALCIUM_DIMER.md`)
   — but VERIFY against current code, they are the pre-July-18 *diagnosis* and the loop was rewritten same-day.
4. EFE / cross-domain: `talon-architectural-north-star` (§"active-inference reach"), `cross-domain-integration`.
5. The two live docs: `PREREG_PO11_VALENCE.md`, `CROSSDOMAIN_SPEC_EFE_SHEAF_OBJECTIVE.md`.

---

## ① PO-11 VALENCE — DONE ✓ (positive, complete ladder)

**Claim proven (weight-readout level):** with a reward-fixed (directional) sign, partition-specific credit is
recoverable from Δw and **leak-immune** — reward can direct the plasticity onto the *right* groups, provably not
via the abundance/drive-timing confound. This is the "to what end" that turns the partition-computation into
directed learning.

**Result (fixed-sign, registered readout = within-condition partial correlation):**
`full` (n=20) RECOVERS (sep +0.707, null-p 0.000); `bindoff`/`scramble`/`λ-short` **all at chance**; magnitude
reference **leaks on every arm** (reads abundance, not the partition). Recorded in `PREREG_PO11_VALENCE.md`
(RESULT + COMPLETENESS sections). **Scope: weight-READOUT only — NOT a claim the closed loop learns** (that's #2).

**Artifacts:** prereg `docs/PREREG_PO11_VALENCE.md`; scorer `sweep/po11_valence_score.py`; synthetic gate
`sweep/po11_valence_synthetic_check.py`; data `results/po10_unitC/*val_*` (fixed-sign, force-added). Key method
note: the leak-immune readout is **within-condition** partial correlation (pooled leaks); P2 (synthetic) and P3
(real-data leak reproduction) both discharged before physics.

**Orphan caught (don't rebuild on it):** `apply_reward_correlated` (`multi_synapse_network.py:1611`) is a dead
stub (zero call sites; writes `_committed_memory_level`, not a loop-closing weight). The valence readout is built
**off-path** in the harness, not by calling it.

---

## ② MAZE LOOP / #2 — LIVE, the main open thread

**The question:** does the spatial-discovery loop learn *via the structured quantum partition*, or is it
classical spine-growth that happens to be goal-directed? This is an **attribution/discrimination** problem
(realism + discrimination + convergence — you must show it learns *differently because of* the partition).

**Audit state (grounded 2026-07-27):** the loop is architecturally complete — the July-18 rewrite fixed the
D19/D20/D21 findings (measurement per-trial `92c623f`; pump ignites `ed43838`; plasticity clock `7b05153`;
`coupling_weights` passed in `run_trial` `sweep/run_spatial_discovery.py:220`). At realistic scale (15 features,
60s) the loop **engages** (thousands of dimers) and shows a **goal-directed signal** (corr(spine,goal_dist)
trends negative across trials; goals reached increasingly). BUT: the partition looks like a **blob** (millions of
Werner cross-bonds over ~15 synapses), and the growth may be **classical**.

**The precise mechanism (Sarah's correction, verified in `spine_plasticity_module.py`):** commitment does NOT
directly change the spine. Spine volume = actin content (`V ∝ actin^1.2`, `:161`). The actin cascade has **two
arms**: **calcium → polymerization** (transient GROWTH, `:91`; runs without commitment — "a never-committed
synapse grew") and **CaMKII/commitment → confinement/stabilization** (CONSOLIDATION/durability — which spines
lock in, `:98`/`:106`/`:167`). "Commitment buys durability, not amplitude."

**The plan (3 steps, sharpened by the two-arm cascade):**
- **Step 1 (gate + tractability unlock):** is the partition STRUCTURED or a BLOB, and why? Measure the
  structure-vs-dimer-count curve (largest-component fraction; goal-assortativity). If structured-at-low-dimers →
  swamped-at-high (birth-inheritance, PO5-2: ~83% of bonds bond every template-bound dimer within 100ms) →
  **fixable** (read early / decay-clean à la Unit C's delay-prune → sparse ⇒ informative AND tractable). If
  blob at all dimer counts → **irreducible** (classical; (A) reading stands; pivot).
  - **Step-1 attempt 1 was INCONCLUSIVE** (`scratchpad/step1_partition_structure.py`): fresh networks + single
    trials that never reached the goal → dimers ~0 → no bonds. **Redesign to guarantee engagement**: accumulate
    across trials (like the b_diag run), or force goal-reaching, or a controlled co-active drive.
  - **Step-1 RESOLVED 2026-07-28 → STRUCTURED / FIXABLE** (research-log `STEP1`; prereg
    `docs/PREREG_STEP1_PARTITION_STRUCTURE.md`; `sweep/step1_partition_structure_probe.py`;
    `results/step1_partition/step1_result.json`). Redesign chosen = **controlled co-active drive** with a
    **controlled engagement IC** (`E_invasion` clamped — the diagnosis of why attempt-1 got 0 dimers was the
    η-gate, not goal-reaching: the live drive never crosses `invasion_threshold`, GAP-2). n=8 interleaved
    C/G at matched distances; only co-activation TIMING varies (SYNC vs STAGGER); 5 seeds. **At matched
    dimer count (800–1600), SYNC blobs (cpef 0.57–0.73, largest 1.0) while STAGGER excludes off-phase
    synapses (cpef 0.000, largest 0.25–0.50)** — the partition tracks input timing, not just geometry.
    STAGGER structure SWAMPS to a blob at **~2200 dimers** (cpef 0.000→0.667). So the partition IS
    input-dependent (refutes the "complete graph regardless of input" fear) but only below the swamp
    threshold → **Step 2 (P31/P32) is worth the compute, conditioned on keeping loop dimers < ~2200
    (read-early/decay-clean) and driving temporally-separated feature activation.** LIMIT: controlled probe,
    not the live nav loop; and Step-1 clears only the PREREQUISITE — it does NOT show quantumness (a classical
    coincidence trace behaves identically; that is exactly what Step 2's isotope test is for).
- **Step 2 (discrimination — the non-circular isotope test):** P31 vs P32 (`fraction_P31`, live in the loop via
  `make_network`/`run_spatial_discovery.py:148`) on **selective consolidation** = cross-trial persistence of
  goal-directed spine volume, NOT transient growth. Non-circular because the arms separate: calcium-growth
  survives P32 by construction; only partition-coordinated consolidation is at stake. Classical predicts
  P32≈P31; partition-driven predicts P31≫P32 on *selective* consolidation.
- **Step 2 RESULT 2026-07-30 → leak-immune selective consolidation NOT demonstrated (real downstream washout)**
  (research-log `STEP2`; prereg `docs/PREREG_STEP2_SELECTIVE_CONSOLIDATION.md`;
  `sweep/step2_selective_consolidation_probe.py`; `results/step2_consolidation/` 48 draws). Positive control
  held (P31 structured {AB}|{CD} + 6.9/8 commit; P32 killed + 2.1/8). But the registered leak-immune readout is
  at CHANCE for P31 (sep +0.019, null-p 0.725); only the abundance LEAK separates P31/P32. Collapse-timing check
  (`scratchpad/step2_collapse_timing.py`) FALSIFIED the "burst blobbed the partition" guess: the measurement fires
  at reward step 0 on the CLEAN {AB}|{CD} partition and it stays structured through the burst (reward-phase η=0 →
  no new cross-bonds). So the partition is measured correctly; the credit is lost DOWNSTREAM — commitment is
  calcium-dominated (~7/8 fire regardless of the per-component coin), so durable consolidation tracks abundance,
  not the partition. Dimer burst root-caused too: `plateau_potential`→−20 mV (correct 2026-07-18 physics, not a bug).
- **Step 3 (NOW THE CONFIRMED NEXT BUILD — the payoff, precisely located by Step 2):** wire the PO-11 valence into
  the **consolidation arm** so the (correctly-measured) partition's per-component credit actually gates *which*
  spines durably lock in — i.e. close the measurement→commitment coupling that calcium currently overwhelms.
  **DISCIPLINE FLAG:** this is where tuning-to-result lurks (`quantum-computation-and-attribution` §6.1) — any
  coupling must be principled/literature-grounded (the candidate is the correlated calcium-return from
  per-component dissolution, which EXISTS but is drowned by baseline calcium), NOT manufactured to force P31≫P32.
  The honest alternative outcome — that the model's durable consolidation is genuinely calcium-dominated and the
  partition does not drive learning — must remain on the table.

**GOTCHA — the loop is compute-hostile:** the blob is the O(n²) entanglement-tracker explosion. The b_diag run
(20 trials, 15 features) ran **8h+ and never finished**. Step 1's sparsification is a *prerequisite* for Step 2's
feasibility. Keep loop probes short and daemonized.

**Loop harness:** `sweep/run_spatial_discovery.py` (at REPO-ROOT `sweep/`, not Model_6/sweep). Key functions
`make_network`, `run_trial`, `run_experiment`. Instrumented probes this session (scratchpad, transient):
`loop_observe.py`, `b_diag_learning.py`, `step1_partition_structure.py`.

---

## ③ CROSS-DOMAIN EFE SPEC (TALON) — DONE ✓ (spec), TALON building

**What:** the active-inference / expected-free-energy objective over the sheaf substrate — the SAME "objective
layer" as #2, one level up. `CROSSDOMAIN_SPEC_EFE_SHEAF_OBJECTIVE.md`. Key results:
- **Q(s) = 𝒩(Λ⁻¹h, Λ⁻¹), Λ = L_F(W) + Σ_ℓ β_ℓ T^ℓ + εI.** L_F (the sheaf Laplacian TALON already computes in
  `consistency_energy`) IS the GMRF precision — obstruction is the prior precision, `w_e`=1/σ_e² are the couplings.
- **β_ℓ calibration closed form (§8):** β_ℓ = 1/reduced-χ² per layer; online EWMA plugs into the Layer-F loop.
- Pragmatic = expected H⁰ obstruction resolved; epistemic = −½log(1+τΣ_vv); policy = argmin G; stop when EFE
  stops dropping. `R_{e,v}` restriction-map form given (linear = transform matrix, free; non-linear = Jacobian).
- **PO-11 is the empirical validation:** within-condition partial correlation = precision off-diagonal — the
  guarantee the coupling block of Q(s) needs.

**Boundary:** math/framing owned in the posner repo; **TALON builds/grounds Q(s) against the murmur substrate
(not visible from here).** TALON-side (Sarah, 2026-07-27): first build = single-layer Q(s) (only `T_stat` live;
ts/struct/sem precisions + the outcome stream `investigations.outcome 0/415` are the single critical path).
**Open next:** (b) Takahashi selected-inverse (scale); the permutation-engine↔EFE handoff (covering-array =
cold-start candidate generator, EFE = warm adaptive selector); multi-layer fusion.

---

## Operational state
- **Branch `claude/trusting-heyrovsky-1338e9` pushed to origin** (`git@github.com:Sarah-in-dev/...`). All work +
  data backed up. Nothing running.
- **Teardown-robust runs:** `run_in_background` dies on Claude teardown (lost ~5h once). Long batches must
  self-daemonize (Python double-fork + `os.setsid`; macOS has NO `setsid` command). Verify ppid=1; poll (no
  auto-notify). Memory: `long-batch-teardown-robust-launch`. Reference daemons in scratchpad.
- **Compute reality:** ~22 min/draw for Unit-C-class physics, concurrency ≤4 (memory-tight, jetsam). The loop is
  far worse (O(n²) blob).

## First move for the fresh thread
Ground per read-order, then pick up **#2**: redesign the Step-1 partition-structure probe to guarantee substrate
engagement (accumulate across trials / force goal-reaching), and get the structure-vs-dimer-count verdict —
fixable-blob vs irreducible-blob. That single result decides whether Step 2 (the P31/P32 selective-consolidation
discrimination) is worth the loop compute. #2 is the hinge for whether the whole program's "quantum-driven
learning" claim lands or resolves to the classical (A) reading.
