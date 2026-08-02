# PRE-REGISTRATION — Step-2: P31/P32 selective-consolidation discrimination

> **CORRECTION 2026-07-30 (research-log `ISO-1`):** this document's framing of P31/P32 as "the
> non-circular isotope test" is **WRONG** and is retracted. The isotope difference is implemented as a
> hardcoded coherence-time swap (`T_singlet` 216 s ↔ 0.4 s), so P32's decoherence is an *input, not a
> derivation* — the "P32 kills it" arm is **circular**. ³²P is also the wrong lever (the real one is
> ⁶Li/⁷Li or Ca/O) and physically un-doable (radioactive). **What survives:** Step 2's *core negative*
> (leak-immune credit lost downstream) rests on the **P31 arm + collapse-timing** and is isotope-independent,
> so it stands. The isotope AXIS of this prereg does not. See `ISO-1` and tasks #8/#9.

**Written 2026-07-30, BEFORE the scored sweep.** Thread #2, Step 2: the non-circular isotope test.
Step 1 (STRUCTURED/FIXABLE) established the partition CAN carry input structure; Step 2 asks whether
that partition **drives selective, durable consolidation** — and whether swapping P31→P32 (which kills
the partition but leaves the chemistry identical) kills the effect. This is the discrimination that
separates "quantum-partition-driven learning" from the classical (A) reading.

## The claim under test
The entanglement partition coordinates which synapses **consolidate together**: quantum measurement
assigns one commit-coin per connected component (`perform_quantum_measurement` / `_find_all_clusters`),
so co-active clusters sharing a component get correlated commitment → correlated durable consolidation.
- **Partition-driven prediction:** under **P31**, per-cluster durable consolidation RECOVERS the
  co-active pairing (co-active clusters correlated); under **P32** (no partition) it is at chance.
- **Classical prediction:** P31 ≈ P32 (consolidation is calcium-driven, partition-independent).

Non-circular because calcium-driven transient growth is isotope-independent **by construction**
(`dimer_particles.py:404` blends T_singlet; P32 only collapses P_S → the partition). Only
partition-coordinated consolidation is at stake.

## Observable (per Sarah's caution — LOAD-BEARING)
Per-cluster **`actin_stable`** (the "consolidation marker" pool), read **POST-GAP**, baseline-subtracted
(Δ over the reward). NOT total `spine_volume`: because the cascade maps calcium→growth AND
commitment→confinement→durability, total volume conflates the isotope-INDEPENDENT growth arm and would
mask the effect. `actin_stable` grows only via `retention = k_stab·confinement·enlargement`, so it is
the commitment-gated durable pool.

## Design (controlled probe — validated 2026-07-30, NOT the live nav loop)
- 8 synapses = 4 clusters (A,B,C,D) × 2, linear @0.5µm, λ_F=214 (distance made irrelevant; structure
  comes from co-activation timing), `use_correlated_sampling=True` (the COORDINATED gate).
- **Engagement = burst-gated η** (`set_backbone_condensation_eta` ON for active synapses only) — the
  transparent form of Step-1's mechanism; keeps the partition STRUCTURED ({AB}|{CD}) instead of a blob.
  `actin_enlargement` left NATURAL so `actin_stable` is not confounded by the clamp.
- Staggered write (two co-active pairs in separate windows) → dopamine (reward + plateau, coordinated
  gate → measurement) → 6 s CaMKII integration → 30 s consolidation gap → read `actin_stable`.
  Integration is stepped WITHOUT the network entanglement tracker (measurement already fired at reward;
  performance only, physics unchanged — ~2× faster at peak dimers).
- **Two pairings** (`pair1`={AB},{CD}; `pair2`={AC},{BD}) and **counterbalanced order** (fwd/rev) —
  the pairing is the decoder label; order removes the early-pair positional confound (Unit C Amendment 1).
- Per draw record (PO-11-scorer compatible): `dw_cluster` = per-cluster Δ`actin_stable`, `mode`, `order`,
  `partition_edges`, `n_committed`, `peak_dimers`. Free (unseeded) draws → stochastic variance for the
  correlation readout.

**Validation (single draws, recorded before the sweep):** P31 → `partition_edges=[[0,1],[2,3]]`
(structured {AB}|{CD}), 7–8/8 committed, AB Δstable > CD; P32 → `partition_edges=[]` (partition killed),
2/8 committed, no AB|CD structure. Peak dimers ~700–1000 (under the Step-1 swamp). ~47 min/P31 draw.

## Scorer (reused as-is)
`sweep/po11_valence_score.py` — within-condition partial correlation of `dw_cluster`, separation =
within-pair minus cross-pair, vs a label-shuffle null. Run per isotope arm:
`--glob 'results/step2_consolidation/P31_*'` and `'P32_*'`.

## PRE-REGISTERED VERDICT (thresholds fixed before the scored run)
1. **PARTITION-DRIVEN SELECTIVE CONSOLIDATION (Step-2 POSITIVE)** iff:
   - **P31** within-condition partial correlation **RECOVERS**: separation > 0 with **null-p < 0.05**
     AND correct pairing decode, AND
   - **P32** is at **chance** (null-p ≥ 0.05 / separation ≈ 0).
   ⇒ the quantum partition drives directed learning; the program's central claim lands at the
   consolidation level.
2. **CLASSICAL / no discrimination** iff P31 and P32 both recover (or both at chance) — consolidation
   does not depend on the partition. ⇒ the (A) reading stands at the learning level.
3. Positive control (must hold): P31 forms structured partitions (`partition_edges` non-empty, within-pair)
   and P32 does not — else the isotope lever is not acting and the run is void.

## Sweep & compute
2 isotopes × 2 pairings × 2 orders × **6 draws** = 48 draws. Run as **4 independent daemons** (one per
(mode,order) cell, each doing P32 then P31), concurrency 4, checkpointed per-cell JSONL, under
`caffeinate` (idle-sleep held off). ~5 hours wall. Partial results survive interruption.

## Limits (stated, not oversold)
- Controlled probe (burst-gated engagement, controlled geometry) — not the live nav loop; shows the
  partition CAN drive selective consolidation, not that the loop as-run does.
- The reward pulse enlarges the dimer population (~8k) at measurement, so the measured partition is
  somewhat larger than the end-of-write {AB}|{CD}; the readout partition (`partition_edges`) is recorded
  at end-of-write. A cleaner measurement-time partition is a follow-on.
- Even a POSITIVE result is Model-6-internal (simulated). Per attribution §5, it strengthens the
  discrimination case but does not measure nature.

## Artifacts
- `sweep/step2_selective_consolidation_probe.py`; scorer `sweep/po11_valence_score.py`;
  data `results/step2_consolidation/*.jsonl` (force-add for provenance).
