# Session Handoff: From Abstract Sweep to Place Field Learning

**Date:** April 7, 2026
**Session focus:** Correcting bond tracking bugs, implementing per-cluster quantum measurement, Hill-function CaMKII drive, calcium feedback loop, and building the first concrete learning task.

---

## What We Accomplished

### 1. Realistic 45s Inter-Traversal Test (Corrected)

Ran the critical test from the April 6 handoff: do dimers survive 45s gaps between traversals?

**Three bugs found and fixed in the bond tracking system:**

**Bug 1: Analytical gap didn't clean network tracker.** The analytical gap code dissolved dimers from per-synapse `dimer_particles.dimers` but never updated the `NetworkEntanglementTracker`, which maintains its own separate bond graph (`self.all_dimers`, `self.entanglement_bonds`). Result: stale bonds from dissolved dimers persisted in all network-level metrics.

**Bug 2: `get_experimental_metrics()` read from stale cache.** `_network_entanglement` was last set during full-physics steps; analytical gaps never refreshed it.

**Bug 3: Sequential `global_id` reassignment corrupted bonds after dissolution.** `collect_dimers()` assigned sequential IDs (0, 1, 2...) from scratch every call. After dissolving 30% of dimers, surviving dimers got new sequential IDs, but old bond tuples like (5, 23) now pointed to different physical dimers. The prune step kept these "phantom bonds" because both IDs existed in the new space — they just referred to different dimers.

**Fix:** Changed `NetworkEntanglementTracker` to use stable `dimer.id` (assigned at creation) as bond graph keys instead of sequential `global_id`. Now `collect_dimers()` preserves identity, and the prune step in `_update_entanglement()` correctly removes bonds for dissolved dimers.

**Corrected results at 45s intervals:**
- Dimers oscillate: ~111 post-traversal → ~79 post-gap (~30% dissolution per gap)
- Bonds do NOT accumulate monotonically — each traversal rebuilds to ~90-160, each gap destroys ~50-70%
- The network is in a build-destroy-rebuild cycle, not steady accumulation
- But the surviving dimer core provides scaffold for faster re-entanglement each lap
- Commitment fires after dopamine at traversal 5 peak (131 dimers, 159 bonds)
- Commitment timing is correct: fires at the dopamine step (peak state), not during post-silence

### 2. Inter-Traversal Interval Sweep (Corrected Bonds)

Ran intervals [10, 15, 30, 45, 60, 90] seconds, 6 traversals each:

| gap_s | surv% | clusters | t5_bonds | mean_drive |
|-------|-------|----------|----------|------------|
| 10    | 96%   | 1        | 774      | 0.941      |
| 15    | 93%   | 1        | 446      | 0.903      |
| 30    | 81%   | 4        | 194      | 0.709      |
| 45    | 67%   | 4        | 106      | 0.560      |
| 60    | 56%   | 6        | 103      | 0.474      |
| 90    | 29%   | 7        | 96       | 0.390      |

**Key insight:** Two regimes visible. At 10-15s (>93% survival), one connected cluster, uniform strong drive. At 60-90s (<56% survival), 6-7 fragmented clusters, weak patchy drive with high per-synapse variance (0.04-0.68 at 90s).

### 3. Per-Cluster Quantum Measurement (New Architecture)

**Old:** Single coin flip for entire entangled network. 73-dimer and 198-dimer clusters produced identical plasticity (fraction = 0 or 1).

**New:** Each connected component gets independent measurement:
- `_find_all_clusters()` — union-find returns List[set] of connected components
- Each cluster: `cluster_P_S = mean(P_S)`, independent flip at that probability
- Returns absolute committed dimer COUNT per synapse (not fraction)
- Fragmented networks (many small clusters) get many independent measurements — some commit, others don't → spatial pattern of commitment

### 4. Hill Function CaMKII Drive (New)

Committed dimer count → plasticity drive via Hill function matching CaMKII cooperative activation:

```python
HILL_N = 4        # CaMKII-like cooperativity
HILL_K_HALF = 20  # Half-max at 20 committed dimers per synapse

drive = count**4 / (20**4 + count**4)
```

Reference: count=5 → drive=0.0004, count=20 → 0.5, count=40 → 0.95.

This creates a 28× difference between the 10s condition (~40 dimers/synapse, drive≈0.95) and 90s condition (~15 dimers/synapse, drive≈0.034).

### 5. Spine-Calcium Feedback Loop (New)

**The problem:** Prior learning (spine enlargement, AMPAR increase) had no effect on subsequent calcium dynamics. A 3× enlarged spine produced identical calcium to a naive spine.

**Two feedback pathways implemented:**

1. **AMPAR→voltage gain:** `v_eff = resting + (voltage - resting) * (AMPAR_count / 80.0)`. More AMPARs → larger EPSP → more VGCC opening.
2. **Spine volume→channel gain:** `channel_gain = spine_volume ** 0.67`. Surface area scaling gives larger spines more effective calcium channels.

Combined at baseline: no change. Post-potentiation (108 AMPAR, vol=1.5): ~77% more calcium.

**Feature flag:** `params.spine_calcium_feedback = True/False` — enables toggling for comparison runs.

### 6. Place Field Learning Task (First Run)

**Design:** Linear track with 5 synapses representing 5 locations. Sequential activation (syn0 at t=0-1s, syn1 at t=1-2s, etc.). Dopamine at t=3.5s (midway through syn3's field). 10 traversals, 20s inter-traversal gaps. Feedback loop ON.

**Results (6 traversals completed before timeout):**

Committed dimer counts by traversal:

| Trav | syn0 | syn1 | syn2 | syn3 (reward) | syn4 |
|------|------|------|------|---------------|------|
| 0    | 13   | 13   | 12   | 3             | 0    |
| 1    | 22   | 21   | 23   | 13            | 13   |
| 2    | 30   | 35   | 45   | 17            | 19   |
| 3    | 51   | 50   | 53   | 20            | 14   |
| 4    | 69   | 93   | 79   | 26            | 15   |
| 5    | 105  | 133  | 105  | 44            | 20   |

Hill drive by traversal:

| Trav | syn0  | syn1  | syn2  | syn3  | syn4  |
|------|-------|-------|-------|-------|-------|
| 0    | 0.151 | 0.151 | 0.115 | 0.001 | 0.000 |
| 3    | 0.977 | 0.975 | 0.980 | 0.500 | 0.194 |
| 5    | 0.999 | 0.999 | 0.999 | 0.959 | 0.500 |

Spine volume (by traversal 5 post-gap): syn0=2.596, syn1=2.829, syn2=2.891, syn3=2.071, syn4=1.580

**Key findings:**
- **Temporal credit assignment WORKS** — graded potentiation based on timing relative to dopamine
- **The gradient is backward-looking** — syn0-2 (active before reward) stronger than syn3 (active during reward). This matches BTSP's asymmetric, predictive plasticity window.
- **Syn3 is weaker than syn0-2** because dopamine arrives mid-burst, giving it only 0.5s of dimer accumulation vs 1.0s for earlier synapses
- **Syn4 gets partial credit** from residual dimers surviving from prior traversal — the long-timescale eligibility trace
- **Feedback loop compounds** — total dimers 72→107→149→195→299→427 across traversals (superlinear growth, possibly runaway)
- **AMPAR unchanged** — 1800s onset delay means AMPAR trafficking hasn't started in the ~160s of simulation

---

## What Needs Doing Next

### Priority 1: Comparison Run (Feedback OFF)
Run same place field task with `params.spine_calcium_feedback = False`. Does the credit assignment gradient emerge from coherence physics alone, or does it require the feedback amplification? This is the critical control.

### Priority 2: Phosphate Pool Check
At 427 dimers across 5 synapses (4 phosphate per dimer = 1708 consumed), is the pool actually limiting? If dissolved dimers return phosphate fast enough to fuel unlimited growth, the runaway is a missing cap, not a physics result. Print phosphate remaining at each traversal.

### Priority 3: Dopamine Timing Fix
Move dopamine from t=3.5s (mid-field for syn3) to t=4.5s (0.5s after syn3's field ends). This matches biological dopamine delay and gives syn3 its full burst before measurement. Expected: syn3 becomes strongest, with graded falloff syn2 > syn1 > syn0 > syn4.

### Priority 4: Classical STDP Comparison
Run with a classical eligibility trace (exponential decay, τ=2s) instead of quantum coherence. The classical model should produce a much narrower credit assignment window — only syn2-3 get credit, not syn0-4. This demonstrates the quantum system's broader temporal integration.

### Priority 5: Performance Optimization
427 dimers → 486s per traversal (O(n²) bonds). Options:
- Skip bond recalc during intra-traversal silence periods (between synapse activations)
- Reduce entanglement tracker update frequency during traversals
- Cap dimer count or add spine volume ceiling to prevent runaway

### Priority 6: Multi-Session Memory Persistence
After learning, simulate protein turnover (CaMKII self-maintenance) and present the track again. Does the strengthened synapse produce more dimers on the next session? This demonstrates that the memory IS the next computation.

---

## Connection to BTSP Literature

The place field learning results directly map to Behavioral Timescale Synaptic Plasticity (BTSP), discovered by Bittner/Magee 2017. Key correspondences:

| BTSP Property | Our Model |
|--------------|-----------|
| One-shot place field formation | Graded potentiation in single traversal |
| Seconds-long eligibility trace | Dimer coherence window (~100s for P31) |
| Backward-looking asymmetry | Syn0-2 stronger than syn3 (pre-reward > reward location) |
| Plateau potential as instructive signal | Dopamine gate triggers quantum measurement |
| CaMKII required (Xiao 2023) | Hill-function CaMKII drive from committed dimer count |
| DDSC: delayed, stochastic CaMKII (Jain/Yasuda, Nature 2024) | Measurement is probabilistic, CaMKII activation depends on dimer count at measurement time, not at formation time |
| Eligibility trace mechanism UNKNOWN | Our model: dimer coherence IS the trace |

The critical finding from Yasuda lab (Nature 2024): CaMKII activation during BTSP is "dendritic, delayed, and stochastic" — occurring 10-100s after induction, requiring both pre- and postsynaptic activity, facilitated by IP3-dependent Ca²⁺ release. Nobody knows what maintains the eligibility trace during the 10-100s gap. Our model proposes: calcium phosphate dimer nuclear spin coherence.

## Three Things Biology Does That ML Doesn't (Articulated This Session)

1. **Learning and inference in single pass** — coherence window simultaneously computes (integrates evidence), learns (sets eligibility), and stores (measurement outcome persists via CaMKII). ML separates forward pass, backprop, and weight update.

2. **Self-maintaining memory** — CaMKII autophosphorylation + GluN2B binding creates a self-reinforcing state that survives protein turnover. The memory actively maintains itself. ML weights persist only because hardware persists.

3. **Storage IS computation** — strengthened spine (from prior learning) directly changes calcium dynamics, dimer formation, and eligibility trace of the next experience. The weight IS the physical structure IS the next computation. ML reads weights from a matrix and multiplies.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `multi_synapse_network.py` | `_find_all_clusters()` helper; `perform_quantum_measurement()` per-cluster with counts; `perform_independent_measurement()` returns counts; Hill function `_committed_count_to_drive()`; stable `dimer.id` in entanglement tracker replacing sequential `global_id`; gate logic uses count > 0 + Hill drive |
| `model6_parameters.py` | Added `spine_calcium_feedback: bool = False` feature flag |
| `model6_core.py` | AMPAR→voltage gain and spine volume→channel gain feedback pathways, gated by `spine_calcium_feedback` flag |
| `analytical_calcium_system.py` | `channel_gain` attribute, applied to channel currents |

## Files Created This Session

| File | Purpose |
|------|---------|
| `sweep/run_theta_burst_45s.py` | Interval sweep with analytical gaps, per-phase timing, corrected bond tracking |
| `sweep/run_place_field_learning.py` | Linear track place field learning experiment with sequential synapse activation |

---

## Key Parameters & Architecture Reference

### Hill Function (CaMKII Drive)
- `HILL_N = 4` (cooperative exponent)
- `HILL_K_HALF = 20` (half-max at 20 committed dimers per synapse)
- Not yet literature-grounded for this specific context — CaMKII's Hill coefficient for Ca²⁺ is ~8 (Bhatt 2024), but our Hill function is on dimer count, not calcium concentration

### Feedback Loop Gains (at various spine states)
| State | AMPAR | Spine vol | v_gain | ch_gain | Combined Ca²⁺ |
|-------|-------|-----------|--------|---------|----------------|
| Baseline | 80 | 1.0 | 1.0× | 1.0× | 1.0× |
| Moderate | 90 | 1.5 | 1.13× | 1.31× | ~1.5× |
| Strong | 108 | 2.5 | 1.35× | 1.84× | ~2.5× |
| Max | 108 | 3.0 | 1.35× | 2.08× | ~2.8× |

### Analytical Gap Implementation
During inter-traversal gaps:
1. Decay P_S per dimer: `exp(-gap_duration / T2)`
2. Compute dissolution probability using coherence-dependent rate
3. Remove dissolved dimers, return phosphate and calcium
4. Call `entanglement_tracker.collect_dimers()` + `_update_entanglement()` to prune stale bonds
5. One full network step to sync state
6. Step spine_plasticity with gap duration using current _committed_memory_level as drive

### Measurement Architecture
- Per connected component (not per whole network)
- Each component: independent flip at component's mean P_S
- Returns absolute count of committed dimers per synapse
- Count → drive via Hill function
- Drive feeds spine_plasticity as structural_drive

### Place Field Learning Protocol
- 5 synapses, sequential 1s activation windows
- Theta burst within each window (4 spikes at 100Hz, 8Hz theta)
- Only active synapse gets -10mV; others at -70mV resting
- Dopamine at t=3.5s (should be moved to t=4.5s)
- 20s inter-traversal gaps (analytical)
- `_camkii_committed` and `network_committed` reset before each traversal
- Spine volume, AMPAR, actin persist across traversals (that's the memory)