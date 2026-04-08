# Session Handoff: Entanglement Topology + Multi-Traversal Scenario

**Date:** April 6, 2026 (evening session)
**Session focus:** Making the computational primitive visible — surfacing entanglement topology through the sweep, fixing the multi-traversal scenario, and correcting the commitment gate.

---

## What We Accomplished

### 1. Entanglement Topology as the Core Observable (Karim's Suggestion)

Karim suggested thinking about latent variables. We identified the entanglement topology — the graph structure of bonded dimers across synapses — as the latent variable that IS the computational content. Scalar outputs (dimer count, field strength, commit yes/no) are projections that lose everything the primitive computes. The topology must be tracked and extracted.

**Key insight:** Two sweep vectors can produce identical dimer counts but encode completely different temporal histories in their entanglement structure. Without topology metrics, the sweep maps shadows.

### 2. Topology Metrics Surfaced (Already Computed, Were Being Dropped)

Discovered that `multi_synapse_network.get_experimental_metrics()` already computes rich topology data — but `theta_burst_scenario.extract_metrics()` was discarding all of it, keeping only mean_field_kT, committed, and commit_level.

**Fixed in theta_burst_scenario.py:** Now passes through all 9 topology fields:
- n_entangled_network, within_synapse_bonds, cross_synapse_bonds, total_bonds
- q2_field_kT, mean_coherence, total_dimers
- n_clusters, mean_connectivity (new — propagated up from dimer_particles)

**Fixed in multi_synapse_network.py:** Added n_clusters and mean_connectivity to get_experimental_metrics().

### 3. Multi-Traversal Scenario Built

Complete rewrite of `theta_burst_scenario.py` from single-burst to multi-traversal CA1 place field stimulation.

**New parameters:**
- `theta_cycles_per_traversal` (replaces old n_bursts, default 12, range 8-16)
- `n_traversals` (default 6, range 1-10)
- `inter_traversal_interval_s` (default 45, range 30-90)
- `silence_duration_s` — now means final post-last-traversal silence only
- `dopamine_after_traversal` — which traversal triggers the single dopamine pulse (default: last)

**Per-traversal snapshots:** Topology metrics extracted after EACH traversal, stored as `traversal_snapshots` list. Commitment snapshot captured at the moment commitment fires.

**Sweep dimensions updated** in quantum_dimensions.py: stim_theta_cycles [8,12,16], stim_n_traversals [1,3,6,10], stim_inter_traversal_s [30,45,60,90].

### 4. Calcium Injection Bug Found and Fixed

**The bug:** Scenario was writing calcium directly to `syn.calcium._ca_field`, but `analytical_calcium_system.step()` calls `self._ca_field.fill(baseline)` on every timestep — wiping the injection immediately. Meanwhile voltage was always -70 mV (resting), so VGCCs never opened.

**The fix:** Drive voltage to -10 mV during burst timesteps (2ms depolarization + 8ms rest per spike within each theta cycle), matching how the working isotope/pharmacology experiments deliver calcium. Channels open, calcium enters through nanodomains, persists through channel physics. Removed direct field injection.

### 5. Dual Commitment Pathway Bug Found and Fixed

**The bug:** Two independent commitment systems existed:
- `_check_network_commitment()` — "auto-commitment" running every timestep, no dopamine required, just field ≥ 20 kT and eligibility > 0.3. This was firing at traversal 0 with only 3 entangled dimers.
- Three-factor AND gate — proper biological gate requiring dopamine + calcium + quantum eligibility. Only evaluated on reward timesteps.

**Fixed:** Set `network.disable_auto_commitment = True` in the scenario. Added propagation from synapse-level `_camkii_committed` up to `network.network_committed` after three-factor gate fires.

### 6. Dopamine Timing Fixed

**The bug:** Dopamine was being delivered after every traversal (0.5s post each one), causing commitment on the first dopamine pulse before evidence accumulated.

**Fixed:** Single dopamine pulse, configurable via `dopamine_after_traversal` parameter. Default: after the last traversal. Sweep can test dopamine after traversal 1 vs 3 vs 6 vs 10 to map how much accumulation is needed.

### 7. First Successful Accumulation Curve

10 traversals, 3s inter-traversal intervals (compressed for testing), dopamine after last traversal:

| Traversal | Dimers | Entangled | Bonds | Cross-Bonds | P_S   | Committed |
|-----------|--------|-----------|-------|-------------|-------|-----------|
| 0         | 29     | 2         | 4     | 1           | 0.994 | False     |
| 1         | 54     | 35        | 40    | 12          | 0.985 | False     |
| 2         | 76     | 68        | 200   | 63          | 0.980 | False     |
| 5         | 113    | 112       | 1231  | 472         | 0.966 | False     |
| 9         | 136    | 133       | 2935  | 1250        | 0.962 | False     |
| final     | 134    | 134       | 3356  | 1477        | 0.951 | True      |

Key observations:
- Dimers grow ~linearly (~13/traversal)
- Bonds grow superlinearly (O(n²) with dimer count — network densifies)
- Cross-synapse bonds = ~42% of total — substantial inter-synapse entanglement
- Coherence well-maintained with P31 (0.994 → 0.951 over 30s)
- Commitment waits for dopamine, fires immediately when it arrives

---

## What Needs Thinking Through Next

### 1. Realistic Inter-Traversal Intervals

The successful diagnostic used 3s gaps. Real biology: 30-60s per lap. At those intervals, coherence decay and dimer dissolution compete with accumulation. **This is where the transition boundary lives.** A single vector at 45s inter-traversal intervals is the critical next test — does the entanglement network survive between laps?

### 2. Bond Density vs. Topology Structure

Bonds are growing as ~O(n²) — approaching a fully-connected graph. A fully-connected graph carries no topological information. The computational primitive requires that the topology REFLECTS input structure (when dimers formed relative to each other, which synapses they span). Questions:
- Is the bond formation too permissive? Does the EM-mediated pathway create bonds too easily?
- At realistic timescales with decoherence, does the graph become sparser and more structured?
- Should we track graph metrics beyond density (clustering coefficient, path length, modularity)?

### 3. Commitment Gate Selectivity

With P31 and short intervals, the gate opens trivially (P_S ≈ 1.0, eligibility = 1.0). The gate only becomes selective when:
- **Coherence has partially decayed** (longer intervals, or P32 isotope)
- **Fewer traversals** have occurred (less accumulation)
- **Parameters are less favorable** (weaker coupling, fewer channels)

The sweep needs to find the boundary where the gate transitions from "always opens" to "never opens." That's the phase transition the paper predicts.

### 4. Dopamine Timing as Sweep Dimension

`dopamine_after_traversal` is now a parameter. This maps a key biological variable: does the system commit differently if dopamine arrives after 1 lap vs 6 laps? Early dopamine tests immature topology. Late dopamine tests mature topology. The difference IS the temporal evidence crystallization primitive.

### 5. Speed for EC2 Sweep

60s wall time for 30s sim time at 3s intervals. At 45s intervals with 10 traversals, sim time is ~460s — expect ~15 min/vector. The critical sweep (500 vectors) would take ~5 days single-threaded. Parallelization on c5.9xlarge (36 vCPUs) brings this to ~4-5 hours. Need to set up multiprocessing before running.

### 6. Sweep Output Schema

The sweep JSON now contains `traversal_snapshots` (list of per-traversal topology dicts) plus `commitment_snapshot`. This is rich time-series data, not just scalar results. Analysis tools need to handle the nested structure. Consider what figures this generates for the paper.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `sweep/theta_burst_scenario.py` | Complete rewrite: voltage-driven calcium, multi-traversal loop, per-traversal snapshots, single configurable dopamine pulse, commitment propagation |
| `sweep/quantum_dimensions.py` | New dimensions: stim_n_traversals, stim_inter_traversal_s, stim_theta_cycles (replaces stim_n_bursts) |
| `sweep/sweep_runner.py` | Console print shows snap count |
| `src/models/Model_6/multi_synapse_network.py` | Added n_clusters + mean_connectivity to get_experimental_metrics(); commitment propagation from three-factor gate to network_committed |

## Files Created This Session

| File | Purpose |
|------|---------|
| `/mnt/skills/user/experimental-validation-bridge/SKILL.md` | Skill: phase transition → experimental predictions bridge (from earlier session) |

---

## Next Session Priority

1. **Run single vector at realistic 45s inter-traversal intervals** — does accumulation survive?
2. If yes: set up EC2 parallelization and run critical sweep
3. If no: investigate decoherence rates and bond survival at biological timescales — this becomes the model gap