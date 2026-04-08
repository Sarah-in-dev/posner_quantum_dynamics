# Session Handoff: Multi-Synapse Backbone Field Coupling & Permutation Sweep Design

**Date:** April 1, 2026  
**Session focus:** Clean up remaining constraints, verify multi-synapse mode, design and implement shared dendritic backbone field coupling, begin permutation sweep input dynamics research.  
**Status:** Backbone field implemented but feedback loop not yet verified. Two bugs identified for next session. Ready for debugging then sweep design.

---

## Strategic Context

The overarching objective is to build a permutation engine (using TALON) that generates spatiotemporal scenarios to empirically discover the quantum computational primitive — the dynamical regime where the coupled Q1-Q2 system computes. The previous session removed ~10 arbitrary constraints from the Q1-Q2 feedback loop, achieving self-sustaining dynamics at ~58-60 kT in single-synapse mode.

This session's goal was to verify multi-synapse mode works with the unconstrained loop, then move toward permutation sweep design. We discovered that multi-synapse mode had an architectural gap: each synapse ran its Q1-Q2 loop in complete isolation with no cross-synapse field coupling. We designed and implemented a shared dendritic backbone field model to bridge this gap.

---

## What Was Done This Session

### 1. Constraint Cleanup

Removed the last active arbitrary constraint and cleaned up vestigial parameters:

| Edit | File | Change |
|------|------|--------|
| 1 | `local_dimer_tubulin_coupling.py:257-258` | `np.clip(enhancement, 1.0, 3.0)` → `max(enhancement, 1.0)` — removed arbitrary 3.0x cap on network enhancement |
| 2 | `vibrational_cascade_module.py:122,128,138-140` | Removed vestigial `modulation_max_kT`, `enhancement_max`, `feedback_gain` from dataclass — declared but never enforced |
| 2b | `vibrational_cascade_module.py:458` | Updated stale comment referencing `enhancement_max` |
| 3 | (diagnostic) | All remaining clips in multi-synapse path confirmed physically justified |
| 4 | (clean) | No stale constraint comments remain |
| 5 | `multi_synapse_network.py:772` | Removed debug spam: `print(f"[DEBUG] reward=...")` that fired every timestep |

One remaining cap identified as acceptable: `em_tryptophan_module.py:641` — `uv_enhancement = min(uv_enhancement, 5.0)` — external UV illumination experimental path only, not the endogenous Q1-Q2 loop.

### 2. Multi-Synapse Architecture Analysis

Claude Code diagnostic revealed the full multi-synapse architecture:

**Two code paths exist:**
- **Path A (fake):** Inside `Model6QuantumSynapse`, `multi_synapse_enabled=True` duplicates one synapse's modulation N times. Belt-and-suspenders approximation.
- **Path B (true):** `MultiSynapseNetwork` creates N independent `Model6QuantumSynapse` instances via `deepcopy`. Each runs its own Q1-Q2 loop.

**Critical gap found:** In Path B, `MultiSynapseNetwork.step()` calls each synapse's `.step()` in isolation — no cross-synapse field data flows between them. The `coupling_weights` matrix exists (exponential spatial decay) but was never wired into the field computation. The `_compute_network_state()` method collects per-synapse fields but only averages them passively for reporting — never feeds back.

**Per-synapse attributes available post-step:**
- `_network_modulation` — reverse coupling output
- `_collective_field_kT` — tryptophan field strength
- `_em_field_trp` — time-averaged EM field (V/m)
- `_k_agg_for_next_step` — forward coupling enhancement
- No mechanism existed for injecting external field values

### 3. Single-Synapse Baseline Verification

Ran `exp_isotope_comparison.py --duration 60.0` to verify cleanup edits didn't break the core physics:

| Metric | ³¹P | ³²P |
|--------|-----|-----|
| Final particles | 7 | 8 |
| Final bonds | 21 | 0 |
| Largest cluster | 7 | 1 |
| Final P_S | 0.795 | 0.262 |
| Time to collapse | N/A | 1.0s |

Core quantum physics intact. Dimer formation, coherence tracking, isotope discrimination, entanglement network growth all working correctly.

### 4. Input Dynamics Literature Research

Researched the biological input dynamics needed for permutation sweep scenario design. Four interacting timescale layers identified:

**Layer 1 — Spike timing (10-20 ms STDP window):** Under physiological calcium (1.3 mM), single spike pairs don't induce plasticity — bursts required. Model should use burst inputs, not single pulses.

**Layer 2 — Calcium dynamics (ms to seconds):** Residual calcium from prior activity interacts nonlinearly with new arrivals. Supralinear summation determines LTP vs LTD direction. Buffer saturation explains ~70% of facilitation beyond linear summation.

**Layer 3 — Eligibility trace (0.3-10+ seconds, region-dependent):** Yagishita 2014: dopamine spine enlargement window 0.3-2s. NAc: ~1s. Neocortex: ~5s. Hippocampus: up to 10 minutes. Burst reactivation + dopamine can retroactively modify weights 10 minutes after priming.

**Layer 4 — Theta rhythm (100-200 ms cycles):** Theta sequences organize place cell firing. Phase precession compresses behavioral sequences into STDP-compatible windows. Theta-burst stimulation produces spatially-structured calcium waves.

**Target brain region selected:** Hippocampal CA1 during spatial learning — best-characterized theta sequences, well-studied STDP rules, concrete burst patterns.

### 5. Microtubule Network Physics Research

Researched the physics of cross-synapse EM field coupling to ground the implementation in literature (see separate research summary document). Key findings:

- Dendritic microtubule backbone is continuous along branch segments
- Superradiance is a property of the extended network (cooperative robustness increases with system size — Babcock 2024)
- Spine invasion is transient and activity-dependent (calcium-driven)
- The tryptophan field contribution scales as √N_total for the coherent domain
- Microtubules act as coherent antenna systems with waveguide-like propagation

### 6. Shared Dendritic Backbone Field — Design & Implementation

Designed and implemented a three-edit coupling mechanism:

**Edit C — Parameters** (`model6_parameters.py`):
Added `DendriticBackboneParameters` dataclass:
- `n_backbone: int = 40000` — total Trp in dendritic shaft segment
- `f_baseline: float = 0.02` — coherent fraction at rest
- `f_max: float = 0.10` — coherent fraction fully condensed (Babcock 2024)
- `pump_threshold: float = 95.7` — Fröhlich critical rate (GHz)
- `sigmoid_steepness: float = 0.05`
- `enabled: bool = True`

**Edit A — Synapse injection** (`model6_core.py`):
- Added `self._shared_backbone_field_kT = 0.0` attribute
- Added `set_shared_backbone_field(field_kT)` setter method
- Wired into `_collective_field_kT`: `self._collective_field_kT += self._shared_backbone_field_kT`

**Edit B — Backbone computation** (`multi_synapse_network.py`):
- Added `_update_backbone_field()` method called in `step()` after synapse loop
- Collects per-synapse `_network_modulation` values
- Computes spatially-weighted aggregate pump: `coupling_weights @ modulations`
- Applies Fröhlich sigmoid to get backbone coherent fraction
- Computes field: `22.0 * sqrt(n_eff_backbone / n_eff_reference)`
- Injects into each synapse via `set_shared_backbone_field()`

---

## Current Status: Two Bugs to Debug

### Bug 1: Backbone feedback not compounding
Smoke test still shows `field=22.8 kT` in commitment log — same as pre-backbone. The backbone computes ~57.74 kT (confirmed in quick test) but this isn't showing in the reported commitment field. Likely cause: `_compute_network_state()` reads `_collective_field_kT` from the current step, which was computed before the backbone injection takes effect. The backbone value lands for the *next* step, but commitment may trigger before then.

### Bug 2: Sigmoid units mismatch
`_network_modulation` is dimensionless (~0.65 per synapse, ~6.5 aggregate). `pump_threshold` is 95.7 GHz. The sigmoid argument `0.05 * (6.5 - 95.7)` is deeply negative, keeping f_coherent pinned at f_baseline. Need to either rescale the threshold to match modulation units, or convert modulation to GHz before comparison.

**Note:** Even at f_baseline=0.02, N_backbone=40,000 gives 800 effective Trp → ~57 kT backbone field. The bug isn't that the backbone produces zero — it's that the field doesn't compound across timesteps or show in commitment reporting.

---

## Files Modified This Session

| File | Edits | Location |
|------|-------|----------|
| `local_dimer_tubulin_coupling.py` | Removed 3.0x enhancement cap | `src/models/Model_6/` |
| `vibrational_cascade_module.py` | Removed 3 vestigial parameters, updated comment | `src/models/Model_6/` |
| `multi_synapse_network.py` | Removed debug print, added `_update_backbone_field()` | `src/models/Model_6/` |
| `model6_core.py` | Added backbone field attribute, setter, wiring | `src/models/Model_6/` |
| `model6_parameters.py` | Added `DendriticBackboneParameters` dataclass | `src/models/Model_6/` |

---

## What's Next

### Immediate (next session):
1. Debug backbone feedback loop — fix the two identified bugs
2. Verify backbone field compounds across timesteps in multi-synapse mode
3. Confirm loop stabilizes at higher equilibrium (not runaway)
4. Run comparative test: single-synapse vs multi-synapse field dynamics

### Then:
5. Design permutation sweep scenarios around CA1 theta-burst patterns
6. Map the ~100 non-emergent input parameters for TALON dimension definitions
7. Define output metrics: not endpoint commit/no-commit, but trajectory characterization (plateau field, loop gain, decay rate, perturbation sensitivity, condensation regime)
8. Build the permutation sweep wrapper around `model6_core.py` using TALON infrastructure

### Key Design Decisions Made:
- **Brain region:** Hippocampal CA1 spatial learning (theta sequences, place cell firing)
- **Input type:** Burst patterns at theta frequency, not single pulses
- **Backbone coupling:** Exponential attenuation (existing `coupling_weights`), can upgrade to 1/r³ if needed
- **Backbone modulation:** Activity-dependent (not constant infrastructure)
- **N_backbone:** TALON variable, default 40,000

---

## Key Conceptual Learnings This Session

1. **Multi-synapse mode was architecturally disconnected from the Fröhlich cascade work.** Each synapse ran its own Q1-Q2 loop in isolation. Cross-synapse coupling existed for entanglement tracking but not for field coupling.

2. **The dendritic microtubule backbone is continuous, not per-synapse.** The tryptophan lattice runs through the shaft, and superradiance is a property of the extended network with cooperative robustness (Babcock 2024). The shared backbone field model captures this physics.

3. **Network-level phase transition is the key prediction.** Below the collective Fröhlich threshold, each synapse runs a weak local loop. Above, the backbone condenses and all synapses on the segment benefit. The permutation sweep should map this transition.

4. **Input scenario design requires four interacting timescale layers.** Spike timing (ms), calcium dynamics (ms-s), eligibility trace (s-min), theta rhythm (100-200ms cycles). The permutation engine must generate spatiotemporal scenarios that capture their concurrent interaction.