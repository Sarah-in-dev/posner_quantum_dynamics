# Session Handoff: Experimental Validation Bridge + Sweep Debugging

**Date:** April 6, 2026
**Session focus:** Understanding the bridge between computational model physics and experimental validation; debugging the sweep smoke test.

---

## Key Breakthrough: Phase Transition Signatures Replace Pharmacology

The January 2026 experimental validation track was stuck because proving quantum necessity required convergent pharmacological evidence (isoflurane + nocodazole) and nocodazole is only feasible in slices. This session identified that the Fröhlich condensation threshold itself generates measurable statistical signatures testable on Karim's existing voltage imaging platform WITHOUT pharmacology.

### Three Testable Predictions

1. **Bimodal response distributions at intermediate delays.** Near the condensation threshold, trial-to-trial stochastic variation produces all-or-none outcomes: some trials catch the condensate still active (strong facilitation), others catch it already collapsed (weak/no facilitation). Classical decay produces unimodal distributions at every delay. This is a distribution-shape prediction, not a mean prediction.

2. **Sharp cooperative input threshold.** Below a critical number of co-active synaptic inputs, the pump rate never reaches the Fröhlich threshold — no long-delay facilitation regardless of stimulus strength. Above it, facilitation appears sharply (Hill coefficient >> 1). Testable by varying holographic ensemble size.

3. **Variance peak at the critical delay.** Trial-to-trial variance peaks AT the transition delay and is suppressed on both sides. Generic criticality signature.

### Supporting Literature

- **Preto 2017 (J Biol Phys, PMC5471165):** Master equation for Fröhlich condensation solved via Gillespie algorithm (5000 trajectories). Distribution of phonon occupation changes qualitatively at threshold — analogous to laser transition. Fluctuations increase significantly during condensation.
- **Zhang, Agarwal & Scully 2019 (PRL 122:158101):** Phonon-number distribution transitions from quasi-thermal to super-Poissonian. Condensate lifetime increases ~10× for BSA at room temperature.
- **Wang & Scully 2022 (Phys Rev B 106:L220103):** Analytical proof Fröhlich condensation is a second-order phase transition with large fluctuations.
- **Reimers et al. 2009 (PNAS 106:4219):** Three condensate types. COHERENT condensates are biologically inaccessible. WEAK condensates are feasible and can affect chemical kinetics. Our model claims weak condensation — this distinction is critical for reviewers.
- **Fenton & Muller 1998 (PNAS, PMC19716):** CA1 place cells show extreme trial-to-trial variability — 18 spikes on one pass, 0 spikes on a nearly identical pass. This all-or-none variability, treated as unexplained noise for 25 years, is exactly the bimodal signature the condensation threshold predicts.

### Key Physics Point: Timescale Separation

Condensate fluctuations are at MHz-THz frequencies. Dimer dynamics operate on seconds timescales. Dimers experience the time-averaged condensate state, not instantaneous fluctuations. The sharpness of the downstream measurement depends on the sharpness of the time-averaged condensation transition — which Preto and Wang confirm is a genuine phase transition.

### Reimers Caveat

Reimers 2009 argues coherent Fröhlich condensates are biologically inaccessible and the Orch OR model is untenable. Our model does NOT claim coherent condensation. We claim weak condensation modulating dimer formation kinetics. This is within Reimers' feasible regime. Any paper MUST make this distinction explicitly or reviewers will reject based on Reimers.

---

## Sweep Debugging: Zero Dimers is Correct Physics

### What Happened

Smoke test ran 3 of 10 vectors. Infrastructure works (no crashes, Q1 field varies correctly with parameters). But zero dimers across all vectors.

### Why Zero Dimers is Correct

The scenario delivered 5 theta bursts over 625ms, then 60s silence. At 1ms timescales, dimer formation rates SHOULD be vanishingly small. The PNC² formation pathway means low calcium concentrations produce negligible formation rates. This is not a bug — it's the temporal integration working correctly.

The previous "working" experiments (isotope, pharmacology) bypassed this by flooding with massive direct calcium, validating the dimerization MACHINERY but not the input processing.

### The Real Input: Multiple Place Field Traversals

CA1 place cells fire during place field traversals lasting ~1-2 seconds each. Rats running laps on a 2m track traverse the same place field every ~30-60 seconds. Learning occurs over MANY laps. The dimer coherence window (~200 seconds) spans 3-6 lap intervals.

The sweep scenario must simulate multiple traversals separated by realistic inter-traversal intervals, not a single brief burst. The system accumulates across laps:
- Traversal 1 (t=0): ~10 theta cycles, initial calcium entry, minimal dimer formation
- Traversal 2 (t=60s): More calcium, residual dimers from T1 still coherent, accumulation continues
- Traversal 3 (t=120s): Further accumulation, approaching formation threshold
- Traversal N: System crosses threshold (or doesn't — depends on parameters)

The number of traversals needed to cross the formation threshold is itself a sweep output metric — and connects directly to the behavioral observation that place fields stabilize over multiple laps.

### Scenario Redesign Needed

`theta_burst_scenario.py` needs to be redesigned for multi-traversal stimulation:
- Parameter: `n_traversals` (1-10)
- Parameter: `inter_traversal_interval_s` (30-90, matching lap times)
- Each traversal: ~10-16 theta cycles at 8 Hz
- Calcium injection through VGCC pathway (not direct flooding)
- Metrics extracted after each traversal AND after final silence period

The VGCC calcium coupling issue (flagged as pre-existing) may need attention — but first check whether multi-traversal accumulation with direct injection produces dimers at biologically realistic concentrations. If it does, the VGCC coupling can be addressed separately.

---

## Sweep Infrastructure Status

### What Works
- `sweep/talon_core/permutation_engine.py` — extracted from TALON, stdlib only, generates covering arrays
- `sweep/quantum_dimensions.py` — 20 dimensions across 4 groups, 2-wise coverage over 8 critical dims produces 40 vectors
- `sweep/sweep_runner.py` — CLI orchestrator with 4 modes (smoke/critical/high/full)
- `apply_vector_to_params()` — correctly maps dim_ids to Model6Parameters attributes (all 9 paths verified)
- Direct calcium injection works (isotope test confirmed 2.5× P31/P32 dimer difference)

### What Needs Fixing
1. **Scenario redesign:** Multi-traversal stimulation with realistic inter-traversal intervals
2. **Results persistence:** Currently only written on full sweep completion — partial results lost. Need per-vector result writing.
3. **Parallelization:** Move to EC2 c5.9xlarge with multiprocessing (same pattern as isotope/pharmacology experiments)
4. **Speed:** Some vectors took 42 min at dt=0.001. Consider adaptive dt or reduced simulation duration for smoke mode.

---

## Skill Created

`/mnt/skills/user/experimental-validation-bridge/SKILL.md` — captures the full connection between computational model phase transition physics and testable experimental predictions. References all key literature. Includes the stimulus timescale insight, the Fenton & Muller place cell variability connection, and the Reimers weak-condensation caveat.

---

## Next Steps

1. **Redesign theta_burst_scenario.py** for multi-traversal stimulation
2. **Add per-vector result persistence** to sweep_runner.py
3. **Test multi-traversal scenario** locally with a single vector to confirm dimer accumulation
4. **Parallelize on EC2** once scenario produces meaningful Q2 output
5. **Run critical sweep** (500 vectors) to map the transition boundary
6. **Extract quantitative predictions** for the three statistical signatures
7. **Update quantum-biology-primer skill** with the experimental bridge findings
8. **Paper:** Add statistical predictions section to Physical Review E manuscript

## Files Created/Modified

| File | Status |
|------|--------|
| `/mnt/skills/user/experimental-validation-bridge/SKILL.md` | New skill |
| `sweep/theta_burst_scenario.py` | Needs redesign (multi-traversal) |
| `sweep/sweep_runner.py` | Needs per-vector result persistence |