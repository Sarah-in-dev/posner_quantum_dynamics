# Research Program Handoff: Deriving the Quantum Computational Primitive via Permutation Analysis

**Date:** March 25, 2026
**Context:** This document captures a research direction developed across a conversation integrating three threads: (1) the quantum synapse Model 6 codebase, (2) the TALON permutation testing engine built for PAUL (healthcare scheduling intelligence), and (3) a strategic reframing of the research objective from "prove quantum effects in the brain" to "derive a computational primitive from quantum synaptic dynamics and use it to solve problems."

---

## The Strategic Shift

The original research program was structured as: model the quantum physics at a synapse → validate biologically → scale to a network → show the brain uses it. This path requires proving quantum biology to skeptics before producing computational results.

The new program inverts this: let the model's own behavior reveal the computational primitive → formalize it → deploy it against real problems → let the results speak. The biology becomes interesting to others after the computation works, not before. This follows the AlphaFold precedent: DeepMind didn't prove how proteins fold. They built a system that predicted structures correctly. The structural biologists validated afterward because the outputs were right.

**Core objective:** We are not trying to prove the brain is quantum. We are trying to show that a computational framework derived from quantum synaptic dynamics — specifically coherence/decoherence rather than backpropagation — can solve certain difficult problems better than current ML. Then implement on biological substrate (organoids) to assign hard problem sets to tissue that can determine solution designs ML cannot.

---

## The Key Insight: Criticality as the Computational Regime

Model 6's interesting behavior consistently lives near phase boundaries, not deep in any single regime:

- The commitment gate declines from 100% to ~55% over 0–120 seconds — operating at the transition between "always commits" and "never commits"
- Both Q1 and Q2 converge on ~20 kT — thermal energy at body temperature, right at the boundary between quantum and classical behavior
- The coherence window of ~100s is just barely long enough for dopamine signaling — marginal, not generous
- The voltage threshold between -40mV (no dimers, no commitment) and -20mV (32 dimers, 100% commitment) is sharp

The hypothesis: the computational power comes from maintaining proximity to the phase boundary between coherent exploration and classical commitment. The primitive isn't "a coherent state" or "a decoherence event." It's the maintained critical regime itself. Systems that hold near this boundary longer, or across more synapses simultaneously, integrate more information before committing.

This reframes neurodegeneration (per the operational stack presentation): substantia nigra neurons fail not because they lose quantum coherence, but because they can no longer afford to maintain the critical regime. Criticality maintenance is the highest operational cost in the stack.

---

## The Method: TALON Permutation Engine Applied to Model 6

### What the Permutation Engine Does (established in PAUL)

TALON's core philosophy: don't prescribe what matters. Crawl the system, extract the dimensions, and let the interaction structure tell you what's important.

For PAUL's scheduling system:
- Layer 1 (Workflow Chain Crawler) parsed 89 files, found 4,031 branches, extracted 2,534 dimensions across 286 chains
- Layer 2 (Permutation Engine) generated 3-wise covering arrays — e.g., the Scheduling chain has 182 dimensions; exhaustive testing = billions of combos; the covering array = ~3,675 vectors while guaranteeing every triplet interaction is tested
- Layer 3 (Executors) ran each test vector against the live system and measured outcomes

### Applying to Model 6

The "system" is no longer a codebase with branch points. It's a physics simulation where the equivalent of branch points are parameter regimes where behavior changes qualitatively — phase boundaries.

**Dimensions to sweep** (at minimum, from Model 6's established layers):

1. **Calcium dynamics** — baseline concentration, channel conductance, voltage threshold, buffering capacity, nanodomain geometry
2. **ATP/J-coupling** — J_PP_atp coupling constant, hydrolysis rate, phosphate speciation equilibria
3. **Dimer formation** — aggregation rate (k=1e6 M⁻¹s⁻¹), Ca/P ratio, template enhancement factor (~1000×), number of template sites (3-4)
4. **Quantum coherence** — T₂ base (100s for dimers), isotope fraction (P-31 vs P-32), decoherence coupling to environment
5. **Tryptophan superradiance (Q1)** — field strength (0–22.1 kT range from isoflurane experiments), coupling to Q2
6. **pH dynamics** — activity-dependent acidification, feedback to phosphate speciation
7. **Dopamine** — timing relative to stimulus (the T₂ ≈ 210s window), concentration, D2 modulation
8. **CaMKII feedback** — barrier modulation by dimer field, bistability threshold
9. **Temperature** — affects everything; body temp = 310K
10. **Stimulus protocol** — voltage, duration, delay before reward

These are continuous parameters that need discretization at meaningful boundary values. The existing experimental results provide some boundaries: the -40mV/-20mV voltage transition, the isoflurane dose-response curve (22.1 → 0.0 kT), the dopamine timing curve, the isotope ratio effect (784×).

### Output Metrics

Two primary metrics, measured together:

1. **Error resolution (temporal credit assignment accuracy):** Present a stimulus at T₁, deliver a reward signal at T₂ with variable delay. Did the correct synapse (the one that received the causal stimulus) get credited? Measured as: correct commitment at the right synapse / total commitment events.

2. **Weighted plasticity changes:** What weight changes resulted? Magnitude, direction, and — critically — variance across slightly different inputs at the same parameter point. Near criticality, small input differences should produce meaningfully different plasticity patterns. Deep in a stable basin, everything looks the same. The variance distinguishes actual computation from reflexive response.

### What the Sweep Reveals

**The phase boundary in parameter space** between "produces correct credit assignment with input-sensitive plasticity" and "produces dynamics but doesn't compute" IS the computational primitive. The parameters that control that boundary define what matters. The parameters that don't affect it are what you can abstract away.

**Dead zones** — parameter regimes where all known factors are favorable but the system still doesn't compute — are signatures of missing mechanisms. The parameter context of the dead zone narrows down what kind of mechanism is needed.

**The four-factor AND gate** (calcium + dimers + EM field + dopamine all required) was discovered experimentally. The permutation engine would have found it systematically via 4-wise interaction analysis, and would find whatever 5-factor, 6-factor, and higher-order interaction patterns exist that haven't been discovered yet.

---

## Phase 2: Network Derivation (Also via Permutation)

Once Phase 1 identifies the single-synapse primitive, Phase 2 treats it as a node and sweeps interaction parameters between nodes:

- How many primitives are coupled?
- What coupling topology?
- What coherence overlap between adjacent primitives?
- How does commitment at one primitive influence the coherent state of neighbors?

The permutation engine doesn't care whether it's sweeping biophysical parameters or coupling topologies. The output metrics are the same (error resolution + plasticity) but measured at the network level. The network architecture that falls out of this sweep is the one where the interaction structure produces good computation. It wasn't designed — the sweep found it.

This is "letting the system inform the system" at the network level.

---

## The Train/Compute Distinction: Why This Is Fundamentally Different from ML

Standard ML has two separate phases:
- **Training:** Show thousands of examples, compute gradients, update weights. System changes.
- **Inference:** Freeze weights, produce outputs. System doesn't change.

In the quantum primitive framework, **there is no separation.** The commitment gate IS the computation AND the weight update simultaneously. When dimers collapse via singlet state measurement, that collapse produces the output (this synapse strengthens) AND permanently changes the system state (the weight is updated). Computing without learning is physically impossible because they're the same event.

This means:
- No backward pass. Only forward exploration (coherent superposition near criticality) followed by commitment (collapse).
- No separate training dataset. The sequence of problems the system encounters IS the training.
- No frozen deployment. Every computation adapts the system in real time.
- Stability comes from the critical regime itself: the system commits when evidence crosses the commitment threshold, and maintains exploration when it doesn't. The threshold IS the stability mechanism.

This eliminates the fundamental brittleness of deployed ML models (encountering situations not in training data) but introduces a different challenge: characterizing how the system maintains what it's learned while continuing to adapt. The permutation sweep of Phase 2 should reveal whether the critical regime naturally provides this balance.

---

## Practical Implementation Plan

### Phase 1: Single-Synapse Permutation Sweep

**Prerequisite:** Build a wrapper around model6_core.py that accepts a parameter vector and returns error resolution + plasticity metrics.

- model6_core.py is an orchestrator stepping all subsystems forward in time
- Need an interface layer that: (a) accepts a dict of parameter overrides, (b) runs a stimulus-delay-reward protocol (stimulus at T₁, dopamine at T₂, variable delay), (c) returns credit assignment accuracy and plasticity pattern (magnitude + variance)
- This is not a rewrite — it's a callable wrapper on top of what exists

**Then:**
1. Define dimensions from Model 6's parameter space with discretization at known boundary values
2. Generate 3-wise covering arrays across those dimensions
3. Run the sweep (parallelizable on EC2, same infrastructure used for the 190-trial isotope comparison)
4. Analyze: where are the phase boundaries? Which parameter interactions dominate computational output? Where are the dead zones?

**Expected output:** The computational primitive defined empirically — the minimal parameter regime that produces reliable temporal credit assignment with input-sensitive plasticity.

### Phase 2: Network-Level Permutation Sweep

Take the primitive from Phase 1, define interaction dimensions (coupling topology, coherence overlap, commitment propagation), generate covering arrays, sweep. Let the network architecture emerge from the interaction structure.

### Phase 3: Benchmark Against Known Problems

Deploy the primitive (and its emergent network) against problems with ground truth:
- **PAUL's scheduling constraint space** — 182 dimensions, known correct solutions, known ML/heuristic baselines. You built the constraints. You have the data.
- **Causal discovery** — Tübingen cause-effect pairs benchmark. Coherent exploration of interaction structure should have structural advantage over correlation-based methods.
- **Delayed signal tasks** — synthetic benchmarks testing temporal credit assignment across varying delays.

### Phase 4: Biological Substrate (FinalSpark)

If Phase 3 works in silico, take the same I/O protocol to organoid tissue. The permutation sweep from Phase 1 tells you which parameter regime to target. The benchmark results from Phase 3 tell you what performance to expect. The organoid experiment becomes: does biological tissue, operating in the identified parameter regime, solve the same problems?

---

## Model 6 Codebase: What Exists

The simulation has 8+ interacting layers at a single synapse:

| Layer | Module | Key Physics |
|-------|--------|-------------|
| Calcium dynamics | calcium_system.py | Channels, diffusion, buffering, nanodomains. Sharp threshold at -40mV → -20mV |
| ATP/J-coupling | atp_system.py | 20 Hz field, hydrolysis, phosphate speciation |
| Dimer chemistry | ca_triphosphate_complex.py | CaHPO₄ ion pairs (K=588 M⁻¹), Ca₆(PO₄)₄ dimer aggregation (k=1e6 M⁻¹s⁻¹) |
| Template enhancement | (in formation logic) | ~1000× concentration at 3-4 sites/synapse, >99% dimer selectivity at synaptic Ca/P |
| Quantum coherence | posner_system.py | T₂=100-225s (P-31), 0.3s (P-32), 784× isotope ratio |
| Tryptophan/Q1 | (EM field coupling) | Femtosecond field, 0–22.1 kT range, dose-dependent isoflurane suppression |
| pH dynamics | pH_dynamics.py | Activity-dependent acidification → phosphate speciation feedback |
| Dopamine readout | dopamine_system.py | D2 modulation, T₂≈210s timing (R²=0.982), commitment gate 100%→55% over 0-120s |
| CaMKII feedback | (feedback loop) | Dimer field → barrier modulation, 100% vs 0% DDSC triggering (±EM) |
| Orchestrator | model6_core.py | Coordinates timesteps across all subsystems |
| Parameters | model6_parameters.py | All physical constants from scipy, organized by subsystem |
| Network | multi_synapse_network.py | Werner state entanglement, all-or-none coordinated collapse |

**Key validated results:**
- Isotope comparison: P-31 T₂ = 225s, P-32 T₂ = 0.3s (~784× ratio, 190 parallelized trials on EC2)
- Dopamine timing: T₂ ≈ 210s, R² = 0.982
- Four-factor AND gate: calcium + dimers + EM field + dopamine all required
- Isoflurane pharmacology: dose-dependent Q1 suppression (22.1 → 0.0 kT) while preserving Q2 dimer formation
- Commitment gate: probabilistic one-shot quantum measurement, decline from 100% to ~55% over 0-120s (T_decay ≈ 110s)
- Correlated decoherence: entangled synapses show 100% all-or-none outcomes vs. 33% for independent (Werner state physics)

---

## TALON System 8 Connection

System 8 (ML Outcome Validation & Adversarial Reasoning) is designed but not yet built. Its four-layer architecture:

- **Layer 2 (Boundary):** Synthetic inputs → assert outputs in expected ranges
- **Layer 3 (Decision Quality):** Four-agent courtroom (Scorer → Critic → Defender → Judge) evaluates whether individual recommendations are good
- **Layer 4 (Consequence):** Analyzes what happens when all recommendations combine — emergent problems from individually-reasonable decisions

System 8's courtroom pattern maps onto the coherence/decoherence cycle: Scorer explores the solution, Critic applies constraints, Defender tests whether constraints are binding, Judge collapses to verdict. If System 8 works for PAUL's scheduling decisions, it validates the architectural pattern before implementation in quantum dynamics.

Building System 8 for PAUL and the Model 6 permutation wrapper can proceed in parallel — they share the philosophy but not the codebase.

---

## Key Principles

1. **Emergent over prescribed.** The computational primitive is not designed. It's discovered through the permutation sweep. The network architecture is not designed. It emerges from the interaction sweep.

2. **The system informs the system.** TALON's core philosophy applies at every level: the model's own behavior tells you what matters, what's missing, and where the computational power lives.

3. **Computation and learning are the same event.** There is no separate training phase. The commitment gate simultaneously produces output and updates weights. This is structurally different from backpropagation, not just biologically inspired.

4. **Criticality is the regime, not a property.** The computational primitive isn't defined by a specific quantum state or a specific decoherence event. It's defined by sustained proximity to the phase boundary between coherent exploration and classical commitment.

5. **Dead zones are informative.** Parameter regimes where all known factors are favorable but computation fails are signatures of missing mechanisms. The gap tells you what's missing; the parameter context tells you what kind of mechanism to look for.

---

## First Concrete Step

Build the callable wrapper around model6_core.py:
- Input: parameter dictionary (overrides to model6_parameters.py defaults)
- Protocol: stimulus at T₁, dopamine at T₂, configurable delay
- Output: (a) did the correct synapse commit? (b) what was the plasticity magnitude and pattern? (c) variance across N runs with slightly different input noise

Then define the dimension list with discretization boundaries and generate the first covering array.

The sweep is parallelizable on EC2 using the same infrastructure that ran the 190-trial isotope comparison. Each parameter vector is independent — embarrassingly parallel.