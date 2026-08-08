# A coherence-gated eligibility trace for temporal credit assignment — F1→F2→F3 synthesis

**2026-08-08. Checkpoint write-up of the unified mechanism built across F1 (isotope lever), F2 (measurement),
and F3 (reward-gated consolidation), grounded against the neuroscience/physics literature. Written so an
outside physicist or neuroscientist can evaluate it: every load-bearing claim is tagged
ESTABLISHED / PREMISE / OPEN, and the falsifiers are stated. This is a MODEL (in-silico); per the program's
attribution discipline it does not measure nature — a positive result strengthens the discrimination case but
attribution always routes through theory.**

---

## 1. The claim, in one paragraph
Temporal credit assignment — linking an action to a reward seconds to minutes later — is an established
behavioral phenomenon whose biological mechanism is **not known**: the "eligibility trace" that must hold a
synapse's candidacy across the delay has no identified molecular substrate, and every measured trace (~0.3–5 s)
is far too short (see §5). This model proposes a candidate answer: **the eligibility trace is a coherent ³¹P
nuclear-spin state in a calcium-phosphate dimer, which persists ~100 s; it decoheres passively on that clock;
and when reward arrives, dopamine gates a spin-selective binding-melt that reads out whatever coherence
remains.** Credit is assigned at the coincidence of a still-coherent tag and dopamine — a three-factor rule in
which the long coherence supplies the one thing biology's short traces cannot: a physical, synapse-local mark
that survives the seconds-to-minutes gap.

## 2. The problem it addresses
Reinforcement-learning theory (TD/RPE; Schultz–Dayan–Montague 1997) says *what* is computed (a dopamine
prediction-error updates the synapses that predicted reward) but not *how* the credit physically lands on a
synapse after a delay. Three-factor plasticity (Reynolds & Wickens 2002; Gerstner 2018) says *how* it should
map — pre×post sets an eligibility trace, a neuromodulator converts it — but **posits an eligibility trace it
cannot identify molecularly**. The measured traces are ~0.3–2 s (Yagishita 2014) to ~2 s (Shindou 2019, whose
authors state "other mechanisms must be invoked"); behavior spans seconds-to-minutes; and **no experiment shows
a single identified synapse holding a mark across that gap** (grounding: `RESEARCH_TCA_MECHANISM_2026-08-08`).
That gap is the hole this model targets.

## 3. The unified mechanism (one story)
1. **Tag formation.** Synaptic activity → Ca²⁺ influx → Ca-phosphate **dimers** whose two ³¹P nuclear spins are
   born entangled (singlet). The coherent tag = the eligibility trace; per-synapse specificity lives here (one
   spine = one nanodomain = one tag). [PREMISE — Fisher 2015 pathway; substrate unproven, §5]
2. **Persistence.** The tag holds coherence **~100 s** — long enough to bridge the reward delay. Lifetime is the
   ³¹P coherence time, which is **dimer-specific** (Agarwal 2022: dimer "hundreds of seconds"; the asymmetric
   trimer decoheres sub-second). [PREMISE for biology; the ~100 s figure is ESTABLISHED for the dimer in
   Agarwal's calculation — verified.]
3. **Passive decoherence + the isotope lever (F1).** The tag leaks coherence on its own via environmental
   coupling (intramolecular dipolar/J modulated by tumbling — symmetry-protected in the dimer; external water
   protons; paramagnetic O₂/metals). **⁶Li/⁷Li doping dials the lifetime**: ⁷Li Larmor-matches ³¹P → efficient
   scalar relaxation → T₂≈14 s; ⁶Li is far off-resonance → negligible, T₂≈216 s ≈ undoped. Derived parameter-free
   from tabulated γ/I/Q (`nuclear_relaxation.py`; Abragam; arXiv 2310.13484). [F1: DERIVED, reproduces the
   literature; the isotope *direction* is ESTABLISHED physics, the magnitude rides on one cited J calibration.]
4. **The readout (F2).** Two spin-correlated dimers **bind and melt** (Fisher 2015): the binding is
   **spin-selective** (Quantum Dynamical Selection, Fisher & Radzihovsky 2018 — only the coherent/singlet channel
   reacts), and the melt releases a Ca²⁺ shower → glutamate → plasticity. This is what makes it a *readout* and
   not leakage: state-selective + amplified + connected to the machinery. Rate derived from Smoluchowski
   encounter × QDS singlet projection (`posner_binding.py`). [F2: DERIVED, reproduces QDS/diffusion limit.]
5. **Reward timing (F3).** Dopamine carries the reward-prediction-error signal (Schultz) and **gates/times** the
   binding-melt — not by touching the spin (that framing was falsified, §5), but via its real signaling
   (D1/D2→cAMP/PKA→local Ca²⁺/pH), which sets the conditions the Ca/pH-dependent melt needs. Credit = the
   spin-selective readout of a still-coherent tag, signed by burst (LTP) vs dip (LTD). The eligibility *window*
   is the coherence lifetime, not a fixed 0.3–2 s. [F3: the coherence gate + reward timing; readout grounded to
   the dopamine-gated binding-melt, `RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08`.]

**One mechanism:** F2 is the readout, F3 is its reward-timing, F1 is the physical lever on how long the tag
survives to be read. Passive decoherence (leakage) erases the tag; the dopamine-gated binding-melt reads it out;
only the second is learning.

## 4. The evidence (in-silico, measured, pre-registered)
- **F1** — `nuclear_relaxation.py` reproduces the ⁶Li/⁷Li scalar-relaxation literature (4/4 acceptance); wired,
  undoped bit-identical.
- **F2** — `posner_binding.py` reproduces the diffusion-limited encounter + QDS 16× spin selectivity + the ⁷Li
  isotope consilience (4/4). Wired as the measurement trigger; the full control ladder (F2-c/F2-e, 60 draws)
  showed the grounded trigger fires early/local — which **corrected** the prior "computation = network partition"
  claim (a real falsification in the model) — and that commitment is calcium-dominated / reward-inert, which
  located F3 as the true gap.
- **F3** — `reward_gating.py` (7/7). **Delayed-credit (F3-b, n=6):** the coherent tag commits across delays to
  30 s while the classical ~2 s trace is dead past 5 s; quantum−classical durable at 10 s = +0.00465, permutation
  p=0.001. **Isotope arm (F3-c, n=6):** ⁶Li ≈ undoped (credits at all delays), ⁷Li short-only (marginal at 2 s,
  dead by 5 s — decoheres below the Werner floor), isotope contrast at 10 s p=0.000. (⁷Li-at-2 s missed its
  pre-registered ≥5/6 threshold at 4/6 — reported, not retuned; ⁷Li's trace is so short 2 s is already its edge.)

**The through-line signature:** change one nuclear-spin degree of freedom (⁶Li↔⁷Li, invisible to classical
chemistry) and the temporal-credit outcome moves — attribution-by-intervention, realized end-to-end in the model.

## 5. Grounding ledger
**ESTABLISHED (cited; the ★ items independently verified at source this arc):**
- ★ Yagishita 2014 (Science): dopamine gates spine plasticity only 0.3–2 s after glutamate, via cAMP/PKA/PDE.
- ★ Shindou 2019 (EJN): striatal eligibility trace ~2 s (Ca-permeable AMPARs); authors declare it incomplete.
- ★ Agarwal 2022 (2210.14812): dimer coherence "hundreds of seconds"; trimer sub-second (asymmetry).
- Fisher 2015 / Fisher & Radzihovsky 2018: readout = spin-selective Posner binding-melt → Ca²⁺; QDS.
- Dopamine ≈ RPE (Schultz 1997/2015); the D1/cAMP/PKA/DARPP-32 readout cascade; dopamine's µm/ms volume-transmission.
- **The mechanism of temporal credit assignment is UNSOLVED**; **no molecular eligibility trace is identified**;
  no proposal exists for a neuromodulator collapsing a nuclear-spin state.

**OUR PREMISE (hypothesis, not measured — the model's load-bearing bets):**
- The eligibility trace IS a ~100 s coherent ³¹P dimer tag (the central claim).
- The Ca-phosphate dimer forms in the spine and holds ³¹P coherence ~100 s in vivo.
- Dopamine's Ca²⁺/pH modulation reaches the Posner microdomain to gate the melt (plausible, unproven).

**OPEN / CONTESTED:**
- The Posner/dimer substrate itself: dimer-vs-trimer; coherence-lifetime estimates disagree by orders of
  magnitude (Fisher ~1 day; Player & Hore ~37 min; Agarwal dimer ~100–1000 s).
- Whether ³¹P entanglement is real at all (unproven; a radical-pair mechanism explains the lithium data too).
- "Readout" here is a FUNCTIONAL definition (state-selective + amplified + connected), not a foundationally
  settled one (the measurement problem).
- In-silico only: the model does not measure nature (attribution gap).

## 6. Falsifiers / what an outside expert should probe
1. **The substrate (root premise).** Does a Ca-phosphate dimer form in spines and hold ³¹P coherence ~100 s?
   This is an AMRIS-class ³¹P-NMR question. If the dimer coherence is not ~100 s in a physiological setting, the
   central premise fails and the trace is not this.
2. **The isotope signature (the one real attribution lever).** Does ⁶Li vs ⁷Li actually shift the *timescale* of
   temporal credit assignment behaviorally (⁷Li crediting only at short delays)? The model predicts it; it is
   testable in principle and is the cleanest falsifier. Caveat: the lithium data is also explained by a
   radical-pair mechanism — the isotope implicates "a nuclear spin," not specifically this dimer mechanism.
3. **The Ca/pH-reaches-the-microdomain step** (whether dopamine signaling can gate a Posner melt where the tag
   lives — a glutamatergic-spine vs dopaminergic-terminal locus question).
4. **Spin-selectivity of the binding-melt** (QDS is a conjecture, not measured).

## 7. The attribution frame (the honest epistemic core)
There is no experiment that measures the quantum state *in the living computational system* AND attributes the
computation to it; attribution routes through theory. Therefore the model's value is **realism + discrimination
+ convergence**, not fit: every constant on a causal path is cited/derived (emergent-only, no tuning to a
result), and the model is built to predict *differently* from classical/radical-pair rivals on an attainable
measurement — here, the isotope-timescale signature. A positive in-silico result (F3-b/F3-c) strengthens the
discrimination case; it does not certify the mechanism in biology.

## 8. Open work
- **Network / living loop (Phase C):** move from the controlled single-synapse probe (hand-set eligibility,
  injected dopamine) toward the multi-synapse loop, with a baseline-robust dopamine classifier.
- **Consolidation-strength ceiling:** credit magnitude is graded by coherence and fades at long delay (F3-b);
  the durable channel's own decay (D20) is a separate, unfixed limit.
- **Substrate grounding:** the premises in §5 are where the whole edifice is anchored; the isotope-behavioral
  test (§6.2) is the nearest falsifiable handle.

*Artifacts: `nuclear_relaxation.py`, `posner_binding.py`, `reward_gating.py`; research-log rows F1/F2-a..e/F3-a..d;
grounding docs RESEARCH_DOPAMINE_REWARD_SIGNAL / RESEARCH_TCA_MECHANISM / RESEARCH_DOPAMINE_READOUT_PHYSICS
(all 2026-08); prereg PREREG_F1 / PREREG_F2 / PREREG_F3.*
