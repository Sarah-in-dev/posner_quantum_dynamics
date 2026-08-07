# Dopamine Reward Signaling & Synaptic Credit Assignment — grounding for F3

**2026-08-06. Cited research pass (4-subagent fan-out + coordinator synthesis) to DERIVE the F3 reward
architecture from the neuroscience, not from intuition (emergent-only).** The one load-bearing parameter
— the eligibility↔dopamine coincidence window — was independently re-verified at the primary source by
the physics thread: **Yagishita et al. 2014, Science (PubMed 25258080), abstract verbatim: "dopamine
promoted spine enlargement only during a narrow time window (0.3 to 2 seconds) after the glutamatergic
inputs."** [VERIFIED-at-source]

## THE REFRAME (corrects both our intuitions — the key finding)
**Per-synapse SPECIFICITY is carried by the local ELIGIBILITY TRACE, not by dopamine.** Dopamine is
volume-transmitted — a near-global scalar third factor. A single well-timed pulse credits *exactly the
synapses that currently hold a trace*, in proportion to it — so for an IMMEDIATE single event a single
pulse is sufficient and already synapse-specific (via the trace). The dopamine **TRAIN** is required for
a different job: **(a) delayed reward (seconds–minutes) via TD chaining to earlier predictive cues;
(b) SIGN** (burst→LTP, dip→LTD); **(c) magnitude / distributional structure.** So Sarah's "multiple
signals" intuition is right for the *temporal/delayed-credit and sign* axes; the *which-synapse*
discrimination is the trace's job. A model using one global scalar pulse under-credits delayed & distal
predictors.

## Load-bearing parameters (for F3 constants — derive, don't choose)
| Parameter | Value | Source | Confidence |
|---|---|---|---|
| **DA↔eligibility coincidence window** | **0.3–2 s (DA AFTER glutamate)** | Yagishita 2014 Science | **VERIFIED-at-source** |
| Tonic DA firing | 2–10 Hz (~4–5 baseline) | Grace 1991 | High |
| Intraburst (phasic) firing | 14–30 Hz; transient ~100–200 ms | Grace&Bunney; Schultz 1998 | High/Med |
| Positive RPE code | ~linear in firing rate | Bayer & Glimcher 2005 Neuron | High |
| Negative RPE code | pause/dip DURATION (asymmetric) | Bayer, Lau & Glimcher 2007 | High |
| Eligibility-trace τ (biophysical) | ~0.5–2 s | Yagishita 2014; Gerstner 2018 | Med |
| BTSP window (seconds-scale) | ~±2 s, single plateau | Bittner & Magee 2017 Science | High |
| STDP window (ms-scale) | ≈±20 ms | Bi & Poo 1998 | High |

## What this says about the F2-e null (precise re-diagnosis)
Our eligibility trace = **P_S / the coherence window per synapse** (we have it). The null (reward inert,
commit ~1/8) is because: (1) reward is disconnected — `apply_reward_correlated` is a dead stub, dopamine's
only reader removed in F2; (2) commitment is calcium-fast, consuming the trace BEFORE any windowed dopamine
read. The grounded mechanism — local trace (specificity) + dopamine gating in a 0.3–2 s window (sign/timing)
+ TD train (delay) — we implemented NONE of the last three.

## F3 architecture (derived)
1. **Eligibility trace = P_S per synapse** — carries specificity. (NB: biology's trace is ~seconds; OUR
   coherence window is ~100 s — that is the Fisher long-trace BET, our program's premise, NOT established
   biology. The dopamine window/structure below IS well-grounded; the long trace is the hypothesis.)
2. **Reward = third factor with an explicit ~0.3–2 s coincidence window + SIGN** (burst→potentiate eligible,
   dip→depress). GATE the trace; do not multiply a scalar. Feed `dopamine_system.py`'s existing phasic/tonic
   structure in as DA(t).
3. **The conversion must WAIT for/integrate the windowed dopamine signal** — not race ahead on calcium.
4. **Delayed reward:** a TRAIN of transients sampling the persistent trace at each transient's 0.3–2 s
   window. Our long (~100 s) trace is what lets a train bridge seconds–minutes — the coherence-gated payoff:
   the delay biology needs TD-chaining + tag-and-capture for, our long trace could bridge directly.
5. **Optional richness:** vectorized/distributional DA (Dabney 2020, heterogeneous reversal points);
   asymmetric negative-RPE (pause-duration) — note the CONTESTED Hart 2014 *symmetric* NAc-release counter.

## Contested / open (carry into design)
- Symmetric (Hart 2014, NAc *release*) vs asymmetric (Bayer/Glimcher, firing *rate*) negative-RPE encoding.
- Scalar-RPE (Kim 2020; Mikhael/Gershman 2022) vs multiplexed/distributional (Berke 2018; Engelhard 2019;
  Dabney 2020) — balance favors structured/multiple, but "fundamentally vector vs many-scalars-parallel" open.
- Eligibility-trace τ has no single agreed value; multiple traces (LTP vs LTD, He et al. 2015) may coexist.
- Fisher/Posner long molecular trace = SPECULATIVE, coherence figures unverified — firewalled from the above.

Key sources: Schultz/Dayan/Montague 1997; Schultz 1998; Grace 1991; Bayer & Glimcher 2005; Bayer/Lau/Glimcher
2007; Cohen 2012; Dabney 2020; Engelhard 2019; Berke 2018; Howe 2013; Hamid 2016; Yagishita 2014 [verified];
Frémaux & Gerstner 2016; Gerstner 2018; Bittner & Magee 2017; Bi & Poo 1998; Frey & Morris 1997; Hart 2014;
Fisher 2015 (speculative).
