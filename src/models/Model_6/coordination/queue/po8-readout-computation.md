# PO-8 queue — items awaiting Sarah (append-only, PO-8-owned)

---

## Q1 — 2026-07-20 12:51Z · Unit A's target changed under grounding. Which rate do I correct?

**The ask:** rule on what Unit A actually implements.

**Why.** The kickoff says "bonds dissolve at ~1e-4/s but the physical rate is 9.63e-3/s, 96x faster."
Grounding says that is true of a code path the physical configuration never runs. Three paths:

| path | code | effective rate | vs physical 9.63e-3/s |
|---|---|---|---|
| cross-synapse | `multi_synapse_network.py:488` `0.1*(1-eta_factor*P_product)` | 0.051–0.094 /s (tau 11–20 s) | **~10x too FAST** |
| intra, non-provenance | `dimer_particles.py:665` `(0.01*(1-coh))/(1+protection)` | ~1e-4/s at coh≈0.99 | 96x too slow (the cited path) |
| intra, `provenance_bonding=True` (**what we run**) | `dimer_particles.py:580-586` | **none** — early-return; death only at P_S<=0.5 | step function at ~107 s |

Independent corroboration for the cross path, from a locked framing the kickoff does not cite —
`model6-research-findings-may29:91`: *"Cross dissolution rate must be derived from physics, not
calibrated to target an eligibility window... The current `K_DISENTANGLE_BASE = 0.1` (cross half-life
~9s) is unmoored from physics."* Two independent routes (my arithmetic, that locked note) agree.

**My recommendation.** Implement on **both live paths**, one opt-in flag, defaulting off:
1. cross (`:488`): replace the self-quenching `0.1*(1-eta*P)` with the derived
   `k_release = 1/T2 + 1/tau_dimer = 9.63e-3/s`, optionally fidelity-graded per the reviewer
   (`t_bond = T2*ln(4F0-1)`), which makes weak bonds die ~4x faster and the graph self-clean.
2. intra-provenance (`:580-586`): add the same rate-based release alongside the existing coherence
   death, so intra bonds have a lifetime instead of surviving intact until the Werner floor.

Path 2 matters most for Unit B: with no intra release, the near-lossless intra core (F=1.0000) is
what makes domains synapse-scale, and it will persist to readout *by construction* rather than by
physics. That directly prejudges advisor Q2 (does the lossless core survive to readout). Correcting
only the cross path would leave the answer baked in.

**Risk to flag:** both edits land in files with existing owners (`multi_synapse_network.py` is on
PO-5's surface map; `dimer_particles.py` is Pathway 1/2, also PO-5). Both are opt-in and off-path
bit-identical, and PO-5 shows WRAPPED-adjacent on the board, but this is a surface-ownership call,
not mine to make unilaterally.

**Alternative if you want minimum blast radius:** cross path only, and Unit B reports the persisting
intra core as a stated limit rather than a measurement.

---

## Q2 — 2026-07-20 12:51Z · Unit B needs per-synapse stimuli. Approve the `net.step` extension?

**The ask:** approve (or redirect) a backwards-compatible per-synapse stimulus argument on
`MultiSynapseNetwork.step`.

**Why.** Unit B's input contrast is synchronous vs staggered activation of neighbour groups at matched
density (the design that avoids the on/off density confound that sank PO-7 Unit 2). That requires
different synapses to see different input at the same instant. Today `net.step` broadcasts one dict to
all synapses — `multi_synapse_network.py:1257-1260`, with the code's own comment *"Each synapse gets
the same stimulus (In future, could have synapse-specific stimuli)"*. There is no other route:
driving synapses individually via `s.step()` bypasses `_update_backbone_field()` (`:1286`), leaves
`_backbone_eta` pinned at 0, and kills the entire cross-synapse channel — the exact trap that produced
the false "ignition is a coin-flip" finding (L-PO7-4 section 7).

**My recommendation.** Minimal, opt-in-by-shape change: `stimulus` stays a dict (broadcast, today's
behaviour, bit-identical) or may be a list/dict of per-index dicts, indexed in the existing
`for i, synapse` loop. Network-level `reward` continues to be read from a network-level dict, so the
dopamine gate is untouched. Gate it on the `MODE=offpath` network regression digest `515772101786800`
plus a broadcast-vs-list equivalence check (same dict replicated N times must reproduce broadcast
exactly).

**Also worth your ruling:** the kickoff frames Unit B as measuring the partition "when a dopamine
event triggers decoherence." Grounding shows dopamine does **not** decohere anything in this model
(see the lead file, correction 2) — what fragments domains is P_S(t) decay under T_singlet=216 s.
Unit B is still well-posed with dopamine as the *clock*, but I want to write it up that way rather
than implying a collapse mechanism the code does not have. Confirming you're content with that
framing before I pre-register.

---

## Q3 — 2026-07-20 13:1xZ · lambda = 5 um vs 214 um. Raised by the advisor (R6 check-in); GROUNDED and load-bearing. NOT my call — it is LOCKED-adjacent.

**The ask:** rule on the entanglement fidelity weight length lambda in `W_ij = exp(-d/lambda)`.

**Why.** The advisor's check-in argues the cross-bond fidelity weight is a category error. The code
uses `coupling_length_um = 5.0` directly as the fidelity weight (`multi_synapse_network.py:1147`),
and L-PO7-5 section 3 derived the F=0.815 ceiling and the "spine spacing sets correlation reach"
prediction from it. But our OWN **feasibility calc #1** (`model6-network-layer-feasibility-may30:73-78`,
LOCKED, PASS) puts the condensate coherence length at **L_coh = v*tau = 214 um at Q=10** (robust
floor ~20 um even at v=100 m/s), verdict verbatim *"internet, not local bus ... a collective mode
has no distance falloff inside its coherence domain."* Reusing the 5 um metabolic/structural length
(handoffs/MICROTUBULERESEARCH.md:210 calls it "MT bundle coherence length," range 1-20 um) as the
entanglement fidelity weight contradicts that locked calc, at a length 43x too short. Verified both
numbers against the skills/docs; the advisor's "feasibility calc #4" is our calc #1.

**Impact — it flips Unit B qualitatively (the advisor's phrase, and I concur):**
- lambda = 5 um: F ceiling 0.815; cross bonds die at ~74 s, intra at ~107 s => a **two-timescale,
  scope-contracting** eligibility trace (multi-synapse binding decays before per-synapse binding).
  Spine-spacing-sets-reach prediction stands.
- lambda = 214 um: W ~ 0.995 at 1 um; ceiling gone; cross and intra die together ~106 s => **one
  timescale, sharp cutoff**; the spine-spacing prediction evaporates.

**My recommendation.** Two things, in order:
1. **Decide lambda before I SCORE Unit B.** It changes the answer, so scoring first would bake in
   whichever value happens to be coded. My physics read: the advisor is right that the fidelity
   weight and the metabolic aggregation length are different constants and 5 um is the wrong one for
   fidelity; the coherence-length floor (>=20 um, 214 um at pinned Q=10) is the defensible fidelity
   scale. But this touches the LOCKED feasibility set, so it is your call, not mine.
2. **I do NOT need it to start.** The readout-time domain sweep I am running now is INTRA-core
   dominated, and the intra channel has no lambda (F = P_S_i*P_S_j, no spatial weight). So the
   flat-then-cliff collapse at ~107 s is lambda-independent and measurable today. lambda only gates
   the CROSS/two-timescale claim. I will measure the intra curve now and hold the cross
   interpretation for your ruling.

**One more framing item the advisor settled (no action needed, just adopting):** drop "program held
in superposition" — this is the (A) model (`quantum-system-canonical` section 5, LOCKED). Correct
phrasing: *stored correlation structure read out by a common-cause collapse; the readout is
classical, but MONOGAMY (degree<=4, no classical analogue) is what makes it informative.* I will
write L-PO8-1 at that altitude.

### UPDATE 2026-07-20 (external lit review) — Q3 has an answer, and it INVERTS the advisor.
`docs/PO8_EXTERNAL_LIT_REVIEW_2026-07-20.md`. Adversarially-checked primary sources:
- **KEEP the short falloff (lambda ~= 5um), do NOT move to 214um.** The advisor argued lambda=214um
  (unity amplitude across the coherence domain, remote ~= local). The broader literature contradicts
  that: the 214um is an ACOUSTIC-phonon length conflated with OPTICAL Trp superradiance (Babcock
  2024, Celardo/Kurian 2019 — coherent scales are sub-um to ~1um); the delocalised superradiant
  state LOCALISES at critical disorder ~10/cm while physiological disorder is ~200/cm (~20x above);
  and Reimers 2009 rules out the coherent Frohlich condensate unity-across-domain needs. **Unity
  remote entanglement is the model's WEAKEST assumption.** So the two-timescale trace (cross ~74s,
  intra ~107s at lambda=5) is the more defensible reading — which is good, it is the richer
  (scope-contracting) computation. My recommendation on Q3 flips: **keep lambda~5um**, and the
  advisor's category-error critique, while correct that 5um was originally a metabolic length, points
  to the WRONG replacement. This is exactly the internal+external cross-check catching a shaky push.
- **The eligibility trace only needs to survive ~2-40s, NOT ~200s** (Bittner/Magee 2017 BTSP ~2s;
  Jain 2024 CaMKII/DDSC 20-40s — the same Jain window the model's analytical_gap already cites). The
  substrate's ~200s over-provisions 5-100x. **Consequence:** a domain that lives ~tens of seconds is
  ON TARGET, not a failure. My retracted L-PO8-1's ~7-30s number was measured via a broken protocol,
  but even a correctly-measured tens-of-seconds trace would be biologically RIGHT — the "contradicts
  the 200s thesis" alarm was itself based on an over-provisioned assumption.
- tau_dimer=200s and T_singlet=216s are theory-only / unobserved / disputed x10^3 (Player&Hore
  ~seconds). Flagged, not actionable by this PO, but it means the trace target is the behavioural
  ~2-40s window, which is what Unit B now measures against.
