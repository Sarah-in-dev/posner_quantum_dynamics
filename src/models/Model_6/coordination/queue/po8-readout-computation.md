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
