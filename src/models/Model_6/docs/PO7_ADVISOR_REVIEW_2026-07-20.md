# Advisor review — round 6: the reframe, and four checks on it
**Model 6 / entanglement-topology · 2026-07-20 · supersedes the R4 packet (withdrawn) and the R5 packet**

Your last response reframed everything, and we ran it down. This asks you to pressure-test the
reframe and four specific claims that now rest on it. We are not looking for reassurance — where a
claim is shaky we want it named. Every number below is from **free-running ensembles** (no fixed
seed — the stochasticity is the mechanism, and you were right that a fixed seed measures one
trajectory of the object). Data: `results/po7_unit17_results.json` (16 draws),
`results/po7_unit18_results.json` (12 draws); derivations in `L·PO7-1 … L·PO7-5`.

---

## 1. Scorecard on your R5 response

| your point | what we did / found |
|---|---|
| **Reject global b-matching (installs an oracle)** | Accepted. Not pursued. |
| **It's a driven NESS; k_release = 1/T₂ + 1/τ_dimer is derivable** | Computed: 1/216 + 1/200 = **9.63e-3/s** (τ≈104 s); the code runs ~1e-4/s — **96× too slow**. Independent check: the model's own constants put P_S at the Werner floor at 107.0 s; τ_release = 103.8 s. **Two routes agree.** Not yet implemented (it's the next PO's first unit). |
| **Fidelity selection emerges locally (k_cross ∝ F already)** | Confirmed in code: k_cross ∝ P_S^a P_S^b W = F. Agreed — the greedy walk sits on a correct kinetic scheme; the fix is the decay rate. |
| **Frustration is matching deficiency (Hall/CKW-LP), NOT H¹** | Corrected in all docs; the H¹ framing is **retracted**. The genuine ℂ¹⁶ partial-trace sheaf answers spin-state consistency, kept separate. |
| **Binary occupancy is an UPPER bound; CKW says weak bonds are ~free** | Confirmed (F=0.55 ⇒ τ=0.01 ⇒ ~100/nucleus). Our 491k "frustrations" **overcount**. CKW not yet adopted. |
| **Slot order is irrelevant; check bonds-per-step** | Permuting slot choice → **bit-identical** partition. Bonds/step bursty (mean 10, max 840); **pair** order not yet isolated (the one order question left). |
| **Run the correlation-length analysis on the ensemble** | Done — §3 below. It did **not** come out as you predicted, and we think the difference is instructive. |

---

## 2. The reframe, as we now hold it (please confirm or correct)

The entanglement graph is a **program**, written at dimer birth (provenance fixes it), held in
superposition, and **executed by dopamine-triggered decoherence** — the correlated collapse is the
readout. Therefore:
- **Writing** (does input determine the graph) and **reading** (how faithfully collapse expresses
  it) are separable, and **the fidelity ceiling cannot falsify the input-selectivity keystone.**
- The output is not a value but an **agreement pattern** — which dimers collapse correlated — read
  in O(1) at decision time. That is a partition readout in the strict sense.
- **Joint collapse is graded, not thresholded:** Werner p=(4F-1)/3, multiplying along paths; the
  computational unit is the **correlated domain** (effective correlation e^{-d}, d=Σ(-ln p_e)), not
  the connected component. No F-cut to nominate.

**Q1. Is this the right ontology?** Specifically: is it legitimate to treat the *pre-dopamine*
graph as a superposed program and the dopamine collapse as its execution, given the model is the
classical-correlation (A)-reading (scalar P_S, no phase)? Or does "program held in superposition"
overclaim what an (A)-model can carry, and we should say "a stored correlation structure read out
by a common-cause collapse"?

---

## 3. The correlation-length result — and where it diverged from your prediction

You predicted correlated domains ~2–5 dimers (ξ≈1.7 bonds at p≈0.56). Measured (12 free draws,
threshold-free correlation metric, bounded Dijkstra):

| quantity | median across draws |
|---|---|
| mean domain size S(u)=Σ_v e^{-d(u,v)} | **468 dimers** (~1.35 synapses) |
| effective domain count Σ 1/S(u) | **45** |
| connected components (≥2) | 17; largest_frac **0.978** (one giant) |

Domains are **synapse-scale, not 2–5.** The reason (verified directly): your ξ≈1.7 assumed uniform
F≈0.67, but **intra-synapse bonds run at F=1.0000** (P_S≈1, so p=1, w=0 — lossless), while only the
**cross**-bridges are p≈0.6. So the graph is near-lossless intra-synapse cores stitched by strong
bridges, not a uniform-p chain. Your qualitative claim **holds** — 45 correlated domains where
connectivity is pinned at 1 — but the unit is a synapse, not a handful of dimers.

**Our reconciliation, which we want you to check (Q2):** F_intra=1 only because P_S has not decayed
at our 20 s runtime (T_singlet=216 s). This is the **write-time** graph. At **readout** (dopamine
at a realistic delay) P_S has decayed, intra F falls toward 0.5, and domains fragment — so your
small domains and our large ones are **the same graph at different times**, related by your own
t_bond=T₂·ln(4F₀-1). **Is that the correct reconciliation** — i.e. is the *readout-time* domain
size (which we have not yet measured) the physically meaningful one, and should it land near your
2–5 estimate once P_S(t) is folded in? Or does the near-lossless intra core persist to readout and
the computational unit really is synapse-scale?

---

## 4. The ceiling is geometry, not η (a correction to your guess)

You expected the F≈0.815 ceiling to trace to η_max (the r=2.828 metabolic ceiling). It doesn't.
F = P_S·P_S·W_ij with W_ij=exp(-d/λ), λ=5µm; at 1µm spacing the nearest-neighbour weight is
exp(-1/5)=**0.8187**, and observed max across 28,673 edges is **0.8151** (gap = P_S<1). η enters the
*rate* (k∝√(η_iη_j)), not the fidelity. **So synapse spacing — not metabolism — sets the
per-bridge correlation, hence the inter-synaptic correlation length** (1µm→p≤0.76, 2µm→0.56,
3µm→0.40).

**Q3.** This makes a falsifiable biological prediction: correlated-domain size (and thus the
network computation's spatial reach) is set by spine spacing, with a ~5µm decay constant. Is that
the right reading, and is the λ=5µm condensate coupling length itself defensible, or is it the next
soft constant to pin?

---

## 5. The one physics question we cannot close ourselves

**Q4 — the eligibility trace.** Your t_bond=T₂·ln(4F₀-1) makes the graph self-clean: strong bonds
(F₀=0.815) live 176 s, weak (0.55) live 39 s, so the graph *at dopamine* is enriched in
high-fidelity bonds relative to birth, and structure written long before reward has decayed while
structure written just before it survives. **This looks like it derives temporal credit assignment
from T₂ alone** — the thing the architecture has asserted for years but never derived. Is that
real, or are we over-reading a dephasing curve? Concretely: does the *combination* of
(a) fidelity-dependent bond lifetime and (b) the readout being the correlated-domain partition
actually implement a behavioural-timescale eligibility trace, or is there a missing ingredient
(e.g. the write rate must also be fidelity-graded, or the dopamine gate must couple to F) before
"the trace falls out of T₂" is earned?

---

## 6. What we are NOT claiming
- Not that the readout-time keystone is settled — we have measured **write-time** domains only;
  the dopamine-at-realistic-delay experiment is the next PO's Unit B.
- Not (B) genuine quantum computation — this remains the (A) classical-correlation reading.
- Not that the release rate is fixed in the model yet — it's derived and queued, not implemented.
- Not that the ℂ¹⁶ spin-state sheaf is needed for the monogamy question — Hall/CKW handle that; the
  sheaf is a separate spin-consistency object and we are keeping them apart.

**The single decision that gates the next phase is Q2** (is readout-time the right clock, and does
folding in P_S(t) recover your domain-size estimate). Everything else we can run; Q1 and Q4 are
where your physics judgement is worth more than our simulation.
