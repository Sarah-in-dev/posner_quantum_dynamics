> # ⚠ SUPERSEDED 2026-07-20 by the reframe — use `PO7_ADVISOR_REVIEW_2026-07-20.md`.
>
> This document's central question (§6: "does a cluster whose bonds top out at F≈0.8 sustain a
> jointly correlated collapse?") is now **answered**: joint collapse is **graded, not thresholded**
> — Werner correlation p=(4F-1)/3 multiplies along paths, so the computational unit is the
> **correlated domain**, not the connected component, and there is no fidelity cliff. The premise
> that we must "prevent percolation" (§7.3) is also superseded — connectivity and correlation are
> different lengths. See `L·PO7-5` and the current advisor review. Kept for the record of how the
> question was posed before it resolved.

# Request for outside guidance
### A quantum-biological synaptic model that builds a real but weakly-entangled network — and the physics question that now gates it
**2026-07-20 · self-contained; assumes no prior contact with this project**

---

## 0. What we are asking you

We have a computational model of quantum entanglement in a network of synapses. Over the last
several sessions we removed the mechanisms that were making its results meaningless, and we now
have a clean, physically-admissible result. That result is **genuine but weak**, and whether
"weak" is fatal is a physics question we cannot settle from inside the model. **We would like your
judgement on one specific question (§6), plus a sanity check on three modelling choices (§7).**

Everything below is measured unless marked otherwise. Where we were wrong earlier this project, we
say so — those corrections are part of what we need you to pressure-test.

---

## 1. The system in one paragraph

The proposed substrate for a long-lived neural "eligibility trace" is the **Ca₆(PO₄)₄ dimer**, a
calcium-phosphate cluster carrying **four ³¹P nuclear spins**. Dimers form in synapses and can
become quantum-entangled with each other. The model's claim is that the **pattern of entanglement
across a network of synapses is the computation**: when the system reads out, each connected
cluster of entangled dimers collapses together as a unit — "one shared coin per cluster." So the
object of interest is the **graph** of entanglement bonds, and specifically how it partitions into
clusters.

There are two ways dimers entangle:
- **Local** (within one synapse): two dimers that inherit phosphates from the same ATP-hydrolysis
  event are born correlated. Pairwise, provenance-based.
- **Cross-synapse** (between synapses): mediated by a microtubule "backbone" that enters a
  coherent (Fröhlich-condensate) state once its metabolic power crosses a threshold. This is the
  channel that lets separate synapses form a *joint* computation. It only switches on ("ignites")
  when the drive is strong enough.

---

## 2. What was broken (and is now fixed)

The graph used to collapse into a single **blob** — every dimer connected to every other, one giant
cluster, which computes nothing (one cluster = one bit, no structure). We found three reasons, all
now corrected:

1. **The graph was physically impossible.** Four spins per dimer means a dimer can hold **at most 4
   bonds**. Measured: mean **715 bonds per dimer**, 99.44% of edges physically inadmissible. Two
   bonding mechanisms in the code — a "same-birth-window clique" rule and a phenomenological
   electromagnetic pathway — were manufacturing **97%** of the bonds, and both were already known
   on independent grounds to be unphysical (they cannot create entanglement).

2. **We enforced the physical bound.** Giving each dimer four spin-slots and requiring every bond
   to consume one at each end (monogamy of entanglement — a spin maximally entangled with one
   partner cannot entangle another) drops the graph to a legal, admissible object. On a single
   synapse this alone breaks the blob (one cluster → 184 clusters).

3. **The graph was effectively write-once.** Bonds dissolved at ~10⁻⁴ s⁻¹, but the physical rate
   (spin coherence + cluster lifetime) is `1/216 + 1/200 ≈ 9.6×10⁻³ s⁻¹` — **~100× faster**. So
   bonds that should recycle within the ~100–200 s computational window instead persisted forever.
   *(Not yet changed in the running model; flagged for the next build.)*

---

## 3. The clean result (16 free-running trials, no random seed fixed anywhere)

The system is stochastic — vesicle release, ion-channel gating, dimer birth are all random, and
**that randomness is the physics, not noise to be averaged away.** So we run it many times and
report distributions. Sixteen independent free runs, 7 synapses, 20 s each:

| quantity | result |
|---|---|
| **P(the backbone ignites)** | **16 / 16 = 1.00** (reliably, not rarely) |
| peak condensation η (igniting runs) | min 0.334, median 0.439, max 0.493 |
| **largest cluster, as fraction of all dimers** | min 0.935, **median 0.968**, max 0.992 |

So even with the physical bound enforced and the unphysical mechanisms removed, **the network still
percolates into one dominant cluster in every trial.** We then asked whether that cluster is real
or a measurement artifact.

---

## 4. Is the giant cluster real? — Yes. (This is the key measurement.)

A connected-component count treats every edge as equal: an edge of fidelity F = 0.501 (barely
entangled — carries ~0.0004 of the maximum possible correlation) joins two clusters exactly as
firmly as an edge of F = 0.99. So the giant cluster *could* be held together by near-worthless
edges, in which case it would be an artifact of using an unweighted measure.

We measured the fidelity of the **bridges** — the specific cross-synapse edges whose removal would
split the giant cluster. Pooled over all 16 runs (28,673 cross edges, of which 1,557 are bridges):

| | all cross edges | bridges only |
|---|---|---|
| median fidelity F | 0.679 | **0.672** |
| fraction near the floor (F < 0.55) | 8.5% | 9.5% |
| **maximum F anywhere** | **0.815** | 0.809 |
| median tangle τ = (2F−1)² | 0.128 | 0.119 |

**The bridges are not floor-level.** Median F ≈ 0.67, only ~9.5% near the worthless zone. They
carry real correlation (τ ≈ 0.12, ~12% of the maximum). **The giant cluster is physically real —
not an artifact of unweighted counting.** Connectivity was the right thing to measure after all.

---

## 5. The new puzzle: real, but capped at F ≈ 0.8

Here is what we did not expect. **No cross-synapse edge, anywhere, across ~29,000 of them in 16
independent runs, exceeds F ≈ 0.815.** The distribution has a hard ceiling well below the
maximally-entangled limit (F = 1). Local intra-synapse bonds reach much higher; the *cross-synapse*
channel does not.

So the network-scale computation is **genuine but only modestly entangled.** It exists, it's
reliable, it's physically legal — and it lives permanently in an intermediate-fidelity regime.

---

## 6. THE QUESTION FOR YOU

The model's central claim is **"one shared coin per connected cluster"**: at readout, every dimer
in a cluster collapses *jointly*, producing one correlated bit. That picture is comfortable when the
bonds are near-maximal. It is not obvious it survives when the bonds holding a cluster together are
**Werner states of intermediate fidelity (F ≈ 0.67 typical, 0.8 ceiling, τ ≈ 0.12)**.

**Does a connected cluster whose inter-synaptic bonds top out at F ≈ 0.8 actually sustain a jointly
correlated collapse — or does correlated collapse require a fidelity floor above what this
condensate channel can deliver?**

Concretely, three sub-questions:
1. Is there a **threshold fidelity** below which "joint collapse of a connected component" stops
   being a meaningful approximation — i.e. the component's dimers decorrelate faster than they
   co-collapse? If so, roughly where is it, and does F ≈ 0.67–0.8 clear it?
2. Should the computational object be the **connected components of the F > ½ graph** (what we
   use), or a **tangle-weighted / entanglement-monotone** notion where an F ≈ 0.5 edge simply does
   not bind two dimers into the same computational unit? (Note: a hard F-threshold cut is a knob we
   refuse to tune; we want a *principled* invariant, not a fitted one.)
3. Is the **F ≈ 0.8 ceiling** on the cross-synapse channel a red flag (the channel cannot deliver
   what a quantum computation would need) or expected physics (condensate-mediated entanglement is
   *supposed* to be modest, and modest is enough for a classical-correlation "(A)-model" readout)?

We are explicitly agnostic and will follow the physics. A "your cluster is too weak to collapse
jointly, the keystone fails at this fidelity" answer is as valuable to us as the reverse.

---

## 7. Three modelling choices we want sanity-checked

1. **Bond release rate.** We claim it should be `k = 1/T₂ + 1/τ_dimer` (spin decoherence +
   cluster dissolution) for a singlet-carrying bond in a driven non-equilibrium steady state. Is
   that the right composition, or does a Werner bond of fidelity F decay at a fidelity-dependent
   rate we're missing? This rate sets whether the graph recycles within the coherence window.

2. **Monogamy as binary vs. continuous.** We currently treat a spin as a binary slot (free/used).
   The continuous statement is CKW: Σ tangle ≤ 1 per spin, so a weak (F ≈ 0.55) bond costs almost
   nothing and a spin could carry many of them. Our binary rule therefore *over*-charges weak bonds
   and may over-fragment. For a system operating at F ≈ 0.67, is the binary approximation
   defensible, or does the intermediate-fidelity regime specifically demand the continuous version?

3. **Percolation vs. bound.** The graph percolates into a giant cluster at mean degree ≈ 1 (the
   standard random-graph threshold), while the monogamy bound is 4. So monogamy *cannot* prevent
   percolation — it's four times too permissive. Is preventing a giant cluster even the right goal,
   or is a percolated-but-weakly-bound network the correct picture of a memory with capacity, where
   what matters is the *weighted* structure inside the giant cluster rather than its existence?

---

## 8. What we are confident of vs. not

| statement | confidence |
|---|---|
| The graph violated the 4-spin monogamy bound ~179× before the fix | **high** — instantaneous count, every trial |
| The clique + EM pathways manufactured ~97% of bonds and are unphysical | **high** |
| Enforcing monogamy breaks the single-synapse blob | **high** |
| The bond release rate is ~100× too slow in the current code | **high** — arithmetic from the model's own constants (two independent routes agree: 103.8 s vs 107.0 s) |
| Ignition is reliable (16/16), not a coin-flip | **high** — corrects our own earlier error (a drive-path bug had left the backbone un-driven) |
| The giant cross-synapse cluster is physically real (bridges F ≈ 0.67, not floor) | **high** — 16 free runs, 28,673 edges |
| The cross-synapse channel is capped at F ≈ 0.8 | **high** — hard ceiling across all runs |
| Whether F ≈ 0.8 sustains joint collapse | **UNKNOWN — this is §6, the ask** |

---

*Prepared by the model's maintainers. Underlying data: `results/po7_unit17_results.json` (16-run
ensemble); full derivation trail in `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries `L·PO7-1`
through `L·PO7-4` and the companion `PO7_TECHNICAL_BRIEF_2026-07-20.md` (same content, more
equations).*
