# Technical brief: entanglement-graph structure in a Posner-dimer synaptic model
### Prepared for external review · 2026-07-20 · Model 6 / entanglement-topology sub-programme

**Purpose.** We have a result we believe is correct and a follow-on question we do not know how
to answer well. This document states the model formally, gives the measurements, and isolates
three open problems — one of which is a modelling-principle question we would like outside help
on. It is self-contained; no familiarity with the codebase is assumed.

**Epistemic markers used throughout:** [MEASURED] = observed in the running model;
[DERIVED] = follows from stated constants; [MODELED] = a defensible modelling choice, not forced;
[OPEN] = we do not know.

---

## 1. The physical setting

The substrate is the **Ca₆(PO₄)₄ dimer** — a calcium-phosphate cluster carrying **four ³¹P
spin-½ nuclei**. (This is the dimer, not the Ca₉(PO₄)₆ Posner trimer, which lacks a rotational
symmetry axis and decoheres sub-second; the dimer is the computational object here. Agarwal 2023.)

Dimers form stochastically in a dendritic spine, may become mutually entangled, and the claim
under test is that the **partition of the entanglement graph into connected components** is the
computational output: readout collapses each component jointly, one correlated bit per component.

Two entangling channels exist in the model:

**(i) Local (intra-synapse), Fisher inheritance.** ATP hydrolysis releases two phosphates in a
correlated (singlet) state. Two dimers that capture the two daughters of one hydrolysis event
inherit that correlation. This is strictly pairwise and provenance-based.

**(ii) Cross-synapse, condensate-mediated.** A microtubule backbone enters a Fröhlich-type
condensate when metabolic power exceeds a threshold; the condensate mediates entanglement between
dimers in *different* spines. Phosphates do **not** travel between synapses — this channel is
field-mediated, not diffusive.

---

## 2. The model, formally

### 2.1 Condensation (the cross-synapse gate)

Critical power, from the Fröhlich condition n_ex ≥ n̄_s:

$$P_c \;=\; \bar n_s\,\hbar\,\omega_{\text{ang}}^2 / Q, \qquad \omega_{\text{ang}} = 2\pi\omega_0$$

with ω₀ = 8×10⁶ Hz, Q = 10, n̄_s = 8.074×10⁵ (Bose–Einstein occupation at ω₀), giving

$$P_c = 21.514\ \text{fW} \qquad \text{[MEASURED, reproduces the documented value]}$$

Per-synapse metabolic power, where E_inv ∈ [0,1] is a microtubule-invasion envelope driven by
actin reorganisation and c ∈ [0,1] is the calcium-channel open fraction:

$$P^{(i)}_{\text{met}} \;=\; P_{\text{basal}} + E^{(i)}_{\text{inv}}\, c^{(i)}\, P_{\text{act}}^{\max}$$

$$P_{\text{basal}} = 0.84\ \text{fW}, \qquad P_{\text{act}}^{\max} = 60.0\ \text{fW}$$

Only the **active** component aggregates spatially (basal is per-spine and never summed):

$$P^{(i)}_{\text{agg}} \;=\; P_{\text{basal}} + \sum_j W_{ij}\,\bigl(P^{(j)}_{\text{met}} - P_{\text{basal}}\bigr), \qquad W_{ij} = e^{-d_{ij}/\lambda},\ \ \lambda = 5\ \mu\text{m}$$

$$r_i = P^{(i)}_{\text{agg}}/P_c, \qquad \eta_i = \begin{cases}\dfrac{r_i-1}{r_i+1} & r_i \ge 1\\[4pt] 0 & r_i < 1\end{cases}$$

Note the ceiling: with E_inv = c = 1, r = (P_basal + P_act^max)/P_c = **2.828**, so
η_max ≈ 0.478. Condensation is reachable but not by a wide margin. [DERIVED]

### 2.2 Bond formation

Cross-synapse formation rate between dimers a ∈ spine i, b ∈ spine j:

$$k^{ab}_{\text{cross}} \;=\; K\,\sqrt{\eta_i\eta_j}\;W_{ij}\;P^a_S P^b_S, \qquad K = 0.5$$

hard-gated on both spines being microtubule-invaded, with per-step formation probability
p = 1 − e^{−k dt}. **η enters as a geometric mean, so η_i = 0 ⇒ k = 0 identically** — a single
un-condensed endpoint zeroes the channel regardless of anything else.

Edges are **Werner states**; an edge counts toward connectivity only above the separability bound:

$$F_{ab} = P^a_S P^b_S W_{ij}, \qquad \text{edge admitted iff } F_{ab} > \tfrac12$$

(Werner 1989. This bound is treated as fixed physics in this programme and has not been moved.)

### 2.3 The object of interest

Let G = (V, E) with V the entangled dimers and E the admitted edges. The computational output is
the partition of V into connected components. Two summary statistics are used throughout:

- **largest_frac** = |largest component| / |V|
- **n_multi** = number of components containing dimers from ≥2 spines

largest_frac → 1 means the partition is trivial: one component, one bit, no structure.

---

## 3. The defect we found

**Monogamy of entanglement was not represented.** A dimer has four ³¹P nuclei; a singlet-strength
bond consumes one nucleus at each endpoint; a nucleus maximally entangled with a partner cannot
also entangle another. Hence

$$\deg(v) \le 4 \quad \forall v \in V \qquad \text{[DERIVED from the molecule]}$$

The model bonded dimers as featureless nodes, with no occupancy accounting. Measured on the
standard single-synapse rig (V = 1034 entangled dimers, 200 steps):

| quantity | measured | bound |
|---|---|---|
| mean degree | **715.16** | 4 |
| max degree | **902** | 4 |
| dimers exceeding bound | **1034 / 1034 (100%)** | 0 |
| edges | 369,740 | ≤ ⌊4·1034/2⌋ = **2,068** |
| **inadmissible edges** | **367,672 (99.44%)** | — |

[MEASURED] The graph exceeded the physical bound by a factor of **179**, and 99.44% of its edges
could not exist.

---

## 4. What we built, and the result

### 4.1 The constraint

Each dimer v carries an occupancy vector σ(v) ∈ ({⊥} ∪ E)⁴. A bond e = (u,v) is admitted only if
a free slot exists at **both** endpoints:

$$\exists\,s,t \in \{1..4\}: \ \sigma_s(u) = \bot \ \wedge\ \sigma_t(v) = \bot$$

on which σ_s(u) ← e, σ_t(v) ← e. Removal releases both. Provenance-inherited bonds must claim a
**named** slot (the inherited nucleus occupies a specific position), so they are not
interchangeable — this is what makes them frustratable.

**Degree ≤ 4 is derived, never imposed as a cap.** Bonds refused for want of a free slot are
counted separately; they are not deleted edges but **frustration** — pairs individually
satisfiable, jointly not.

### 4.2 Single synapse: the blob breaks [MEASURED]

Same rig, occupancy accounting off vs on:

| | OFF | ON |
|---|---|---|
| edges | 369,740 | **2,031** (0.55%) |
| mean degree | 715.16 | **3.93** |
| max degree | 902 | **4** |
| dimers over bound | 1034 | **0** |
| components | 1 | **184** |
| **largest_frac** | **1.000** | **0.112** |
| frustrated (refused) bonds | — | **491,566** |

The graph goes from a single component containing every dimer to 184 components with the largest
holding 11%. This is the first physically admissible entanglement graph the programme has
produced, and it is not trivial. Nothing was tuned; the Werner bound was untouched.

### 4.3 Why the unconstrained graph was a blob [MEASURED]

At the multi-synapse scale (7 spines, 1 µm spacing), before condensation each spine is its own
component — largest_frac = 0.22, exactly as expected for one nanodomain per spine. Then:

```
t = 9.05 s   components = 7   largest_frac = 0.218   cross-bonds = 0
t = 9.55 s   components = 1   largest_frac = 1.000   cross-bonds = 98
```

**98 cross-synapse bonds — all already above the Werner bound — collapse 100% of dimers into one
component.** Because each spine was internally a near-complete clique (the degree-715 violation),
a single admitted cross-bond fuses two dense balls wholesale. Percolation is trivial when the
parts are already complete graphs.

### 4.4 Network-wide occupancy: cross-synapse structure vanishes [MEASURED]

Extending the ledger so intra- and cross-synapse bonds spend from the *same* four nuclei
(7 spines, 12 s):

| quantity | value |
|---|---|
| max total degree (intra + cross) | **4** (0 violations, all 240 samples) |
| mean total degree | **3.69** — ≈92% of nuclei spent on intra bonds |
| cross-synapse bonds ever formed | **1** (none above the Werner bound) |
| cross-bond frustration | 710 |
| components / largest_frac / **n_multi** | 373 / 0.0119 / **0** |

Monogamy holds globally and the blob is gone — but **no cross-synapse structure survives at all**
(n_multi = 0 throughout, versus a peak of 33 when only intra bonds were constrained).

**This is the crux.** With four nuclei per dimer and dense local formation, the local channel
consumes essentially the entire spin budget, leaving nothing for the network scale.

---

## 5. Three open problems

### 5.1 ⚠ Is the starvation physical, or an artifact of the update order? [OPEN — the main ask]

Bonds are admitted by a **greedy sequential matching**: the simulation walks candidate pairs in
whatever order the update loop produces and claims slots first-come-first-served. Intra-synapse
bonds form from t ≈ 0; condensation ignites at t ≈ 10 s. **By the time the cross-synapse channel
opens, the slots are gone.**

Nothing in the physics privileges local bonds. The outcome we measured may be a property of the
algorithm, not the model. Formally, we are computing one maximal (not maximum) constrained
subgraph:

$$\mathcal{G}^\ast = \text{greedy}_{\prec}\bigl\{\,e \in E : \deg_\sigma(u),\deg_\sigma(v) < 4\,\bigr\}$$

where ≺ is arrival order. The physically meaningful object is presumably not this. Candidates:

1. **Fidelity-weighted matching** — admit edges in decreasing F, i.e. approximate
   $\arg\max_{H \subseteq E,\ \Delta(H)\le 4} \sum_{e\in H} F_e$. A degree-constrained maximum-weight
   subgraph (b-matching, b ≡ 4), solvable exactly in polynomial time.
2. **Detailed balance** — bonds form *and* break with rates satisfying a stationarity condition, so
   occupancy equilibrates rather than latching. Physically the most defensible; the current
   dissolution rate may simply be far too slow relative to formation.
3. **Timescale separation as physics** — if local bonds genuinely form and lock before the
   condensate ignites, first-come-first-served *is* the physics, and starvation is the answer.

**We do not know which is right, and the three give qualitatively different partitions.** This is
where we would most value outside judgement: *what is the correct variational or kinetic principle
for allocating a conserved, discrete, per-node resource among competing entangling channels
operating on different timescales?*

### 5.2 Is occupancy counting a faithful H¹, or only a lower bound? [OPEN]

We treat a nucleus as a binary slot: free or spent. The full description is a state in
(ℂ²)^⊗4 = ℂ¹⁶ per dimer, with the six intra-dimer J-couplings

$$H = \sum_{k<l} J_{kl}\,\mathbf{I}_k\cdot\mathbf{I}_l$$

acting on it. Under that description, restriction of a dimer's state to the nucleus mediating a
given bond is a **partial trace over the other three** — which mixes coordinates, so the
associated cellular sheaf does **not** decompose.

An earlier construction in this programme took the six J-couplings themselves as the stalk
(ℝ⁶) with restriction maps given by coordinate projection ℝ⁶ → ℝ. We verified numerically that
this decomposes: cross-block edges **0 / 369,740**, and H⁰ equals the sum of per-block component
counts exactly. It is a direct sum of three ordinary graph Laplacians, not sheaf structure —
projections are diagonal, so no cohomological information beyond per-channel circuit rank is
available. (The stalk was additionally input-blind: initialised as i.i.d. `normal(0.15, 0.15, 6)`
per dimer.)

Our 491,566 frustration events are the **combinatorial shadow** of monogamy. The question:
**is that count a faithful H¹, or a lower bound that ignores partial entanglement?** Whether
nucleus *a* can still bond depends, in the full theory, on how it is entangled with the other
three in its own dimer — a continuous quantity our binary occupancy cannot see. If the answer is
"lower bound," the ℂ¹⁶ construction becomes necessary rather than optional, and we would want to
know whether a tractable reduction (e.g. restriction to the singlet–triplet manifold) preserves
the obstruction.

### 5.3 The rig is not reproducible [MEASURED — being fixed]

Two runs of identical code at a fixed seed diverge (three RNG instances are constructed without a
seed argument and draw from OS entropy). Consequence: **all multi-synapse numbers in §4.3–§4.4 are
single draws from a distribution, not reproducible values.** An independent record shows η_max
across four nominally identical driven runs coming out 0.0, 0.0709, 0.0940, 0.1069 — i.e. *whether
the condensate ignites at all* varies run to run.

The single-synapse results in §3 and §4.2 are on a deterministic path and are reproducible.

**Nothing in §4.3–§4.4 should be treated as a measurement until the seeding is fixed and the runs
are repeated across seeds.** We state it rather than omit it.

---

## 6. Summary of what we believe, with confidence

| claim | status |
|---|---|
| The entanglement graph violated monogamy by ~179× (99.44% of edges inadmissible) | **[MEASURED], reproducible, high confidence** |
| Enforcing four-nucleus occupancy fragments the graph: largest_frac 1.000 → 0.112, 184 components | **[MEASURED], reproducible, high confidence** |
| Refused bonds constitute genuine frustration (pairwise satisfiable, jointly not) | [DERIVED] — but see §5.2 on whether it is the full obstruction |
| The prior ℝ⁶ "sheaf" is a direct sum of 3 graph Laplacians, carrying no sheaf-specific information | **[MEASURED]**, identity verified exactly |
| A handful (~98) of cross-bonds percolate the unconstrained graph completely | [MEASURED], single draw |
| Cross-synapse structure does not survive network-wide occupancy accounting | [MEASURED], **single draw, and confounded by §5.1** |
| Condensation is reachable (r up to ≈2.5, all 7 spines) and strobes rather than latches | [MEASURED], single draw |

**The one-line question for a reviewer:** given a conserved discrete resource (four nuclei per
node) contested by two entangling channels with a ~10 s timescale separation, is greedy
first-come allocation defensible physics, or must the allocation be derived from a stationarity
or optimality principle — and if the latter, which?
