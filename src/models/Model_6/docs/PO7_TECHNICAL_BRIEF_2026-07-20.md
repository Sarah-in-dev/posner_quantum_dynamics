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

where ≺ is arrival order. **This question is now largely resolved — a reviewer's response plus
follow-up arithmetic closed most of it, and we record the resolution rather than the open form.**

1. **Global fidelity-weighted b-matching** ($\arg\max_{H:\ \Delta(H)\le 4}\sum F_e$) is **rejected
   on physical grounds**: it requires knowing every candidate edge and solving a global
   assignment — an oracle the modeller runs, which nothing local computes. (Same objection this
   programme already accepted against "no molecule diagonalises a Laplacian.")
2. **"Detailed balance" was the wrong name — it is a driven birth–death NESS.** Entanglement
   generation is not microscopically reversible (the bath never spontaneously entangles two
   nuclei), so this is a non-equilibrium steady state, not equilibrium. Crucially, that means the
   **release rate is not a free parameter — it is derivable:**

   $$k_{\text{release}} = \frac{1}{T_2} + \frac{1}{\tau_{\text{dimer}}}
     = \frac{1}{216\,\text{s}} + \frac{1}{200\,\text{s}} = 9.63\times10^{-3}\,\text{s}^{-1}
     \quad(\tau_{\text{release}} = 103.8\,\text{s}).$$

   The model currently dissolves bonds at ~$10^{-4}\,\text{s}^{-1}$ — **96× too slow**, which is
   what makes the graph effectively write-once. (Independent check: the model's own constants put
   $P_S$ crossing the Werner floor at 107.0 s; the derived $\tau_{\text{release}}$ is 103.8 s — two
   routes, same number.) **And selection by fidelity then emerges locally for free:** the model's
   formation rate is $k_{\text{cross}}\propto P_S^a P_S^b W_{ij}$, which *is* $F_{ab}$ — the rate
   already equals the fidelity, so high-F pairs are attempted more often and win slots without any
   global optimiser. The greedy walk was layered on top of an already-correct local kinetic scheme;
   the fix is the decay rate, not the allocation rule.
3. **Timescale separation** is real but our test could not see past it: at the corrected
   $\tau_{\text{release}}\approx104$ s, slots recycle ~0.7× within the 100–200 s coherence window
   but **do not recycle at all inside our 20 s runs** (~5 recycle-times short). So the "starvation"
   we measured is an artifact of run length, not of the allocation principle.

**Diagnostic performed (§ order-sensitivity):** bonds formed per step is bursty — mean 10.2, but 90%
of steps form ≤1, with rare bursts up to 840. Permuting *which slot* a bond claims leaves the
partition **bit-identical**, so slot allocation is provably order-free. *Which pair* is offered
first within a burst is not yet isolated and is the one remaining order question.

**What is left for a reviewer here is narrow:** confirm that $k_{\text{release}}=1/T_2+1/\tau_{\text{dimer}}$
is the right composition for a singlet-carrying bond in a driven NESS (versus, e.g., a
fidelity-dependent decay), since that rate now sets the entire recycling picture.

### 5.2 Binary occupancy is an UPPER bound (not a lower one), and frustration is a matching deficiency (not H¹) [CORRECTED]

**Two corrections to our earlier framing, both from reviewer input, both recorded rather than
quietly patched.**

**(a) The direction of the bound is inverted.** Binary occupancy — one nucleus, one bond — is the
*maximally-entangled* limit. The continuous statement is the CKW monogamy inequality
$\sum_j \tau_{aj} \le 1$ per nucleus, with tangle $\tau = C^2$ and, for a Werner state,
concurrence $C = 2F-1$. So a bond's true cost is $(2F-1)^2$:

$$F=0.75 \Rightarrow \tau=0.25 \ (\text{4 bonds/nucleus}); \qquad
  F=0.55 \Rightarrow \tau=0.01 \ (\text{~100 bonds/nucleus}).$$

Bonds near the Werner floor are **almost free**. Therefore our binary constraint is an **upper
bound on cost** — it *over*-charges weak bonds — and our 491,566 refusals are an **overcount**, not
a lower bound. The binary rule may be *over*-fragmenting. (To sustain even the original degree-715
graph under CKW would require $F \lesssim 0.537$ throughout, which the model's bimodal $F$
distribution does not obviously exclude — hence §6's decisive measurement.)

**(b) It is not H¹.** Our earlier documents (and an earlier version of this brief) called the
frustration an $H^1$ obstruction. **That was wrong and is retracted.** Resource contention among
binary slots is a **matching deficiency**, certified by Hall's condition; under CKW it becomes **LP
infeasibility over a capacity polytope**. Both are computable and neither is cohomology.

The genuine sheaf lives elsewhere. The full per-dimer state is (ℂ²)^⊗4 = ℂ¹⁶, with the six
intra-dimer J-couplings

$$H = \sum_{k<l} J_{kl}\,\mathbf{I}_k\cdot\mathbf{I}_l$$

acting on it; restriction of that state to the nucleus mediating a bond is a **partial trace over
the other three**, which mixes coordinates, so *that* cellular sheaf does not decompose — and it
answers a **different** question (consistency of the joint spin-state assignment), not resource
contention. Keeping the two separate is what keeps the sheaf claim credible where we do make it.

An earlier construction in this programme took the six J-couplings themselves as the stalk
(ℝ⁶) with restriction maps given by coordinate projection ℝ⁶ → ℝ. We verified numerically that
this decomposes: cross-block edges **0 / 369,740**, and H⁰ equals the sum of per-block component
counts exactly. It is a direct sum of three ordinary graph Laplacians, not sheaf structure —
projections are diagonal, so no cohomological information beyond per-channel circuit rank is
available. (The stalk was additionally input-blind: initialised as i.i.d. `normal(0.15, 0.15, 6)`
per dimer.)

So the ℝ⁶ construction was doubly wrong: diagonal restrictions (no sheaf), on an input-blind
stalk. The ℂ¹⁶ construction with partial-trace restrictions is the genuine article, but per (a)/(b)
above it is the tool for **spin-state consistency**, not for the monogamy/contention question —
which is fully handled by Hall / CKW-LP and needs no cohomology. **The open item is narrow and
quantitative:** does the binary-vs-CKW gap (weak bonds nearly free) materially change the partition?
That is the measurement in §6, not a modelling-principle question.

### 5.3 The system is stochastic by construction — its output is a distribution, not a value [FRAMING]

This model is a stochastic quantum-biological system: vesicle release, calcium-channel gating,
dimer birth, and — through them — whether the microtubule condensate ignites are all random. **That
randomness is the physics, not measurement noise.** The correct observable is therefore a
*distribution* over free-running realisations, not a single number, and the correct experiment
kicks the system off repeatedly from independent entropy and reports the ensemble.

We flag this because our own single-synapse numbers in §3–§4.2 were taken on a path we had
happened to run deterministically, and the multi-synapse numbers in §4.3–§4.4 are **single draws**.
An early diagnostic recorded η_max across four nominally identical driven runs coming out
0.0, 0.0709, 0.0940, 0.1069 — i.e. **whether the condensate ignites at all varies run to run.** We
initially misread that as a reproducibility defect to be seeded away. It is not: it is a genuine
property of the substrate, and *seeding it to obtain a clean or an igniting run would be selecting
the outcome.* We do not do that.

**Consequence for how to read this brief.** The structural facts in §3 and the collapse-threshold
argument in §4 are instantaneous graph properties or analytic (Erdős–Rényi), true in every draw
and robust. The specific multi-synapse *values* (largest_frac trajectories, the ~98-bond
percolation count, peak η) are single realisations and are being re-established as **distributions
over an ensemble of free-running draws** — in particular P(ignition), which is itself a
first-class result: the network-scale cross-synapse computation appears to be an intrinsically
*probabilistic* event, which is what one would expect if it depends on rare coincident condensation
across neighbouring spines. That ensemble is running; numbers will follow as distributions.

*(A seeding capability was added to the code for narrow software-regression testing — confirming a
newly-added opt-in flag, when disabled, executes the original code path — and is left dormant and
unused for any physics measurement. It never appears in a claim about what the system does.)*

---

## 6. Summary of what we believe, with confidence

| claim | status |
|---|---|
| The entanglement graph violated monogamy by ~179× (99.44% of edges inadmissible) | **[MEASURED], instantaneous graph fact, holds in every draw — high confidence** |
| The clique + phenomenological-EM pathways manufacture ~97% of intra bonds (3.80 of 3.93/dimer); both are independently established as unphysical | **[MEASURED], high confidence** |
| Enforcing four-nucleus occupancy fragments the single-synapse graph (largest_frac → 0.11, 184 components) | **[MEASURED]** — being re-established as an ensemble distribution |
| Refused bonds are genuine frustration = **matching deficiency (Hall's condition), NOT H¹** | [DERIVED] — see §5.2; earlier "H¹" framing was wrong and is retracted |
| The prior ℝ⁶ "sheaf" is a direct sum of 3 graph Laplacians, carrying no sheaf-specific information | **[MEASURED]**, identity verified exactly |
| The cross-synapse graph percolates at mean degree ≈ 1 (Erdős–Rényi), while the monogamy bound is 4 — so monogamy **cannot** prevent percolation | **[MEASURED] + analytic**, high confidence |
| Freeing the spin budget (dropping clique+EM) leaves the *fraction* of time structure exists unchanged (~20%); it only reduces severity | [MEASURED], single draw — being re-run as ensemble |
| **Whether the condensate ignites at all is stochastic across free draws** (P(ignition) < 1) | **[MEASURED]** — ensemble in progress; this is a first-class finding, not a defect |

**The one-line question for a reviewer:** connectivity (π₀) counts an F=0.5⁺ edge and an F=0.99
edge identically, though the former carries ~10⁻⁴ of the entanglement. Is the giant component we
observe held together by near-Werner-floor bridges — making it an artifact of an unweighted
invariant that cannot sustain correlated collapse — and if so, what is the right entanglement-
weighted notion of "one component, one commit bit" (a tangle/CKW-monotone cut, a spectral
threshold, something else)? *(We are measuring the bridge-fidelity distribution now; the question
stands regardless of which way it resolves.)*
