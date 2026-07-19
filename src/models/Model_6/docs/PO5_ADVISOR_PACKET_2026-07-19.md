# Advisor packet — §8 keystone, round 2
**PO-5 · 2026-07-19 · after executing the round-1 directives**

Round 1 was reasoned from a narrative write-up. Two of four directives did not survive execution.
This packet leads with **what your directives produced**, then the **measurements**, then what would
**falsify my current reading** — so you can disagree from data rather than from my framing.

---

## 1. SCORECARD ON THE ROUND-1 DIRECTIVES

| directive | outcome |
|---|---|
| **"Unit 3 proves the keystone fails; it isn't underpowered"** | **CORRECT, and sharper than my version.** The P0 partition is a deterministic function of the sorted birth times. I had run the keystone test (Units 9/10) at **bus=0 — pure P0** — where pair-level selectivity is *structurally* absent. The test was incapable of a positive. Directive accepted; powered Unit 10 cancelled. |
| **Q1 "the clique is unphysical, not an overclaim"** | **PHYSICS CORRECT, DIAGNOSIS WRONG.** Coincidence is not entangling, product state across pairs, entanglement not transitive — all granted. **But removing the clique does not fragment the graph.** Degree cap k=4 removes **65% of all edges**; `largest_frac` stays **1.0000**. Even k=1 stays 1.0000. The clique was never what held the graph together. |
| **Q2 "a global multiplier is rank-one and cannot carry pair information"** | **FACTUALLY OFF.** `g` is *already* in the rate: `em_rate = k_base × (Φ/reference_kT) × coh × g`. The proposed fix — "let `Φ_ij = Φ₀·g(r_ij)` reach the bond rate" — already exists. Unit 1's measured `D = 33.5` **is** that term working. The rate matrix is not rank-one. *(I also briefly claimed a "137× too strong" mismatch here — that was me conflating the trp→dimer near-field with dimer–dimer dipolar coupling. Withdrawn.)* |
| **Q3/4/5 "is the sheaf Laplacian the spin Hamiltonian?"** | **PARTLY ACTIONABLE, and your correction stands.** The relaxation argument (`ẋ = −Lx` converges to `ker L`; no diagonalisation needed) is a genuine answer to A5 and I've adopted it. **Your caveat is the important part**: relaxation delivers a *section* `x* ∈ H⁰`, not `dim H⁰`. I had been treating the dimension as the readout — wrong. Not yet testable: the J-compatibility arm was null (below). |
| **Q6 "don't build the homeostat yet"** | **AGREED, unchanged.** And now better supported: the supercriticality is not artefactual-from-clique either (Q1), so the homeostat would have been correcting the wrong thing twice over. |
| **"latent dimension is the deep fix" (1D interval graph vs d≥2)** | **DEEPEST POINT MADE, STILL UNTESTED.** Routing ℝ⁶ to formation produced a **null** (−0.8% edges). But that tests one *tolerance rule*, not the dimensionality claim. Your argument may be right and my implementation just too permissive — see §4. |

---

## 2. THE MEASUREMENTS YOU DID NOT HAVE

### 2.1 Removing the clique does nothing (Unit 11, native field, 2 seeds)

| arm | edges | components | largest_frac |
|---|---|---|---|
| baseline | 407,332 | 1.0 | 1.0000 |
| degree cap k=4 (Fisher-consistent) | 140,697 (**−65%**) | 1.0 | **1.0000** |
| degree cap k=1 | 138,722 (−66%) | 1.0 | **1.0000** |
| J-compatibility on formation | 404,109 (−0.8%) | 1.0 | 1.0000 |
| coupling_length 5→1 nm | 372,170 | 20.0 | 0.9821 |
| **all three combined** | 105,536 (**−74%**) | 20.0 | 0.9821 |

**Two-thirds of edges can be deleted with zero effect on connectivity.** Density is not the problem.

*(Note: true per-event matching is **not implementable** — dimers are born from a concentration
field, `dimer_particles.py:210-213`. No pyrophosphate objects, no per-phosphate provenance. The
100 ms clique is a proxy for a representation the model does not have. The degree cap is the nearest
available proxy.)*

### 2.2 The fidelity cut is the only lever that moves (Unit 12)

Intra bonds have **no fidelity threshold** — connectivity is *bare existence*. Cross-synapse bonds
count only above the Werner bound. `model6-entanglement-partition-werner` flags the intra layer as
carrying *"the same dead-store pattern (1/r³ in rate, not in stored fidelity; bare-existence
connectivity)"* and pre-authorises the treatment *"if its own blob/saturation needs it."*

Storing `F = P_S_i · P_S_j · g(r_ij)` and sweeping the threshold (**no value nominated**):

```
F distribution over 857,193 bonds:
  p1=0.010  p10=0.019  p25=0.037  med=0.090  p75=0.335  p90=0.980  p99=0.994
```

| threshold | edges kept | components | largest_frac | sheaf H⁰ |
|---|---|---|---|---|
| 0.00 | 100% | 1.0 | 1.0000 | 3.0 |
| 0.10 | 47% | 2.0 | 0.9991 | 10.0 |
| 0.20 | 32% | 5.0 | 0.9957 | 10.5 |
| 0.30 | 26% | 7.0 | 0.9943 | 14.5 |
| 0.50 | 21% | 11.0 | **0.8314** | **15.0** |

**The F distribution is bimodal** — median 0.09, top decile 0.98 — split at the 5 nm plateau where
`g` saturates. **Sheaf H⁰ rises monotonically 3 → 15** as the cut sharpens: readout resolution and
graph structure moving together for the first time.

### 2.3 The pathway carrying the percolation has no microscopic mechanism

`dimer_particles.py`, the EM pathway's own docstring:

> *"The microscopic Hamiltonian for EM-mediated nuclear spin coupling in biological systems
> **remains an open question**. The UV-frequency tryptophan field (~10¹⁵ Hz) **cannot couple
> directly to nuclear spin dynamics (~Hz)** via standard dispersive mechanisms. Our
> **phenomenological treatment** captures…"*

**A ~15-order-of-magnitude frequency gap, self-declared, on the 17% pathway that Units 7 and 11 show
spans the graph by itself.** Nobody in this program has raised it. I think it is the most important
item in this packet.

---

## 3. CONSTRAINTS — what is locked, what is ungrounded

**Locked / do-not-re-derive:** the 20 kT trp-superradiance coupling (derived two ways;
`model6-network-layer-feasibility-may30`). `reference_kT = 20.0` ties to it. **Not in question.**

**Ungrounded, and load-bearing:**
- `coupling_length = 5.0` nm — recorded *"declared, never read"* on 2026-05-29, wired into the live
  rate later, **no citation anywhere.** A dormant constant became load-bearing unobserved.
- `k_entangle_em_base = 1.0` — function-body literal, unsweepable until this session. **Not** tied
  to the derivation. At `k_base = 0.01`: 21.5 components, `largest_frac` 0.98.
- `birth_window = 0.1` s — grounded only by an *upper* bound (Fisher ~1 s), flagged 2026-05-29 as
  *"tunable, candidate for sweep"*, never swept.

**Explicit caution being respected:** the same Werner skill says *"Do NOT apply the 0.5 bound to
intra bonds — would wrongly cut working intra edges at ~7 nm."* Measured: at 7 nm, `F ≈ 0.36`. The
caution is quantitatively correct. **No threshold is nominated in §2.2 for this reason.**

---

## 4. THE FOUR QUESTIONS, WITH THE DATA ATTACHED

**Q-A. What fidelity should a bond need to count?** The F distribution is bimodal with the split at
the `g`-saturation radius. Is the sub-5 nm core the real entanglement and the long tail an artefact
of a rate-shaped-but-not-fidelity-shaped construction? The threshold *is* the physics claim.

**Q-B. Does a geometric cut satisfy §8, or relocate the failure?** The fidelity cut fragments by
removing **long-range** bonds — a geometric criterion. §8 says geometry is not enough (`g` is
geometry, not input). **Risk: it produces beautifully structured, input-blind topology — the same
trap in a new guise.** Untested: every Unit 12 run is a single drive condition.

**Q-C. The 10¹⁵ Hz / Hz coupling gap (§2.3).** If the EM pathway has no mechanism, what is the
status of a result that depends on it spanning the graph?

**Q-D. Your latent-dimension argument, given the null.** Routing ℝ⁶ via a min-|ΔJ| ≤ 0.15 Hz
tolerance changed 0.8% of edges — with 6 couplings from one distribution, some pair nearly always
matches. Is there a *right* way to make bonding depend on ℝ⁶ compatibility that I implemented too
permissively, or does the null indicate the data cannot carry it?

---

## 5. WHAT WOULD FALSIFY MY CURRENT READING

- **If the fidelity cut's structure is input-blind** — the decisive untested experiment: run the cut
  under two drive conditions with a density covariate. If the partition doesn't move with input, the
  lever is cosmetic. *I consider this the single most likely way I am wrong.*
- **If the F bimodality is an artefact of the 5 nm plateau** rather than physics — the split sits
  exactly where `g` saturates, which is suspicious. A different `coupling_length` would relocate it.
- **If `dim H⁰` rising 3→15 is counting free coordinates rather than structure** — it is scored on
  engaged coordinates only, but the measure deserves independent scrutiny.

## 6. STANDING RESULTS THAT CONSTRAIN ANY PROPOSAL

Dissolution is inert (`k≈1e-4/s`; graph is write-once) · no transient window (field saturated from
first sample) · both mechanisms percolate independently · P0's partition is a deterministic function
of birth times · the keystone is **not supported**, at power that could not detect an effect below
~18–20 components.
