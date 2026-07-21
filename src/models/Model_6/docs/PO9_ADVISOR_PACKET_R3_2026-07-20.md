# PO-9/10 Advisor Packet R3 — the localization caveat is discharged; the keystone is unconditional

**2026-07-20 · supersedes `PO9_ADVISOR_PACKET_R2_2026-07-20.md` · closes the loop on your review**

R2 left one load-bearing question open (your point 1): the whole graded-computation result was
**conditional on the mediating mode being delocalized over 15 µm.** That question is now settled — by the
Anderson-localization estimate you asked for (not a sweep), plus your own mechanistic confirmation of the
mode identity. **Verdict: delocalized. The keystone is no longer conditional.** Full arithmetic, tags, and
sources: `PO10_LOCALIZATION_ESTIMATE_2026-07-20.md`.

What changed since R2:
- The localization estimate is **done** and returns **delocalized** — on two independent grounds (disorder
  *and* structural continuity), §3 below.
- You **confirmed the mode identity mechanistically** (the 8 MHz band-bottom mode is the mediator *by
  construction* of Fröhlich condensation), which removes the last assumption R2 flagged. §3.3.
- The graded-overlap experiment (R2 §5) is **complete** — GRADED, step null rejected. Unchanged; restated
  in §5 for a self-contained packet.
- Net: the readout partition performs a **graded computation over temporal input overlap**, over a
  cross-synapse channel that **exists**. Not a precondition — a computation.

---

## 1. The durable result (unchanged from R2, brief)

PO-8's "~25 s death" was a modeling artifact — a bulk template ×33 catalyst applied to gap dissolution
during silence, where formation is gated off (a one-sided loss). Removed: dimers 19% lost by 120 s (was
76%). The substrate now survives to a behavioural readout delay. `L·PO9-1`, N=12, committed. We also
decoupled λ into a metabolic aggregation length (λ_met = 5 µm) and a fidelity/coherence length (λ_F) —
which is what surfaced your point 1.

## 2. The one measured cell (unchanged from R2, brief)

Two clusters of 4 synapses, 15 µm apart, igniting independently (driving one leaves the other dark). Input =
timing. `cross_w` = A–B block weight of the synapse correlation matrix. **SYNC + delocalized (λ_F=214):
8/8 draws bind, cross_w 848 [338,1336].** The other three cells are hard structural zeros across 21 draws
(analytic, not measured). Honest framing (your point 2): **one measurement + three analytic predictions.**
Clock reconciled (your point 3): death at the Werner-floor crossing, T_eff ≈ 158 s, delay ≈ 57 s.

---

## 3. Point 1, SETTLED — the mode is delocalized over 15 µm

R2 promised a calculation, not a sweep, and asked you to pin the mode bandwidth and 310 K disorder. The
calculation is done (`PO10_...`). The headline: **those disorder inputs turned out not to be load-bearing** —
the mode is delocalized by orders of margin regardless — and the real constraint is structural, which also
clears.

### 3.1 Disorder does not localize it
The mediating mode is at **f₀ = 8 MHz** (our `omega_0`; the mode whose Q=10 gives L_coh = v·τ ≈ 214 µm).
Its wavelength is **λ = v/f₀ = 134 µm** — already ~9× the 15 µm gap; even its reduced wavelength
1/k = 21.4 µm exceeds 15 µm (Ioffe–Regel: a wave can't localize below ~its own wavelength). Quantitatively,
with the 1D weak-scattering law ξ ∝ (v/ω)²/(a·σ²) and the ω⁻² acoustic scaling (Ishii 1973; Monthus–Garel
PRB 81:224208), the localization length at 8 MHz is **hundreds of metres to hundreds of km** for realistic
GTP/GDP disorder — to localize at 15 µm you'd need σ ≈ 175 (17,500%), impossible. **Your R2-era worry that
"optical superradiance localizes at ~1 µm" does NOT port:** that is a *high-frequency excitonic* mode; ours
is 3–7 decades lower in frequency, and ξ ∝ ω⁻² makes it 10⁵–10⁷× longer. The scoping doc's "2% disorder
threshold" was evaluated at a band-top mode; at our band-bottom operating frequency the true margin is far
larger.

### 3.2 The real constraint is structural continuity — and it also clears
A lattice **terminus** is not weak disorder — it's a near-total reflector, and no ω⁻² scaling rescues a bare
phonon from a wall. Individual dendritic MTs are only a few µm, so a single-MT picture would fail at 15 µm.
**But the mediator is the shared condensate BUS on the backbone *bundle*, not one MT.** In a staggered bundle
of N = 5–15 MTs, a single MT's end removes only ~1/N ≈ 10–20% of the local cross-sectional stiffness (the
other N−1 carry through) — a *dip*, not a wall. That reduces to weak disorder (σ ~ 1/N), which the same
calculation puts at ξ ~ metres ≫ 15 µm. A true wall needs the *whole bundle* to terminate — a branch point —
and continuous unbranched dendritic segments run tens of µm (> 15 µm). **So the bus spans 15 µm; the reach is
set by backbone segment length (tens of µm), not by individual MT length or by disorder.**

**Testable prediction (yours, sharpened):** cross-cluster binding requires both clusters on the *same
unbranched backbone segment* — robust to individual MT turnover, but failing across a branch point. A
spine-placement/branch-topology claim, more informative than a decay constant.

### 3.3 Mode identity — confirmed (your argument, recorded)
You closed the one assumption we couldn't close ourselves: the 8 MHz band-bottom mode is the mediator **by
construction**, because Fröhlich condensation funnels quanta to the lowest mode — "the condensing mode" and
"the lowest mode" are the same statement. A GHz mediator is structurally impossible for a condensate (and
would independently collapse L_coh). Thermal occupation says the same (n̄ ~124× higher at 8 MHz than 1 GHz at
310 K). We've adopted this as the primary argument; our L_coh self-consistency check is kept as a second line.

---

## 4. What the result now establishes (caveat discharged)

- **Does:** with the (now-established) delocalized bus, the readout partition carries — and *grades* — a
  cross-synapse coincidence: it encodes **how long two synapse populations were simultaneously active**, and
  it survives to the Werner-floor crossing (~57 s), a behavioural delay.
- **No longer conditional:** R2's load-bearing "if delocalized" is discharged. What remains genuinely open is
  the standing program-level bet on **Q ≈ 10** (the AMRIS-class coherence-time measurement) — but that is the
  pre-existing backbone premise, not something this result introduces.
- **A vs B:** still the (A) reading — cross_w is a common-cause correlation magnitude; the quantum content is
  the monogamy constraint (≤ 4 bonds / four ³¹P spins), not the readout. No non-classicality is claimed.

---

## 5. The graded-overlap result (complete) — GRADED, step null rejected

Sweep the temporal offset continuously; score cross_w vs **co-ignition duration** (ignition lags drive
~13–15 s, so the physical x-axis is co-*ignition* time; windows raised to 40 s).

| co-ignition (s) | 0.0 | 0.4 | 2.8 | 6.5 | 10.0 | 25.0 |
|---|---|---|---|---|---|---|
| cross_w (mean±sd) | 0 | 26±16 | 209±55 | 600±87 | 926±23 | 848±316 |

Spearman ρ = **+0.936** (40 draws); step-rejection Welch **t = 24.25** (2.8 s co-ignition → 209 ≪ saturated
926). cross_w is a smooth monotone function of co-ignition duration, width ~5–7 s (bond kinetics). **The
partition encodes a CONTINUOUS property of the input — a computation, not a presence detector.** This clears
the §8 bar, now unconditionally (§3).

---

## 6. Where we're headed, and where we'd value your judgement

1. **Unit C — what can this engine actually compute?** With inputs (spatiotemporal cluster activation) and a
   readout (the graded correlation-partition / agreement pattern) both characterized, the open question is the
   **expressivity of the input→partition map.** We've launched a focused research pass to pick the right
   theoretical lens (reservoir/fading-memory kernels vs. similarity/kernel computation vs. correlation
   clustering vs. associative memory), each with concrete simulator-runnable benchmark tasks — foregrounding
   (a) temporal credit assignment and (b) whether the entanglement structure buys anything over a classical
   correlation reservoir. Your steer on the right framing would be valuable before we commit to benchmarks.
2. **Scoping note (for Sarah):** the model's `N_backbone` default encodes a ~10 µm backbone segment — below
   the 15 µm experiment geometry. Not a physics limit (sweepable), but it should be set ≥ 15 µm before a
   "15 µm reach" is quoted externally.
