# PO-9 Advisor Packet R2 — readout-time input-selectivity (Unit B), reframed on your review

**2026-07-20 · supersedes `PO9_ADVISOR_PACKET_2026-07-20.md` · for external review**

Your four points are all accepted; R2 rebuilds the claim around them. Summary of what changed:
- The λ "sweep" is retired as malformed (point 1). λ is not a number to pick — the question is the
  **binary** delocalized-vs-Anderson-localized, and the honest coupling form is **flat-then-cutoff**,
  not exp(−d/λ). Settled by a localization estimate, not a sweep.
- The 2×2 is demoted to **ONE measurement + THREE analytic predictions confirmed** (point 2).
- The clock is reconciled exactly (point 3): death at the Werner-floor crossing, delay ≈57 s,
  T_eff≈158 s (not 216), origin at write-end.
- The graded-overlap experiment (point 4) is **running**; its pre-registered verdict is in §5.

---

## 1. The durable result (holds independent of everything below)

**PO-8's blocker was a modeling artifact, and removing it is the real advance.** The correlated-domain
graph "died at ~25 s" because gap dissolution carried a bulk template ×33 catalyst during silence,
where formation is gated off — a one-sided loss with nothing to balance it. In the confined nanodomain
the dissolution products are conserved locally (they don't diffuse to a bulk sink), and our own A3
probe had already measured the confined lifetime at ~200 s. Dropping the term: dimers 19% lost by
120 s (was 76%). **This is what makes a readout at a behavioural delay measurable at all** — before it,
the object was erased, not answered. `L·PO9-1`, committed, N=12.

We also decoupled λ into a metabolic aggregation length (λ_met, diffusive) and a fidelity/coherence
length (λ_F) — they were one constant doing two jobs. That decoupling is what surfaced the question in
your point 1, which we now think is the crux.

---

## 2. The one measured cell (point 2: honestly, this is the finding)

Geometry: two clusters of 4 synapses, 15 µm apart, that ignite independently (verified: driving one
leaves the other dark). Input = timing: SYNC (both clusters co-active) vs STAGGER (never co-active),
density matched. Scored quantity `cross_w` = A–B block weight of the synapse correlation matrix
(Σ p_e over cross-cluster bridges above the Werner floor) = do the clusters share a correlated domain.

**Measured (N=8 free draws, readout delay 20 s):**

| condition | cross_w > 0 | cross_w (mean, range) | status |
|---|---|---|---|
| **SYNC, delocalized (λ_F=214)** | **8 / 8 draws** | 848, [338, 1336] | **MEASURED** |
| STAGGER, delocalized | 0 / 7 | 0 | analytic (never co-exist ⇒ no bond can form) |
| SYNC, localized (λ_F=5) | 0 / 8 | 0 | analytic (bridge below floor: F=0.04) |
| STAGGER, localized | 0 / 6 | 0 | analytic (both fail) |

The three null cells are **hard structural zeros across 21 draws** — not small noisy values. So the
honest statement is: **one measurement** — *under synchronous drive with a delocalized mode, two
15-µm-separated clusters bind into one correlated domain (8/8 draws), and it persists to readout* —
**plus three analytic predictions confirmed at the data level.** We are not selling a 4-cell finding;
you correctly flagged that as the PO-5 mis-registration shape (a criterion satisfiable without
answering the question).

**Persistence + clock (point 3, reconciled exactly).** cross_w holds flat then collapses when the
Werner floor is crossed:

| delay (s) | 0 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|
| cross_w | 1180 | 1167 | 1152 | 1122 | 1103 | 311 |
| mean P_S | 0.945 | 0.901 | 0.860 | 0.822 | 0.787 | 0.726 |
| F_cross = P_S²·0.932 | 0.833 | 0.756 | 0.689 | 0.629 | 0.578 | **0.491** |

Binding survives exactly while F_cross > ½ and crashes when it dips below (delay 40→60). Fitting the
decay gives **T_eff ≈ 158 s, not T_singlet = 216 s** — `analytical_gap` applies `spread_factor`
(j-coupling disorder) and `template_factor` to T_base, which is the faster decay you suspected. The
origin is **write-end** (P_S = 0.943 at delay 0, already decayed during the write). Solving F_cross=½
with the fitted T_eff gives death at **delay ≈ 57 s** — your ~95 s used 216 s and a stimulus-onset
clock; with the true T_eff and origin it lands at 57 s, matching the measured 40→60 s crash.

---

## 3. The math — reframed as flat-then-cutoff / binary (point 1)

A cross-synapse bond is a Werner state, F = P_S^a·P_S^b·W. Werner floor: counts iff F > ½. p=(4F−1)/3.

**We agree the coupling form W = exp(−d/λ) is wrong for this channel.** It is lossy-propagation form —
correct for metabolic aggregation along the dendrite (where λ_met = 5 µm came from and is defensible),
but the cross-synapse channel is **mediation by a delocalized collective mode**, not propagation. Both
dimers couple to the same coherent object, so inside the mediator's coherence domain the coupling is
order-unity and roughly flat in separation; outside it, they couple to uncorrelated domains and the
correlation dies. **The honest form is flat-then-cutoff, and the question is binary:**

- **Delocalized (L_coh ≈ vτ ≈ 214 µm):** at d = 15 µm we are deep inside ⇒ W ≈ O(1). Our λ_F=214 cell
  (W = 0.93) sits here — numerically the delocalized case, now for a defensible reason rather than
  "the larger of two numbers."
- **Anderson-localized (~1 µm, disorder-limited):** the mode is not a network object at all; it cannot
  coordinate synapses 15 µm apart. The cross-synapse channel **does not exist** — not "is weak." Our
  λ_F=5 cell is this limit; its cross_w=0 is analytic, not a datum.

λ_F=5 as a *number* is neither regime — an interpolation with no physical interpretation, which is why
the sweep was unsatisfying. **What settles it is a calculation, not a sweep:** an Anderson-localization
estimate for the acoustic Fröhlich mode — disorder strength against bandwidth, giving a localization
length to compare with L_coh — which returns delocalized-or-not.

**Where we need you:** the two inputs to that estimate are contested and are yours to pin — the mode
**bandwidth** and the effective **disorder strength** at body temperature. Our external pass found
optical superradiance localizes ~1 µm at physiological disorder (~20× critical), and Reimers-style
arguments that coherent Fröhlich condensation is "extremely fragile" — both point localized. The
feasibility calc (L_coh = vτ, no falloff within the coherence domain) points delocalized. We did **not**
assert 214; we refuse to pick the value that makes the ceiling vanish. The estimate is the tiebreaker.

---

## 4. What the one measurement does and does not establish

- **Does:** with a delocalized mode, the readout partition carries a cross-synapse coincidence — "these
  populations fired together" (bind) vs "apart" (stay separate) — and it survives to the Werner-floor
  crossing at ~57 s, a behavioural delay.
- **Does NOT:** establish that biology has a delocalized mode. The whole result is conditional on the
  point-1 binary landing "delocalized." That is the load-bearing open question.
- **A vs B:** this is the (A) reading — cross_w is a common-cause correlation magnitude, no
  non-classicality witness. The quantum content is the monogamy constraint (≤4 bonds from the four ³¹P
  spins), not the readout.

---

## 5. The graded-overlap experiment (point 4) — RUNNING; verdict pre-registered

You were right that this is not optional. STAGGER's zero comes from material absence (clusters never
co-exist ⇒ no bond CAN form), which is the density confound in a temporal costume — a presence
detector, not a computation. The graded version sweeps the temporal offset continuously and measures
cross_w vs **co-ignition duration** (a subtlety we hit and fixed: ignition lags drive by ~13–15 s, so
the physically meaningful x-axis is co-*ignition* time, not co-*drive* overlap; windows raised to 40 s
so partial overlaps produce resolvable co-ignition).

**Pre-registered verdict (Amendment 2, before scoring):**
- **STEP** (cross_w ≈ saturated for any co-ignition > ~0, then cliffs): presence detector. The null.
- **GRADED** (cross_w rises smoothly with co-ignition duration, width ~ dimer/bond lifetime): the
  partition encodes a continuous property of the input — the first result in the sub-programme that
  would clear the §8 bar (computation, not detector).

One early draw (offset=10 → 11 s co-ignition → cross_w ≈ 1050 ≈ full) smells STEP/fast-saturating, but
the low-co-ignition end is not yet sampled and we will not conclude before it lands. Result to follow
in R3.

---

## 6. The three questions we would most value your judgement on

1. **The localization estimate (point 1).** Is delocalized-vs-localized the right binary, and what are
   the defensible mode bandwidth and disorder strength for the acoustic Fröhlich mode at 310 K? This
   replaces the sweep and decides whether the cross-synapse channel exists at all.
2. **Is co-activity too easy an input contrast even in the graded version?** If cross_w saturates fast
   (STEP), is there a subtler input manipulation that could show a graded computation, or is a
   presence detector the honest ceiling of this mechanism?
3. **If it is a presence detector,** is that still a meaningful primitive (coincidence-gated
   correlated update to a synapse set), or does "computation" require the graded encoding — i.e. where
   is the §8 bar exactly?
