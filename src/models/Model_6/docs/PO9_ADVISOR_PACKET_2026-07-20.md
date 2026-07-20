# PO-9 Advisor Packet — the readout-time input-selectivity keystone (Unit B)

**2026-07-20 · Model 6 / entanglement-topology sub-programme · for external review**
**Status: interim (1–3 free draws/cell; full 8/cell ensemble running). The pattern is clean; the magnitudes are provisional.**

---

## 0. One-paragraph result

We asked whether the correlated-domain partition **at readout** (a dopamine event at a behaviourally
realistic delay) carries *which input was presented*. It does — **but only when the entanglement
coherence length λ_F is long enough to bridge two synapses.** With two spatially separated synapse
clusters, cross-cluster binding (the observable that says "the two clusters are one correlated
domain") appears in **exactly one of four conditions**: synchronous co-activation *and* long λ_F.
It is a logical AND of a **co-activity** gate (input) and a **Werner-floor** gate (coherence length).
This ties the whole result to the λ decision that has been in dispute: input-selectivity at the
network scale is real, and it is **λ_F-gated**.

---

## 1. Setup

**Substrate fix that makes this runnable (L·PO9-1).** The gap-dissolution term carried a bulk
template ×33 catalyst during silence (formation gated off ⇒ one-sided loss). Removing it restores the
confined-niche lifetime (dimers 19% lost by 120 s, was 76%), so the substrate survives to readout.
Without this the readout object was erased before it could be scored (PO-8's blocker).

**λ decoupling (L·PO9-1).** One constant `coupling_length_um` was doing two physically distinct jobs:
metabolic-power aggregation (diffusive, gates *which synapses co-ignite*) and the Werner fidelity
weight (the condensate coherence length, sets *how far an entangled bond stays correlated*). Split
into λ_met (=5 µm, kept) and λ_F (swept). This experiment sweeps λ_F.

**Geometry (Amendment 1).** Two clusters of 4 synapses each:
- within-cluster spacing **0.5 µm**; between-cluster gap **15 µm**.
- Reason for two *separated* clusters rather than a linear array: (i) ignition needs a spatial quorum
  — a tight cluster of ~4–6 ignites, 3–4 spread on a line do not; (ii) metabolic aggregation reaches
  ~λ_met=5 µm, so *adjacent* groups sit inside each other's aggregation range and co-ignite
  regardless of drive (branch-global, L·ETA-4). At 15 µm the clusters ignite independently — verified:
  driving cluster A gives per-synapse peak η = [0.19, 0.20, 0.19, 0.18 | 0, 0, 0, 0]; B stays dark.

**Two input conditions, density matched by construction (all synapses driven in both):**
- **SYNC** — both clusters driven simultaneously, t ∈ [0, 20 s].
- **STAGGER** — cluster A driven t ∈ [0, 20 s], cluster B t ∈ [20, 40 s]. Never co-active.
Each synapse receives the same total drive; only the *timing* differs. No seed; free-running draws.
Drive is per-synapse **through `net.step`** (so the backbone η field still updates; per-synapse
`s.step()` would leave η≡0 and nothing ignites).

**Protocol per draw:** WRITE (per-synapse, SYNC or STAGGER) → `analytical_gap(delay)` (the confined
intrinsic aging) → score at delays {0, 10, 20, 30, 40, 60} s.

---

## 2. The math

**Cross-synapse bond fidelity (Werner state):**

    F_ij = P_S^a · P_S^b · W_ij ,      W_ij = exp(−d_ij / λ_F)

P_S ∈ [0.25, 1] is the per-dimer singlet probability (coherence); W_ij the spatial weight.

**Werner separability bound (Werner 1989):** a bond is entangled — counts toward the correlated
partition — iff

    F_ij > 1/2 .

**Correlation coefficient of a Werner bond:** p = (4F − 1)/3   (F=1/2 → p=1/3; F=0.815 → p=0.75).
Correlation multiplies along a path, so connectivity ≠ correlation (L·PO7-5).

**Scored quantity — `cross_w`:** the total A–B block weight of the synapse-level correlation matrix,

    cross_w = Σ_{a∈A, b∈B, F_ab>1/2}  p_ab ,    p_ab = (4 F_ab − 1)/3 ,

i.e. how much correlation crosses between the two clusters. cross_w > 0 ⇔ the clusters share a
correlated domain; cross_w = 0 ⇔ they are two separate domains. (We retired Newman modularity Q_act
as the scored statistic — it is trivially high for two spatial clusters against a grouping equal to
those clusters; the failing-first control confirmed SYNC was *not* null under Q_act. `cross_w` is the
input-clean observable.)

**Why λ_F is the gate.** Take the post-write coherence P_S ≈ 0.9 (so P_S² ≈ 0.81). A *cross-cluster*
bond (d = 15 µm) counts iff

    F_cross = P_S² · exp(−15/λ_F) > 1/2  ⟺  P_S² > 0.5 · exp(15/λ_F).

- **λ_F = 5 µm:** exp(−15/5) = e^−3 = **0.0498**, so F_cross = 0.81·0.0498 = **0.040**. Requires
  P_S² > 10.0 — impossible. **Cross-cluster bonds can never clear the floor.** The two clusters are
  always two domains, for *any* input. Geometry wins; input is invisible.
- **λ_F = 214 µm:** exp(−15/214) = e^−0.0701 = **0.9323**, so F_cross = 0.81·0.9323 = **0.755 > 1/2**.
  Cross-cluster bonds count while P_S² > 0.536, i.e. P_S > 0.732. **The bridge is above the floor.**

(For reference, within-cluster d = 0.5 µm: W = 0.905 at λ_F=5, 0.9977 at λ_F=214 — within-cluster
bonds are strong at both λ_F, which is why within_w is large everywhere and only the *cross* block
carries the input signal.)

**Why co-activity is the second gate.** A cross-cluster bond needs dimers present at *both* ends
*simultaneously* and both ends co-ignited (condensate mediation k_cross ∝ √(η_a η_b) · W · P_S²). In
STAGGER the clusters are never co-active: when A's dimers exist B has none, and by the time B forms A
has aged. So even at λ_F = 214, STAGGER forms no cross bonds. In SYNC they co-exist and co-ignite.

**The conjunction (the selectivity):**

    cross_w > 0   ⟺   (co-active: SYNC)  AND  (bridge above floor: λ_F large).

| | λ_F = 5 µm | λ_F = 214 µm |
|---|---|---|
| **SYNC**    | 0 (floor gate fails) | **> 0** |
| **STAGGER** | 0 (both fail)        | 0 (co-activity gate fails) |

---

## 3. Result (interim: 1–3 draws/cell, delay = 20 s unless noted)

| condition | cross_w (per draw) | within_w (mean) | ignited |
|---|---|---|---|
| **SYNC λ_F=214**   | **337.7, 1083.3**  | 2963 | 2/2 |
| STAGGER λ_F=214    | 0.0                | 1825 | 1/1 |
| SYNC λ_F=5         | 0.0, 0.0, 0.0      | 2214 | 3/3 |
| STAGGER λ_F=5      | 0.0                | 1671 | 1/1 |

The three null cells are **hard structural zeros** (no admissible cross-cluster bond forms at all),
not small noisy values. Only SYNC λ_F=214 is positive. cross_frac = cross_w/(within_w+cross_w) ≈ 0.10
— the clusters are *weakly* bridged (a detectable inter-cluster domain, ~10% of total correlation),
not fused into one blob.

**Persistence across the readout delay (SYNC λ_F=214, one draw):**

| delay (s) | 0 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|
| cross_w | 349 | 345 | 338 | 329 | 321 | 132 |

The inter-cluster binding **survives ~40 s**, then falls — consistent with the closed form: cross_w
drops when P_S decays below 0.732 (F_cross = P_S²·0.932 through the Werner floor). This is the
coherence-limited self-cleaning of L·PO7-5, now observed on the cross-cluster channel.

---

## 4. Interpretation

- **The network computes a cross-synapse coincidence.** The readout distinguishes "these two synapse
  populations fired together" (SYNC → one bridged domain) from "they fired apart" (STAGGER → two
  domains). That distinction is carried in the correlated-domain partition at dopamine time and
  survives a behaviourally realistic delay.
- **It is λ_F-gated.** The capability exists *iff* the coherence length is long enough for the
  inter-synaptic bridge to clear the Werner floor. At the short, disorder-limited λ_F it is absent.
  So the contested parameter (λ_F: ~1 µm localised … ~214 µm ballistic) is not a detail — it is the
  switch that determines whether the substrate can read its input at all.
- **A vs B.** This is the (A) coherence-gated classical correlated-partition reading: cross_w is a
  common-cause correlation magnitude, no non-classicality witness. The quantum content remains the
  monogamy constraint (degree ≤ 4 from the four ³¹P spins), not the readout itself.

---

## 5. What we would value the advisor's judgement on

1. **Is co-activity the right operationalisation of "input"?** STAGGER's cross_w = 0 is almost
   definitional (clusters never co-exist, so no bond *can* form). We read the informative content as
   the *contrast* (the partition distinguishes the two inputs) — but a subtler contrast (partial
   temporal overlap, graded) would test *graded* selectivity rather than a binary. Worth building?
2. **The λ_F value, still unresolved.** This shows the capability is real *conditional on* λ_F being
   long. It does **not** establish that biology operates at long λ_F. The disorder/localisation
   physics (optical superradiance localises ~1 µm; coherent Fröhlich "extremely fragile") argues for
   short; the feasibility calc (L_coh = vτ ≈ 214 µm, no falloff within the coherence domain) argues
   for long. We deliberately did **not** assert 214; we swept it. Which bound does the advisor weight,
   and is decoupling λ_met from λ_F the right resolution?
3. **Normalisation.** cross_w is absolute; the three hard zeros make the contrast unambiguous, but for
   the graded question a normalised cross_frac (reported) may be the better statistic.
4. **Does the coincidence primitive survive P_S²·W < ½ being the *only* gate?** i.e. is there any
   route to cross-synapse correlation that does not pass through the condensate co-ignition, which
   would change the input-selectivity story.

**Registered before scoring:** `PREREG_PO9_UNIT_B_READOUT_KEYSTONE.md` (Amendment 1 records the
metric change and the failing-first result that motivated it). Full 8-draw distributions to follow.
