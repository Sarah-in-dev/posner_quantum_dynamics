# Coherence-ordered fragmentation of the entanglement partition (T1′)

*Results write-up. Model 6, cross-synapse entanglement topology. 2026-07-17.*
*Provenance: `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries T1'-1…T1'-5, dt-1, ERR-1.*

---

## 1. Summary

The cross-synapse entanglement partition is governed by a coherence-dependent distance
cutoff that follows algebraically from the Werner separability bound, with no fitted
parameters. As coherence decays this cutoff contracts, and the prediction is that the
partition must therefore fragment **in order of synapse spacing — most-separated pair
first**. We tested the ordering on a four-rung ladder across four independent stochastic
realizations. All four produced the pre-registered order exactly (p ≈ 3.0 × 10⁻⁶ against a
uniform-permutation null).

The result is discriminating: a scalar eligibility trace — the standard model of a decaying
synaptic memory in both neuroscience and machine learning — decays uniformly and carries no
spatial information, and therefore cannot produce a spacing-ordered cascade under any
parameterisation. The result is a statement about the **model**, not about biology; see
§7.

---

## 2. The prediction, derived

Cross-synapse bond fidelity in the model is a Werner fidelity over the two endpoints'
coherences and their spatial coupling:

```
F = P_S_i · P_S_j · w ,      w = exp(−d / λ) ,     λ = 5 µm
```

where `P_S ∈ [0.25, 1]` is a dimer's singlet probability (its coherence) and `d` is the
inter-synapse distance. A Werner state is entangled iff its fidelity exceeds ½ (Werner
1989), so an edge is counted iff `F > 0.5`. Substituting and solving for `d`:

```
P_product · exp(−d/λ) > 0.5
exp(−d/λ)  >  0.5 / P_product
−d/λ       >  ln(0.5 / P_product)
 d         <  λ · ln(P_product / 0.5)  ≡  d*
```

**`d*` is a coherence-set interaction radius.** Two synapses are entangled if and only if
they are closer than `d*`, and `d*` is set entirely by the coherence of the dimers involved.

Two consequences follow immediately, and neither is a tunable quantity:

1. **A hard coherence floor.** Since `w ≤ 1`, we have `F ≤ P_S²`, so `F > 0.5` requires
   **`P_S > 1/√2 ≈ 0.7071`**. Below this coherence no two synapses can be entangled *at any
   separation whatsoever*. The separability bound is simultaneously a coherence threshold.
2. **A contracting radius.** `d*` is monotonically increasing in `P_product`. As coherence
   decays, `d*` shrinks, and it crosses each pair's separation in descending order of
   separation. **The partition must fragment far-pairs-first.**

Inverting for the coherence at which a given gap `g` loses its edge (taking both endpoints
at equal coherence, `P_product = P_S²`):

```
P_crit(g) = sqrt( 0.5 · exp(g/λ) )
```

Nothing here is fitted. The `0.5` is the Werner separability bound; `λ = 5 µm` is the
model's pre-existing coupling length; the functional forms are the model's existing bond
fidelity. The prediction is a two-line algebraic consequence of a 1989 theorem.

---

## 3. Prior confirmation: the static half, and a free retrodiction

Before testing the dynamic ordering, the static form of the rule was confirmed: measured
`P_S ≈ 0.998` gives `d* = 3.45 µm`, and an eight-synapse ladder with mixed gaps
(2.0/2.5/4.5/3.0/2.0/4.5/2.8 µm) was pre-registered against it. The probe called **7 of 7
gaps correctly**, reproduced the exact predicted edge list, and returned
`betti0_cross = 3`, `component_sizes = [3,3,2]`, `betti1_cross = 0`.

The same `d*` **retrodicts an earlier, independent validation with no free parameters.** In
the previously-run chain-versus-ring geometry at 2.5 µm spacing: nearest-neighbour 2.5 µm
< 3.45 bonds; the ring's next chord (4.33 µm) and the opposite/next-nearest vertex (5.0 µm)
do not. That yields exactly the observed 5 path edges for the chain and 6 hexagon edges for
the ring — a result obtained before the distance rule was derived, and explained by it
afterwards without adjustment.

---

## 4. Method

**Rig and geometry.** Eight synapses in 1D. Four "live" gaps carry edges; each is isolated
from its neighbours by a 4.5 µm **dark control**, for which `P_crit = 1.1090 > 1` — an edge
there is impossible at any coherence, since `P_S ≤ 1` by construction. Because connectivity
in 1D is decided entirely by consecutive gaps, each live pair forms its own component, and
each break is independent and unambiguous (betti0: 4 → 5 → 6 → 7 → 8).

| gap (µm) | P_crit | role |
|---|---|---|
| 3.35 | 0.9885 | live — predicted to break **first** |
| 2.90 | 0.9450 | live |
| 2.45 | 0.9034 | live |
| 2.00 | 0.8637 | live — predicted to break **last** |
| 4.50 (×3) | 1.1090 | dark control — cannot bond at any coherence |

**`d*(0)` was measured, not assumed.** In the rig at t = 0.08 s (n ≈ 2200 dimers): `P_S`
median 0.9987 (min 0.9922, max 1.0), giving `d*` median **3.4522 µm** (min 3.3870, max
3.4657). The *distribution* matters: all four live gaps sit below the population **minimum**
`d*`, so every live edge forms on the whole population at t = 0 rather than on a fortunate
tail.

**The governing statistic is the tail, not the median.** A synapse-pair edge survives while
*any* bonded dimer pair clears `F > 0.5`. The relevant radius is therefore
`d*_eff = λ·ln(max_pair(P_S²)/0.5)` — an extreme-value statistic — and it decays roughly
3.6× more slowly than the median-derived radius. This is not a detail: a median-based
reading predicts breaks that do not occur, and directly caused three successive incorrect
break-time derivations in earlier work. A control run confirmed the effect: at t = 7.5 s the
median radius had fallen below all four live gaps while all four edges remained intact.

**Geometry was chosen for statistical power, and the power was measured.** Each synapse's
tail is set by its own longest-lived dimer, whose effective lifetime `T_eff` is fixed at
creation; the resulting between-synapse scatter in `d*_eff` is ≈ 0.29 µm. Rung spacings
finer than that scatter are decided by chance. Measured order-recovery across 10 seeds:

| rung spacing | live gaps (µm) | order recovered |
|---|---|---|
| 0.10 µm | 3.35 / 3.25 / 3.15 / 3.05 | 6 / 10 |
| 0.25 µm | 3.35 / 3.10 / 2.85 / 2.60 | 5 / 10 |
| **0.45 µm** | **3.35 / 2.90 / 2.45 / 2.00** | **10 / 10** |

The widest ladder was adopted. Note the direction: wider rungs push the cascade *later* and
cost more compute, but are the only configuration in which the ordering is resolvable above
the intrinsic scatter. This also establishes that the ordering is **not** an algebraic
identity — it fails 40% of the time when the spacing is too fine to resolve (see §7).

**Pre-registration.** Only the **order** was pre-registered: 3.35 → 2.90 → 2.45 → 2.00.
Break *times* were explicitly not predicted and are not scored. They are an extreme-value
statistic over a multiplicative random walk (`P_excess` carries ~±5.8% noise accumulated
over ~10⁵ steps) across hundreds of pairs, and three prior attempts to predict them
analytically (9.5 s → ~13 s → 19.3 s against an observed 34.0 s) failed. The order, by
contrast, is invariant to which quantile governs.

**Verdict guards.** An edge must be absent for 3 consecutive samples to count as broken
(transient absences are logged as flickers and excluded); the verdict returns
**INCONCLUSIVE** below 3 clean breaks. Both guards exist because an earlier version of this
experiment reported a false confirmation from a single transient absence scored by a
monotonicity test that was vacuously true over one point. The verdict function used here can
return INCONCLUSIVE, CONFIRMED, or FALSIFIED, and returned INCONCLUSIVE on control runs.

**Integration step.** dt = 10⁻³ s. `P_S` and the Werner edge set are dt-converged
(`d*_med = 3.45`, edges = 5, unchanged across dt from 10⁻⁴ to 5×10⁻³), so the scored
quantities are step-size independent. dt = 10⁻² was rejected (dimer count overflows). The
drive-transient dimer count is not fully converged at 10⁻³ (~+38% versus 10⁻⁴), but the
scored quantity — the edge set — is.

**Drive protocol.** Theta-burst drive for 0.08 s to form the partition, then silence for up
to 90 s. `η` (the condensation/pump variable) is clamped at 0.26 throughout, including during
silence. This is a control, not a convenience: it holds the pump fixed so that coherence is
the only variable moving. Bonds continue to *form* under clamped η, but formation cannot
rescue an edge, since the Werner cut is applied at read time against current `P_S`.

---

## 5. Results

Four independent seeds. All four broke in the exact pre-registered order.

| seed | gap 3.35 | gap 2.90 | gap 2.45 | gap 2.00 | order |
|---|---|---|---|---|---|
| 0 | 14.5 s | 32.5 s | 61.5 s | 78.0 s | ✓ |
| 1 | 14.0 s | 37.0 s | 55.0 s | 82.5 s | ✓ |
| 2 | 14.0 s | 42.0 s | 54.5 s | 71.0 s | ✓ |
| 3 | 11.0 s | 32.5 s | 64.5 s | 82.0 s | ✓ |

All three dark controls remained unbonded in every run, as required.

**Statistical assessment.** Under the null hypothesis that the fragmentation order carries
no spatial information, the observed order is a uniformly random permutation of four rungs:
P(correct) = 1/4! = 1/24 per seed. Four independent seeds all correct gives

```
p = (1/24)⁴ = 1/331 776 ≈ 3.0 × 10⁻⁶
```

**Order is invariant; times are not.** The 2.90 µm gap broke at 32.5, 37.0, 42.0 and 32.5 s
across the four seeds — a spread of nearly 30% — while the ordering never varied. This is
the direct empirical vindication of scoring the order and refusing to score the times.

---

## 6. Confounds and controls

**Dimer population collapse (the principal confound).** The dimer population falls from
≈ 2200 to under 100 over the course of a run, because dissolution is coherence-protected and
therefore accelerates as coherence decays. The last two breaks in each seed occur in this
depleted regime, so their break *times* are confounded: an edge may be lost because its
synapses ran out of dimers rather than because `d*` contracted past the gap.

**Why this does not compromise the ordering.** Dissolution is spatially uniform — it lowers
every pair's effective radius equally, independent of separation. Its tendency is therefore
to make edges fail at *similar* times, not in separation order. It cannot generate a
consistent spacing-ordered cascade, and certainly not the same one across four independent
stochastic realizations. Replication is what converts this from an argument into a control:
uniform attrition has no mechanism by which to reproduce a specific ordering repeatably,
whereas the contracting-radius mechanism predicts exactly that ordering every time. The
first two breaks in each seed additionally occur while the population is still healthy
(≈ 1800 and ≈ 1000 dimers in seed 0), and are unconfounded on their own.

**Dark controls.** Three 4.5 µm gaps with `P_crit = 1.1090 > 1` are structurally incapable of
bonding. They remained dark throughout every run, confirming that edges are not being
produced spuriously.

**Flicker rejection.** Transient edge absences occurred (e.g. seed 0 on the 3.35 µm gap at
t = 14.0 s, resolving before the true break at 14.5 s) and were correctly excluded.

**Step-size control.** See §4 — the scored quantities are dt-converged.

---

## 7. Limitations

**This is a result about the model, not about biology.** The simulation has no privileged
access to whether the underlying physics in a living system is quantum. Nothing here
addresses that question, and no experiment currently proposed measures the quantum state in
a living computational system *and* attributes the system's computation to it. The
appropriate framing is that the model is a theory of how the system is expected to operate,
made discriminating so that it predicts differently from its rivals on a measurable
quantity.

**This is not a claim of quantum computation.** The model's readout assigns one shared
stochastic outcome per connected component — a classical common-cause correlation over scalar
magnitudes, with no phase and no non-classicality witness. The present result establishes
that the partition carries *spatial* structure; it says nothing about whether it carries
*quantum* structure. Establishing the latter requires a separate microscopic construction
with phase and an explicit non-classicality test.

**On circularity.** The edge criterion contains distance, so it is reasonable to ask whether
observing distance-ordered fragmentation merely recovers the model's own definition. The
algebra fixes the *rule*; it does not establish that the resulting *ordering survives* the
system's stochasticity — per-dimer frozen lifetimes, multiplicative noise compounding over
~10⁵ steps, extreme-value statistics over hundreds of pairs, and a collapsing population.
That survival is an empirical question, and it demonstrably has a negative answer in part of
the parameter space: at 0.10 µm rung spacing the correct order is recovered only 6 times in
10 (§4). The ordering is a robustness property of the noisy system, not a restatement of the
definition. It is, however, correctly described as a demonstration that the model behaves as
its own physics implies under realistic noise — not as a discovery about nature.

**Scope.** One geometry (1D, eight synapses, four rungs), one drive protocol, four seeds,
and a fixed spatial arrangement. The partition here encodes *distance*, which is anatomy and
is fixed before the run begins; whether the partition also encodes *input* — which synapses
were driven, in what pattern — is a separate and necessary question, and is the designated
next line of work.

---

## 8. Methods appendix — reproduction

All scripts run from `src/models/Model_6/sweep/`:

| script | purpose |
|---|---|
| `coherence_fragmentation_probe.py` | the T1′ experiment (`--seconds 90 --seeds 0 1 2 3`) |
| `coherence_radius_probe.py` | the static half (7/7 gap confirmation) |
| `measure_dstar0.py` | measures `d*(0)` and the `P_S` distribution |
| `dstar_eff_replay.py` | tail-statistic replay; window sizing (upper bound on break times) |
| `order_power_probe.py` | order-recovery power across candidate geometries |
| `dt_convergence_drive.py`, `dt_convergence_operating_point.py`, `dt_independence_tail.py` | step-size controls |

Runtime ≈ 2.7 h per seed (single-threaded, ~500 MB); seeds parallelise across cores with no
measurable interference. Runs self-terminate once all live edges are confirmed broken.

**References.** Werner, R. F. (1989), *Phys. Rev. A* **40**, 4277 — separability bound,
entangled iff F > ½. Agarwal et al. (2023), arXiv:2210.14812 — dimer coherence lifetimes.
