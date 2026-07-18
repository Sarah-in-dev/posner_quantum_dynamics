# Coherence-ordered fragmentation of the entanglement partition (T1′)

*Results write-up. Model 6, cross-synapse entanglement topology. 2026-07-17.*
*Provenance: `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries T1'-1…T1'-6, dt-1, ERR-1.*
*Revised 2026-07-18 following adversarial review: §6 rewritten on a measured basis (T1′-6),
§4 power figure revised 10/10 → 37/40, §5 conditioning and §6 dark-control scope stated.*

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
| **0.45 µm** | **3.35 / 2.90 / 2.45 / 2.00** | **37 / 40 (92%)** |

The widest ladder was adopted. Note the direction: wider rungs push the cascade *later* and
cost more compute, but are the only configuration in which the ordering is resolvable above
the intrinsic scatter. This also establishes that the ordering is **not** an algebraic
identity — it fails 40% of the time when the spacing is too fine to resolve (see §7).

The adopted geometry's power figure was **revised downward** on re-measurement (research log
T1′-6): 37 of 40 independent noise draws, not the 10/10 originally recorded. The two figures
are different estimators of the same quantity — the original used unguarded first-crossing
detection, whereas 37/40 applies the **guarded** break criterion the experiment itself uses
(`CONSECUTIVE_ABSENT = 3`, below), which is stricter and converts marginal orderings into
violations. The guarded figure is the operative one. Order recovery is therefore high but
**not deterministic**: one of the four seed populations recovers the order in only 7 of 10
draws, its last two rungs breaking within ≈ 1.5 s of each other. This does not bear on the
significance in §5, which is computed against a permutation null that instrument power does
not move; it bears on how strongly the geometry choice can be justified.

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

The null is unaffected by the choice of geometry, but note that this `p` is conditional on a
ladder selected for order-recovery power (§4). The four scored seeds are fresh, so this is
not data reuse; and at the revised 92% per-seed power, observing 4/4 has probability
0.92⁴ ≈ 0.72 — an unremarkable outcome for a working instrument, not a fortunate one.

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

**The objection this raises, stated at full strength.** Because edge survival is governed by
`max_pair(P_S²)` — an *extreme-value* statistic (§4) — population loss is not obviously
order-neutral. The maximum of few draws is typically smaller than the maximum of many, and
bonded pairs scale as ~N². If losing dimers contracts `d*_eff` on its own, then by §2 it
would cross the gaps widest-first *as well*, producing the same signature by a different
mechanism — and reproducibly, so replication across seeds would not distinguish the two.
An earlier version of this section dismissed this by appeal to dissolution's *spatial
uniformity*. That was the wrong property, and the objection deserves a measurement.

**Why this does not compromise the ordering — measured, not argued.** The run was replayed
with the two channels separated (research log T1′-6; `sweep/population_channel_arms.py`,
traces under `results/T1prime6_arms/`). Four arms, pre-registered before execution together
with the reading of every outcome, including the outcomes that would have refuted this
section:

| arm | coherence | population | result, all four seeds |
|---|---|---|---|
| A | decays | decays | reproduces the cascade |
| B | decays | **held fixed** | **bit-identical to A** (`max|A−B| = 0.000e+00`) |
| C | **frozen** | decays | **zero breaks** |
| D | frozen | held fixed | zero breaks (null) |

Holding coherence fixed while the population collapses 2223 → ≈ 50 produces **no edge breaks
at all**, and retains **100.0000%** of `max_pair(P_S²)`. Enabling attrition alongside
coherence decay changes nothing: arms A and B agree to machine zero across every sample of
every pair's `d*_eff`. The population channel is not small here — it is *inert*.

The reason is structural rather than statistical, and it is visible in the removal rule.
Dimers are deleted **lowest-coherence-first** (`dimer_particles.py:230-241`, sorting on a
strictly increasing map of `P_S`), re-sorted every step. Attrition is therefore
**rank-selective**: the population maximum is the *last* element removed, so the statistic
that governs edge survival is precisely the one attrition cannot reach. The extreme-value
argument above assumes random removal, which this model does not perform.

That assumption was tested on its own terms rather than dismissed. Forcing **uniform random**
removal — the rule under which population loss genuinely would erode an extreme-value
statistic — still yields **zero breaks**, with `Δd*_eff = −0.002 µm` against the 1.35 µm span
the cascade must traverse. The reason is measured: `P_S(0)` is packed against its ceiling
(median 0.9987, max 1.0000), leaving no headroom below the maximum, so the max over ≈ 50
survivors is indistinguishable from the max over ≈ 2200. The objection fails twice over —
once on the model's actual removal rule, and once on the shape of the coherence distribution.

**Where population loss does bite, reported for completeness.** Under random removal
*combined with* coherence decay, the cascade still orders correctly in every seed but breaks
systematically earlier: once `P_S` spreads out under decay, the tail is carried by a few
long-lived dimers that random deletion can destroy. A model that removed dimers at random
would thus carry a live population channel affecting break *times* — though not their order.
This model does not remove randomly, and the claim here is scoped to this model.

**Dark controls.** Three 4.5 µm gaps with `P_crit = 1.1090 > 1` are structurally incapable of
bonding, and remained dark throughout every run. This is an **implementation** check — it
confirms that no edge is fabricated where the algebra forbids one — and not a physics
control: since `P_crit > 1` cannot be met at any coherence, their darkness is a foregone
conclusion rather than an observation about the mechanism.

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

**On circularity.** The right description of this result is that the model **behaves as its
own physics implies under realistic noise** — not that anything was discovered about nature.
The edge criterion contains distance, so distance-ordered fragmentation partly recovers the
model's own definition, and that framing should govern how the result is read. What the
algebra fixes is the *rule*; what it does not fix is whether the resulting *ordering
survives* the system's stochasticity — per-dimer frozen lifetimes, multiplicative noise
compounding over ~10⁵ steps, extreme-value statistics over hundreds of pairs, and a
collapsing population. That survival is an empirical question with a genuinely contingent
answer: the order fails to be recovered in part of the parameter space (6/10 at 0.10 µm rung
spacing), and even at the adopted spacing it is recovered 92% of the time rather than
always (§4). Both failures are, admittedly, what the algebra itself predicts once the
between-synapse scatter in `d*_eff` (≈ 0.29 µm) exceeds the rung spacing — so this
demonstrates robustness of the model's behaviour under noise, and should not be oversold as
an independent discovery.

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
| `order_power_probe.py` | order-recovery power across candidate geometries (unguarded detection; see §4) |
| `population_channel_arms.py` | §6 channel separation — coherence vs population, four arms + two counterfactual |
| `dt_convergence_drive.py`, `dt_convergence_operating_point.py`, `dt_independence_tail.py` | step-size controls |

Runtime ≈ 2.7 h per seed (single-threaded, ~500 MB); seeds parallelise across cores with no
measurable interference. Runs self-terminate once all live edges are confirmed broken.

**Data availability.** The four scored T1′ runs (§5) wrote to session-scoped scratchpad and
those raw logs **are lost**; their break tables survive only as transcription into the
research log, so §5 is not independently re-derivable without a re-run. This was corrected
going forward: from 2026-07-18 all scored traces are persisted to a tracked path
(`src/models/Model_6/results/`), and the §6 channel-separation arms are the first result
whose per-sample traces — `n_dimers`, `max_pair(P_S²)` and per-pair `d*_eff` at 0.5 s
resolution — ship alongside the claim.

**References.** Werner, R. F. (1989), *Phys. Rev. A* **40**, 4277 — separability bound,
entangled iff F > ½. Agarwal et al. (2023), arXiv:2210.14812 — dimer coherence lifetimes.
