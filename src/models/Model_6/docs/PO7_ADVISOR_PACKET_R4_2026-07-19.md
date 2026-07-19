> # ⚠ WITHDRAWN — DO NOT SEND · 2026-07-19
>
> **This packet's core framing is wrong and it must not go to the advisor as written.**
>
> It assumes phosphate provenance can act as a CROSS-SYNAPSE channel. It cannot. Per
> `model6-entanglement-partition-werner` §2 the cross-synapse mechanism is
> `k_cross = K_ENTANGLE_EM_BASE(0.5)·sqrt(eta_i·eta_j)·w_spatial·P_product`, with
> `w_spatial = exp(-d/5 µm)` the **condensate coupling length** — cross-synapse entanglement is
> mediated by the Fröhlich backbone, not by a phosphate travelling between synapses. Direct
> entanglement is only ever local.
>
> **Consequently withdrawn from this packet:** §2.1's "landmine" (the 2 µm structural zero is
> CORRECT physics, not a settings bug); §2.3's keystone verdict (scored a mechanism that should
> not exist); §4's whole architectural question (premised on cross-synapse provenance); §4.2's
> claim-radius ask (moot — reach was never limiting inside one 400 nm nanodomain).
>
> **What survives and is still worth the advisor's time — all LOCAL:** the monogamy violation
> (mean degree 715 vs a bound of 4; 99.44% of edges inadmissible), provenance being
> monogamy-CLEAN without a cap, the ≤50 ms coincidence window making `provenance_net_age_s`
> non-operative, and the Unit-5 sheaf being a direct sum of 3 graph Laplacians rather than
> irreducible sheaf structure.
>
> **The finding that replaces the question:** if entanglement is only local there is no η-free
> cross-synapse route, so **η being dead is a hard blocker, not something to engineer around.**
>
> Full correction: `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entry `L·PO7-2`.

# Advisor packet — round 4: the multi-synapse scale, and a confound we could not design out
**PO-7 · 2026-07-19 · after building the network-shared provenance layer and running the keystone test**

Round 3 you took reading **(a)**: the pair-level input channel needs the MULTI-SYNAPSE scale, and the
next test is provenance across N synapses with input selecting *which* synapses' events overlap. We
built that and ran it.

**Headline: the mechanism works and is now proven η-free. The keystone test returned a
pre-registered NEGATIVE — but we do not believe that negative is trustworthy, for a reason that is
structural rather than fixable by better bookkeeping.** That structural problem is the question we
put to you.

---

## 1. SCORECARD ON ROUND-3 DIRECTIVES

| directive | outcome |
|---|---|
| **"Lift provenance to a network-shared event pool"** | **BUILT and VALIDATED at the data level.** Hydrolysis events carry absolute network coordinates; a dimer in any synapse claims events within reach; two dimers in *different* synapses sharing one event's daughters bond. Opt-in, **off-path bit-identical `1034/369740/0.991922159684`** (gated before the build and re-gated after scoring). |
| **"Cross-synapse edges without η"** | **CONFIRMED, and this is the strongest result in the packet.** `eta_cross = 0` in **all 20 scored runs**. Every cross-synapse edge observed is Fisher-inherited from a shared hydrolysis origin. The dead pump (r≈0.077, η=0) does not block the channel — exactly as predicted. |
| **"6 synapses, {1,2,3} vs {4,5,6}, partial pool overlap, score against activation identity"** | **Run, 5 seeds, 2 arms — verdict NEGATIVE (decomposition null).** But see §3: the specified split *is* the spatial half under a linear layout, so we added an interleaved arm; and the activation-identity label turns out to be density-confounded in a way we could not design out. |
| **"Pre-registration guard: state what would flip the verdict"** | **Done, and it earned its keep.** Three achievable routes to a negative were named before running. The guard caught a would-be third false positive (§3.2). |

---

## 2. THE MEASUREMENTS

### 2.1 The layer works — and a landmine in the code as shipped

| spacing | mean cross edges | mean overlap | seeds with cross |
|---|---|---|---|
| 0.2 µm | 2.5 | 0.046 | 2/2 |
| 0.4 µm | 0.5 | 0.007 | 1/2 |
| 0.6 / 0.8 / 1.2 / 2.0 µm | 0.0 | 0.000 | 0/2 |

A geometric prediction registered *before* the run held exactly: each synapse's grid spans 400 nm
and the claim reach is 500 nm, so the minimum cross-synapse distance is `spacing_nm − 400`, giving
zero cross edges above 0.9 µm and nonzero below.

**⚠ The inherited code's DEFAULT is `spacing_um = 2.0` — squarely in the structural-zero regime.**
The layer as committed could never have produced a single cross edge. A run at defaults would have
reported a physics-looking null caused entirely by a settings choice. We flag this because it is the
exact failure mode that has produced under-founded nulls in this program before.

### 2.2 A hard ceiling on cross-edge yield

Cross edges appear **once** (t=0.25 s: 2 edges, 1 of 15 synapse pairs, one 2-synapse component) and
then **never again through t=2.0 s**, while total provenance bonds grow **153 → 481 (+328 bonds, zero
new cross edges)** — including two ~85-dimer birth cohorts that produced none.

**Longer runs do not buy power.** The cross channel is not merely sparse; it saturates.

### 2.3 The keystone test — pre-registered verdict: NEGATIVE

6 synapses, 0.2 µm (chosen by a pre-registered overlap rule, not by yield), 5 seeds, 2 arms.

| arm | mean Q_act | mean Q_shuf | d | **mean n_multi** |
|---|---|---|---|---|
| arm1 contiguous {0,1,2} vs {3,4,5} | −0.0000 | −0.1608 | **+1.609** | **0.60** |
| arm2 interleaved {0,2,4} vs {1,3,5} | +0.0000 | −0.1525 | **+1.332** | **0.50** |

> **NEGATIVE — DECOMPOSITION NULL: the partition splits cleanly per-synapse (mean n_multi < 1).
> Input-LOCATED, not input-COMPUTING.**

Seed dominates the condition contrast: one seed is high-yield in every arm, and other seeds flip
direction between conditions. Which synapses are active does not systematically determine whether a
cross-synapse component forms. Same shape as `L·PO5-13`'s `d = 0.02`, reached by a different route.

---

## 3. THREE DESIGN DEFECTS PO-7 FOUND IN ITS OWN PRE-REGISTRATION

Disclosed, not buried — same policy as round 3.

### 3.1 The specified split IS the spatial half (caught before running)

Under `pattern="linear"`, synapses 0,1,2 are the left half of the dendrite. So round 3's literal
`{1,2,3}` vs `{4,5,6}` makes "activation identity" and "spatial half" the *same label* — it would
have re-committed the `L·PO5-13` false positive. We added **arm 2, interleaved `{0,2,4}` vs
`{1,3,5}`**, where identity is orthogonal to position.

*(Arm 2 has its own flaw: interleaving moves co-active synapses from 0.2 µm to 0.4 µm apart, so it
conflates layout with distance. It still breaks the space/identity alignment, but its criterion is
not a clean spatial test.)*

### 3.2 ⚠ The Q statistic is DENSITY-CONFOUNDED — a would-be third false positive

The pre-registration claimed *"density matched by construction: every condition activates exactly 3
of 6 synapses."* **That is false in the way that matters.** Inactive synapses sit at −70 mV, produce
almost no calcium, hence almost no dimers, hence no events and no edges. **Cross edges can therefore
form only among active synapses — trivially, by density, not by computation.**

The signature is unmistakable in the data: **`Q_act = +0.0000` in all 20 runs.** That is Newman
modularity's *degenerate* value when every edge lies inside one community (`e=1, a=1 ⇒ Q = 1−1 = 0`).
The apparent effect size `d = 1.61 / 1.33` is produced entirely by the *shuffled null* going
negative. It is an artifact of the null, not a signal.

**A PASS on the Q criteria would have been the third false positive in this series.** It was
prevented only because the verdict function required all three criteria to hold and the
decomposition criterion failed independently. We are reporting the Q result as confounded regardless
of what the function printed.

*(A first version of the metric was worse and was caught before scoring: computed at the dimer level,
Q would have been ≈1.0 by construction, since all ~370,000 intra-synapse clique edges sit inside one
activation label. It was moved to the synapse-level graph §5 names.)*

---

## 4. THE ARCHITECTURAL QUESTION FOR YOU

**Can "which synapses are active" ever be varied independently of "where the material is" — and if
not, is the density confound a flaw in our design, or is it telling us something about the physics?**

The problem is structural, not bookkeeping:

- Activation drives calcium; calcium drives dimer formation; dimers are what bond. So **activating a
  synapse is inseparable from putting material there.**
- Therefore any ON/OFF input contrast makes the activation label and the density label *the same
  label*. §8 rules density-alone insufficient — but at this scale we cannot construct an input
  contrast where they come apart.
- This is `L·PO5-11`'s "topology is a function of density alone" reappearing at the network scale,
  inside a metric we specifically designed to dodge `L·PO5-13`.

**Three readings, and we think it is your call which:**

- **(a) Design fix.** Drive *all six* synapses above dimer-forming threshold and vary **HIGH vs LOW
  drive** rather than ON vs OFF. Density is then genuinely matched and activation identity still
  varies. **Our recommendation** — but we flag a risk: `L·PO5-11` found the population is
  *peak-saturated*, so a HIGH/LOW contrast may not reach the population at all, and we would be back
  to a non-contrast.
- **(b) The confound is the physics.** If the partition tracks where the material is, and material
  necessarily follows input, then "input-located" may be all this substrate can do — and §8's
  demand for structure *beyond* density is asking for something the mechanism cannot supply. That
  would be a real answer, not a failure.
- **(c) The test needs a different observable.** Modularity against a node label may be the wrong
  instrument. If the computation is *which components form*, perhaps the observable should be
  component identity/stability across repeated presentations of the same input, rather than
  assortativity against a label at all.

**Second, narrower question:** the cross-edge yield ceiling (§2.2) blocks any statistic on the
synapse graph. Raising `provenance_net_reach_nm` (500 nm) or `provenance_net_event_rate` (0.5) would
fix the power — but we will not tune them toward a verdict. **Is there a principled physical basis
for the claim radius?** It currently reads as a modeling choice, and we would rather it be derived
from phosphate diffusion than swept.

---

## 5. WHAT PO-7 IS NOT CLAIMING

- **Not that the mechanism failed.** It works, it is validated at the data level, and the η-free
  cross-synapse channel is demonstrated for the first time — with the pump dead.
- **Not that the keystone is disproven.** The verdict is a decomposition null measured under a
  density-confounded label and a ceiling-limited edge count. **§8 at multi-synapse scale is OPEN,
  not answered negative.** We are explicitly declining to bank a negative the measurement did not
  earn.
- **Not that the rate constants are physical.** `reach_nm`, `event_rate`, `age_s` are modeling
  choices reported as swept, not certified — hence the question in §4.
- **Not that the geometry generalizes.** The claim radius is a 2D projection (z is dropped while the
  layout jitters z by ±0.2 µm), biasing cross-edge formation *upward*; and cross edges require
  spacing < 0.9 µm, excluding the upper half of the physiological spine-spacing range.
- This remains an **(A)-reading** result and is not evidence for (B).

---

## 6. THE DELIVERABLE STATE

The network-shared provenance layer is committed, opt-in, bit-identical-off (gated twice), and
validated. Six explicit-path commits. Full evidence — the geometry sweep, the power trace, the scored
test, and all three self-caught defects — is in `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entry
`L·PO7-1`, with the pre-registration in `PREREG_PO7_UNIT2_MULTISYNAPSE_KEYSTONE.md` and the running
open-questions list in `coordination/requests/po7-provenance-network/notes.md`.

**The single decision that gates the next unit is §4 above.** The concrete next design — all-six
driven, HIGH vs LOW, constant co-active spacing, plus a yield increase — is ready to run, but it
depends on reading (a) being the right one, and on §4's second question having an answer better than
"sweep it."
