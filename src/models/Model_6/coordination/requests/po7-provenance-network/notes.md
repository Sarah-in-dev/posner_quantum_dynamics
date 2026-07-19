# PO-7 — open questions, with my recommendation (not blocking; I keep working)

## Q0 — STATISTICAL POWER: the scored test may be underpowered, and a thin null would overclaim

**Raised 2026-07-19 ~20:52 UTC, from Unit 1's first 6 rows, BEFORE Unit 2 was run.**

Unit 1 measures, at the pre-registered spacing of 0.2 µm, only **2–3 cross-synapse provenance
edges per run** (against ~200 intra). At 0.4 µm it is 0–1; at ≥0.6 µm it is 0.

Unit 2's statistic is Newman modularity on the **6-node synapse graph**. With 2–3 edges among 6
nodes, `Q` is dominated by noise and `n_multi` (components spanning ≥2 synapses) can be at most
2–3 by construction. **A "decomposition null" verdict from a probe this thin would be
overclaiming a negative** — asserting the mechanism does not compute when the measurement simply
could not have detected it either way. That is the same criterion-mis-registration family as
Units 8/9/13/16b, just pointing the other direction.

**Recommendation (adopted): run a cheap POWER CHECK before committing to the scored Unit 2** —
cross-edge count vs `T_SIM` at 0.2 µm, one seed, no scoring. Two outcomes, both useful:
- **Edges accumulate with sim time** ⇒ raise `T_SIM` for Unit 2 to reach adequate power, then run
  the pre-registered test as written. Raising power is decided BEFORE scoring and is not
  tuning-to-verdict; the verdict function and spacing rule are unchanged.
- **Edges saturate at ~3** ⇒ the honest finding is that **the layer cannot generate enough
  cross-synapse structure to support the §8 keystone test at all** at the committed
  `reach_nm=500` / `event_rate=0.5`. That is a stronger, more useful result than a thin null, and
  it is reported as a **characterization of the mechanism's ceiling**, not as a keystone verdict.

Either way the §8 keystone question would remain **not-yet-answered** rather than answered
negative — and saying so is the honest report. Reporting the ceiling is not a failure to deliver;
it is the finding that the cross-synapse channel, as built, is too sparse to carry the test.

## Q1 — the kickoff's {1,2,3} vs {4,5,6} split IS the spatial half, i.e. the L·PO5-13 trap

**This is the most important thing on this page.** The kickoff specifies conditions
`{1,2,3}` vs `{4,5,6}` and metric "modularity against the ACTIVATION-IDENTITY label (NOT
spatial half — that was PO-5's false positive, L·PO5-13)".

But in `pattern="linear"` (`multi_synapse_network.py:961-963`) positions are
`positions[:, 0] = np.arange(n) * spacing_um`. So synapses 0,1,2 are the LEFT half of the
dendrite and 3,4,5 the RIGHT half. **Under a linear layout, "activation identity" and
"spatial half" are the SAME LABEL** — the exact confound that made L·PO5-13 a false positive
would be re-introduced by following the kickoff literally.

Quoting the scar (`RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md:126-128`):
> the "input label" was **which spatial half a dimer sits in**, so Q=0.15 detects **spatial
> locality** (dimers bond to neighbours because they claim the nearest events). **That is
> GEOMETRY, which §8 explicitly rules insufficient**

**Recommendation (adopted, and built into the pre-registration):** run the contiguous split
as ARM 1, and add **ARM 2, an INTERLEAVED split — active `{0,2,4}` vs `{1,3,5}`** — in which
activation identity is orthogonal to spatial position. A result that only appears in ARM 1
is spatial and must be reported as a negative. A result that survives ARM 2 is genuinely
activation-identity. **ARM 2 is the discriminator; ARM 1 alone cannot distinguish the two
hypotheses and would repeat the L·PO5-13 error.**

This does not relitigate a locked decision — it *adds* the control the kickoff's own stated
rationale demands.

## Q2 — is the spacing needed for pool overlap biologically defensible?

Geometry (verified, see heartbeat) forces `spacing < 0.9 µm` for any cross-synapse claim at
`reach_nm = 500`. Real dendritic spine spacing is commonly ~0.5–2 µm, so the low end of the
required band is not absurd, but the upper half of the plausible range is excluded.

**Recommendation:** treat spacing as a **swept independent variable, reported in full with
the nulls included**. Do NOT select the spacing that maximises Q. If the only spacing that
produces overlap sits at the edge of physical plausibility, that is itself the finding and
gets reported as a limit on the mechanism, not hidden.

## Q3 — the z-axis is dropped in the provenance distance

`abs_xy` uses `d['dimer'].position[:2]` and `positions[syn_idx][:2]` — the claim radius is a
**2D projection**, while `pattern="linear"` puts ±0.2 µm of jitter on BOTH y and z
(`:964-965`). So z-separated synapses are treated as closer than they are, biasing cross-edge
formation *upward*.

**Recommendation:** leave the code as-is for this unit (changing it is a physics change that
would need its own bit-identity proof and is not what PO-7 was chartered to do), but **report
it as a stated limit** — any positive result is measured under a 2D-projected reach and would
need re-checking in 3D. Flagged for the next worker.

## Q4 — `_prov_last_stats['overlap_frac']` is a single-step quantity, not a run quantity

Events fully claimed (`slots_free == 0`) are pruned at the top of the NEXT step (`:448-450`),
so `overlap_frac` only ever reflects claims made in the current step. It is a valid
instantaneous diagnostic but must NOT be read as "the fraction of the pool that is shared
over the run."

**Recommendation:** the probe takes the **peak over steps** and reports it as such; the
run-level overlap claim rests on the independently recomputed final cross/intra edge split,
not on this field. Done in `po7_unit1_cross_edge_validation.py`.

## Q5 — dimers born before any event exists never get provenance

`:504-505` marks new dimers seen even when the event pool is empty, so they can never claim
later ("born-with" semantics).

**Recommendation:** keep it — it is faithful to the per-synapse build — but report the
affected fraction so a large value cannot silently suppress edge formation.

## Q6 — MY PREREG'S "density matched by construction" CLAIM IS WRONG (found mid-run, recorded)

**Found 2026-07-19 ~21:20 UTC, from Unit 2's first scored row, BEFORE the verdict was read.**

The prereg asserts *"Density matched by construction: every condition activates exactly 3 of 6
synapses."* **That is false in the way that matters.** Equal *counts* of active synapses is not
matched *density across the graph's nodes*: inactive synapses sit at -70 mV, generate almost no
calcium, hence almost no dimers, hence no events and no edges. Cross edges can therefore form
ONLY among active synapses — **trivially, by density**, not by computation.

Evidence, first scored row: `Q_act = +0.0000` exactly, with `Q_shuf = -0.3075`. Q_act=0 is
Newman modularity's DEGENERATE value when every edge lies inside one community (e_11=1, a_1=1
=> Q = 1-1 = 0). The apparent effect is an artifact of the null, not a signal.

**ARM 2 does not rescue this.** ARM 2 was registered as the guard against the SPATIAL confound
and it does that job correctly. But interleaved `{0,2,4}` has the identical density structure —
edges can only appear among the three driven synapses. **Neither arm separates input-COMPUTING
from density.** This is L·PO5-11's finding ("topology is a function of density alone") reappearing
in a metric I designed to avoid L·PO5-13, and I did not see it when writing the prereg.

**Consequence:** a "PASS" printed by the Unit 2 verdict function on the Q criteria would be a
THIRD false positive in this series, not a keystone result. It is reported as confounded
regardless of what the function prints. The `n_multi` (decomposition) criterion is unaffected and
remains meaningful.

**THE DESIGN FIX for the next worker (this is the real deliverable of Unit 2):**
drive **all six synapses above dimer-forming threshold** so every node carries comparable
density, and vary the input as **HIGH vs LOW drive** on the two groups rather than
**ON vs OFF**. Then activation identity varies while density is genuinely matched, and Q against
activation identity means what it was supposed to mean. Combined with the interleaved ARM 2
layout, that design is confounded by neither space nor density.

**Blocking caveat on that fix:** it does not solve the sparsity ceiling (Unit 1b: 2 cross edges,
1/15 synapse pairs, saturating). Any redesign must ALSO raise cross-edge yield — via
`provenance_net_reach_nm`, `provenance_net_event_rate`, or the born-with claiming rule — before
a modularity statistic on the synapse graph has enough edges to carry a verdict. **Fix the
confound and the power together, or the next test is unreadable too.**
