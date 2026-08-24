# BENCHMARK 10 — mushroom contextual bandit: a SECOND negative, and CGL's mechanisms actively HURT here

Canonical mushroom bandit (Blundell et al. 2015; Riquelme et al. 2018). T=20000, regret vs the per-round
oracle (eat edible +5, abstain on poisonous 0; eating poisonous is -35 half the time, expected -15).

## Headline

| algorithm | regret |
|---|---|
| uniform random | 98167 |
| eps-greedy LINEAR (the no-conjunction control) | 9621 |
| LinUCB (Li et al. 2010) | 9348 |
| LinTS (Agrawal & Goyal 2013) | 26251 |
| CGL-single (full 22-way conjunction) | 51659 = always abstains |
| CGL-single, coin select | 98194 (worse than random) |
| CGL-ensemble M=32 d=3, argmax + novelty | 17228 (bimodal: 2/8 seeds deadlock) |
| **CGL-ensemble, exact-key optimism** | **4534** |
| **CONTROL: same ensemble, running mean, NO CGL** | **2581**  <- best |

## Two faults of MINE, found and fixed before reporting

1. **The first run was invalid.** Every baseline had an exploration device (eps / UCB bonus / posterior
   sampling); the CGL argmax arms had none. Both deadlocked into always abstaining -- predicted
   `5.0 x 20000 x P(edible) = 51800`, measured 51659, and CGL-single and CGL-ensemble returned *byte-identical*
   regret, which is what exposed it. Fixed with a novelty/optimism bonus (dopamine novelty response, Kakade &
   Dayan 2002 -- which is also textbook optimistic initialisation, stated plainly rather than claimed as ours).
2. **The novelty bonus was then neutralised by partial reactivation**, which reports "evidence" for any
   overlapping component, so optimism never fired and 2/8 seeds still deadlocked (sd 18452 > mean). Treating a
   *guess* as evidence is the bug. With optimism gated on the EXACT component being untried: 0/8 deadlocks,
   4534 mean, per-seed 3140-6285.

## The attribution control — and CGL loses it

| variant | regret |
|---|---|
| **CONTROL: conjunctions only (running mean, no CGL)** | **2581** |
| + leaky / active-depression update only | 4534 |
| + Jaccard partial reactivation only | 23714 |
| FULL CGL (leaky + Jaccard) | 17228 |

**Every CGL mechanism makes it worse.** The entire 2-4x win over the linear control comes from "use feature
conjunctions with a running mean and optimistic init" -- a plain random-subspace tabular learner with no CGL
in it. As in Benchmark 9, the apparent CGL win is fully explained by a non-CGL control, and is not claimed.

## Why, and what it predicts

The mechanisms are mismatched to this task, for reasons that follow from what they are FOR:
- **Active depression** buys revisability under change. Mushroom is **stationary**, where a running mean is
  optimal and discounting old data is pure loss.
- **Partial reactivation** buys answers for unseen combinations. With d=3 slots the same conjunctions recur
  constantly, so exact evidence is nearly always available and the Jaccard blur only corrupts it -- and it
  additionally suppresses the optimism that prevents deadlock.
- **Graph binding was never exercised at all** (see below).

## The graph topology was inert in BOTH established benchmarks

Audited by call-site: `activate` / `decay` / `components` are called by benchmark_structure, _scaling,
_generalization, _delayed_credit and the regression suite -- and by NEITHER benchmark 9 NOR 10, nor foraging,
nor the no-show tests. Those used EXPLICIT mode: a `frozenset -> value` store with Jaccard retrieval.

This is structural, not an oversight of convenience. Verified directly: presenting all 22 mushroom features
**simultaneously** makes the graph return **one component of size 22** -- byte-identical to the conjunction key
built by hand. The same units **separated in time** return **three** components. **Binding only computes
something when grouping is determined by TIMING.** Static tabular rows and context-free arms have no temporal
structure, so the graph is a no-op by construction, and neither benchmark could have tested it.

## Standing conclusion after two established benchmarks

Two attempted, zero wins, and neither exercised the distinctive machinery. Every win to date that DID use the
graph is on temporally-structured tasks we designed; every loss is on static data where the graph was inert.
The next test must have all three: conjunctive structure (so the representation matters), non-stationarity
(so active depression matters), and temporal grouping (so binding matters). Prediction to be registered
BEFORE running it, and a null reported as a null.
