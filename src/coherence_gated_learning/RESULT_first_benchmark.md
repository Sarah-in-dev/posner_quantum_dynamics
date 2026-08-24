# First honest benchmark of the abstract primitive — NEGATIVE (2026-08-24)

## Result

| task | coherence-gated | best trace baseline |
|---|---|---|
| **A. sample efficiency** (episodes to criterion) | median **11**, mean 69.6, solved **85%** | median 13, mean **13.4**, solved **100%** |
| **B. retention** after 400 interference episodes | **5.0%** | **100%** |

**The classical eligibility-trace baseline matches or beats the coherence-gated primitive on both tasks.**
Task A is a tie on the median but our tail is much worse (15% of runs never reach criterion). Task B we lose
outright.

## The crux, and it is not a detail

The baseline's best configuration used **trace_tau = 216 — the same time constant as ours**. Given an equally
long eligibility trace, the classical three-factor rule (`w += lr * r * e`) does everything the coherence-gated
primitive does on this task. **The long horizon was the whole advantage, and it is classically available.**

That is the same conclusion the substrate-necessity test reached one level down: there we found a plain
decaying scalar reproduces the emergent quantum coherence (p = 0.16, indistinguishable). Here we find a plain
decaying scalar reproduces the *architecture* too. The two results are consistent, and together they say the
distinctive contribution is NOT the long trace per se — because nothing about a long trace requires this
substrate or this architecture.

## Diagnosis of the retention loss (stated, not used as an excuse)

Two things are tangled in Task B and they should be separated before anyone reads too much into the 5% vs 100%:

1. **The task as designed rewards inertia.** Interference episodes pair distractor activity with a spurious
   reward 15% of the time. In that world distractors genuinely *are* (spuriously) predictive, so a correct
   learner should credit them. The baseline "wins" partly by having lr = 0.01 and barely moving at all.
2. **Update magnitudes were not matched.** Our commitment applies a full-size weight step (lr = 1.0, one-shot
   latch) against the baseline's 0.01 nudge. That is not a like-for-like comparison of architectures.
3. **A faithfulness gap in the abstraction:** in Model 6 commitment is ONE-SHOT — the measurement token is
   consumed (`_measurement_gate_opened = False`) and a committed synapse does not re-commit. `cgl.py` latches
   `committed` but still allows repeated weight updates on later rewards, so noise accumulates without bound.
   That is a real defect in the extraction, not in the biology.

## What this does and does not license

- It does NOT show the primitive is worthless. It shows the *first regime we guessed* is not where any
  advantage lives, and that our headline framing ("native long-horizon credit assignment") is matched by a
  classical trace with a long time constant.
- It does NOT rescue itself by tuning. The fix for (2) and (3) is to make the comparison fair and the
  abstraction faithful, then re-run — and if it still loses, that is the answer.
- The claim "solves problems current ML cannot" is **not supported by any evidence we currently have**, and
  should not be made until a benchmark shows a separation that survives a hyperparameter-swept baseline.

## The honest next question

If a long classical eligibility trace does this, what is left that is distinctive? Candidates, in order of how
testable they are:
1. **One-shot commitment under a single reward** (sample efficiency in the extreme-sparse limit, n=1 reward)
2. **Per-cluster stochastic readout producing structured exploration** — the primitive we have NOT tested at
   all, and the only one that is genuinely unlike a scalar trace
3. **Capacity/competition effects** at scale, where continuous proportional updates interfere and discrete
   latched ones may not
If none of these separate, the honest conclusion is that the biology is interesting and the architecture is
not novel — which is a publishable finding about the biology, not a new ML paradigm.
