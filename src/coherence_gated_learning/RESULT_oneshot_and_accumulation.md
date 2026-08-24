# One-shot baselines, and the accumulation mechanism the abstraction was missing (2026-08-24)

## Q1 — how does it compare to standard one-shot / single-pass learners?

Sarah asked the right question, and it is the sharpest available falsification: our readout is close in
spirit to k-NN, so if plain k-NN matches it, the "primitive" adds nothing over instance-based retrieval.
All models given the IDENTICAL temporal representation (context units + decayed history units).

| model | AUC | bal-acc | fit |
|---|---|---|---|
| online logistic / SGD (single epoch) | **0.7052** | **0.5995** | 0.0 s |
| Bernoulli naive Bayes (single pass) | 0.7034 | 0.5917 | 0.0 s |
| k-NN Jaccard, k=25 | 0.6943 | 0.5904 | 21.6 s |
| **coherence-gated, as built** | **0.6886** | 0.5962 | 2.5 s |

**We were LAST on AUC**, and beaten by k-NN — i.e. the extra machinery was adding nothing over similarity
retrieval. Reported straight; this is the comparison that should have been in Benchmark 5.

## Q2 — the biology is NOT one-shot. What was missing?

Sarah's point: real synapses are tagged, weighted through the plasticity cascade, and synapses that fire
again ACCUMULATE. Our abstraction had latched that into a binary commit and then read out the MEAN outcome
per component — so **a pattern seen ONCE scored a full-strength ±1**: maximum confidence from a single noisy
observation, and no benefit whatsoever from repetition. That is exactly backwards from the biology, where one
event produces a WEAK change and strength BUILDS with confirmation.

Implemented as evidence-weighted accumulation, `score = sum_reward / (count + k)` — the Bayesian form of
"one event moves you a little, repetition moves you a lot". `k = 0` reproduces the old mean behaviour.

| k | AUC |
|---|---|
| **0 (old behaviour)** | **0.6886** |
| 1 | 0.7044 |
| **2** | **0.7054** |
| 5 | 0.7028 |
| 10 | 0.6976 |
| 20 | 0.6900 |
| 50 | 0.6786 |

**+0.017 AUC, moving us from LAST to level with the best one-shot learner** (SGD 0.7052), past naive Bayes
(0.7034) and k-NN (0.6943). The improvement is broad across k=1–5 rather than knife-edge, and the mechanism
was motivated by the biology rather than fitted — but `k=2` is a chosen value and is flagged as such.
Balanced accuracy moved the other way slightly (0.5962 → 0.5911); AUC is the threshold-free measure and the
honest headline, but the mixed direction is recorded.

## What accumulation did NOT fix

It raises the LEVEL, not the SLOPE. AUC per successive chunk of the test period:

| | chunks | trend |
|---|---|---|
| no accumulation | 0.6816 0.6943 0.6773 0.7082 0.6818 | +0.0014/chunk |
| with accumulation | 0.7011 0.7081 0.6980 0.7218 0.6963 | +0.0004/chunk |

Every chunk lifts by ~0.02, but the model still does not IMPROVE as more data arrives. So saturation is a
SEPARATE deficiency from over-confidence, and it is still there.

## What is still missing versus Model 6

Two further accumulation mechanisms the biology has and this abstraction still does not:

1. **Tag magnitude should grow with repeated activation.** Model 6 forms MORE dimers on repeated drive, so
   the tag is bigger and more readable. Our `activate()` sets `trace = 1.0` — saturating, so a synapse driven
   ten times is indistinguishable from one driven once.
2. **Structural weight should accumulate across commitments.** Model 6 grows spine volume / stable actin over
   repeated commitment events. We latch a boolean and never grow the structure.

Either could plausibly restore "more data helps". They are the next concrete step, and they come straight
from the biology rather than from tuning.
