# Real data: public medical appointment no-shows — the primitive LOSES, and the dataset explains why (2026-08-24)

110,527 de-identified Brazilian clinic appointments, 35 binary features, chronological 70/30 split,
no-show rate 0.202. Structurally the same problem PAUL solves.

| | AUC | balanced acc | fit |
|---|---|---|---|
| coherence-gated (online, single pass) | **0.6486** | 0.5725 | 10.8 s |
| logistic regression (additive) | **0.7087** | 0.5952 | 0.0 s |
| gradient boosting (interactions) | **0.7133** | 0.5975 | 0.5 s |

**The primitive is beaten by both baselines.** It is above chance (0.5), so it learned something real, but it
is not competitive as a tabular predictor. Reported as-is.

## The number that matters more than our loss

**Gradient boosting beats logistic regression by 0.0046 AUC — essentially nothing.**

A strong interaction-capturing model performing level with a purely additive one means **this dataset has
almost no interaction structure to find.** Nearly all its signal is additive (lead time dominates, then
SMS/age/etc., each contributing independently).

So this dataset **cannot test our claim**. The primitive's demonstrated advantage (B2/B3) is finding
CONJUNCTIONS that additive models are blind to. Where there are no conjunctions, there is nothing to find,
and a similarity memory just loses to a well-fit linear model. This is a negative result about the TEST BED
as much as about the method.

## Why the primitive underperforms here (mechanism, not excuse)

1. **The tabular mapping degrades it.** With no time dimension, every feature of a row binds into one
   component and the learner becomes a SET-SIMILARITY MEMORY — effectively k-NN over 11,338 stored
   feature-sets. The temporal-binding half of the primitive is not exercised at all.
2. **k-NN-style retrieval is simply weaker than logistic regression** on additive tabular data. That is
   expected, not surprising.
3. Single online pass versus full-batch optimisation.

## The actionable consequence for PAUL — a cheap pre-test

Before anyone extracts scheduling data to try this, run **one** diagnostic on PAUL's own features:

> Fit gradient boosting and logistic regression on the same features and compare AUC.

- **Gap ~0 (as here):** the data is additive. Our primitive will not help, and no further work is justified.
  This costs an afternoon and could save weeks.
- **Meaningful gap:** genuine interaction structure exists — that is the regime where the primitive's
  conjunction-finding is worth testing, and where surfacing WHICH combinations matter has value beyond the
  score itself.

That diagnostic is the honest gate, and it does not require our method at all.

## What is NOT concluded

- Not that the primitive is worthless: B2 showed it solves conjunctions that a scalar trace provably cannot,
  and B3 that it holds 100% across 64 overlapping rules. Those results stand.
- Not that it is a general tabular learner. It is not, and should not be presented as one.
- The next fair real-data test needs a dataset with DEMONSTRATED interaction structure (GBM >> linear).
  Finding one is the obvious next step; this one was chosen for its resemblance to PAUL, not for its
  interactions.
