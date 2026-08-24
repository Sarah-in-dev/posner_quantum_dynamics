# Real data WITH TIME — the primitive beats the additive model that was handed engineered history (2026-08-24)

## Why this supersedes Benchmark 5

Benchmark 5 flattened each appointment into one row with all features simultaneous. That was the wrong test:
it discarded the temporal dimension, which IS the primitive's mechanism, and it withheld real signal from
every model. The data has strong temporal structure — 66% of appointments belong to repeat patients, and the
no-show rate is **35.2% after a previous no-show versus 18.7% after a previous show**. Benchmark 5 measured
three handicapped learners and concluded, wrongly, that the dataset had no interaction structure.

## Setup

Per-patient appointment SEQUENCES. For the primitive, a past outcome lays a trace that decays with REAL
ELAPSED DAYS (tau = 45 days); traces still readable when the next appointment arrives BIND with that
appointment's context, so history-x-context conjunctions become learnable components. The baselines are given
the SAME history, hand-engineered the standard way (previous outcome, days since last visit, number of prior
visits, prior no-show rate). Chronological 70/30 split.

## Result

| model | AUC | balanced acc |
|---|---|---|
| **coherence-gated, native time (online, single pass)** | **0.6886** | **0.5962** |
| logistic regression + hand-built history | 0.6692 | 0.5829 |
| gradient boosting + hand-built history | **0.7340** | **0.6156** |
| logistic regression, NO history | 0.6629 | — |
| gradient boosting, NO history | 0.7122 | — |

**The primitive beats logistic regression even though logistic regression was handed hand-engineered history
features and the primitive built its own temporal representation** — online, in a single pass, with no feature
engineering, in 2.4 s, storing 5,815 components.

**Gradient boosting still wins** (0.7340 vs 0.6886). It is batch, multi-pass, and also received the
hand-built history. That gap is real and is not explained away.

## The diagnostic flipped, and that is the important part

Interaction headroom (GBM − logistic regression, both with history): **+0.0648**, versus **+0.0046** in the
flattened Benchmark 5.

So the earlier conclusion — "this dataset has no interaction structure" — was **an artefact of removing time**.
Once the temporal dimension is present, there is substantial interaction structure, which is precisely the
regime where conjunction-finding is worth something. Benchmark 5's negative was largely a self-inflicted
wound, and it is superseded by this entry rather than deleted.

## What this does and does not support

- It DOES support: the primitive's native temporal machinery is competitive on real data and beats the
  additive baseline without any feature engineering — a like-for-like win over the model class that most
  production scoring pipelines actually use.
- It does NOT support: beating strong batch models. Gradient boosting is ahead by 0.045 AUC.
- The honest framing of the advantage is therefore NOT accuracy. It is: **no feature engineering, single
  online pass, learns from outcomes as they arrive, and names the conjunctions it used.**

## Consequence for the PAUL pre-test (revised)

The gate proposed after Benchmark 5 still holds but must be run **with temporal features included**:
fit GBM and logistic regression on PAUL's features *plus* patient history, and compare AUC. Running that
diagnostic on a time-flattened table would have produced a false negative here, and would do the same there.
