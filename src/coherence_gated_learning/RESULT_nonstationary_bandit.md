# BENCHMARK 9 — established non-stationary bandit: a NEGATIVE, with the cause identified

First test against PUBLISHED algorithms rather than baselines we wrote. Environments follow the
abruptly-changing Bernoulli bandit protocol (Garivier & Moulines 2011); regret is against the PER-STEP
oracle. D-UCB and SW-UCB were tuned with the paper's rules **using the true breakpoint count**, i.e. the
baselines were given information our agent was not.

## Result: the primitive LOSES

| algorithm | ENV-A regret (K=3, 2 breakpoints) | ENV-B regret (K=5, 9 breakpoints) |
|---|---|---|
| UCB1 (stationary) | 364.6 | 862.8 |
| Thompson Beta (stationary) | 912.1 | 1663.2 |
| D-UCB (GM2011) | 836.0 | 1528.0 |
| **SW-UCB (GM2011)** | **333.1** | 811.1 |
| EXP3.S (Auer 2002) | 1998.1 | 2225.0 |
| **Discounted Thompson** | 433.4 | **681.4** |
| CGL primitive (raw {0,1}) | 1162.8 | 1653.7 |
| CGL primitive (centered) | 1320.6 | 822.9 |

Worst or near-worst in ENV-A. Reported as measured.

## Cause: the SELECTION rule, not the learning

Diagnostic 1 — the centered arm is structurally broken here. In ENV-A segment 1 the best arm has mean 0.5,
so its centered value is exactly 0.0 and its commit probability is `clip(1.6 x 0.0) = 0`. Nothing ever fires;
selection falls through to uniform random.

Diagnostic 2 — with PERFECT value estimates supplied (no learning involved), "one independent coin per
candidate" only commits reliably when values are WELL SEPARATED:

| true values | P(pick best) |
|---|---|
| 1.0 / -1.0 / -1.0 | **1.000**  <- the foraging regime |
| 0.9 / 0.1 / 0.1 | 0.849 |
| 0.6 / 0.4 / 0.4 | 0.481 |
| 0.5 / 0.45 / 0.4 | 0.384  (uniform = 0.333)  <- the bandit regime |

So the ceiling is set by value separation, independent of learning. When several coins fire the tie is broken
UNIFORMLY, discarding the ranking. Benchmark 8's perfect foraging score sits on the top row because a
conjunctive rule is satisfied or not, making values near-binary; Bernoulli arm means of 0.5/0.4/0.3 sit on
the bottom row.

**This is a boundary condition, and it is the honest generalisation of the earlier wins: per-component
collapse is a CONSOLIDATION rule -- an independent local decision about whether THIS memory commits -- not
an action-selection rule for choosing among close competing alternatives. The brain does not select actions
with independent local coins either; that is striatal/basal-ganglia competition, a different circuit.**

## Two grounded fixes were tried and BOTH FAILED

| variant | ENV-A | ENV-B |
|---|---|---|
| fixed baseline, no consolidation | 1155.7 | 1632.4 |
| adaptive baseline (tonic DA = average reward rate; Niv/Daw/Dayan 2007) | 1658.2 | 1804.7 |
| + consolidation, gain 0.3 / 1.0 / 3.0 | 1545.7 / 1417.0 / 1288.4 | 1623.5 / 1455.3 / 1307.9 |

The adaptive baseline made ENV-A **worse**. Consolidation helped monotonically but plateaued near 1290,
still ~4x the best published. Hill-climbing stopped here deliberately: tuning toward a benchmark is what the
discipline forbids.

## The apparent fix is NOT a CGL result — checked, and retracted

Swapping the selection module for winner-take-all (keeping the CGL credit machinery) looked like a large win:

| | ENV-A | ENV-B |
|---|---|---|
| CGL credit + WTA, rate 0.05 / 0.1 / 0.2 | **82.9** / 117.0 / 157.5 | 675.1 / 742.9 / 793.6 |
| best published | 333.1 | 681.4 |

ENV-A recovers the new best arm in **22 steps** after each switch. But the control settles it:

```
plain recency-weighted greedy (no CGL at all): ENV-A regret 82.9   <- IDENTICAL
```

On a context-free bandit every arm is a SINGLETON component, so `value()` is just a leaky average and
`argmax` is just greedy. **The CGL machinery contributes exactly nothing to that number, and the win must
not be reported as a CGL result.** What the number does show is the well-known fact that a simple
recency-weighted greedy rule beats D-UCB / SW-UCB / EXP3.S on these environments.

## What this actually tells us

The context-free bandit **cannot test what we built**: with one unit per arm there are no conjunctions, so
the distinctive machinery (graph binding, per-component collapse, partial reactivation) is inert by
construction. The right established family is the **CONTEXTUAL** bandit, where components are feature
conjunctions and the conjunctive representation is load-bearing. That is the next benchmark.
