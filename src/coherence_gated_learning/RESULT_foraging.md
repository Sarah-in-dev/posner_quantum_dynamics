# The fourth primitive: learned state steers sampling — demonstrated, and it cuts both ways (2026-08-24)

The last untested claim: *"stronger synapses pull the agent's trajectory toward those features on future
trials. No separate policy network. Spine enlargement IS the policy update."* Every earlier benchmark,
including the closed-loop bandit, drew inputs from a FIXED distribution. Here the agent chooses which patch to
visit, so what it learns determines what data it subsequently gets.

## Result (12 patches, hidden feature-conjunction rule, unsignalled switch at trial 2,000, 30 seeds)

| agent | pre | post | never recovered | patches sampled pre | patches post |
|---|---|---|---|---|---|
| **coherence-gated (no exploration parameter)** | **1.000** | 0.695 | 0% | **8.1 / 12** | **2.2** |
| softmax T=0.25 | 0.998 | **0.888** | 0% | 11.8 | 4.3 |
| softmax T=0.5 | 0.875 | 0.634 | 7% | 12.0 | 11.9 |
| eps-greedy fixed 0.05 | 0.955 | 0.361 | 50% | 12.0 | 10.7 |
| eps-greedy fixed 0.15 | 0.866 | 0.226 | 67% | 12.0 | 12.0 |
| eps-greedy decaying | 0.951 | 0.034 | 97% | 12.0 | 8.3 |

## What IS demonstrated

**The primitive shapes its own input distribution, and no baseline does.** Before the switch it sampled only
**8.1 of 12 patches** while EVERY baseline sampled all 12. Nothing told it to stop visiting unrewarding
patches — the same stored structure that represents what it has learned is what steers where it looks. That is
the fourth primitive, and it is visible in the coverage column rather than in the reward column.

It also reached a perfect pre-switch score (1.000) while doing it, i.e. the focusing was efficient, not lucky.

## What it COSTS — and this is the finding

**Softmax T=0.25 beats us after the switch (0.888 vs 0.695).** The coverage column says why: post-switch we
sampled only **2.2 patches**. The agent narrowed onto what had been working and then could not see the
alternatives, because it had stopped visiting them. **Self-selected data plus consolidation produces tunnel
vision.**

Note the contrast with Benchmark 7, where the primitive RECOVERED BETTER than softmax. There the contexts were
handed to the agent, so it kept receiving evidence about everything whether it liked it or not. **Closing the
loop introduces a failure mode that does not exist in the open-loop version** — the very coupling that makes
sampling efficient is what blinds it after a change.

The double-edge was pre-registered in the harness docstring as the thing to watch for, and it materialised.

## What this implies is still missing

Biology does not rely on value alone to decide where to look — it has novelty and surprise signals that force
re-sampling of things currently believed worthless. Our agent has no such drive: once a patch's value is low
it is simply never revisited, so the evidence that would overturn the belief is never collected.

That is the next mechanism, and like the accumulation fix it comes from the biology rather than from tuning.
Until it exists, the honest statement is: **the primitive steers its own sampling efficiently, and is
vulnerable to lock-in when the world changes underneath it.**

---

# UPDATE: the lock-in is fixed by the ACTIVE DEPRESSION arm (2026-08-24)

## First attempt — a novelty drive — FAILED, and the failure was informative

Added an unfamiliarity bonus (`drive = gain*value + nov/sqrt(count+1)`), motivated by the brain's
novelty/familiarity response. Result:

| novelty weight | pre | post | patches pre | patches post |
|---|---|---|---|---|
| 0 (none) | 1.000 | 0.695 | 8.1 | 2.2 |
| 0.2 | 1.000 | 0.699 | 8.6 | 2.0 |
| 0.5 | 1.000 | 0.693 | 9.5 | 2.0 |
| 0.8 | 1.000 | 0.690 | 11.8 | 1.8 |

It broadened EARLY sampling (8.1 -> 11.8 patches) but post-switch coverage stayed at ~2.0 and recovery did not
move. **Novelty-by-count decays permanently: once every patch is familiar nothing is novel again, however
wrong the beliefs have become.** Reported as a failed mechanism, not quietly dropped.

## The actual cause: accumulated evidence could not be REVISED

A component with `sum=+50, count=50` scores ~0.96 and needs ~50 contradicting observations to flip. **The
evidence accumulation that cured over-confidence is precisely what caused the lock-in.** Model 6 does not wait
for decay — it has an ACTIVE DEPRESSION arm (PP1 strips CaMKII-pThr286; DAPK1 disrupts the GluN2B complex),
so a memory is actively taken apart when the reward signal turns against it. The abstraction had kept the
potentiation arm and dropped the depression arm.

## Result with active depression

| agent | pre | post | never recovered | patches pre | patches post |
|---|---|---|---|---|---|
| coherence-gated, sticky sum | 1.000 | 0.695 | 0% | 8.1 | 2.2 |
| **+ active depression (LTD arm)** | **1.000** | **1.000** | **0%** | 9.7 | 10.8 |
| softmax T=0.25 (best baseline) | 0.998 | 0.888 | 0% | 11.8 | 4.3 |
| eps-greedy fixed 0.05 | 0.955 | 0.361 | 50% | 12.0 | 10.7 |
| eps-greedy decaying | 0.951 | 0.034 | 97% | 12.0 | 8.3 |

**Perfect before AND after the unsignalled change**, beating the best tuned baseline (0.888), with no
exploration parameter. Robust across recency rates 0.05–0.4 (an 8x range, all 1.000) and unchanged by adding
the asymmetric-LTD variant, so it is not knife-edge. At the fastest rate it also KEEPS the focused sampling
that made the fourth primitive visible (8.1 patches pre, 7.3 post) while still recovering fully.

## Honest naming

The value update is now `value <- value + rate*(r - value)` — the standard incremental / exponentially-weighted
rule. It is NOT novel and is not claimed as such. What remains distinctive is the **per-component conjunctive
representation** and **exploration arising from commit probability rather than a temperature**; the fix here
restores revisability, which the sticky sum had removed.
