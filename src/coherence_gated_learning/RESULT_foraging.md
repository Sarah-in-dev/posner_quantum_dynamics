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
