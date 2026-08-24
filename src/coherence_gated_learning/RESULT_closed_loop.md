# The closed loop: exploration emerging from learned structure (2026-08-24)

Benchmarks 1–6 were passive prediction, which makes the primitive's two most distinctive claims invisible.
This tests the one with no analogue in TD(lambda), k-NN or gradient boosting:

> "Fragmented graphs -> many independent stochastic samples -> diverse outcomes. Connected graphs -> one
>  sample -> uniform outcome. This creates natural exploration without epsilon-greedy, softmax temperature,
>  or a separate exploration policy. **The structure of what's been learned determines how exploratory the
>  next decision is.**"

## Task

Contextual bandit, 4 contexts x 6 actions, sparse reward, 4,000 trials, 30 seeds, with an **UNSIGNALLED goal
switch** at trial 2,000: the correct action in every context changes with no cue. Action selection uses the
same per-component collapse as everywhere else — one coin per candidate component, probability from
accumulated evidence. **No epsilon, no temperature, no schedule.**

## Result (baselines swept; each family's best reported)

| agent | pre-switch | post-switch | never recovered | explore-rate pre | explore-rate post |
|---|---|---|---|---|---|
| **coherence-gated (no exploration parameter)** | **1.000** | **0.748** | **0%** | **0.02** | **0.39** |
| softmax, T=0.5 (best of sweep) | 0.922 | 0.707 | 0% | — | — |
| eps-greedy fixed, best (0.05) | 0.958 | 0.316 | 85% | 0.05 | 0.05 |
| eps-greedy decaying, best | 0.535 | 0.140 | 100% | — | — |

**The core claim is demonstrated: the exploration rate rose from 0.02 to 0.39 by itself when the world
changed.** Nothing was re-tuned; the old answer stopped paying, its component weakened, and variability
returned as a consequence of the structure.

## The overclaim I had to correct

A first pass used softmax T=0.2 and it failed 53% of the time, which made this look like a capability only the
primitive had. **Sweeping the baselines properly shows softmax at T=0.5 also recovers in 100% of seeds
(post 0.707).** So "only this architecture can recover" is FALSE and is not claimed.

## What IS defensible

Softmax buys its recovery with a **permanent exploitation tax**: a fixed temperature keeps it noisy forever,
so it never exceeds 0.922 even when fully consolidated and the environment is stable. The primitive pays no
such tax — **1.000 before the switch AND 0.748 after**, because its exploration is ADAPTIVE rather than
constant: near-zero when structure is consolidated, high when it is disrupted.

So the honest claim is not "we can do something nothing else can." It is: **the exploration/exploitation
balance is set by the state of what has been learned, rather than by a hyperparameter someone has to choose
in advance** — and that yields strictly better behaviour in both regimes than a tuned constant-temperature
policy.

## Robustness

Our own `gain` is not knife-edge: 0% recovery failure across gain 1.2–5.0 (post-switch 0.763 -> 0.614),
failing only at 0.8. A broad plateau.

## Still untested

The fourth primitive — **learned state modifying the input distribution** ("stronger synapses pull the agent's
trajectory toward those features on future trials; spine enlargement IS the policy update"). This benchmark
has a fixed context distribution, so the agent's learning does not change what it subsequently encounters.
That closed loop remains unbuilt.
