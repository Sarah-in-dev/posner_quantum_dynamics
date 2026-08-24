# Structured credit assignment — the graph primitive separates from a scalar trace (2026-08-24)

## Result

| | accuracy | note |
|---|---|---|
| **graph (per-component joint collapse)** | **100.0%** (sd 0.0, n=30) | |
| scalar eligibility-trace baseline (best of swept lr) | 63.8% (sd 12.3) | = chance on every informative case |
| chance | 50.0% | |

**Permutation p = 0.0000.**

The baseline's 63.8% is not partial learning: the task has four equiprobable cases, and "neither present"
is answered correctly for free (25%), leaving coin-flip on the rest — 25 + 0.75x50 = 62.5%, which is what it
scores. **The scalar learner is at chance on every case that carries information.**

## The task, and why it discriminates

Delayed-reward XOR over a hidden pair (a, b), embedded in distractors:
`a alone -> +1`, `b alone -> +1`, `both -> -1`, `neither -> -1`.
So `P(a present | +) = P(a present | -) = 0.5`: **every individual feature carries exactly zero marginal
information**, and all of it lives in which features CO-OCCUR. A per-feature learner (`w += lr*r*e`) is
provably at chance — feature `a` receives +1 exactly as often as -1. The graph learner sees different
CONNECTED COMPONENTS in the two cases (`{a}` vs `{a,b}`) because co-active nodes bind, and credit attaches to
the component, so the conjunction is learnable.

Solved **online, from a single global scalar arriving after the activity, with no gradient, no hidden layer,
and no stored computation graph.** A 2-layer network with backprop also solves XOR; the claim here is about
MECHANISM, not about XOR being hard.

## Two defects found by diagnosis (both mine, both real)

The first two runs of this benchmark returned **49.7% — chance**. Instrumenting an actual training run, rather
than assuming, found two faults in the abstraction:

1. **Singletons never formed components.** A lone active node has no edges, so it was skipped entirely and
   could never be credited. Wrong against Model 6, where a single synapse's dimer cloud is itself one
   connected component ("one synapse = one nanodomain = one component"). Fixed: an edgeless node whose own
   trace exceeds sqrt(edge_threshold) — the same Werner floor the pairwise edges use — forms a singleton.
2. **BINDING AND MEMORY WERE THE SAME TIMESCALE.** Any node with a trace above the floor bound to any other,
   so with tau=216 everything stayed "live" all trial, every trial produced ONE junk component containing the
   targets plus all distractors, and no structure could accumulate. **Model 6 separates two timescales that I
   had collapsed:** bond FORMATION is fast and local (dimers bond within a single calcium event), while bond
   PERSISTENCE is long (~100 s coherence). Fixed by using the `bind_window` parameter that the first version
   declared and then ignored. This is the substantive correction — the primitive does not work without it.

## Honest limits

- **Small structure space.** Four distinct cases; the readout is effectively a lookup over discovered
  components. That is powerful here but combinatorial in general, and scaling to many overlapping structures
  is UNTESTED.
- **Distractors were temporally separated by construction.** Real inputs will not be so cleanly segmented; how
  the binding window behaves under realistic temporal overlap is untested.
- **One task.** This shows a separation exists in the regime the architecture was designed for. It does not
  establish a general advantage, and gradient methods remain better on dense-reward supervised problems.
- The earlier scalar extraction (`cgl.py`) tying with the baseline is now explained: it was not the primitive.
  It had no graph and used independent per-unit draws, i.e. it WAS TD(lambda). That benchmark says nothing
  about this architecture.

## What this licenses

That the graph-with-component-collapse is doing something a scalar eligibility trace provably cannot — which
is the first evidence for a distinctive computational contribution, as opposed to "a long trace", which the
substrate test already showed is classically available and unremarkable.
