"""
BENCHMARK 2 — credit assignment where the answer is a STRUCTURE, not a set of individual features.

WHY THIS TASK. Benchmark 1 used independent features and asked "which single unit earns reward". That task
has no correlation structure, so a graph can represent nothing a scalar cannot — which is exactly why the
scalar baseline tied. This task is built so that the per-feature marginals are EXACTLY UNINFORMATIVE and all
the information lives in which features CO-OCCUR.

THE TASK (a delayed-reward XOR over a hidden pair, embedded in distractors):
  A hidden pair (a, b) is drawn per run. Each trial presents one of four cases with equal probability:
      a alone  -> reward +1        both a and b -> reward -1
      b alone  -> reward +1        neither      -> reward -1
  so P(a present | +) = P(a present | -) = 0.5, and likewise for b. Every individual feature carries ZERO
  marginal information; only the CONJUNCTION is informative. Distractors fire at other times in the trial.
  Reward arrives at the end of the trial, as one global scalar.

WHY IT DISCRIMINATES. A per-feature eligibility learner (w += lr*r*e) is provably at chance: feature a
receives +1 as often as -1. The graph learner sees different CONNECTED COMPONENTS in the two cases -- {a}
versus {a,b} -- because co-active nodes bind. Credit attaches to the component, so the structure is
learnable without gradients, without a hidden layer, and from a single global scalar.

This is the canonical linearly-inseparable problem. A 2-layer network with backprop also solves it; the claim
here is narrower and about MECHANISM: solved online, from one delayed scalar, with no gradient and no stored
computation graph. A null (the graph learner also fails) is a real result and is reported as such.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from graph_cgl import GraphCoherenceGatedLearner, GraphCGLParams
from cgl import CoherenceGatedLearner, CGLParams

N_UNITS, N_DISTRACT = 20, 3
# GAP must keep the TOTAL trial shorter than the pairwise readable horizon
# tau*ln((1-floor)/(sqrt(thresh)-floor)) = 216*ln(0.75/0.457) = 107 time units.
# The first version used GAP=60 -> total 240, so the target pair had decayed below threshold BEFORE the
# reward arrived and nothing could ever be credited. That was a mis-specified benchmark, not a result.
GAP = 20.0
CASES = [("a", +1.0), ("b", +1.0), ("ab", -1.0), ("none", -1.0)]


class ScalarBaseline:
    """Per-feature eligibility-trace learner: w += lr * r * e (the three-factor / TD(lambda) rule)."""
    def __init__(self, n, lr=0.1, tau=216.0, seed=0):
        self.n, self.lr, self.tau = n, lr, tau
        self.w = np.zeros(n); self.e = np.zeros(n)
    def activate(self, active): self.e[np.asarray(active, dtype=bool)] = 1.0
    def decay(self, dt=1.0): self.e *= np.exp(-dt / self.tau)
    def reward(self, r):
        if r != 0.0: self.w += self.lr * r * self.e
    def clear_episode(self): self.e[:] = 0.0
    def predict(self, present): return float(np.sign(self.w[list(present)].sum())) if present else -1.0


def trial(learner, a, b, case, rng, is_graph):
    """One trial: target features co-activate (binding), distractors fire later (no binding), then reward."""
    present = {"a": [a], "b": [b], "ab": [a, b], "none": []}[case]
    act = np.zeros(N_UNITS, dtype=bool)
    for i in present:
        act[i] = True
    if present:
        learner.activate(act)          # co-activation -> these bind into one component
    learner.decay(GAP)                 # gap: distractors below will NOT bind to the target set
    for _ in range(N_DISTRACT):        # distractors, each isolated in time
        d = np.zeros(N_UNITS, dtype=bool)
        d[rng.choice([k for k in range(N_UNITS) if k not in (a, b)])] = True
        learner.activate(d)
        learner.decay(GAP)
    return present


def evaluate(learner, a, b, rng, is_graph, n_test=200):
    """Accuracy at predicting the reward sign from the presented structure."""
    ok = 0
    for _ in range(n_test):
        case, r = CASES[rng.integers(4)]
        present = {"a": [a], "b": [b], "ab": [a, b], "none": []}[case]
        if is_graph:
            key = frozenset(present)
            pred = np.sign(learner.group_w.get(key, 0.0)) if present else -1.0
            if pred == 0.0:
                pred = -1.0
        else:
            pred = learner.predict(present)
        ok += int(pred == r)
    return ok / n_test


def run(make, is_graph, n_seeds=30, n_train=600):
    accs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(500 + seed)
        a, b = rng.choice(N_UNITS, size=2, replace=False)
        L = make(seed)
        for _ in range(n_train):
            case, r = CASES[rng.integers(4)]
            trial(L, int(a), int(b), case, rng, is_graph)
            L.reward(r)
            L.clear_episode()
        accs.append(evaluate(L, int(a), int(b), rng, is_graph))
    return np.array(accs)


if __name__ == "__main__":
    print("=" * 94)
    print("BENCHMARK 2 — STRUCTURED CREDIT: delayed-reward XOR over a hidden pair, with distractors")
    print("per-feature marginals are exactly uninformative; all information is in CO-OCCURRENCE")
    print("=" * 94)

    g = run(lambda s: GraphCoherenceGatedLearner(
        N_UNITS, GraphCGLParams(tau=216.0, edge_threshold=0.5, commit_gain=3.0, lr=1.0, seed=s)),
        is_graph=True)

    best = None
    for lr in (0.01, 0.05, 0.1, 0.5):
        acc = run(lambda s, lr=lr: ScalarBaseline(N_UNITS, lr=lr, seed=s), is_graph=False)
        if best is None or acc.mean() > best[0].mean():
            best = (acc, lr)
    b_acc, b_lr = best

    print(f"\n  {'graph (per-component collapse)':>34}: accuracy {g.mean()*100:5.1f}%  "
          f"(sd {g.std()*100:.1f}, n={len(g)} runs)")
    print(f"  {'scalar trace baseline (best lr=%.2f)'%b_lr:>34}: accuracy {b_acc.mean()*100:5.1f}%  "
          f"(sd {b_acc.std()*100:.1f}, n={len(b_acc)} runs)")
    print(f"  {'chance':>34}: 50.0%")

    from itertools import chain
    def perm(x, y, T=20000, seed=1):
        obs = abs(x.mean()-y.mean()); pool = np.concatenate([x, y]); nx = len(x)
        rng = np.random.default_rng(seed); ge = 0
        for _ in range(T):
            rng.shuffle(pool)
            if abs(pool[:nx].mean()-pool[nx:].mean()) >= obs-1e-12: ge += 1
        return (ge+1)/(T+1)
    print(f"\n  permutation p (graph vs baseline) = {perm(g, b_acc):.4f}")
    verdict = (g.mean() > b_acc.mean() + 0.1) and perm(g, b_acc) < 0.05
    print(f"\n  => {'STRUCTURE IS LEARNABLE BY THE GRAPH AND NOT BY THE SCALAR TRACE' if verdict else 'no separation demonstrated — report it, do not tune'}")
    print("=" * 94)
