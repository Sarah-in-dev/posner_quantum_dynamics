"""
BENCHMARK 3 — SCALING THE STRUCTURE SPACE. Built to BREAK the graph primitive, not to flatter it.

Benchmark 2 showed the graph solves a single hidden conjunction that a scalar trace provably cannot (100% vs
chance). But its readout is effectively a LOOKUP over discovered components, and the space of possible
components is combinatorial. So the honest next question is capacity and interference:

  - Does accuracy hold as the number of hidden rules K grows?
  - Does it survive OVERLAPPING rules, where one feature appears in both a rewarded and a punished rule
    (A+B -> +1 while B+C -> -1)? Overlap is what makes per-feature credit provably useless AND what should
    make a naive lookup confuse itself.
  - How many distinct components does it end up storing? (capacity blow-up indicator)

TASK. 40 features. K hidden rules, each a PAIR drawn from a small shared pool so overlap is forced; half the
rules are rewarded (+1), half punished (-1). Each trial presents one rule's pair co-actively, plus distractors
at separated times, then delivers one global scalar reward at the end. Accuracy is measured on predicting the
sign of a presented pair.

A per-feature learner is at chance BY CONSTRUCTION whenever a feature appears in both a + and a - rule.
A degradation of the graph with K is a REAL RESULT and is reported as such -- this benchmark exists to find
the ceiling, not to avoid it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from graph_cgl import GraphCoherenceGatedLearner, GraphCGLParams

N_FEATURES, POOL, N_DISTRACT, GAP = 40, 12, 3, 20.0


class ScalarBaseline:
    def __init__(self, n, lr=0.05, tau=216.0):
        self.n, self.lr, self.tau = n, lr, tau
        self.w = np.zeros(n); self.e = np.zeros(n)
    def activate(self, a): self.e[np.asarray(a, dtype=bool)] = 1.0
    def decay(self, dt=1.0): self.e *= np.exp(-dt / self.tau)
    def reward(self, r):
        if r != 0.0: self.w += self.lr * r * self.e
    def clear_episode(self): self.e[:] = 0.0
    def predict(self, pair): return float(np.sign(self.w[list(pair)].sum())) or -1.0


def make_rules(K, rng):
    """K distinct pairs drawn from a SMALL pool -> features are shared across rules by construction."""
    seen, rules = set(), []
    while len(rules) < K:
        p = tuple(sorted(rng.choice(POOL, size=2, replace=False)))
        if p not in seen:
            seen.add(p); rules.append(p)
    signs = np.array([1.0] * (K // 2) + [-1.0] * (K - K // 2))
    rng.shuffle(signs)
    return rules, signs


def trial(L, pair, rng, is_graph):
    a = np.zeros(N_FEATURES, dtype=bool); a[list(pair)] = True
    L.activate(a); L.decay(GAP)
    for _ in range(N_DISTRACT):
        d = np.zeros(N_FEATURES, dtype=bool)
        d[rng.integers(POOL, N_FEATURES)] = True     # distractors drawn OUTSIDE the rule pool
        L.activate(d); L.decay(GAP)


def run(K, is_graph, n_seeds=15, n_train_per_rule=80):
    accs, caps = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(7000 + seed)
        rules, signs = make_rules(K, rng)
        L = (GraphCoherenceGatedLearner(N_FEATURES, GraphCGLParams(seed=seed))
             if is_graph else ScalarBaseline(N_FEATURES))
        order = np.arange(K)
        for _ in range(n_train_per_rule):
            rng.shuffle(order)
            for i in order:
                trial(L, rules[i], rng, is_graph)
                L.reward(float(signs[i]))
                L.clear_episode()
        ok = 0
        for i, pair in enumerate(rules):
            if is_graph:
                v = L.group_w.get(frozenset(pair), 0.0)
                pred = float(np.sign(v)) if v != 0 else -1.0
            else:
                pred = L.predict(pair)
            ok += int(pred == signs[i])
        accs.append(ok / K)
        caps.append(len(L.group_w) if is_graph else 0)
    return np.array(accs), np.array(caps)


if __name__ == "__main__":
    print("=" * 96)
    print("BENCHMARK 3 — SCALING: K overlapping conjunctive rules drawn from a %d-feature pool" % POOL)
    print("(features are shared across rules by construction, so per-feature credit is useless)")
    print("=" * 96)
    print(f"  {'K rules':>8} {'graph acc':>11} {'baseline acc':>13} {'chance':>8} {'graph components stored':>25}")
    print("-" * 96)
    for K in (2, 4, 8, 16, 32, 64):
        g, cap = run(K, True)
        b, _ = run(K, False)
        print(f"  {K:>8} {g.mean()*100:>10.1f}% {b.mean()*100:>12.1f}% {50.0:>7.1f}% "
              f"{cap.mean():>24.0f}")
    print("-" * 96)
    print("  accuracy = predicting the reward sign of each learned rule; chance = 50%")
