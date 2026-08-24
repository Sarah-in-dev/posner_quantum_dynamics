"""
BENCHMARK 4 — GENERALIZATION. Designed to expose the graph primitive's weakness, not its strength.

Benchmark 3 showed the graph holds 100% across 64 overlapping conjunctive rules while a per-feature learner
falls to 68%. But those rules were ARBITRARY, so memorising a lookup IS the correct solution and the graph's
component store is exactly that -- a table that grew linearly with the number of rules.

The unasked question: can it handle a combination it has never seen? A pure lookup cannot, by construction.

TASK (deliberately favourable to the per-feature learner). A hidden subset S of the feature pool is drawn.
For a presented PAIR: reward +1 iff BOTH members are in S, else -1. This rule is LINEARLY SEPARABLE at the
feature level -- give members of S a positive weight and the sum discriminates -- so a per-feature learner
should generalise to held-out pairs, while a component lookup should score chance on pairs it never saw.

We train on a random 60% of pairs and test ONLY on the held-out 40%. If the graph collapses to chance here,
that is the honest boundary of the primitive and it gets recorded as such.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from graph_cgl import GraphCoherenceGatedLearner, GraphCGLParams

N_FEATURES, POOL, N_DISTRACT, GAP = 40, 10, 3, 20.0


class ScalarBaseline:
    def __init__(self, n, lr=0.05, tau=216.0):
        self.n, self.lr, self.tau = n, lr, tau
        self.w = np.zeros(n); self.e = np.zeros(n)
    def activate(self, a): self.e[np.asarray(a, dtype=bool)] = 1.0
    def decay(self, dt=1.0): self.e *= np.exp(-dt / self.tau)
    def reward(self, r):
        if r != 0.0: self.w += self.lr * r * self.e
    def clear_episode(self): self.e[:] = 0.0
    def predict(self, pair):
        s = self.w[list(pair)].sum()
        return 1.0 if s > 0 else -1.0


def trial(L, pair, rng):
    a = np.zeros(N_FEATURES, dtype=bool); a[list(pair)] = True
    L.activate(a); L.decay(GAP)
    for _ in range(N_DISTRACT):
        d = np.zeros(N_FEATURES, dtype=bool)
        d[rng.integers(POOL, N_FEATURES)] = True
        L.activate(d); L.decay(GAP)


def run(is_graph, n_seeds=15, epochs=60, train_frac=0.6):
    tr_acc, te_acc = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(9000 + seed)
        S = set(rng.choice(POOL, size=POOL // 2, replace=False).tolist())
        pairs = [(i, j) for i in range(POOL) for j in range(i + 1, POOL)]
        rng.shuffle(pairs)
        n_tr = int(len(pairs) * train_frac)
        train, test = pairs[:n_tr], pairs[n_tr:]
        sign = lambda p: 1.0 if (p[0] in S and p[1] in S) else -1.0

        L = (GraphCoherenceGatedLearner(N_FEATURES, GraphCGLParams(seed=seed))
             if is_graph else ScalarBaseline(N_FEATURES))
        for _ in range(epochs):
            for p in np.random.permutation(len(train)):
                pr = train[p]
                trial(L, pr, rng); L.reward(sign(pr)); L.clear_episode()

        def pred(pr):
            if is_graph:
                v = L.predict_group(pr)
                return float(np.sign(v)) if v != 0 else -1.0
            return L.predict(pr)
        def balanced(ps):
            """Mean of per-class recall. Raw accuracy is misleading here: only ~22% of pairs are +1, so
            'always predict -1' already scores ~78%. Balanced accuracy has chance at 50% for any constant
            predictor, which is what we need to see whether anything is actually generalising."""
            pos = [p for p in ps if sign(p) > 0]; neg = [p for p in ps if sign(p) < 0]
            rp = np.mean([pred(p) > 0 for p in pos]) if pos else np.nan
            rn = np.mean([pred(p) < 0 for p in neg]) if neg else np.nan
            return np.nanmean([rp, rn])
        tr_acc.append(balanced(train))
        te_acc.append(balanced(test))
    return np.array(tr_acc), np.array(te_acc)


if __name__ == "__main__":
    print("=" * 92)
    print("BENCHMARK 4 — GENERALIZATION to UNSEEN combinations (rule: +1 iff both features in hidden set S)")
    print("this rule is linearly separable per-feature, so it FAVOURS the scalar baseline by design")
    print("=" * 92)
    gtr, gte = run(True)
    btr, bte = run(False)
    print("  BALANCED accuracy (mean of per-class recall; any constant predictor scores 50%)")
    print(f"  {'':>34} {'seen pairs':>12} {'HELD-OUT pairs':>16}")
    print("-" * 92)
    print(f"  {'graph (component lookup)':>34} {gtr.mean()*100:>11.1f}% {gte.mean()*100:>15.1f}%")
    print(f"  {'scalar trace baseline':>34} {btr.mean()*100:>11.1f}% {bte.mean()*100:>15.1f}%")
    print(f"  {'chance':>34} {50.0:>11.1f}% {50.0:>15.1f}%")
    print("-" * 92)
    if gte.mean() < 0.60:
        print("\n  => BOUNDARY FOUND: on balanced accuracy the graph does NOT generalise to unseen")
        print("     combinations -- it memorises what it has observed. Recorded as the honest limit.")
    else:
        print("\n  => the graph also generalises here — report the numbers, do not over-read either way.")
