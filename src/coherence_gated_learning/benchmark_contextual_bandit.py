#!/usr/bin/env python3
"""
BENCHMARK 10 — ESTABLISHED CONTEXTUAL BANDIT (UCI Mushroom), where the conjunctive machinery is NOT inert.

WHY THIS ONE, AFTER BENCHMARK 9. The context-free bandit could not test what we built: one unit per arm
means every component is a singleton, so `value()` degenerates to a leaky average and `argmax` to plain
greedy -- verified there by a control that scored an IDENTICAL regret with no CGL at all. A contextual
bandit is the smallest established setting where components are genuine feature CONJUNCTIONS.

THE TASK (Blundell et al. 2015; Riquelme et al. 2018 "Deep Bayesian Bandits Showdown" -- the canonical
mushroom bandit). Each round one of 8124 UCI mushrooms is drawn. The agent eats or abstains:
    abstain            ->  0
    eat edible         -> +5
    eat poisonous      -> +5 with prob 0.5, -35 with prob 0.5   (expected -15)
The per-round oracle eats edible mushrooms and abstains on poisonous ones. Regret is against that oracle.

TWO REPRESENTATIONS, BOTH REPORTED. Measured properties of this dataset drive the design, and they were
checked BEFORE building (not after seeing scores):
  - all 8124 rows are DISTINCT 22-way conjunctions -> a component built from the whole context is a unique
    key, i.e. pure memorisation with no generalisation except through Jaccard overlap;
  - edibility is LOW-ORDER: feature 4 (odor) alone gives 98.5% accuracy.
So:
  CGL-single   : one component per (full 22-feature context, action). The direct reading of the primitive.
  CGL-ensemble : M synapses, each with its OWN local view of d randomly-chosen feature slots, each storing
                 and crediting its own conjunction; the value is the mean over synapses. This is the
                 grounded architecture -- a synapse does not receive the whole input, it sees the subset
                 arriving at its own nanodomain, and credit is assigned per synapse. It yields LOW-ORDER
                 conjunctions across an ensemble, which is the grain this task actually has.

SELECTION. Benchmark 9 established that per-component collapse is a CONSOLIDATION rule (an independent local
decision about whether THIS memory commits), not an action-selection rule among close alternatives, and that
an action whose value is exactly 0 can never commit. Action selection here is therefore competitive argmax
(the striatal/basal-ganglia role), and the CGL machinery supplies the REPRESENTATION and CREDIT. Both
selection rules are reported so the difference is visible rather than assumed.

THE CONTROL THAT MATTERS. Benchmark 9's apparent win evaporated against a no-CGL control. The equivalent
control here is a LINEAR model on the same one-hot features: linear IS the no-conjunction representation. If
CGL-ensemble does not beat it, the conjunctive machinery is not doing the work and must not be credited.
"""
import os, sys, urllib.request, zipfile, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from cgl_primitive import CoherenceGatedPrimitive, CGLParams

URL = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
CACHE = os.environ.get("CGL_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))
T = 20000
R_EAT_GOOD, R_EAT_BAD, R_ABSTAIN = 5.0, -35.0, 0.0


def load():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "agaricus-lepiota.data")
    if not os.path.exists(path):
        with urllib.request.urlopen(URL, timeout=60) as f:
            z = zipfile.ZipFile(io.BytesIO(f.read()))
        open(path, "wb").write(z.read("agaricus-lepiota.data"))
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    y = np.array([r[0] == "e" for r in rows])
    vocab, X = {}, []
    for r in rows:
        X.append([vocab.setdefault((i, v), len(vocab)) for i, v in enumerate(r[1:])])
    return np.array(X), y, len(vocab)


def onehot(X, V):
    H = np.zeros((len(X), V))
    H[np.arange(len(X))[:, None], X] = 1.0
    return H


# ------------------------------------------------------------------ baselines
class Uniform:
    def __init__(self, V, rng, **kw): self.rng = rng
    def act(self, x, h): return int(self.rng.integers(2))
    def learn(self, x, h, a, r): pass


class EpsGreedyLinear:
    """Linear value per action on the one-hot context. THE CONTROL: linear = no conjunctions."""
    def __init__(self, V, rng, eps=0.05, lr=0.01, **kw):
        self.w = np.zeros((2, V)); self.rng, self.eps, self.lr = rng, eps, lr
    def act(self, x, h):
        if self.rng.random() < self.eps: return int(self.rng.integers(2))
        return int(np.argmax(self.w @ h))
    def learn(self, x, h, a, r): self.w[a] += self.lr * (r - self.w[a] @ h) * h


class LinUCB:
    """Li et al. 2010, disjoint model, rank-1 Sherman-Morrison updates."""
    def __init__(self, V, rng, alpha=1.0, **kw):
        self.Ainv = np.array([np.eye(V) for _ in range(2)]); self.b = np.zeros((2, V)); self.alpha = alpha
    def act(self, x, h):
        s = []
        for a in range(2):
            th = self.Ainv[a] @ self.b[a]
            s.append(th @ h + self.alpha * np.sqrt(max(h @ self.Ainv[a] @ h, 0.0)))
        return int(np.argmax(s))
    def learn(self, x, h, a, r):
        Av = self.Ainv[a] @ h
        self.Ainv[a] -= np.outer(Av, Av) / (1.0 + h @ Av)
        self.b[a] += r * h


class LinTS:
    """Linear Thompson sampling (Agrawal & Goyal 2013), diagonal-free posterior via Ainv."""
    def __init__(self, V, rng, v=0.25, **kw):
        self.Ainv = np.array([np.eye(V) for _ in range(2)]); self.b = np.zeros((2, V))
        self.rng, self.v = rng, v
    def act(self, x, h):
        s = []
        for a in range(2):
            mu = self.Ainv[a] @ self.b[a]
            s.append((mu + self.v * (self.Ainv[a] @ self.rng.standard_normal(len(mu)))) @ h)
        return int(np.argmax(s))
    def learn(self, x, h, a, r):
        Av = self.Ainv[a] @ h
        self.Ainv[a] -= np.outer(Av, Av) / (1.0 + h @ Av)
        self.b[a] += r * h


# ------------------------------------------------------------------ the primitive
# NOVELTY BONUS (added after the first run; see RESULT_contextual_bandit.md). The first run gave the CGL
# argmax arms NO exploration device while every baseline had one (eps / UCB bonus / posterior sampling), and
# they deadlocked into always abstaining: both action values start at 0, argmax picks abstain, abstain pays
# exactly 0 forever, so eat is never tried. OPT is applied ONLY where the learner has no evidence at all --
# no stored component and no overlapping one. Grounded in the dopamine novelty response (novel stimuli evoke
# bursts; Kakade & Dayan 2002 novelty bonuses), and it is also the textbook optimistic-initialisation device
# that makes a greedy selector explore -- stated plainly rather than claimed as ours.
OPT = 1.0        # scaled units; +1.0 == the best attainable reward (+5)


class CGLSingle:
    """One component per (full 22-feature context, action) -- the direct reading."""
    def __init__(self, V, rng, seed=0, rate=0.1, coin=False, **kw):
        self.c = CoherenceGatedPrimitive(V + 2, CGLParams(seed=seed, learn_rate=rate))
        self.V, self.coin = V, coin
    def _k(self, x, a): return list(x) + [self.V + a]
    def _v(self, x, a):
        v, ev = self.c.value_evidence(self._k(x, a))
        return v if ev else OPT
    def act(self, x, h):
        if self.coin: return self.c.choose([self._k(x, 0), self._k(x, 1)])
        return int(np.argmax([self._v(x, a) for a in range(2)]))
    def learn(self, x, h, a, r): self.c.learn(self._k(x, a), r / 5.0)


class CGLEnsemble:
    """M synapses, each seeing its OWN d randomly-chosen feature slots (its nanodomain's local view)."""
    def __init__(self, V, rng, seed=0, M=32, d=3, rate=0.1, coin=False, **kw):
        r = np.random.default_rng(seed)
        self.slots = [r.choice(22, size=d, replace=False) for _ in range(M)]
        self.syn = [CoherenceGatedPrimitive(V + 2, CGLParams(seed=seed * 1000 + i, learn_rate=rate))
                    for i in range(M)]
        self.V, self.coin, self.rng = V, coin, np.random.default_rng(seed)
    def _k(self, x, a, i): return [int(x[j]) for j in self.slots[i]] + [self.V + a]
    def _v(self, x, a):
        vs = []
        for i, s in enumerate(self.syn):
            v, ev = s.value_evidence(self._k(x, a, i))
            vs.append(v if ev else OPT)
        return float(np.mean(vs))
    def act(self, x, h):
        v = np.array([self._v(x, 0), self._v(x, 1)])
        if self.coin:
            p = np.clip(1.6 * v, 0, 1); f = np.flatnonzero(self.rng.random(2) < p)
            if len(f) == 1: return int(f[0])
            if len(f) > 1: return int(self.rng.choice(f))
            return int(self.rng.integers(2))
        return int(np.argmax(v))
    def learn(self, x, h, a, r):
        for i, s in enumerate(self.syn): s.learn(self._k(x, a, i), r / 5.0)


# ------------------------------------------------------------------ harness
def run(make, X, y, H, V, seed):
    rng = np.random.default_rng(seed)
    agent = make(V, rng)
    idx = rng.integers(len(X), size=T)
    coin = rng.random(T) < 0.5
    regret = 0.0
    for t in range(T):
        i = idx[t]; x, h = X[i], H[i]
        a = agent.act(x, h)                                  # 1 = eat, 0 = abstain
        if a == 1:
            r = R_EAT_GOOD if y[i] else (R_EAT_BAD if coin[t] else R_EAT_GOOD)
        else:
            r = R_ABSTAIN
        agent.learn(x, h, a, r)
        best = R_EAT_GOOD if y[i] else R_ABSTAIN             # oracle: eat edible, abstain on poisonous
        got = (R_EAT_GOOD if y[i] else -15.0) if a == 1 else R_ABSTAIN   # EXPECTED value of the action taken
        regret += best - got
    return regret


class Ablation:
    """ATTRIBUTION CONTROL (the test that decides whether CGL may be credited at all).

    Same random-subspace conjunctive ensemble, same optimistic init, same greedy argmax -- but switchable:
      leaky=False, jac=False  -> plain running mean: 'use feature conjunctions', with NO CGL mechanism
      leaky=True              -> the leaky / active-depression update
      jac=True                -> Jaccard partial reactivation
    Benchmark 9's apparent win vanished against exactly this kind of control; run it before claiming anything.
    """
    def __init__(self, V, rng, seed=0, M=32, d=3, leaky=False, jac=False, rate=0.1):
        r = np.random.default_rng(seed)
        self.slots = [r.choice(22, size=d, replace=False) for _ in range(M)]
        self.V, self.M, self.leaky, self.jac, self.rate = V, M, leaky, jac, rate
        self.tab = [{} for _ in range(M)]; self.idx = [{} for _ in range(M)]
    def _k(self, x, a, i): return frozenset([int(x[j]) for j in self.slots[i]] + [self.V + a])
    def _one(self, i, k):
        t = self.tab[i]
        if k in t: return t[k] if self.leaky else t[k][0] / t[k][1]
        if not self.jac: return OPT
        cand = set()
        for u in k: cand |= self.idx[i].get(u, set())
        num = den = 0.0
        for kk in cand:
            w = (len(k & kk) / len(k | kk)) ** 4
            num += w * (t[kk] if self.leaky else t[kk][0] / t[kk][1]); den += w
        return num / den if den > 0 else OPT
    def act(self, x, h):
        return int(np.argmax([np.mean([self._one(i, self._k(x, a, i)) for i in range(self.M)]) for a in (0, 1)]))
    def learn(self, x, h, a, r):
        r = r / 5.0
        for i in range(self.M):
            k = self._k(x, a, i); t = self.tab[i]
            if k not in t:
                t[k] = 0.0 if self.leaky else [0.0, 0]
                for u in k: self.idx[i].setdefault(u, set()).add(k)
            if self.leaky: t[k] += self.rate * (r - t[k])
            else: t[k][0] += r; t[k][1] += 1


def main():
    X, y, V = load()
    H = onehot(X, V)
    n_seeds = 5
    algos = [
        ("uniform random",                       lambda V, rng: Uniform(V, rng)),
        ("eps-greedy LINEAR (the control)",      lambda V, rng: EpsGreedyLinear(V, rng, eps=0.05)),
        ("LinUCB (Li et al. 2010)",              lambda V, rng: LinUCB(V, rng, alpha=1.0)),
        ("LinTS (Agrawal & Goyal 2013)",         lambda V, rng: LinTS(V, rng)),
        ("CGL-single, argmax + novelty",          lambda V, rng: CGLSingle(V, rng, seed=int(rng.integers(1e6)))),
        ("CGL-single, coin select",              lambda V, rng: CGLSingle(V, rng, seed=int(rng.integers(1e6)), coin=True)),
        ("CGL-ensemble M=32 d=3, argmax+novelty",lambda V, rng: CGLEnsemble(V, rng, seed=int(rng.integers(1e6)))),
        ("CGL-ensemble M=64 d=2, argmax+novelty",lambda V, rng: CGLEnsemble(V, rng, seed=int(rng.integers(1e6)), M=64, d=2)),
        ("CGL-ensemble M=32 d=3, coin",          lambda V, rng: CGLEnsemble(V, rng, seed=int(rng.integers(1e6)), coin=True)),
    ]
    print("=" * 94)
    print(f"BENCHMARK 10 — MUSHROOM CONTEXTUAL BANDIT | T={T}, {n_seeds} seeds | regret vs per-round oracle")
    print(f"  {len(X)} mushrooms, {V} feature-values, all rows distinct 22-way conjunctions; odor alone = 98.5%")
    print("=" * 94)
    rows = []
    for name, make in algos:
        R = np.array([run(make, X, y, H, V, 100 + s) for s in range(n_seeds)])
        rows.append((name, R.mean(), R.std()))
        print(f"  {name:>34}  regret {R.mean():9.1f}  (sd {R.std():7.1f})", flush=True)
    best = min(r[1] for r in rows)
    print("-" * 94)
    print(f"  best: {[r[0] for r in rows if r[1] == best][0]}")

    print("\n" + "=" * 94)
    print("ATTRIBUTION — which mechanism earns the win over the linear control?")
    print("=" * 94)
    for lab, kw in ((" CONTROL: conjunctions only (running mean, NO CGL)", dict(leaky=False, jac=False)),
                    (" + leaky / active-depression update only",           dict(leaky=True,  jac=False)),
                    (" + Jaccard partial reactivation only",               dict(leaky=False, jac=True)),
                    (" FULL CGL (leaky + Jaccard)",                        dict(leaky=True,  jac=True))):
        R = np.array([run(lambda V, rng, kw=kw: Ablation(V, rng, seed=int(rng.integers(1e6)), **kw),
                          X, y, H, V, 100 + s) for s in range(n_seeds)])
        print(f"  {lab:>50}  regret {R.mean():9.1f}", flush=True)


if __name__ == "__main__":
    main()
