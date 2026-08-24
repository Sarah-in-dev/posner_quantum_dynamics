"""
BENCHMARK 8 — THE FOURTH PRIMITIVE: learned state modifies the INPUT DISTRIBUTION.

The last untested claim:
    "In our spatial experiment: stronger synapses pull the agent's trajectory toward those features on future
     trials. No separate policy network. Spine enlargement IS the policy update. The representation and the
     policy are the same physical structure."

Every previous benchmark, including the closed-loop bandit, drew the agent's inputs from a FIXED distribution.
Here the agent CHOOSES WHERE TO LOOK, so what it learns determines what data it subsequently receives. That
coupling is the primitive: there is no policy network, and no separate exploration policy -- the same stored
structure that represents what has been learned is what steers the next sample.

TASK. 12 patches, each carrying 3 binary features drawn from a shared vocabulary. A hidden rule makes patches
containing a particular FEATURE CONJUNCTION rewarding. Each trial the agent picks ONE patch to visit, observes
its features, and receives a sparse reward. Because it picks, it selects its own training data. The rule
SWITCHES, unsignalled, half way through.

WHAT THIS CAN EXPOSE (and is meant to). Self-reinforcing attention is double-edged: seeking what you already
value gives efficient exploitation but risks premature LOCK-IN -- the agent stops sampling elsewhere and never
discovers it is wrong. We therefore measure COVERAGE (distinct patches sampled) alongside reward, so lock-in
shows up rather than hiding behind a good pre-switch score. A result where the primitive locks in and cannot
recover is a real finding and is reported as one.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

N_PATCHES, N_FEATURES, FEATS_PER_PATCH = 12, 10, 3
TRIALS, SWITCH = 4000, 2000


def make_world(rng):
    patches = [rng.choice(N_FEATURES, size=FEATS_PER_PATCH, replace=False) for _ in range(N_PATCHES)]
    return patches


def rule_reward(feats, rule):
    return 1.0 if set(rule).issubset(set(feats)) else -1.0


class CGLForager:
    """The SAME stored structure both represents what is known and steers where to look next."""
    def __init__(self, seed=0, k=2.0, gain=1.6):
        self.rng = np.random.default_rng(seed); self.k, self.gain = k, gain
        self.store = {}                                  # frozenset(features of a patch) -> [sum, count]

    def _val(self, feats):
        s = self.store.get(frozenset(feats))
        if s is not None:
            return s[0] / (s[1] + self.k)
        # partial reactivation: unfamiliar patches answered by the overlapping ones already known
        num = den = 0.0
        key = frozenset(feats)
        for kk, s in self.store.items():
            inter = len(key & kk)
            if not inter:
                continue
            w = (inter / len(key | kk)) ** 4
            num += w * (s[0] / (s[1] + self.k)); den += w
        return num / den if den else 0.0

    def choose(self, patches):
        vals = np.array([self._val(p) for p in patches])
        p = np.clip(self.gain * vals, 0.0, 1.0)          # one coin per patch-component
        fired = np.flatnonzero(self.rng.random(len(patches)) < p)
        if len(fired) == 1:
            return int(fired[0])
        if len(fired) > 1:
            return int(self.rng.choice(fired))
        return int(self.rng.integers(len(patches)))
    def learn(self, feats, r):
        s = self.store.setdefault(frozenset(feats), [0.0, 0]); s[0] += r; s[1] += 1


class EpsForager:
    def __init__(self, seed=0, eps=0.1, decay=None):
        self.rng = np.random.default_rng(seed); self.q = {}; self.n = {}
        self.eps0, self.decay, self.t = eps, decay, 0
    def _eps(self):
        return self.eps0 if self.decay is None else max(0.01, self.eps0 * np.exp(-self.t / self.decay))
    def choose(self, patches):
        self.t += 1
        if self.rng.random() < self._eps():
            return int(self.rng.integers(len(patches)))
        vals = [self.q.get(frozenset(p), 0.0) for p in patches]
        return int(np.argmax(vals))
    def learn(self, feats, r):
        k = frozenset(feats); self.n[k] = self.n.get(k, 0) + 1
        self.q[k] = self.q.get(k, 0.0) + (r - self.q.get(k, 0.0)) / self.n[k]


class SoftmaxForager:
    def __init__(self, seed=0, temp=0.5):
        self.rng = np.random.default_rng(seed); self.q = {}; self.n = {}; self.T = temp
    def choose(self, patches):
        v = np.array([self.q.get(frozenset(p), 0.0) for p in patches]) / self.T
        v -= v.max(); p = np.exp(v); p /= p.sum()
        return int(self.rng.choice(len(patches), p=p))
    def learn(self, feats, r):
        k = frozenset(feats); self.n[k] = self.n.get(k, 0) + 1
        self.q[k] = self.q.get(k, 0.0) + (r - self.q.get(k, 0.0)) / self.n[k]


def episode(agent, seed):
    rng = np.random.default_rng(20_000 + seed)
    patches = make_world(rng)
    rule1 = rng.choice(N_FEATURES, size=2, replace=False)
    good = [i for i, p in enumerate(patches) if set(rule1).issubset(set(p))]
    while not good:                                       # ensure the rule is satisfiable
        rule1 = rng.choice(N_FEATURES, size=2, replace=False)
        good = [i for i, p in enumerate(patches) if set(rule1).issubset(set(p))]
    for _ in range(200):
        rule2 = rng.choice(N_FEATURES, size=2, replace=False)
        g2 = [i for i, p in enumerate(patches) if set(rule2).issubset(set(p))]
        if g2 and set(g2) != set(good):
            break
    R, visits = np.zeros(TRIALS), np.zeros((TRIALS, N_PATCHES))
    for t in range(TRIALS):
        rule = rule1 if t < SWITCH else rule2
        i = agent.choose(patches)
        r = rule_reward(patches[i], rule)
        agent.learn(patches[i], r)
        R[t] = (r > 0); visits[t, i] = 1
    return R, visits


def run(make, label, n=30):
    R, V = [], []
    for s in range(n):
        r, v = episode(make(s), s); R.append(r); V.append(v)
    R, V = np.array(R), np.array(V)
    pre, post = R[:, SWITCH-500:SWITCH].mean(), R[:, -500:].mean()
    cov_pre = np.array([(v[:SWITCH].sum(0) > 0).sum() for v in V]).mean()
    cov_post = np.array([(v[SWITCH:SWITCH+500].sum(0) > 0).sum() for v in V]).mean()
    rec = []
    for row in R:
        roll = np.convolve(row[SWITCH:], np.ones(200)/200, mode="valid")
        idx = np.flatnonzero(roll > 0.6); rec.append(idx[0] if len(idx) else np.nan)
    failed = 100 * np.isnan(np.array(rec, dtype=float)).mean()
    print(f"  {label:>32} {pre:>8.3f} {post:>9.3f} {failed:>8.0f}% {cov_pre:>10.1f} {cov_post:>11.1f}")


if __name__ == "__main__":
    print("=" * 100)
    print("BENCHMARK 8 — FORAGING: the agent CHOOSES its own inputs (learned state steers sampling)")
    print(f"{N_PATCHES} patches | hidden feature-conjunction rule | unsignalled rule switch at {SWITCH}")
    print("=" * 100)
    print(f"  {'agent':>32} {'pre':>8} {'post':>9} {'failed':>9} {'patches pre':>10} {'patches post':>11}")
    print("  " + "-" * 88)
    run(lambda s: CGLForager(seed=s), "coherence-gated (no expl. param)")
    for e in (0.05, 0.15):
        run(lambda s, e=e: EpsForager(seed=s, eps=e), f"eps-greedy fixed {e}")
    run(lambda s: EpsForager(seed=s, eps=1.0, decay=600), "eps-greedy decaying")
    for T in (0.25, 0.5):
        run(lambda s, T=T: SoftmaxForager(seed=s, temp=T), f"softmax T={T}")
    print("  " + "-" * 88)
    print("  'patches' = distinct patches sampled (low pre = focused; low post = LOCKED IN)")
