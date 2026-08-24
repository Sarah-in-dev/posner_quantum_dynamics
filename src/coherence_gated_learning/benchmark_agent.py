"""
BENCHMARK 7 — THE CLOSED LOOP. An agent that ACTS, with exploration emerging from learned structure.

WHY THIS IS DIFFERENT FROM EVERYTHING BEFORE IT. Benchmarks 1-6 were passive prediction on fixed data: show a
pattern, predict an outcome. That is exactly the setting where standard methods are strongest, and it makes
the primitive's two most distinctive claims INVISIBLE, because both require an agent that acts:

    "Fragmented graphs -> many independent stochastic samples -> diverse outcomes. Connected graphs -> one
     sample -> uniform outcome. This creates natural exploration without epsilon-greedy, softmax temperature,
     or a separate exploration policy. The structure of what's been learned determines how exploratory the
     next decision is."

Nothing in TD(lambda), k-NN or gradient boosting does that. It has never been tested here.

THE TASK. A contextual bandit with an UNSIGNALLED GOAL SWITCH. Several contexts; in each, one of N actions
pays off. Reward is sparse and delayed to the end of the trial. Part-way through, the correct action in every
context CHANGES, with no cue. This is the sharpest discriminator available:
  - epsilon-greedy with a DECAY schedule explores early then commits -- after the switch epsilon has already
    annealed, so it cannot recover without someone re-tuning it.
  - epsilon-greedy with FIXED epsilon can recover but never fully exploits, paying a permanent tax.
  - the claim under test is that the primitive needs NEITHER: when the old answer stops paying, its component
    weakens, the graph fragments, and exploration RETURNS BY ITSELF.

HOW ACTION SELECTION WORKS (faithful to the mechanism, not bolted on). Each candidate action forms a
prospective component with the current context. Each component gets ONE stochastic coin whose probability
comes from its accumulated evidence -- exactly the per-component collapse used everywhere else. If precisely
one component commits, the agent takes it (exploitation). If none or several commit, the agent picks among
the live candidates (exploration). Exploration is therefore a CONSEQUENCE of how consolidated the structure
is, with no temperature and no schedule.

A null -- the primitive failing to recover, or matching a tuned baseline -- is a real result and is reported.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

N_ACTIONS, N_CONTEXTS = 6, 4
TRIALS, SWITCH = 4000, 2000


class CGLAgent:
    """Per-component collapse used for ACTION SELECTION. No epsilon, no temperature, no schedule."""
    def __init__(self, seed=0, k=2.0, gain=1.6, lr=1.0):
        self.rng = np.random.default_rng(seed)
        self.k, self.gain, self.lr = k, gain, lr
        self.store = {}                                   # (context, action) -> [sum_reward, count]

    def _val(self, c, a):
        s = self.store.get((c, a))
        return 0.0 if s is None else s[0] / (s[1] + self.k)

    def act(self, c):
        # one coin per candidate component; probability from accumulated evidence
        vals = np.array([self._val(c, a) for a in range(N_ACTIONS)])
        p = np.clip(self.gain * vals, 0.0, 1.0)           # negative evidence -> 0 -> never committed
        fired = np.flatnonzero(self.rng.random(N_ACTIONS) < p)
        if len(fired) == 1:
            return int(fired[0]), "exploit"               # consolidated: one component collapses
        if len(fired) > 1:
            return int(self.rng.choice(fired)), "explore" # several live components: pick among them
        return int(self.rng.integers(N_ACTIONS)), "explore"  # nothing committed: fully exploratory

    def learn(self, c, a, r):
        s = self.store.setdefault((c, a), [0.0, 0])
        s[0] += self.lr * r; s[1] += 1


class EpsGreedy:
    def __init__(self, seed=0, eps0=1.0, decay=None, eps_min=0.01):
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((N_CONTEXTS, N_ACTIONS)); self.n = np.zeros((N_CONTEXTS, N_ACTIONS))
        self.eps0, self.decay, self.eps_min, self.t = eps0, decay, eps_min, 0
    def eps(self):
        if self.decay is None: return self.eps0
        return max(self.eps_min, self.eps0 * np.exp(-self.t / self.decay))
    def act(self, c):
        self.t += 1
        if self.rng.random() < self.eps():
            return int(self.rng.integers(N_ACTIONS)), "explore"
        return int(np.argmax(self.q[c])), "exploit"
    def learn(self, c, a, r):
        self.n[c, a] += 1
        self.q[c, a] += (r - self.q[c, a]) / self.n[c, a]


class Softmax:
    def __init__(self, seed=0, temp=0.2):
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((N_CONTEXTS, N_ACTIONS)); self.n = np.zeros((N_CONTEXTS, N_ACTIONS)); self.T = temp
    def act(self, c):
        z = self.q[c] / self.T; z -= z.max()
        p = np.exp(z); p /= p.sum()
        a = int(self.rng.choice(N_ACTIONS, p=p))
        return a, ("exploit" if a == int(np.argmax(self.q[c])) else "explore")
    def learn(self, c, a, r):
        self.n[c, a] += 1
        self.q[c, a] += (r - self.q[c, a]) / self.n[c, a]


def episode(agent, seed):
    rng = np.random.default_rng(10_000 + seed)
    best = rng.integers(N_ACTIONS, size=N_CONTEXTS)
    best2 = np.array([(b + 1 + rng.integers(N_ACTIONS - 1)) % N_ACTIONS for b in best])   # different answer
    rewards, explore_flags = np.zeros(TRIALS), np.zeros(TRIALS)
    for t in range(TRIALS):
        tgt = best if t < SWITCH else best2
        c = int(rng.integers(N_CONTEXTS))
        a, mode = agent.act(c)
        r = 1.0 if a == tgt[c] else -1.0                  # sparse: only the right action pays
        agent.learn(c, a, r)
        rewards[t] = (r > 0); explore_flags[t] = (mode == "explore")
    return rewards, explore_flags


def run(make, label, n_seeds=30):
    R, E = [], []
    for s in range(n_seeds):
        r, e = episode(make(s), s); R.append(r); E.append(e)
    R, E = np.array(R), np.array(E)
    pre = R[:, SWITCH-500:SWITCH].mean()
    post = R[:, -500:].mean()
    # recovery: first trial after the switch where a 200-trial rolling hit-rate exceeds 0.6
    rec = []
    for row in R:
        seg = row[SWITCH:]
        roll = np.convolve(seg, np.ones(200)/200, mode="valid")
        idx = np.flatnonzero(roll > 0.6)
        rec.append(idx[0] if len(idx) else np.nan)
    rec = np.array(rec, dtype=float)
    print(f"  {label:>34} {pre:>9.3f} {post:>10.3f} {np.nanmean(rec):>11.0f} "
          f"{100*np.isnan(rec).mean():>9.0f}% {E[:, :SWITCH].mean():>9.2f} {E[:, SWITCH:].mean():>9.2f}")


if __name__ == "__main__":
    print("=" * 108)
    print("BENCHMARK 7 — CLOSED LOOP: contextual bandit with an UNSIGNALLED goal switch at trial %d" % SWITCH)
    print(f"{N_CONTEXTS} contexts x {N_ACTIONS} actions | sparse reward | {TRIALS} trials | 30 seeds")
    print("=" * 108)
    print(f"  {'agent':>34} {'pre-switch':>9} {'post-switch':>10} {'recovery':>11} {'failed':>10} "
          f"{'expl pre':>9} {'expl post':>9}")
    print("  " + "-" * 100)
    run(lambda s: CGLAgent(seed=s), "coherence-gated (no epsilon at all)")
    run(lambda s: EpsGreedy(seed=s, decay=400), "eps-greedy, DECAYING (tuned)")
    run(lambda s: EpsGreedy(seed=s, eps0=0.10, decay=None), "eps-greedy, FIXED 0.10")
    run(lambda s: EpsGreedy(seed=s, eps0=0.02, decay=None), "eps-greedy, FIXED 0.02")
    run(lambda s: Softmax(seed=s, temp=0.2), "softmax, T=0.2")
    print("  " + "-" * 100)
    print("  recovery = trials after the switch to regain a 60% hit-rate | failed = never regained it")
