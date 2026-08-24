#!/usr/bin/env python3
"""
BENCHMARK 9 — ESTABLISHED NON-STATIONARY BANDIT BENCHMARK (the first test against published algorithms).

WHY THIS ONE. Every previous benchmark used a task WE designed, and baselines we wrote. The measured
strengths -- non-stationarity handled with no exploration parameter -- point at exactly one established
literature: switching/abruptly-changing bandits. So this is the honest place to meet published work.

THE ENVIRONMENT (abruptly-changing Bernoulli bandit, Garivier & Moulines 2011 setting).
K arms with Bernoulli rewards whose means change ABRUPTLY at a small number of breakpoints. Regret is
measured against the ORACLE that plays the best arm at each t (not the best fixed arm), which is what makes
the problem a switching problem rather than a stationary one.
  ENV-A "few breakpoints"  : K=3, T=10000, means change at t=3000 and t=5000  (the GM setting)
  ENV-B "frequent"         : K=5, T=10000, means redrawn every 1000 steps     (harder; more switches)
NB: this is the standard PROTOCOL re-implemented, not a claim to reproduce any specific published figure.

THE BASELINES (published algorithms, implemented from their definitions; hyperparameters stated, not tuned
against our agent):
  UCB1                Auer, Cesa-Bianchi & Fischer 2002 -- stationary; expected to fail after a switch
  Thompson (Beta)     Thompson 1933 / Chapelle & Li 2011 -- stationary; same expectation
  D-UCB               Kocsis & Szepesvari 2006, analysed by Garivier & Moulines 2011 (discounted counts)
  SW-UCB              Garivier & Moulines 2011 (sliding window)
  EXP3.S              Auer, Cesa-Bianchi, Freund & Schapire 2002 (adversarial, tracks the best arm)
  Discounted Thompson a discounted-Beta variant, the common practical non-stationary TS
D-UCB / SW-UCB use the paper's tuning rules with xi=0.6, gamma = 1 - (1/4)sqrt(Upsilon/T),
tau = 2 sqrt(T log T / Upsilon), given the TRUE number of breakpoints Upsilon -- i.e. these baselines are
handed information about the switch structure that our agent is NOT given.

HOW THE PRIMITIVE PLAYS. Each arm is ONE unit, so components are singletons and the CONJUNCTIVE machinery
is inert here -- this benchmark isolates mechanisms 6 and 7 (commit-probability selection + the active
depression arm). That is the point: the non-stationarity claim should stand on its own.

THE REWARD-SCALE QUESTION, RUN BOTH WAYS (this is a real fork, not a detail). The primitive was validated on
+/-1 rewards. Bandit benchmarks use Bernoulli {0,1}, where EVERY arm has positive value, so every arm's coin
fires and selection degenerates toward uniform. Two arms are reported:
  'raw {0,1}'     : rewards fed unchanged -- expected to do badly, and reported if it does
  'centered'      : r -> 2r-1, i.e. reward measured against the BASELINE rather than against zero. This is
                    the dopamine convention (phasic bursts ABOVE and dips BELOW tonic baseline encode a
                    signed prediction error; Schultz), so it is a grounded modelling choice rather than a
                    tuned one -- but it IS a choice, and the raw arm is shown so the reader can see its size.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from cgl_primitive import CoherenceGatedPrimitive, CGLParams

T = 10000


# ------------------------------------------------------------------ environments
def env_few(rng):
    """K=3, breakpoints at 3000 and 5000 (Garivier & Moulines setting). Returns means[t, arm]."""
    K = 3
    seg = [(0, [0.5, 0.3, 0.4]), (3000, [0.2, 0.6, 0.3]), (5000, [0.3, 0.2, 0.7])]
    mu = np.zeros((T, K))
    for i, (start, m) in enumerate(seg):
        end = seg[i + 1][0] if i + 1 < len(seg) else T
        mu[start:end] = m
    return mu, 2                                   # 2 breakpoints


def env_frequent(rng):
    """K=5, means redrawn uniformly every 1000 steps -- many switches."""
    K, block = 5, 1000
    mu = np.zeros((T, K))
    for b in range(T // block):
        mu[b * block:(b + 1) * block] = rng.uniform(0.1, 0.9, size=K)
    return mu, T // block - 1


ENVS = [("ENV-A few breakpoints (K=3)", env_few), ("ENV-B frequent switches (K=5)", env_frequent)]


# ------------------------------------------------------------------ published baselines
class UCB1:
    def __init__(self, K, rng, **kw): self.K, self.rng = K, rng; self.n = np.zeros(K); self.s = np.zeros(K)
    def act(self, t):
        if (self.n == 0).any(): return int(np.flatnonzero(self.n == 0)[0])
        return int(np.argmax(self.s / self.n + np.sqrt(2 * np.log(t + 1) / self.n)))
    def learn(self, a, r): self.n[a] += 1; self.s[a] += r


class ThompsonBeta:
    def __init__(self, K, rng, **kw): self.rng = rng; self.a = np.ones(K); self.b = np.ones(K)
    def act(self, t): return int(np.argmax(self.rng.beta(self.a, self.b)))
    def learn(self, a, r): self.a[a] += r; self.b[a] += 1 - r


class DiscountedThompson:
    def __init__(self, K, rng, gamma=0.99, **kw): self.rng, self.g = rng, gamma; self.a = np.ones(K); self.b = np.ones(K)
    def act(self, t): return int(np.argmax(self.rng.beta(self.a, self.b)))
    def learn(self, a, r):
        self.a = self.g * self.a + (1 - self.g); self.b = self.g * self.b + (1 - self.g)   # decay toward prior
        self.a[a] += r; self.b[a] += 1 - r


class DUCB:
    """Discounted UCB. N_t(g,i)=sum g^{t-s}1{A_s=i}; c = 2B sqrt(xi log n_t(g) / N_t(g,i))."""
    def __init__(self, K, rng, gamma=0.99, xi=0.6, **kw):
        self.K, self.rng, self.g, self.xi = K, rng, gamma, xi
        self.N = np.zeros(K); self.X = np.zeros(K)
    def act(self, t):
        if (self.N == 0).any(): return int(np.flatnonzero(self.N == 0)[0])
        n = self.N.sum()
        return int(np.argmax(self.X / self.N + 2 * np.sqrt(self.xi * np.log(max(n, 2)) / self.N)))
    def learn(self, a, r):
        self.N *= self.g; self.X *= self.g
        self.N[a] += 1; self.X[a] += r


class SWUCB:
    """Sliding-Window UCB: statistics over the last tau plays only."""
    def __init__(self, K, rng, tau=1000, xi=0.6, **kw):
        self.K, self.rng, self.tau, self.xi = K, rng, int(tau), xi
        self.hist = []                                        # (arm, reward), truncated to tau
    def act(self, t):
        N = np.zeros(self.K); X = np.zeros(self.K)
        for (a, r) in self.hist:
            N[a] += 1; X[a] += r
        if (N == 0).any(): return int(np.flatnonzero(N == 0)[0])
        w = min(t + 1, self.tau)
        return int(np.argmax(X / N + np.sqrt(self.xi * np.log(max(w, 2)) / N)))
    def learn(self, a, r):
        self.hist.append((a, r))
        if len(self.hist) > self.tau: self.hist.pop(0)


class EXP3S:
    """EXP3.S (Auer et al. 2002): exponential weights with a mixing term that lets it TRACK a changing best arm."""
    def __init__(self, K, rng, gamma=None, alpha=None, **kw):
        self.K, self.rng = K, rng
        self.gamma = gamma if gamma is not None else min(1.0, np.sqrt(K * np.log(K * T) / ((np.e - 1) * T)))
        self.alpha = alpha if alpha is not None else 1.0 / T
        self.w = np.ones(K)
    def _p(self):
        w = self.w / self.w.sum()
        return (1 - self.gamma) * w + self.gamma / self.K
    def act(self, t):
        self.p = self._p(); return int(self.rng.choice(self.K, p=self.p))
    def learn(self, a, r):
        xhat = np.zeros(self.K); xhat[a] = r / self.p[a]
        self.w = self.w * np.exp(self.gamma * xhat / self.K) + (np.e * self.alpha / self.K) * self.w.sum()
        self.w = np.clip(self.w, 1e-12, 1e12)


# ------------------------------------------------------------------ the primitive
class CGLBandit:
    """Each arm = one unit (singleton components). Isolates commit-probability selection + active depression."""
    def __init__(self, K, rng, seed=0, rate=0.1, centered=True, **kw):
        self.K, self.centered = K, centered
        self.c = CoherenceGatedPrimitive(K, CGLParams(seed=seed, learn_rate=rate))
        self.cands = [[i] for i in range(K)]
    def act(self, t): return self.c.choose(self.cands)
    def learn(self, a, r): self.c.learn([a], (2.0 * r - 1.0) if self.centered else r)


# ------------------------------------------------------------------ harness
def run(make, mu, n_break, seed):
    rng = np.random.default_rng(seed)
    K = mu.shape[1]
    agent = make(K, rng, n_break)
    draw = rng.random((T, K)) < mu                      # pre-drawn Bernoulli outcomes
    regret = 0.0
    for t in range(T):
        a = agent.act(t)
        r = float(draw[t, a])
        agent.learn(a, r)
        regret += mu[t].max() - mu[t, a]                # regret vs the per-step oracle
    return regret


def main():
    n_seeds = 20
    for env_name, env_fn in ENVS:
        mu, n_break = env_fn(np.random.default_rng(7))
        gamma = 1 - 0.25 * np.sqrt(max(n_break, 1) / T)
        tau = 2 * np.sqrt(T * np.log(T) / max(n_break, 1))
        algos = [
            ("UCB1 (stationary)",            lambda K, rng, nb: UCB1(K, rng)),
            ("Thompson Beta (stationary)",   lambda K, rng, nb: ThompsonBeta(K, rng)),
            ("D-UCB (GM2011)",               lambda K, rng, nb: DUCB(K, rng, gamma=gamma)),
            ("SW-UCB (GM2011)",              lambda K, rng, nb: SWUCB(K, rng, tau=tau)),
            ("EXP3.S (Auer2002)",            lambda K, rng, nb: EXP3S(K, rng)),
            ("Discounted Thompson",          lambda K, rng, nb: DiscountedThompson(K, rng)),
            ("CGL primitive (raw {0,1})",    lambda K, rng, nb: CGLBandit(K, rng, seed=int(rng.integers(1e6)), centered=False)),
            ("CGL primitive (centered)",     lambda K, rng, nb: CGLBandit(K, rng, seed=int(rng.integers(1e6)), centered=True)),
        ]
        print("=" * 92)
        print(f"{env_name} | T={T}, {n_break} breakpoint(s) | regret vs PER-STEP oracle | {n_seeds} seeds")
        print(f"  D-UCB gamma={gamma:.5f}  SW-UCB tau={tau:.0f}  (both given the TRUE breakpoint count)")
        print("=" * 92)
        rows = []
        for name, make in algos:
            if env_name.startswith("ENV-B"):
                rs = []
                for s in range(n_seeds):
                    m, nb = env_frequent(np.random.default_rng(7 + s))
                    rs.append(run(make, m, nb, 1000 + s))
                R = np.array(rs)
            else:
                R = np.array([run(make, mu, n_break, 1000 + s) for s in range(n_seeds)])
            rows.append((name, R.mean(), R.std()))
        best = min(r[1] for r in rows)
        for name, m, sd in rows:
            mark = "  <-- best" if m == best else ""
            print(f"  {name:>30}  regret {m:8.1f}  (sd {sd:6.1f}){mark}")
        print()


if __name__ == "__main__":
    main()
