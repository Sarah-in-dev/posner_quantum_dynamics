"""
BENCHMARK — delayed credit from a GLOBAL scalar reward, with distractors and interference.

THE CLAIM UNDER TEST (stated narrowly, on purpose). We are NOT claiming better asymptotic accuracy, and we
are NOT claiming anything about dense-reward supervised learning, where gradient methods win. The claim is
about a REGIME: a single scalar reward, arriving long after the causal activity, with no gradient, no stored
computation graph, and no replay.

BASELINE: a classical eligibility-trace learner -- w += lr * r * e, with e decaying geometrically and
replaced on activity. This is the three-factor / REINFORCE-with-traces rule, and it is the FAIREST possible
comparison because it already has the one thing usually missing: a trace that bridges the delay. Its
hyperparameters (learning rate, trace decay) are SWEPT and we report its BEST configuration, not a strawman.

WHAT DIFFERS ARCHITECTURALLY (this is what the benchmark isolates):
  1. a hard READABLE THRESHOLD -- below it a trace earns exactly zero, not a little
  2. STOCHASTIC one-shot COMMITMENT rather than a small proportional nudge
  3. a STRUCTURAL LATCH -- committed weights are protected, not continuously rewritten

TWO TASKS:
  A. SAMPLE EFFICIENCY  -- episodes to criterion (correct unit is argmax, held for 10 consecutive episodes)
  B. RETENTION          -- after learning, a long stretch of unrewarded/noisy-reward activity. Does the
                           learned answer survive? This is where a structural latch should differ from
                           weights that keep being updated.
A null (baseline matches or beats us) is a real result and is reported as such.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cgl import CoherenceGatedLearner, CGLParams

N_UNITS, DELAY, N_DISTRACT = 20, 40, 3


class TraceBaseline:
    """Classical eligibility-trace learner: w += lr * r * e ; e decays geometrically."""
    def __init__(self, n, lr, tau, seed=0):
        self.n, self.lr, self.tau = n, lr, tau
        self.w = np.zeros(n); self.e = np.zeros(n)
        self.rng = np.random.default_rng(seed)
    def activate(self, active):
        self.e[np.asarray(active, dtype=bool)] = 1.0
    def decay(self, dt=1.0):
        self.e *= np.exp(-dt / self.tau)
    def reward(self, r):
        if r != 0.0:
            self.w += self.lr * r * self.e
        return None


def episode(learner, target, rng, rewarded=True, reward_val=1.0):
    """Target fires; distractors fire during the delay; ONE global scalar reward at the end."""
    a = np.zeros(N_UNITS, dtype=bool)
    if rewarded:
        a[target] = True
    learner.activate(a)
    step = max(1, DELAY // (N_DISTRACT + 1))
    for _ in range(N_DISTRACT):
        learner.decay(step)
        d = np.zeros(N_UNITS, dtype=bool)
        pick = rng.choice([i for i in range(N_UNITS) if i != target], size=1)
        d[pick] = True
        learner.activate(d)
    learner.decay(DELAY - step * N_DISTRACT)
    learner.reward(reward_val if rewarded else 0.0)


def run_taskA(make, n_seeds=40, max_ep=400):
    """Episodes to criterion: argmax(w) == target for 10 consecutive episodes."""
    outs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        target = int(rng.integers(N_UNITS))
        L = make(seed)
        streak, ep_to_crit = 0, None
        for ep in range(max_ep):
            episode(L, target, rng)
            if L.w.max() > 0 and int(np.argmax(L.w)) == target:
                streak += 1
                if streak >= 10:
                    ep_to_crit = ep + 1; break
            else:
                streak = 0
        outs.append(ep_to_crit if ep_to_crit is not None else max_ep)
    return np.array(outs, dtype=float)


def run_taskB(make, n_seeds=40, learn_ep=200, interfere_ep=400, noise_p=0.15):
    """Retention: learn, then a long stretch of activity with NO true reward (occasional noise reward)."""
    keep = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(2000 + seed)
        target = int(rng.integers(N_UNITS))
        L = make(seed)
        for _ in range(learn_ep):
            episode(L, target, rng)
        learned = (L.w.max() > 0 and int(np.argmax(L.w)) == target)
        for _ in range(interfere_ep):          # distractor activity, target never fires
            a = np.zeros(N_UNITS, dtype=bool)
            a[rng.choice([i for i in range(N_UNITS) if i != target])] = True
            L.activate(a); L.decay(DELAY)
            L.reward(1.0 if rng.random() < noise_p else 0.0)   # spurious reward, unrelated to target
        kept = (L.w.max() > 0 and int(np.argmax(L.w)) == target)
        keep.append(bool(learned and kept))
    return float(np.mean(keep))


if __name__ == "__main__":
    cgl_make = lambda s: CoherenceGatedLearner(
        N_UNITS, CGLParams(tau=216.0, trace_floor=0.25, readable_threshold=0.707,
                           commit_gain=3.0, lr=1.0, seed=s))

    print("=" * 92)
    print("DELAYED-CREDIT BENCHMARK — one global scalar reward, %d steps after the causal activity" % DELAY)
    print(f"{N_UNITS} units, {N_DISTRACT} distractors firing during the delay")
    print("=" * 92)

    print("\nBaseline hyperparameter sweep (reporting its BEST, not a strawman):")
    best = None
    for lr in (0.01, 0.05, 0.2, 0.5, 1.0):
        for tau in (20.0, 60.0, 216.0, 600.0):
            r = run_taskA(lambda s, lr=lr, tau=tau: TraceBaseline(N_UNITS, lr, tau, s))
            med = float(np.median(r))
            if best is None or med < best[0]:
                best = (med, lr, tau, r)
    med_b, lr_b, tau_b, rb = best
    print(f"  best baseline: lr={lr_b}  trace_tau={tau_b}  ->  median episodes to criterion = {med_b:.0f}")

    ra = run_taskA(cgl_make)
    print("\nTASK A — SAMPLE EFFICIENCY (episodes to criterion; 400 = never reached)")
    print(f"  {'coherence-gated':>18}: median {np.median(ra):>6.0f}   mean {ra.mean():>6.1f}   "
          f"solved {np.mean(ra < 400)*100:.0f}%")
    print(f"  {'best trace baseline':>18}: median {med_b:>6.0f}   mean {rb.mean():>6.1f}   "
          f"solved {np.mean(rb < 400)*100:.0f}%")

    print("\nTASK B — RETENTION (learn, then 400 episodes of distractor activity + spurious reward)")
    kb = run_taskB(lambda s: TraceBaseline(N_UNITS, lr_b, tau_b, s))
    kc = run_taskB(cgl_make)
    print(f"  {'coherence-gated':>18}: retained correct answer in {kc*100:5.1f}% of runs")
    print(f"  {'best trace baseline':>18}: retained correct answer in {kb*100:5.1f}% of runs")
    print("=" * 92)
