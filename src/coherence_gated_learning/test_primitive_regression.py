#!/usr/bin/env python3
"""
REGRESSION SUITE for the consolidated primitive (cgl_primitive.py).

The seven mechanisms were each validated in a DIFFERENT benchmark script, on a DIFFERENT partial
implementation. Consolidating them into one class is only safe if the one class still reproduces each
original result on the ORIGINAL task -- otherwise a benchmark number would not say which subset it measured.

Each test below re-runs an original protocol (imported from its benchmark module where possible, so the task
cannot silently drift) against CoherenceGatedPrimitive, and asserts the previously MEASURED outcome. Thresholds
are set below the measured values with headroom for seed noise; they are regression guards, not targets.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from cgl_primitive import CoherenceGatedPrimitive, CGLParams
import benchmark_structure as BS
import benchmark_foraging as BF


def _p(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} {detail}")
    return ok


# ---------------------------------------------------------------- 1+2+3: conjunctive credit (XOR)
def test_structured_credit():
    """ORIGINAL (RESULT_structured_credit): graph ~100%, scalar trace baseline at chance (~50%).
    Exercises trace decay, two-timescale binding, and per-component joint collapse together."""
    accs = []
    for seed in range(30):
        rng = np.random.default_rng(500 + seed)
        a, b = rng.choice(BS.N_UNITS, size=2, replace=False)
        L = CoherenceGatedPrimitive(BS.N_UNITS, CGLParams(seed=seed))
        for _ in range(600):
            case, r = BS.CASES[rng.integers(4)]
            BS.trial(L, int(a), int(b), case, rng, is_graph=True)
            L.reward(r)
            L.clear_episode()
        ok = 0
        for _ in range(200):
            case, r = BS.CASES[rng.integers(4)]
            present = {"a": [a], "b": [b], "ab": [a, b], "none": []}[case]
            v = L.value(present) if present else -1.0
            ok += int((np.sign(v) if v != 0 else -1.0) == r)
        accs.append(ok / 200)
    m = float(np.mean(accs))
    return _p("structured credit (XOR conjunction)", m > 0.90, f"accuracy {m*100:.1f}% (was ~100%, scalar 50%)")


# ---------------------------------------------------------------- 2: the two timescales are load-bearing
def _xor_acc(nseed=12, **kw):
    out = []
    for seed in range(nseed):
        rng = np.random.default_rng(500 + seed)
        a, b = rng.choice(BS.N_UNITS, size=2, replace=False)
        L = CoherenceGatedPrimitive(BS.N_UNITS, CGLParams(seed=seed, **kw))
        for _ in range(600):
            case, r = BS.CASES[rng.integers(4)]
            BS.trial(L, int(a), int(b), case, rng, is_graph=True)
            L.reward(r); L.clear_episode()
        ok = 0
        for _ in range(200):
            case, r = BS.CASES[rng.integers(4)]
            present = {"a": [a], "b": [b], "ab": [a, b], "none": []}[case]
            v = L.value(present) if present else -1.0
            ok += int((np.sign(v) if v != 0 else -1.0) == r)
        out.append(ok / 200)
    return float(np.mean(out)), L


def test_bind_window_load_bearing():
    """Collapsing formation and persistence into ONE constant lets every live node bind to every other, so
    components fill with distractors.

    MEASURED 2026-08-24, and it CORRECTS the original claim's scope. The 2x2:
                          overlap ON   overlap OFF
        bind_window=2       100.0%       100.0%
        bind_window=inf     100.0%        50.7%   <- chance, the originally-reported collapse
    The binding window is load-bearing EXACTLY AS FOUND -- but only once partial reactivation is switched
    off. With it on, similarity-weighted retrieval RECOVERS the structure from junk components. The two
    mechanisms are partially redundant on this task, and the cost shows up in the REPRESENTATION instead:
    21 stored keys of mean size 1.05 (clean {a},{b},{a,b}) versus 547 keys of mean size 3.81 -- a 26x
    memory blow-up for the same accuracy. So binding buys COMPACTNESS; partial reactivation buys
    ROBUSTNESS to bad binding. Asserting only the accuracy would have hidden that."""
    collapsed_alone, _ = _xor_acc(overlap_generalization=False, bind_window=1e9)
    clean_alone, _ = _xor_acc(overlap_generalization=False, bind_window=2.0)
    _, L_bad = _xor_acc(nseed=1, bind_window=1e9)
    _, L_ok = _xor_acc(nseed=1, bind_window=2.0)
    blowup = len(L_bad.store) / max(len(L_ok.store), 1)
    ok = collapsed_alone < 0.65 and clean_alone > 0.90 and blowup > 5.0
    return _p("bind_window is load-bearing", ok,
              f"no-overlap {clean_alone*100:.0f}% -> {collapsed_alone*100:.0f}%; keys x{blowup:.0f} when collapsed")


# ---------------------------------------------------------------- 4: singletons
def test_singleton_component():
    """A lone readable node is its own component and CAN be credited (defect found 2026-08-24)."""
    L = CoherenceGatedPrimitive(5, CGLParams(seed=0))
    L.activate([1, 0, 0, 0, 0]); L.decay(1.0)
    comps = L.components()
    return _p("singleton component is creditable", comps == [{0}], f"components={comps}")


# ---------------------------------------------------------------- reward necessity
def test_reward_necessity():
    """r == 0 must commit nothing, anywhere (the F3 result the biology run also has to satisfy)."""
    L = CoherenceGatedPrimitive(5, CGLParams(seed=0))
    L.activate([1, 1, 0, 0, 0]); L.decay(1.0)
    fired, n = L.reward(0.0), L.n_commits
    return _p("reward is necessary (r=0 -> no commit)", fired == [] and n == 0, f"fired={fired} commits={n}")


# ---------------------------------------------------------------- 5: partial reactivation
def test_partial_reactivation():
    """An unseen combination must be answered by overlapping memories, not sit at 0 (was exactly chance)."""
    L = CoherenceGatedPrimitive(10, CGLParams(seed=0))
    for _ in range(50):
        L.learn([0, 1], +1.0); L.learn([2, 3], -1.0)
    near_pos, near_neg = L.value([0, 1, 4]), L.value([2, 3, 4])
    off = CoherenceGatedPrimitive(10, CGLParams(seed=0, overlap_generalization=False))
    for _ in range(50):
        off.learn([0, 1], +1.0)
    ok = near_pos > 0.3 and near_neg < -0.3 and off.value([0, 1, 4]) == 0.0
    return _p("partial reactivation of unseen combos", ok, f"near+ {near_pos:+.2f} near- {near_neg:+.2f}, off=0")


# ---------------------------------------------------------------- capacity
def test_capacity():
    """RESULT_scaling_and_limits: distinct conjunctions stay separable at K=64."""
    K = 64
    L = CoherenceGatedPrimitive(2 * K, CGLParams(seed=0))
    for k in range(K):
        for _ in range(30):
            L.learn([2 * k, 2 * k + 1], +1.0 if k % 2 == 0 else -1.0)
    ok = sum(int(np.sign(L.value([2 * k, 2 * k + 1])) == (1 if k % 2 == 0 else -1)) for k in range(K))
    return _p("capacity at K=64 conjunctions", ok == K, f"{ok}/{K} recovered with correct sign")


# ---------------------------------------------------------------- 6+7: non-stationary foraging
def test_foraging_nonstationary():
    """RESULT_foraging: with the active depression arm, pre 1.000 AND post 1.000 across an unsignalled
    rule switch, beating the best tuned baseline (softmax 0.888). Exercises commit-probability selection,
    partial reactivation and the LTD arm on self-chosen data."""
    class Agent:                       # thin adapter: the foraging env speaks features, not unit indices
        def __init__(self, seed):
            self.c = CoherenceGatedPrimitive(BF.N_FEATURES, CGLParams(seed=seed))
        def choose(self, patches): return self.c.choose([[int(f) for f in p] for p in patches])
        def learn(self, feats, r):  self.c.learn([int(f) for f in feats], r)

    R = np.array([BF.episode(Agent(s), s)[0] for s in range(20)])
    pre, post = R[:, BF.SWITCH - 500:BF.SWITCH].mean(), R[:, -500:].mean()
    return _p("non-stationary foraging (pre AND post)", pre > 0.95 and post > 0.95,
              f"pre {pre:.3f} post {post:.3f} (sticky variant was post 0.695; best baseline 0.888)")


if __name__ == "__main__":
    print("=" * 96)
    print("REGRESSION — does the CONSOLIDATED primitive still reproduce every original result?")
    print("=" * 96)
    results = [t() for t in (test_structured_credit, test_bind_window_load_bearing, test_singleton_component,
                             test_reward_necessity, test_partial_reactivation, test_capacity,
                             test_foraging_nonstationary)]
    print("=" * 96)
    print(f"  {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
