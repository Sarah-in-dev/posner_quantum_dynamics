#!/usr/bin/env python3
"""
F3 DELAYED-CREDIT probe — does the coherent P_S tag assign credit at delays where a classical trace can't?

REFRAMED against docs/RESEARCH_TCA_MECHANISM_2026-08-08. The temporal gap (measured eligibility traces
~0.3–5 s vs behaviour spanning seconds-to-minutes) is UNSOLVED at the single synapse. Model-6's candidate
answer: the trace is the coherent P_S tag (~100 s), read out by dopamine (decoherence) at ANY delay while
still coherent. This tests exactly that — and contrasts it against the CLASSICAL 0.3–2 s trace (biology's,
Yagishita/Shindou) as the baseline the quantum tag must beat.

Design: drive one synapse to eligibility (the tag), hold at rest injecting tonic dopamine, then at reward
DELAY inject ONE 100 ms dopamine burst, then a FIXED post-reward consolidation window (equal for all
delays), and measure commit + durable Δactin_stable. Sweep delay ∈ {1,2,5,10,30} s × mode ∈ {quantum,
classical}.

Pre-registered (PREREG_F3) prediction: the QUANTUM tag commits/consolidates across delays up to ~the
coherence lifetime (~107 s, undoped), while the CLASSICAL baseline dies past ~2 s. The quantum−classical
gap at delays ≥5 s IS the result: a trace long enough to solve the temporal gap. Honest premise (the ~100 s
coherence) is flagged; a null (quantum also dies by ~2 s) is reportable.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S = 2.0
CONSOLIDATE_S = 15.0     # FIXED post-reward consolidation (equal for every delay)
DA_BURST = 10e-6
DA_TONIC = 20e-9
BURST_DUR_S = 0.1
DELAYS = [1.0, 2.0, 5.0, 10.0, 30.0]
MODES = ["quantum", "classical"]


def one_run(mode, delay, seed):
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    s = Model6QuantumSynapse(p)
    s._network_controlled = True
    s._reward_gated_consolidation = True
    s._reward_gating_mode = mode
    for _ in range(int(BUILD_S / DT)):
        s.step(DT, {"voltage": -40e-3})
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    ps_at_tag = float(getattr(s, "_mean_singlet_prob", np.nan))
    stable_pre = s.spine_plasticity.actin_stable
    da_step = int(round(delay / DT))
    burst_steps = int(round(BURST_DUR_S / DT))
    total_steps = int((delay + CONSOLIDATE_S) / DT)
    ps_at_reward = None
    for k in range(total_steps):
        s._da_signal = DA_BURST if (0 <= k - da_step < burst_steps) else DA_TONIC
        s.step(DT, {"voltage": -70e-3})
        if k == da_step:
            ps_at_reward = float(getattr(s, "_mean_singlet_prob", np.nan))
    return dict(mode=mode, delay=delay, committed=bool(s._camkii_committed),
                d_stable=float(s.spine_plasticity.actin_stable - stable_pre),
                ps_at_tag=ps_at_tag, ps_at_reward=ps_at_reward)


def perm_p(a, b, T=20000, seed=1):
    a, b = np.array(a), np.array(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(T):
        pr = rng.permutation(pool)
        ge += (pr[:na].mean() - pr[na:].mean()) >= obs
    return obs, ge / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {(m, d): [] for m in MODES for d in DELAYS}
    for m in MODES:
        for d in DELAYS:
            for seed in range(a.n):
                R[(m, d)].append(one_run(m, d, seed))
    print(f"  {'mode':<10}{'delay s':>8}{'commit':>10}{'d_stable mean':>15}{'P_S@reward':>12}")
    for m in MODES:
        for d in DELAYS:
            rs = R[(m, d)]
            nc = sum(r["committed"] for r in rs)
            dm = np.mean([r["d_stable"] for r in rs])
            pr = np.mean([r["ps_at_reward"] for r in rs if r["ps_at_reward"] is not None])
            print(f"  {m:<10}{d:>8}{str(nc)+'/'+str(a.n):>10}{dm:>15.5f}{pr:>12.3f}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "f3_delayed_credit.json"), "w") as f:
            json.dump({f"{m}_{d}": R[(m, d)] for m in MODES for d in DELAYS}, f, indent=1)

    print("\n=== ACCEPTANCE (pre-registered) ===")
    checks = []
    # 1. classical baseline dies past the 2 s window
    cl_long = sum(r["committed"] for d in (5.0, 10.0, 30.0) for r in R[("classical", d)])
    ok_cl = cl_long == 0
    checks.append(ok_cl); print(f"  [{'PASS' if ok_cl else 'FAIL'}] classical baseline: 0 commits at delays >2 s ({cl_long} seen)")
    # 2. quantum tag still credits at long delays
    q10 = sum(r["committed"] for r in R[("quantum", 10.0)])
    q30 = sum(r["committed"] for r in R[("quantum", 30.0)])
    ok_q = q10 >= a.n - 1 and q30 >= a.n - 1
    checks.append(ok_q); print(f"  [{'PASS' if ok_q else 'FAIL'}] quantum tag commits at 10 s ({q10}/{a.n}) and 30 s ({q30}/{a.n})")
    # 3. the temporal-gap gap: quantum durable > classical durable at 10 s, significant
    q = [r["d_stable"] for r in R[("quantum", 10.0)]]
    c = [r["d_stable"] for r in R[("classical", 10.0)]]
    diff, p = perm_p(q, c)
    ok_gap = diff > 0 and p < 0.05
    checks.append(ok_gap); print(f"  [{'PASS' if ok_gap else 'FAIL'}] temporal-gap @10 s: quantum−classical durable = {diff:+.5f} null-p={p:.3f}")
    allok = all(checks)
    print(f"\n  VERDICT: {'PASS — coherence-window tag solves the temporal gap the classical trace cannot' if allok else 'MIXED/NEGATIVE — report honestly, do not retune'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
