#!/usr/bin/env python3
"""
F3 ISOTOPE-CREDIT probe — the physical lever (F1) on the eligibility-trace lifetime (F3), in one experiment.

The coherence-window thesis (F3-b) predicts the eligibility-trace lifetime = the ³¹P coherence lifetime.
F1's ⁶Li/⁷Li dopant is the DERIVED physical lever on that lifetime (nuclear_relaxation.T2_observed: ⁷Li
Larmor-matches ³¹P → fast scalar decoherence T₂≈14 s; ⁶Li far → negligible, T₂≈216 s ≈ undoped). So the
isotope should DIAL delayed-credit: ⁷Li's tag decoheres (crosses the Werner floor) by ~5 s and credits only
at SHORT delays; ⁶Li ≈ undoped and credits out to ~100 s. This is the falsifiable, mechanism-level isotope
signature that ties F1 → F2 → F3: change one nuclear-spin degree of freedom, watch temporal credit move.

Design = the F3 delayed-credit probe with an added dopant axis (quantum mode only).
Sweep: dopant ∈ {None, Li6, Li7} × delay ∈ {2, 5, 10, 30} s.

Pre-registered prediction (PREREG_F3, isotope arm): (1) ⁶Li ≈ undoped — credits at ALL delays; (2) ⁷Li
credits at short delay (2 s) but FAILS at long delays (10, 30 s) — its tag has decohered; (3) the isotope
contrast at ≥10 s (undoped/⁶Li credit, ⁷Li does not) is significant. A null (⁷Li also credits at 30 s, or
⁶Li also fails) is reportable — it would falsify the coherence-lifetime = trace-lifetime claim.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S = 2.0
CONSOLIDATE_S = 15.0
DA_BURST = 10e-6
DA_TONIC = 20e-9
BURST_DUR_S = 0.1
DOPANTS = [None, "Li6", "Li7"]
DELAYS = [2.0, 5.0, 10.0, 30.0]


def one_run(dopant, delay, seed):
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    p.environment.dopant = dopant                      # F1 lever on the coherence (tag) lifetime
    s = Model6QuantumSynapse(p)
    s._network_controlled = True
    s._reward_gated_consolidation = True
    s._reward_gating_mode = "quantum"
    for _ in range(int(BUILD_S / DT)):
        s.step(DT, {"voltage": -40e-3})
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    stable_pre = s.spine_plasticity.actin_stable
    da_step = int(round(delay / DT))
    burst_steps = int(round(BURST_DUR_S / DT))
    ps_at_reward = None
    for k in range(int((delay + CONSOLIDATE_S) / DT)):
        s._da_signal = DA_BURST if (0 <= k - da_step < burst_steps) else DA_TONIC
        s.step(DT, {"voltage": -70e-3})
        if k == da_step:
            ps_at_reward = float(getattr(s, "_mean_singlet_prob", np.nan))
    return dict(dopant=str(dopant), delay=delay, committed=bool(s._camkii_committed),
                d_stable=float(s.spine_plasticity.actin_stable - stable_pre), ps_at_reward=ps_at_reward)


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
    R = {(str(d), dl): [] for d in DOPANTS for dl in DELAYS}
    for d in DOPANTS:
        for dl in DELAYS:
            for seed in range(a.n):
                R[(str(d), dl)].append(one_run(d, dl, seed))
    print(f"  {'dopant':<8}{'delay s':>8}{'commit':>10}{'P_S@reward':>12}{'d_stable':>12}")
    for d in DOPANTS:
        for dl in DELAYS:
            rs = R[(str(d), dl)]
            nc = sum(r["committed"] for r in rs)
            pr = np.mean([r["ps_at_reward"] for r in rs if r["ps_at_reward"] is not None])
            dm = np.mean([r["d_stable"] for r in rs])
            print(f"  {str(d):<8}{dl:>8}{str(nc)+'/'+str(a.n):>10}{pr:>12.3f}{dm:>12.5f}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "f3_isotope_credit.json"), "w") as f:
            json.dump({f"{d}_{dl}": R[(str(d), dl)] for d in DOPANTS for dl in DELAYS}, f, indent=1)

    print("\n=== ACCEPTANCE (pre-registered isotope arm) ===")
    checks = []
    # 1. Li6 ≈ undoped: both credit at all delays
    none_all = all(sum(r["committed"] for r in R[("None", dl)]) >= a.n - 1 for dl in DELAYS)
    li6_all = all(sum(r["committed"] for r in R[("Li6", dl)]) >= a.n - 1 for dl in DELAYS)
    ok1 = none_all and li6_all
    checks.append(ok1); print(f"  [{'PASS' if ok1 else 'FAIL'}] ⁶Li ≈ undoped: both commit at ALL delays (none={none_all}, Li6={li6_all})")
    # 2. Li7 short-only: credits at 2 s, fails at 10 & 30 s
    li7_2 = sum(r["committed"] for r in R[("Li7", 2.0)])
    li7_10 = sum(r["committed"] for r in R[("Li7", 10.0)])
    li7_30 = sum(r["committed"] for r in R[("Li7", 30.0)])
    ok2 = li7_2 >= a.n - 1 and li7_10 == 0 and li7_30 == 0
    checks.append(ok2); print(f"  [{'PASS' if ok2 else 'FAIL'}] ⁷Li short-only: 2 s={li7_2}/{a.n}, 10 s={li7_10}/{a.n}, 30 s={li7_30}/{a.n}")
    # 3. isotope contrast @10 s: undoped+Li6 durable > Li7 durable, significant
    long_coh = [r["d_stable"] for dl in (10.0,) for k in ("None", "Li6") for r in R[(k, dl)]]
    li7_long = [r["d_stable"] for r in R[("Li7", 10.0)]]
    diff, p = perm_p(long_coh, li7_long)
    ok3 = diff > 0 and p < 0.05
    checks.append(ok3); print(f"  [{'PASS' if ok3 else 'FAIL'}] isotope contrast @10 s: (undoped/⁶Li)−⁷Li durable = {diff:+.5f} null-p={p:.3f}")
    allok = all(checks)
    print(f"\n  VERDICT: {'PASS — the ⁶Li/⁷Li lever dials the eligibility-trace lifetime; isotope moves temporal credit (F1→F3)' if allok else 'MIXED/NEGATIVE — report honestly, do not retune'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
