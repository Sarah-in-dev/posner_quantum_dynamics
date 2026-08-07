#!/usr/bin/env python3
"""
F3 Phase B — the windowed-dopamine DISCRIMINATION probe (the minimal first experiment).

Tests the load-bearing F3 claim: does a dopamine transard gating an eligibility trace ONLY inside the
0.3-2 s coincidence window (Yagishita 2014) produce TIMING-discriminated durable consolidation? This is
the thing F2-e showed we lack (reward inert, calcium-fast commit).

Design (within-synapse timing manipulation = the clean form of "synapse A in-window vs B out-of-window";
controls for synapse identity entirely). Each run: drive one synapse to eligibility (build dimers → P_S
trace), set the eligibility event, then deliver ONE 100 ms dopamine transient at a controlled latency
`t_since` after the event, then let the spine consolidate, and measure durable Δactin_stable.

Conditions (ONE grounded parameter set — the 0.3-2 s window + Garris DA levels; nothing retuned):
  IN        burst @ t_since=1.0 s   (inside window)      → expect potentiation
  OUT_LATE  burst @ t_since=3.0 s   (after window)       → expect none
  OUT_EARLY burst @ t_since=0.1 s   (before window)      → expect none
  NODA      no transient                                 → expect none (DA necessary)
  DIP       dip   @ t_since=1.0 s   (inside window)      → expect depression / no potentiation (sign)

Pre-registered (PREREG_F3) acceptance: IN > {OUT_LATE, OUT_EARLY, NODA} (window discriminates timing) and
IN > DIP (sign), each by a permutation null-p < 0.05. A result that lifts IN only by also lifting the
out-of-window/NODA arms is an artifact ⇒ report the negative.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S = 2.0          # drive to build the dimer/P_S eligibility trace
CONSOLIDATE_S = 20.0   # let actin_stable respond after a commit (confinement ~tens of s)
DA_BURST = 10e-6       # M, phasic peak (Garris 1994; model6_parameters dopamine_phasic_peak)
DA_TONIC = 20e-9       # M, tonic baseline (Garris 1994) — injected explicitly when no transient
DA_DIP = 5e-9          # M, below tonic 20 nM (a dip)
BURST_DUR_S = 0.1      # 100 ms transient (dopamine_system.burst_duration)

CONDS = {  # name: (t_since or None, kind)
    "IN":        (1.0, "burst"),
    "OUT_LATE":  (3.0, "burst"),
    "OUT_EARLY": (0.1, "burst"),
    "NODA":      (None, "none"),
    "DIP":       (1.0, "dip"),
}


def one_run(t_since, kind, seed):
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True   # EM path carries the commitment machinery
    s = Model6QuantumSynapse(p)                            # (the non-EM baseline has NO commitment gate)
    s._network_controlled = True          # disable the single-synapse auto measurement-commit path
    s._reward_gated_consolidation = True  # F3 windowed gate is the ONLY commit path
    for _ in range(int(BUILD_S / DT)):    # build dimers → P_S eligibility trace
        s.step(DT, {"voltage": -40e-3})
    # set the eligibility event (network normally does this)
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    elig_ps = float(getattr(s, "_mean_singlet_prob", np.nan))
    stable_pre = s.spine_plasticity.actin_stable
    da_step = None if kind == "none" else int(round(t_since / DT))
    burst_steps = int(round(BURST_DUR_S / DT))
    for k in range(int(CONSOLIDATE_S / DT)):
        if da_step is not None and 0 <= (k - da_step) < burst_steps:
            s._da_signal = DA_BURST if kind == "burst" else DA_DIP
        else:
            s._da_signal = DA_TONIC       # explicit tonic — the live DA field drifts >tonic during dynamics
                                          # (exactly tonic only at rest), which reads as a spurious burst
        s.step(DT, {"voltage": -70e-3})
    stable_post = s.spine_plasticity.actin_stable
    return dict(d_stable=float(stable_post - stable_pre), committed=bool(s._camkii_committed),
                P_S=elig_ps, t_commit=(None if not s._camkii_committed else float(getattr(s, "_commitment_time", -1))))


def permutation_p(a, b, T=20000, seed=1):
    a, b = np.array(a), np.array(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(T):
        p = rng.permutation(pool)
        ge += (p[:na].mean() - p[na:].mean()) >= obs      # one-sided: IN greater
    return obs, ge / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = {c: [] for c in CONDS}
    for c, (t, kind) in CONDS.items():
        for seed in range(a.n):
            results[c].append(one_run(t, kind, seed))
        ds = [r["d_stable"] for r in results[c]]
        nc = sum(r["committed"] for r in results[c])
        print(f"  {c:<10} Δactin_stable mean={np.mean(ds):+.5f} sd={np.std(ds):.5f}  committed={nc}/{a.n}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "f3_discrimination.json"), "w") as f:
            json.dump({c: results[c] for c in CONDS}, f, indent=1)

    IN = [r["d_stable"] for r in results["IN"]]
    print("\n=== ACCEPTANCE (pre-registered; one-sided permutation, IN greater) ===")
    checks = []
    for other in ["OUT_LATE", "OUT_EARLY", "NODA", "DIP"]:
        o = [r["d_stable"] for r in results[other]]
        diff, p = permutation_p(IN, o)
        ok = (diff > 0) and (p < 0.05)
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] IN > {other:<9} diff={diff:+.5f} null-p={p:.3f}")
    allok = all(checks)
    print(f"\n  VERDICT: {'PASS — windowed dopamine produces timing+sign-discriminated consolidation' if allok else 'MIXED/NEGATIVE — see arms above (report honestly, do not retune)'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
