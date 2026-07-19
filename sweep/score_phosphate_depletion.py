#!/usr/bin/env python3
"""OFFLINE SCORER for PO-2's phosphate depletion bound — the authoritative verdict.

Composed from `src/models/Model_6/sweep/score_leta5.py`'s shape, per MO gen-2's standing rule:
compute buys the trace, the verdict is derived from the trace. The runner
(`phosphate_depletion_bound_probe.py`) computes no verdict, so a bug in THIS file costs zero
compute to fix — which is the 58 minutes PO-5 lost to a scorer bug on an unpersisted intermediate.

Applies the criteria registered in PREREG_PO2_PHOSPHATE.md AMENDMENT A2.6, fixed before the run:

  NO_DRAIN_TO_BOUND  |t| < 2            -> report 95% upper bound on |slope| and the implied
                                            MINIMUM time-to-depletion. A BOUND, never a proof of zero.
  DRAIN_DETECTED     t <= -2 AND monotonic -> overturns PO2-9 on a longer horizon.
  NONLINEAR          quadratic term significant.
  INVALID            conservation drifted past eps during the run.

Safe to run against a partial trace while the runner is still going.

Usage:  venv/bin/python sweep/score_phosphate_depletion.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "..", "src", "models", "Model_6", "results",
                     "phosphate_depletion", "depletion_grounded_seed1000.json")

EPS_REL = 1e-12       # same tolerance registered in A2.3 §3
T_CRIT = 2.0          # |t| threshold registered in A2.6


def fit(x, y, deg=1):
    """Least squares with slope standard error and t statistic."""
    n = len(x)
    coeffs = np.polyfit(x, y, deg)
    yhat = np.polyval(coeffs, x)
    resid = y - yhat
    dof = n - (deg + 1)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sxx = float(((x - x.mean()) ** 2).sum())
    se_slope = np.sqrt(ss_res / dof / sxx) if dof > 0 and sxx > 0 else float("nan")
    slope = coeffs[-2] if deg >= 1 else float("nan")
    t = slope / se_slope if se_slope else float("nan")
    return dict(slope=float(slope), se=float(se_slope), t=float(t), r2=float(r2),
                dof=dof, coeffs=[float(c) for c in coeffs])


def main():
    if not os.path.exists(TRACE):
        print("no trace persisted yet — runner has not written a sample.")
        return 1
    with open(TRACE) as fh:
        d = json.load(fh)

    s = d["samples"]
    if len(s) < 4:
        print(f"only {len(s)} samples — need >=4 to fit. Runner still warming up.")
        return 1

    t = np.array([p["t_sim"] for p in s], float)
    struct = np.array([p["structural"] for p in s], float)
    cons = np.array([p["cons_rel"] for p in s], float)
    S0 = d["structural_initial"]

    print("=" * 78)
    print("PO-2 · PHOSPHATE DEPLETION BOUND — offline scorer (A2.6 criteria, fixed pre-run)")
    print("=" * 78)
    print(f"  fraction (grounded)  : {d['fraction']}")
    print(f"  samples              : {len(s)}")
    print(f"  simulated span       : {t[0]:.1f} -> {t[-1]:.1f} s   ({t[-1]-t[0]:.1f} s)")
    print(f"  vs the 20 s window   : {(t[-1]-t[0])/20.0:.2f}x")
    print(f"  wall elapsed         : {d['wall_elapsed_s']/60:.1f} min")
    print()

    # ---- INVALID gate first: a depletion number off a broken ledger is unreadable ----
    max_cons = float(np.max(np.abs(cons)))
    print(f"  conservation guard   : max |dP|/P = {max_cons:.3e}  (eps = {EPS_REL:.0e})")
    if max_cons > EPS_REL:
        print()
        print(f"  VERDICT : INVALID")
        print(f"  because : conservation drifted past eps during the run; no depletion claim")
        print(f"            is readable from a broken ledger.")
        return 2

    lin = fit(t, struct, 1)
    quad = fit(t, struct, 2)
    diffs = np.diff(struct)
    monotonic_down = bool(np.all(diffs < 0))

    pct_per_s = lin["slope"] / S0 * 100.0
    print(f"  slope                : {lin['slope']:+.6e} pool-units/s  ({pct_per_s:+.6f} %/s)")
    print(f"  slope SE             : {lin['se']:.6e}")
    print(f"  t                    : {lin['t']:+.2f}   (dof {lin['dof']}, |t| crit {T_CRIT})")
    print(f"  R^2 (linear)         : {lin['r2']:.6f}")
    print(f"  monotonic decreasing : {monotonic_down}")
    print(f"  quadratic term       : {quad['coeffs'][0]:+.3e}  (R^2 {quad['r2']:.6f})")
    print()

    # 95% upper bound on |slope| ~ |slope| + 2*SE, and what it implies
    ub = abs(lin["slope"]) + T_CRIT * lin["se"]
    min_time_min = (S0 / ub) / 60.0 if ub > 0 else float("inf")

    if abs(lin["t"]) < T_CRIT:
        print(f"  VERDICT : NO_DRAIN_TO_BOUND")
        print(f"  because : |t| = {abs(lin['t']):.2f} < {T_CRIT}; no drain distinguishable")
        print(f"            from noise over this span.")
        print()
        print(f"  THE BOUND (this is the deliverable, not a proof of zero):")
        print(f"    95% upper bound on |slope| : {ub:.6e} pool-units/s "
              f"({ub/S0*100:.6f} %/s)")
        print(f"    => ANY depletion is SLOWER than {min_time_min:.0f} min simulated "
              f"({min_time_min/60:.1f} h)")
        print(f"    compare frac=0.02 (retired) : 34.4 min  -> bound is "
              f"{min_time_min/34.4:.0f}x slower")
        return 0

    if lin["t"] <= -T_CRIT and monotonic_down:
        print(f"  VERDICT : DRAIN_DETECTED")
        print(f"  because : t = {lin['t']:+.2f} <= -{T_CRIT} and the series is monotonic.")
        print(f"            PO2-9's 'valve closed at the grounded value' is OVERTURNED on")
        print(f"            this longer horizon. Time to depletion: "
              f"{(S0/abs(lin['slope']))/60:.1f} min simulated.")
        return 3

    print(f"  VERDICT : NONLINEAR")
    print(f"  because : slope is significant (t={lin['t']:+.2f}) but the series is not")
    print(f"            monotonic decreasing; neither flat nor a linear drain describes it.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
