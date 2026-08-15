#!/usr/bin/env python3
"""
Scorer for f3_delay_sweep — the temporal-credit CURVE under the corrected mechanism.

Reports, per delay:
  * commit probability for each arm (quantum-undoped / classical-base / quantum-Li7)
  * a two-sided PERMUTATION null on quantum-undoped vs each control arm
  * P_S at reward (the tag's remaining coherence) and the Werner floor crossing

TWO RATES ARE REPORTED, and the distinction matters:
  - UNCONDITIONAL commit rate (the pre-registered primary): includes trials where the brief eligibility drive
    happened to form NO dimers at all (P_S sits at the 0.25 thermal floor, n_dimers=0). Those trials cannot
    credit because there was never a tag — a real source of stochasticity, not a failure of the readout.
  - CONDITIONAL rate | tag formed (secondary diagnostic): restricted to trials that actually built a tag
    (n_dimers > 0). This isolates "was the tag READ?" from "did a tag EXIST?".
Both are shown so neither can be quietly cherry-picked.
"""
import json
import os
import sys

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "f3_delay_sweep", "runs.jsonl")
WERNER = 1.0 / np.sqrt(2)


def perm_p(a, b, n_perm=20000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def main():
    if not os.path.exists(RESULTS):
        print(f"no results at {RESULTS}"); return 1
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    arms = ["quantum-undoped", "classical-base", "quantum-Li7"]
    delays = sorted({r["delay"] for r in rows})

    print("=" * 104)
    print("F3 DELAY SWEEP — temporal-credit curve under the CORRECTED mechanism (supersedes F3-b/F3-c)")
    print(f"n runs = {len(rows)} | Werner floor = {WERNER:.3f} | commit = the coherence-gated CaMKII-GluN2B latch")
    print("=" * 104)
    hdr = (f"  {'delay':>6} | {'quantum':>18} {'classical':>18} {'Li7':>18} | "
           f"{'P_S@rew(q)':>10} | {'p(q vs cls)':>11} {'p(q vs Li7)':>11}")
    print(hdr); print("-" * len(hdr))

    for d in delays:
        cell = {}
        for a in arms:
            rs = [r for r in rows if r["delay"] == d and r["arm"] == a]
            unc = [1.0 if r["committed"] else 0.0 for r in rs]
            withtag = [1.0 if r["committed"] else 0.0 for r in rs if r.get("n_dimers", 0) > 0]
            cell[a] = (unc, withtag, rs)

        def fmt(a):
            unc, wt, _ = cell[a]
            if not unc:
                return f"{'--':>18}"
            return f"{np.mean(unc):>8.2f} ({np.mean(wt) if wt else float('nan'):>4.2f}|tag)"

        q_unc = cell["quantum-undoped"][0]
        ps_q = [r["ps_reward"] for r in cell["quantum-undoped"][2]]
        p_cls = perm_p(q_unc, cell["classical-base"][0])
        p_li7 = perm_p(q_unc, cell["quantum-Li7"][0])
        print(f"  {d:>6.1f} | {fmt('quantum-undoped')} {fmt('classical-base')} {fmt('quantum-Li7')} | "
              f"{np.mean(ps_q) if ps_q else float('nan'):>10.3f} | {p_cls:>11.4f} {p_li7:>11.4f}")

    print("-" * len(hdr))
    print("  cells show: UNCONDITIONAL commit rate (CONDITIONAL on a tag having formed)")

    # pooled across delays >= 5 s: the regime where the classical trace is dead by construction
    long_d = [d for d in delays if d >= 5.0]
    if long_d:
        q = [1.0 if r["committed"] else 0.0 for r in rows
             if r["arm"] == "quantum-undoped" and r["delay"] in long_d]
        c = [1.0 if r["committed"] else 0.0 for r in rows
             if r["arm"] == "classical-base" and r["delay"] in long_d]
        l7 = [1.0 if r["committed"] else 0.0 for r in rows
              if r["arm"] == "quantum-Li7" and r["delay"] in long_d]
        print(f"\n  POOLED at delays >= 5 s (the temporal-gap regime), n={len(q)}/{len(c)}/{len(l7)}:")
        print(f"    quantum-undoped commit = {np.mean(q):.3f}")
        print(f"    classical-base  commit = {np.mean(c):.3f}   permutation p = {perm_p(q, c):.4f}")
        print(f"    quantum-Li7     commit = {np.mean(l7):.3f}   permutation p = {perm_p(q, l7):.4f}")
        print("\n  => the temporal-gap claim needs quantum > classical at these delays; the isotope claim needs")
        print("     quantum > Li7. Both are scored against permutation nulls; a null is a reportable result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
