#!/usr/bin/env python3
"""
Scorer for f4_specificity — does credit land on the DRIVEN synapses?

Primary statistic: commit rate among DRIVEN vs UNDRIVEN synapses in the REWARD arm, tested against a
WITHIN-RUN label-permutation null. The permutation shuffles which synapses are called "driven" inside each
run, holding that run's total number of commitments fixed, so the null asks exactly one question: given how
many synapses committed, is it the DRIVEN ones? That controls for run-to-run differences in overall
excitability, which a pooled shuffle would not.

Also reported:
  - the NO-REWARD arm (reward necessity at network scale — the single-synapse F3 result must survive here)
  - precision / recall of "committed" as a predictor of "driven"
  - tag occupancy: dimers on driven vs undriven synapses (the specificity of the SUBSTRATE, which should be
    perfect by construction — undriven synapses form no tag at all — so any loss of specificity in the
    COMMIT is introduced downstream, by the readout or the cascade, not by the eligibility)
A null (driven ≈ undriven) is a real result: it would mean a global reward smears credit across the segment.
"""
import json
import os
import sys

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "f4_specificity", "runs.jsonl")


def within_run_perm_p(runs, n_perm=20000, seed=1):
    """Null: within each run, reassign the driven labels at random (same count), recompute the contrast."""
    rng = np.random.default_rng(seed)

    def contrast(label_sets):
        d, u = [], []
        for r, dset in zip(runs, label_sets):
            for i in range(r["n_syn"]):
                (d if i in dset else u).append(1.0 if r["committed"][i] else 0.0)
        if not d or not u:
            return np.nan
        return np.mean(d) - np.mean(u)

    obs = contrast([set(r["driven"]) for r in runs])
    if not np.isfinite(obs):
        return obs, np.nan
    ge = 0
    for _ in range(n_perm):
        shuffled = [set(rng.choice(r["n_syn"], size=len(r["driven"]), replace=False).tolist()) for r in runs]
        c = contrast(shuffled)
        if np.isfinite(c) and abs(c) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def arm_stats(runs):
    d, u, td, tu = [], [], [], []
    for r in runs:
        dset = set(r["driven"])
        for i in range(r["n_syn"]):
            c = 1.0 if r["committed"][i] else 0.0
            t = r["tags"][i]
            if i in dset:
                d.append(c); td.append(t)
            else:
                u.append(c); tu.append(t)
    return d, u, td, tu


def main():
    if not os.path.exists(RESULTS):
        print(f"no results at {RESULTS}"); return 1
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    rw = [r for r in rows if r["rewarded"]]
    nr = [r for r in rows if not r["rewarded"]]

    print("=" * 96)
    print("F4 SPECIFICITY — does the RIGHT synapse get the credit? (global reward x local eligibility)")
    print(f"runs: {len(rw)} rewarded, {len(nr)} no-reward | {rows[0]['n_syn']} synapses, "
          f"{len(rows[0]['driven'])} driven per run")
    print("=" * 96)

    for name, runs in (("REWARD", rw), ("NO-REWARD (control)", nr)):
        if not runs:
            continue
        d, u, td, tu = arm_stats(runs)
        print(f"\n  [{name}]")
        print(f"    tag occupancy   driven {np.mean(td):8.1f} dimers | undriven {np.mean(tu):8.1f} dimers")
        print(f"    commit rate     driven {np.mean(d):8.3f} (n={len(d)}) | undriven {np.mean(u):8.3f} (n={len(u)})")
        if name == "REWARD":
            obs, p = within_run_perm_p(runs)
            tp = sum(d); fp = sum(u)
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec = tp / len(d) if d else float("nan")
            print(f"    contrast        {obs:+.3f}   within-run permutation p = {p:.4f}")
            print(f"    precision (committed are driven) = {prec:.3f} | recall (driven commit) = {rec:.3f}")
            specific = np.isfinite(p) and p < 0.05 and obs > 0
            print(f"\n  => {'CREDIT IS SYNAPSE-SPECIFIC (p<0.05)' if specific else 'NOT shown specific at this power — report it; do not tune'}")
            if (sum(d) + sum(u)) == 0:
                print("     NB: nothing committed anywhere in the reward arm — the contrast is vacuous,")
                print("     not a refutation. Check tag formation / commit machinery before interpreting.")
    if nr:
        d, u, _, _ = arm_stats(nr)
        tot = sum(d) + sum(u)
        print(f"\n  reward necessity: {int(tot)} commitments across the entire NO-REWARD arm "
              f"({'PASS — reward is necessary at network scale' if tot == 0 else 'commitments occurred without reward — investigate'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
