#!/usr/bin/env python3
"""
Is there a DOWNSTREAM isotope lever to find? (cheap single-synapse check, no O(n²) network tracker.)

F2-c showed the isotope CANNOT move the measurement TRIGGER (it fires ~90 ms, before ⁷Li's 14 s
decoherence acts). The open question for the FULL run: does the isotope move CONSOLIDATION downstream,
via P_S-dependent dissolution over the coherence window? This checks the PREREQUISITE cheaply: does the
mean singlet probability P_S(t) actually diverge between ⁶Li and ⁷Li over the hold window at all? If
Li6 ≈ Li7 here, the expensive network C2 has nothing to find.

Drives one synapse to build dimers (write), then HOLDS and records mean P_S(t) under None/Li6/Li7.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
WRITE_S = 3.0
HOLD_SAMPLES_S = [0, 1, 2, 5, 10, 15, 20, 30]


def run(dopant):
    np.random.seed(0)
    p = Model6Parameters(); p.em_coupling_enabled = True
    p.environment.dopant = dopant
    syn = Model6QuantumSynapse(p); syn.set_microtubule_invasion(True)
    # write: depolarize to build calcium → dimers
    for _ in range(int(WRITE_S / DT)):
        syn.step(DT, {"voltage": -40e-3})
    ps0 = getattr(syn, "_mean_singlet_prob", None)
    ndim = len(syn.dimer_particles.dimers)
    # hold at rest; sample P_S at the registered times
    traj = {}
    t = 0.0; nxt = list(HOLD_SAMPLES_S)
    max_t = HOLD_SAMPLES_S[-1]
    while t <= max_t + 1e-9:
        while nxt and t >= nxt[0] - 1e-9:
            traj[nxt.pop(0)] = float(getattr(syn, "_mean_singlet_prob", np.nan))
        syn.step(DT, {"voltage": -70e-3}); t += DT
    committed = bool(getattr(syn, "_camkii_committed", False))
    return dict(dopant=str(dopant), n_dimers_after_write=ndim, ps_after_write=ps0,
                traj=traj, committed=committed)


if __name__ == "__main__":
    print(f"{'dopant':>6} {'ndim':>5} " + " ".join(f"P_S@{s}s" for s in HOLD_SAMPLES_S) + "  committed")
    out = {}
    for d in [None, "Li6", "Li7"]:
        r = run(d); out[str(d)] = r
        vals = " ".join(f"{r['traj'].get(s, float('nan')):6.3f}" for s in HOLD_SAMPLES_S)
        print(f"{str(d):>6} {r['n_dimers_after_write']:>5}  {vals}   {r['committed']}")
    # the prerequisite verdict
    li6, li7 = out["Li6"]["traj"], out["Li7"]["traj"]
    diffs = {s: li6[s] - li7[s] for s in HOLD_SAMPLES_S if s in li6 and s in li7}
    max_gap = max(abs(v) for v in diffs.values()) if diffs else 0.0
    at = max(diffs, key=lambda s: abs(diffs[s])) if diffs else None
    print(f"\nmax |P_S(⁶Li) − P_S(⁷Li)| over the window = {max_gap:.3f}  at t={at}s")
    print("VERDICT:", "downstream isotope lever EXISTS (P_S diverges over the window) — full C2 worth running"
          if max_gap > 0.05 else
          "NO downstream P_S divergence — full C2 network run has nothing to find (report + reconsider)")
