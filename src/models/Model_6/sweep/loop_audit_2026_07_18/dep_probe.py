#!/usr/bin/env python3
"""DIAGNOSTIC: r's dependence on N and spacing; and the measured ca_open ceiling."""
import sys, os
import numpy as np
M6 = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6/src/models/Model_6"
sys.path.insert(0, M6)
from model6_parameters import (Model6Parameters, P_BASAL_W,
                               bose_einstein_occupation, hbar)

bp = Model6Parameters().dendritic_backbone
P_c = bose_einstein_occupation(bp.omega_0)*hbar*(2*np.pi*bp.omega_0)**2/bp.Q
pam = bp.p_active_max_W
LAM = 5.0

print(f"P_c={P_c*1e15:.2f}fW  P_BASAL={P_BASAL_W*1e15:.2f}fW  "
      f"p_active_max={pam*1e15:.0f}fW  lambda={LAM}um  Q={bp.Q}")

# analytic steady-state ca_open from calcium_system.py:96-105 (alpha=1e3,beta=2e3)
def ca_ss(v):
    P = 1.0/(1.0+np.exp(-(v-(-0.030))/0.012))
    a, b = 1e3*P, 2e3*(1-P)
    return a/(a+b)
print("\nSteady-state ca_open_fraction (analytic, calcium_system.py:96-124):")
for v in [-70e-3, -50e-3, -40e-3, -30e-3, -10e-3]:
    print(f"   V={v*1e3:6.1f}mV  ca_open_ss={ca_ss(v):.4f}")

# central-synapse row-sum for a linear chain of N at given spacing
def rowsum(N, sp):
    idx = np.arange(N); c = N//2
    return np.exp(-np.abs(idx-c)*sp/LAM).sum()

print("\nCentral-synapse W row-sum  ->  critical drive d* = (P_c-P_BASAL)/(p_act_max*rowsum)")
print(f"{'N':>4} " + " ".join(f"{s:>13}" for s in ["sp=0.5um","sp=1um","sp=2um","sp=5um"]))
for N in [1, 2, 4, 7, 10, 20, 40, 100]:
    cells = []
    for sp in [0.5, 1.0, 2.0, 5.0]:
        rs = rowsum(N, sp)
        ds = (P_c-P_BASAL_W)/(pam*rs)
        cells.append(f"{rs:5.1f}/d*{ds:5.3f}")
    print(f"{N:>4} " + " ".join(f"{c:>13}" for c in cells))

print("\nMax achievable drive d = E_inv_max * ca_open_ss(V):")
for v, lab in [(-40e-3, "spatial-discovery peak"), (-10e-3, "theta-burst peak (instantaneous)")]:
    c = ca_ss(v)
    for e in [0.0005, 0.1, 0.5, 0.74, 1.0]:
        print(f"   {lab:34s} E_inv={e:5.3f} ca={c:.4f} -> d={e*c:.4f}")
    print()

print("Theta-burst DUTY CYCLE (coherence_fragmentation_probe.py:196-198): "
      "4 spikes x 2ms depol per 125ms theta = 8/125 = 6.4% at -10mV")
print(f"   duty-averaged ca_open = 0.064*{ca_ss(-10e-3):.4f} + 0.936*{ca_ss(-70e-3):.4f} "
      f"= {0.064*ca_ss(-10e-3)+0.936*ca_ss(-70e-3):.4f}")
