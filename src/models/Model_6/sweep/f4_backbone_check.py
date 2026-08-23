#!/usr/bin/env python3
"""De-risk check for the FULL-SYSTEM specificity run: can the tryptophan/MT backbone actually condense?

F4-a ran with eta == 0.0000 (backbone inert, local tags only) because a 2 s drive left actin_enlargement at
0.099 against an invasion threshold of 0.1, so E_invasion was exactly 0 and the active metabolic term vanished.
Cross-synapse bonds are impossible in that regime. Before paying for a long full-system batch, drive ALL
synapses (maximum aggregate power -- if it cannot cross here it cannot cross with a driven subset) and report
E_invasion, eta, and the cross-synapse bond count as they build. A null here is decisive and cheap.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import logging; logging.disable(logging.INFO)
import numpy as np
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

DT = 5e-3
N_SYN = 6
DRIVE_V = -20e-3          # plateau-level depolarization (strong drive)

def main():
    np.random.seed(0)
    p = Model6Parameters(); p.em_coupling_enabled = True; p.multi_synapse_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, use_correlated_sampling=True)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    print(f"[{time.strftime('%H:%M:%S')}] backbone check: {N_SYN} synapses ALL driven at {DRIVE_V*1e3:.0f} mV", flush=True)
    print("   t(s) | E_inv_max  eta_max | dimers | xbonds(any) xbonds(F>0.5)", flush=True)
    t = 0.0
    while t < 40.0:
        for _ in range(int(1.0 / DT)):
            net.step(DT, {"per_synapse": [{"voltage": DRIVE_V}] * N_SYN, "reward": False})
        t += 1.0
        if t in (1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40):
            ei = max(getattr(s.spine_plasticity, "E_invasion", 0.0) for s in net.synapses)
            eta = max(getattr(s, "_backbone_eta", 0.0) for s in net.synapses)
            nd = sum(len(s.dimer_particles.dimers) for s in net.synapses)
            tr = net.entanglement_tracker
            xb = len(tr.cross_synapse_bonds)
            xw = sum(1 for F in tr.cross_synapse_bonds.values() if F > 0.5)
            flag = "  <== BACKBONE CONDENSED" if eta > 0 else ""
            print(f"   {t:4.0f} | {ei:9.3f} {eta:8.4f} | {nd:6d} | {xb:11d} {xw:13d}{flag}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] CHECK COMPLETE", flush=True)

if __name__ == "__main__":
    main()
