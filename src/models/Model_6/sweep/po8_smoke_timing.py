#!/usr/bin/env python3
"""PO-8 smoke + timing probe. NOT a measurement — sizes the readout-time sweep.

Runs ONE free-running draw (NO seed) on the Unit-18 rig, snapshots the correlated-domain
partition at a few readout times WITHIN the single run (read-only snapshots do not perturb
the trajectory), and prints wall-clock per second-of-sim so I can size the full sweep.
Reuses the Unit-18 metric verbatim by import — no reimplementation.
"""
import sys, os, time, math
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)
import logging; logging.disable(logging.INFO)
import numpy as np

# reuse the Unit-18 metric verbatim
from po7_unit18_correlation_domains import (
    build_weighted_graph, bounded_dijkstra, connected_components, D_MAX)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease

N_SYN, SPACING = 7, 1.0
DT = 5e-3
VOLT = -40e-3
ACT = 0.95
SNAP_TIMES = [2.0, 5.0, 10.0]   # short probe only — size the full sweep


def domain_summary(tr):
    nodes, adj, edges, max_cross_F = build_weighted_graph(tr)
    V = len(nodes)
    if V == 0:
        return dict(V=0, edges=0, comps=0, lg_frac=0.0, mean_dom=0.0, maxF=0.0,
                    meanPS=0.0, intraF_med=0.0, crossF_med=0.0, n_cross=0)
    comp_sizes = connected_components(nodes, edges.keys())
    largest = comp_sizes[0] if comp_sizes else 0
    S = []
    for u in nodes:
        dist = bounded_dijkstra(u, adj)
        S.append(sum(math.exp(-d) for d in dist.values()))
    S = np.asarray(S, float)
    # channel fidelity split (read directly off the tracker)
    intraF = [float(v) for v in tr.intra_synapse_bonds_cache.values()]
    crossF = [float(v) for k, v in tr.cross_synapse_bonds.items()
              if float(v) > 0.5 and k[0][0] != k[1][0]]
    ps = [float(d['P_S']) for d in tr.all_dimers]
    return dict(V=V, edges=len(edges), comps=sum(1 for s in comp_sizes if s >= 2),
                lg_frac=largest / V, mean_dom=float(S.mean()), maxF=max_cross_F,
                meanPS=float(np.mean(ps)) if ps else 0.0,
                intraF_med=float(np.median(intraF)) if intraF else 0.0,
                crossF_med=float(np.median(crossF)) if crossF else 0.0,
                n_cross=len(crossF))


def main():
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)  # NO seed
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker
    rel = PresynapticRelease(None)

    t = 0.0
    t_max = SNAP_TIMES[-1]
    nsteps = int(round(t_max / DT))
    snap_steps = {int(round(x / DT)): x for x in SNAP_TIMES}
    peak_eta = 0.0
    wall0 = time.time()
    for i in range(1, nsteps + 1):
        g = rel.step(ACT, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        e = max((float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses), default=0.0)
        peak_eta = max(peak_eta, e)
        if i in snap_steps:
            d = domain_summary(tr)
            wall = time.time() - wall0
            print(f"t={snap_steps[i]:6.1f}s  wall={wall:6.1f}s  peak_eta={peak_eta:.3f}  "
                  f"V={d['V']:4d} E={d['edges']:5d} comps>=2={d['comps']:3d} "
                  f"lg_frac={d['lg_frac']:.3f} mean_dom={d['mean_dom']:7.1f} "
                  f"meanPS={d['meanPS']:.3f} intraF_med={d['intraF_med']:.3f} "
                  f"crossF_med={d['crossF_med']:.3f} n_cross={d['n_cross']}")
            sys.stdout.flush()
    wall = time.time() - wall0
    print(f"\nTOTAL: {t_max:.0f}s sim in {wall:.1f}s wall  =>  {wall/t_max:.2f} wall-s per sim-s")


if __name__ == "__main__":
    main()
