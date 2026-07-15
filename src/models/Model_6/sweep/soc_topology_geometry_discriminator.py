#!/usr/bin/env python3
"""
SOC topology — GEOMETRY DISCRIMINATOR (Stage 3, the decisive test).

Stage 2 found betti1_cross=6 on two K4 clusters, but a K4 is COMPLETE only
because 4 nodes ~1um apart all couple — so the loop count might be trivial
small-cluster completeness rather than genuine topology. This test settles it:

  CHAIN of 6 synapses at 2.5um  -> nearest-neighbour edges only (w=exp(-0.5)=0.61
     clears Werner; next-nearest 5um w=0.37 does not) => PATH graph => betti1=0
  RING of 6 synapses, adjacent chord 2.5um -> same near-neighbour coupling, but
     the ends close => 6-CYCLE => betti1=1

Chain and ring have the SAME node count and the SAME per-node coupling; they
differ by exactly ONE edge (the closing one). If betti1_cross reads 0 for the
chain and 1 for the ring, the instrument is tracking real loops, not node count.
Pump is clamped to the ignited regime (eta=0.26), same as Stage 2.
"""
import sys, os, time, types, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from entanglement_topology import compute_synapse_quotient_betti


def chain_positions(n=6, spacing_um=2.5):
    p = np.zeros((n, 3)); p[:, 0] = np.arange(n) * spacing_um
    return p


def ring_positions(n=6, adj_um=2.5):
    # adjacent chord = 2 R sin(pi/n) = adj_um  => R = adj_um / (2 sin(pi/n))
    R = adj_um / (2.0 * np.sin(np.pi / n))
    ang = 2.0 * np.pi * np.arange(n) / n
    p = np.zeros((n, 3)); p[:, 0] = R * np.cos(ang); p[:, 1] = R * np.sin(ang)
    return p


def build(positions):
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=len(positions), pattern="clustered", spacing_um=1.0)
    net.positions = positions
    net.distances = net._compute_distances()
    net.coupling_weights = net._compute_coupling_weights()
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def clamp_eta(target):
    def _c(self):
        for s in self.synapses:
            s.set_backbone_condensation_eta(target)
    return _c


def run_geometry(name, positions, eta=0.26, seconds=0.08, dt=0.001):
    net = build(positions)
    net._update_backbone_field = types.MethodType(clamp_eta(eta), net)
    spike_period, depol_dur, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * spike_period, 0.125
    for k in range(int(seconds / dt)):
        t = k * dt
        phase = t % theta_period
        v = (-10e-3 if (phase < burst_active and (phase % spike_period) < depol_dur)
             else -70e-3)
        net.step(dt, {"voltage": v, "reward": False})
    tr = net.entanglement_tracker
    q = compute_synapse_quotient_betti(tr.all_dimers, tr.cross_synapse_bonds,
                                       werner_bound=tr.WERNER_ENTANGLEMENT_BOUND,
                                       crosscheck=True)
    # Which synapse-pairs actually cleared the Werner bound (the quotient edges).
    gid_syn = {d['global_id']: d['synapse_idx'] for d in tr.all_dimers}
    syn_edges = set()
    for (i, j), f in tr.cross_synapse_bonds.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                syn_edges.add((min(a, b), max(a, b)))
    print(f"{name:>6}: betti0_cross={q.betti0} betti1_cross={q.betti1}  "
          f"quotient V/E={q.n_nodes}/{q.n_edges}  ok={q.crosscheck_ok}")
    print(f"         synapse edges = {sorted(syn_edges)}")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.08)
    args = ap.parse_args()
    t0 = time.time()
    print("=== GEOMETRY DISCRIMINATOR (eta=0.26 ignited) ===")
    qc = run_geometry("CHAIN", chain_positions(6, 2.5), seconds=args.seconds)
    qr = run_geometry("RING",  ring_positions(6, 2.5),  seconds=args.seconds)
    print(f"\n[{time.time()-t0:.0f}s] PREDICTION: chain betti1=0, ring betti1=1")
    ok = (qc.betti1 == 0 and qr.betti1 >= 1)
    print("VERDICT:", "INSTRUMENT TRACKS REAL TOPOLOGY (chain=tree, ring=loop)"
          if ok else "MISMATCH — instrument does NOT cleanly discriminate; investigate")


if __name__ == "__main__":
    main()
