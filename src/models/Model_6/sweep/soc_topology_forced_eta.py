#!/usr/bin/env python3
"""
SOC topology — FORCED-eta probe (Stage 2, accelerated).

The open question is TOPOLOGICAL (when the pump is ignited, does the cross-
synapse graph form loops?), not "how long does E_invasion take to climb."
Stage 1 already proved eta is naturally reachable (d* ~ 0.087 in-cluster, ceiling
~0.74). So we clamp eta into the ignited regime from t=0 and measure the
downstream cross-synapse topology on a SHORT window — decoupling the slow actin
ramp (and the O(n^2) intra-blob explosion) from the topology question.

HONEST CAVEAT: eta is CLAMPED here, not naturally climbed. This is a valid probe
of the downstream topology BECAUSE Stage 1 established eta is naturally reachable;
it is not a claim about natural emergence timing. betti0_cross / betti1_cross are
read on the synapse-quotient graph (Werner-cleared cross-bonds only).

Prediction (from k_cross ∝ √(η_i η_j)·w_spatial·P_S² and F_werner = P_S²·w_spatial):
  within-cluster (≈1µm, w≈0.82): F≈0.82 > 0.5  -> quotient EDGE
  cross-cluster (15µm, w≈0.05): F≈0.05 < 0.5  -> NO edge
  => betti0_cross -> 2 (two synapse-clusters); betti1_cross > 0 iff the 4 nodes
     in a cluster cross-bond into cycles.
"""
import sys, os, time, types, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soc_topology_stage2 import build_two_cluster_network, pump_probe


def clamp_eta_factory(target_eta):
    def _clamped(self):
        for s in self.synapses:
            s.set_backbone_condensation_eta(target_eta)
    return _clamped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.26)   # natural eta at d≈0.15
    ap.add_argument("--seconds", type=float, default=0.08)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--log-every", type=float, default=0.02)
    args = ap.parse_args()

    t0 = time.time()
    net = build_two_cluster_network()
    # Clamp the pump into the ignited regime.
    net._update_backbone_field = types.MethodType(clamp_eta_factory(args.eta), net)

    spike_period, depol_dur, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * spike_period, 0.125
    n_steps = int(args.seconds / args.dt)
    next_log = 0.0
    print(f"forced eta={args.eta} on 8 synapses (two clusters); "
          f"driving {args.seconds}s bursts...")
    for k in range(n_steps):
        t = k * args.dt
        phase = t % theta_period
        v = (-10e-3 if (phase < burst_active and (phase % spike_period) < depol_dur)
             else -70e-3)
        net.step(args.dt, {"voltage": v, "reward": False})
        if t >= next_log:
            m = net.get_experimental_metrics()
            tr = net.entanglement_tracker
            n_cross = len(tr.cross_synapse_bonds)
            n_cross_cleared = sum(1 for f in tr.cross_synapse_bonds.values()
                                  if f > tr.WERNER_ENTANGLEMENT_BOUND)
            print(f"  t={t:5.3f}s  dimers={m.get('total_dimers',0):5.0f}  "
                  f"cross_bonds={n_cross:4d} (Werner-cleared {n_cross_cleared:3d})  "
                  f"betti0_cross={m.get('betti0_cross',0):2d}  "
                  f"betti1_cross={m.get('betti1_cross',0):3d}", flush=True)
            next_log += args.log_every

    # Final honest read with the coboundary cross-check.
    from entanglement_topology import compute_synapse_quotient_betti
    tr = net.entanglement_tracker
    q = compute_synapse_quotient_betti(tr.all_dimers, tr.cross_synapse_bonds,
                                       werner_bound=tr.WERNER_ENTANGLEMENT_BOUND,
                                       crosscheck=True)
    print(f"\n[{time.time()-t0:.0f}s] FINAL synapse-quotient topology:")
    print(f"  betti0_cross (synapse-clusters) = {q.betti0}")
    print(f"  betti1_cross (cross-synapse loops) = {q.betti1}")
    print(f"  quotient nodes/edges = {q.n_nodes}/{q.n_edges}   "
          f"component_sizes={q.component_sizes}  cyc/comp={q.cycles_per_component}")
    print(f"  crosscheck_ok = {q.crosscheck_ok}")
    verdict = ("NO cross-synapse loops yet (forest/blob)" if q.betti1 == 0
               else f"CROSS-SYNAPSE LOOPS PRESENT — betti1_cross={q.betti1}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
