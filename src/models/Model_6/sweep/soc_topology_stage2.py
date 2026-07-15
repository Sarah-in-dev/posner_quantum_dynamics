#!/usr/bin/env python3
"""
SOC experiment — STAGE 2 (expensive, gated on Stage 1): drive the two-cluster
network above the pump threshold and read betti1_cross (loops that SPAN
synapses) in the ignited regime.

Stage 1 (soc_pump_threshold_stage1.py) proved the pump ignites in-cluster at
drive d* ~ 0.087 (E_inv*ca_open), well below the honest ceiling ~0.74 — so this
regime is reachable. Here we actually run the physics: 8 synapses in two
clusters, sustained theta bursting so E_invasion climbs and ca_open stays high,
logging r/eta AND betti1_cross over time.

*** COMPUTE WALL ***  On the current O(n^2) bond recalc, ~0.375 s of sim took
~6 min at 5 synapses / 2400 dimers. E_invasion only reaches ~0.5 by ~30 s, so a
faithful ignition run is many hours and grows with dimer count. Options, in
order of preference:
  (1) extend the analytical-gap acceleration (run_theta_burst_45s.analytical_gap)
      to the DRIVE phase, or throttle the tracker recalc cadence;
  (2) cap dimers/bonds per spine (the intra-blob clique-fill is what makes it
      O(n^2) and it is topologically trivial anyway — betti1_cross ignores it);
  (3) run as a long background / Fargate job.
Do NOT read a short smoke run as the result — E_invasion will still be ~0.

Usage:
  python3 soc_topology_stage2.py --seconds 2      # smoke test (eta still ~0)
  python3 soc_topology_stage2.py --seconds 40     # real ignition run (hours)
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model6_parameters import (
    Model6Parameters, P_BASAL_W, bose_einstein_occupation, hbar,
)
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from soc_pump_threshold_stage1 import two_cluster_positions


def build_two_cluster_network():
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0

    net = MultiSynapseNetwork(n_synapses=8, pattern="clustered", spacing_um=1.0)
    # Override with the faithful two-cluster geometry, recompute coupling.
    net.positions = two_cluster_positions(per_cluster=4, sep_um=15.0, within_um=1.0)
    net.distances = net._compute_distances()
    net.coupling_weights = net._compute_coupling_weights()
    net.initialize(Model6QuantumSynapse, params)
    for syn in net.synapses:
        syn.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def pump_probe(net):
    """Read live r/eta the same way _update_backbone_field computes them."""
    bp = net.params.dendritic_backbone
    omega_ang = 2.0 * np.pi * bp.omega_0
    P_c = bose_einstein_occupation(bp.omega_0) * hbar * omega_ang**2 / bp.Q
    from model6_parameters import compute_metabolic_power
    p_met = np.array([
        compute_metabolic_power(getattr(s.spine_plasticity, 'E_invasion', 0.0),
                                s.calcium.channels.get_open_fraction(),
                                bp.p_active_max_W) for s in net.synapses])
    p_agg = P_BASAL_W + net.coupling_weights @ (p_met - P_BASAL_W)
    r = p_agg / P_c
    eta = np.where(r >= 1, (r - 1) / (r + 1), 0.0)
    e_inv = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0) for s in net.synapses])
    return float(r.mean()), float(eta.mean()), float(e_inv.mean())


def drive_bursts(net, dt, seconds, log_every_s=2.0):
    """Sustained theta bursting; log pump + betti1_cross periodically."""
    spike_period, depol_dur, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * spike_period, 0.125
    n_steps = int(seconds / dt)
    next_log = 0.0
    for k in range(n_steps):
        t = k * dt
        phase = t % theta_period
        if phase < burst_active:
            v = -10e-3 if (phase % spike_period) < depol_dur else -70e-3
        else:
            v = -70e-3
        net.step(dt, {"voltage": v, "reward": False})
        if t >= next_log:
            r, eta, e_inv = pump_probe(net)
            m = net.get_experimental_metrics()
            print(f"  t={t:6.2f}s  E_inv={e_inv:.3f}  r={r:.3f}  eta={eta:.4f}  "
                  f"cross_bonds={m.get('cross_synapse_bonds',0):4d}  "
                  f"betti1_cross={m.get('betti1_cross',0):3d}  "
                  f"betti0_cross={m.get('betti0_cross',0):2d}", flush=True)
            next_log += log_every_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.001)
    args = ap.parse_args()

    t0 = time.time()
    net = build_two_cluster_network()
    r0, eta0, _ = pump_probe(net)
    print(f"built two-cluster 8-synapse net; initial r={r0:.3f} eta={eta0:.4f}")
    print(f"driving {args.seconds}s of theta bursts (dt={args.dt})...")
    drive_bursts(net, args.dt, args.seconds)
    print(f"[{time.time()-t0:.0f}s] done. Read betti1_cross ABOVE, in rows where eta>0.")


if __name__ == "__main__":
    main()
