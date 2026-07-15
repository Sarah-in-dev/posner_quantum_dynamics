#!/usr/bin/env python3
"""
Read the FIRST live Betti0/Betti1 of the cross-synapse entanglement graph.

Reuses make_network() + run_burst_traversal() from run_theta_burst_45s so the
drive is faithful, runs a short stimulation to form cross-synapse bonds, then
reads NetworkEntanglementTracker.compute_entanglement_topology(crosscheck=True).

Purpose: answer whether the current Werner-thresholded partition already carries
intra-component loops (Betti1 > 0) or is still a forest (Betti1 == 0, "armed but
not firing"). Betti1 is what entanglement-topology-measurement wants; if it is 0
today, persistent-homology vineyards have nothing to measure yet.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_theta_burst_45s import make_network, run_burst_traversal


def main():
    t0 = time.time()
    net = make_network()
    dt = 0.001

    # A couple of theta cycles is enough to spawn dimers and cross-synapse bonds.
    print("driving 3 theta cycles...", flush=True)
    run_burst_traversal(net, dt, n_theta_cycles=3)

    m = net.get_experimental_metrics()
    topo = net.entanglement_tracker.compute_entanglement_topology(crosscheck=True)

    print(f"\n[{time.time()-t0:.1f}s]  total_dimers={m['total_dimers']:.0f}  "
          f"cross_bonds={m['cross_synapse_bonds']}  "
          f"within_bonds={m['within_synapse_bonds']}  "
          f"total_bonds={m['total_bonds']}")
    print("--- Betti (Werner-thresholded entanglement graph) ---")
    print(f"  nodes (V)           = {topo.n_nodes}")
    print(f"  edges (E)           = {topo.n_edges}")
    print(f"  Betti0 (components) = {topo.betti0}")
    print(f"  Betti1 (loops)      = {topo.betti1}")
    print(f"  component_sizes     = {topo.component_sizes}")
    print(f"  cycles_per_comp     = {topo.cycles_per_component}")
    print(f"  rank(delta)         = {topo.rank_delta}")
    print(f"  crosscheck_ok       = {topo.crosscheck_ok}")

    # Sanity: Betti0 must equal the union-find cluster count.
    n_clusters = len(net.entanglement_tracker._find_all_clusters())
    print(f"\n  _find_all_clusters() = {n_clusters}  "
          f"(matches Betti0: {n_clusters == topo.betti0})")

    verdict = ("FOREST — armed but not firing (H1 detection has nothing to see yet)"
               if topo.betti1 == 0 else
               f"LOOPS PRESENT — Betti1={topo.betti1} closed entanglement path(s)")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
