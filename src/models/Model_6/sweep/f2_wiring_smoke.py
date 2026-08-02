#!/usr/bin/env python3
"""F2 wiring smoke — proves the measurement TRIGGER is the spin-selective binding event, decoupled
from reward. Fast + deterministic (seeds the tracker; no 47-min model draw). Exercises the REAL
`_evaluate_coordinated_gate` / `_evaluate_independent_gate` / `_binding_measurement_fires` code.

Acceptance proven here (prereg PREREG_F2, control-ladder C1 + the decoupling core):
  1. reward=False still MEASURES when a coherent singlet cluster exists  (the F2 decoupling).
  2. the trigger rides on P_S: a fully-coherent cluster fires; a thermal-floor cluster (with an
     unlucky draw) does not → re-arms.
  3. reward=True with NO coherent cluster does NOT measure (reward cannot force a measurement).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multi_synapse_network import MultiSynapseNetwork


N_CLUSTER = 800   # realistic single-synapse coherent dimer cloud (step1 probe range 800–1600)


def _seed_tracker(net, p_s, gids=tuple(range(N_CLUSTER))):
    """Give the tracker one coherent cluster at singlet probability p_s (bypasses model dynamics)."""
    tr = net.entanglement_tracker
    tr.all_dimers = [{'global_id': g, 'P_S': p_s, 'synapse_idx': i} for i, g in enumerate(gids)]
    tr.collect_dimers = lambda *a, **k: None                       # no-op refresh
    tr._find_all_clusters = lambda: [set(gids)]                    # one connected component
    calls = {'n': 0}
    def _pqm(synapses):
        calls['n'] += 1
        tr._last_measurement = {'total_dimers': len(gids), 'total_bonds': 1,
                                'n_clusters_measured': 1, 'n_clusters_singlet': 1,
                                'singlet_outcomes': len(gids), 'committed_counts': np.zeros(len(net.synapses))}
        return np.zeros(len(net.synapses))
    tr.perform_quantum_measurement = _pqm
    return calls


def _fresh_net():
    net = MultiSynapseNetwork(n_synapses=2, pattern="linear", spacing_um=1.0)
    net._coordinated_measurement_performed = False
    for s in net.synapses:
        setattr(s, '_peak_calcium_uM', 5.0)   # calcium factor satisfied
    return net


def main():
    dt = 1e-3
    results = []

    # --- 1. reward=False + coherent cluster ⇒ MEASURES (the decoupling) ---
    np.random.seed(0)
    net = _fresh_net(); calls = _seed_tracker(net, p_s=1.0)
    net._evaluate_coordinated_gate({'reward': False}, dt)
    ok1 = calls['n'] == 1 and net._coordinated_measurement_performed
    results.append(("reward=False + coherent cluster ⇒ measurement fires", ok1))

    # --- 2. the trigger RIDES ON P_S: coherent fires ~always; thermal-floor materially less ---
    def fire_rate(p_s, n=120):
        hits = 0
        for sd in range(n):
            np.random.seed(sd)
            net = _fresh_net(); calls = _seed_tracker(net, p_s=p_s)
            net._evaluate_coordinated_gate({'reward': False}, dt)
            hits += (calls['n'] == 1)
        return hits / n
    r_coh, r_floor = fire_rate(1.0), fire_rate(0.25)
    ok2 = (r_coh > 0.99) and (r_floor < r_coh - 0.15)   # spin-0-selective: singlet drives the trigger
    results.append((f"trigger rides on P_S: fire-rate coherent={r_coh:.2f} > floor={r_floor:.2f}", ok2))

    # --- 3. reward=True but NO cluster ⇒ does NOT measure (reward cannot force it) ---
    np.random.seed(0)
    net = _fresh_net()
    tr = net.entanglement_tracker
    tr.collect_dimers = lambda *a, **k: None
    tr._find_all_clusters = lambda: []          # no coherent cluster
    tr.all_dimers = []
    calls3 = {'n': 0}
    tr.perform_quantum_measurement = lambda synapses: (calls3.__setitem__('n', calls3['n'] + 1) or np.zeros(2))
    net._evaluate_coordinated_gate({'reward': True}, dt)
    ok3 = calls3['n'] == 0
    results.append(("reward=True + NO cluster ⇒ no measurement (reward cannot force it)", ok3))

    # --- 4. independent control gate also decoupled (reward=False + cluster ⇒ measures) ---
    np.random.seed(0)
    net = _fresh_net(); net._independent_measurement_performed = False
    calls4 = _seed_tracker(net, p_s=1.0)
    # independent gate uses perform_independent_measurement — patch that too
    net.entanglement_tracker.perform_independent_measurement = net.entanglement_tracker.perform_quantum_measurement
    net._evaluate_independent_gate({'reward': False}, dt)
    ok4 = calls4['n'] == 1
    results.append(("independent control gate: reward=False + cluster ⇒ measures", ok4))

    print("=== F2 WIRING SMOKE ===")
    for msg, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    allok = all(ok for _, ok in results)
    print(f"\n  ALL: {'PASS — measurement trigger is the binding event, decoupled from reward' if allok else 'FAIL'}")
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
