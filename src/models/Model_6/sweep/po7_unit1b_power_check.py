#!/usr/bin/env python3
"""
PO-7 UNIT 1b — POWER CHECK for the scored keystone test. NOT a scored run, no verdict.

Unit 1 measures only 2-3 cross-synapse provenance edges per run at the pre-registered 0.2um
spacing. Unit 2's statistic is Newman modularity on a 6-node synapse graph; with 2-3 edges that
statistic is noise-dominated, and a "decomposition null" verdict from it would OVERCLAIM a
negative -- asserting the mechanism does not compute when the measurement could not have
detected it either way. See coordination notes.md Q0.

This probe asks one question: does cross-synapse edge count GROW with simulation time, or
saturate? It scores nothing and has no verdict function.

Two quantities, because they are different objects and the choice between them must be made
BEFORE scoring, not after seeing a result:
  - INSTANTANEOUS: len(_prov_bonds) cross pairs alive at a given moment (bonds are pruned on
    coherence death / dimer loss). This is what Unit 2 currently snapshots at the end of a run.
  - CUMULATIVE: distinct cross-synapse pairs that have EVER bonded during the run. This is the
    "eligibility = persisting topology" reading (quantum-system-canonical SS5) and carries far
    more signal, but it is a different object and adopting it is a design decision to be
    recorded, not a silent power grab.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

DT = 0.005
TRACKER_EVERY = 10
T_MAX = 2.0
SPACING = 0.2
SEED = 31337
N_SYN = 6
WERNER = 0.5


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters()
    p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    tr = net.entanglement_tracker
    tr.provenance_network = True

    rel = PresynapticRelease(seed=SEED)
    ever_cross, ever_pairs, trace = set(), set(), []
    for i in range(int(round(T_MAX / DT))):
        g = rel.step(0.95, DT)
        for s in net.synapses:
            s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))
            inst = 0
            for (a, b), f in tr._prov_bonds.items():
                if f > WERNER and a[0] != b[0]:
                    inst += 1
                    ever_cross.add((a, b))
                    ever_pairs.add((min(a[0], b[0]), max(a[0], b[0])))
            comps = tr._find_all_clusters()
            n_multi = sum(1 for c in comps if len({g0[0] for g0 in c}) >= 2)
            row = {'t_s': round((i + 1) * DT, 3), 'inst_cross': inst,
                   'cum_cross': len(ever_cross), 'synapse_pairs_ever': len(ever_pairs),
                   'n_multi': n_multi, 'n_prov_total': len(tr._prov_bonds),
                   'n_dimers': len(tr.all_dimers)}
            trace.append(row)
            print(f"t={row['t_s']:>5}s inst_cross={inst:>3} cum_cross={row['cum_cross']:>4} "
                  f"syn_pairs_ever={row['synapse_pairs_ever']:>2}/15 n_multi={n_multi:>3} "
                  f"prov_total={row['n_prov_total']:>4} dimers={row['n_dimers']}")
            sys.stdout.flush()

    path = os.path.join(SWEEP_DIR, 'po7_unit1b_power_results.json')
    with open(path, 'w') as f:
        json.dump({'spacing_um': SPACING, 'seed': SEED, 't_max_s': T_MAX, 'trace': trace}, f, indent=2)
    print(f"\nwrote {path}")
    print("\nREAD THIS AS: does cum_cross / synapse_pairs_ever keep RISING, or flatten? "
          "Rising => longer T_SIM buys real power for Unit 2. Flat => the layer's cross-synapse "
          "channel has a CEILING and cannot support the keystone test as built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
