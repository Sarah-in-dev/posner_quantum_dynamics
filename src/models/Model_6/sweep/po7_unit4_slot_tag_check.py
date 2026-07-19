#!/usr/bin/env python3
"""
PO-7 UNIT 4 — validate the WHICH-SPIN provenance tag (advisor R4, step 1).

GATE (must pass, else the tag is not admissible):
  provenance-ON REGRESSION. The slot tag is side-band: nothing reads it, so turning
  provenance ON must reproduce Unit 1's recorded row EXACTLY at the same config
  (spacing 0.2 um, seed 31337, 6 synapses, T_SIM 0.4): cross=2, intra=215.
  Bit-identity of the OFF path is necessary but NOT sufficient here — the tag lives on
  the ON path, so the ON path is what has to be shown unchanged.

THEN, what the tag buys immediately — the monogamy audit on the provenance graph.
Each Ca6(PO4)4 dimer has 4 x 31P spins, and a bond consumes ONE spin at each endpoint.
So a given (dimer, spin) may mediate AT MOST ONE bond. With the slot now recorded we can
count, for the first time, how many bonds each individual spin is being asked to mediate.
This is the bound the advisor says is derivable rather than assumed.
"""
import sys, os, json, logging
from collections import Counter
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

T_SIM, DT = 0.4, 0.005
TRACKER_EVERY = 10
SPACING, SEED, N_SYN = 0.2, 31337, 6
EXPECTED = {'cross': 2, 'intra': 215}          # Unit 1, spacing=0.2 seed=31337


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    tr = net.entanglement_tracker
    tr.provenance_network = True

    rel = PresynapticRelease(seed=SEED)
    for i in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        for s in net.synapses:
            s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))

    cross = sum(1 for (a, b) in tr._prov_bonds if a[0] != b[0])
    intra = len(tr._prov_bonds) - cross
    print("=" * 88)
    print("GATE — provenance-ON regression (the tag must change nothing)")
    print("=" * 88)
    print(f"  cross = {cross:>4}   expected {EXPECTED['cross']}")
    print(f"  intra = {intra:>4}   expected {EXPECTED['intra']}")
    gate = (cross == EXPECTED['cross'] and intra == EXPECTED['intra'])
    print(f"  ON-PATH UNCHANGED: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  => INVALID: the side-band tag perturbed the ON path. Do not build on it.")
        return 1

    # ---- tag integrity ----
    n_bonds = len(tr._prov_bonds)
    tagged = sum(1 for k in tr._prov_bonds if k in tr._prov_bond_spins)
    spins_seen = Counter()
    for (sa, sb) in tr._prov_bond_spins.values():
        spins_seen[sa] += 1; spins_seen[sb] += 1
    print("\n" + "=" * 88)
    print("TAG INTEGRITY")
    print("=" * 88)
    print(f"  provenance bonds: {n_bonds}   with a recorded mediating spin pair: {tagged}")
    print(f"  dimers carrying a slot map: {len(tr._prov_slot_of)}")
    print(f"  spin-index usage across bond endpoints: {dict(sorted(spins_seen.items()))}")
    print(f"  (K=2 claims per dimer => indices must be a subset of {{0,1}})")
    idx_ok = set(spins_seen) <= {0, 1}
    print(f"  INDICES IN RANGE: {'PASS' if idx_ok else 'FAIL'}")

    # ---- monogamy audit, now possible for the first time ----
    load = Counter()
    for (ka, kb), (sa, sb) in tr._prov_bond_spins.items():
        load[(ka, sa)] += 1
        load[(kb, sb)] += 1
    over = {k: v for k, v in load.items() if v > 1}
    print("\n" + "=" * 88)
    print("MONOGAMY AUDIT ON THE PROVENANCE GRAPH (a spin mediates AT MOST ONE bond)")
    print("=" * 88)
    print(f"  distinct (dimer, spin) mediators in use: {len(load)}")
    print(f"  max bonds asked of a single spin: {max(load.values()) if load else 0}")
    print(f"  spins over the bound: {len(over)} / {len(load)}")
    if over:
        print(f"  worst offenders: {sorted(over.items(), key=lambda x: -x[1])[:5]}")
    print("  => " + ("provenance graph is monogamy-CLEAN at this operating point"
                     if not over else
                     "provenance graph VIOLATES monogamy even at this sparsity"))

    out = {'gate_on_path_unchanged': bool(gate), 'cross': cross, 'intra': intra,
           'n_bonds': n_bonds, 'n_tagged': tagged,
           'spin_index_usage': {str(k): v for k, v in spins_seen.items()},
           'indices_in_range': bool(idx_ok),
           'n_mediators': len(load),
           'max_bonds_per_spin': max(load.values()) if load else 0,
           'n_spins_over_bound': len(over)}
    path = os.path.join(SWEEP_DIR, 'po7_unit4_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
