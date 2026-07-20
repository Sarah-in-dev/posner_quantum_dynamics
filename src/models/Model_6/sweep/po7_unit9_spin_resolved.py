#!/usr/bin/env python3
"""
PO-7 UNIT 9 — SPIN-RESOLVED BONDING: the missing birth/entanglement representation.

WHAT WAS MISSING
----------------
The model bonded dimers as featureless nodes. But a Ca6(PO4)4 dimer carries FOUR 31P
spin-1/2 nuclei [quantum-system-canonical:43, LOCKED], a singlet-strength bond consumes ONE
spin at each end, and monogamy of entanglement forbids a spin mediating two bonds. Nothing
in the model represented that, so the graph reached mean degree 715 against a hard bound of
4 — 99.44% of edges physically inadmissible (Unit 3).

THE BUILD (opt-in `spin_resolved`; off => byte-identical, gated)
  - every dimer owns 4 spin slots;
  - a bond must claim a FREE slot at both endpoints or it does not form;
  - provenance-inherited bonds must claim their NAMED slot (the inherited nucleus sits in a
    specific slot — Unit 4's which-spin tag), so two inheritances competing for one slot
    cannot both be satisfied;
  - degree <= 4 is DERIVED, never capped.

WHAT REJECTED BONDS ARE
  Not lost edges — FRUSTRATION. Pairs individually satisfiable, jointly not. That is the
  H^1 obstruction the Unit-5 "sheaf" could not express (it decomposed into 3 ordinary graph
  Laplacians, cross-block edges 0/369740, verified in Unit 3).

MEASURED HERE, on the standing fingerprint rig (1 synapse, 200 steps, seed 31337):
  degree distribution, admissibility, frustration count, and the PARTITION
  (components / largest_frac) with spin resolution OFF vs ON.
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

T_SIM, DT, SEED = 1.0, 0.005, 31337
SPINS = 4


def components(ids, edges):
    parent = {i: i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    bonded = set()
    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
            bonded.add(a); bonded.add(b)
    groups = {}
    for i in bonded:
        groups.setdefault(find(i), []).append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    return sizes


def run(spin_resolved):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles
    dp.spin_resolved = bool(spin_resolved)

    rel = PresynapticRelease(seed=SEED)
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})

    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]
    idset = set(ids)
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in idset and b in idset]
    deg = Counter()
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    degs = np.array([deg.get(i, 0) for i in ids], float) if ids else np.zeros(1)
    sizes = components(ids, edges)
    V, E = len(ids), len(edges)
    return {'spin_resolved': bool(spin_resolved), 'V': V, 'E': E,
            'mean_deg': float(degs.mean()), 'max_deg': float(degs.max()),
            'over_bound': int((degs > SPINS).sum()),
            'frac_over': float((degs > SPINS).sum() / V) if V else 0.0,
            'inadmissible_edges': float(np.clip(degs - SPINS, 0, None).sum() / 2.0),
            'n_components': len(sizes),
            'largest_frac': (sizes[0] / V) if (sizes and V) else 0.0,
            'frustrated': int(dp._spin_frustrated),
            'n_bond_spins': len(dp._bond_spins)}


def main():
    print("=" * 96)
    print("PO-7 UNIT 9 — spin-resolved bonding (4x 31P per Ca6(PO4)4, monogamy derived)")
    print("=" * 96)
    rows = {}
    for flag in (False, True):
        r = run(flag); rows['ON' if flag else 'OFF'] = r
        tag = 'ON ' if flag else 'OFF'
        print(f"\n  spin_resolved={tag}  V={r['V']}  E={r['E']}")
        print(f"     mean degree      {r['mean_deg']:10.2f}   (bound {SPINS})")
        print(f"     max degree       {r['max_deg']:10.0f}")
        print(f"     dimers over bound{r['over_bound']:10d}  ({100*r['frac_over']:.1f}%)")
        print(f"     inadmissible edges{r['inadmissible_edges']:9.0f}")
        print(f"     components       {r['n_components']:10d}")
        print(f"     largest_frac     {r['largest_frac']:10.3f}")
        print(f"     frustrated bonds {r['frustrated']:10d}")

    off, on = rows['OFF'], rows['ON']
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    adm = on['over_bound'] == 0
    print(f"  monogamy satisfied with spin resolution ON: {'YES' if adm else 'NO'} "
          f"({on['over_bound']} dimers over bound)")
    print(f"  edges: {off['E']} -> {on['E']}  ({100.0*on['E']/off['E']:.2f}% retained)")
    print(f"  frustration (bonds refused for want of a free spin): {on['frustrated']}")
    print(f"  largest_frac: {off['largest_frac']:.3f} -> {on['largest_frac']:.3f}")
    print(f"  components:   {off['n_components']} -> {on['n_components']}")
    if on['largest_frac'] < 0.95 <= off['largest_frac']:
        print("  => THE BLOB BREAKS. Enforcing the physical spin bound fragments the graph;")
        print("     the partition is no longer trivial.")
    elif on['largest_frac'] >= 0.95:
        print("  => Blob PERSISTS even under monogamy. Percolation is intrinsic, not a")
        print("     consequence of inadmissible edges — a real negative, and a stronger one.")
    with open(os.path.join(SWEEP_DIR, 'po7_unit9_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print("\nwrote po7_unit9_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
