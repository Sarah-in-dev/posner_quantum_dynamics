#!/usr/bin/env python3
"""
PO-7 UNIT 14 — the STOICHIOMETRIC prediction (advisor R5 step 3).

THE PREDICTION, stated before running
-------------------------------------
Provenance gives each dimer K = 2 hydrolysis events, hence at most 2 inherited bonds, leaving
**2 of its 4 31P nuclei free**. But Unit 9 measured mean intra degree 3.93 with the legacy
mechanisms live. The ~1.9 excess bonds per dimer come from the two mechanisms already
established as unphysical:
  - the 100 ms birth-window CLIQUE (a proxy for a missing representation; LOCC-non-entangling)
  - the phenomenological EM pathway (no microscopic Hamiltonian, ~15-order frequency gap)

**Prediction: with provenance ON (which replaces the clique and skips the EM path) and spin
resolution ON, mean intra degree falls to ~2, leaving ~2 free nuclei per dimer.** If it holds,
cross-synapse capacity is ~2 slots/dimer as a STOICHIOMETRIC fact, independent of update order —
and the starvation measured in Unit 11 was caused by bonds that should not exist.

No new physics is written here: `provenance_bonding` already replaces the clique entirely
(dimer_particles.py — the `elif template_bound:` legacy branch is skipped when it is on) and
skips the EM pathway. This is a 2x2 over two existing flags.

FALSIFIER: if intra degree stays near 4 with provenance ON, the excess is NOT the clique/EM and
the diagnosis is wrong. If it falls well below 2, provenance is producing fewer bonds than its
own K=2 allows and something else is limiting.
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
    return sorted((len(v) for v in groups.values()), reverse=True)


def run(provenance, spin):
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
    dp.provenance_bonding = bool(provenance)   # replaces the clique, skips the EM pathway
    dp.spin_resolved = bool(spin)

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
    free = float(np.clip(SPINS - degs, 0, SPINS).mean())
    return {'provenance': bool(provenance), 'spin_resolved': bool(spin),
            'V': len(ids), 'E': len(edges),
            'mean_deg': float(degs.mean()), 'max_deg': float(degs.max()),
            'median_deg': float(np.median(degs)),
            'mean_free_nuclei': free,
            'over_bound': int((degs > SPINS).sum()),
            'n_components': len(sizes),
            'largest_frac': (sizes[0] / len(ids)) if (sizes and ids) else 0.0,
            'frustrated': int(dp._spin_frustrated)}


def main():
    print("=" * 100)
    print("PO-7 UNIT 14 — stoichiometry: does dropping the clique + EM pathway free ~2 nuclei?")
    print("=" * 100)
    print("  PREDICTION (registered above, before running): provenance ON + spin ON")
    print("  => mean intra degree ~2, mean free nuclei ~2\n")
    rows = {}
    for prov in (False, True):
        for spin in (False, True):
            r = run(prov, spin)
            key = f"prov={'ON ' if prov else 'OFF'} spin={'ON ' if spin else 'OFF'}"
            rows[key] = r
            print(f"  {key}   V={r['V']:>5} E={r['E']:>7}  "
                  f"deg mean={r['mean_deg']:7.2f} med={r['median_deg']:6.1f} max={r['max_deg']:5.0f}  "
                  f"free={r['mean_free_nuclei']:4.2f}  comps={r['n_components']:>4} "
                  f"lgfrac={r['largest_frac']:.3f}")
            sys.stdout.flush()

    target = rows["prov=ON  spin=ON "]
    print("\n" + "=" * 100)
    print("VERDICT ON THE PREDICTION")
    print("=" * 100)
    d, free = target['mean_deg'], target['mean_free_nuclei']
    print(f"  provenance ON + spin ON: mean intra degree = {d:.2f}, mean free nuclei = {free:.2f}")
    if 1.5 <= d <= 2.5:
        print("  => PREDICTION HOLDS. Dropping the clique and the EM pathway leaves ~2 nuclei")
        print("     free per dimer. Cross-synapse capacity is ~2 slots/dimer as a STOICHIOMETRIC")
        print("     fact, independent of update order — Unit 11's starvation was caused by bonds")
        print("     that should not exist.")
    elif d > 2.5:
        print("  => PREDICTION FAILS HIGH. The excess degree is NOT the clique/EM pathway;")
        print("     something else is generating bonds and the diagnosis needs revisiting.")
    else:
        print("  => PREDICTION FAILS LOW. Provenance yields fewer bonds than its own K=2 allows;")
        print("     something else is limiting formation.")
    base = rows["prov=OFF spin=ON "]
    print(f"\n  for reference, clique+EM with spin ON: mean degree {base['mean_deg']:.2f}, "
          f"free {base['mean_free_nuclei']:.2f}")
    print(f"  excess attributable to clique+EM: {base['mean_deg'] - d:.2f} bonds/dimer")

    with open(os.path.join(SWEEP_DIR, 'po7_unit14_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print("\nwrote po7_unit14_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
