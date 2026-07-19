#!/usr/bin/env python3
"""
PO-7 UNIT 2 — the multi-synapse SS8 keystone test.

Pre-registered in docs/PREREG_PO7_UNIT2_MULTISYNAPSE_KEYSTONE.md. Read that first; this file
implements it and must not drift from it.

WHY THE GRAPH IS SYNAPSE-LEVEL, NOT DIMER-LEVEL (registered before scoring):
  intra_synapse_bonds_cache is the dense per-synapse clique -- the single-synapse fingerprint
  alone carries E=369740 intra edges. EVERY intra edge lies inside one synapse and therefore
  inside one activation label, so dimer-level modularity against activation identity would sit
  near 1.0 BY CONSTRUCTION, for any input whatsoever. That statistic could only ever pass, which
  is precisely the mis-registration the pre-registration guard exists to reject.
  The correct object is the one quantum-system-canonical SS5 / model6-entanglement-partition-werner
  name: "synapses = nodes, cross-synapse bonds = edges". Q is computed on the 6-node synapse
  graph whose edges are the CROSS-synapse bonds (provenance + eta, each above the Werner bound).

Verdict is the pre-registered function; a negative is a result and is reported as one.
"""
import sys, os, json, logging, itertools
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

T_SIM, DT = 0.4, 0.005
TRACKER_EVERY = 10
SEEDS = [31337, 4242, 90210, 7, 123456]          # >=5 seeds (the 3-seed scars)
N_SYN = 6
WERNER = 0.5

ARMS = {
    'arm1_contiguous':  {'A': {0, 1, 2}, 'B': {3, 4, 5}},   # CONFOUNDED with spatial half
    'arm2_interleaved': {'A': {0, 2, 4}, 'B': {1, 3, 5}},   # ORTHOGONAL to space -- the discriminator
}


def newman_Q(W: np.ndarray, labels: np.ndarray) -> float:
    """Newman modularity of a weighted undirected node graph under a given labelling.

    Q = sum_ij [ W_ij/2m - k_i k_j / (2m)^2 ] * delta(l_i, l_j)
    Returns 0.0 for an empty graph (no edges => no structure to report).
    """
    W = np.asarray(W, float)
    m2 = W.sum()
    if m2 <= 0:
        return 0.0
    k = W.sum(axis=1)
    same = (labels[:, None] == labels[None, :]).astype(float)
    return float((((W / m2) - np.outer(k, k) / (m2 ** 2)) * same).sum())


def run_one(active: set, spacing_um: float, seed: int) -> dict:
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(seed)
    p = Model6Parameters()
    p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=spacing_um)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    tr = net.entanglement_tracker
    tr.provenance_network = True

    rel = PresynapticRelease(seed=seed)
    for i in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        for j, s in enumerate(net.synapses):
            if j in active:
                s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
            else:
                s.step(DT, {"voltage": -70e-3, "reward": False, "glutamate": 0.0})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))

    # --- synapse-level cross-bond graph (provenance + eta, each above the Werner bound) ---
    W = np.zeros((N_SYN, N_SYN))
    n_prov_cross = n_eta_cross = 0
    for (a, b), f in tr._prov_bonds.items():
        if f > WERNER and a[0] != b[0]:
            W[a[0], b[0]] += 1.0; W[b[0], a[0]] += 1.0; n_prov_cross += 1
    for (a, b), f in tr.cross_synapse_bonds.items():
        if f > WERNER and a[0] != b[0]:
            W[a[0], b[0]] += 1.0; W[b[0], a[0]] += 1.0; n_eta_cross += 1

    labels = np.array([1 if j in active else 0 for j in range(N_SYN)])
    Q_act = newman_Q(W, labels)

    # Null: permute which synapses carry the label, same label counts.
    rng = np.random.RandomState(seed + 999)
    Q_shuf = float(np.mean([newman_Q(W, rng.permutation(labels)) for _ in range(200)]))

    # Decomposition statistic: components spanning >=2 synapses.
    comps = tr._find_all_clusters()
    n_multi = sum(1 for c in comps if len({g[0] for g in c}) >= 2)

    return {'seed': seed, 'active': sorted(active), 'Q_act': Q_act, 'Q_shuf': Q_shuf,
            'n_multi': n_multi, 'n_components': len(comps),
            'n_prov_cross': n_prov_cross, 'n_eta_cross': n_eta_cross,
            'n_prov_bonds_total': len(tr._prov_bonds),
            'n_synapse_pairs_linked': int((W > 0).sum() // 2)}


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sqrt(((x.var(ddof=1) if len(x) > 1 else 0.0) +
                 (y.var(ddof=1) if len(y) > 1 else 0.0)) / 2.0)
    if s == 0:
        return 0.0 if abs(x.mean() - y.mean()) < 1e-12 else float('inf')
    return float((x.mean() - y.mean()) / s)


def verdict(per_arm: dict) -> tuple:
    """The PRE-REGISTERED verdict function. PASS requires all three; else NEGATIVE."""
    reasons = []
    c1 = all(a['mean_n_multi'] >= 1 for a in per_arm.values())
    if not c1:
        reasons.append("DECOMPOSITION NULL: partition splits cleanly per-synapse "
                       "(mean n_multi < 1) -- input-LOCATED but not input-COMPUTING")
    c2 = all(a['d_act_vs_shuf'] >= 0.8 for a in per_arm.values())
    if not c2:
        reasons.append("FLAT NULL: d(Q_act vs Q_shuf) < 0.8 in at least one arm")
    d_arms = abs(cohens_d([per_arm['arm1_contiguous']['mean_Q_act']],
                          [per_arm['arm2_interleaved']['mean_Q_act']]))
    c3 = per_arm['arm2_interleaved']['d_act_vs_shuf'] >= 0.3
    if not c3:
        reasons.append("SPATIAL CONFOUND: arm2 (interleaved) collapses to null while the "
                       "signal survives only in the spatially-contiguous arm -- this is the "
                       "L.PO5-13 false positive, caught")
    ok = c1 and c2 and c3
    return (("PASS: provenance carries input-dependent cross-synapse partition" if ok
             else "NEGATIVE: " + " | ".join(reasons)), ok)


def main():
    spacing = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    print(f"PO-7 UNIT 2 keystone test -- spacing={spacing}um, {len(SEEDS)} seeds, "
          f"{N_SYN} synapses\n")
    rows, per_arm = [], {}
    for arm, conds in ARMS.items():
        Qa, Qs, nm = [], [], []
        for cond, active in conds.items():
            for sd in SEEDS:
                r = run_one(active, spacing, sd); r['arm'] = arm; r['cond'] = cond
                rows.append(r); Qa.append(r['Q_act']); Qs.append(r['Q_shuf']); nm.append(r['n_multi'])
                print(f"  {arm:<18} {cond} seed={sd:>6}  Q_act={r['Q_act']:+.4f} "
                      f"Q_shuf={r['Q_shuf']:+.4f} n_multi={r['n_multi']:>3} "
                      f"prov_cross={r['n_prov_cross']:>4} eta_cross={r['n_eta_cross']:>4}")
                sys.stdout.flush()
        per_arm[arm] = {'mean_Q_act': float(np.mean(Qa)), 'mean_Q_shuf': float(np.mean(Qs)),
                        'mean_n_multi': float(np.mean(nm)),
                        'd_act_vs_shuf': cohens_d(Qa, Qs)}
        a = per_arm[arm]
        print(f"  -> {arm}: mean Q_act={a['mean_Q_act']:+.4f} Q_shuf={a['mean_Q_shuf']:+.4f} "
              f"d={a['d_act_vs_shuf']:+.3f} mean_n_multi={a['mean_n_multi']:.2f}\n")

    v, ok = verdict(per_arm)
    print("=== PRE-REGISTERED VERDICT ===")
    print(f"  {v}")
    out = {'spacing_um': spacing, 'seeds': SEEDS, 'rows': rows, 'per_arm': per_arm,
           'verdict': v, 'passed': bool(ok),
           'prereg': 'docs/PREREG_PO7_UNIT2_MULTISYNAPSE_KEYSTONE.md'}
    path = os.path.join(SWEEP_DIR, 'po7_unit2_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
