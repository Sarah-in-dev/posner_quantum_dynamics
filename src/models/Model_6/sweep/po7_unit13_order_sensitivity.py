#!/usr/bin/env python3
"""
PO-7 UNIT 13 — does within-step arrival order matter? (advisor R5 step 4)

THE QUESTION
------------
Spin slots are claimed by a GREEDY walk over candidate pairs in whatever order the update loop
produces. If many bonds form per timestep, that order is doing real work and the measured
partition is partly an artifact of it. If ~<=1 bond forms per step, the order cannot matter and
the concern is moot.

The advisor's test, verbatim in substance: "how many bonds form per timestep? If it's >>1,
within-step arrival order is doing real work and you should shrink dt or randomize the order and
measure sensitivity. If it's <=1, the order artifact is negligible."

TWO ARMS
  A. INSTRUMENT — count bonds formed per timestep (intra), and the slot-claim attempts per step.
  B. SENSITIVITY — run the same seed twice, once with the natural claim order and once with the
     within-step candidate order RANDOMLY PERMUTED, and compare the resulting partition. If the
     partition is insensitive to the permutation, order is not load-bearing.

Single synapse (the deterministic path — the multi-synapse rig is not reproducible, so an
order-sensitivity comparison there would be confounded by the unseeded-RNG divergence).
Nothing is tuned; this is a diagnostic with no verdict function beyond the stated threshold.
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


def run(spin_resolved=True, permute=False, seed=SEED):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles
    dp.spin_resolved = bool(spin_resolved)

    # --- instrument: wrap _create_bond to count calls and successes per step ---
    orig_create = dp._create_bond
    stats = {'attempts': [], 'formed': []}
    counter = {'att': 0, 'made': 0}

    def wrapped(id_i, id_j, strength, spins=None):
        counter['att'] += 1
        before = len(dp._bond_lookup)
        orig_create(id_i, id_j, strength, spins) if spins is not None else \
            orig_create(id_i, id_j, strength)
        if len(dp._bond_lookup) > before:
            counter['made'] += 1
    dp._create_bond = wrapped

    # --- ARM B: permute the order in which free slots are offered ---
    if permute:
        rng = np.random.default_rng(seed + 7)
        orig_claim = dp._claim_spin
        def permuted_claim(dimer_id, key, required=None):
            if required is not None:
                return orig_claim(dimer_id, key, required)
            occ = dp._spin_occ.setdefault(dimer_id, [None] * dp.SPINS_PER_DIMER)
            order = rng.permutation(dp.SPINS_PER_DIMER)
            for s in order:
                if occ[s] is None:
                    occ[s] = key
                    return int(s)
            return None
        dp._claim_spin = permuted_claim

    rel = PresynapticRelease(seed=seed)
    for _ in range(int(round(T_SIM / DT))):
        counter['att'] = 0; counter['made'] = 0
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        stats['attempts'].append(counter['att'])
        stats['formed'].append(counter['made'])

    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]
    idset = set(ids)
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in idset and b in idset]
    sizes = components(ids, edges)
    att = np.array(stats['attempts'], float)
    frm = np.array(stats['formed'], float)
    return {'permute': permute, 'V': len(ids), 'E': len(edges),
            'n_components': len(sizes),
            'largest_frac': (sizes[0] / len(ids)) if (sizes and ids) else 0.0,
            'frustrated': int(dp._spin_frustrated),
            'attempts_per_step_mean': float(att.mean()), 'attempts_per_step_max': float(att.max()),
            'formed_per_step_mean': float(frm.mean()), 'formed_per_step_max': float(frm.max()),
            'formed_per_step_p90': float(np.percentile(frm, 90)),
            'steps_with_gt1_formed': int((frm > 1).sum()), 'n_steps': int(len(frm))}


def main():
    print("=" * 96)
    print("PO-7 UNIT 13 — within-step order sensitivity (advisor R5 step 4)")
    print("=" * 96)

    a = run(spin_resolved=True, permute=False)
    print(f"\n  ARM A — INSTRUMENT (natural claim order)")
    print(f"    bonds FORMED per step: mean={a['formed_per_step_mean']:.2f} "
          f"p90={a['formed_per_step_p90']:.0f} max={a['formed_per_step_max']:.0f}")
    print(f"    steps forming >1 bond: {a['steps_with_gt1_formed']}/{a['n_steps']} "
          f"({100.0*a['steps_with_gt1_formed']/a['n_steps']:.1f}%)")
    print(f"    claim ATTEMPTS per step: mean={a['attempts_per_step_mean']:.1f} "
          f"max={a['attempts_per_step_max']:.0f}")
    print(f"    partition: E={a['E']} components={a['n_components']} "
          f"largest_frac={a['largest_frac']:.3f} frustrated={a['frustrated']}")

    verdict_order_matters = a['formed_per_step_mean'] > 1.0
    print(f"\n    => bonds/step {'>>1' if a['formed_per_step_mean'] > 5 else ('>1' if verdict_order_matters else '<=1')}: "
          f"order is {'LOAD-BEARING — arm B decides how much' if verdict_order_matters else 'NOT load-bearing; §5.1 is moot'}")

    b = run(spin_resolved=True, permute=True)
    print(f"\n  ARM B — SENSITIVITY (slot offer order randomly permuted, same seed)")
    print(f"    partition: E={b['E']} components={b['n_components']} "
          f"largest_frac={b['largest_frac']:.3f} frustrated={b['frustrated']}")

    dE = abs(b['E'] - a['E']) / max(a['E'], 1)
    dC = abs(b['n_components'] - a['n_components']) / max(a['n_components'], 1)
    dL = abs(b['largest_frac'] - a['largest_frac'])
    print(f"\n  DELTA natural vs permuted:  |dE|={100*dE:.2f}%  "
          f"|dComponents|={100*dC:.2f}%  |dLargest_frac|={dL:.4f}")
    insensitive = dE < 0.05 and dC < 0.05 and dL < 0.05
    print(f"  => partition is {'INSENSITIVE' if insensitive else 'SENSITIVE'} to slot order")
    if insensitive:
        print("     The greedy claim order is not determining the partition. §5.1's ordering")
        print("     concern does not apply to WHICH SLOT is taken.")
    else:
        print("     Slot order changes the partition — the greedy walk IS doing work and the")
        print("     allocation principle must be settled before the partition is interpreted.")

    out = {'arm_a_natural': a, 'arm_b_permuted': b,
           'delta': {'edges_frac': dE, 'components_frac': dC, 'largest_frac_abs': dL},
           'order_load_bearing': bool(verdict_order_matters),
           'partition_insensitive_to_slot_order': bool(insensitive)}
    with open(os.path.join(SWEEP_DIR, 'po7_unit13_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote po7_unit13_results.json")
    print("\nNOTE: this permutes WHICH SLOT a bond takes, not which PAIR is offered first.")
    print("Pair order is set by the physics loop and is not separately permutable here; the")
    print("bonds-per-step count in ARM A is the evidence bearing on that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
