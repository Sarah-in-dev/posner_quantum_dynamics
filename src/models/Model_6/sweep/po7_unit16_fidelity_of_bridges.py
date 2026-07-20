#!/usr/bin/env python3
"""
PO-7 UNIT 16 — is the percolation held together by bonds too weak to matter?

THE QUESTION (advisor R5 §5.2, sharpened by Unit 15)
----------------------------------------------------
Unit 15 established the collapse is a mean-degree~1 percolation, not a spin-budget effect. But
pi_0 counts every admitted edge identically: an edge at F=0.501 joins two components exactly as
firmly as one at F=0.99, while carrying

    tau = C^2 = (2F-1)^2  =>  F=0.501 -> tau = 4e-6 ;  F=0.99 -> tau = 0.96

i.e. ~0.0004% of the entanglement. If the BRIDGES — the edges whose removal would disconnect
components — sit near the Werner floor, then the giant component is an artifact of an unweighted
invariant and cannot sustain correlated collapse.

WHAT THIS MEASURES, at the percolated end state:
  1. the fidelity distribution of all cross-synapse edges;
  2. the distribution restricted to BRIDGES (edges whose removal increases the component count);
  3. what the partition becomes if edges are admitted on a tangle (CKW) criterion instead of the
     bare Werner bound — i.e. how much of the giant component survives when weak edges are
     discounted.

(3) is reported as a SWEEP over the tangle threshold, not a single tuned value. No threshold is
nominated here; the shape of the curve is the result. If largest_frac falls off a cliff at a
tangle level that carries negligible entanglement, the blob is not physical.

NOTE ON DIRECTION (correcting an error I made earlier): CKW does NOT limit weak bonds — under
tau=(2F-1)^2 a weak bond is CHEAP, so a CKW budget permits MORE of them. The point is not that
CKW prunes bridges; it is that the entanglement CARRIED by a bridge is what should determine
whether two dimers are in the same component at all.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

N_SYN, SPACING = 7, 1.0
DT, T_SIM = 5e-3, float(os.environ.get('U16_SECONDS','20.0'))
# Seed 4242 is a KNOWN-IGNITING draw (po7_unit12_seeding_check). Ignition is
# seed-dependent: seed 9137 never condenses, seed 0 did not either. With the RNGs now
# threaded (commit 57ccd75) this is reproducible rather than luck, but it means any
# claim from ONE seed is a claim about that draw.
SEED = int(os.environ.get('U16_SEED','4242'))
VOLT = -40e-3
TRACKER_EVERY = 10
WERNER = 0.5


def comps_from_edges(nodes, edges):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted((len(v) for v in groups.values()), reverse=True)


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters()
    p.em_coupling_enabled = True; p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING,
                              seed=SEED)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True   # drop clique + EM (Unit 14)
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    peak_eta=[0.0]; peak_xb=[0]
    rel = PresynapticRelease(seed=SEED)
    for i in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        for s in net.synapses:
            s.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))
            etas = [float(getattr(s_, '_backbone_eta', 0.0)) for s_ in net.synapses]
            peak_eta[0] = max(peak_eta[0], max(etas))
            peak_xb[0] = max(peak_xb[0], len(tr.cross_synapse_bonds))

    # --- the graph at the end state ---
    nodes = [d['global_id'] for d in tr.all_dimers]
    nodeset = set(nodes)
    cross = [(k, f) for k, f in tr.cross_synapse_bonds.items()
             if f > WERNER and k[0] in nodeset and k[1] in nodeset and k[0][0] != k[1][0]]
    intra = [(k, f) for k, f in tr.intra_synapse_bonds_cache.items()
             if k[0] in nodeset and k[1] in nodeset]
    all_edges = [k for k, _ in cross] + [k for k, _ in intra]
    base = comps_from_edges(nodes, all_edges)
    V = len(nodes)
    print("=" * 96)
    print("PO-7 UNIT 16 — fidelity of the bridges")
    print("=" * 96)
    print(f"  V={V}  cross edges={len(cross)}  intra edges={len(intra)}")
    print(f"  DIAGNOSTIC: peak eta over run = {peak_eta[0]:.4f}; "
          f"peak cross-bond count ever = {peak_xb[0]}")
    if not cross:
        print("  no cross edges at the end state — nothing to analyse.")
        return 1
    print(f"  components={len(base)}  largest_frac={base[0]/V:.4f}")

    F = np.array([f for _, f in cross], float)
    tau = (2.0 * F - 1.0) ** 2
    print(f"\n  CROSS-EDGE FIDELITY: min={F.min():.4f} p10={np.percentile(F,10):.4f} "
          f"median={np.median(F):.4f} p90={np.percentile(F,90):.4f} max={F.max():.4f}")
    print(f"  CROSS-EDGE TANGLE tau=(2F-1)^2: median={np.median(tau):.6f} "
          f"mean={tau.mean():.6f} max={tau.max():.6f}")
    print(f"  fraction of cross edges with F < 0.55 (tau < 0.01): "
          f"{100.0*(F < 0.55).mean():.1f}%")

    # --- which cross edges are BRIDGES? (removal increases component count) ---
    intra_only = comps_from_edges(nodes, [k for k, _ in intra])
    n_intra_comp = len(intra_only)
    bridges, nonbridges = [], []
    for k, f in cross:
        without = [kk for kk, _ in cross if kk != k] + [kk for kk, _ in intra]
        if len(comps_from_edges(nodes, without)) > len(base):
            bridges.append(f)
        else:
            nonbridges.append(f)
    print(f"\n  cross edges that are BRIDGES: {len(bridges)} / {len(cross)}")
    if bridges:
        B = np.array(bridges, float); tb = (2*B - 1)**2
        print(f"    bridge F: min={B.min():.4f} median={np.median(B):.4f} max={B.max():.4f}")
        print(f"    bridge tangle: median={np.median(tb):.6f} max={tb.max():.6f}")

    # --- sweep: admit edges only above a tangle threshold ---
    print(f"\n  PARTITION vs TANGLE THRESHOLD (sweep, no value nominated)")
    print(f"    {'tau_min':>10} {'F_equiv':>8} {'edges kept':>11} {'components':>11} {'largest_frac':>13}")
    sweep = []
    for tmin in [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5]:
        fmin = 0.5 * (1 + np.sqrt(tmin))
        keep = [k for k, f in cross if (2*f - 1)**2 >= tmin] + [k for k, _ in intra]
        c = comps_from_edges(nodes, keep)
        n_cross_kept = sum(1 for k, f in cross if (2*f - 1)**2 >= tmin)
        sweep.append({'tau_min': tmin, 'F_equiv': float(fmin), 'cross_kept': n_cross_kept,
                      'components': len(c), 'largest_frac': c[0]/V})
        print(f"    {tmin:10.6f} {fmin:8.4f} {n_cross_kept:11d} {len(c):11d} {c[0]/V:13.4f}")

    print(f"\n  reference: intra-only partition = {n_intra_comp} components, "
          f"largest_frac={intra_only[0]/V:.4f}")
    print("\n  READ AS: if largest_frac collapses toward the intra-only value at a tangle")
    print("  threshold carrying negligible entanglement, the giant component is an artifact")
    print("  of unweighted connectivity and cannot sustain correlated collapse.")

    out = {'V': V, 'n_cross': len(cross), 'n_intra': len(intra),
           'components': len(base), 'largest_frac': base[0]/V,
           'cross_F': {'min': float(F.min()), 'p10': float(np.percentile(F,10)),
                       'median': float(np.median(F)), 'p90': float(np.percentile(F,90)),
                       'max': float(F.max())},
           'cross_tau_median': float(np.median(tau)),
           'frac_F_below_0p55': float((F < 0.55).mean()),
           'n_bridges': len(bridges),
           'bridge_F_median': float(np.median(bridges)) if bridges else None,
           'intra_only_components': n_intra_comp,
           'intra_only_largest_frac': intra_only[0]/V,
           'sweep': sweep}
    with open(os.path.join(SWEEP_DIR, 'po7_unit16_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nwrote po7_unit16_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
