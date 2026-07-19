#!/usr/bin/env python3
"""
PO-5 UNIT 12 — the two repairs Unit 11 missed, aimed at the pathway that actually percolates.

ARM D — INTRA-BOND FIDELITY CUT (analysis layer; NO source change)
  Cross-synapse bonds count only above the Werner bound (F = P_S_i*P_S_j*w, edge iff F>0.5).
  INTRA bonds have NO such cut -- connectivity is BARE EXISTENCE, so a bond formed once counts
  forever at full weight however weak or long-range. `model6-entanglement-partition-werner`
  flags exactly this: the intra layer "carries the same dead-store pattern (1/r^3 in rate, not
  in stored fidelity; bare-existence connectivity) and could get the same treatment IF ITS OWN
  BLOB/SATURATION NEEDS IT". It does.

  *** The same skill also says: "Do NOT apply the 0.5 bound to intra bonds (different scale;
  would break the working intra layer)." That instruction is RESPECTED. This does not assert
  0.5 or any other value -- it stores F = P_S_i*P_S_j*g(r_ij) and SWEEPS the threshold,
  reporting the response curve and the F distribution. No threshold is nominated. ***

  Implemented purely in analysis: the edge set is filtered at read time. The dynamics are
  untouched, so nothing standing is invalidated.

ARM E — k_entangle_em_base sweep
  A function-body literal (1.0), ungrounded, unsweepable before this -- the same class as
  kT_ref. NOT tied to the derived 20 kT; only `reference_kT` is. Promoted to an attribute,
  verified bit-identical.

PRE-REGISTERED:
  P1 at threshold 0 the filtered graph == the unfiltered graph (sanity anchor).
  P2 some threshold gives largest_frac < 0.9 -> the fidelity cut fragments the blob.
  P3 if NO threshold below the point where the graph empties gives an intermediate state,
     the cut only ever goes blob -> dust, and is not a usable lever. Reported as such.
"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
sys.path.insert(0, SWEEP_DIR)
from po5_unit5_sheaf_laplacian import sheaf_cohomology
DT, T_SIM, SEEDS = 0.005, 1.0, [61, 62]
THRESHOLDS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

def clusters(ids, edges):
    p = {i: i for i in ids}
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb: p[ra] = rb
    sz = {}
    for i in ids: r = f(i); sz[r] = sz.get(r, 0) + 1
    s = sorted(sz.values(), reverse=True)
    return len(s), (s[0]/len(ids) if ids else 0.0)

def snapshot(seed, kbase=None):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses: s.set_microtubule_invasion(True)
    syn = net.synapses[0]; dp = syn.dimer_particles
    if kbase is not None: dp.k_entangle_em_base = kbase
    rel = PresynapticRelease(seed=seed)
    for _ in range(int(round(T_SIM/DT))):
        g = rel.step(0.95, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
    ent = [d for d in dp.dimers if d.is_entangled]
    st = {d.id: (np.asarray(d.position, float), d.singlet_probability) for d in ent}
    es = set(st)
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    L = dp.coupling_length
    F = {}
    for (a, b) in edges:
        r = float(np.linalg.norm(st[a][0] - st[b][0]))
        g_ = (L / max(r, L)) ** 3
        F[(a, b)] = st[a][1] * st[b][1] * g_
    return list(st), edges, F, st

def main():
    print("=" * 96)
    print("PO-5 UNIT 12 — intra-bond fidelity cut (arm D) + base-rate sweep (arm E)")
    print("=" * 96, flush=True)
    rows = []

    print("\nARM D — fidelity F = P_S_i*P_S_j*g(r); threshold SWEPT, none nominated", flush=True)
    allF = []
    for seed in SEEDS:
        ids, edges, F, _ = snapshot(seed)
        allF += list(F.values())
    q = np.percentile(allF, [1, 10, 25, 50, 75, 90, 99])
    print(f"  F distribution over {len(allF)} bonds: "
          f"p1={q[0]:.4f} p10={q[1]:.4f} p25={q[2]:.4f} med={q[3]:.4f} "
          f"p75={q[4]:.4f} p90={q[5]:.4f} p99={q[6]:.4f}", flush=True)
    print(f"  {'thresh':>8s} {'edges kept':>11s} {'frac':>7s} {'comps':>7s} {'largest_frac':>13s} {'H0':>5s}")
    for th in THRESHOLDS:
        cs, lfs, h0s, ke, tot = [], [], [], [], []
        for seed in SEEDS:
            ids, edges, F, st = snapshot(seed)
            kept = [e for e in edges if F[e] >= th]
            c, lf = clusters(ids, kept) if len(ids) >= 2 else (len(ids), 1.0)
            pos = {i: st[i][0] for i in ids}
            h0 = sheaf_cohomology(ids, pos, kept, "geometry")["H0_engaged"] if len(ids) >= 2 else 0
            cs.append(c); lfs.append(lf); h0s.append(h0); ke.append(len(kept)); tot.append(len(edges))
        row = {"arm": "D", "threshold": th, "edges_kept": float(np.mean(ke)),
               "edge_frac": float(np.mean(ke)/np.mean(tot)), "components": float(np.mean(cs)),
               "largest_frac": float(np.mean(lfs)), "sheaf_H0": float(np.mean(h0s))}
        rows.append(row)
        print(f"  {th:8.2f} {np.mean(ke):11.0f} {row['edge_frac']:7.3f} "
              f"{np.mean(cs):7.1f} {np.mean(lfs):13.4f} {np.mean(h0s):5.1f}", flush=True)

    print("\nARM E — k_entangle_em_base (ungrounded literal; NOT the derived 20 kT)", flush=True)
    print(f"  {'k_base':>8s} {'edges':>9s} {'comps':>7s} {'largest_frac':>13s}")
    for kb in (0.01, 0.05, 0.1, 0.3, 1.0):
        cs, lfs, es_ = [], [], []
        for seed in SEEDS:
            ids, edges, F, st = snapshot(seed, kbase=kb)
            c, lf = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0)
            cs.append(c); lfs.append(lf); es_.append(len(edges))
        rows.append({"arm": "E", "k_base": kb, "edges": float(np.mean(es_)),
                     "components": float(np.mean(cs)), "largest_frac": float(np.mean(lfs))})
        print(f"  {kb:8.3f} {np.mean(es_):9.0f} {np.mean(cs):7.1f} {np.mean(lfs):13.4f}", flush=True)

    with open(os.path.join(SWEEP_DIR, "po5_unit12_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    D = [r for r in rows if r["arm"] == "D"]
    print("\n" + "=" * 96)
    print(f"P1 anchor: threshold 0 -> largest_frac {D[0]['largest_frac']:.4f}, "
          f"edge_frac {D[0]['edge_frac']:.3f}  {'(OK)' if D[0]['edge_frac'] > 0.999 else '(FAIL)'}")
    mid = [r for r in D if 0.05 < r["largest_frac"] < 0.9]
    if mid:
        print("P2 CONFIRMED — the fidelity cut yields an INTERMEDIATE regime:")
        for r in mid:
            print(f"    thresh={r['threshold']:.2f}: components {r['components']:.1f}, "
                  f"largest_frac {r['largest_frac']:.4f}, edges kept {r['edge_frac']:.3f}, "
                  f"H0 {r['sheaf_H0']:.1f}")
        print("    => a lever exists that fragments the blob WITHOUT touching the 20 kT.")
    else:
        print("P3 — no intermediate regime: the cut goes blob -> dust with nothing between.")
        print("    Not a usable lever.")
    E = [r for r in rows if r["arm"] == "E"]
    print(f"\nARM E: largest_frac {min(r['largest_frac'] for r in E):.4f} .. "
          f"{max(r['largest_frac'] for r in E):.4f} over k_base 0.01-1.0")
    print("\nLIMITS: 1 synapse, 1 s, 2 seeds. NO threshold and NO k_base is nominated as correct;")
    print("the 0.5 Werner bound is NOT applied to intra bonds, per the skill's explicit caution.")

if __name__ == "__main__":
    main()
