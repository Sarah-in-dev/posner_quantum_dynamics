#!/usr/bin/env python3
"""PO-5 UNIT 11 — the three repair arms. Pre-registered: docs/PREREG_PO5_UNIT11_REPAIR_ARMS.md"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
sys.path.insert(0, SWEEP_DIR)
from po5_unit5_sheaf_laplacian import sheaf_cohomology
DT, T_SIM, SEEDS = 0.005, 1.0, [51, 52]

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

def run(seed, cl=None, cap=0, jc=False):
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
    if cl is not None: dp.coupling_length = cl
    dp.birth_degree_cap = cap
    dp.j_compat_formation = jc
    rel = PresynapticRelease(seed=seed); t0 = time.time()
    for _ in range(int(round(T_SIM/DT))):
        g = rel.step(0.95, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; es = set(ids)
    pos = {d.id: np.asarray(d.position, float) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    c, lf = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0)
    h0 = sheaf_cohomology(ids, pos, edges, "geometry")["H0_engaged"] if len(ids) >= 2 else 0
    return {"V": len(ids), "E": len(edges), "components": c, "largest_frac": lf,
            "sheaf_H0": h0, "elapsed_s": time.time()-t0}

def cell(label, **kw):
    rs = [run(s, **kw) for s in SEEDS]
    m = lambda k: float(np.mean([r[k] for r in rs]))
    row = {"label": label, **{k: m(k) for k in ("V","E","components","largest_frac","sheaf_H0")}, **kw}
    print(f"  {label:34s} E={row['E']:9.0f} comps={row['components']:7.1f} "
          f"largest_frac={row['largest_frac']:.4f} H0={row['sheaf_H0']:5.1f}", flush=True)
    return row

def main():
    print("="*96); print("PO-5 UNIT 11 — repair arms (all opt-in, all-off = today's physics)"); print("="*96, flush=True)
    rows = []
    print("\nBASELINE + ARM A (coupling_length sweep; diagnostic override):", flush=True)
    rows.append(cell("baseline (all off)"))
    for cl in (1.0, 2.0, 5.0, 10.0):
        rows.append(cell(f"A: coupling_length={cl:g}nm", cl=cl))
    print("\nARM B (degree cap on birth-pairing):", flush=True)
    for k in (1, 2, 4, 8):
        rows.append(cell(f"B: birth_degree_cap={k}", cap=k))
    print("\nARM C (J-compatibility gating formation):", flush=True)
    rows.append(cell("C: j_compat_formation", jc=True))
    print("\nCOMBINED:", flush=True)
    rows.append(cell("B(k=4) + C", cap=4, jc=True))
    rows.append(cell("B(k=4) + C + A(1nm)", cap=4, jc=True, cl=1.0))
    with open(os.path.join(SWEEP_DIR, "po5_unit11_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    base = rows[0]
    print("\n" + "="*96)
    print(f"P1 baseline largest_frac = {base['largest_frac']:.4f}"
          f"  {'(anchor OK)' if base['largest_frac'] > 0.95 else '(UNEXPECTED)'}")
    A = [r for r in rows if r["label"].startswith("A:")]
    spread = max(r["largest_frac"] for r in A) - min(r["largest_frac"] for r in A)
    print(f"P3 coupling_length: largest_frac spread over 1-10nm = {spread:.4f} "
          f"=> {'LOAD-BEARING (and ungrounded)' if spread > 0.05 else 'NOT load-bearing'}")
    B4 = [r for r in rows if r["label"] == "B: birth_degree_cap=4"][0]
    print(f"P2 arm B k=4: largest_frac = {B4['largest_frac']:.4f} "
          f"=> {'FRAGMENTS' if B4['largest_frac'] < 0.9 else 'does NOT fragment (P2 carries it, per Unit 7)'}")
    C = [r for r in rows if r["label"].startswith("C:")][0]
    print(f"P4 arm C: E {base['E']:.0f} -> {C['E']:.0f} "
          f"({100*(C['E']-base['E'])/base['E']:+.2f}%) "
          f"=> {'changes the bond set' if abs(C['E']-base['E'])/base['E'] > 0.01 else 'NULL (formation gating inert too)'}")
    best = min(rows, key=lambda r: r["largest_frac"])
    print(f"\nlowest largest_frac achieved: {best['largest_frac']:.4f} by [{best['label']}]")
    print("\nLIMITS: 1 synapse, 1 s, 2 seeds. A fragmenting arm is NOT endorsed — B and C encode")
    print("physics claims needing grounding independent of any result here.")

if __name__ == "__main__":
    main()
