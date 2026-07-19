#!/usr/bin/env python3
"""
PO-5 UNIT 6 — does J-coupling mismatch, as a dissolution term, fragment the graph?

Pre-registered: docs/PREREG_PO5_UNIT6_J_MISMATCH.md (committed before the code change).
Flag-off verified BIT-IDENTICAL to pre-change code (1034 / 369740 / 0.991922159684).

ARM A OFF       -- current physics, the standing baseline
ARM B REAL      -- dissolution scaled by true J-mismatch (the hypothesis)
ARM C SCRAMBLED -- same magnitudes, permuted across pairs (THE DECISIVE CONTROL)

C exists because PO-5 proposed this mechanism while predicting it would fragment the graph.
If B ~= C the effect is "PO-5 added dissolution", NOT "J-structure organises the graph",
and the hypothesis is NOT supported whatever the component count does.
"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
sys.path.insert(0, SWEEP_DIR)
from po5_unit5_sheaf_laplacian import sheaf_cohomology, ordinary_topology

BUS = [0.5, 20.0, None]
SEEDS = [101, 102, 103]
T_SIM, DT = 1.0, 0.005

def run(arm, bus, seed):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses: s.set_microtubule_invasion(True)
    dp = net.synapses[0].dimer_particles
    dp.j_mismatch_dissolution = (arm != "OFF")
    dp.j_mismatch_scramble = (arm == "SCRAMBLED")
    o_step = dp.step; native = []
    def w(*a, **k):
        if "collective_field_kT" in k:
            native.append(float(k["collective_field_kT"]))
            if bus is not None: k["collective_field_kT"] = bus
        return o_step(*a, **k)
    dp.step = w
    rel = PresynapticRelease(seed=seed); t0 = time.time()
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; idset = set(ids)
    pos = {d.id: np.asarray(d.position, float) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in idset and b in idset]
    c, cyc = ordinary_topology(ids, edges)
    sh = sheaf_cohomology(ids, pos, edges, "geometry")
    return {"arm": arm, "bus": bus, "seed": seed, "V": len(ids), "E": len(edges),
            "components": c, "cycle_rank": cyc, "sheaf_H0_engaged": sh["H0_engaged"],
            "native_bus": float(np.mean(native)) if native else None,
            "elapsed_s": time.time() - t0}

def main():
    print("=" * 96)
    print("PO-5 UNIT 6 — J-mismatch dissolution: A=OFF  B=REAL  C=SCRAMBLED(control)")
    print("=" * 96, flush=True)
    rows = []
    for bus in BUS:
        lbl = "NATIVE" if bus is None else f"{bus:.1f}"
        print(f"\n--- bus={lbl} ---", flush=True)
        for arm in ("OFF", "REAL", "SCRAMBLED"):
            cs, hs, es = [], [], []
            for sd in SEEDS:
                r = run(arm, bus, sd); rows.append(r)
                cs.append(r["components"]); hs.append(r["sheaf_H0_engaged"]); es.append(r["E"])
            print(f"  {arm:10s} components {np.mean(cs):8.1f} +/- {np.std(cs):5.1f}   "
                  f"bonds {np.mean(es):9.0f}   sheafH0 {np.mean(hs):5.1f} +/- {np.std(hs):4.1f}",
                  flush=True)
            with open(os.path.join(SWEEP_DIR, "po5_unit6_results.json"), "w") as f:
                json.dump(rows, f, indent=2)
    print("\n" + "=" * 96)
    for bus in BUS:
        lbl = "NATIVE" if bus is None else f"{bus:.1f}"
        g = lambda a: np.array([r["components"] for r in rows if r["arm"] == a and r["bus"] == bus])
        A, B, C = g("OFF"), g("REAL"), g("SCRAMBLED")
        spread = max(np.std(A), np.std(B), np.std(C), 1e-9)
        b_vs_a = (B.mean() - A.mean()) / spread
        b_vs_c = (B.mean() - C.mean()) / spread
        print(f"bus={lbl:>6s}  A={A.mean():7.1f}  B={B.mean():7.1f}  C={C.mean():7.1f}   "
              f"(B-A)/spread={b_vs_a:6.2f}  (B-C)/spread={b_vs_c:6.2f}")
        if abs(b_vs_a) < 1.0:
            print("          -> NOT SUPPORTED (no effect): B ~= A, the rule changes nothing")
        elif abs(b_vs_c) < 1.0:
            print("          -> NOT SUPPORTED (generic): B ~= C, this is added dissolution,")
            print("             NOT J-structure organising the graph")
        else:
            print("          -> SUPPORTED at this bus: B differs from BOTH A and C")
    print("\nLIMITS: single synapse, 1 s, 3 seeds. A SUPPORTED verdict raises the physics")
    print("question (does intra-dimer J gate inter-dimer entanglement?) — it does not settle it.")

if __name__ == "__main__":
    main()
