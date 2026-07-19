#!/usr/bin/env python3
"""
PO-5 UNIT 9 — §8's KEYSTONE, finally askable: in a regime where structure SURVIVES,
does INPUT modulate the topology?

WHY THIS IS DIFFERENT FROM UNIT 2's Q-B
---------------------------------------
Q-B asked "does input change the bond set?" at the NATIVE operating point -- where Unit 4/7/8
have since shown the graph is a single blob from the first measurable instant. A readout pinned
at 1 component cannot show an effect of anything. Q-B was asking a good question of a saturated
instrument.

Unit 7 measured the regime where structure survives: BOTH levers must be reduced (P0's window
below ~10 ms AND the field low), because each percolates independently. This unit puts the
system there and only then varies the input.

THE SWEEP IS A RESPONSE CURVE, NOT A SEARCH FOR A VALUE
-------------------------------------------------------
`model6-research-findings-may29:66` records birth_window as "Tunable parameter -- candidate for
TALON sweep, not arbitrary calibration", grounded only by an UPPER bound (Fisher ~1 s). This
reports the response across the permitted band. It does NOT nominate a correct value, and a
window that yields nice topology is NOT thereby endorsed.

PRE-REGISTERED (before the run)
-------------------------------
  Effect size  d = |mean(SUSTAINED) - mean(PULSED)| / pooled_std(within condition, across seeds)
  computed on: components, largest_frac, sheaf H0_engaged.

  P1 SATURATION CHECK: at bw=0.1 (native) with bus low, if largest_frac > 0.95 the readout is
     saturated there and no effect can be detected AT THAT POINT regardless of the physics.
     Reported so a null at native is not misread as a physics result.
  P2 KEYSTONE SUPPORTED: in at least one unsaturated cell (largest_frac < 0.9), d >= 2.0 on at
     least one measure.
  P3 KEYSTONE NOT SUPPORTED: every unsaturated cell gives d <= 1.0. Then which dimers bond does
     NOT depend on input even where the graph CAN show structure -- §8's keystone fails on its
     own terms, and that is reported as a finding, not a protocol problem.
  Between 1.0 and 2.0 -> INCONCLUSIVE at that cell.

  The seed spread IS the null: same input, different RNG. No arm is "silent" -- both drive
  conditions are fully live (BASELINE_RATE_HZ=0.5 makes silence unavailable anyway).
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

DT, T_SIM = 0.005, 1.0
BUS = [0.0, 1.0]
BW = [0.002, 0.01, 0.1]
SEEDS = [31, 32, 33]

def drive(kind, t):
    if kind == "SUSTAINED":
        return 0.95
    return 0.95 if (t % 0.6) < 0.2 else 0.0     # matched total is not required here;
                                               # the contrast is PATTERN (Unit 3 showed it
                                               # changes burst structure)

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
    s = sorted(sz.values(), reverse=True); n = len(ids)
    return len(s), (s[0]/n if n else 0.0)

def run(bus, bw, kind, seed):
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
    dp.birth_window = bw
    o = dp.step
    def w(*a, **k):
        if "collective_field_kT" in k and bus is not None:
            k["collective_field_kT"] = bus
        return o(*a, **k)
    dp.step = w
    rel = PresynapticRelease(seed=seed)
    t = 0.0; mg = 0.0
    for _ in range(int(round(T_SIM/DT))):
        a = drive(kind, t)
        g = rel.step(a, DT); mg = max(mg, g)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        t += DT
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; es = set(ids)
    pos = {d.id: np.asarray(d.position, float) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    c, lf = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0)
    sh = sheaf_cohomology(ids, pos, edges, "geometry")["H0_engaged"] if len(ids) >= 2 else 0
    return {"bus": bus, "bw": bw, "input": kind, "seed": seed, "V": len(ids), "E": len(edges),
            "components": c, "largest_frac": lf, "sheaf_H0": sh, "max_glu": mg}

def main():
    print("=" * 100)
    print("PO-5 UNIT 9 — does INPUT modulate topology where structure SURVIVES?")
    print("=" * 100, flush=True)
    rows = []
    for bus in BUS:
        for bw in BW:
            cell = {}
            for kind in ("SUSTAINED", "PULSED"):
                rs = [run(bus, bw, kind, s) for s in SEEDS]
                rows += rs; cell[kind] = rs
            def stat(k, key):
                return np.array([r[key] for r in cell[k]], float)
            lf = np.mean([r["largest_frac"] for r in cell["SUSTAINED"] + cell["PULSED"]])
            print(f"\nbus={bus:g} bw={bw:g}   largest_frac={lf:.4f}"
                  f"{'   [SATURATED - no effect detectable here]' if lf > 0.95 else ''}")
            for key in ("components", "largest_frac", "sheaf_H0"):
                A, B = stat("SUSTAINED", key), stat("PULSED", key)
                pooled = np.sqrt((A.std(ddof=1)**2 + B.std(ddof=1)**2) / 2) if len(A) > 1 else 0.0
                d = abs(A.mean() - B.mean()) / pooled if pooled > 1e-12 else float('nan')
                print(f"    {key:14s} SUS {A.mean():8.3f}+/-{A.std(ddof=1):6.3f}   "
                      f"PUL {B.mean():8.3f}+/-{B.std(ddof=1):6.3f}   d={d:6.2f}")
            with open(os.path.join(SWEEP_DIR, "po5_unit9_results.json"), "w") as f:
                json.dump(rows, f, indent=2)

    print("\n" + "=" * 100)
    unsat, best = [], 0.0
    for bus in BUS:
        for bw in BW:
            cell = [r for r in rows if r["bus"] == bus and r["bw"] == bw]
            lf = np.mean([r["largest_frac"] for r in cell])
            if lf >= 0.9:
                continue
            unsat.append((bus, bw, lf))
            for key in ("components", "largest_frac", "sheaf_H0"):
                A = np.array([r[key] for r in cell if r["input"] == "SUSTAINED"], float)
                B = np.array([r[key] for r in cell if r["input"] == "PULSED"], float)
                pooled = np.sqrt((A.std(ddof=1)**2 + B.std(ddof=1)**2) / 2)
                if pooled > 1e-12:
                    best = max(best, abs(A.mean() - B.mean()) / pooled)
    if not unsat:
        print("NO UNSATURATED CELL — every tested combination is a blob (largest_frac >= 0.9).")
        print("The keystone cannot be tested here; this is a statement about the INSTRUMENT.")
    else:
        print(f"unsaturated cells: {[(b, w, round(l,3)) for b, w, l in unsat]}")
        print(f"largest effect size across them: d = {best:.2f}")
        if best >= 2.0:
            print("P2 KEYSTONE SUPPORTED at >=1 unsaturated cell: input modulates the topology")
            print("   where the graph can show structure.")
        elif best <= 1.0:
            print("P3 KEYSTONE NOT SUPPORTED: even where the graph CAN show structure, which")
            print("   dimers bond does NOT depend on input beyond seed noise. §8 fails on its")
            print("   own terms. Reported as a finding.")
        else:
            print("INCONCLUSIVE: effect between seed noise and the registered bar.")
    print("\nLIMITS: single synapse, 1 s, 3 seeds/cell. bus is a diagnostic OVERRIDE (Tier 2);")
    print("bw is swept across its permitted band and NO value is nominated as correct.")

if __name__ == "__main__":
    main()
