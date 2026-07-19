#!/usr/bin/env python3
"""
PO-5 UNIT 7 — where is the percolation threshold actually?
Pre-registered: docs/PREREG_PO5_UNIT7_CRITICAL_POINT.md.

P1 self-correction: largest_frac >= 0.85 at EVERY bus incl. 0 => the bus does NOT form the
                    giant component, and L-PO5-5's framing is corrected.
P2: some birth_window gives largest_frac < 0.5 -- the real transition.
P3: susceptibility chi = sum(s^2)/sum(s) over FINITE clusters PEAKS there, not at any bus.
"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
T_SIM, DT = 1.0, 0.005
SEEDS = [11, 12]

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
    n = len(ids)
    fin = s[1:]                       # finite clusters = all but the largest
    chi = (sum(x * x for x in fin) / sum(fin)) if fin else 0.0
    return len(s), (s[0] / n if n else 0), chi, (s[1] if len(s) > 1 else 0)

def run(bus, bw, seed):
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
    dp.birth_window = bw
    o = dp.step; nat = []
    def w(*a, **k):
        if "collective_field_kT" in k:
            nat.append(float(k["collective_field_kT"]))
            if bus is not None: k["collective_field_kT"] = bus
        return o(*a, **k)
    dp.step = w
    rel = PresynapticRelease(seed=seed); t0 = time.time()
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
    ent = [d.id for d in dp.dimers if d.is_entangled]; es = set(ent)
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    c, lf, chi, s2 = clusters(ent, edges)
    na = np.array(nat) if nat else np.array([0.0])
    return {"bus": bus, "bw": bw, "seed": seed, "V": len(ent), "E": len(edges),
            "components": c, "largest_frac": lf, "chi": chi, "second_largest": s2,
            "native_mean": float(na.mean()), "native_std": float(na.std()),
            "native_min": float(na.min()), "native_max": float(na.max()),
            "elapsed_s": time.time() - t0}

def block(title, combos):
    print(f"\n--- {title} ---", flush=True)
    out = []
    for bus, bw in combos:
        rs = [run(bus, bw, s) for s in SEEDS]
        out += rs
        lf = np.mean([r["largest_frac"] for r in rs]); chi = np.mean([r["chi"] for r in rs])
        c = np.mean([r["components"] for r in rs]); s2 = np.mean([r["second_largest"] for r in rs])
        bl = "NATIVE" if bus is None else f"{bus:g}"
        print(f"  bus={bl:>6s} bw={bw:<6g} V={rs[0]['V']:5d} comps={c:6.1f} "
              f"largest_frac={lf:.4f} 2nd={s2:6.1f} chi={chi:8.2f}", flush=True)
    return out

def main():
    print("=" * 92)
    print("PO-5 UNIT 7 — locating the percolation threshold (chi peaks AT it)")
    print("=" * 92, flush=True)
    rows = []
    rows += block("ARM BUS (bw=0.1 native): does the bus form the giant component?",
                  [(b, 0.1) for b in (0.0, 1.0, 10.0, None)])
    rows += block("ARM BW @ bus=0 (P0 isolated): where does P0 percolate?",
                  [(0.0, w) for w in (0.002, 0.01, 0.05, 0.1)])
    rows += block("ARM BW @ NATIVE bus: does P2 rescue a fragmented P0?",
                  [(None, w) for w in (0.002, 0.01, 0.05, 0.1)])
    with open(os.path.join(SWEEP_DIR, "po5_unit7_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 92)
    busr = [r for r in rows if r["bw"] == 0.1]
    p1 = all(r["largest_frac"] >= 0.85 for r in busr)
    print(f"P1 {'CONFIRMED' if p1 else 'REJECTED'}: largest_frac >= 0.85 at every bus "
          f"(min {min(r['largest_frac'] for r in busr):.4f})")
    if p1:
        print("   => the BUS does not form the giant component. L-PO5-5's framing of it as")
        print("      'the percolation control parameter' is CORRECTED: it absorbs stragglers.")
    for lbl, bus in (("bus=0", 0.0), ("NATIVE", None)):
        arm = [r for r in rows if r["bus"] == bus and r["bw"] != 0.1 or (r["bus"] == bus and r["bw"] == 0.1)]
        arm = sorted({r["bw"] for r in arm})
        print(f"\n   {lbl}:")
        best, bchi = None, -1
        for w in arm:
            rs = [r for r in rows if r["bus"] == bus and r["bw"] == w]
            if not rs: continue
            lf = np.mean([r["largest_frac"] for r in rs]); chi = np.mean([r["chi"] for r in rs])
            frag = "  <-- FRAGMENTED (largest_frac<0.5)" if lf < 0.5 else ""
            if chi > bchi: bchi, best = chi, w
            print(f"     bw={w:<6g} largest_frac={lf:.4f} chi={chi:8.2f}{frag}")
        print(f"     chi peaks at bw={best}")
    nat = [r for r in rows if r["bus"] is None]
    if nat:
        print(f"\n   NATIVE bus distribution: mean {np.mean([r['native_mean'] for r in nat]):.3f} "
              f"std {np.mean([r['native_std'] for r in nat]):.3f} "
              f"range [{min(r['native_min'] for r in nat):.3f}, "
              f"{max(r['native_max'] for r in nat):.3f}]")
    print("\nLIMITS: single synapse, 1 s, 2 seeds/point.")

if __name__ == "__main__":
    main()
