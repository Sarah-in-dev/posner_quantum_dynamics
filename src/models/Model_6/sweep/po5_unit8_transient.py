#!/usr/bin/env python3
"""
PO-5 UNIT 8 — is there an informative window DURING the field's rise?

Unit 7 recorded the native collective field as mean 21.984, std 1.558, range [0.000, 22.095].
It STARTS AT ZERO and climbs. Every measurement in this program -- including every one of
PO-5's -- sampled topology at the ENDPOINT, after saturation. Nobody has looked during the rise.

Unit 4/7 established the graph is fragmented at low field (bus 0 -> 46-60 components) and a
single blob at native (1 component). If the field passes through that range every run, the
topology should be informative EARLY and saturate later.

PRE-REGISTERED (before the run):
  P1. The field rises monotonically from ~0 toward ~22 within the run. (If not, the premise
      is wrong and the unit reports that.)
  P2. There exists a sampling time at which components > 1 AND largest_frac < 0.99, i.e. an
      informative window before saturation.
  P3. If no such window exists -- if the graph is already a single blob by the first sample --
      then saturation precedes any structure and the transient hypothesis FAILS. That closes
      off the most hopeful remaining possibility and is reported as such.

No overrides. This is the system running natively; the only change is WHEN we look.
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

DT = 0.005
SAMPLES = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
SEEDS = [21, 22, 23]

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
    fin = s[1:]
    chi = (sum(x*x for x in fin) / sum(fin)) if fin else 0.0
    return len(s), (s[0]/n if n else 0.0), chi

def run(seed):
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
    rel = PresynapticRelease(seed=seed)
    out, t, nxt = [], 0.0, 0
    for _ in range(int(round(SAMPLES[-1] / DT))):
        g = rel.step(0.95, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        t += DT
        if nxt < len(SAMPLES) and t >= SAMPLES[nxt] - DT/2:
            ent = [d for d in dp.dimers if d.is_entangled]
            ids = [d.id for d in ent]; es = set(ids)
            pos = {d.id: np.asarray(d.position, float) for d in ent}
            edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
            c, lf, chi = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0, 0.0)
            sh = sheaf_cohomology(ids, pos, edges, "geometry")["H0_engaged"] if len(ids) >= 2 else 0
            out.append({"t": round(t,3), "seed": seed,
                        "field_kT": float(getattr(syn, "_collective_field_kT", 0.0)),
                        "n_dimers": len(dp.dimers), "V": len(ids), "E": len(edges),
                        "components": c, "largest_frac": lf, "chi": chi, "sheaf_H0": sh})
            nxt += 1
    return out

def main():
    print("=" * 104)
    print("PO-5 UNIT 8 — topology DURING the field's rise (no overrides; only WHEN we look)")
    print("=" * 104, flush=True)
    rows = []
    for sd in SEEDS:
        print(f"\nseed {sd}", flush=True)
        hdr = f"  {'t':>6s} {'field_kT':>9s} {'V':>6s} {'E':>8s} {'comps':>6s} {'lg_frac':>8s} {'chi':>8s} {'sheafH0':>8s}"
        print(hdr); print("  " + "-"*(len(hdr)-2), flush=True)
        for r in run(sd):
            rows.append(r)
            print(f"  {r['t']:6.3f} {r['field_kT']:9.3f} {r['V']:6d} {r['E']:8d} "
                  f"{r['components']:6d} {r['largest_frac']:8.4f} {r['chi']:8.2f} {r['sheaf_H0']:8d}",
                  flush=True)
        with open(os.path.join(SWEEP_DIR, "po5_unit8_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print("\n" + "=" * 104)
    f0 = [r["field_kT"] for r in rows if r["t"] == SAMPLES[0]]
    f1 = [r["field_kT"] for r in rows if r["t"] == SAMPLES[-1]]
    print(f"P1 field rise: {np.mean(f0):.3f} kT at t={SAMPLES[0]} -> {np.mean(f1):.3f} kT at t={SAMPLES[-1]}")
    win = [r for r in rows if r["components"] > 1 and r["largest_frac"] < 0.99]
    if win:
        ts = sorted({r["t"] for r in win})
        print(f"P2 CONFIRMED — informative window exists at t in {ts}")
        for t in ts:
            g = [r for r in win if r["t"] == t]
            print(f"    t={t}: components {np.mean([x['components'] for x in g]):.1f}, "
                  f"largest_frac {np.mean([x['largest_frac'] for x in g]):.4f}, "
                  f"field {np.mean([x['field_kT'] for x in g]):.2f} kT, "
                  f"sheafH0 {np.mean([x['sheaf_H0'] for x in g]):.1f}")
        print("    => the topology IS informative before saturation. The computation, if there")
        print("       is one, would have to be read DURING the transient, not at the endpoint.")
    else:
        print("P3 — NO informative window. The graph is saturated by the first sample.")
        print("    => saturation precedes any structure; the transient hypothesis FAILS and the")
        print("       most hopeful remaining possibility is closed off.")
    print("\nLIMITS: single synapse, 3 seeds, one drive condition. Says nothing about whether")
    print("INPUT modulates whatever structure exists in the window.")

if __name__ == "__main__":
    main()
