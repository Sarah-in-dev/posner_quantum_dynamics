#!/usr/bin/env python3
"""
PO-5 UNIT 13 — is the fidelity cut a real lever, or cosmetic?

Unit 12 found the ONLY intervention that fragments the graph (largest_frac 1.0000 -> 0.8314,
sheaf H0 3 -> 15). But it fragments by removing LONG-RANGE bonds -- a geometric criterion --
and every Unit 12 run used ONE drive condition. PO-5 named this as the single most likely way
it is wrong.

THE COVARIATE IS SPATIAL, NOT DENSITY (advisor, round 2). A cut on g(r_ij) gives a partition
determined by the distance matrix. That is input-blind IF positions are input-independent. But
dimers are born from a calcium concentration field, so there is a live path:
    input -> Ca spatial pattern -> birth positions -> g(r_ij) -> partition
So the question is not "does density differ" but "does the SPATIAL STRUCTURE of births differ".

PRE-REGISTERED:
  P1 (the covariate). Do the two drive conditions produce different spatial statistics?
     Scored on the pairwise-separation distribution: |median_SUS - median_PUL| / pooled seed SD.
  P2 (the topology). Does the partition after the cut differ between conditions?
     Same effect-size form, on components / largest_frac / sheaf H0.
  READING:
     P1 null  AND P2 null  -> the cut is COSMETIC. Geometry is input-blind; the lever is real
                              but carries nothing. This is the outcome PO-5 flagged as most likely.
     P1 diff AND P2 diff  -> geometry IS input-shaped; the cut is a live channel.
     P1 null  AND P2 diff  -> something other than birth geometry carries it. Interesting; would
                              need a separate explanation and is NOT claimed as keystone support.
  Bar: d >= 2.0 differs, d <= 1.0 null, between = inconclusive. 5 SEEDS per condition -- Unit 9
  reported an effect off 3 seeds that reversed sign on replication.

NO threshold is nominated. F is the Unit-12 rate-derived form and is a KNOWN CATEGORY ERROR
(advisor: fidelity should come from state, F(t)=1/4+(F0-1/4)exp(-t/T2)). This unit therefore
tests whether the LEVER moves with input, not whether the threshold is correct.
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
DT, T_SIM, SEEDS = 0.005, 1.0, [71, 72, 73, 74, 75]
THRESHOLDS = [0.20, 0.30, 0.50]

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

def run(kind, seed):
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
    t = 0.0
    for _ in range(int(round(T_SIM/DT))):
        a = 0.95 if kind == "SUSTAINED" else (0.95 if (t % 0.6) < 0.2 else 0.0)
        g = rel.step(a, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        t += DT
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; es = set(ids)
    st = {d.id: (np.asarray(d.position, float), d.singlet_probability) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    L = dp.coupling_length
    # --- P1 covariate: spatial statistics of the dimer cloud itself ---
    pos = np.array([st[i][0] for i in ids])
    iu, ju = np.triu_indices(len(ids), k=1)
    sub = np.random.default_rng(0).choice(len(iu), size=min(200000, len(iu)), replace=False)
    rr = np.linalg.norm(pos[iu[sub]] - pos[ju[sub]], axis=1)
    spatial = {"r_med": float(np.median(rr)), "r_p10": float(np.percentile(rr, 10)),
               "r_p90": float(np.percentile(rr, 90))}
    F = {}
    for (a, b) in edges:
        r = float(np.linalg.norm(st[a][0] - st[b][0]))
        F[(a, b)] = st[a][1] * st[b][1] * (L / max(r, L)) ** 3
    out = {"input": kind, "seed": seed, "V": len(ids), "E": len(edges), **spatial}
    for th in THRESHOLDS:
        kept = [e for e in edges if F[e] >= th]
        c, lf = clusters(ids, kept) if len(ids) >= 2 else (len(ids), 1.0)
        h0 = sheaf_cohomology(ids, {i: st[i][0] for i in ids}, kept, "geometry")["H0_engaged"] \
             if len(ids) >= 2 else 0
        out[f"comps@{th}"] = c; out[f"lf@{th}"] = lf; out[f"H0@{th}"] = h0
    return out

def eff(rows, key):
    A = np.array([r[key] for r in rows if r["input"] == "SUSTAINED"], float)
    B = np.array([r[key] for r in rows if r["input"] == "PULSED"], float)
    pooled = np.sqrt((A.std(ddof=1)**2 + B.std(ddof=1)**2)/2)
    d = abs(A.mean()-B.mean())/pooled if pooled > 1e-12 else float('nan')
    return A.mean(), B.mean(), d

def main():
    print("="*100); print("PO-5 UNIT 13 — is the fidelity cut real or cosmetic? (spatial covariate, 5 seeds)")
    print("="*100, flush=True)
    rows = []
    for kind in ("SUSTAINED", "PULSED"):
        for s in SEEDS:
            r = run(kind, s); rows.append(r)
            print(f"  {kind:10s} seed {s}  V={r['V']:5d} r_med={r['r_med']:6.2f}nm  "
                  f"comps@0.5={r['comps@0.5']:4d} lf@0.5={r['lf@0.5']:.4f}", flush=True)
    with open(os.path.join(SWEEP_DIR, "po5_unit13_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "="*100)
    print("P1 — SPATIAL COVARIATE (does birth geometry differ by input?)")
    p1 = []
    for k in ("r_med", "r_p10", "r_p90", "V"):
        a, b, d = eff(rows, k); p1.append(d)
        print(f"    {k:8s} SUS {a:9.3f}  PUL {b:9.3f}   d={d:6.2f}")
    print("\nP2 — TOPOLOGY AFTER THE CUT")
    p2 = []
    for th in THRESHOLDS:
        print(f"  threshold {th}:")
        for k in (f"comps@{th}", f"lf@{th}", f"H0@{th}"):
            a, b, d = eff(rows, k); p2.append(d)
            print(f"    {k:12s} SUS {a:9.3f}  PUL {b:9.3f}   d={d:6.2f}")
    m1, m2 = max(p1), max(p2)
    print("\n" + "="*100)
    print(f"max effect size — spatial {m1:.2f} | topology {m2:.2f}")
    if m1 <= 1.0 and m2 <= 1.0:
        print("VERDICT: COSMETIC. Birth geometry does not differ by input and neither does the")
        print("  partition after the cut. The lever fragments the graph but carries no input")
        print("  information. This is the outcome PO-5 flagged as most likely.")
    elif m1 >= 2.0 and m2 >= 2.0:
        print("VERDICT: LIVE CHANNEL. Birth geometry IS input-shaped and the partition follows.")
    elif m1 <= 1.0 and m2 >= 2.0:
        print("VERDICT: ANOMALOUS — topology moves without spatial statistics moving. Needs a")
        print("  separate explanation; NOT claimed as keystone support.")
    else:
        print("VERDICT: INCONCLUSIVE between the registered bounds.")
    print("\nLIMITS: 1 synapse, 1 s, 5 seeds. F is the rate-derived form and is a KNOWN category")
    print("error (advisor round 2); this tests the LEVER, not the threshold.")

if __name__ == "__main__":
    main()
