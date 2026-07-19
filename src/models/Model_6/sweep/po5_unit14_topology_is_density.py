#!/usr/bin/env python3
"""
PO-5 UNIT 14 — is the topology a FUNCTION OF DENSITY ALONE?

THE BIND (measured, not assumed):
  - Unit 13: drive PATTERN does not reach the dimer population. SUSTAINED and PULSED gave
    BIT-IDENTICAL V, positions and topology on seeds 72/73. `target_count` derives from
    `peak_conc = np.max(dimer_concentration)` -- a PEAK, not an integral -- and both protocols
    hit the same peak in their first burst. With write-once bonds the population never revisits.
  - Unit 10: drive AMPLITUDE does reach it (V spanned 1065-1163).

So the only input dimension with purchase on the graph is amplitude, and amplitude acts through
density. That makes the keystone question sharp and answerable:

    Is the topology a function of V alone, or does it carry residual structure?

  function of V alone  -> the partition carries exactly what a scalar carries. This is
                          LITERALLY §8's "scalar as computation". Keystone fails, cleanly,
                          with a contrast that actually varies -- unlike Units 9/10/13.
  residual structure   -> something beyond density survives; the keystone has a channel.

PRE-REGISTERED:
  P1 amplitude must move V (else this contrast fails like the pattern one did).
  P2 fit topology ~ V by least squares across amplitudes. Compare the FIT RESIDUAL scatter to
     the WITHIN-CELL (seed) scatter:
        ratio = SD(residual) / SD(within-cell)
        ratio <= 1.5 -> topology is a function of density alone. §8 FAILS.
        ratio >= 3.0 -> residual structure beyond density. §8 has a channel.
        between      -> INCONCLUSIVE.
  P3 pattern arm at matched amplitude must reproduce Unit 13's null (confirms the bind).
  Thresholds fixed here; 3 seeds/cell, and the ratio is scale-free so it does not inherit
  Unit 9's small-n effect-size fragility.
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
AMPS = [0.25, 0.40, 0.55, 0.70, 0.85, 0.95]
SEEDS = [81, 82, 83]
CUT = 0.50

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

def run(amp, seed, pulsed=False):
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
    rel = PresynapticRelease(seed=seed); t = 0.0
    for _ in range(int(round(T_SIM/DT))):
        a = amp if not pulsed else (amp if (t % 0.6) < 0.2 else 0.0)
        g = rel.step(a, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        t += DT
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; es = set(ids)
    st = {d.id: (np.asarray(d.position, float), d.singlet_probability) for d in ent}
    edges = [(a, b) for (a, b) in dp._bond_lookup if a in es and b in es]
    L = dp.coupling_length
    kept = []
    for (a, b) in edges:
        r = float(np.linalg.norm(st[a][0] - st[b][0]))
        if st[a][1]*st[b][1]*(L/max(r, L))**3 >= CUT: kept.append((a, b))
    c0, lf0 = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0)
    c1, lf1 = clusters(ids, kept) if len(ids) >= 2 else (len(ids), 1.0)
    h0 = sheaf_cohomology(ids, {i: st[i][0] for i in ids}, kept, "geometry")["H0_engaged"] \
         if len(ids) >= 2 else 0
    return {"amp": amp, "seed": seed, "pulsed": pulsed, "V": len(ids), "E": len(edges),
            "comps_raw": c0, "lf_raw": lf0, "comps_cut": c1, "lf_cut": lf1, "H0_cut": h0}

def main():
    print("="*98); print("PO-5 UNIT 14 — is the topology a function of DENSITY ALONE?"); print("="*98, flush=True)
    rows = []
    print(f"\nAMPLITUDE SWEEP (cut={CUT}):")
    print(f"  {'amp':>5s} {'V':>7s} {'comps_cut':>10s} {'lf_cut':>8s} {'H0_cut':>7s}")
    for amp in AMPS:
        rs = [run(amp, s) for s in SEEDS]; rows += rs
        print(f"  {amp:5.2f} {np.mean([r['V'] for r in rs]):7.1f} "
              f"{np.mean([r['comps_cut'] for r in rs]):10.1f} "
              f"{np.mean([r['lf_cut'] for r in rs]):8.4f} "
              f"{np.mean([r['H0_cut'] for r in rs]):7.1f}", flush=True)
    print("\nPATTERN ARM at amp=0.95 (P3 — must reproduce Unit 13's null):")
    for pul in (False, True):
        rs = [run(0.95, s, pulsed=pul) for s in SEEDS]; rows += rs
        print(f"  {'PULSED' if pul else 'SUSTAINED':10s} V={np.mean([r['V'] for r in rs]):7.1f} "
              f"comps_cut={np.mean([r['comps_cut'] for r in rs]):6.1f}", flush=True)
    with open(os.path.join(SWEEP_DIR, "po5_unit14_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    sweep = [r for r in rows if not r["pulsed"] and r["amp"] in AMPS]
    V = np.array([r["V"] for r in sweep], float)
    print("\n" + "="*98)
    print(f"P1 amplitude moved V over [{V.min():.0f}, {V.max():.0f}]  "
          f"{'(OK — the contrast varies)' if V.max()-V.min() > 30 else '(FAILS like the pattern arm)'}")
    print("\nP2 — is topology a function of V alone?")
    for key in ("comps_cut", "lf_cut", "H0_cut", "comps_raw"):
        y = np.array([r[key] for r in sweep], float)
        sl, ic = np.polyfit(V, y, 1)
        resid = y - (sl*V + ic)
        within = []
        for amp in AMPS:
            g = np.array([r[key] for r in sweep if r["amp"] == amp], float)
            within.append(g.std(ddof=1))
        wsd = float(np.mean(within)); rsd = float(np.std(resid, ddof=2))
        ratio = rsd/wsd if wsd > 1e-12 else float('nan')
        verdict = ("DENSITY ALONE" if ratio <= 1.5 else
                   "RESIDUAL STRUCTURE" if ratio >= 3.0 else "inconclusive")
        print(f"  {key:11s} slope={sl:+.5f}  resid_SD={rsd:7.3f}  seed_SD={wsd:7.3f}  "
              f"ratio={ratio:5.2f}  -> {verdict}")
    print("\n" + "="*98)
    print("READING: ratio<=1.5 on the topology measures => the partition carries exactly what a")
    print("scalar (V) carries. That is literally §8's 'scalar as computation', established with a")
    print("contrast that DOES vary — unlike Units 9/10/13, whose pattern contrast did not reach")
    print("the population at all.")
    print("\nLIMITS: 1 synapse, 1 s, 3 seeds/cell. F is the rate-derived form (known category")
    print("error); the cut is used as a fixed lens, not as a nominated threshold.")

if __name__ == "__main__":
    main()
