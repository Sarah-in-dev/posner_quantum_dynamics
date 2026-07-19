#!/usr/bin/env python3
"""
PO-5 UNIT 10 — THE DECISIVE CONTROL: is Unit 9's effect PAIR-LEVEL or just DENSITY?

Unit 9 measured, in the unsaturated regime, SUSTAINED 60.3 components vs PULSED 52.0 (d=2.92).
But PULSED also made 4.4% fewer dimers (1152 -> 1101). §8's bar is explicitly that the partition
must carry MORE THAN ACTIVE-REGION DENSITY, so Unit 9's criterion did not test the thing §8 asks.
A 4.4% population difference yielding a 14% component difference is a 3x amplification -- more
than proportional density explains, but a NONLINEAR density relation could still account for it.

THE DESIGN. Matching dimer count exactly is fragile, so density is used as a COVARIATE:
run SUSTAINED across a range of amplitudes to trace the components-vs-V curve that DENSITY ALONE
produces, then ask where PULSED lands relative to that curve.

  ON the curve   -> the effect is density -> GATE-LEVEL -> §8 says this "collapses to
                    'scalar as computation'". The keystone FAILS on its own terms.
  OFF the curve  -> the same population produces a different partition under a different input
                    pattern -> PAIR-LEVEL -> the keystone is SUPPORTED.

PRE-REGISTERED (before the run):
  P1 BRACKET CHECK. The SUSTAINED amplitudes must produce a V range that BRACKETS PULSED's V.
     If they do not, the covariate cannot be evaluated at PULSED's density and the control is
     INCONCLUSIVE -- reported, not patched by extrapolating.
  P2 Fit components ~ V by least squares on the SUSTAINED points only. Compute PULSED's residual
     from that fit, in units of the pooled within-condition SD.
       |resid| >= 2.0 SD -> PAIR-LEVEL (keystone SUPPORTED)
       |resid| <= 1.0 SD -> DENSITY (keystone NOT SUPPORTED; §8 fails on its own terms)
       otherwise         -> INCONCLUSIVE
  Thresholds fixed here and do not move after the run.

Regime: bus=0 (diagnostic override, Tier 2), bw in the unsaturated band. NO value is nominated
as correct; this is a control, not a calibration.
"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
DT, T_SIM, BUS = 0.005, 1.0, 0.0
BW = [0.002, 0.01]
AMPS = [0.35, 0.50, 0.65, 0.80, 0.95]
SEEDS = [41, 42, 43]

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

def run(bw, kind, amp, seed):
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
        if "collective_field_kT" in k: k["collective_field_kT"] = BUS
        return o(*a, **k)
    dp.step = w
    rel = PresynapticRelease(seed=seed)
    t = 0.0
    for _ in range(int(round(T_SIM/DT))):
        a = amp if kind == "SUSTAINED" else (amp if (t % 0.6) < 0.2 else 0.0)
        g = rel.step(a, DT)
        syn.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
        t += DT
    ent = [d for d in dp.dimers if d.is_entangled]
    ids = [d.id for d in ent]; es = set(ids)
    edges = [(x, y) for (x, y) in dp._bond_lookup if x in es and y in es]
    c, lf = clusters(ids, edges) if len(ids) >= 2 else (len(ids), 1.0)
    return {"bw": bw, "input": kind, "amp": amp, "seed": seed,
            "V": len(ids), "components": c, "largest_frac": lf}

def main():
    print("=" * 96)
    print("PO-5 UNIT 10 — density covariate control: is Unit 9's effect PAIR-LEVEL or DENSITY?")
    print("=" * 96, flush=True)
    rows = []
    for bw in BW:
        print(f"\n=== bw={bw:g} (bus=0) ===", flush=True)
        print(f"  {'cond':>12s} {'amp':>5s} {'V':>7s} {'components':>11s} {'lg_frac':>8s}")
        for amp in AMPS:
            rs = [run(bw, "SUSTAINED", amp, s) for s in SEEDS]; rows += rs
            print(f"  {'SUSTAINED':>12s} {amp:5.2f} {np.mean([r['V'] for r in rs]):7.1f} "
                  f"{np.mean([r['components'] for r in rs]):11.1f} "
                  f"{np.mean([r['largest_frac'] for r in rs]):8.4f}", flush=True)
        rs = [run(bw, "PULSED", 0.95, s) for s in SEEDS]; rows += rs
        print(f"  {'PULSED':>12s} {0.95:5.2f} {np.mean([r['V'] for r in rs]):7.1f} "
              f"{np.mean([r['components'] for r in rs]):11.1f} "
              f"{np.mean([r['largest_frac'] for r in rs]):8.4f}", flush=True)
        with open(os.path.join(SWEEP_DIR, "po5_unit10_results.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print("\n" + "=" * 96)
    for bw in BW:
        S = [r for r in rows if r["bw"] == bw and r["input"] == "SUSTAINED"]
        P = [r for r in rows if r["bw"] == bw and r["input"] == "PULSED"]
        Vs = np.array([r["V"] for r in S], float); Cs = np.array([r["components"] for r in S], float)
        Vp = np.mean([r["V"] for r in P]); Cp = np.mean([r["components"] for r in P])
        print(f"\nbw={bw:g}:  SUSTAINED V range [{Vs.min():.0f}, {Vs.max():.0f}]   PULSED V={Vp:.0f}")
        if not (Vs.min() <= Vp <= Vs.max()):
            print("  P1 BRACKET CHECK FAILED — PULSED's density is outside the SUSTAINED range.")
            print("  INCONCLUSIVE at this bw; not extrapolating.")
            continue
        print("  P1 BRACKET CHECK PASSED")
        slope, intercept = np.polyfit(Vs, Cs, 1)
        pred = slope * Vp + intercept
        resid_pts = Cs - (slope * Vs + intercept)
        sd = np.std(resid_pts, ddof=2) if len(Cs) > 2 else np.std(resid_pts)
        z = (Cp - pred) / sd if sd > 1e-12 else float('nan')
        print(f"  density curve: components = {slope:.4f}*V + {intercept:.1f}  (scatter SD {sd:.2f})")
        print(f"  at PULSED's V={Vp:.0f} density alone predicts {pred:.1f} components; "
              f"PULSED actually has {Cp:.1f}")
        print(f"  residual = {Cp-pred:+.1f} components = {z:+.2f} SD")
        if abs(z) >= 2.0:
            print("  => PAIR-LEVEL. Same density, different partition under a different input")
            print("     pattern. §8's keystone SUPPORTED at this cell.")
        elif abs(z) <= 1.0:
            print("  => DENSITY. PULSED sits on the curve density alone produces. This is")
            print("     GATE-LEVEL — the case §8 says collapses to 'scalar as computation'.")
            print("     The keystone FAILS on its own terms.")
        else:
            print("  => INCONCLUSIVE (between the registered bounds).")
    print("\nLIMITS: single synapse, 1 s, 3 seeds. bus=0 is a diagnostic override; this is a")
    print("control, not a calibration, and no birth-window value is nominated as correct.")

if __name__ == "__main__":
    main()
