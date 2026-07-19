#!/usr/bin/env python3
"""
PO-7 UNIT 6 — synchrony vs stagger at matched density (advisor R4 step 2).

Pre-registered in docs/PREREG_PO7_UNIT6_SYNCHRONY.md. Read that first; this implements it.

Density is matched BY CONSTRUCTION: every synapse receives an identical burst train in both
conditions (0.1 s on per 0.4 s period). Only the PHASE differs — whether neighbouring synapses
are elevated at the same time. Adjacent synapses (the only pairs close enough to bond at
0.2 um) are always in different groups, so stagger removes their co-elevation entirely.

Timescale is set by Unit 5's MEASURED coincidence window (<=50 ms), not by guesswork:
the 200 ms stagger is a 4x margin over it.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

DT = 0.005
TRACKER_EVERY = 10
SPACING, N_SYN = 0.2, 6
BURST_S, PERIOD_S, N_PERIODS = 0.1, 0.4, 3
STAGGER_S = 0.2                      # 4x the measured <=50 ms coincidence window
T_SIM = PERIOD_S * N_PERIODS
SEEDS = [31337, 4242, 90210, 7, 123456]
GROUP_A, GROUP_B = {0, 2, 4}, {1, 3, 5}
DENSITY_TOL = 0.10                   # pre-registered: >10% dimer-count gap => INVALID


def is_on(syn_idx, t, condition):
    """Identical burst train in both conditions; only the PHASE of group B differs."""
    phase = t % PERIOD_S
    off = 0.0 if (syn_idx in GROUP_A or condition == "SYNC") else STAGGER_S
    lo = off
    return lo <= phase < lo + BURST_S


def run_one(condition, seed):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    tr = net.entanglement_tracker
    tr.provenance_network = True

    rel = PresynapticRelease(seed=seed)
    on_steps = [0] * N_SYN
    for i in range(int(round(T_SIM / DT))):
        t = i * DT
        g = rel.step(0.95, DT)
        for j, s in enumerate(net.synapses):
            if is_on(j, t, condition):
                on_steps[j] += 1
                s.step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})
            else:
                s.step(DT, {"voltage": -70e-3, "reward": False, "glutamate": 0.0})
        if (i + 1) % TRACKER_EVERY == 0:
            tr.step(DT * TRACKER_EVERY, net.synapses, net.positions,
                    coupling_weights=getattr(net, "coupling_weights", None))

    n_cross = sum(1 for (a, b) in tr._prov_bonds if a[0] != b[0])
    comps = tr._find_all_clusters()
    n_multi = sum(1 for c in comps if len({g0[0] for g0 in c}) >= 2)
    return {'condition': condition, 'seed': seed, 'n_cross': n_cross, 'n_multi': n_multi,
            'n_prov_bonds': len(tr._prov_bonds), 'n_dimers': len(tr.all_dimers),
            'on_steps_per_synapse': on_steps}


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sqrt(((x.var(ddof=1) if len(x) > 1 else 0.0) +
                 (y.var(ddof=1) if len(y) > 1 else 0.0)) / 2.0)
    if s == 0:
        return 0.0 if abs(x.mean() - y.mean()) < 1e-12 else float('inf')
    return float((x.mean() - y.mean()) / s)


def verdict(d, density_ok):
    if not density_ok:
        return ("INVALID — density check FAILED: dimer counts differ between conditions, so this "
                "repeats Unit 2's confound. Not scored."), False
    if d >= 0.8:
        return ("POSITIVE — coincidence detection: the partition depends on TEMPORAL input "
                "structure at matched density."), True
    if d <= -0.8:
        return ("ANOMALY — stagger exceeds synchrony; the mechanism gives no account of this. "
                "Reported as anomaly, not as a result."), False
    if abs(d) < 0.3:
        return ("NEGATIVE — no coincidence dependence detected (|d| < 0.3). The Unit-1b ceiling "
                "reads as a power limit, not a temporal window. NOTE: the 15%-unclaimed event "
                "leak biases toward exactly this outcome."), False
    return (f"INCONCLUSIVE — |d|={abs(d):.2f} falls between the registered thresholds "
            "(0.3 / 0.8). No verdict claimed."), False


def main():
    print(f"PO-7 UNIT 6 — synchrony vs stagger, matched density")
    print(f"  {N_SYN} synapses @ {SPACING}um | burst {BURST_S}s / period {PERIOD_S}s x {N_PERIODS}"
          f" | stagger {STAGGER_S}s (4x the measured <=50ms window)")
    print(f"  A={sorted(GROUP_A)} B={sorted(GROUP_B)} (adjacent synapses always in DIFFERENT groups)\n")
    rows, agg = [], {}
    for cond in ("SYNC", "STAGGER"):
        for sd in SEEDS:
            r = run_one(cond, sd); rows.append(r)
            print(f"  {cond:<8} seed={sd:>6}  n_cross={r['n_cross']:>3} n_multi={r['n_multi']:>3} "
                  f"prov={r['n_prov_bonds']:>4} dimers={r['n_dimers']:>5} "
                  f"on_steps={r['on_steps_per_synapse']}")
            sys.stdout.flush()
        sub = [r for r in rows if r['condition'] == cond]
        agg[cond] = {'cross': [r['n_cross'] for r in sub],
                     'multi': [r['n_multi'] for r in sub],
                     'dimers': [r['n_dimers'] for r in sub]}
        print(f"  -> {cond}: mean n_cross={np.mean(agg[cond]['cross']):.2f} "
              f"mean n_multi={np.mean(agg[cond]['multi']):.2f} "
              f"mean dimers={np.mean(agg[cond]['dimers']):.0f}\n")

    ds, dg = np.mean(agg['SYNC']['dimers']), np.mean(agg['STAGGER']['dimers'])
    gap = abs(ds - dg) / max(ds, dg)
    density_ok = gap <= DENSITY_TOL
    print("=== DENSITY CHECK (the thing Unit 2 lacked) ===")
    print(f"  mean dimers SYNC={ds:.0f}  STAGGER={dg:.0f}  gap={100*gap:.2f}% "
          f"(tolerance {100*DENSITY_TOL:.0f}%)  -> {'PASS' if density_ok else 'FAIL'}")

    d_cross = cohens_d(agg['SYNC']['cross'], agg['STAGGER']['cross'])
    d_multi = cohens_d(agg['SYNC']['multi'], agg['STAGGER']['multi'])
    print(f"\n=== EFFECT SIZE (SYNC vs STAGGER) ===")
    print(f"  d(n_cross) = {d_cross:+.3f}")
    print(f"  d(n_multi) = {d_multi:+.3f}")
    v, ok = verdict(d_cross, density_ok)
    print(f"\n=== PRE-REGISTERED VERDICT ===\n  {v}")

    out = {'rows': rows, 'd_cross': d_cross, 'd_multi': d_multi,
           'density_gap': gap, 'density_ok': bool(density_ok),
           'mean_dimers': {'SYNC': float(ds), 'STAGGER': float(dg)},
           'mean_cross': {k: float(np.mean(agg[k]['cross'])) for k in agg},
           'mean_multi': {k: float(np.mean(agg[k]['multi'])) for k in agg},
           'verdict': v, 'passed': bool(ok),
           'prereg': 'docs/PREREG_PO7_UNIT6_SYNCHRONY.md',
           'config': {'spacing_um': SPACING, 'burst_s': BURST_S, 'period_s': PERIOD_S,
                      'n_periods': N_PERIODS, 'stagger_s': STAGGER_S, 'seeds': SEEDS}}
    path = os.path.join(SWEEP_DIR, 'po7_unit6_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
