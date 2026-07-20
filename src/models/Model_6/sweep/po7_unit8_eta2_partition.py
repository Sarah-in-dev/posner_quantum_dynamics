#!/usr/bin/env python3
"""
PO-7 UNIT 8 — reproduce L·ETA-2's ignition rig and MEASURE THE PARTITION ON IT.

THE GAP THIS FILLS
------------------
L·ETA-2 established that the pump IGNITES on a 7-synapse @1um co-driven rig:
    NMDAR open 0.0000 -> 0.3806, r 0.3509 -> 1.6234, eta 0.0000 -> 0.2376
and then STOPPED. Nobody looked at the entanglement graph on that rig.
L·ETA-5 did measure cross edges but drove only ONE feature, so sqrt(eta_i*eta_j) = 0 by
construction and the zero it found was structural, not informative.

So the one configuration where eta is known non-zero across co-driven NEIGHBOURS has never
had its partition measured. That is this unit. k_cross ~ sqrt(eta_i*eta_j) is non-zero here
(0.2376), so cross-synapse bonds CAN form — whether they DO, and what partition results, is
open.

COMPOSED FROM (not rebuilt):
  sweep/loop_audit_2026_07_18/eta_probe.py  -- build() rig and probe_r() instrumentation.
  ⚠ That probe drives `net.step(dt, {"voltage": v, "reward": False})` — VOLTAGE ONLY, which
  is precisely the ERR-2 defect that made L·ETA-1 measure NMDARs structurally silent.
  L·ETA-2's ignition came from glutamate reaching the drivers. So glutamate is supplied here
  via PresynapticRelease, as L·ETA-2's corrected rig requires.

REPRODUCTION GATE (must pass or nothing downstream is admissible):
  r must reach ~1.6234 and eta ~0.2376 (10% tolerance). If it does not, this is NOT
  L·ETA-2's rig and no partition claim may be made from it.

PERSISTENCE (the L·PO5-3 scar: a scoring bug destroyed 58 minutes of physics):
  every tracker sample is appended to a JSONL trace as it is taken, so a crash at any point
  leaves the physics on disk to be scored offline.
"""
import sys, os, json, logging, time
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

from model6_parameters import (Model6Parameters, compute_metabolic_power, P_BASAL_W,
                               bose_einstein_occupation, hbar)

N_SYN, SPACING = 7, 1.0
DT = 5e-3
T_SIM = float(os.environ.get("PO7_U8_SECONDS", "2.0"))   # short by default; scale after costing
VOLT = -40e-3                    # subthreshold synaptic band (plateau NOT merged into the knob)
SEED = 0
PATTERN = os.environ.get("PO7_U8_PATTERN", "linear")
# PO-7 Unit 10: intra-synapse spin resolution ON. NOTE this governs INTRA bonds only
# (dimer_particles._create_bond); cross-synapse bonds are made in
# multi_synapse_network._update_entanglement and are NOT spin-accounted yet. So this
# asks the narrower question: does shattering the intra cliques alone stop ~98
# cross-bonds from percolating the network?
SPIN_RESOLVED = os.environ.get("PO7_U8_SPIN", "0") == "1"
# PO-7 Unit 15 (advisor R5 step 5): drop the clique + EM pathway. Unit 14 measured
# that they manufacture 97% of the intra graph (3.80 of 3.93 bonds/dimer); with them
# gone ~3.87 of 4 nuclei are free, so the cross-synapse channel has the budget it was
# starved of in Unit 11. provenance_bonding already replaces the clique and skips EM.
PROVENANCE = os.environ.get("PO7_U8_PROV", "0") == "1"
TRACKER_EVERY = 10
ETA2_R, ETA2_ETA = 1.6234, 0.2376
TOL = 0.10


def pc(bp):
    return bose_einstein_occupation(bp.omega_0) * hbar * (2 * np.pi * bp.omega_0) ** 2 / bp.Q


def build(n, spacing, invaded=True, pattern=None):
    """Rig from eta_probe.py:25-36. PATTERN is parameterised because eta_probe used
    "clustered", which places synapses at randn*spacing*0.5 — RANDOM geometry, so the
    coupling row-sums (hence r) differ run to run and L·ETA-2's peak r is not
    reproducible by construction. "linear" gives deterministic spacing."""
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=n, pattern=(pattern or PATTERN), spacing_um=spacing)
    net.initialize(Model6QuantumSynapse, params)
    if invaded:
        for s in net.synapses:
            s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    for s_ in net.synapses:
        if SPIN_RESOLVED:
            s_.dimer_particles.spin_resolved = True
        if PROVENANCE:
            s_.dimer_particles.provenance_bonding = True
    return net


def probe_r(net, P_c):
    """Verbatim from eta_probe.py:39-49."""
    bp = net.params.dendritic_backbone
    e = np.array([getattr(s.spine_plasticity, 'E_invasion', 0.0) for s in net.synapses])
    ca = np.array([s.calcium.channels.get_open_fraction() for s in net.synapses])
    pm = np.array([compute_metabolic_power(e[i], ca[i], bp.p_active_max_W)
                   for i in range(len(net.synapses))])
    agg = P_BASAL_W + net.coupling_weights @ (pm - P_BASAL_W)
    r = agg / P_c
    eta_native = np.array([(x - 1) / (x + 1) if x >= 1.0 else 0.0 for x in r])
    eta_set = np.array([getattr(s, '_backbone_eta', np.nan) for s in net.synapses])
    return e, ca, r, eta_native, eta_set


def main():
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    net = build(N_SYN, SPACING, pattern=PATTERN)
    P_c = pc(net.params.dendritic_backbone)
    tr = net.entanglement_tracker
    rows = net.coupling_weights.sum(axis=1)
    trace_path = os.path.join(SWEEP_DIR, 'po7_unit8_trace.jsonl')
    open(trace_path, 'w').close()

    print(f"PO-7 UNIT 8 — L·ETA-2 rig + partition measurement")
    print(f"  N={N_SYN} spacing={SPACING}um pattern={PATTERN}  P_c={P_c*1e15:.2f}fW  "
          f"row-sums min={rows.min():.3f} max={rows.max():.3f}")
    print(f"  drive {VOLT*1e3:.0f}mV sustained + glutamate, {T_SIM}s @ dt={DT}  "
          f"spin={SPIN_RESOLVED} provenance(drops clique+EM)={PROVENANCE}")
    print(f"  GATE: r ~ {ETA2_R} and eta ~ {ETA2_ETA} (+-{TOL:.0%})\n")
    print(f"  {'t(s)':>7} {'ca_mx':>7} {'E_inv':>7} {'r_max':>8} {'eta_mx':>7} "
          f"{'n_cond':>6} {'dimers':>7} {'xbond':>6} {'comps':>6} {'nmulti':>6} {'lgfrac':>6} {'s/step':>7}")

    rel = PresynapticRelease(seed=SEED)
    peak = dict(r=0.0, eta=0.0, e=0.0, ca=0.0, xbond=0, nmulti=0, ncond=0)
    n_steps = int(round(T_SIM / DT))
    t0 = time.time()
    for k in range(n_steps):
        g = rel.step(0.95, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        if (k + 1) % TRACKER_EVERY:
            continue
        e, ca, r, en, es = probe_r(net, P_c)
        n_cond = int((r >= 1.0).sum())
        xb = sum(1 for (a, b), f in tr.cross_synapse_bonds.items()
                 if f > tr.WERNER_ENTANGLEMENT_BOUND and a[0] != b[0])
        comps = tr._find_all_clusters()
        nmulti = sum(1 for c in comps if len({g0[0] for g0 in c}) >= 2)
        nd = len(tr.all_dimers)
        # largest_frac: fraction of ALL dimers in the biggest component. This is the
        # program's standard blob measure (1.0 = blob). Without it, n_components==1 is
        # AMBIGUOUS — _find_all_clusters omits unbonded dimers as singletons, so 1 can mean
        # "everything merged" OR "only one small bonded cluster exists".
        sizes = sorted((len(c) for c in comps), reverse=True)
        largest = sizes[0] if sizes else 0
        largest_frac = (largest / nd) if nd else 0.0
        n_bonded = sum(sizes)
        peak['r'] = max(peak['r'], float(r.max())); peak['eta'] = max(peak['eta'], float(en.max()))
        peak['e'] = max(peak['e'], float(e.max())); peak['ca'] = max(peak['ca'], float(ca.max()))
        peak['xbond'] = max(peak['xbond'], xb); peak['nmulti'] = max(peak['nmulti'], nmulti)
        peak['ncond'] = max(peak['ncond'], n_cond)
        peak['lgfrac'] = max(peak.get('lgfrac', 0.0), largest_frac)
        rec = {'t': round((k + 1) * DT, 4), 'ca_max': float(ca.max()), 'E_inv_max': float(e.max()),
               'r': [float(x) for x in r], 'r_max': float(r.max()), 'eta_max': float(en.max()),
               'n_condensed': n_cond, 'n_dimers': nd, 'n_cross_bonds': xb,
               'n_components': len(comps), 'n_multi': nmulti,
               'largest_comp': largest, 'largest_frac': largest_frac,
               'n_bonded': n_bonded, 'frac_bonded': (n_bonded / nd) if nd else 0.0}
        with open(trace_path, 'a') as f:          # persist AS WE GO (L·PO5-3 scar)
            f.write(json.dumps(rec) + "\n")
        print(f"  {rec['t']:7.3f} {ca.max():7.4f} {e.max():7.4f} {r.max():8.4f} "
              f"{en.max():7.4f} {n_cond:6d} {nd:7d} {xb:6d} {len(comps):6d} {nmulti:6d} "
              f"{(time.time()-t0)/(k+1):7.3f}")
        sys.stdout.flush()

    print(f"\n  PEAK: ca={peak['ca']:.4f} E_inv={peak['e']:.4f} r={peak['r']:.4f} "
          f"eta={peak['eta']:.4f} condensed={peak['ncond']}/{N_SYN} "
          f"cross_bonds={peak['xbond']} n_multi={peak['nmulti']} "
          f"max_largest_frac={peak.get('lgfrac',0.0):.3f}")
    # L·ETA-2's peak r is NOT reproducible: its rig used the RANDOM 'clustered'
    # geometry and the row-sums were never recorded, so r is not comparable across
    # runs. E_invasion IS comparable (deterministic actin integrator) and is the
    # primary gate; r/eta are reported against L·ETA-2 for context, not as pass/fail.
    e_inv_ok = abs(peak['e'] - 0.3518) / 0.3518 <= TOL
    ignited = peak['ncond'] > 0 and peak['r'] >= 1.0
    print(f"  E_invasion GATE: {'PASS' if e_inv_ok else 'FAIL'} "
          f"({peak['e']:.4f} vs L·ETA-2 0.3518)")
    print(f"  IGNITION: {'YES' if ignited else 'NO'} "
          f"(peak r={peak['r']:.4f}, max condensed={peak['ncond']}/{N_SYN})")
    r_ok = abs(peak['r'] - ETA2_R) / ETA2_R <= TOL
    e_ok = abs(peak['eta'] - ETA2_ETA) / ETA2_ETA <= TOL if peak['eta'] > 0 else False
    print(f"\n  REPRODUCTION GATE: r {'PASS' if r_ok else 'FAIL'} "
          f"({peak['r']:.4f} vs {ETA2_R}) | eta {'PASS' if e_ok else 'FAIL'} "
          f"({peak['eta']:.4f} vs {ETA2_ETA})")
    if not (r_ok and e_ok):
        print("  => NOT L·ETA-2's rig at this duration. No partition claim is admissible.")
        print(f"     (If eta never ignited, T_SIM={T_SIM}s is likely too short — L·ETA-2 ran 20 s.")
        print("      Re-run with PO7_U8_SECONDS=20 once the per-step cost above is acceptable.)")
    else:
        print(f"  => rig REPRODUCED. Partition on it: cross_bonds={peak['xbond']}, "
              f"n_multi={peak['nmulti']}")
        print("     " + ("CROSS-SYNAPSE STRUCTURE EXISTS where eta ignites."
                         if peak['nmulti'] > 0 else
                         "eta ignites but NO multi-synapse component forms — a real negative."))
    out = {'config': {'n_syn': N_SYN, 'spacing_um': SPACING, 'dt': DT, 't_sim': T_SIM,
                      'volt_mV': VOLT * 1e3, 'seed': SEED},
           'peak': peak, 'gate': {'r_ok': bool(r_ok), 'eta_ok': bool(e_ok),
                                  'eta2_r': ETA2_R, 'eta2_eta': ETA2_ETA},
           'trace': 'po7_unit8_trace.jsonl'}
    with open(os.path.join(SWEEP_DIR, 'po7_unit8_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote po7_unit8_results.json (+ trace {trace_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
