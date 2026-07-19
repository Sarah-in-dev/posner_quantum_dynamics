#!/usr/bin/env python3
"""
PO-7 UNIT 7 — DERIVE the provenance claim radius as a reaction-diffusion length.
Advisor R4 step 3. Derivation + measurement; no verdict function, nothing tuned.

THE ARGUMENT (advisor R4 §4.2)
------------------------------
The claim radius is NOT a diffusion length. Phosphate diffuses far in a second
(sqrt(6Dt) ~ tens of um), so diffusion alone would give a radius far larger than the 500 nm
currently coded. The physical scale is set by COMPETITION between diffusing away and being
incorporated into a growing cluster:

    r_claim ~= sqrt(6 D / k_incorp)

Both quantities already exist in the model, so this is a derivation from committed constants
rather than a new parameter:

  D        = params.phosphate.D_phosphate = 890e-12 m^2/s  (H2PO4^-)
             NOTE: this is a FREE-SOLUTION value. Cytoplasmic crowding reduces effective D by
             roughly 3-4x; that correction is applied as a reported sensitivity arm, not
             silently folded in.
  k_incorp = pseudo-first-order consumption rate of a PNC unit
           = k_eff * [PNC],  with k_eff = k_base * template_enhancement
             k_base = productive_fraction * 4*pi*D_ion_pair*R_ion_pair * N_A ~= 1.9e4 M^-1 s^-1
             (ca_triphosphate_complex.py:140-153)

[PNC] and template_enhancement are FIELDS, so they are measured from a short run rather than
assumed, and the radius is reported across their observed range.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))

T_SIM, DT, SEED = 0.2, 0.005, 31337
CODED_REACH_NM = 500.0
CROWDING_FACTORS = {'free solution (as coded)': 1.0,
                    'cytoplasmic crowding /3': 3.0,
                    'cytoplasmic crowding /4': 4.0}


def r_claim_m(D, k_incorp):
    return float(np.sqrt(6.0 * D / k_incorp)) if k_incorp > 0 else float('inf')


def main():
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    np.random.seed(SEED)
    p = Model6Parameters(); p.em_coupling_enabled = True
    D_free = float(p.phosphate.D_phosphate)
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    rel = PresynapticRelease(seed=SEED)
    for _ in range(int(round(T_SIM / DT))):
        g = rel.step(0.95, DT)
        net.synapses[0].step(DT, {"voltage": -10e-3, "reward": False, "glutamate": g})

    syn = net.synapses[0]
    ctc = getattr(syn, 'ca_triphosphate', None) or getattr(syn, 'triphosphate', None)
    if ctc is None:
        for attr in dir(syn):
            o = getattr(syn, attr, None)
            if hasattr(o, 'dimerization') and hasattr(o, 'template_enhancement'):
                ctc = o; break
    if ctc is None:
        print("FAIL: could not locate the Ca-triphosphate subsystem; derivation not attempted.")
        return 1

    pnc = np.asarray(ctc.dimerization.pnc_concentration, float)
    k_base = float(ctc.dimerization.k_base)
    te = np.asarray(ctc.template_enhancement, float)
    te_mean = float(np.mean(te)) if te.size > 1 else float(te)

    active = pnc[pnc > 0]
    if active.size == 0:
        print("FAIL: [PNC] is identically zero in this run; cannot derive k_incorp.")
        return 1

    print("=" * 92)
    print("MEASURED INPUTS (from the run, not assumed)")
    print("=" * 92)
    print(f"  D_phosphate (coded, free solution) = {D_free:.3e} m^2/s  = {D_free*1e12:.0f} um^2/s")
    print(f"  k_base                              = {k_base:.3e} M^-1 s^-1")
    print(f"  template_enhancement (mean)         = {te_mean:.3f}")
    print(f"  [PNC] over active grid cells: median={np.median(active):.3e} M  "
          f"p90={np.percentile(active,90):.3e} M  max={active.max():.3e} M")
    print(f"  active cells: {active.size} / {pnc.size}")

    k_eff = k_base * te_mean
    rows = []
    print("\n" + "=" * 92)
    print("DERIVED CLAIM RADIUS   r = sqrt(6D / k_incorp),  k_incorp = k_eff * [PNC]")
    print("=" * 92)
    for label, div in CROWDING_FACTORS.items():
        D = D_free / div
        print(f"\n  D = {D*1e12:>6.1f} um^2/s   [{label}]")
        for pl, pv in (('median', float(np.median(active))),
                       ('p90', float(np.percentile(active, 90))),
                       ('max', float(active.max()))):
            k_inc = k_eff * pv
            r = r_claim_m(D, k_inc)
            rows.append({'D_m2_s': D, 'crowding': label, 'pnc_level': pl, 'pnc_M': pv,
                         'k_incorp_s': k_inc, 'r_claim_m': r})
            print(f"    [PNC]={pv:.3e} M ({pl:>6})  k_incorp={k_inc:.3e} s^-1  "
                  f"r_claim={r*1e9:>12,.0f} nm  ({r*1e6:>10,.2f} um)")

    rmin = min(r['r_claim_m'] for r in rows) * 1e9
    rmax = max(r['r_claim_m'] for r in rows) * 1e9
    print("\n" + "=" * 92)
    print("VERDICT ON THE CODED CONSTANT")
    print("=" * 92)
    print(f"  derived r_claim spans {rmin:,.0f} nm ... {rmax:,.0f} nm")
    print(f"  coded provenance_net_reach_nm = {CODED_REACH_NM:.0f} nm")
    if rmin > CODED_REACH_NM:
        print(f"  => The coded reach is SMALLER than every derived value "
              f"(by {rmin/CODED_REACH_NM:,.0f}x at the most conservative end).")
        print("     Per advisor R4 §4.2: 'If it comes out much larger, the yield ceiling is a")
        print("     settings artifact.' The cross-edge ceiling is therefore NOT a physical")
        print("     property of the mechanism — it is the reach constant being far too small.")
    elif rmax < CODED_REACH_NM:
        print("  => The coded reach is LARGER than every derived value: cross-synapse edges")
        print("     are genuinely rare and the ceiling is physical. That is a finding.")
    else:
        print("  => The coded 500 nm falls inside the derived range; it is defensible as-is.")

    out = {'D_phosphate_free_m2_s': D_free, 'k_base_M_s': k_base,
           'template_enhancement_mean': te_mean,
           'pnc_M': {'median': float(np.median(active)), 'p90': float(np.percentile(active, 90)),
                     'max': float(active.max()), 'n_active_cells': int(active.size)},
           'rows': rows, 'coded_reach_nm': CODED_REACH_NM,
           'derived_r_min_nm': rmin, 'derived_r_max_nm': rmax}
    path = os.path.join(SWEEP_DIR, 'po7_unit7_claim_radius_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print("\nSTATED ASSUMPTIONS: (1) k_incorp is the PSEUDO-FIRST-ORDER PNC consumption rate")
    print("k_eff*[PNC]; (2) the competing sink is dimer formation only — no phosphate buffering,")
    print("efflux, or metabolic re-uptake is modelled (params note phosphate buffering is absent);")
    print("(3) sqrt(6D/k) is the 3-D reaction-diffusion length. Each would move the number, and")
    print("all three push the SAME way: adding sinks raises k_incorp and SHRINKS r.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
