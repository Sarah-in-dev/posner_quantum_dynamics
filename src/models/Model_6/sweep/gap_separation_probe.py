#!/usr/bin/env python3
"""
L·GAP-2 — committed vs uncommitted spine volume across an HONEST gap.  PO-4.
============================================================================
The board's acceptance (MO_MODEL6.md Sec.3 PO-4). Pre-registered in
docs/PREREG_PO4_GAP.md AMENDMENT D (commit d5b0c00) BEFORE this ran: the sign, a
4-sigma magnitude floor, two nulls, two positive controls, and the saturation limit.

Scored on the EXISTENCE and SIGN of a separation, with a floor that keeps it
distinguishable from thermal noise -- NOT against the 1.291/2.389 pair in
MO_MODEL6.md:140, which grep shows to be coordination prose with no artifact behind
it (MO ruling 003 confirmed this as the MO's own defect and instructed: do not bend
the measurement to reach the quoted pair).

Same controlled initial condition and the same stated limits as L·GAP-1 -- see
gap_retention_probe.py's docstring. The live drive path is not exercised.

AMENDMENT D NULL 2 -- THE FROZEN-CLOCK CONTROL, and how to reproduce it.
  Registered: on PRE-FIX code the same measurement must yield dV ~ 0, because
  neither arm's clock runs. MEASURED, not asserted:

      committed    V @+300 s = 1.0006 +/- 0.0005   (spine clock advanced 0.0010 s)
      uncommitted  V @+300 s = 1.0003 +/- 0.0006   (spine clock advanced 0.0010 s)
      dV = +0.000299   against the post-fix +0.7764   -- a factor of 2595

  That is what "the full model has never been allowed to show it"
  (MO_MODEL6.md:140) means, demonstrated rather than claimed.

  To reproduce: extract the pre-fix gap and point this probe's import at it --
      git show 806adc7:src/models/Model_6/sweep/run_theta_burst_45s.py > /tmp/prefix_gap.py
  then import analytical_gap from that module instead of from run_theta_burst_45s.
  Commit 806adc7 is the last commit before stage 6; `grep -c 'PLASTICITY CLOCK'`
  on the extracted file returns 0, which is the check that it is genuinely pre-fix.
"""
import sys, os, json, logging
import numpy as np

logging.disable(logging.INFO)
M6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, M6)
sys.path.insert(0, os.path.join(M6, 'sweep'))
for _n in ['model6_core', 'multi_synapse_network', 'dimer_particles',
           'analytical_calcium_system', 'atp_system', 'ca_triphosphate_complex',
           'quantum_coherence', 'pH_dynamics', 'dopamine_system', 'em_tryptophan_module',
           'em_coupling_module', 'local_dimer_tubulin_coupling', 'camkii_module',
           'spine_plasticity_module', 'photon_emission_module', 'photon_receiver_module',
           'ddsc_module', 'vibrational_cascade_module']:
    logging.getLogger(_n).setLevel(logging.ERROR)

from gap_retention_probe import make_network, set_arm, read
from run_theta_burst_45s import analytical_gap

# ---- PRE-REGISTERED (AMENDMENT D, d5b0c00) ---------------------------------
GAP_S = 300.0
N_REPS = 5
SIGMA = 0.065          # measured single-run thermal spread
FLOOR = 4 * SIGMA      # 0.26 — separation must clear 4 sigma
CEILING = 3.80         # actin-limited, spine_plasticity_module.py:332-333 (D20)


def run(committed, seed):
    np.random.seed(seed)
    net = make_network()
    set_arm(net, committed)
    v0 = read(net)[0]['vol']
    analytical_gap(net, GAP_S, dt_sub=1.0)
    st = read(net)[0]
    return v0, st['vol'], st['conf'], st['committed']


def arm(committed, seeds):
    rows = [run(committed, s) for s in seeds]
    return (np.array([r[1] for r in rows]), rows[0][0],
            np.mean([r[2] for r in rows]), all(r[3] for r in rows))


def main():
    print("=" * 78)
    print("L·GAP-2 — committed vs uncommitted spine volume across an HONEST gap")
    print(f"pre-registered: PREREG_PO4_GAP.md AMENDMENT D (d5b0c00)   gap = {GAP_S:.0f} s, "
          f"{N_REPS} reps/arm")
    print("=" * 78)

    seeds_a = list(range(N_REPS))
    seeds_b = list(range(100, 100 + N_REPS))

    Vc, v0, conf_c, cm_c = arm(True, seeds_a)
    Vu, _, conf_u, _ = arm(False, seeds_a)

    print(f"\n  start volume (both arms, identical state): {v0:.4f}")
    print(f"\n  {'arm':<14}{'V @+300s':>22}{'conf':>9}{'committed':>11}")
    print("  " + "-" * 56)
    print(f"  {'committed':<14}{np.mean(Vc):>12.4f} +/- {np.std(Vc):.4f}{conf_c:>9.3f}"
          f"{str(cm_c):>11}")
    print(f"  {'uncommitted':<14}{np.mean(Vu):>12.4f} +/- {np.std(Vu):.4f}{conf_u:>9.3f}"
          f"{'False':>11}")

    dV = float(np.mean(Vc) - np.mean(Vu))
    print(f"\n  SEPARATION  dV = {dV:+.4f}   (registered: > 0, and > {FLOOR:.2f} = 4 sigma)")

    # ---- NULL 1: two uncommitted arms, differing only by seed
    Vu2, _, _, _ = arm(False, seeds_b)
    dV_null = float(np.mean(Vu) - np.mean(Vu2))
    print(f"  NULL (uncommitted vs uncommitted, seeds only)  dV = {dV_null:+.4f}   "
          f"(registered: |dV| <= {FLOOR:.2f})")

    # ---- Registered checks
    problems, saturated = [], (np.mean(Vc) >= CEILING - 1e-3 and np.mean(Vu) >= CEILING - 1e-3)
    if not cm_c:
        problems.append("positive control dead: _camkii_committed not True in committed arm")
    if conf_c <= 0.5:
        problems.append(f"positive control dead: confinement {conf_c:.3f} <= 0.5")
    if saturated:
        problems.append(f"both arms at the {CEILING} ceiling — separation compressed")
    if abs(dV_null) > FLOOR:
        problems.append(f"uncommitted-pair null separated by {dV_null:+.4f} > {FLOOR:.2f}")

    print("\n" + "=" * 78)
    if problems:
        v = "INCONCLUSIVE" if (saturated or not cm_c or conf_c <= 0.5) else "FALSIFIED"
        print(f"VERDICT: {v}")
        for p in problems:
            print(f"  - {p}")
    elif dV > FLOOR:
        v = "SEPARATION CONFIRMED"
        print(f"VERDICT: {v}")
        print(f"  - committed {np.mean(Vc):.4f} vs uncommitted {np.mean(Vu):.4f}, "
              f"dV = {dV:+.4f} > {FLOOR:.2f}")
        print(f"  - seed-only null separates by {dV_null:+.4f}, "
              f"{abs(dV/dV_null) if dV_null else float('inf'):.0f}x smaller than the effect")
        print(f"  - both positive controls fired (committed={cm_c}, conf={conf_c:.3f})")
        print(f"  - neither arm at the {CEILING} ceiling")
    else:
        v = "FALSIFIED"
        print(f"VERDICT: {v}")
        print(f"  - dV = {dV:+.4f} does not clear the registered {FLOOR:.2f} floor")
    print("=" * 78)
    print("LIMITS: controlled initial condition; the live drive path is NOT exercised and")
    print("        does not reach this regime (measured: a traversal leaves E_invasion at")
    print("        0.0000, 10x below invasion_threshold). Two synapses, one network.")
    print("        NOTE: this measurement was taken with K_CLASSICAL = 0.05, the retired")
    print("        rate; it was corrected to the grounded 0.005 later the same day. The")
    print("        separation is an ACTIN/volume result and does not read the dissolution")
    print("        rate, so the correction does not move it -- but the run predates it.")

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'gap_separation_probe_results.json')
    with open(p, 'w') as f:
        json.dump(dict(verdict=v, gap_s=GAP_S, v_start=v0,
                       committed=dict(mean=float(np.mean(Vc)), sd=float(np.std(Vc)),
                                      conf=float(conf_c)),
                       uncommitted=dict(mean=float(np.mean(Vu)), sd=float(np.std(Vu))),
                       dV=dV, dV_null=dV_null, floor=FLOOR), f, indent=2)
    print(f"\npersisted -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
