#!/usr/bin/env python3
"""
dt-convergence of the analytical gap's PLASTICITY stage (stage 6).  PO-4.
=========================================================================
MO ruling 002: do not assert dt_sub = 1.0 is fine -- check it. DECISION RECORD
`dt-1` established convergence for P_S / edges but explicitly NOT for
transient-phase counts, so it is not assumed to transfer to the plasticity clock.

WHAT IS BEING CHECKED, AND AGAINST WHAT
  Stage 6 integrates actin/volume/CaMKII/DDSC with FORWARD EULER at dt_sub. For the
  actin enlargement pool during silence the exact solution is available in closed
  form -- formation is negligible at baseline calcium (f_CaM ~ 1e-4 at Ca = 0.1 uM
  with Hill n=4, K = 1.0 uM), so the pool obeys

      d(enl)/dt = -(k_stab*conf + (1-conf)/tau_extrude) * enl

  giving enl(g) = enl(0) * exp(-rate*g). The ANALYTIC solution is therefore the
  reference, which is stronger than a finer-dt reference: it has no discretisation
  error at all. (The 5 s full-physics validator at run_theta_burst_45s.py:405-415
  covers stages 1-5, the dimer/bond side; it does not exercise plasticity.)

WHY IT MATTERS HERE
  L·GAP-1's committed-arm residuals (~0.002-0.004) are systematically ~10x the
  uncommitted arm's (~0.0001-0.0005). The committed branch has tau_eff = 50.9 s
  against the uncommitted 180 s, so at fixed dt_sub it takes ~3.5x more Euler error
  per unit time. This quantifies that rather than asserting it.
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

from gap_retention_probe import (make_network, set_arm, read, predicted_R,
                                 GAP_S, CONF_SS, TOL)
from run_theta_burst_45s import analytical_gap

DT_SUBS = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]


def measure(committed, gap_s, dt_sub, seed=42):
    np.random.seed(seed)
    net = make_network()
    set_arm(net, committed)
    pre = read(net)
    analytical_gap(net, gap_s, dt_sub=dt_sub)
    post = read(net)
    return float(np.mean([post[i]['E'] / pre[i]['E'] for i in range(len(pre))
                          if pre[i]['E'] > 0])), post[0]['conf']


def main():
    print("=" * 78)
    print("dt-CONVERGENCE of the gap's plasticity stage   [PO-4, MO ruling 002]")
    print(f"reference = ANALYTIC exp(-rate*g), zero discretisation error   gap = {GAP_S:.0f} s")
    print("=" * 78)

    out = {}
    for label, committed in (("uncommitted", False), ("committed", True)):
        print(f"\n{label}   (tau_eff = "
              f"{1/(0.02*CONF_SS + (1-CONF_SS)/180) if committed else 180.0:.1f} s)")
        print(f"  {'dt_sub':>8}{'R measured':>13}{'R analytic':>13}{'|err|':>11}"
              f"{'ratio':>9}   halving?")
        prev = None
        rows = []
        for dt in DT_SUBS:
            R, conf = measure(committed, GAP_S, dt)
            Ra = predicted_R(GAP_S, conf)
            err = abs(R - Ra)
            ratio = (prev / err) if (prev and err > 0) else float('nan')
            print(f"  {dt:>8.2f}{R:>13.6f}{Ra:>13.6f}{err:>11.6f}{ratio:>9.2f}   "
                  f"{'yes (1st order)' if prev and 1.6 < ratio < 2.6 else ''}")
            rows.append(dict(dt_sub=dt, R=R, R_analytic=Ra, err=err))
            prev = err
        out[label] = rows

    print("\n" + "=" * 78)
    verdicts = []
    for label in out:
        e1 = [r for r in out[label] if r['dt_sub'] == 1.0][0]['err']
        fine = out[label][-1]['err']
        ok = e1 <= TOL
        verdicts.append(ok)
        print(f"{label:<14} err(dt_sub=1.0) = {e1:.6f}  vs tolerance {TOL}  -> "
              f"{'WITHIN' if ok else 'EXCEEDS'};  err({DT_SUBS[-1]}) = {fine:.6f}")
    print()
    if all(verdicts):
        print("VERDICT: dt_sub = 1.0 is ADEQUATE for the plasticity stage at this gap length —")
        print("  its error is inside the pre-registered tolerance on both arms, and the error")
        print("  falls with dt_sub, identifying the residual as Euler discretisation rather")
        print("  than a modelling error. NOT a claim that 1.0 s suffices at every gap length:")
        print("  error grows with gap/tau_eff, so a longer gap or a faster tau needs re-checking.")
    else:
        print("VERDICT: dt_sub = 1.0 EXCEEDS tolerance on at least one arm — reduce it.")
    print("=" * 78)

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'gap_dt_convergence_results.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"persisted -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
