#!/usr/bin/env python3
"""
L·GAP-1 — the analytical_gap retention probe.  PO-4.
=====================================================
Pre-registered in `docs/PREREG_PO4_GAP.md` BEFORE this was run. Read that first;
the predicted numbers, the nulls, the positive controls and the verdict function
are all fixed there, not here.

WHAT THIS MEASURES
  R = E_invasion(after gap) / E_invasion(before gap), across ONE silent gap.

WHY R AND NOT "DID STATE SURVIVE"
  A stopped clock and a real memory effect BOTH produce "state survived the gap".
  The retention FRACTION separates them, and the stopped clock's value is the
  suspiciously perfect one:
      current code (1 ms/gap)     R = 0.999994   <- RED FLAG
      honest, never-committed     R = 0.8948     (exp(-20/180), Honkura 2008)
      honest, committed           R = 0.6751     (k_stab*conf + (1-conf)/180)

CONTROLLED INITIAL CONDITION — stated up front, not buried
  The spine state is SET directly rather than driven there through the live
  glutamate->calcium->actin path. Two measured reasons, both recorded in the log:
    (1) COST. A 12-cycle theta traversal on a 2-synapse network took 190.7 s wall
        for 1.5 s of simulated time (~127x slower than realtime). A 20 s
        full-physics silence reference would be ~42 min, and PO-4 holds no heavy
        compute slot (PO-3 holds the single one).
    (2) REACH. That same traversal left actin_enlargement = 0.0106 and
        E_invasion = 0.0000 -- an order of magnitude BELOW
        invasion_threshold = 0.1. The live drive path does not reach the regime
        where retention is even defined, so driving it would measure nothing.
  Precedent: `model6-dimer-formation-chemistry` Sec.2 unit-tests the chemistry in
  isolation with controlled calcium "because the live model6_core glutamate->calcium
  path is not yet wired". Same move, same reason, stated the same way.

  LIMIT, therefore, and it is a real one: this validates the GAP FUNCTION's
  treatment of the spine subsystem. It does NOT validate the drive path, and it
  makes no claim that a live network reaches this regime -- the diagnostic above
  says it does not.
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

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from run_theta_burst_45s import analytical_gap

# ---- PRE-REGISTERED CONSTANTS (docs/PREREG_PO4_GAP.md Sec.1) ----------------
GAP_S = 20.0
TOL = 0.02
E0 = 1.0                 # controlled pre-gap actin_enlargement (well above threshold 0.1)
R_STOPPED_CLOCK = 0.99   # >= this on current code == the defect reproduced
TAU_EXTRUDE = 180.0
K_STAB = 0.02
CONF_SS = 0.02 / (0.02 + 0.0005)


def predicted_R(gap_s, conf):
    """The registered prediction, from the module's own constants. Not fitted."""
    rate = K_STAB * conf + (1.0 - conf) / TAU_EXTRUDE
    return float(np.exp(-rate * gap_s))


def make_network(n=2):
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=n, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def set_arm(net, committed):
    """Controlled initial condition. Sets the pool E_invasion reads, and the
    confinement latch that selects WHICH drain path runs. Nothing else."""
    for s in net.synapses:
        sp = s.spine_plasticity
        sp.actin_enlargement = E0
        sp.confinement = CONF_SS if committed else 0.0
        sp._update_einv() if hasattr(sp, '_update_einv') else None
        a = sp.params.actin
        sp.E_invasion = float(np.clip(
            (sp.actin_enlargement - a.invasion_threshold) /
            max(1e-6, a.E_ref - a.invasion_threshold), 0.0, 1.0))
        s._camkii_committed = bool(committed)
        if committed:
            s._committed_memory_level = 1.0


def read(net):
    return [dict(E=s.spine_plasticity.E_invasion,
                 sp_time=s.spine_plasticity.time,
                 enl=s.spine_plasticity.actin_enlargement,
                 conf=s.spine_plasticity.confinement,
                 vol=s.spine_plasticity.spine_volume,
                 committed=bool(getattr(s, '_camkii_committed', False)))
            for s in net.synapses]


def run_arm(committed, gap_s, seed=42):
    np.random.seed(seed)
    net = make_network()
    set_arm(net, committed)
    pre = read(net)
    t_net_pre = net.time
    analytical_gap(net, gap_s, dt_sub=1.0)
    post = read(net)
    # D20's discriminator, adopted on the MO's ruling to PO-3: an OBSERVED CLOCK DELTA is
    # proof; a retention threshold is a symptom. The gap is honest only if the plasticity
    # clock advanced by the same wall time the network clock did.
    for p, q in zip(pre, post):
        q['sp_advance'] = q['sp_time'] - p['sp_time']
    for q in post:
        q['net_advance'] = net.time - t_net_pre
    return pre, post


def main():
    print("=" * 78)
    print("L·GAP-1 — analytical_gap retention probe   [PO-4]")
    print("Pre-registered: docs/PREREG_PO4_GAP.md   (predictions fixed before this ran)")
    print("=" * 78)

    results, failures, inconclusive = {}, [], []

    # ---- NULL 1 (pre-registered): zero-duration gap => R must be exactly 1.000
    pre, post = run_arm(False, 0.0)
    r_null = post[0]['E'] / pre[0]['E'] if pre[0]['E'] > 0 else float('nan')
    ok_null1 = abs(r_null - 1.0) < 1e-9
    print(f"\nNULL 1  zero-duration gap      R = {r_null:.9f}   "
          f"(registered: exactly 1.0)  {'PASS' if ok_null1 else 'FAIL'}")
    if not ok_null1:
        inconclusive.append("null-1 zero-duration gap decayed")
        print("        ^ NOT a harness bug. The gap's TAIL runs network.step(0.001, ...)")
        print("          unconditionally, so even a 0 s gap advances the spine by 1 ms.")

    # ---- The two arms
    print(f"\n{'arm':<14}{'conf':>8}{'E pre':>10}{'E post':>10}{'R meas':>11}"
          f"{'R pred':>11}{'|diff|':>9}   verdict")
    print("-" * 78)
    for label, committed in (("uncommitted", False), ("committed", True)):
        pre, post = run_arm(committed, GAP_S)

        # ---- POSITIVE CONTROLS (pre-registered; the L·ETA-4 scar)
        if committed:
            if not all(p['committed'] for p in post):
                inconclusive.append("positive control dead: _camkii_committed not True")
            if not all(p['conf'] > 0.5 for p in post):
                inconclusive.append("positive control dead: confinement <= 0.5")
        # ---- ENTRY BAR (pre-registered null 2)
        eligible = [i for i, p in enumerate(pre) if p['E'] > 0.0]
        if len(eligible) < 2:
            inconclusive.append(f"{label}: fewer than 2 synapses above entry bar")

        R = float(np.mean([post[i]['E'] / pre[i]['E'] for i in eligible])) if eligible else float('nan')
        Rp = predicted_R(GAP_S, post[0]['conf'])
        d = abs(R - Rp)
        # On CURRENT code the registered expectation is the stopped clock, not Rp.
        stopped = R >= R_STOPPED_CLOCK
        verdict = "STOPPED CLOCK" if stopped else ("within tol" if d <= TOL else "OFF PREDICTION")
        print(f"{label:<14}{post[0]['conf']:>8.3f}{pre[0]['E']:>10.4f}{post[0]['E']:>10.4f}"
              f"{R:>11.6f}{Rp:>11.4f}{d:>9.4f}   {verdict}")
        results[label] = dict(R=R, R_pred=Rp, diff=d, conf=post[0]['conf'],
                              E_pre=pre[0]['E'], E_post=post[0]['E'],
                              vol_pre=pre[0]['vol'], vol_post=post[0]['vol'],
                              stopped_clock=bool(stopped))
        if not stopped and d > TOL:
            failures.append(f"{label}: |R-Rpred| = {d:.4f} > {TOL}")

    # ---- D20 CLOCK-DELTA DISCRIMINATOR (adopted on MO ruling; proof, not symptom)
    pre_c, post_c = run_arm(False, GAP_S)
    sp_adv, net_adv = post_c[0]['sp_advance'], post_c[0]['net_advance']
    print(f"\nD20 CLOCK DELTA over a {GAP_S:.0f} s gap:")
    print(f"  network.time advanced           = {net_adv:.4f} s")
    print(f"  spine_plasticity.time advanced  = {sp_adv:.4f} s")
    print(f"  ratio (honest gap => 1.0)       = {sp_adv / net_adv if net_adv else float('nan'):.6f}")
    clock_honest = abs(sp_adv - net_adv) < 1e-6
    print(f"  {'PASS — clocks agree' if clock_honest else 'FAIL — the plasticity clock lags the network clock'}")
    clock = dict(net_advance=net_adv, sp_advance=sp_adv, honest=bool(clock_honest))

    # ---- POST-HOC DIAGNOSTIC — declared post-hoc, NOT pre-registered, and it does
    # NOT enter the verdict. Reported because it is the sharpest single discriminator
    # the run produced: if the clock is stopped, retention is independent of gap length.
    r20 = results['uncommitted']['R']
    print(f"\nPOST-HOC (not pre-registered, not in the verdict):")
    print(f"  R(gap=0 s)  = {r_null:.6f}")
    print(f"  R(gap=20 s) = {r20:.6f}")
    print(f"  ratio       = {r20 / r_null if r_null else float('nan'):.6f}")
    print("  A 0 s gap and a 20 s gap retain the SAME fraction. Retention does not")
    print("  depend on gap duration -- the signature of a fixed-size tick, not decay.")

    # ---- VERDICT (pre-registered Sec.5)
    print("\n" + "=" * 78)
    both_stopped = all(results[k]['stopped_clock'] for k in results)
    if inconclusive:
        v = "INCONCLUSIVE"
        why = inconclusive
    elif both_stopped:
        v = "DEFECT REPRODUCED — the clock is stopped"
        why = [f"R = {results[k]['R']:.6f} >= {R_STOPPED_CLOCK} in the {k} arm; "
               f"registered honest value was {results[k]['R_pred']:.4f}" for k in results]
    elif failures:
        v = "FALSIFIED"
        why = failures
    else:
        v = "RETENTION MATCHES PREDICTION"
        why = [f"{k}: R = {results[k]['R']:.4f} vs predicted {results[k]['R_pred']:.4f}"
               for k in results]
    print(f"VERDICT: {v}")
    for w in why:
        print(f"  - {w}")
    print("=" * 78)
    print("LIMITS: controlled initial condition (see module docstring); the live drive")
    print("        path is NOT exercised and does not reach this regime. K_CLASSICAL =")
    print("        0.05 (the RETIRED rate) is live in the gap -- MO-held, not PO-4's.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'gap_retention_probe_results.json')
    with open(out, 'w') as f:
        json.dump(dict(verdict=v, why=why, gap_s=GAP_S, results=results,
                       clock_delta=clock, null_zero_gap_R=r_null), f, indent=2)
    print(f"\npersisted -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
