#!/usr/bin/env python3
"""L·ETA-6 — how much NMDAR calcium did L·ETA-4's "silent" synapses actually get?

PRE-REGISTERED: docs/PREREG_L_ETA_6_NMDAR_MAGNITUDE.md (commit ca694df, BEFORE this ran).
Thresholds are fixed there and are not renegotiable here.

AUDIT, NOT A RE-RUN. This reconstructs L·ETA-4's CONDITIONS read-only to measure an INPUT
magnitude. It does not re-derive L·ETA-4's verdict and does not modify
plateau_vgcc_leak_probe.py.

Rotation 001 showed L·ETA-4's NMDAR metric (open FRACTION) is voltage-independent by
construction, so it cannot see NMDAR current. This measures the current directly:
  I_NMDA = sum(current[is_nmda])   -- analytical_calcium_system.py:136-138, base*B(V) when open
and the NMDAR-attributable calcium via the model's own APV control (nmda_blocked, :109/:124).
"""
import sys, os, logging
import numpy as np
logging.disable(logging.INFO)
HERE = os.path.dirname(os.path.abspath(__file__)); M6 = os.path.dirname(HERE)
REPO = os.path.normpath(os.path.join(M6, '..', '..', '..'))
sys.path.insert(0, M6); sys.path.insert(0, os.path.join(REPO, 'sweep')); sys.path.insert(0, HERE)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse, PLATEAU_VOLTAGE_V
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease

# L·ETA-4's OWN constants (plateau_vgcc_leak_probe.py:76-80)
N_SYN, DRIVEN, T_S, DT, SEED = 7, 3, 12.0, 0.005, 11


def build():
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net, p


def run(plateau, nmda_block):
    np.random.seed(SEED)
    net, p = build()
    if nmda_block:
        for s in net.synapses:
            s.calcium.channels.params.nmda_blocked = True
    rel = [PresynapticRelease(seed=3000+i) for i in range(N_SYN)]
    acts = np.zeros(N_SYN); acts[DRIVEN] = 1.0

    # PREREG L·ETA-6 AMENDMENT 1 (MO ruling 008): §4 gates on a PEAK, and a peak is NOT
    # sampling-invariant — 20-step sampling can only UNDER-report it, biasing toward
    # dCa < 0.05 uM i.e. toward NEGLIGIBLE, the branch under which L·ETA-4 survives.
    # Fixed at zero cost: AnalyticalCalciumSystem already maintains a TRUE per-step running
    # max internally (analytical_calcium_system.py:419,
    # peak_concentration = max(peak_concentration, np.max(_local_ca))), updated every step
    # regardless of how often this probe samples. Read it at the end of the arm.
    q_nmda = np.zeros(N_SYN)   # integrated NMDAR charge
    q_vgcc = np.zeros(N_SYN)
    ca_sum = np.zeros(N_SYN); ca_peak = np.zeros(N_SYN); n = 0
    glu_events = np.zeros(N_SYN, dtype=int)

    for k in range(int(T_S/DT)):
        for i, syn in enumerate(net.synapses):
            v = -70e-3 + acts[i]*30e-3
            stim = {'voltage': v, 'reward': False}
            g = rel[i].step(acts[i], DT)
            if g:
                stim['glutamate'] = g
                glu_events[i] += 1
            if plateau:
                stim['plateau_potential'] = True
            syn.step(DT, stim)
            ch = syn.calcium.channels
            m = ch.is_nmda
            q_nmda[i] += float(np.sum(ch.current[m])) * DT
            q_vgcc[i] += float(np.sum(ch.current[~m])) * DT
            # Calcium sampled every 20 steps — L·ETA-4's own logging cadence
            # (plateau_vgcc_leak_probe.py:137 `if k % 20 == 0`). Physics unchanged;
            # the field max is the expensive call. Charge integrals stay PER STEP.
            if k % 20 == 0:
                ca = float(np.max(syn.calcium.get_concentration())) * 1e6
                ca_sum[i] += ca; ca_peak[i] = max(ca_peak[i], ca)
        if k % 20 == 0:
            n += 1
        if k % 400 == 0:
            print(f"      ... step {k}/{int(T_S/DT)}", flush=True)
    # EXACT per-step peak (the scored quantity), not the sampled one.
    ca_peak_true = np.array([float(getattr(s.calcium, 'peak_concentration', np.nan))*1e6
                             for s in net.synapses])
    return dict(q_nmda=q_nmda, q_vgcc=q_vgcc, ca_mean=ca_sum/n,
                ca_peak=ca_peak_true, ca_peak_sampled=ca_peak, glu=glu_events)


def main():
    print("="*100)
    print("L·ETA-6 — NMDAR-attributable calcium at L·ETA-4's 'silent' synapses")
    print("pre-registered: docs/PREREG_L_ETA_6_NMDAR_MAGNITUDE.md   |   AUDIT, not a re-run")
    print("="*100)
    print(f"  L·ETA-4 conditions: {N_SYN} synapses @1um, ONLY {DRIVEN} driven (act=1.0), "
          f"T={T_S}s, dt={DT}, seed={SEED}, plateau={PLATEAU_VOLTAGE_V*1e3:.0f}mV")
    print()
    import json
    OUT = os.path.join(M6, 'results', 'nmdar_magnitude'); os.makedirs(OUT, exist_ok=True)
    arms = {}
    # ARM_SET is overridable so the cheap (plateau OFF) pair can be run alone. The
    # plateau ON arms cost >10x and must be MO-sequenced (see queue Q4).
    ARM_SET = os.environ.get('ARMS', 'off').lower()
    pls = (False,) if ARM_SET == 'off' else ((True,) if ARM_SET == 'on' else (False, True))
    for pl in pls:
        for nb in (False, True):
            print(f"  RUNNING plateau={int(pl)} nmda_blocked={int(nb)} ...", flush=True)
            import time as _t; _t0=_t.time()
            arms[(pl, nb)] = run(pl, nb)
            print(f"  done plateau={int(pl)} nmda_blocked={int(nb)} in {_t.time()-_t0:.1f}s", flush=True)
            # INCREMENTAL PERSIST — a kill must not cost completed arms (L·ETA-5 lesson).
            with open(os.path.join(OUT, f'arm_pl{int(pl)}_nb{int(nb)}.json'), 'w') as fh:
                json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v)
                           for k, v in arms[(pl, nb)].items()}, fh, indent=1)
    print()
    silent = [i for i in range(N_SYN) if i != DRIVEN]

    R_on = dca_on = None
    for pl in pls:
        a = arms[(pl, False)]; b = arms[(pl, True)]
        tag = "PLATEAU ON " if pl else "PLATEAU OFF"
        print("-"*100)
        print(f"{tag}")
        print(f"  glutamate events: driven {a['glu'][DRIVEN]}, silent {[int(a['glu'][i]) for i in silent]}")
        qn_s = float(np.mean([a['q_nmda'][i] for i in silent]))
        qn_d = float(a['q_nmda'][DRIVEN])
        R = qn_s/qn_d if qn_d > 0 else float('nan')
        dca_pk = float(np.mean([a['ca_peak'][i]-b['ca_peak'][i] for i in silent]))
        dca_mn = float(np.mean([a['ca_mean'][i]-b['ca_mean'][i] for i in silent]))
        print(f"  NMDAR charge   silent(mean) {qn_s:.4e}   driven {qn_d:.4e}   R = {R:.4f}")
        print(f"  dCa_NMDA silent: peak {dca_pk:+.5f} uM   mean {dca_mn:+.5f} uM")
        print(f"  silent Ca peak  intact {np.mean([a['ca_peak'][i] for i in silent]):.4f} uM"
              f"   blocked {np.mean([b['ca_peak'][i] for i in silent]):.4f} uM  (EXACT per-step max)")
        print(f"    [20-step-sampled peak would have been "
              f"{np.mean([a['ca_peak_sampled'][i]-b['ca_peak_sampled'][i] for i in silent]):+.5f} uM "
              f"vs exact {dca_pk:+.5f} uM — sampling bias check]")
        if pl:
            R_on, dca_on = R, dca_pk
        else:
            R_off, dca_off = R, dca_pk
    print("-"*100)

    # ---- PRE-REGISTERED VERDICT (thresholds from the prereg, §4) ----
    NEG_R, NEG_CA = 0.05, 0.05
    MAT_R, MAT_CA = 0.124, 0.10
    print()
    print("="*100)
    print("VERDICT — thresholds fixed in the pre-registration, before measuring")
    print("="*100)
    if R_on is not None:
        print(f"  R (silent/driven NMDAR charge, plateau ON) : {R_on:.4f}")
        print(f"  dCa_NMDA(silent) peak, plateau ON          : {dca_on:+.5f} uM")
    else:
        print("  plateau-ON arms NOT RUN (scored condition unmeasured)")
    print(f"  NEGLIGIBLE iff R <= {NEG_R} AND dCa < {NEG_CA} uM")
    print(f"  MATERIAL   iff R >= {MAT_R} OR  dCa >= {MAT_CA} uM   (Jain 2024: 7/56.3 = 0.124)")
    print()
    if R_on is None:
        print("  => EQUIVOCAL — SCORED CONDITION NOT MEASURED. The plateau-ON arms, which are")
        print("     the pre-registered scored condition, were not completed (>10x the cost of")
        print("     plateau-OFF; killed per the compute cap). Plateau-OFF numbers above are")
        print("     REAL but are NOT the scored quantity. Cannot determine whether L·ETA-4's")
        print("     -0.0019 survives. Escalated with cost; not guessed.")
        print()
        print(f"  plateau-OFF (measured, NOT scored): R = {R_off:.4f}, dCa peak = {dca_off:+.5f} uM")
        print()
        print("SCORED VERDICT: EQUIVOCAL_SCORED_CONDITION_NOT_MEASURED")
        return
    if R_on <= NEG_R and dca_on < NEG_CA:
        v = "NEGLIGIBLE"
        print("  => NEGLIGIBLE. L·ETA-4's -0.0019 SURVIVES as approximately correct: the")
        print("     spontaneous floor delivered too little NMDAR calcium at the silent")
        print("     synapses to matter. P_product is WEAKLY SUPPORTED rather than unevidenced;")
        print("     PO-5 survives roughly as scoped.")
    elif R_on >= MAT_R or dca_on >= MAT_CA:
        v = "MATERIAL"
        print("  => MATERIAL. The silent synapses had a LIVE NMDAR AND-gate. L·ETA-4's")
        print("     -0.0019 is CONTRADICTED, not merely unevidenced — its metric could not")
        print("     see a real NMDAR contribution. PO-5's P_product premise needs re-scoping")
        print("     (Sarah's call, not this PO's).")
    else:
        v = "EQUIVOCAL"
        print("  => EQUIVOCAL. Between the pre-registered bands. CANNOT DETERMINE whether")
        print("     L·ETA-4's finding survives without re-running it with a current-based")
        print("     metric. Escalated with cost, not guessed.")
    print()
    print(f"SCORED VERDICT: {v}")


if __name__ == "__main__":
    main()
