#!/usr/bin/env python3
"""
F3 COHERENCE CONTROLS — is the producing readout genuinely COHERENCE-GATED, or would it commit anyway?
All arms get an identical DELAYED DOPAMINE BURST at 10 s. The only differences are the tag's coherence
and the readout mode:

  A. undoped, quantum   -> tag still coherent at 10 s (P_S > Werner 0.707)  => EXPECT COMMIT
  B. Li6,     quantum   -> negligible dopant effect (F1)                     => EXPECT COMMIT (~A)
  C. Li7,     quantum   -> T2 216->~14 s, tag DECOHERED by 10 s              => EXPECT NO COMMIT
  D. undoped, classical -> biology's fixed 0.3-2 s window, closed by 10 s    => EXPECT NO COMMIT

C is the isotope lever (F1->F2->F3, the model's one real attribution handle); D is the temporal-gap contrast
(the classical trace is dead where the coherent tag still credits). If C/D commit anyway, the readout is NOT
coherence-gated and that is a finding. Nothing is tuned.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sarahdavidson/posner_quantum_dynamics/src/models/Model_6")
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S, DELAY_S, REWARD_S, SETTLE_S = 0.5, 10.0, 20.0, 10.0
DA_BURST_S = 0.5   # GROUNDED: real phasic DA bursts are sub-second (Yagishita window 0.3-2 s), not 20 s
DA_TONIC, DA_BURST = 20e-9, 10e-6

ARMS = [("undoped", "quantum", None), ("Li6", "quantum", "Li6"),
        ("Li7", "quantum", "Li7"), ("classical-base", "classical", None)]


def run(mode, dopant, seed):
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    if dopant is not None:
        p.environment.dopant = dopant
    s = Model6QuantumSynapse(p)
    s._network_controlled = True
    s._reward_gated_consolidation = True
    s._reward_gating_mode = mode

    for _ in range(int(BUILD_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -40e-3})
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    ps_tag = float(getattr(s, "_mean_singlet_prob", np.nan))

    for _ in range(int(DELAY_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    ps_reward = float(getattr(s, "_mean_singlet_prob", np.nan))

    nburst = int(DA_BURST_S / DT)
    for k in range(int(REWARD_S / DT)):          # identical delayed BRIEF burst, then tonic
        s._da_signal = DA_BURST if k < nburst else DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    for _ in range(int(SETTLE_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})

    return dict(ps_tag=ps_tag, ps_reward=ps_reward, committed=bool(s._camkii_committed),
                mem=float(s.camkii.molecular_memory),
                readout_ca=float(getattr(s, "_readout_ca_uM", 0.0)))


if __name__ == "__main__":
    NS = 3
    WERNER = 1.0 / np.sqrt(2)
    print("=" * 96)
    print("F3 COHERENCE CONTROLS — identical delayed BURST at 10 s in every arm; only coherence/mode differ")
    print(f"Werner floor = {WERNER:.3f} (tag readable only above this)")
    print("=" * 96)
    print(f"  {'arm':>15} | {'P_S@tag':>8}{'P_S@reward':>11} | {'readout_Ca_uM':>13} | {'commit':>7} | {'final_mem':>9}")
    print("-" * 96)
    res = {}
    for name, mode, dop in ARMS:
        rs = [run(mode, dop, sd) for sd in range(NS)]
        cr = float(np.mean([r["committed"] for r in rs]))
        res[name] = cr
        print(f"  {name:>15} | {np.mean([r['ps_tag'] for r in rs]):>8.3f}"
              f"{np.mean([r['ps_reward'] for r in rs]):>11.3f} | "
              f"{np.mean([r['readout_ca'] for r in rs]):>13.2f} | {cr:>7.2f} | "
              f"{np.mean([r['mem'] for r in rs]):>9.3f}")
    print("-" * 96)
    ok_q = res["undoped"] >= 0.99 and res["Li6"] >= 0.99
    ok_iso = res["Li7"] <= 0.01
    ok_cls = res["classical-base"] <= 0.01
    print(f"\n  coherent tag credits at 10 s (undoped & Li6):      {'PASS' if ok_q else 'FAIL'}")
    print(f"  ISOTOPE LEVER — Li7 (decohered) does NOT credit:   {'PASS' if ok_iso else 'FAIL'}")
    print(f"  TEMPORAL GAP — classical baseline dead at 10 s:    {'PASS' if ok_cls else 'FAIL'}")
    print(f"\n  => {'COHERENCE-GATED reward-signed readout CONFIRMED' if (ok_q and ok_iso and ok_cls) else 'see failures above — report, do not tune'}")
