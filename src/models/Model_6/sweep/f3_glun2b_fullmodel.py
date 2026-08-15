#!/usr/bin/env python3
"""
F3 END-TO-END in the FULL MODEL with the grounded memory architecture (glun2b_memory):
  eligibility (tag forms) -> DELAY (tag holds coherence; CaMKII resets) -> delayed reward {none/dip/burst}
  -> settle. Commit = the persistent CaMKII-GluN2B complex (molecular_memory).

Tests the three things that must hold for a reward-signed readout:
  (1) HEBBIAN-SAFE: eligibility alone does not commit (checked at delay end, before reward).
  (2) DA-DECISIVE: burst commits; dip/none do not (DAPK1 gates the binding on dopamine).
  (3) PERSISTENT: the complex survives to the end of settle (pT286 has decayed by then).
Also reports pT286 (should be TRANSIENT) and P_S at reward (the coherent tag = the eligibility carrier).
A null is a result; nothing here is tuned.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sarahdavidson/posner_quantum_dynamics/src/models/Model_6")
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S, DELAY_S, REWARD_S, SETTLE_S = 0.5, 10.0, 20.0, 10.0
DA_TONIC, DA_BURST, DA_DIP = 20e-9, 10e-6, 5e-9
CONDS = {"none": DA_TONIC, "dip": DA_DIP, "burst": DA_BURST}


def run(cond, seed):
    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    s = Model6QuantumSynapse(p)
    s._network_controlled = True
    s._reward_gated_consolidation = True
    s._reward_gating_mode = "quantum"

    for _ in range(int(BUILD_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -40e-3})
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    ps_tag = float(getattr(s, "_mean_singlet_prob", np.nan))
    n_dim = len(s.dimer_particles.dimers)

    for _ in range(int(DELAY_S / DT)):                       # hold: tag decoheres passively; CaMKII resets
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    pre = dict(committed=bool(s._camkii_committed), pT286=float(s.camkii.pT286),
               mem=float(s.camkii.molecular_memory), ps=float(getattr(s, "_mean_singlet_prob", np.nan)))

    for _ in range(int(REWARD_S / DT)):                      # the delayed reward event
        s._da_signal = CONDS[cond]
        s.step(DT, {"voltage": -70e-3})
    at_reward_pt = float(s.camkii.pT286)

    for _ in range(int(SETTLE_S / DT)):                      # settle: DA back to tonic
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})

    return dict(cond=cond, n_dim=n_dim, ps_tag=ps_tag, ps_delay_end=pre["ps"],
                pre_commit=pre["committed"], pre_mem=pre["mem"], pre_pT286=pre["pT286"],
                reward_pT286=at_reward_pt, final_pT286=float(s.camkii.pT286),
                final_mem=float(s.camkii.molecular_memory), committed=bool(s._camkii_committed))


if __name__ == "__main__":
    NS = 3
    print("=" * 100)
    print("F3 END-TO-END (full model) — grounded GluN2B structural-latch memory, delayed reward")
    print(f"build {BUILD_S}s | delay {DELAY_S}s | reward {REWARD_S}s | settle {SETTLE_S}s | n={NS}")
    print("=" * 100)
    print(f"  {'cond':>6} | {'commit':>7} | {'pre-commit(delay end)':>21} | {'pT286: delay/reward/final':>26} | "
          f"{'final_mem':>9} | {'P_S@reward':>10}")
    print("-" * 100)
    summ = {}
    for cond in CONDS:
        rs = [run(cond, sd) for sd in range(NS)]
        cr = float(np.mean([r["committed"] for r in rs]))
        pre = float(np.mean([r["pre_commit"] for r in rs]))
        summ[cond] = (cr, pre)
        print(f"  {cond:>6} | {cr:>7.2f} | {pre:>21.2f} | "
              f"{np.mean([r['pre_pT286'] for r in rs]):>8.3f}"
              f"{np.mean([r['reward_pT286'] for r in rs]):>9.3f}"
              f"{np.mean([r['final_pT286'] for r in rs]):>9.3f} | "
              f"{np.mean([r['final_mem'] for r in rs]):>9.3f} | "
              f"{np.mean([r['ps_delay_end'] for r in rs]):>10.3f}")
    print("-" * 100)
    hebbian_safe = all(summ[c][1] <= 0.01 for c in CONDS)
    decisive = summ["burst"][0] >= 0.99 and summ["dip"][0] <= 0.01 and summ["none"][0] <= 0.01
    print(f"\n  (1) HEBBIAN-SAFE (no pre-commit before reward): {'PASS' if hebbian_safe else 'FAIL'}")
    print(f"  (2) DA-DECISIVE (burst commits; dip/none do not): {'PASS' if decisive else 'FAIL'}")
    print(f"  => {'PRODUCES end-to-end in the full model' if (hebbian_safe and decisive) else 'NOT yet — report the unmet condition; do NOT tune.'}")
