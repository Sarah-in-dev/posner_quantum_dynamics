#!/usr/bin/env python3
"""
SWITCH-REGIME BENCH (fast, modules only) — is there a GROUNDED regime where the bistable CaMKII switch is
DOWN-stable under Hebbian eligibility calcium at tonic dopamine, and commits only when the reward readout
calcium arrives WITH dopamine's PP1-inhibition? (Yagishita: Hebbian Ca alone insufficient; dopamine reinforces.)

Mechanism under test (grounded): dopamine sets PP1 via DARPP-32 (burst -> PP1 inhibited ~0.12x; dip -> ~1.07x;
tonic -> 1.0x). PP1 sets the switch's commitment threshold. So the SAME calcium commits with a burst (PP1 low,
threshold low) but not with a dip/none (PP1 active, threshold high). The DARPP-32 cascade runs CONTINUOUSLY
(physiological), so Hebbian Ca -> PP2B -> PP1 active during eligibility too.

Protocol (per cell): eligibility (Ca_elig, tonic DA) -> hold (basal, tonic DA) -> reward (Ca_readout, DA cond)
-> settle (basal, tonic DA). Commit = final pT286 in the UP basin (> 0.3).
DA-DECISIVE iff: eligibility does NOT latch (pt286 at hold-end < 0.3), burst commits, dip & none do not.

Sweep PP1 Vmax over a GROUNDED range: Zhabotinsky/Graupner say PP1 dephosphorylation is COMPARABLE to the
autophosphorylation rate; autophos ~ k_phos_max * autocat = 0.1*6 = 0.6, so Vmax in [0.1, 0.6] is the grounded
band. Ca_elig = 1 uM (physiological single-EPSP Hebbian peak); Ca_readout = 3 uM (binding-melt shower, ~F3-e
sustained). These are GROUNDED inputs, not tuned. If NO grounded regime produces it, that is the finding.
"""
import sys, os
import numpy as np
sys.path.insert(0, "/Users/sarahdavidson/posner_quantum_dynamics/src/models/Model_6")
import logging; logging.disable(logging.INFO)
from camkii_module import CaMKIIModule, CaMKIIParameters
from darpp32_pp1_module import DARPP32PP1Module

DT = 0.05
T_ELIG, T_HOLD, T_REWARD, T_SETTLE = 2.0, 6.0, 2.0, 6.0
CA_ELIG, CA_READOUT, CA_BASAL = 1.0, 3.0, 0.1     # uM — grounded (physiological Hebbian / readout shower / rest)
KD_D1 = 1e-6
DA_TONIC, DA_BURST, DA_DIP = 20e-9, 10e-6, 5e-9
def occ(da): return da / (da + KD_D1)
COMMIT_PT286 = 0.3
CONDS = {"none": DA_TONIC, "dip": DA_DIP, "burst": DA_BURST}


def build_ck(pp1_vmax, seed):
    p = CaMKIIParameters()
    p.t286.bistable = True
    p.t286.k_dephosphorylation = pp1_vmax
    return CaMKIIModule(p, seed=seed)


def run(pp1_vmax, cond, seed):
    ck = build_ck(pp1_vmax, seed)
    dp = DARPP32PP1Module(da_tonic_occupancy=occ(DA_TONIC), ca_basal_uM=CA_BASAL)

    def phase(dur, ca, da):
        for _ in range(int(round(dur / DT))):
            s = dp.step(DT, da, ca)                 # DARPP-32 runs CONTINUOUSLY (physiological)
            ck.step(DT, ca, pp1_factor=s["pp1_factor"])

    phase(T_ELIG, CA_ELIG, occ(DA_TONIC))
    phase(T_HOLD, CA_BASAL, occ(DA_TONIC))
    pt_hold_end = ck.pT286                          # did eligibility latch it? (before reward)
    phase(T_REWARD, CA_READOUT, occ(CONDS[cond]))
    phase(T_SETTLE, CA_BASAL, occ(DA_TONIC))
    return pt_hold_end, ck.pT286


if __name__ == "__main__":
    NS = 3
    print("=" * 96)
    print("SWITCH-REGIME BENCH — grounded search for a DA-decisive, Hebbian-safe regime")
    print(f"Ca_elig={CA_ELIG}uM (Hebbian) | Ca_readout={CA_READOUT}uM (shower) | autocat=6.0 Km=0.2 | "
          f"autophos~{0.1*6.0:.1f} (Zhabotinsky band Vmax<=0.6)")
    print("=" * 96)
    print(f"  {'PP1_Vmax':>8} | {'elig_latched?':>13} | {'none':>6}{'dip':>6}{'burst':>7} (commit-rate) | verdict")
    print("-" * 96)
    for vmax in [0.1, 0.2, 0.3, 0.4, 0.6]:
        cr, hold = {}, []
        for cond in CONDS:
            commits = 0
            for seed in range(NS):
                phe, ptf = run(vmax, cond, seed)
                commits += (ptf > COMMIT_PT286)
                if cond == "none":
                    hold.append(phe)
            cr[cond] = commits / NS
        elig_latched = np.mean(hold) > COMMIT_PT286
        decisive = (not elig_latched) and cr["burst"] >= 0.99 and cr["dip"] <= 0.01 and cr["none"] <= 0.01
        v = ("*** DA-DECISIVE + Hebbian-safe" if decisive else
             ("elig latches (Hebbian commits)" if elig_latched else
              ("no separation" if cr["burst"] <= cr["none"] + 0.01 else "partial")))
        print(f"  {vmax:>8.2f} | {'YES' if elig_latched else 'no':>13} "
              f"(pt={np.mean(hold):.2f}) | {cr['none']:>6.2f}{cr['dip']:>6.2f}{cr['burst']:>7.2f} | {v}")
    print("-" * 96)
    print("A grounded DA-decisive + Hebbian-safe row => wire (Ca-feed already landed; add continuous DARPP-32 +")
    print("that PP1 Vmax + reward-readout Ca) into model6_core and re-validate. NO row => structural finding.")
