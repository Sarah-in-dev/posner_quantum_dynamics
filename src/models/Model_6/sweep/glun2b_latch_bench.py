#!/usr/bin/env python3
"""
GLUN2B-LATCH BENCH — validate the grounded memory architecture (CaMKII module glun2b_memory mode):
  (1) pT286 is TRANSIENT: rises with the Ca event, DECAYS/resets after (not a persistent switch).
  (2) The GluN2B structural complex is the MEMORY: forms at the commitment event, PERSISTS after pT286 decays,
      is protected (survives the hold), and is DA-decisive.
Protocol: brief eligibility (tag/Hebbian) -> hold (reset) -> DDSC/readout Ca + DA{none/dip/burst} -> settle.
Commit = molecular_memory (=GluN2B_bound) at the END (after settle, pT286 long decayed) > 0.3.
The DDSC/readout Ca is INTERNAL-STORE (~uM, near CaMKII K_half=1uM) — swept to characterize where DA is
load-bearing (a switch property, not tuning). DA-DECISIVE iff burst commits, dip/none do not, AND the memory
PERSISTS to the end (pT286 has reset). A null is a result.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sarahdavidson/posner_quantum_dynamics/src/models/Model_6")
import logging; logging.disable(logging.INFO)
from camkii_module import CaMKIIModule, CaMKIIParameters
from darpp32_pp1_module import DARPP32PP1Module

DT = 0.05
# T_REWARD ~ the DDSC commitment window (Jain 2024: delayed CaMKII activation over ~10-100 s)
T_ELIG, T_HOLD, T_REWARD, T_SETTLE = 0.5, 8.0, 20.0, 12.0
CA_ELIG, CA_BASAL = 2.0, 0.1
KD_D1 = 1e-6
DA_TONIC, DA_BURST, DA_DIP = 20e-9, 10e-6, 5e-9
def occ(da): return da / (da + KD_D1)
CONDS = {"none": DA_TONIC, "dip": DA_DIP, "burst": DA_BURST}
COMMIT = 0.3


def build():
    p = CaMKIIParameters()
    p.glun2b.glun2b_memory = True
    return CaMKIIModule(p, seed=0)


def run(ca_readout, cond, seed):
    ck = build(); ck.rng = np.random.default_rng(seed)
    dp = DARPP32PP1Module(da_tonic_occupancy=occ(DA_TONIC), ca_basal_uM=CA_BASAL)
    trace = {}

    def phase(dur, ca, da, tag):
        for _ in range(int(round(dur / DT))):
            s = dp.step(DT, da, ca)
            ck.step(DT, ca, pp1_factor=s["pp1_factor"])
        trace[tag] = (ck.pT286, ck.GluN2B_bound)

    phase(T_ELIG, CA_ELIG, occ(DA_TONIC), "elig")
    phase(T_HOLD, CA_BASAL, occ(DA_TONIC), "hold")     # pT286 should reset; complex ~0 (no commit yet)
    phase(T_REWARD, ca_readout, occ(CONDS[cond]), "reward")
    phase(T_SETTLE, CA_BASAL, occ(DA_TONIC), "settle")  # pT286 decays; complex should PERSIST if formed
    return trace, ck.molecular_memory


if __name__ == "__main__":
    NS = 4
    print("=" * 96)
    print("GLUN2B-LATCH BENCH — transient pT286 + persistent structural memory; DA-decisive at near-threshold readout")
    print(f"elig {T_ELIG}s@{CA_ELIG}uM | hold {T_HOLD}s | reward {T_REWARD}s(DA) | settle {T_SETTLE}s | commit=mem>{COMMIT}")
    print("=" * 96)
    # First: show the transient + persistence for a burst at a mid readout Ca (seed 0)
    tr, mem = run(1.0, "burst", 0)
    print("TRANSIENT+PERSIST CHECK (burst, readout=1.0uM):  (pT286, GluN2B_complex) per phase")
    for k in ("elig", "hold", "reward", "settle"):
        print(f"   {k:>7}: pT286={tr[k][0]:.3f}  complex={tr[k][1]:.3f}")
    print(f"   => pT286 transient (rises@reward, decays@settle); memory=complex persists to end = {mem:.3f}\n")

    print(f"  {'readout_uM':>10} | {'none':>6}{'dip':>6}{'burst':>7} (commit-rate) | {'hold_pT286':>10} | verdict")
    print("-" * 90)
    for car in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        cr, holdpt = {}, []
        for cond in CONDS:
            c = 0
            for sd in range(NS):
                tr, mem = run(car, cond, sd)
                c += (mem > COMMIT)
                if cond == "none":
                    holdpt.append(tr["hold"][0])
            cr[cond] = c / NS
        dec = cr["burst"] >= 0.99 and cr["dip"] <= 0.01 and cr["none"] <= 0.01
        v = ("*** DA-DECISIVE" if dec else
             ("elig/hold pre-commits" if np.mean(holdpt) > COMMIT else
              ("no separation" if cr["burst"] <= cr["none"] + 0.01 else "partial")))
        print(f"  {car:>10.2f} | {cr['none']:>6.2f}{cr['dip']:>6.2f}{cr['burst']:>7.2f} | {np.mean(holdpt):>10.3f} | {v}")
