#!/usr/bin/env python3
"""
F3-e RESOLUTION EXPERIMENT — is there a (calcium, PP1-strength) regime where CaMKII commitment is DECIDED by
dopamine, rather than saturated by calcium alone?

CONTEXT (F3-e, RESEARCH_LOG_CALCIUM_DIMER.md): the through-CaMKII correction is biologically correct but the
reward channel is INERT because the readout Ca²⁺ (~3 µM sustained) saturates CaMKII (k_phos ≫ k_dephos=0.001),
so calcium commits on its own and dopamine's DARPP-32/PP1 reinforcement (pp1_factor≈0.95) can't get a vote.
Biology (Zhabotinsky 2000; Graupner 2007; Fernandez 2006; Nakano 2010): CaMKII is a bistable switch set by a
Ca–PP1 BALANCE, and "potentiation does not occur to dopamine OR glutamate alone" — Ca-alone must be sub-threshold
and dopamine (PP1 inhibition) must be load-bearing.

THE MEASUREMENT (pre-registered): sweep calcium × PP1-strength × dopamine, drive a CaMKII + DARPP-32/PP1 pair to
consolidation from rest, and ask where dopamine DECIDES commitment. A (ca, pp1_strength) cell is DA-DECISIVE iff
  burst commit-rate ≥ 0.8  AND  dip commit-rate ≤ 0.2  AND  no-DA commit-rate ≤ 0.2.
If ≥1 DA-decisive cell exists at physiological calcium (≤ a few µM), the reward-signed readout is achievable
there (resolves the F3-e blocker → set the operating point, re-validate, then network). If NONE, the model cannot
support DA-decisive CaMKII commitment in this regime — a structural finding.

EMERGENT DISCIPLINE: pp1_strength (the PP1 dephosphorylation of CaMKII-pT286) is the grounded counterforce whose
scale the bistable-switch models set relative to k_phos; we SWEEP it to LOCATE the balance regime, we do NOT tune
it to a downstream decode. The DA-decisive criterion is fixed BEFORE the run.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from camkii_module import CaMKIIModule, CaMKIIParameters
from darpp32_pp1_module import DARPP32PP1Module

DT = 0.05                 # CaMKII/DARPP dynamics are ~seconds; 50 ms is fine
CONSOLIDATE_S = 25.0      # long enough for GluN2B binding to follow pT286 (tau ~ s)
KD_D1 = 1e-6              # D1 low-affinity Kd (grounded)
DA_TONIC, DA_BURST, DA_DIP = 20e-9, 10e-6, 5e-9
def occ(da):
    return da / (da + KD_D1)
DA_CONDS = [("none", occ(DA_TONIC)), ("dip", occ(DA_DIP)), ("burst", occ(DA_BURST))]
# The DRIVE toward the CaMKII threshold = quantum dimer field (barrier reduction, the model's actual
# commitment driver) at a physiological calcium. Sweep the field to cross the threshold; find where PP1
# (dopamine-controlled) makes the crossing DA-DECISIVE.
CA_UM = 2.0                                  # µM — physiological LTP-range spine calcium (a few µM)
FIELDS = [0.0, 6.0, 12.0, 18.0, 24.0]        # kT dimer field (0=none .. 24=~10-synapse collective)
PP1_MULT = [1, 30, 100, 300]                 # PP1 strength = k_dephos_base(0.001) × mult vs k_phos_max(0.1)
COMMIT_THR = 0.5


def pp1_steady_map():
    """Steady pp1_factor per dopamine condition at CA_UM (captures Ca→PP2B→PP1-up vs DA→Thr34→PP1-down)."""
    out = {}
    for name, o in DA_CONDS:
        m = DARPP32PP1Module(da_tonic_occupancy=occ(DA_TONIC), ca_basal_uM=0.1)
        s = {"pp1_factor": 1.0}
        for _ in range(int(6.0 / 0.02)):
            s = m.step(0.02, o, CA_UM)
        out[name] = s["pp1_factor"]
    return out


def commit_rate(field, mult, pf, n_seeds):
    committed = 0
    for seed in range(n_seeds):
        p = CaMKIIParameters(); p.t286.k_dephosphorylation = 0.001 * mult
        ck = CaMKIIModule(p, seed=seed)
        for _ in range(int(CONSOLIDATE_S / DT)):
            ck.step(DT, CA_UM, quantum_field_kT=field, pp1_factor=pf)
        committed += (ck.molecular_memory > COMMIT_THR)
    return committed / n_seeds


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
    pf = pp1_steady_map()
    print(f"CA={CA_UM} µM; consolidation {CONSOLIDATE_S}s; commit = molecular_memory > {COMMIT_THR}; n={a.n}/cell")
    print(f"pp1_factor at CA: none={pf['none']:.2f} dip={pf['dip']:.2f} burst={pf['burst']:.2f}")
    print(f"{'field(kT)':>9}{'pp1x':>6} | {'none':>6}{'dip':>6}{'burst':>7}  DA-decisive?")
    print("-" * 60)
    decisive, any_commit = [], False
    for field in FIELDS:
        for mult in PP1_MULT:
            r = {name: commit_rate(field, mult, pf[name], a.n) for name in ("none", "dip", "burst")}
            any_commit = any_commit or any(v > 0 for v in r.values())
            dec = r["burst"] >= 0.8 and r["dip"] <= 0.2 and r["none"] <= 0.2
            if dec:
                decisive.append((field, mult))
            print(f"{field:>9.1f}{mult:>6d} | {r['none']:>6.2f}{r['dip']:>6.2f}{r['burst']:>7.2f}"
                  f"  {'*** YES' if dec else ''}")
    print("-" * 60)
    if not any_commit:
        print("INVALID — positive control failed: nothing commits at any field/PP1. Raise field/calcium/time.")
    elif decisive:
        print(f"DA-DECISIVE regime FOUND at (field kT, pp1×): {decisive}")
        print("=> reward-signed readout is achievable there. Resolve: set operating point (grounded), re-validate.")
    else:
        print("NO DA-decisive cell (but some commit) — dopamine cannot flip commitment in this range.")
        print("=> structural finding: the Ca+field drive is above threshold regardless of PP1/dopamine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
