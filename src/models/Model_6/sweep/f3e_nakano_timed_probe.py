#!/usr/bin/env python3
"""
F3-e NAKANO-TIMED PROBE — can the reward-signed readout mechanism be DA-DECISIVE when dopamine FOLLOWS
calcium (Nakano 2010 timing) at a NEAR-THRESHOLD drive?

WHAT THIS TESTS (and, explicitly, what it does NOT):
  This asks whether the MECHANISM (bistable CaMKII + DARPP-32/PP1, standalone) CAN be tipped by dopamine when
  the drive sits near the switch threshold and dopamine arrives AFTER the calcium (and PP2B) has decayed.
  It does NOT claim the physiological readout drive actually sits near threshold — that is constraint #1
  (grounding that CaMKII integrates the diffuse PSD-distance ~1 µM calcium, not the nanodomain peak) and it
  remains a SEPARATE, later step. So:
    PASS  ⇒ "the mechanism supports a reward-signed readout with Nakano timing; NEXT, ground whether the
             physiological drive is near-threshold (constraint #1)."
    FAIL  ⇒ report the most likely unmet grounded constraint; do NOT parameter-hunt to force a pass.

THE FOUR COUPLED CONDITIONS (RESEARCH_DOPAMINE_CAMKII_REINFORCEMENT_2026-08-09, Part-2 constraint map):
  1. CaMKII sees PSD-distance ~1 µM calcium (near its K_calcium_half=1.0 µM) — SEPARATE later step; here we
     simply FEED 1 µM as the eligibility calcium (a modeled operating point, not a claim about the readout).
  2. The drive must sit NEAR the switch threshold — we CHARACTERIZE the switch by sweeping the field and read
     off where dopamine becomes load-bearing (this is a switch property, NOT tuning-to-decode).
  3. Dopamine must FOLLOW calcium, not overlap it — the protocol is TEMPORAL: brief 1 µM eligibility → decay
     (PP2B subsides) → DELAYED dopamine burst/dip/none.
  4. The bistable switch must HOLD near-threshold through the reward delay — VERIFIED IN THE DATA: we report
     the tonic-DA switch state at the END OF THE DELAY (pt286_delay_end). If it has already latched UP before
     the reward arrives, that IS the finding (reported, not fought by shrinking the delay).

THE SWITCH (bistable=True, PP1 Vmax=0.15, autocat=6.0, Km_pp1=0.2). At tonic DA / no field the saturating-PP1
switch has TWO stable attractors and an unstable separatrix (analytic, k_phos=0.1·autocat·p(1-p),
dephos=Vmax·p/(Km+p)):  DOWN ≈ 0.0,  separatrix ≈ 0.069,  UP ≈ 0.732.  Commitment is a BASIN classifier:
final pT286 > COMMIT_PT286 (=0.3, sitting between the separatrix 0.069 and the UP attractor 0.732) ⇒ latched UP.
This reads the switch's OWN attractor state, measured after DA returns to tonic and the field is OFF (so a
"commit" means a SELF-SUSTAINING memory, not a transient held up by the burst). molecular_memory is reported
too for continuity with f3e (but pT286 is the switch variable; mm folds in GluN2B occupancy nonlinearity).

DA-DECISIVE at a field row iff:  burst commit-rate ≥ 0.8  AND  dip ≤ 0.2  AND  none ≤ 0.2.
Positive control: a saturating field row must commit in EVERY DA condition (machinery works ⇒ a null is valid).
Negative control: the zero-drive row must stay DOWN.

EMERGENT DISCIPLINE: calcium (1 µM) and the DA-follows-Ca timing are grounded/cited; the field is SWEPT to
locate the switch threshold, NOT tuned to a decode; the commit threshold is the switch's own basin boundary.
No model source is edited; nothing is committed.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from camkii_module import CaMKIIModule, CaMKIIParameters
from darpp32_pp1_module import DARPP32PP1Module

DT = 0.05
# --- Nakano-timed protocol phases (grounded: DA FOLLOWS Ca; brief eligibility so the switch is not saturated) ---
T_ELIG   = 4.0     # s, brief near-threshold Hebbian coincidence (Ca + tag)
T_DELAY  = 8.0     # s, calcium decays, PP2B subsides, tag/field held — the "hold" window (constraint #4)
T_REWARD = 4.0     # s, the DELAYED dopamine event (burst / dip / none)
T_SETTLE = 6.0     # s, DA back to tonic AND field OFF — tests a SELF-SUSTAINING latch, not a held transient

CA_ELIG  = 1.0     # µM — PSD-distance diffuse spine calcium, near CaMKII K_calcium_half=1.0 (modeled op-point)
CA_BASAL = 0.1     # µM — basal

KD_D1 = 1e-6                                   # D1 low-affinity Kd (grounded, as in f3e harness)
DA_TONIC, DA_BURST, DA_DIP = 20e-9, 10e-6, 5e-9
def occ(da):
    return da / (da + KD_D1)

VMAX_PP1     = 0.15    # bistable PP1 Vmax (handoff: k_dephosphorylation ≈ 0.15 in bistable mode)
COMMIT_PT286 = 0.3     # basin boundary between separatrix (0.069) and UP attractor (0.732) at tonic/no-field
COMMIT_MM    = 0.5     # secondary (f3e continuity)

FIELDS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 24.0]   # 0 = negative control; 24 = saturating positive control
DA_CONDS = ["none", "dip", "burst"]
DA_OCC = {"none": occ(DA_TONIC), "dip": occ(DA_DIP), "burst": occ(DA_BURST)}


def build_camkii(seed):
    p = CaMKIIParameters()
    p.t286.bistable = True
    p.t286.k_dephosphorylation = VMAX_PP1   # PP1 Vmax (saturating) in bistable mode
    return CaMKIIModule(p, seed=seed)


def run_protocol(field, da_cond, seed):
    """One Nakano-timed run. Returns final pT286 / molecular_memory (measured at settle: DA tonic, field OFF),
    plus pt286 at the END OF THE DELAY under tonic DA (constraint-#4 hold check)."""
    ck = build_camkii(seed)
    dp = DARPP32PP1Module(da_tonic_occupancy=occ(DA_TONIC), ca_basal_uM=CA_BASAL)

    def phase(duration, ca, da_occupancy, fld):
        n = int(round(duration / DT))
        for _ in range(n):
            s = dp.step(DT, da_occupancy, ca)
            ck.step(DT, ca, quantum_field_kT=fld, pp1_factor=s["pp1_factor"])

    # 1. eligibility: brief 1 µM calcium + tag/field, tonic DA (DA has no vote yet — PP2B active)
    phase(T_ELIG, CA_ELIG, occ(DA_TONIC), field)
    # 2. delay: calcium decays to basal, field/tag HELD, DA still tonic (the hold window)
    phase(T_DELAY, CA_BASAL, occ(DA_TONIC), field)
    pt286_delay_end = ck.pT286                       # <-- constraint #4: did it hold, or latch prematurely?
    # 3. reward: the DELAYED dopamine event (burst/dip/none); calcium basal, field still held
    phase(T_REWARD, CA_BASAL, DA_OCC[da_cond], field)
    # 4. settle: DA back to tonic, field OFF — measure the self-sustaining state
    phase(T_SETTLE, CA_BASAL, occ(DA_TONIC), 0.0)

    return {"pt286": ck.pT286, "mm": ck.molecular_memory, "pt286_delay_end": pt286_delay_end}


def rates(field, n_seeds):
    """commit-rate (pT286 basin) + mean final pT286/mm per DA condition, and the tonic delay-end hold state."""
    out = {}
    delay_end = []
    for cond in DA_CONDS:
        commits, pts, mms = 0, [], []
        for seed in range(n_seeds):
            r = run_protocol(field, cond, seed)
            commits += (r["pt286"] > COMMIT_PT286)
            pts.append(r["pt286"]); mms.append(r["mm"])
            if cond == "none":
                delay_end.append(r["pt286_delay_end"])
        out[cond] = {"cr": commits / n_seeds, "pt": float(np.mean(pts)), "mm": float(np.mean(mms))}
    out["_delay_end_tonic"] = float(np.mean(delay_end))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    print("=" * 84)
    print("F3-e NAKANO-TIMED PROBE — is the mechanism DA-decisive with DA-follows-Ca timing?")
    print("=" * 84)
    print(f"protocol: elig {T_ELIG}s @Ca={CA_ELIG}µM+field | delay {T_DELAY}s @basal+field(held) | "
          f"reward {T_REWARD}s (DA) | settle {T_SETTLE}s (tonic, field OFF)")
    print(f"switch: bistable, PP1 Vmax={VMAX_PP1}, autocat=6.0, Km=0.2 | attractors DOWN~0.0 / sep~0.069 / UP~0.732")
    print(f"commit = final pT286 > {COMMIT_PT286} (basin); n={a.n} seeds/cell; mm=molecular_memory (secondary)")
    print()
    hdr = (f"{'field':>6} | {'none cr':>7}{'dip cr':>7}{'burst cr':>9} | "
           f"{'none pt':>8}{'dip pt':>7}{'burst pt':>9} | {'delayEnd(tonic)':>16}  DA-decisive?")
    print(hdr); print("-" * len(hdr))

    decisive, any_commit = [], False
    saturating_ok = None
    negctrl_down = None
    for field in FIELDS:
        r = rates(field, a.n)
        crs = {c: r[c]["cr"] for c in DA_CONDS}
        any_commit = any_commit or any(v > 0 for v in crs.values())
        dec = crs["burst"] >= 0.8 and crs["dip"] <= 0.2 and crs["none"] <= 0.2
        if dec:
            decisive.append(field)
        de = r["_delay_end_tonic"]
        latch_flag = "  <-LATCHED-in-delay" if de > COMMIT_PT286 else ""
        print(f"{field:>6.1f} | {crs['none']:>7.2f}{crs['dip']:>7.2f}{crs['burst']:>9.2f} | "
              f"{r['none']['pt']:>8.2f}{r['dip']['pt']:>7.2f}{r['burst']['pt']:>9.2f} | "
              f"{de:>16.3f}{latch_flag}  {'*** YES' if dec else ''}")
        if field == max(FIELDS):
            saturating_ok = all(crs[c] >= 0.8 for c in DA_CONDS)
        if field == 0.0:
            negctrl_down = all(crs[c] <= 0.2 for c in DA_CONDS)

    print("-" * len(hdr))
    # -------- controls + verdict --------
    print(f"positive control (field={max(FIELDS)}: all DA conditions commit): "
          f"{'PASS' if saturating_ok else 'FAIL'}")
    print(f"negative control (field=0: nothing commits): {'PASS' if negctrl_down else 'FAIL'}")
    if not any_commit:
        print("\nINVALID — nothing commits anywhere. The machinery does not drive the switch; a null is not "
              "interpretable. (Would need to check field/calcium/durations before any verdict.)")
    elif not saturating_ok:
        print("\nINVALID — positive control failed (the switch cannot be latched even by a saturating field). "
              "A negative DA-decisive result is not interpretable until the machinery is shown to work.")
    elif decisive:
        print(f"\nVERDICT: DA-DECISIVE at near-threshold field(s) {decisive}.")
        print("  => The MECHANISM supports a reward-signed readout with Nakano timing: at a near-threshold drive,")
        print("     a delayed dopamine burst latches the switch UP while dip/none stay DOWN.")
        print("  => NEXT (separate step): ground whether the PHYSIOLOGICAL readout drive sits near this threshold")
        print("     (constraint #1: CaMKII integrating diffuse PSD ~1 µM calcium, not the nanodomain peak).")
    else:
        print("\nVERDICT: NOT DA-decisive at any field (but the switch DOES latch — positive control passed).")
        print("  => Report the unmet grounded constraint (see delayEnd column for premature-latch evidence);")
        print("     do NOT parameter-hunt to force a pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
