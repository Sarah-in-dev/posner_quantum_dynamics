"""
DARPP-32 / PP1 signaling module — dopamine REINFORCES CaMKII (it does not bypass it)
====================================================================================

Grounding: docs/RESEARCH_DOPAMINE_CAMKII_REINFORCEMENT_2026-08-09.md (Yagishita 2014; Nakano 2010;
Fernandez 2006; Greengard/Svenningsson DARPP-32/PP-1 cascade; Xiao 2023; Jain 2024).

WHY THIS EXISTS: the reward-gated readout must act THROUGH CaMKII, by controlling PP1's dephosphorylation
of CaMKII-pThr286. PP1 activity is the control variable dopamine reaches. This module computes PP1 activity
from dopamine (D1 occupancy) and calcium via the DARPP-32 phospho-state; the CaMKII module multiplies its
k_dephosphorylation by `pp1_factor` (normalized so tonic dopamine ⇒ 1.0 ⇒ bit-identical when unwired).

THE CASCADE (Nakano 2010 structure):
  LTP arm : D1→cAMP→PKA → phosphorylate DARPP-32-Thr34 → phospho-Thr34 INHIBITS PP1 → CaMKII-pT286 persists.
  LTD arm : Ca→PP2B(calcineurin) DEphosphorylates Thr34 → PP1 disinhibited (active) → strips CaMKII-pT286.
  Gain    : Ca→Cdk5→DARPP-32-Thr75 → INHIBITS PKA (dominant at WEAK Ca);  strong Ca→PP2A→dephos-Thr75 →
            disinhibits PKA. This Ca-amplitude switch sets LTD (weak Ca) vs LTP (strong Ca) directionality.
  So the LTP/LTD SIGN is EMERGENT from PP1 activity (PP1 down → LTP, PP1 up → LTD), NOT an imposed ±1.

EMERGENT-PHYSICS DISCIPLINE: the cascade STRUCTURE and the Ca thresholds / DA window are grounded and cited.
Rate constants without a direct source are tagged [MODELED] and set from the biology's timescales — NEVER
tuned to make a downstream readout decode. If the grounded cascade does not produce the correct sign, that
is a FINDING, not a license to tune.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DARPP32PP1Parameters:
    # --- concentrations ---
    DARPP32_total_uM: float = 50.0     # [GROUNDED] DARPP-32 ~50 µM in MSN spines (Ouimet/Greengard)
    # --- Thr34 (PKA up; PP2B down) — the PP1-inhibition site ---
    k_pka_thr34: float = 4.0           # [MODELED] s⁻¹, PKA phosphorylation of Thr34 (fast, DA-window)
    k_pp2b_thr34: float = 6.0          # [MODELED] s⁻¹, PP2B(calcineurin) dephosphorylation of Thr34
    k_basal_thr34: float = 0.3         # [MODELED] s⁻¹, basal Thr34 dephosphorylation
    # --- Thr75 (Cdk5 up; PP2A down) — the PKA-gain / Ca-amplitude switch ---
    k_cdk5_thr75: float = 0.8          # [MODELED] s⁻¹, Cdk5 phosphorylation of Thr75 (constitutive-ish)
    k_pp2a_thr75: float = 3.0          # [MODELED] s⁻¹, PP2A dephosphorylation of Thr75 (strong-Ca gated)
    # --- calcium-activated phosphatases (Hill) ---
    ca_pp2b_half_uM: float = 0.5       # [GROUNDED] calcineurin Ca/CaM half-activation ~0.5 µM (Stemmer/Klee)
    ca_pp2b_hill: float = 3.0          # [MODELED] cooperativity
    ca_pp2a_half_uM: float = 1.0       # [GROUNDED-ish] PP2A/Thr75 strong-Ca switch ~1 µM (Nakano directionality)
    ca_pp2a_hill: float = 3.0          # [MODELED]
    # --- PKA drive from dopamine (D1 occupancy → cAMP → PKA), gated by Thr75 inhibition ---
    da_pka_half: float = 0.5           # [MODELED] D1 occupancy for half-max PKA drive
    da_pka_hill: float = 2.0           # [MODELED] cooperative (AC5/cAMP threshold)
    # --- Ser130 (Ser137 in rat): the CK1 site that PROTECTS Thr34 from calcineurin ---
    # [GROUNDED — Desdouits et al. 1995, title result: "Phosphorylation of Ser-137 by casein kinase I inhibits
    # dephosphorylation of Thr-34 by calcineurin"; confirmed in vitro AND in vivo (Frontiers Behav Neurosci 2011
    # review). DARPP-32 is phosphorylated on this site by CK1 under BASAL conditions; it is dephosphorylated by
    # PP2C, and PP2C-mediated loss of Ser-137 FACILITATES Thr-34 dephosphorylation by calcineurin.]
    # [GROUNDED — the incoherent feedforward] PP2B (calcineurin) dephosphorylates CK1 and thereby ENHANCES Ser137
    # phosphorylation, so Ca²⁺ simultaneously ACTIVATES (via PP2B→Thr34) and INHIBITS (via CK1→Ser137→protection)
    # the removal of Thr34. This brake is why a BRIEF dopamine burst can still leave Thr34 standing while the
    # commitment calcium is present — the tension that made the model's brief-burst arm fail with no Ser130 state.
    # The RATES are not published quantitatively (the review states the CK1 mechanism "remains incompletely
    # understood"), so they are [MODELED] from the biology's timescales and the qualitative constraints:
    # basal Ser137 substantially phosphorylated, and calcium INCREASING it.
    ser130_protection: bool = True     # opt-out flag; False reproduces the pre-Ser130 behaviour exactly
    k_ck1_ser130: float = 0.5          # [MODELED] s⁻¹, CK1 phosphorylation of Ser130
    k_pp2c_ser130: float = 0.3         # [MODELED] s⁻¹, PP2C dephosphorylation of Ser130
    ck1_basal: float = 0.5             # [MODELED] fraction of CK1 active at basal Ca (basal phosphorylation)
    ck1_ca_gain: float = 0.5           # [MODELED] additional CK1 activation as PP2B rises (the feedforward)
    prot_frac_max: float = 0.8         # [MODELED] maximal fractional inhibition of PP2B→Thr34 by phospho-Ser130

    # --- PP1 inhibition by phospho-Thr34 (potent) ---
    K_inhib_thr34: float = 0.02        # [MODELED] effective D34 fraction for half PP1 inhibition. The raw
                                       #   Hemmings 1984 Ki≈1 nM with 50 µM DARPP-32 gives a MUCH steeper
                                       #   switch; this effective constant reflects the fraction of PP1
                                       #   locally accessible to the inhibitor. Flagged, not outcome-tuned.
    pp1_leak: float = 0.05             # [MODELED] residual PP1 activity when fully inhibited (never exactly 0)


class DARPP32PP1Module:
    """State = DARPP-32 phospho-fractions (Thr34, Thr75). Output = PP1 activity factor for CaMKII k_dephos."""

    def __init__(self, params: Optional[DARPP32PP1Parameters] = None,
                 da_tonic_occupancy: float = 0.1, ca_basal_uM: float = 0.1):
        self.p = params or DARPP32PP1Parameters()
        self.thr34 = 0.0
        self.thr75 = 0.0
        self.ser130 = 0.0          # CK1 site protecting Thr34 from calcineurin (Desdouits 1995)
        self.time = 0.0
        # settle to the tonic steady state so pp1_factor is normalized to 1.0 at tonic DA / basal Ca
        self._relax_to_steady(da_tonic_occupancy, ca_basal_uM)
        self._pp1_ref = self._pp1_activity(self.thr34)
        self.history = {'time': [], 'thr34': [], 'thr75': [], 'ser130': [], 'pka': [], 'pp1_factor': []}

    # ---- calcium-gated phosphatase activities (Hill) ----
    def _pp2b(self, ca_uM):
        c = max(ca_uM, 0.0) ** self.p.ca_pp2b_hill
        return c / (self.p.ca_pp2b_half_uM ** self.p.ca_pp2b_hill + c)

    def _pp2a(self, ca_uM):
        c = max(ca_uM, 0.0) ** self.p.ca_pp2a_hill
        return c / (self.p.ca_pp2a_half_uM ** self.p.ca_pp2a_hill + c)

    def _pka(self, da_occ):
        """PKA drive from D1 occupancy (cAMP/AC5), INHIBITED by phospho-Thr75 (the Ca-amplitude gain switch)."""
        d = max(da_occ, 0.0) ** self.p.da_pka_hill
        drive = d / (self.p.da_pka_half ** self.p.da_pka_hill + d)
        return drive * (1.0 - self.thr75)          # Thr75-P inhibits PKA (Nakano/Greengard)

    def _pp1_activity(self, thr34):
        """Phospho-Thr34-DARPP-32 inhibits PP1. More Thr34-P → less PP1 activity."""
        inhib = thr34 / (thr34 + self.p.K_inhib_thr34)
        return self.p.pp1_leak + (1.0 - self.p.pp1_leak) * (1.0 - inhib)

    def _derivs(self, da_occ, ca_uM):
        pka = self._pka(da_occ)
        pp2b = self._pp2b(ca_uM)
        pp2a = self._pp2a(ca_uM)
        # Ser130/CK1: PP2B dephosphorylates (activates) CK1, so calcium RAISES Ser130 — the incoherent
        # feedforward. Phospho-Ser130 then PROTECTS Thr34 from calcineurin (Desdouits 1995).
        if self.p.ser130_protection:
            ck1_act = min(1.0, self.p.ck1_basal + self.p.ck1_ca_gain * pp2b)
            d_ser130 = (self.p.k_ck1_ser130 * ck1_act * (1.0 - self.ser130)
                        - self.p.k_pp2c_ser130 * self.ser130)
            protection = 1.0 - self.p.prot_frac_max * self.ser130
        else:
            d_ser130 = 0.0
            protection = 1.0
        d_thr34 = (self.p.k_pka_thr34 * pka * (1.0 - self.thr34)
                   - (self.p.k_pp2b_thr34 * pp2b * protection + self.p.k_basal_thr34) * self.thr34)
        d_thr75 = (self.p.k_cdk5_thr75 * (1.0 - self.thr75)
                   - self.p.k_pp2a_thr75 * pp2a * self.thr75)
        return d_thr34, d_thr75, d_ser130, pka

    def _relax_to_steady(self, da_occ, ca_uM, dt=0.01, n=20000):
        for _ in range(n):
            d34, d75, d130, _ = self._derivs(da_occ, ca_uM)
            self.thr34 = float(np.clip(self.thr34 + d34 * dt, 0.0, 1.0))
            self.thr75 = float(np.clip(self.thr75 + d75 * dt, 0.0, 1.0))
            self.ser130 = float(np.clip(self.ser130 + d130 * dt, 0.0, 1.0))

    def step(self, dt: float, da_occupancy: float, calcium_uM: float) -> Dict:
        """Advance the cascade; return PP1 factor (normalized to tonic=1.0) for CaMKII k_dephos scaling."""
        self.time += dt
        # sub-step for stability with the fast Thr34 rates
        nsub = max(1, int(np.ceil(dt / 0.005)))
        h = dt / nsub
        for _ in range(nsub):
            d34, d75, d130, pka = self._derivs(da_occupancy, calcium_uM)
            self.thr34 = float(np.clip(self.thr34 + d34 * h, 0.0, 1.0))
            self.thr75 = float(np.clip(self.thr75 + d75 * h, 0.0, 1.0))
            self.ser130 = float(np.clip(self.ser130 + d130 * h, 0.0, 1.0))
        pp1 = self._pp1_activity(self.thr34)
        pp1_factor = pp1 / self._pp1_ref if self._pp1_ref > 0 else 1.0
        self.history['time'].append(self.time); self.history['thr34'].append(self.thr34)
        self.history['thr75'].append(self.thr75); self.history['ser130'].append(self.ser130)
        self.history['pka'].append(pka)
        self.history['pp1_factor'].append(pp1_factor)
        return {'thr34': self.thr34, 'thr75': self.thr75, 'ser130': self.ser130, 'pka': pka,
                'pp1_activity': pp1, 'pp1_factor': float(pp1_factor)}

    def get_pp1_factor(self) -> float:
        return self.history['pp1_factor'][-1] if self.history['pp1_factor'] else 1.0


# =============================================================================
# VALIDATION — does the correct LTP/LTD sign EMERGE from the grounded cascade?
# =============================================================================
if __name__ == "__main__":
    def run(da_occ, ca_uM, seconds=3.0, dt=0.02, da_tonic=0.1, ca_basal=0.1):
        m = DARPP32PP1Module(da_tonic_occupancy=da_tonic, ca_basal_uM=ca_basal)
        for _ in range(int(seconds / dt)):
            s = m.step(dt, da_occ, ca_uM)
        return s

    print("=" * 74)
    print("DARPP-32 / PP1 MODULE — does the LTP/LTD sign emerge from grounded biology?")
    print("=" * 74)
    print("  pp1_factor < 1  ⇒  PP1 inhibited  ⇒  CaMKII k_dephos ↓  ⇒  LTP")
    print("  pp1_factor > 1  ⇒  PP1 active     ⇒  CaMKII k_dephos ↑  ⇒  LTD\n")

    tonic   = run(0.1, 0.1)    # baseline: DA tonic, basal Ca — normalization point
    burst   = run(0.9, 2.0)    # reward burst + strong Ca (glutamate coincidence) → expect LTP
    dip     = run(0.01, 2.0)   # dopamine DIP + strong Ca → expect LTD (no PKA to inhibit PP1)
    weak_ca = run(0.9, 0.3)    # DA burst but WEAK Ca → Cdk5/Thr75 dominant → LTD-leaning
    strong  = run(0.9, 3.0)    # DA burst + strong Ca → PP2A clears Thr75 → strong LTP

    for name, s in [("tonic (ref)", tonic), ("BURST+strongCa", burst), ("DIP+strongCa", dip),
                    ("burst+weakCa", weak_ca), ("burst+strongerCa", strong)]:
        print(f"  {name:18s} thr34={s['thr34']:.3f} thr75={s['thr75']:.3f} "
              f"pka={s['pka']:.3f} pp1_factor={s['pp1_factor']:.3f}")

    print("\n=== ACCEPTANCE (the sign must EMERGE, and in the grounded direction/ordering) ===")
    # NOTE (grounded, cited): the resting striatal state is PKA-suppressed / PP1-ACTIVE — DARPP-32-Thr75-P
    # (Cdk5) holds PKA off until dopamine arrives (Svenningsson/Greengard; Nishi). So the LTP/LTD range is
    # ASYMMETRIC by biology: dopamine drives strong LTP (large PP1 inhibition) from a resting point where PP1
    # is already active, so the dip→LTD headroom is inherently SMALL. We test the emergent DIRECTION and
    # ordering (burst<tonic<dip; strong-Ca more LTP than weak-Ca), not a symmetric magnitude.
    checks = [
        ("tonic normalizes to ~1.0",                       abs(tonic['pp1_factor'] - 1.0) < 0.05),
        ("BURST → PP1 inhibited (LTP), burst < tonic",     burst['pp1_factor'] < 0.8),
        ("DIP → PP1 more active than tonic (LTD direction)", dip['pp1_factor'] > tonic['pp1_factor'] + 0.03),
        ("ordering burst < tonic < dip (bidirectional)",   burst['pp1_factor'] < tonic['pp1_factor'] < dip['pp1_factor']),
        ("Ca-amplitude switch: weak-Ca less LTP than strong-Ca", weak_ca['pp1_factor'] > strong['pp1_factor']),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  LTD/LTP asymmetry (grounded, reported not gated): dip {dip['pp1_factor']:.3f}× vs "
          f"burst {burst['pp1_factor']:.3f}× — resting PP1 is already active, so LTD headroom is small.")
    print(f"  ALL: {'PASS — LTP/LTD direction EMERGES from DARPP-32/PP1 (grounded), not an imposed sign' if all(o for _, o in checks) else 'FAIL — see above'}")
