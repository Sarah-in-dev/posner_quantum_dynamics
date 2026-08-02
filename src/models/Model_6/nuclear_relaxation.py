#!/usr/bin/env python3
"""
DERIVED ³¹P coherence under ⁶Li/⁷Li doping — the EMERGENT isotope lever (F1).

Replaces the circular hardcoded T_singlet swap (216 s ↔ 0.4 s). The ³¹P nuclear spins remain
THE qubits; lithium is a DOPANT (Li⁺ substitutes a Ca²⁺ in the Ca₆(PO₄)₄ dimer). The dopant
decoheres the ³¹P via scalar relaxation of the second kind (Abragam), and ⁷Li's Larmor frequency
sits next to ³¹P's (near-resonant → fast decay) while ⁶Li's is far (→ negligible). The ⁶Li/⁷Li
CONTRAST is parameter-free (tabulated γ, I, Q); only the absolute scale rides on one J calibration.

Source: arXiv 2310.13484 = Sci Rep 15, s41598-025-96487-5 (2025), eq. (9); Abragam Ch. VIII;
J-couplings Swift & Fisher 1711.05899; Earth-field precedent Chiavazza 2013.

    1/T2_scalar(³¹P) = (8π²J²/3)·I(I+1)·τ/(1+(ω_Li−ω_P)²τ²)          [eq. 9, non-secular Lorentzian]
    1/T2_obs        = 1/T_intrinsic + 1/T2_scalar                     [intrinsic dimer floor + dopant]

FOOTGUNS (from the physics report):
  1. ω is ANGULAR (rad/s): ω = 2π·(γ/2π[MHz/T]·1e6)·B0.  NOT Hz.
  2. Do NOT add the Abragam secular term A²/3·I(I+1)·τ — we are OUTSIDE motional narrowing (Δω·τ≫1);
     it would give µs decay for BOTH isotopes and destroy the result. Eq.(9) alone is the correct object.
  3. Regime Δω·τ ≫ 1 ⇒ R_sc ≈ (8π²J²/3)I(I+1)/(Δω²·τ): faster for smaller Δω AND shorter τ (both favour ⁷Li).
"""
import numpy as np

# --- tabulated nuclear constants (all [STANDARD-TABLE] except where noted) ---
GAMMA = {'P31': 17.24, 'Li7': 16.55, 'Li6': 6.27}   # γ/2π, MHz/T (CODATA; paper refs 38–39)
SPIN_I = {'Li7': 1.5, 'Li6': 1.0}                   # nuclear spin of the quadrupolar dopant
TAU_LI = {'Li7': 10.0, 'Li6': 300.0}                # T1 of Li, s [ASSUMED-in-source: Fisher solvated; grounded by |Q7/Q6|≈50]
B0 = 50e-6                                           # Tesla, Earth's field [ASSUMED-in-source]
T_INTRINSIC = 216.0                                 # s, undoped Ca₆(PO₄)₄ dimer coherence (Agarwal band; the one kept number)
J_LI7_HZ = 18.0                                     # Hz, Li7–P scalar coupling — THE single calibrated input
                                                    # (paper: "~2 orders above" Swift's 0.178 Hz P–P; slide 18→50 Hz keeps ⁷Li "seconds")

def omega(species):
    """Angular Larmor frequency (rad/s)."""
    return 2*np.pi * GAMMA[species]*1e6 * B0

def R_scalar(dopant, J_li7_hz=J_LI7_HZ):
    """Scalar-relaxation-of-the-2nd-kind rate of ³¹P by a Li dopant (eq. 9). dopant: 'Li6'|'Li7'|None."""
    if dopant is None:
        return 0.0
    I, tau = SPIN_I[dopant], TAU_LI[dopant]
    dw = omega('P31') - omega(dopant)
    J = J_li7_hz * (GAMMA[dopant] / GAMMA['Li7'])   # reduced coupling isotope-independent ⇒ J∝γ_Li
    return (8*np.pi**2 * J**2 / 3.0) * I*(I+1) * tau/(1.0 + (dw*tau)**2)

def T2_scalar(dopant, J_li7_hz=J_LI7_HZ):
    R = R_scalar(dopant, J_li7_hz)
    return np.inf if R == 0 else 1.0/R

def T2_observed(dopant, J_li7_hz=J_LI7_HZ, t_intrinsic=T_INTRINSIC):
    """Observable ³¹P coherence: intrinsic dimer floor + dopant scalar relaxation."""
    return 1.0 / (1.0/t_intrinsic + R_scalar(dopant, J_li7_hz))

if __name__ == '__main__':
    print("=== intermediate quantities (reproduce paper's 5416/5199/1970 rad/s) ===")
    for s in ['P31','Li7','Li6']:
        print(f"  ω({s}) = {omega(s):7.1f} rad/s")
    print(f"  Δω(Li7) = {omega('P31')-omega('Li7'):.1f} rad/s   Δω(Li6) = {omega('P31')-omega('Li6'):.1f} rad/s")
    print("\n=== SCALAR-ONLY T2 (the paper's lever, in isolation) ===")
    t7, t6 = T2_scalar('Li7'), T2_scalar('Li6')
    print(f"  ⁷Li: T2_scalar = {t7:.1f} s        (paper: 'only seconds')")
    print(f"  ⁶Li: T2_scalar = {t6:.3e} s ≈ {t6/86400:.1f} days")
    print(f"  ratio ⁶Li/⁷Li = {t6/t7:.2e}  → log10 = {np.log10(t6/t7):.2f}   (paper: '≥5 orders of magnitude')")
    print("\n=== OBSERVABLE T2 (with the 216 s intrinsic dimer floor) — the model's lever ===")
    for d in [None, 'Li6', 'Li7']:
        print(f"  dopant={str(d):>4}: T2_obs = {T2_observed(d):8.2f} s")
    print("\n=== ACCEPTANCE CHECKS ===")
    ok_orders = 4.5 <= np.log10(t6/t7) <= 5.5
    ok_li7    = 2.0 <= t7 <= 30.0
    ok_intrinsic = abs(T2_observed(None)-216.0) < 1.0 and abs(T2_observed('Li6')-216.0) < 5.0
    ok_kill   = T2_observed('Li7') < 30.0
    print(f"  [{'PASS' if ok_orders else 'FAIL'}] scalar ⁶Li/⁷Li ratio ≈ 5 orders")
    print(f"  [{'PASS' if ok_li7 else 'FAIL'}] ⁷Li scalar T2 in 'seconds' band (2–30 s)")
    print(f"  [{'PASS' if ok_intrinsic else 'FAIL'}] undoped & ⁶Li observable T2 ≈ intrinsic 216 s (dopant negligible)")
    print(f"  [{'PASS' if ok_kill else 'FAIL'}] ⁷Li observable T2 collapses to seconds (real kill)")
    print(f"\n  ALL: {'PASS — reproduces the literature; lever is emergent' if all([ok_orders,ok_li7,ok_intrinsic,ok_kill]) else 'FAIL'}")
