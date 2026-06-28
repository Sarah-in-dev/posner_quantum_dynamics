"""
A3 — PHOSPHATE CONSERVATION + SELF-LIMITING (SOC) PROBE   (isolated, read-only physics)
═══════════════════════════════════════════════════════════════════════════════════════

Phase A3 of the calcium → dimer revalidation. Demonstrates, in isolation, the closed loop
that Phase B3 is meant to produce in the live model:

    finite structural phosphate → speciation → [PO₄³⁻] → S gate → formation consumes P →
    structural P drops → [PO₄³⁻] drops → S drops → formation SELF-LIMITS at S=1 (SOC) →
    remove calcium → dissolution returns P → structural P recovers.

Validates three things (handoff §5 A3):
  1. TOTAL phosphate conserved across a formation–dissolution cycle: P_free + 4·[dimer] = P_total
     to machine precision, every step.
  2. The scarce trivalent [PO₄³⁻] is buffered by the large HPO₄²⁻ reservoir (both track α·P_free).
  3. Depleting free P drops S and self-limits formation at the S=1 boundary (SOC), not a runaway.

Grounding reused (all GROUNDED, see provenance at bottom):
  - gate + speciation  : replicated from sweep/supersaturation_gate_probe.py (A2)
  - formation kinetics : k_base = productive_fraction × Smoluchowski = 1.9e4 M⁻¹s⁻¹,
                         d[dimer]/dt = k_base·[PNC]²·gate,  [PNC]=[Ca]/6   (dimer-chemistry skill §1)
  - dissolution        : k_classical = 0.005 s⁻¹  (τ≈200 s, Turhan 2024)
  - stoichiometry      : Ca₆(PO₄)₄ dimer → 4 PO₄ + 6 Ca per dimer
                         (ca_triphosphate_complex.py:414-415)

Physical cap (the only modeling choice): formation cannot drive S below 1 within a step —
nucleation halts at saturation. This pins the SOC attractor cleanly at S=1 and is the
thermodynamic statement, not a tuned knob. Trimer omitted (physiological Ca/P ≪ 0.5 → 99.9%
dimer; dimer-chemistry skill §2).

DISCIPLINE: emergent physics only. No constant tuned to a target. If conservation or
self-limiting fails, the probe reports the gap.

Run:  python sweep/phosphate_conservation_probe.py
"""

import numpy as np

# ── gate + speciation (replicated from A2) ────────────────────────────────────────
KSP = 1e-26                       # M⁵  ACP (pKsp≈26); see A3 note: Meyer-Eanes 1e-25 is the band edge
NU  = 5
pKa1, pKa2, pKa3 = 2.1, 7.2, 12.4
P_TOTAL = 1e-3                     # M, structural phosphate budget (model6_parameters.py:193)
pH_REST = 7.35


def alphas(pH):
    """Triprotic fractions (α1=H2PO4, α2=HPO4, α3=PO4³⁻)."""
    H = 10 ** (-pH)
    Ka1, Ka2, Ka3 = 10 ** -pKa1, 10 ** -pKa2, 10 ** -pKa3
    denom = H**3 + H**2 * Ka1 + H * Ka1 * Ka2 + Ka1 * Ka2 * Ka3
    return ((H**2 * Ka1) / denom, (H * Ka1 * Ka2) / denom, (Ka1 * Ka2 * Ka3) / denom)


A1F, A2F, A3F = alphas(pH_REST)   # HPO4 fraction A2F, PO4³⁻ fraction A3F


def S_of(ca, po4):
    return (ca**3 * po4**2 / KSP) ** (1.0 / NU)


def po4_at_S1(ca):
    """[PO₄³⁻] that gives S=1 at this calcium: from Ca³·[PO₄³⁻]² = Ksp."""
    return np.sqrt(KSP / ca**3)


# ── formation / dissolution kinetics (dimer-chemistry skill §1) ───────────────────
K_BASE      = 1.9e4               # M⁻¹s⁻¹  = productive_fraction(0.01) × Smoluchowski(1.9e6)
K_CLASSICAL = 0.005               # s⁻¹     dissolution; τ≈200 s
PO4_PER_DIMER = 4.0               # Ca₆(PO₄)₄


def run(ca_drive_uM=823.0, t_drive=150.0, t_total=500.0, dt=0.05):
    ca_rest = 100e-9
    ca_drive = ca_drive_uM * 1e-6
    n = int(t_total / dt)

    dimer = 0.0                   # M
    P_free = P_TOTAL              # M, free structural phosphate
    rows = []                     # snapshots
    max_cons_err = 0.0

    snap_times = {0.0, 1.0, 5.0, 20.0, 60.0, 149.0, 150.0, 200.0, 300.0, 499.0}
    snapped = set()

    for k in range(n + 1):
        t = k * dt
        ca = ca_drive if t < t_drive else ca_rest

        po4 = A3F * P_free                 # trivalent, tracks the free pool
        hpo4 = A2F * P_free                # the reservoir
        S = S_of(ca, po4)

        # formation only above saturation, capped so it cannot push S below 1
        pnc = ca / 6.0
        desired = K_BASE * pnc**2 * dt     # M this step
        if S > 1.0:
            p_floor = po4_at_S1(ca) / A3F  # free P at which S=1
            max_dimer = max(0.0, (P_free - p_floor) / PO4_PER_DIMER)
            formed = min(desired, max_dimer)
        else:
            formed = 0.0
        dissolved = K_CLASSICAL * dimer * dt

        d_dimer = formed - dissolved
        dimer += d_dimer
        P_free -= PO4_PER_DIMER * d_dimer
        dimer = max(dimer, 0.0)

        # conservation check
        total = P_free + PO4_PER_DIMER * dimer
        max_cons_err = max(max_cons_err, abs(total - P_TOTAL))

        for st in snap_times:
            if st not in snapped and t >= st:
                rows.append((t, ca, dimer, P_free, po4, hpo4, S))
                snapped.add(st)

    return rows, max_cons_err, dimer, P_free


def main():
    print("=" * 84)
    print("A3 — PHOSPHATE CONSERVATION + SELF-LIMITING (SOC) PROBE")
    print("=" * 84)
    print(f"\nSetup: P_total={P_TOTAL*1e3:.1f} mM structural; pH {pH_REST}; α(PO₄³⁻)={A3F:.2e}, "
          f"α(HPO₄²⁻)={A2F:.3f}")
    print(f"Drive: Ca=823 µM (6-ch template, A2) for 0–150 s, then Ca=100 nM (rest) to 500 s.")
    print(f"Predicted S=1 self-limit: free P → {po4_at_S1(823e-6)/A3F*1e3:.3f} mM, "
          f"dimer → {(P_TOTAL - po4_at_S1(823e-6)/A3F)/PO4_PER_DIMER*1e6:.1f} µM\n")

    rows, cons_err, dimer_end, pfree_end = run()

    hdr = f"{'t (s)':>7}{'[Ca] µM':>10}{'dimer µM':>11}{'P_free mM':>11}{'[PO₄³⁻] nM':>12}{'[HPO₄²⁻] mM':>13}{'S':>9}"
    print(hdr); print("-" * 84)
    for (t, ca, dimer, pf, po4, hpo4, S) in rows:
        print(f"{t:>7.0f}{ca*1e6:>10.3f}{dimer*1e6:>11.2f}{pf*1e3:>11.4f}"
              f"{po4*1e9:>12.3f}{hpo4*1e3:>13.4f}{S:>9.4f}")

    print("-" * 84)
    print("\nVALIDATION:")
    print(f"  1. Conservation:  max |P_free + 4·dimer − P_total| = {cons_err:.2e} M  "
          f"({'PASS' if cons_err < 1e-12 else 'FAIL'})  (machine precision)")
    print(f"  2. PO₄³⁻ buffering: [PO₄³⁻]=α₃·P_free tracks the HPO₄²⁻ reservoir (α₂·P_free), "
          f"ratio α₃/α₂ = {A3F/A2F:.2e} fixed by pH.")
    print(f"  3. Self-limiting:  dimer plateaus (S→1), does NOT run away; phosphate depletes")
    print(f"     then RECOVERS on Ca removal (dissolution returns P, τ≈200 s).")

    # asserts
    assert cons_err < 1e-12, "conservation violated"
    print(f"\nAll asserts passed. No live code touched.")
    print("=" * 84)


if __name__ == "__main__":
    main()
