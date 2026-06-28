"""
A2 — ACP SUPERSATURATION NUCLEATION GATE PROBE  (isolated, read-only physics)
═══════════════════════════════════════════════════════════════════════════════

Phase A2 of the calcium → dimer revalidation. Validates the amorphous-calcium-phosphate
(ACP) supersaturation nucleation gate as PURE ALGEBRA, before any live file is touched.

The gate:
    Ksp(ACP) = [Ca²⁺]³ · [PO₄³⁻]²  = 1e-26 M⁵   (pKsp ≈ 26, 310 K, pH 7.4)
    IAP      = [Ca²⁺]³ · [PO₄³⁻]²
    S        = (IAP / Ksp) ** (1/5)             (v = 5 ions per formula unit)
    nucleation thermodynamically allowed only for S > 1.

Two reused groundings (see GROUNDING NOTES at bottom for provenance + line refs):
  (b) closed-form nanodomain calcium  — formula replicated verbatim from
      sweep/nanodomain_closedform_probe.py:115-123 (ca_physics); self-test asserted.
  (c) free trivalent [PO₄³⁻]          — Henderson-Hasselbalch over the STRUCTURAL pool,
      replicated from src/models/Model_6/atp_system.py:382-401, with the model's actual
      params (model6_parameters.py): pKa=2.1/7.2/12.4, structural=1mM, pH_rest=7.35.

DISCIPLINE (locked): emergent physics only. No constant is tuned to make the bare/template
split come out selective. If it is not selective, the probe says so and shows the gap.

Run:  python sweep/supersaturation_gate_probe.py
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# (b) CLOSED-FORM NANODOMAIN CALCIUM  — replicated from nanodomain_closedform_probe.py
# ═══════════════════════════════════════════════════════════════════════════════
z           = 2              # Ca²⁺ valence
F           = 96485.0        # C/mol  Faraday
D_ca        = 220e-12        # m²/s   free Ca diffusion        (Allbritton 1992)
ca_baseline = 100e-9         # M      resting [Ca²⁺]           (Helmchen 1996)
i_ch_ref    = 0.3e-12        # A      single-channel current   (Borst & Sakmann 1996)

# λ (buffer length scale). The A1 probe pins B_FREE = B_total = 300 µM with the
# calbindin-D28k k_on (Nägerl 2000) → λ ≈ 165 nm, which is the value that reproduces
# the 137.2 µM @4 nm self-test. The handoff's λ ≈ 117 nm uses 600 µM binding sites
# (κ_s interpretation). At the 4 nm read radius the two differ by ~1% (exp(-r/λ)≈1),
# so the near-mouth PEAK — the only thing this gate reads — is robust to the choice.
B_FREE   = 300e-6            # M
K_ON_REF = 2.7e7            # M⁻¹s⁻¹  calbindin-D28k (Nägerl 2000)
LAMBDA_165 = np.sqrt(D_ca / (K_ON_REF * B_FREE))        # ≈ 165 nm (A1 self-test value)
LAMBDA_117 = np.sqrt(D_ca / (K_ON_REF * 600e-6))        # ≈ 117 nm (handoff)


def ca_physics(r_m, i_ch, lam):
    """Naraghi-Neher buffered point-source. [Ca](r) in M. (verbatim formula)"""
    r = np.maximum(r_m, 1e-12)
    amp = i_ch / (z * F * 4 * np.pi * D_ca * r) * 1e-3   # mol/m³ → M
    return ca_baseline + amp * np.exp(-r / lam)


def ca_co_located(n_channels, r_m, i_ch=i_ch_ref, lam=LAMBDA_165):
    """n identical channels co-located at the same read radius r: the excess over
    baseline sums linearly (superposition of point sources at equal distance)."""
    excess = ca_physics(r_m, i_ch, lam) - ca_baseline
    return ca_baseline + n_channels * excess


# ── self-test (b): single 0.3 pA channel at 4 nm must reproduce 137.2 µM ──────────
_ca_4nm = ca_physics(4e-9, i_ch_ref, LAMBDA_165)
assert abs(_ca_4nm * 1e6 - 137.2) < 0.3, f"closed-form self-test FAILED: {_ca_4nm*1e6:.2f} µM"


# ═══════════════════════════════════════════════════════════════════════════════
# (c) FREE TRIVALENT [PO₄³⁻]  — replicated from atp_system.py update_speciation
# ═══════════════════════════════════════════════════════════════════════════════
pKa1, pKa2, pKa3   = 2.1, 7.2, 12.4        # model6_parameters.py:209-211
PHOSPHATE_STRUCT   = 1e-3                   # M, model6_parameters.py:193 (phosphate_total)
pH_REST            = 7.35                   # model6_parameters.py:824 (pH_rest)
pH_ACTIVE          = 6.8                    # model6_parameters.py:827 (pH_active_min)


def po4_3minus(pH, P_struct=PHOSPHATE_STRUCT):
    """[PO₄³⁻] = α₃ · structural phosphate, via full triprotic speciation."""
    H = 10 ** (-pH)
    Ka1, Ka2, Ka3 = 10 ** -pKa1, 10 ** -pKa2, 10 ** -pKa3
    denom = H**3 + H**2 * Ka1 + H * Ka1 * Ka2 + Ka1 * Ka2 * Ka3
    alpha3 = (Ka1 * Ka2 * Ka3) / denom
    return alpha3 * P_struct, alpha3


PO4_REST, ALPHA3_REST = po4_3minus(pH_REST)        # the model's actual resting value
PO4_pH74, _           = po4_3minus(7.4)            # the gate-literature pH 7.4
PO4_ACTIVE, _         = po4_3minus(pH_ACTIVE)      # burst pH — PO4 drops further


# ═══════════════════════════════════════════════════════════════════════════════
# THE GATE
# ═══════════════════════════════════════════════════════════════════════════════
KSP = 1e-26     # M⁵  amorphous calcium phosphate (pKsp ≈ 26, 310 K, pH 7.4)
NU  = 5         # ions per formula unit


def supersaturation(ca_M, po4_M):
    iap = ca_M**3 * po4_M**2
    return (iap / KSP) ** (1.0 / NU)


def threshold_ca(po4_M):
    """Solve S = 1 for [Ca]:  [Ca] = (Ksp / [PO₄³⁻]²) ** (1/3)."""
    return (KSP / po4_M**2) ** (1.0 / 3.0)


# ── self-tests (the §8 asserts) ───────────────────────────────────────────────────
assert supersaturation(ca_baseline, PO4_REST) < 1.0, "rest must be undersaturated"
assert supersaturation(0.5e-6, PO4_REST) < 1.0, "0.5 µM calibration must be undersaturated"


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    PO4 = PO4_REST   # use the model's actual [PO₄³⁻] (resting pH 7.35) for the table

    print("=" * 78)
    print("A2 — ACP SUPERSATURATION NUCLEATION GATE PROBE")
    print("=" * 78)
    print(f"\nGROUNDED INPUTS (from code, not estimates):")
    print(f"  λ (near-mouth)        = {LAMBDA_165*1e9:.0f} nm  (A1 self-test value; "
          f"handoff 117 nm differs <1% at 4 nm)")
    print(f"  closed-form self-test = {_ca_4nm*1e6:.1f} µM  @ 4 nm, single 0.3 pA channel  [PASS]")
    print(f"  pKa1/2/3              = {pKa1}/{pKa2}/{pKa3}")
    print(f"  structural phosphate  = {PHOSPHATE_STRUCT*1e3:.1f} mM")
    print(f"  α₃ @ pH {pH_REST}        = {ALPHA3_REST:.3e}")
    print(f"  [PO₄³⁻] @ rest pH7.35 = {PO4_REST*1e9:.2f} nM   (handoff estimated ~10 nM — "
          f"code is ~2× lower)")
    print(f"  [PO₄³⁻] @ pH 7.40     = {PO4_pH74*1e9:.2f} nM")
    print(f"  [PO₄³⁻] @ burst pH6.8 = {PO4_ACTIVE*1e9:.2f} nM   (drops further during activity)")
    print(f"  Ksp                   = {KSP:.0e} M⁵   (pKsp ≈ 26)")

    # ── scenario table ────────────────────────────────────────────────────────────
    print(f"\n{'-'*78}")
    print(f"SCENARIO TABLE  (all use the model's [PO₄³⁻] = {PO4*1e9:.2f} nM @ rest pH)")
    print(f"{'-'*78}")
    hdr = f"{'scenario':<34}{'[Ca] µM':>10}{'S':>12}{'nucleates?':>12}"
    print(hdr); print("-" * 78)

    scenarios = [
        ("1. rest (100 nM)",                 ca_baseline),
        ("2. old calibration (0.5 µM)",      0.5e-6),
        ("3. bare 1 channel @4 nm",          ca_co_located(1, 4e-9)),
        ("4. template 3 channels @4 nm",     ca_co_located(3, 4e-9)),
        ("5. template 6 channels @4 nm",     ca_co_located(6, 4e-9)),
    ]
    for name, ca in scenarios:
        S = supersaturation(ca, PO4)
        print(f"{name:<34}{ca*1e6:>10.3f}{S:>12.3e}{('YES' if S>1 else 'no'):>12}")

    # ── r × n sweep ───────────────────────────────────────────────────────────────
    print(f"\n{'-'*78}")
    print("SWEEP:  read radius r  ×  n co-located channels   (S value; * = nucleates)")
    print(f"{'-'*78}")
    radii_nm = [4, 8, 16, 30, 50]
    n_list   = [1, 3, 6]
    print(f"{'r (nm)':>8}" + "".join(f"{f'n={n}':>16}" for n in n_list))
    for r_nm in radii_nm:
        row = f"{r_nm:>8}"
        for n in n_list:
            ca = ca_co_located(n, r_nm * 1e-9)
            S = supersaturation(ca, PO4)
            mark = "*" if S > 1 else " "
            cell = f"{S:.3f}{mark}"
            row += f"{cell:>16}"
        print(row)

    # ── threshold + selectivity verdict ───────────────────────────────────────────
    print(f"\n{'-'*78}")
    print("THRESHOLD & SELECTIVITY")
    print(f"{'-'*78}")
    thr_rest = threshold_ca(PO4_REST)
    thr_74   = threshold_ca(PO4_pH74)
    print(f"  [Ca] at S=1, model [PO₄³⁻] @pH7.35 = {thr_rest*1e6:.0f} µM")
    print(f"  [Ca] at S=1,        [PO₄³⁻] @pH7.40 = {thr_74*1e6:.0f} µM")
    ca_bare   = ca_co_located(1, 4e-9)
    ca_t3     = ca_co_located(3, 4e-9)
    ca_t6     = ca_co_located(6, 4e-9)
    print(f"\n  bare 1-ch @4nm  = {ca_bare*1e6:6.1f} µM  → S={supersaturation(ca_bare,PO4):.3f}  "
          f"{'nucleates' if supersaturation(ca_bare,PO4)>1 else 'BELOW threshold'}")
    print(f"  template 3-ch   = {ca_t3*1e6:6.1f} µM  → S={supersaturation(ca_t3,PO4):.3f}  "
          f"{'nucleates' if supersaturation(ca_t3,PO4)>1 else 'BELOW threshold'}")
    print(f"  template 6-ch   = {ca_t6*1e6:6.1f} µM  → S={supersaturation(ca_t6,PO4):.3f}  "
          f"{'nucleates' if supersaturation(ca_t6,PO4)>1 else 'BELOW threshold'}")

    bare_S, t6_S = supersaturation(ca_bare, PO4), supersaturation(ca_t6, PO4)
    print(f"\n  VERDICT: at the 4 nm grid-floor read radius, the bare single channel sits")
    if bare_S < 1 < t6_S:
        print(f"  BELOW S=1 (S={bare_S:.2f}) while the 6-channel template sits ABOVE (S={t6_S:.2f}).")
        print(f"  → The gate + multi-channel geometry IS selective on its own — and sharper")
        print(f"    than the handoff anticipated: at the code's lower [PO₄³⁻], NUCLEATION")
        print(f"    REQUIRES A CHANNEL CLUSTER. A single bare channel crosses S=1 only at")
        print(f"    r≲0.7 nm (sub-physical contact distance) — so it effectively never")
        print(f"    nucleates. At the 4 nm grid floor it takes ≥6 co-located channels;")
        print(f"    a 3-channel template (the model's channel_positions[:3]) needs r≲2 nm.")
    elif bare_S > 1:
        print(f"  ABOVE S=1 (S={bare_S:.2f}) — even a bare single channel nucleates at 4 nm.")
    else:
        print(f"  NOT selective in the expected direction — bare S={bare_S:.2f}, 6-ch S={t6_S:.2f}.")

    # ── 3.9 kT heterogeneous-catalysis note (rate factor; NOT computed into S) ─────
    print(f"\n{'-'*78}")
    print("TEMPLATE HETEROGENEOUS CATALYSIS (note, not applied to the gate)")
    print(f"{'-'*78}")
    ddg_kT = np.log(50)
    print(f"  The 50× template enhancement = nucleation-barrier reduction ΔΔG = ln(50)·kT")
    print(f"  ≈ {ddg_kT:.2f} kT, i.e. a RATE factor (rate ∝ exp(-ΔG/kT)) applied ABOVE threshold.")
    print(f"  This is what would lift the sub-threshold 3-channel case (S={supersaturation(ca_t3,PO4):.2f})")
    print(f"  into formation. Mapping 3.9 kT into an effective critical-S SHIFT needs the ACP")
    print(f"  interfacial energy — a deferred pin; flagged, NOT computed here.")

    print(f"\n{'='*78}")
    print("All asserts passed (self-test 137.2 µM; rest & 0.5 µM undersaturated). "
          "No live code touched.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
