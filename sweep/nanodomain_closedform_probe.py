#!/usr/bin/env python3
"""
Closed-Form Nanodomain Reference Probe
=======================================
Pure-algebra steady-state buffered point-source nanodomain.
NO PDE, no grid, no time-stepping.

Formula (Naraghi & Neher 1997 / Neher 1998):
  [Ca](r) = ca_baseline + i_ch / (z·F·4π·D_ca·r) · exp(-r/λ)

where:
  λ = sqrt(D_ca / (k_on · [B]_free))

k_on pinned from literature for the dominant endogenous fast buffer
(calbindin-D28k), with sensitivity bracketing.

Reports:
  1. Single open channel: peak [Ca] at r = 1, 2, 4, 10, 20, 30 nm
  2. Full 50-channel cluster at -70/-55/-40 mV (ensemble-average)
  3. Side-by-side with production's 0.5 µM/channel + 190 nm λ

Usage:
  python sweep/nanodomain_closedform_probe.py
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. CONSTANTS — all from model6_parameters.py (no imports needed)
# ═══════════════════════════════════════════════════════════════════

# Electrochemistry
z       = 2              # Ca²⁺ valence
F       = 96485.0        # Faraday constant  (C/mol)

# Calcium dynamics (CalciumParameters)
D_ca        = 220e-12    # m²/s  free calcium diffusion  (Allbritton 1992)
ca_baseline = 100e-9     # M     resting [Ca²⁺]          (Helmchen 1996)
B_total     = 300e-6     # M     total buffer             (Neher & Augustine 1992)
buffer_kd   = 10e-6      # M     buffer Kd
kappa_s     = 60         # dimensionless buffer capacity   (Neher & Augustine 1992)
i_ch_ref    = 0.3e-12    # A     single channel current   (Borst & Sakmann 1996)

# Pump parameters (for production λ)
pump_vmax = 50e-6        # M/s   (Scheuss 2006)
pump_km   = 0.5e-6       # M

# Channel model
N_CHANNELS     = 50
NMDA_FRACTION  = 0.5
MG_CONC_MM     = 1.0     # mM extracellular [Mg²⁺]
ALPHA          = 1e3      # s⁻¹  opening rate
BETA           = 2e3      # s⁻¹  closing rate
V_HALF         = -0.020   # V    VGCC Boltzmann midpoint
V_SLOPE        = 0.006    # V    VGCC Boltzmann slope

# Geometry (matches Mode B/C)
SEED       = 42
GRID_SIZE  = 100
AZ_RADIUS  = 200e-9      # m
dx         = 2 * AZ_RADIUS / GRID_SIZE   # 4 nm

# Production analytical model constants
CA_PER_CH_PROD = 0.5e-6  # M  (calibrated amplitude)
D_eff_prod     = D_ca / (1 + kappa_s)
k_pump         = pump_vmax / pump_km
LAMBDA_PROD    = np.sqrt(D_eff_prod / max(k_pump, 1))   # ~190 nm


# ═══════════════════════════════════════════════════════════════════
# 2. k_on PINNING  &  λ SENSITIVITY
# ═══════════════════════════════════════════════════════════════════
#
# Literature source for the endogenous fast buffer (calbindin-D28k):
#
#   Nägerl et al. 2000, J Physiol 529:625-636
#   "Binding kinetics of calbindin-D28k determined by flash photolysis
#    of caged Ca²⁺"
#   k_on = 2.7 × 10⁷ M⁻¹s⁻¹  (apparent per-molecule rate,
#   calbindin-D28k in cerebellar Purkinje neurons)
#
# Consistent with:
#   Naraghi & Neher 1997, J Neurosci 17:6961 — endogenous buffer
#   modelling framework that this formula derives from.
#
# Calmodulin fast (N-lobe) sites for comparison:
#   Faas et al. 2011, Nat Neurosci 14:301: k_on ≈ 5 × 10⁸ M⁻¹s⁻¹
#   (much faster, but calmodulin is at lower free [B] and partially
#    Ca²⁺-saturated at rest)

K_ON_REF = 2.7e7         # M⁻¹s⁻¹  calbindin-D28k  (Nägerl et al. 2000)
B_FREE   = B_total       # ≈ B_total at rest ([Ca]_rest << Kd)

def lambda_buf(k_on):
    """Buffer-limited decay length λ = sqrt(D_ca / (k_on · [B]_free))."""
    return np.sqrt(D_ca / (k_on * B_FREE))

LAMBDA_REF = lambda_buf(K_ON_REF)

# Sensitivity bracket
K_ON_BRACKET = [1.0e7, 2.7e7, 5.5e7, 1.0e8, 5.0e8]
BRACKET_LABELS = [
    "parvalbumin-like (slow)",
    "calbindin-D28k  (Nägerl 2000, PINNED)",
    "calbindin fast sites (Bhatt 2000)",
    "Naraghi-Neher 1997 endog.",
    "calmodulin N-lobe (Faas 2011)",
]


# ═══════════════════════════════════════════════════════════════════
# 3. PHYSICS FORMULA  &  PRODUCTION FORMULA
# ═══════════════════════════════════════════════════════════════════

def ca_physics(r_m, i_ch, lam):
    """Naraghi-Neher buffered point-source:
    [Ca](r) = ca_baseline + i_ch / (z·F·4π·D_ca·r) · exp(-r/λ)

    i/(z·F·4π·D·r) yields mol/m³;  ×1e-3 converts to M (mol/L).
    """
    r = np.maximum(r_m, 1e-12)  # guard div-by-zero
    amp = i_ch / (z * F * 4 * np.pi * D_ca * r) * 1e-3   # mol/m³ → M
    return ca_baseline + amp * np.exp(-r / lam)

def ca_production(r_m, i_ch, r_min=dx):
    """Production analytical model (AnalyticalNanodomainCalculator):
    ca_per_channel × (i_ch / 0.3 pA) × exp(-r / λ_prod)
    """
    r = np.maximum(r_m, r_min)
    return ca_baseline + CA_PER_CH_PROD * (i_ch / i_ch_ref) * np.exp(-r / LAMBDA_PROD)


# ═══════════════════════════════════════════════════════════════════
# 4. CHANNEL GEOMETRY — identical to Mode B/C
# ═══════════════════════════════════════════════════════════════════

np.random.seed(SEED)
center = GRID_SIZE // 2
channel_positions_grid = np.array([
    [center + np.random.randint(-2, 3),
     center + np.random.randint(-2, 3)]
    for _ in range(N_CHANNELS)
])
# Physical positions (m)
channel_positions_m = channel_positions_grid * dx

# Channel type assignment (first NMDA_FRACTION are NMDAR)
n_nmda = int(round(NMDA_FRACTION * N_CHANNELS))
is_nmda = np.zeros(N_CHANNELS, dtype=bool)
is_nmda[:n_nmda] = True


# ═══════════════════════════════════════════════════════════════════
# 5. STEADY-STATE GATING  (deterministic ensemble average)
# ═══════════════════════════════════════════════════════════════════

def vgcc_P_open_v(V):
    """VGCC Boltzmann voltage factor."""
    return 1.0 / (1.0 + np.exp(-(V - V_HALF) / V_SLOPE))

def vgcc_P_open(V):
    """VGCC steady-state open probability (from production gating model)."""
    Pv = vgcc_P_open_v(V)
    alpha_eff = ALPHA * Pv
    beta_eff  = BETA * (1.0 - Pv)
    return alpha_eff / (alpha_eff + beta_eff)

def nmdar_P_open(glutamate=1.0):
    """NMDAR steady-state open probability (saturating glutamate)."""
    return ALPHA * glutamate / (ALPHA * glutamate + BETA)   # = 1/3 for glut=1

def jahr_stevens_B(V):
    """Jahr & Stevens 1990 Mg-unblock: B(V) = 1/(1 + exp(-0.062·V_mV + 1.2726)·[Mg])."""
    V_mV = V * 1e3
    return 1.0 / (1.0 + np.exp(-0.062 * V_mV + 1.2726) * MG_CONC_MM)


def channel_expected_current(ch_idx, V):
    """Expected conducting current for channel ch_idx at voltage V (A).
    = P_open × I_open.
    """
    if is_nmda[ch_idx]:
        return nmdar_P_open() * i_ch_ref * jahr_stevens_B(V)
    else:
        return vgcc_P_open(V) * i_ch_ref


# ═══════════════════════════════════════════════════════════════════
# 6. CLUSTER FIELD EVALUATION
# ═══════════════════════════════════════════════════════════════════

def cluster_field_physics(query_m, V, lam):
    """Ensemble-average [Ca] from all 50 channels at physical query point(s).

    query_m: (N, 2) array of physical positions in metres.
    Returns: (N,) array in Molarity (M).  mol/m³ → M via ×1e-3.
    """
    query_m = np.atleast_2d(query_m)
    ca = np.full(len(query_m), ca_baseline)
    for ch_i in range(N_CHANNELS):
        i_exp = channel_expected_current(ch_i, V)
        if i_exp < 1e-30:
            continue
        r = np.sqrt(np.sum((query_m - channel_positions_m[ch_i]) ** 2, axis=1))
        r = np.maximum(r, 1e-9)   # r_min = 1 nm
        amp = i_exp / (z * F * 4 * np.pi * D_ca * r) * 1e-3   # mol/m³ → M
        ca += amp * np.exp(-r / lam)
    return ca


def cluster_field_physics_rmin(query_m, V, lam, r_min_m):
    """Same as above but with explicit r_min floor."""
    query_m = np.atleast_2d(query_m)
    ca = np.full(len(query_m), ca_baseline)
    for ch_i in range(N_CHANNELS):
        i_exp = channel_expected_current(ch_i, V)
        if i_exp < 1e-30:
            continue
        r = np.sqrt(np.sum((query_m - channel_positions_m[ch_i]) ** 2, axis=1))
        r = np.maximum(r, r_min_m)
        amp = i_exp / (z * F * 4 * np.pi * D_ca * r) * 1e-3   # mol/m³ → M
        ca += amp * np.exp(-r / lam)
    return ca


def cluster_field_production(query_m, V):
    """Production model field from all 50 channels (ensemble-average)."""
    query_m = np.atleast_2d(query_m)
    ca = np.full(len(query_m), ca_baseline)
    for ch_i in range(N_CHANNELS):
        if is_nmda[ch_i]:
            p_open = nmdar_P_open()
            i_when_open = i_ch_ref * jahr_stevens_B(V)
        else:
            p_open = vgcc_P_open(V)
            i_when_open = i_ch_ref
        r = np.sqrt(np.sum((query_m - channel_positions_m[ch_i]) ** 2, axis=1))
        r = np.maximum(r, dx)     # production r_min = dx = 4 nm
        ca += p_open * CA_PER_CH_PROD * (i_when_open / i_ch_ref) * np.exp(-r / LAMBDA_PROD)
    return ca


# ═══════════════════════════════════════════════════════════════════
# 7. QUERY POINTS — same local region as Mode C
# ═══════════════════════════════════════════════════════════════════

# Template positions (first 3 channels, same as production)
template_positions_grid = channel_positions_grid[:3]
template_positions_m = template_positions_grid * dx

# Local points: circular region radius 30 nm around templates + channels
LOCAL_RADIUS_NM = 30.0
local_radius_grid = int(LOCAL_RADIUS_NM * 1e-9 / dx) + 1

local_set = set()
for pos in np.vstack([template_positions_grid, channel_positions_grid]):
    cx, cy = int(pos[0]), int(pos[1])
    for di in range(-local_radius_grid, local_radius_grid + 1):
        for dj in range(-local_radius_grid, local_radius_grid + 1):
            x, y = cx + di, cy + dj
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if di*di + dj*dj <= local_radius_grid**2:
                    local_set.add((x, y))
local_points_grid = np.array(sorted(local_set))
local_points_m = local_points_grid * dx


# ═══════════════════════════════════════════════════════════════════
#  MAIN — REPORTS
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 95)
    print("CLOSED-FORM NANODOMAIN REFERENCE PROBE")
    print("Naraghi & Neher 1997 / Neher 1998 buffered point-source")
    print("Pure algebra — no PDE, no grid, no time-stepping")
    print("=" * 95)

    # ── derived constants ──
    print()
    print("PARAMETERS (from model6_parameters.py)")
    print(f"  D_ca          = {D_ca*1e12:.0f} µm²/s")
    print(f"  z             = {z}")
    print(f"  F             = {F:.0f} C/mol")
    print(f"  ca_baseline   = {ca_baseline*1e9:.0f} nM")
    print(f"  B_total       = {B_total*1e6:.0f} µM")
    print(f"  buffer_kd     = {buffer_kd*1e6:.0f} µM")
    print(f"  κ_s           = {kappa_s}")
    print(f"  i_ch          = {i_ch_ref*1e12:.1f} pA")
    print(f"  dx (grid)     = {dx*1e9:.0f} nm")

    # ───────────────────────────────────────────────────────────────
    # k_on PINNING  +  λ SENSITIVITY TABLE
    # ───────────────────────────────────────────────────────────────
    print()
    print("-" * 95)
    print("k_on PINNING  (endogenous fast buffer: calbindin-D28k)")
    print()
    print("  PINNED VALUE:  k_on = 2.7 × 10⁷ M⁻¹s⁻¹")
    print("  Source: Nägerl et al. 2000, J Physiol 529:625-636")
    print("          \"Binding kinetics of calbindin-D28k determined by")
    print("           flash photolysis of caged Ca²⁺\"")
    print()
    print(f"  [B]_free ≈ B_total = {B_total*1e6:.0f} µM  (at rest, [Ca] << Kd)")
    print(f"  λ = sqrt(D_ca / (k_on · [B]_free))")
    print(f"  λ_ref = sqrt({D_ca*1e12:.0f}e-12 / ({K_ON_REF:.1e} × {B_FREE*1e6:.0f}e-6))")
    print(f"        = {LAMBDA_REF*1e9:.1f} nm")
    print()
    print(f"  Production λ_prod = sqrt(D_eff / k_pump)")
    print(f"                    = sqrt({D_eff_prod*1e12:.2f}e-12 / {k_pump:.0f})")
    print(f"                    = {LAMBDA_PROD*1e9:.1f} nm")
    print()

    hdr = f"  {'k_on (M⁻¹s⁻¹)':>16}  {'λ (nm)':>8}  {'label'}"
    print(hdr)
    print("  " + "-" * 80)
    for kon, label in zip(K_ON_BRACKET, BRACKET_LABELS):
        lam = lambda_buf(kon)
        marker = "  ◄──" if kon == K_ON_REF else ""
        print(f"  {kon:>16.1e}  {lam*1e9:>8.1f}  {label}{marker}")
    print()
    print(f"  Production decay length (pump-based):  λ_prod = {LAMBDA_PROD*1e9:.1f} nm")
    print(f"  LENGTH-CONSTANT GAP:  λ_prod / λ_ref = {LAMBDA_PROD/LAMBDA_REF:.2f}×")
    print()

    # ───────────────────────────────────────────────────────────────
    # SELF-TEST: single channel at r = 4 nm must give ≈ 140 µM
    # ───────────────────────────────────────────────────────────────
    selftest = ca_physics(4e-9, i_ch_ref, LAMBDA_REF) - ca_baseline
    print(f"SELF-TEST:  single channel, full open (0.3 pA), r = 4 nm")
    print(f"  [Ca] − baseline = {selftest*1e6:.1f} µM   (expect ≈ 140 µM)")
    assert 130e-6 < selftest < 150e-6, f"FAIL: got {selftest*1e6:.1f} µM"
    print(f"  PASS")
    print()

    # ───────────────────────────────────────────────────────────────
    # REPORT 1: Single open channel at specific distances
    # ───────────────────────────────────────────────────────────────
    print("=" * 95)
    print("REPORT 1:  SINGLE OPEN CHANNEL  (i = 0.3 pA, full current, no Mg-block)")
    print("           Naraghi-Neher vs Production at r = 1, 2, 4, 10, 20, 30 nm")
    print("=" * 95)

    radii_nm = [1, 2, 4, 10, 20, 30]
    radii_m  = np.array(radii_nm) * 1e-9

    # Amplitude prefactor for a single channel
    amp_prefactor = i_ch_ref / (z * F * 4 * np.pi * D_ca) * 1e-3  # mol/m³·m → M·m
    print(f"\n  Amplitude prefactor: i/(z·F·4π·D) × 1e-3 = {amp_prefactor:.4e} M·m")
    print(f"  (This × 1/r gives [Ca] contribution in M before spatial decay)\n")

    hdr1 = (f"  {'r (nm)':>7} | {'Physics [Ca]':>14} | {'Production [Ca]':>14} | "
            f"{'Ratio P/Prod':>12} | {'Phys − base':>14} | {'Prod − base':>14} | {'Amp ratio':>10}")
    print(hdr1)
    print("  " + "-" * (len(hdr1) - 2))

    for r_nm, r_m in zip(radii_nm, radii_m):
        ca_phys = ca_physics(r_m, i_ch_ref, LAMBDA_REF)
        ca_prod = ca_production(r_m, i_ch_ref, r_min=dx)

        delta_phys = ca_phys - ca_baseline
        delta_prod = ca_prod - ca_baseline

        ratio = delta_phys / delta_prod if delta_prod > 1e-15 else float('inf')

        # Format concentration intelligently
        def fmt_ca(val):
            if val >= 1e-3:
                return f"{val*1e3:>10.2f} mM"
            elif val >= 1e-6:
                return f"{val*1e6:>10.2f} µM"
            else:
                return f"{val*1e9:>10.1f} nM"

        def fmt_delta(val):
            if val >= 1e-3:
                return f"{val*1e3:>10.2f} mM"
            elif val >= 1e-6:
                return f"{val*1e6:>10.2f} µM"
            else:
                return f"{val*1e9:>10.1f} nM"

        print(f"  {r_nm:>5d}   | {fmt_ca(ca_phys)} | {fmt_ca(ca_prod)} | "
              f"{ratio:>12.0f}× | {fmt_delta(delta_phys)} | {fmt_delta(delta_prod)} | {ratio:>10.0f}×")

    # Also show at r_min = 4 nm (the production floor)
    print()
    print("  KEY COMPARISON at r_min floors:")
    r_1nm = ca_physics(1e-9, i_ch_ref, LAMBDA_REF) - ca_baseline
    r_2nm = ca_physics(2e-9, i_ch_ref, LAMBDA_REF) - ca_baseline
    r_4nm = ca_physics(4e-9, i_ch_ref, LAMBDA_REF) - ca_baseline
    prod_4nm = ca_production(4e-9, i_ch_ref) - ca_baseline
    print(f"  Physics at r_min=1 nm :  {r_1nm*1e6:.1f} µM")
    print(f"  Physics at r_min=2 nm :  {r_2nm*1e6:.1f} µM")
    print(f"  Physics at r_min=4 nm :  {r_4nm*1e6:.1f} µM")
    print(f"  Production at r_min=4 nm: {prod_4nm*1e6:.3f} µM")
    print(f"  AMPLITUDE GAP (4 nm) :  {r_4nm/prod_4nm:.0f}×")
    print()

    # ───────────────────────────────────────────────────────────────
    # REPORT 2: Full 50-channel cluster at operating-range voltages
    # ───────────────────────────────────────────────────────────────
    print("=" * 95)
    print("REPORT 2:  FULL 50-CHANNEL CLUSTER — operating-range steady states")
    print("           Ensemble-average [Ca] peak over template/local points")
    print(f"           {N_CHANNELS} channels ({n_nmda} NMDAR + {N_CHANNELS - n_nmda} VGCC)")
    print(f"           Cluster extent: ±2 grid pts = ±{2*dx*1e9:.0f} nm")
    print(f"           {len(local_points_m)} local query points (30 nm radius)")
    print("=" * 95)

    VOLTAGES_MV = [-70, -55, -40]
    R_MIN_VALUES = [1e-9, 2e-9, 4e-9]   # 1, 2, 4 nm floors

    for V_mV in VOLTAGES_MV:
        V = V_mV * 1e-3

        Pv = vgcc_P_open_v(V)
        p_vgcc = vgcc_P_open(V)
        p_nmdar = nmdar_P_open()
        B = jahr_stevens_B(V)
        n_vgcc = N_CHANNELS - n_nmda

        n_open_vgcc = n_vgcc * p_vgcc
        n_open_nmdar = n_nmda * p_nmdar
        i_per_nmdar = i_ch_ref * B
        i_per_vgcc  = i_ch_ref

        total_conducting = n_open_vgcc * i_per_vgcc + n_open_nmdar * i_per_nmdar

        print(f"\n  V = {V_mV:+d} mV")
        print(f"    VGCC : P_open_v = {Pv:.5f},  P_open = {p_vgcc:.6f},  ⟨N_open⟩ = {n_open_vgcc:.3f}")
        print(f"    NMDAR: P_open = {p_nmdar:.4f},  B(V) = {B:.5f},  ⟨N_open⟩ = {n_open_nmdar:.2f}")
        print(f"           effective conducting current/ch = {i_per_nmdar*1e15:.2f} fA")
        print(f"    Total conducting current = {total_conducting*1e15:.1f} fA")
        print()

        # Physics: peak over local points at different r_min
        print(f"    {'':6} {'Physics peak':>14}  {'at r_min':>8}  {'Production peak':>15}  {'Amp gap':>8}")
        print(f"    {'':6} {'-'*14}  {'-'*8}  {'-'*15}  {'-'*8}")

        # Production field (r_min = dx = 4 nm always)
        ca_prod_field = cluster_field_production(local_points_m, V)
        prod_peak = np.max(ca_prod_field)

        for r_min_m in R_MIN_VALUES:
            r_min_nm = r_min_m * 1e9
            ca_phys_field = cluster_field_physics_rmin(local_points_m, V, LAMBDA_REF, r_min_m)
            phys_peak = np.max(ca_phys_field)

            delta_phys = phys_peak - ca_baseline
            delta_prod = prod_peak - ca_baseline
            gap = delta_phys / delta_prod if delta_prod > 1e-15 else float('inf')

            def fmt(val):
                if val >= 1e-3:
                    return f"{val*1e3:.2f} mM"
                elif val >= 1e-6:
                    return f"{val*1e6:.4f} µM"
                else:
                    return f"{val*1e9:.1f} nM"

            print(f"    {'':6} {fmt(phys_peak):>14}  {r_min_nm:>6.0f} nm  {fmt(prod_peak):>15}  {gap:>7.0f}×")

    # ───────────────────────────────────────────────────────────────
    # REPORT 3: Side-by-side — amplitude gap + length-constant gap
    # ───────────────────────────────────────────────────────────────
    print()
    print("=" * 95)
    print("REPORT 3:  SIDE-BY-SIDE — physics vs production snapshot")
    print("           Amplitude gap + length-constant gap at each voltage")
    print("=" * 95)
    print()
    print("  The production analytical model uses:")
    print(f"    amplitude  = {CA_PER_CH_PROD*1e6:.1f} µM / channel  (empirically calibrated)")
    print(f"    λ_prod     = {LAMBDA_PROD*1e9:.1f} nm  (pump-equilibration length)")
    print(f"    r_min      = dx = {dx*1e9:.0f} nm")
    print(f"    spatial    = pure exp(-r/λ)  (NO 1/r divergence)")
    print()
    print("  The Naraghi-Neher physics formula uses:")
    print(f"    amplitude  = i/(z·F·4π·D·r)  (point-source Green's function)")
    print(f"    λ_ref      = {LAMBDA_REF*1e9:.1f} nm  (buffer-attenuation length)")
    print(f"    spatial    = (1/r) · exp(-r/λ)  (diverges as r→0)")
    print()

    hdr3 = (f"  {'V (mV)':>7} | {'Δ Phys (r=1nm)':>16} | {'Δ Phys (r=4nm)':>16} | "
            f"{'Δ Prod':>16} | {'Amp gap @4nm':>12} | {'λ ratio':>8}")
    print("  (All values = peak above baseline)")
    print(hdr3)
    print("  " + "-" * (len(hdr3) - 2))

    for V_mV in VOLTAGES_MV:
        V = V_mV * 1e-3

        # Physics peak over local points at r_min = 1 nm and 4 nm
        ca_1nm = cluster_field_physics_rmin(local_points_m, V, LAMBDA_REF, 1e-9)
        ca_4nm = cluster_field_physics_rmin(local_points_m, V, LAMBDA_REF, 4e-9)
        ca_prod = cluster_field_production(local_points_m, V)

        pk_1 = np.max(ca_1nm) - ca_baseline
        pk_4 = np.max(ca_4nm) - ca_baseline
        pk_p = np.max(ca_prod) - ca_baseline

        gap_4 = pk_4 / pk_p if pk_p > 1e-15 else float('inf')

        def fmt_uM(val):
            if val >= 1e-3:
                return f"{val*1e3:.2f} mM"
            elif val >= 1e-6:
                return f"{val*1e6:.4f} µM"
            else:
                return f"{val*1e9:.1f} nM"

        print(f"  {V_mV:>+6d}  | {fmt_uM(pk_1):>16} | {fmt_uM(pk_4):>16} | "
              f"{fmt_uM(pk_p):>16} | {gap_4:>11.0f}× | "
              f"{LAMBDA_PROD/LAMBDA_REF:>7.2f}×")

    print()
    print("  SUMMARY OF GAPS:")
    print(f"    Amplitude gap  : ~10²×  (physics 1/r point-source vs production flat 0.5 µM)")
    print(f"    Length-constant : λ_prod/λ_ref = {LAMBDA_PROD/LAMBDA_REF:.2f}×"
          f"  ({LAMBDA_PROD*1e9:.0f} nm pump-based vs {LAMBDA_REF*1e9:.0f} nm buffer-based)")
    print()
    print("  The production model's 0.5 µM/channel is calibrated to match the PDE's")
    print("  emergent nanodomain peak (which includes buffering, diffusion, extrusion).")
    print("  The physics formula gives the steady-state free [Ca²⁺] at the channel")
    print("  mouth — two orders of magnitude higher than the calibrated snapshot.")
    print("  The 190 nm decay length is set by pump kinetics (macroscopic recovery),")
    print("  not by buffer-limited attenuation (the nanodomain-relevant length scale).")
    print()
    print("Done.")
