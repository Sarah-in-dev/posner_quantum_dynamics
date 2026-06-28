#!/usr/bin/env python3
"""
Calcium PDE Reference Probe — steady-state voltage clamp comparison
====================================================================
Three modes:

MODE A  (original):  Each solver uses its OWN channel class.
  PDE   → all-VGCC (V_half=-30 mV, slope=12 mV)
  Ana   → 50% NMDAR + 50% VGCC (V_half=-20 mV, slope=6 mV), glutamate=1.0

MODE B  (like-for-like):  SHARED production CalciumChannels instance.
  Both solvers see identical per-step channel states & currents.
  PDE path:  flux -> CalciumDiffusion (buffering + diffusion + extrusion)
  Ana path:  AnalyticalNanodomainCalculator.calculate_field_at_points()
  The ONLY difference is spatial handling.

MODE C  (onset transfer function):  Fine voltage sweep across operating range.
  Same like-for-like setup as Mode B (shared production CalciumChannels).
  Glutamate ON/saturating (NMDAR limited only by Mg block).
  Sweep: -70 to -30 mV in 3 mV steps (operating range -70→-40, context above).
  Reports VGCC open%, NMDAR conducting%, PDE peak/mean, Ana peak per voltage.

Usage:
  python sweep/calcium_pde_reference_probe.py         # all modes
  python sweep/calcium_pde_reference_probe.py C       # Mode C only
  python sweep/calcium_pde_reference_probe.py A B     # Modes A and B only
"""

import sys, os, time
import numpy as np

# ── path setup ──────────────────────────────────────────────────────
SWEEP_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR   = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)

import logging
logging.disable(logging.INFO)

from model6_parameters import Model6Parameters, CalciumParameters

# PDE spatial components (orphaned solver)
from calcium_system import CalciumDiffusion

# Production channel model + analytical spatial solver
from analytical_calcium_system import (
    CalciumChannels as ProductionChannels,
    AnalyticalNanodomainCalculator,
    AnalyticalCalciumSystem,
)
# Original PDE full system (for Mode A)
from calcium_system import CalciumSystem as PDECalciumSystem


# ═══════════════════════════════════════════════════════════════════
# Shared geometry — identical to model6_core.py production wiring
# ═══════════════════════════════════════════════════════════════════
SEED = 42
N_CHANNELS = 50
GRID_SIZE = 100

params = Model6Parameters()
grid_shape = (GRID_SIZE, GRID_SIZE)
dx = 2 * params.spatial.active_zone_radius / GRID_SIZE   # 4 nm

np.random.seed(SEED)
center = GRID_SIZE // 2
channel_positions = np.array([
    [center + np.random.randint(-2, 3),
     center + np.random.randint(-2, 3)]
    for _ in range(N_CHANNELS)
])
template_positions = [(x, y) for x, y in channel_positions[:3]]

# Full-grid query points for analytical evaluation (same domain as PDE)
_idx = np.indices((GRID_SIZE, GRID_SIZE))
ALL_GRID_POINTS = np.column_stack([_idx[0].ravel(), _idx[1].ravel()])  # (10000, 2)

# ═══════════════════════════════════════════════════════════════════
# Sweep parameters
# ═══════════════════════════════════════════════════════════════════
VOLTAGES_MV = [-40, -30, -20, -10, 0]

DT        = 10e-6       # 10 µs per step
T_TOTAL   = 20e-3       # 20 ms
N_STEPS   = int(T_TOTAL / DT)

# Readout: average over last 5 ms
READOUT_START = int(15e-3 / DT)

N_SEEDS_A = 5           # Mode A seeds (kept short — already have results)
N_SEEDS_B = 10          # Mode B seeds (higher — isolating spatial only)
N_SEEDS_C = 10          # Mode C seeds (onset sweep, same as B)

# Mode C — fine onset sweep voltages
VOLTAGES_C_MV = [-70, -67, -64, -61, -58, -55, -52, -49, -46, -43, -40, -35, -30]

# Flux constants (same as PDE CalciumChannels.get_calcium_flux)
Z_CA = 2
FARADAY = 96485
DZ = 20e-9              # cleft width (Zuber et al. 2005)
DV = dx * dx * DZ       # voxel volume (m³)

# Derived constants
D_eff = params.calcium.D_ca / (1 + params.calcium.buffer_capacity_kappa_s)
k_pump = params.calcium.pump_vmax / params.calcium.pump_km
decay_length = np.sqrt(D_eff / max(k_pump, 1))


# ═══════════════════════════════════════════════════════════════════
# MODE A — original (different channel models)
# ═══════════════════════════════════════════════════════════════════

def run_modeA_pde(voltage_V, seed):
    np.random.seed(seed)
    sys_pde = PDECalciumSystem(grid_shape, dx, channel_positions.copy(), params)
    peaks, means = [], []
    for step_i in range(N_STEPS):
        sys_pde.step(DT, {'voltage': voltage_V})
        if step_i >= READOUT_START:
            peaks.append(np.max(sys_pde.diffusion.ca_free))
            means.append(np.mean(sys_pde.diffusion.ca_free))
    return np.mean(peaks), np.mean(means)


def run_modeA_analytical(voltage_V, seed):
    np.random.seed(seed)
    sys_an = AnalyticalCalciumSystem(
        grid_shape, dx, channel_positions.copy(), params,
        template_positions=template_positions,
    )
    peaks, means = [], []
    for step_i in range(N_STEPS):
        sys_an.step(DT, {'voltage': voltage_V, 'glutamate': 1.0})
        if step_i >= READOUT_START:
            peaks.append(sys_an.get_peak_concentration())
            means.append(sys_an.get_mean_concentration())
    return np.mean(peaks), np.mean(means)


# ═══════════════════════════════════════════════════════════════════
# MODE B — like-for-like (shared production channels)
# ═══════════════════════════════════════════════════════════════════

def run_modeB(voltage_V, seed):
    """
    Single run at one clamped voltage.
    Returns dict with pde_peak, pde_mean, ana_peak, ana_mean, open_frac
    (all in raw M, not µM).
    """
    np.random.seed(seed)

    # ── shared production channels ──
    channels = ProductionChannels(channel_positions.copy(), params.calcium)

    # ── PDE spatial path (CalciumDiffusion only, no CalciumSystem channels) ──
    pde_diff = CalciumDiffusion(grid_shape, dx, params.calcium)

    # ── analytical spatial path ──
    ana_calc = AnalyticalNanodomainCalculator(params.calcium, dx)

    pde_peaks, pde_means = [], []
    ana_peaks, ana_means = [], []
    open_fracs = []

    for step_i in range(N_STEPS):
        # 1. Shared gating update (production model: NMDAR + VGCC)
        channels.update_gating(DT, voltage=voltage_V, glutamate=1.0)

        # 2a. PDE path: build flux from shared channel state
        flux = np.zeros(grid_shape)
        for ch_i in range(channels.n_channels):
            if channels.state[ch_i]:
                cx, cy = int(channels.positions[ch_i, 0]), int(channels.positions[ch_i, 1])
                flux[cx, cy] += channels.current[ch_i] / (Z_CA * FARADAY * DV)
        pde_diff.add_flux(flux, DT)
        pde_diff.update_buffering(DT)
        pde_diff.update_diffusion(DT)
        pde_diff.update_extrusion(DT)

        # 2b. Analytical path: instantaneous field from same channel state
        ana_field = ana_calc.calculate_field_at_points(
            channels.positions,
            channels.state,
            channels.current,
            ALL_GRID_POINTS,
            dx,
        )

        # 3. Readout (last 5 ms)
        if step_i >= READOUT_START:
            pde_peaks.append(np.max(pde_diff.ca_free))
            pde_means.append(np.mean(pde_diff.ca_free))
            ana_peaks.append(np.max(ana_field))
            ana_means.append(np.mean(ana_field))
            open_fracs.append(np.sum(channels.state) / channels.n_channels)

    return dict(
        pde_peak=np.mean(pde_peaks),
        pde_mean=np.mean(pde_means),
        ana_peak=np.mean(ana_peaks),
        ana_mean=np.mean(ana_means),
        open_frac=np.mean(open_fracs),
    )


# ═══════════════════════════════════════════════════════════════════
# MODE C — onset transfer function (fine sweep across operating range)
# ═══════════════════════════════════════════════════════════════════

def _jahr_stevens_B(voltage_V, mg_conc_mM=1.0):
    """Jahr-Stevens Mg-unblock factor B(V)."""
    V_mV = voltage_V * 1e3
    return 1.0 / (1.0 + np.exp(-0.062 * V_mV + 1.2726) * mg_conc_mM)


def run_modeC(voltage_V, seed):
    """
    Single onset-sweep run at one clamped voltage.
    Returns dict with pde_peak, pde_mean, ana_peak, vgcc_open, nmdar_open, B_V
    (calcium values in raw M, fractions in [0,1]).
    """
    np.random.seed(seed)

    # ── shared production channels ──
    channels = ProductionChannels(channel_positions.copy(), params.calcium)

    # ── PDE spatial path ──
    pde_diff = CalciumDiffusion(grid_shape, dx, params.calcium)

    # ── analytical spatial path ──
    ana_calc = AnalyticalNanodomainCalculator(params.calcium, dx)

    n_vgcc = int(np.sum(~channels.is_nmda))
    n_nmda = int(np.sum(channels.is_nmda))

    pde_peaks, pde_means = [], []
    ana_peaks = []
    vgcc_opens, nmdar_opens = [], []

    for step_i in range(N_STEPS):
        # 1. Shared gating update (glutamate saturating → NMDAR limited only by Mg block)
        channels.update_gating(DT, voltage=voltage_V, glutamate=1.0)

        # 2a. PDE path: flux from shared channel state
        flux = np.zeros(grid_shape)
        for ch_i in range(channels.n_channels):
            if channels.state[ch_i]:
                cx, cy = int(channels.positions[ch_i, 0]), int(channels.positions[ch_i, 1])
                flux[cx, cy] += channels.current[ch_i] / (Z_CA * FARADAY * DV)
        pde_diff.add_flux(flux, DT)
        pde_diff.update_buffering(DT)
        pde_diff.update_diffusion(DT)
        pde_diff.update_extrusion(DT)

        # 2b. Analytical path
        ana_field = ana_calc.calculate_field_at_points(
            channels.positions, channels.state, channels.current,
            ALL_GRID_POINTS, dx,
        )

        # 3. Readout (last 5 ms)
        if step_i >= READOUT_START:
            pde_peaks.append(np.max(pde_diff.ca_free))
            pde_means.append(np.mean(pde_diff.ca_free))
            ana_peaks.append(np.max(ana_field))
            if n_vgcc > 0:
                vgcc_opens.append(
                    np.sum(channels.state[~channels.is_nmda]) / n_vgcc)
            else:
                vgcc_opens.append(0.0)
            if n_nmda > 0:
                nmdar_opens.append(
                    np.sum(channels.state[channels.is_nmda]) / n_nmda)
            else:
                nmdar_opens.append(0.0)

    # Jahr-Stevens B(V) — deterministic, constant for clamped voltage
    B = _jahr_stevens_B(voltage_V, params.calcium.mg_conc_mM)

    return dict(
        pde_peak=np.mean(pde_peaks),
        pde_mean=np.mean(pde_means),
        ana_peak=np.mean(ana_peaks),
        vgcc_open=np.mean(vgcc_opens),
        nmdar_open=np.mean(nmdar_opens),
        B_V=B,
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── mode selection ──
    requested = [a.upper() for a in sys.argv[1:] if a.upper() in ('A', 'B', 'C')]
    if not requested:
        requested = ['A', 'B', 'C']       # default: run all

    print("=" * 90)
    print("CALCIUM PDE REFERENCE PROBE")
    print("=" * 90)
    print()
    print(f"Geometry : {N_CHANNELS} channels in 5×5 cluster, grid {GRID_SIZE}×{GRID_SIZE}, dx={dx*1e9:.0f} nm")
    print(f"Time     : {T_TOTAL*1e3:.0f} ms at dt={DT*1e6:.0f} µs  ({N_STEPS} steps)")
    print(f"Readout  : mean over last {(T_TOTAL - READOUT_START*DT)*1e3:.0f} ms")
    print(f"D_eff    : {D_eff*1e12:.2f} µm²/s")
    print(f"k_pump   : {k_pump:.0f} s⁻¹   λ = {decay_length*1e9:.1f} nm")
    print(f"Flux dV  : dx²·dz = ({dx*1e9:.0f} nm)²·{DZ*1e9:.0f} nm = {DV*1e27:.1f} nm³")
    print(f"Modes    : {', '.join(requested)}")
    print()

    # ───────────────────────────────────────────────────────────────
    # MODE A
    # ───────────────────────────────────────────────────────────────
    if 'A' in requested:
        print("=" * 90)
        print(f"MODE A — different channel models  ({N_SEEDS_A} seeds)")
        print("  PDE  : all-VGCC (V_half=-30 mV, slope=12 mV)")
        print("  Ana  : 50% NMDAR (glut=1, Mg-unblock) + 50% VGCC (V_half=-20 mV, slope=6 mV)")
        print("=" * 90)

        results_a = []
        for v_mV in VOLTAGES_MV:
            v_V = v_mV * 1e-3
            print(f"  V={v_mV:+4d} mV ...", end="", flush=True)
            t0 = time.time()

            pde_p, pde_m, ana_p, ana_m = [], [], [], []
            for s in range(N_SEEDS_A):
                sid = 1000 + s
                pp, pm = run_modeA_pde(v_V, sid)
                ap, am = run_modeA_analytical(v_V, sid)
                pde_p.append(pp); pde_m.append(pm)
                ana_p.append(ap); ana_m.append(am)

            results_a.append(dict(
                v_mV=v_mV,
                pde_peak=np.mean(pde_p)*1e6, pde_mean=np.mean(pde_m)*1e6,
                ana_peak=np.mean(ana_p)*1e6, ana_mean=np.mean(ana_m)*1e6,
            ))
            print(f"  {time.time()-t0:.1f}s")

        hdr_a = (f"{'V(mV)':>7} | {'PDE peak':>10} {'PDE mean':>10} | "
                 f"{'Ana peak':>10} {'Ana mean':>10} | {'pk ratio':>9}")
        print()
        print(hdr_a)
        print("-" * len(hdr_a))
        for r in results_a:
            ratio = r['ana_peak'] / r['pde_peak'] if r['pde_peak'] > 0.001 else float('nan')
            print(f"{r['v_mV']:>+7d} | {r['pde_peak']:>10.3f} {r['pde_mean']:>10.4f} | "
                  f"{r['ana_peak']:>10.3f} {r['ana_mean']:>10.4f} | {ratio:>9.3f}")
        print()

    # ───────────────────────────────────────────────────────────────
    # MODE B
    # ───────────────────────────────────────────────────────────────
    if 'B' in requested:
        print("=" * 90)
        print(f"MODE B — LIKE-FOR-LIKE  ({N_SEEDS_B} seeds)")
        print("  Shared production CalciumChannels (NMDAR+VGCC, glutamate=1.0)")
        print("  PDE path  : flux[x,y] += I/(zF·dV) → buffer → diffuse → extrude")
        print("  Ana path  : 0.5 µM/ch · (I/0.3pA) · exp(-r/λ)")
        print("  ONLY difference = spatial handling")
        print("=" * 90)

        results_b = []
        for v_mV in VOLTAGES_MV:
            v_V = v_mV * 1e-3
            print(f"  V={v_mV:+4d} mV ...", end="", flush=True)
            t0 = time.time()

            seed_runs = []
            for s in range(N_SEEDS_B):
                seed_runs.append(run_modeB(v_V, 2000 + s))

            row = dict(v_mV=v_mV)
            for key in ['pde_peak', 'pde_mean', 'ana_peak', 'ana_mean', 'open_frac']:
                vals = [r[key] for r in seed_runs]
                row[key + '_mean'] = np.mean(vals) * (1e6 if key != 'open_frac' else 1.0)
                row[key + '_sd']   = np.std(vals)  * (1e6 if key != 'open_frac' else 1.0)

            results_b.append(row)
            print(f"  {time.time()-t0:.1f}s")

        print()
        hdr_b = (f"{'V(mV)':>7} | {'open%':>6} | "
                 f"{'PDE peak':>12} {'PDE mean':>12} | "
                 f"{'Ana peak':>12} {'Ana mean':>12} | {'pk ratio':>9}")
        print(hdr_b)
        print("-" * len(hdr_b))
        for r in results_b:
            of = r['open_frac_mean']
            pp = r['pde_peak_mean'];  pm = r['pde_mean_mean']
            ap = r['ana_peak_mean'];  am = r['ana_mean_mean']
            pp_sd = r['pde_peak_sd']; ap_sd = r['ana_peak_sd']
            ratio = ap / pp if pp > 0.001 else float('nan')
            print(f"{r['v_mV']:>+7d} | {of*100:>5.1f} | "
                  f"{pp:>7.3f}±{pp_sd:>4.2f} {pm:>7.4f}±{r['pde_mean_sd']:>4.3f} | "
                  f"{ap:>7.3f}±{ap_sd:>4.2f} {am:>7.4f}±{r['ana_mean_sd']:>4.3f} | {ratio:>9.3f}")

        print()
        print("NOTE: In Mode B both columns see identical channel states at every step.")
        print("      PDE peak is the emergent nanodomain from flux→buffer→diffuse→pump.")
        print("      Ana peak is the calibrated 0.5 µM/channel × exp(−r/190 nm) snapshot.")
        print()

    # ───────────────────────────────────────────────────────────────
    # MODE C — onset transfer function
    # ───────────────────────────────────────────────────────────────
    if 'C' in requested:
        print("=" * 90)
        print(f"MODE C — ONSET TRANSFER FUNCTION  ({N_SEEDS_C} seeds, mean ± sd)")
        print("  Shared production CalciumChannels (NMDAR+VGCC)")
        print("  Glutamate ON/saturating → NMDAR limited only by Mg block")
        print("  Activation map: act = (V_mV + 70) / 30   (-70→0.0, -40→1.0)")
        print("  VGCC: V_half=-20 mV, slope=6 mV")
        print("  NMDAR: Jahr-Stevens B(V) = 1/(1 + exp(-0.062·V + 1.2726)·[Mg])")
        print("=" * 90)

        results_c = []
        for v_mV in VOLTAGES_C_MV:
            v_V = v_mV * 1e-3
            act = (v_mV + 70) / 30.0
            print(f"  act={act:.2f}  V={v_mV:+4d} mV ...", end="", flush=True)
            t0 = time.time()

            seed_runs = []
            for s in range(N_SEEDS_C):
                seed_runs.append(run_modeC(v_V, 3000 + s))

            row = dict(v_mV=v_mV, act=act)

            # Aggregate scalar fields (calcium in M → µM)
            for key in ['pde_peak', 'pde_mean', 'ana_peak']:
                vals = [r[key] for r in seed_runs]
                row[key + '_mean'] = np.mean(vals) * 1e6
                row[key + '_sd']   = np.std(vals) * 1e6

            # Aggregate fraction fields (already [0,1])
            for key in ['vgcc_open', 'nmdar_open']:
                vals = [r[key] for r in seed_runs]
                row[key + '_mean'] = np.mean(vals)
                row[key + '_sd']   = np.std(vals)

            # B(V) is deterministic — same across seeds
            row['B_V'] = seed_runs[0]['B_V']

            # NMDAR conducting = stochastic open fraction × B(V)
            nmdar_cond_vals = [r['nmdar_open'] * r['B_V'] for r in seed_runs]
            row['nmdar_cond_mean'] = np.mean(nmdar_cond_vals)
            row['nmdar_cond_sd']   = np.std(nmdar_cond_vals)

            results_c.append(row)
            print(f"  {time.time()-t0:.1f}s")

        # ── results table ──
        print()
        print("Onset transfer function: PDE-grounded calcium vs activation")
        print("  NMDAR_cond% = NMDAR_stochastic_open × B(V)  [effective conducting fraction]")
        print()
        hdr_c = (f"{'act':>5} | {'V(mV)':>6} | {'VGCC%':>10} | "
                 f"{'NMDAR_c%':>10} | {'B(V)':>5} | "
                 f"{'PDE_peak':>12} | {'PDE_mean':>12} | {'Ana_peak':>12}")
        print(hdr_c)
        print("-" * len(hdr_c))
        for r in results_c:
            act_str = f"{r['act']:.2f}"
            if r['act'] > 1.0:
                act_str += "*"       # flag context rows above operating range
            vgcc_pct  = r['vgcc_open_mean'] * 100
            nmdar_pct = r['nmdar_cond_mean'] * 100
            pp = r['pde_peak_mean']; pp_sd = r['pde_peak_sd']
            pm = r['pde_mean_mean']; pm_sd = r['pde_mean_sd']
            ap = r['ana_peak_mean']; ap_sd = r['ana_peak_sd']
            print(f"{act_str:>6} | {r['v_mV']:>+5d}  | "
                  f"{vgcc_pct:>5.2f}±{r['vgcc_open_sd']*100:>4.2f} | "
                  f"{nmdar_pct:>5.2f}±{r['nmdar_cond_sd']*100:>4.2f} | "
                  f"{r['B_V']:>5.3f} | "
                  f"{pp:>7.4f}±{pp_sd:>4.3f} | "
                  f"{pm:>7.4f}±{pm_sd:>4.3f} | "
                  f"{ap:>7.4f}±{ap_sd:>4.3f}")

        print()
        print("  * = above operating range (context rows)")
        print("  act = (V+70)/30 :  -70 mV → 0.0,  -40 mV → 1.0")
        print("  NMDAR_cond% = fraction of NMDAR channels open (stochastic) × B(V) Mg-unblock")
        print("  PDE_peak/mean, Ana_peak in µM (averaged over readout window & seeds)")
        print()

    print("Done.")
