#!/usr/bin/env python3
"""
input_engine_probe.py

Characterization harness for the input -> calcium transfer (the tunable input engine).
Drives AnalyticalCalciumSystem in ISOLATION (channels + nanodomain -> calcium, nothing
downstream) with controlled voltage protocols, runs N stochastic realizations per
protocol, and records the calcium response distribution.

BASELINE instrument: what does voltage do to calcium today, before the channel rework?
The current channel is a voltage-gated Boltzmann. 'glutamate' is plumbed into the
stimulus dict but is inert until the NMDAR coincidence detector exists.

Nothing here is tuned to an outcome. Protocol parameters (amplitudes, durations,
timings) are the axes a permutation sweep would turn.
"""

import sys, os
import numpy as np

# --- model import (matches sweep/observe_network_partition.py convention) ---
SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)

from analytical_calcium_system import AnalyticalCalciumSystem
from model6_parameters import Model6Parameters

# ----------------------------------------------------------------------------
# Config (geometry mirrors model6_core construction)
# ----------------------------------------------------------------------------
GRID_SHAPE = (100, 100)
ACTIVE_ZONE_RADIUS = 200e-9
DX = 2 * ACTIVE_ZONE_RADIUS / GRID_SHAPE[0]   # 4 nm per grid point
N_CHANNELS = 50
V_REST = -70e-3

DT = 1e-4        # 0.1 ms (resolves channel gating; channel_open_time ~0.5 ms)
N_REAL = 20      # stochastic realizations per protocol (lower if slow)


def make_channel_positions(n_channels, grid_shape, spread_cells=10, seed=0):
    """Fixed channel layout: random integer offsets from grid center.
    Built once and reused across realizations, so the LAYOUT is fixed while only
    the channel gating is stochastic per realization."""
    rng = np.random.default_rng(seed)
    center = np.array([grid_shape[0] // 2, grid_shape[1] // 2])
    offs = rng.integers(-spread_cells, spread_cells + 1, size=(n_channels, 2))
    pos = np.clip(center + offs, 0, np.array(grid_shape) - 1)
    return pos.astype(int)


def voltage_at(t, segments, baseline=V_REST):
    """Protocol = list of (t_start_s, t_end_s, voltage_V); baseline elsewhere."""
    for (t0, t1, v) in segments:
        if t0 <= t < t1:
            return v
    return baseline


def glutamate_at(t, segments, base=0.0):
    """Protocol = list of (t_start_s, t_end_s, glutamate_level); base elsewhere."""
    for (t0, t1, g) in segments:
        if t0 <= t < t1:
            return g
    return base


def run_realization(segments, total_time, dt, channel_positions, params,
                    glutamate=0.0, glutamate_segments=None):
    """One stochastic realization. Fresh system -> fresh gating RNG; layout fixed."""
    ca_sys = AnalyticalCalciumSystem(GRID_SHAPE, DX, channel_positions, params)
    n_steps = int(round(total_time / dt))
    ca = np.empty(n_steps)
    openf = np.empty(n_steps)
    cur = np.empty(n_steps)
    g_eng = np.empty(n_steps)
    for k in range(n_steps):
        t = k * dt
        glu = glutamate_at(t, glutamate_segments) if glutamate_segments is not None else glutamate
        stim = {"voltage": voltage_at(t, segments), "glutamate": glu}
        ca_sys.step(dt, stim)
        ca[k] = float(np.max(ca_sys.get_concentration()))
        openf[k] = float(ca_sys.channels.get_open_fraction())
        cur[k] = float(np.sum(ca_sys.channels.current))
        g_eng[k] = float(ca_sys.channels.g_engaged)
    return np.arange(n_steps) * dt, ca, openf, cur, g_eng


def run_protocol(name, segments, total_time, dt, n_real, channel_positions, params,
                 glutamate=0.0, glutamate_segments=None):
    ca_all, open_all, cur_all, t_arr, g_eng_first = [], [], [], None, None
    for i in range(n_real):
        t_arr, ca, openf, cur, g_eng = run_realization(
            segments, total_time, dt, channel_positions, params, glutamate,
            glutamate_segments)
        ca_all.append(ca); open_all.append(openf); cur_all.append(cur)
        if i == 0:
            g_eng_first = g_eng
    ca_all = np.array(ca_all); open_all = np.array(open_all); cur_all = np.array(cur_all)

    peak = ca_all.max(axis=1)            # per-realization peak [M]
    integral = ca_all.sum(axis=1) * dt   # per-realization integral [M*s]

    print(f"\n=== {name} ===")
    print(f"  realizations {n_real} | total {total_time*1e3:.0f} ms | dt {dt*1e3:.2f} ms")
    print(f"  peak Ca [uM]:    mean {peak.mean()*1e6:8.3f}   std {peak.std()*1e6:8.3f}")
    print(f"  integral [uM*s]: mean {integral.mean()*1e6:8.4f}   std {integral.std()*1e6:8.4f}")
    print(f"  max open frac:   mean {open_all.max(1).mean():8.3f}")
    return {"name": name, "t": t_arr,
            "ca_mean": ca_all.mean(0), "ca_std": ca_all.std(0),
            "open_mean": open_all.mean(0), "cur_mean": cur_all.mean(0),
            "peak": peak, "integral": integral,
            "g_engaged": g_eng_first}


def main():
    params = Model6Parameters()
    channel_positions = make_channel_positions(N_CHANNELS, GRID_SHAPE)
    results = []

    # STRENGTH sweep: single 50 ms depolarizing pulse at increasing amplitude
    for v_mV in (-60, -50, -40, -30, -20, -10, 0):
        seg = [(0.05, 0.10, v_mV * 1e-3)]
        results.append(run_protocol(f"strength_{v_mV:+d}mV", seg, 0.15, DT,
                                    N_REAL, channel_positions, params))

    # SEQUENCE: train of three 10 ms pulses at -20 mV, 20 ms apart
    train = [(0.02, 0.03, -20e-3), (0.05, 0.06, -20e-3), (0.08, 0.09, -20e-3)]
    results.append(run_protocol("train_3x10ms_-20mV", train, 0.12, DT,
                                N_REAL, channel_positions, params))

    # --- COINCIDENCE characterization (NMDAR needs glutamate AND depolarization) ---
    for v_mV in (-40, -20, 0):
        seg = [(0.05, 0.10, v_mV * 1e-3)]
        results.append(run_protocol(f"Vonly_{v_mV:+d}mV", seg, 0.15, DT, N_REAL,
                                    channel_positions, params, glutamate=0.0))
        results.append(run_protocol(f"coinc_{v_mV:+d}mV", seg, 0.15, DT, N_REAL,
                                    channel_positions, params, glutamate=1.0))
    results.append(run_protocol("Glu_only_rest", [], 0.15, DT, N_REAL,
                                channel_positions, params, glutamate=1.0))

    # --- PULSE protocol: measure g_engaged decay directly ---
    pulse_glu_seg = [(0.010, 0.012, 1.0)]   # 2 ms glutamate pulse
    pulse_v_seg   = [(0.0, 0.6, -20e-3)]    # held at -20 mV whole window
    r_pulse = run_protocol("pulse_glu2ms_-20mV", pulse_v_seg, 0.6, DT, N_REAL,
                           channel_positions, params,
                           glutamate_segments=pulse_glu_seg)
    results.append(r_pulse)

    # Diagnostic: g_engaged decay timing
    g = r_pulse["g_engaged"]
    t = r_pulse["t"]
    peak_g = g.max()
    t_off = 0.012
    mask_post = t >= t_off
    g_post = g[mask_post]
    t_post = t[mask_post]
    tau_nmda = params.calcium.tau_nmda
    t_1e = t_half = None
    for j in range(len(g_post)):
        if t_1e is None and g_post[j] <= peak_g * np.exp(-1):
            t_1e = t_post[j] - t_off
        if t_half is None and g_post[j] <= peak_g * 0.5:
            t_half = t_post[j] - t_off
    print(f"\n=== pulse g_engaged decay ===")
    print(f"  peak g_engaged:        {peak_g:.4f}")
    print(f"  1/e crossing:          {t_1e*1e3:.1f} ms after pulse-off   (expected: {tau_nmda*1e3:.1f} ms)")
    print(f"  half-life crossing:    {t_half*1e3:.1f} ms after pulse-off   (expected: {tau_nmda*np.log(2)*1e3:.1f} ms)")

    out = os.path.join(SWEEP_DIR, "input_engine_probe_results.npz")
    save = {}
    for r in results:
        for k in ("t", "ca_mean", "ca_std", "open_mean", "cur_mean", "peak", "integral"):
            save[f"{r['name']}__{k}"] = r[k]
    np.savez_compressed(out, **save)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
