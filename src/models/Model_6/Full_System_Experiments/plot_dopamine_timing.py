#!/usr/bin/env python3
"""
Dopamine Timing Experiment: Three-Panel Figure
================================================

Panel A: Eligibility at readout (quantum memory persistence)
Panel B: Commit probability vs delay (probabilistic gate output)
Panel C: Δ Synaptic strength vs delay (functional outcome)

Usage:
    python plot_dopamine_timing.py [--raw PATH] [--summary PATH] [--output PATH]
    
    Defaults look for results/ directory in current folder.

Author: Sarah Davidson
Date: February 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from pathlib import Path
import argparse


def load_data(raw_path: str = None, summary_path: str = None):
    """Load raw and/or summary data"""
    
    raw = None
    summary = None
    
    if raw_path:
        with open(raw_path) as f:
            raw = json.load(f)
    
    if summary_path:
        with open(summary_path) as f:
            summary = json.load(f)
    
    return raw, summary


def compute_stats_from_raw(raw_data):
    """Compute per-delay statistics from raw trial data"""
    
    stats = {}
    
    for ctype in ['stim_plus_dopamine', 'stim_no_dopamine', 'dopamine_only']:
        ctype_trials = [r for r in raw_data if r['condition_type'] == ctype]
        delays = sorted(set(r['delay'] for r in ctype_trials))
        
        stats[ctype] = {}
        for delay in delays:
            d_trials = [r for r in ctype_trials if r['delay'] == delay]
            
            eligs = [r['eligibility_at_readout'] for r in d_trials]
            commits = [1 if r['committed'] else 0 for r in d_trials]
            strengths = [r['delta_strength'] for r in d_trials]
            
            n = len(d_trials)
            commit_rate = np.mean(commits)
            
            # Wilson score interval for commit rate (better than normal approx)
            z = 1.96  # 95% CI
            if n > 0:
                p_hat = commit_rate
                denom = 1 + z**2 / n
                center = (p_hat + z**2 / (2 * n)) / denom
                spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
                commit_ci_low = max(0, center - spread)
                commit_ci_high = min(1, center + spread)
            else:
                commit_ci_low = commit_ci_high = 0
            
            stats[ctype][delay] = {
                'eligibility_mean': float(np.mean(eligs)),
                'eligibility_std': float(np.std(eligs)),
                'eligibility_sem': float(np.std(eligs) / np.sqrt(n)) if n > 1 else 0,
                'commit_rate': float(commit_rate),
                'commit_ci_low': float(commit_ci_low),
                'commit_ci_high': float(commit_ci_high),
                'delta_strength_mean': float(np.mean(strengths)),
                'delta_strength_std': float(np.std(strengths)),
                'delta_strength_sem': float(np.std(strengths) / np.sqrt(n)) if n > 1 else 0,
                'n_trials': n,
                # Individual trials for scatter
                'eligibilities': eligs,
                'strengths': strengths,
                'commits': commits,
            }
    
    return stats


def fit_exponential(x, y, weights=None):
    """Fit y = A * exp(-x / tau) and return params + R²"""
    try:
        def decay(t, A, tau):
            return A * np.exp(-t / tau)
        
        popt, pcov = curve_fit(
            decay, x, y,
            p0=[max(y) if max(y) > 0 else 0.5, 100.0],
            bounds=([0, 1], [2.0, 1000]),
            sigma=weights,
            maxfev=5000
        )
        
        A, tau = popt
        y_pred = decay(x, A, tau)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return A, tau, r_sq
    except Exception:
        return None, None, None


def plot_three_panel(stats, output_path=None):
    """Generate three-panel dopamine timing figure"""
    
    # --- Setup ---
    fig = plt.figure(figsize=(15, 4.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    # Colors
    c_main = '#2E86AB'       # Blue for main condition
    c_no_da = '#8B8B8B'      # Gray for stim-only control
    c_da_only = '#C4C4C4'    # Light gray for dopamine-only control
    c_fit = '#E94F37'        # Red-orange for fitted curve
    c_scatter = '#2E86AB'    # Blue scatter dots
    
    spd = stats.get('stim_plus_dopamine', {})
    sno = stats.get('stim_no_dopamine', {})
    dop = stats.get('dopamine_only', {})
    
    delays_main = sorted(spd.keys())
    delays_ctrl = sorted(sno.keys())
    
    # =====================================================================
    # PANEL A: Eligibility at readout
    # =====================================================================
    ax_a = fig.add_subplot(gs[0])
    
    eligs = [spd[d]['eligibility_mean'] for d in delays_main]
    elig_sem = [spd[d]['eligibility_sem'] for d in delays_main]
    
    # Individual trial scatter
    for d in delays_main:
        jitter = np.random.normal(0, 0.8, len(spd[d]['eligibilities']))
        ax_a.scatter(
            [d + j for j in jitter],
            spd[d]['eligibilities'],
            c=c_scatter, alpha=0.15, s=12, edgecolors='none', zorder=2
        )
    
    # Mean with error bars
    ax_a.errorbar(delays_main, eligs, yerr=elig_sem,
                  fmt='o-', color=c_main, markersize=5, linewidth=1.8,
                  capsize=3, capthick=1, zorder=5, label='Stim + Dopamine')
    
    # Fit eligibility decay
    x_elig = np.array(delays_main, dtype=float)
    y_elig = np.array(eligs)
    A_e, tau_e, r2_e = fit_exponential(x_elig, y_elig)
    
    if A_e is not None:
        t_smooth = np.linspace(0, max(delays_main) + 10, 200)
        y_fit = A_e * np.exp(-t_smooth / tau_e)
        ax_a.plot(t_smooth, y_fit, '--', color=c_fit, linewidth=1.5, alpha=0.8,
                  label=f'Fit: τ = {tau_e:.0f}s (R² = {r2_e:.2f})')
    
    # Controls
    if sno:
        ctrl_eligs = [sno[d]['eligibility_mean'] for d in delays_ctrl]
        ax_a.plot(delays_ctrl, ctrl_eligs, 's--', color=c_no_da, markersize=4,
                  linewidth=1, alpha=0.7, label='Stim only')
    
    ax_a.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax_a.set_xlabel('Dopamine Delay (s)', fontsize=11)
    ax_a.set_ylabel('Eligibility (mean singlet prob.)', fontsize=11)
    ax_a.set_title('A. Quantum Memory Persistence', fontsize=12, fontweight='bold')
    ax_a.legend(fontsize=8, loc='upper right')
    ax_a.set_xlim(-5, max(delays_main) + 10)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    
    # =====================================================================
    # PANEL B: Commit probability (THE key new result)
    # =====================================================================
    ax_b = fig.add_subplot(gs[1])
    
    commits = [spd[d]['commit_rate'] for d in delays_main]
    ci_low = [spd[d]['commit_ci_low'] for d in delays_main]
    ci_high = [spd[d]['commit_ci_high'] for d in delays_main]
    
    # Error bars from Wilson CI
    yerr_low = [c - lo for c, lo in zip(commits, ci_low)]
    yerr_high = [hi - c for c, hi in zip(commits, ci_high)]
    
    ax_b.errorbar(delays_main, commits, yerr=[yerr_low, yerr_high],
                  fmt='o-', color=c_main, markersize=6, linewidth=2,
                  capsize=4, capthick=1.2, zorder=5, label='Stim + Dopamine')
    
    # Fit commit rate decay
    x_c = np.array(delays_main, dtype=float)
    y_c = np.array(commits)
    A_c, tau_c, r2_c = fit_exponential(x_c, y_c)
    
    if A_c is not None:
        t_smooth = np.linspace(0, max(delays_main) + 10, 200)
        y_fit_c = A_c * np.exp(-t_smooth / tau_c)
        ax_b.plot(t_smooth, y_fit_c, '--', color=c_fit, linewidth=1.5, alpha=0.8,
                  label=f'Fit: τ = {tau_c:.0f}s (R² = {r2_c:.2f})')
    
    # Control flat lines
    if sno:
        ctrl_commits = [sno[d]['commit_rate'] for d in delays_ctrl]
        ax_b.plot(delays_ctrl, ctrl_commits, 's--', color=c_no_da, markersize=4,
                  linewidth=1, alpha=0.7, label='Stim only (0%)')
    if dop:
        dop_commits = [dop[d]['commit_rate'] for d in delays_ctrl]
        ax_b.plot(delays_ctrl, dop_commits, '^--', color=c_da_only, markersize=4,
                  linewidth=1, alpha=0.7, label='Dopamine only (0%)')
    
    ax_b.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax_b.set_xlabel('Dopamine Delay (s)', fontsize=11)
    ax_b.set_ylabel('Commit Probability', fontsize=11)
    ax_b.set_title('B. Probabilistic Gate Output', fontsize=12, fontweight='bold')
    ax_b.legend(fontsize=8, loc='upper right')
    ax_b.set_xlim(-5, max(delays_main) + 10)
    ax_b.set_ylim(-0.05, 1.1)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    
    # Annotate the key timescale
    if tau_c is not None:
        ax_b.annotate(f'Functional T₂ ≈ {tau_c:.0f}s',
                      xy=(tau_c, 0.37), fontsize=9, color=c_fit,
                      ha='center', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=c_fit, alpha=0.8))
    
    # =====================================================================
    # PANEL C: Δ Synaptic strength (functional outcome)
    # =====================================================================
    ax_c = fig.add_subplot(gs[2])
    
    strengths = [spd[d]['delta_strength_mean'] for d in delays_main]
    strength_sem = [spd[d]['delta_strength_sem'] for d in delays_main]
    
    # Individual trial scatter
    for d in delays_main:
        jitter = np.random.normal(0, 0.8, len(spd[d]['strengths']))
        ax_c.scatter(
            [d + j for j in jitter],
            spd[d]['strengths'],
            c=c_scatter, alpha=0.15, s=12, edgecolors='none', zorder=2
        )
    
    # Mean with error bars
    ax_c.errorbar(delays_main, strengths, yerr=strength_sem,
                  fmt='o-', color=c_main, markersize=5, linewidth=1.8,
                  capsize=3, capthick=1, zorder=5, label='Stim + Dopamine')
    
    # Fit strength decay
    x_s = np.array(delays_main, dtype=float)
    y_s = np.array(strengths)
    A_s, tau_s, r2_s = fit_exponential(x_s, y_s)
    
    if A_s is not None:
        t_smooth = np.linspace(0, max(delays_main) + 10, 200)
        y_fit_s = A_s * np.exp(-t_smooth / tau_s)
        ax_c.plot(t_smooth, y_fit_s, '--', color=c_fit, linewidth=1.5, alpha=0.8,
                  label=f'Fit: T₂ = {tau_s:.0f}s (R² = {r2_s:.2f})')
    
    # Controls
    if sno:
        ctrl_str = [sno[d]['delta_strength_mean'] for d in delays_ctrl]
        ax_c.plot(delays_ctrl, ctrl_str, 's--', color=c_no_da, markersize=4,
                  linewidth=1, alpha=0.7, label='Stim only')
    if dop:
        dop_str = [dop[d]['delta_strength_mean'] for d in delays_ctrl]
        ax_c.plot(delays_ctrl, dop_str, '^--', color=c_da_only, markersize=4,
                  linewidth=1, alpha=0.7, label='Dopamine only')
    
    ax_c.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)
    ax_c.set_xlabel('Dopamine Delay (s)', fontsize=11)
    ax_c.set_ylabel('Δ Synaptic Strength', fontsize=11)
    ax_c.set_title('C. Functional LTP Outcome', fontsize=12, fontweight='bold')
    ax_c.legend(fontsize=8, loc='upper right')
    ax_c.set_xlim(-5, max(delays_main) + 10)
    ax_c.set_ylim(-0.05, 0.6)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    
    # =====================================================================
    # Figure-level annotation
    # =====================================================================
    fig.suptitle(
        'Dopamine Timing Experiment: Quantum Eligibility Trace Enables Temporal Credit Assignment',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    # Caption-style annotation
    caption = (
        f'N = {spd[delays_main[0]]["n_trials"]} trials/condition, '
        f'10 synapses/network, coordination mode. '
        f'Controls: activity-only and dopamine-only both 0% at all delays.'
    )
    fig.text(0.5, -0.04, caption, ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved figure to {output_path}")
    
    return fig


def print_summary_table(stats):
    """Print formatted summary"""
    spd = stats.get('stim_plus_dopamine', {})
    delays = sorted(spd.keys())
    
    print("\n" + "=" * 85)
    print("DOPAMINE TIMING: PROBABILISTIC GATE RESULTS")
    print("=" * 85)
    print(f"{'Delay (s)':<10} {'Eligibility':<16} {'Commit Rate':<14} {'95% CI':<16} {'Δ Strength':<16} {'N':>4}")
    print("-" * 85)
    
    for d in delays:
        s = spd[d]
        print(f"{d:<10.0f} "
              f"{s['eligibility_mean']:.3f} ± {s['eligibility_std']:.3f}  "
              f"{s['commit_rate']:>6.0%}         "
              f"[{s['commit_ci_low']:.2f}, {s['commit_ci_high']:.2f}]   "
              f"{s['delta_strength_mean']:.3f} ± {s['delta_strength_std']:.3f}  "
              f"{s['n_trials']:>4}")
    
    # Fit summary
    x = np.array(delays, dtype=float)
    
    y_e = np.array([spd[d]['eligibility_mean'] for d in delays])
    A_e, tau_e, r2_e = fit_exponential(x, y_e)
    
    y_c = np.array([spd[d]['commit_rate'] for d in delays])
    A_c, tau_c, r2_c = fit_exponential(x, y_c)
    
    y_s = np.array([spd[d]['delta_strength_mean'] for d in delays])
    A_s, tau_s, r2_s = fit_exponential(x, y_s)
    
    print("\n" + "-" * 85)
    print("FITTED DECAY CONSTANTS:")
    if tau_e: print(f"  Eligibility:   τ = {tau_e:.1f}s, A = {A_e:.3f}, R² = {r2_e:.3f}")
    if tau_c: print(f"  Commit rate:   τ = {tau_c:.1f}s, A = {A_c:.3f}, R² = {r2_c:.3f}")
    if tau_s: print(f"  Δ Strength:    τ = {tau_s:.1f}s, A = {A_s:.3f}, R² = {r2_s:.3f}")
    print("=" * 85)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot dopamine timing results')
    parser.add_argument('--raw', type=str, default='results/dopamine_timing_parallel_raw.json',
                        help='Path to raw results JSON')
    parser.add_argument('--summary', type=str, default='results/dopamine_timing_results.json',
                        help='Path to summary JSON')
    parser.add_argument('--output', type=str, default='results/dopamine_timing_3panel.png',
                        help='Output figure path')
    args = parser.parse_args()
    
    print("Loading data...")
    raw, summary = load_data(args.raw, args.summary)
    
    if raw:
        print(f"  Raw: {len(raw)} trials loaded")
        stats = compute_stats_from_raw(raw)
    else:
        print("  ERROR: Raw data required for this visualization")
        exit(1)
    
    print_summary_table(stats)
    
    fig = plot_three_panel(stats, output_path=args.output)
    print("\nDone.")
    plt.show()