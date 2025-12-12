"""
Visualization Suite
====================

Comprehensive, publication-quality figures for quantum cascade experiments.

Figure types:
1. System architecture diagram
2. Single experiment plots (per experiment type)
3. Multi-panel summary figures
4. Time course / cascade flow plots
5. Statistical comparison plots
6. Dose-response and threshold curves
7. Heatmaps for parameter sweeps
8. 3D surface plots for interactions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Ellipse
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from core import ExperimentResult, SystemState


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color schemes
COLORS = {
    # Quantum systems
    'Q1': '#1f77b4',      # Blue - Tryptophan
    'Q2': '#2ca02c',      # Green - Dimers
    'classical': '#9467bd',  # Purple - Classical
    
    # Conditions
    'control': '#2ca02c',
    'blocked': '#d62728',
    'partial': '#ff7f0e',
    
    # Data
    'data': '#1f77b4',
    'fit': '#d62728',
    'theory': '#7f7f7f',
    
    # Isotopes
    'P31': '#1f77b4',
    'P32': '#d62728',
}

# Figure sizes (inches)
FIGSIZE = {
    'single': (8, 6),
    'double': (12, 5),
    'triple': (16, 5),
    'quad': (12, 10),
    'full_page': (12, 16),
    'presentation': (14, 8),
}


def setup_style():
    """Set up matplotlib style for publication"""
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'errorbar.capsize': 3,
    })
    
    if HAS_SEABORN:
        sns.set_palette("colorblind")


# =============================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================

def plot_system_architecture(save_path: Optional[str] = None):
    """
    Create detailed system architecture diagram showing
    Q1 → Q2 → Classical cascade with all connections
    """
    
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # === QUANTUM SYSTEM 1: Tryptophan ===
    q1_box = FancyBboxPatch((0.5, 6.5), 4, 2.5, 
                            boxstyle="round,pad=0.1",
                            facecolor='#e6f2ff', edgecolor=COLORS['Q1'], linewidth=2)
    ax.add_patch(q1_box)
    ax.text(2.5, 8.5, "QUANTUM SYSTEM 1", fontweight='bold', ha='center', fontsize=12, color=COLORS['Q1'])
    ax.text(2.5, 8.0, "Tryptophan Superradiance", ha='center', fontsize=11)
    ax.text(2.5, 7.5, "• MT lattice organization", ha='center', fontsize=9)
    ax.text(2.5, 7.1, "• Ground-state coupling", ha='center', fontsize=9)
    ax.text(2.5, 6.7, "• EM field: 16-20 kT", ha='center', fontsize=9)
    
    # Q1 modulators
    ax.annotate("MT invasion\n(+1200 Trp)", xy=(0.5, 7.5), xytext=(-1, 7.5),
                ha='right', fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate("Anesthetics\n(block)", xy=(0.5, 7.0), xytext=(-1, 6.5),
                ha='right', fontsize=8, color='red',
                arrowprops=dict(arrowstyle='-|>', color='red', lw=1))
    
    # === FORWARD COUPLING ===
    ax.annotate("", xy=(7, 7.5), xytext=(4.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q1'], lw=3))
    ax.text(5.75, 7.9, "Forward Coupling", fontsize=9, ha='center', color=COLORS['Q1'])
    ax.text(5.75, 7.2, "k_agg enhancement", fontsize=8, ha='center', style='italic')
    
    # === QUANTUM SYSTEM 2: Dimers ===
    q2_box = FancyBboxPatch((7, 6.5), 4, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#e6ffe6', edgecolor=COLORS['Q2'], linewidth=2)
    ax.add_patch(q2_box)
    ax.text(9, 8.5, "QUANTUM SYSTEM 2", fontweight='bold', ha='center', fontsize=12, color=COLORS['Q2'])
    ax.text(9, 8.0, "Ca-Phosphate Dimers", ha='center', fontsize=11)
    ax.text(9, 7.5, "• Ca₆(PO₄)₄ formation", ha='center', fontsize=9)
    ax.text(9, 7.1, "• Nuclear spin coherence", ha='center', fontsize=9)
    ax.text(9, 6.7, "• T₂ ≈ 67s (P31)", ha='center', fontsize=9)
    
    # Q2 modulators
    ax.annotate("Calcium\n(NMDA)", xy=(11, 8.0), xytext=(12.5, 8.5),
                ha='left', fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax.annotate("APV\n(block Ca)", xy=(11, 7.5), xytext=(12.5, 7.5),
                ha='left', fontsize=8, color='red',
                arrowprops=dict(arrowstyle='-|>', color='red', lw=1))
    ax.annotate("P32 isotope\n(short T₂)", xy=(11, 7.0), xytext=(12.5, 6.5),
                ha='left', fontsize=8, color='red',
                arrowprops=dict(arrowstyle='-|>', color='red', lw=1))
    
    # === REVERSE COUPLING ===
    ax.annotate("", xy=(4.5, 6.8), xytext=(7, 6.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q2'], lw=2, ls='--'))
    ax.text(5.75, 6.4, "Reverse Coupling", fontsize=9, ha='center', color=COLORS['Q2'])
    
    # === PLASTICITY GATE ===
    gate_box = FancyBboxPatch((3.5, 4), 4, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#ffffd0', edgecolor='orange', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(5.5, 5.1, "PLASTICITY GATE", fontweight='bold', ha='center', fontsize=11, color='orange')
    ax.text(5.5, 4.5, "Eligibility + Dopamine + Calcium", ha='center', fontsize=9)
    ax.text(5.5, 4.15, "(All three required)", ha='center', fontsize=8, style='italic')
    
    # Inputs to gate
    ax.annotate("", xy=(5.5, 5.5), xytext=(9, 6.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q2'], lw=2))
    ax.text(7.5, 6.2, "Eligibility", fontsize=8, color=COLORS['Q2'])
    
    ax.annotate("Dopamine\n(reward)", xy=(3.5, 4.5), xytext=(1.5, 4.5),
                ha='right', fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    
    # === CLASSICAL BRIDGE ===
    ax.annotate("", xy=(9, 4), xytext=(7.5, 4.75),
                arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    
    classical_box = FancyBboxPatch((7, 1.5), 6, 2.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#f0e6ff', edgecolor=COLORS['classical'], linewidth=2)
    ax.add_patch(classical_box)
    ax.text(10, 3.5, "CLASSICAL CASCADE", fontweight='bold', ha='center', fontsize=12, color=COLORS['classical'])
    
    # Classical cascade components
    cascade_y = 2.8
    components = ["CaMKII\npT286", "Molecular\nMemory", "Actin\nDynamics", "AMPAR\nTrafficking", "Synaptic\nStrength"]
    x_positions = [7.5, 8.7, 9.9, 11.1, 12.3]
    
    for i, (comp, x) in enumerate(zip(components, x_positions)):
        ax.text(x, cascade_y, comp, ha='center', fontsize=8, va='top')
        if i < len(components) - 1:
            ax.annotate("", xy=(x_positions[i+1]-0.3, cascade_y-0.3), 
                       xytext=(x+0.3, cascade_y-0.3),
                       arrowprops=dict(arrowstyle='->', color=COLORS['classical'], lw=1))
    
    ax.text(10, 1.8, "THE FUNCTIONAL OUTPUT", fontweight='bold', ha='center', 
            fontsize=10, color=COLORS['classical'], style='italic')
    
    # === TEMPLATE FEEDBACK ===
    ax.annotate("", xy=(9, 6.5), xytext=(10, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls=':'))
    ax.text(10.5, 5.2, "Template\nFeedback", fontsize=8, color='gray', ha='left')
    
    # === TIMESCALES ===
    time_box = FancyBboxPatch((0.5, 0.5), 5, 1.5,
                              boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='gray', linewidth=1)
    ax.add_patch(time_box)
    ax.text(3, 1.7, "TIMESCALES", fontweight='bold', ha='center', fontsize=10)
    ax.text(3, 1.3, "Q1: fs-ps (superradiance)", ha='center', fontsize=8, color=COLORS['Q1'])
    ax.text(3, 0.95, "Q2: 1-100s (spin coherence)", ha='center', fontsize=8, color=COLORS['Q2'])
    ax.text(3, 0.6, "Classical: min-hours (consolidation)", ha='center', fontsize=8, color=COLORS['classical'])
    
    plt.title("Quantum-Classical Cascade Architecture\n", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 2. NETWORK THRESHOLD FIGURE
# =============================================================================

def plot_network_threshold(result: ExperimentResult, save_path: Optional[str] = None):
    """
    Plot network size threshold experiment results
    
    Shows:
    - Dimer count vs N (with threshold line)
    - Synaptic strength vs N (with Hill fit)
    - Cooperativity analysis
    """
    
    setup_style()
    fig = plt.figure(figsize=FIGSIZE['triple'])
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    summary = result.summary_stats
    n_values = sorted(summary.keys())
    
    # Panel A: Dimer count
    ax1 = fig.add_subplot(gs[0])
    
    means = [summary[n]['dimers']['mean'] for n in n_values]
    sems = [summary[n]['dimers']['sem'] for n in n_values]
    
    ax1.errorbar(n_values, means, yerr=sems, fmt='o-', color=COLORS['Q2'], 
                 capsize=3, markersize=8, linewidth=2)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50 dimer threshold')
    ax1.fill_between([0, max(n_values)+2], 0, 50, alpha=0.1, color='red')
    
    ax1.set_xlabel('Number of Synapses (N)')
    ax1.set_ylabel('Peak Dimer Count')
    ax1.set_title('A. Dimer Formation', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, max(n_values) + 2)
    ax1.set_ylim(0, None)
    
    # Panel B: Synaptic strength
    ax2 = fig.add_subplot(gs[1])
    
    means = [summary[n]['strength']['mean'] for n in n_values]
    sems = [summary[n]['strength']['sem'] for n in n_values]
    
    ax2.errorbar(n_values, means, yerr=sems, fmt='s-', color=COLORS['classical'],
                 capsize=3, markersize=8, linewidth=2, label='Data')
    
    # Add Hill fit if available
    if 'hill_fit' in result.fitted_params:
        hf = result.fitted_params['hill_fit']
        if hf['EC50'] > 0:
            x_fit = np.linspace(min(n_values), max(n_values), 100)
            y_fit = hf['bottom'] + (hf['top'] - hf['bottom']) / (1 + (hf['EC50'] / x_fit) ** hf['hill_n'])
            ax2.plot(x_fit, y_fit, '--', color=COLORS['fit'], linewidth=2,
                    label=f"Hill fit (EC₅₀={hf['EC50']:.1f})")
    
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Number of Synapses (N)')
    ax2.set_ylabel('Final Synaptic Strength')
    ax2.set_title('B. Plasticity Output', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, max(n_values) + 2)
    
    # Panel C: Cooperativity
    ax3 = fig.add_subplot(gs[2])
    
    # Plot dimer scaling
    dimers = np.array([summary[n]['dimers']['mean'] for n in n_values])
    n_arr = np.array(n_values)
    
    # Normalize to N=1 (or smallest)
    if dimers[0] > 0:
        dimers_norm = dimers / dimers[0]
        n_norm = n_arr / n_arr[0]
        
        ax3.loglog(n_norm, dimers_norm, 'o-', color=COLORS['Q2'], 
                   markersize=8, linewidth=2, label='Observed')
        
        # Linear reference
        ax3.loglog(n_norm, n_norm, '--', color='gray', alpha=0.5, label='Linear (slope=1)')
        
        # Cooperative reference (slope=2)
        ax3.loglog(n_norm, n_norm**2, ':', color='red', alpha=0.5, label='Cooperative (slope=2)')
    
    coop = result.fitted_params.get('cooperativity', 1.0)
    ax3.set_xlabel('Normalized N')
    ax3.set_ylabel('Normalized Dimer Count')
    ax3.set_title(f'C. Cooperativity = {coop:.2f}', fontweight='bold')
    ax3.legend(loc='upper left')
    
    plt.suptitle('Network Size Threshold Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 3. DOPAMINE TIMING FIGURE
# =============================================================================

def plot_dopamine_timing(result: ExperimentResult, save_path: Optional[str] = None):
    """
    Plot dopamine timing experiment showing T2 decay
    """
    
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['double'])
    
    summary = result.summary_stats
    delays = sorted(summary.keys())
    
    # Panel A: Eligibility decay
    ax1 = axes[0]
    
    means = [summary[d]['pre_da_eligibility']['mean'] for d in delays]
    sems = [summary[d]['pre_da_eligibility']['sem'] for d in delays]
    
    ax1.errorbar(delays, means, yerr=sems, fmt='o-', color=COLORS['Q2'],
                 capsize=3, markersize=8, linewidth=2, label='Data')
    
    # Theoretical decay
    t_theory = np.linspace(0, max(delays), 100)
    T2 = 67  # seconds
    y_theory = np.exp(-t_theory / T2)
    ax1.plot(t_theory, y_theory, '--', color=COLORS['theory'], linewidth=2, 
             alpha=0.7, label=f'Theory (T₂=67s)')
    
    # Fitted decay
    if 'decay_fit' in result.fitted_params:
        df = result.fitted_params['decay_fit']
        if df['tau'] > 0:
            y_fit = df['A'] * np.exp(-t_theory / df['tau']) + df['baseline']
            ax1.plot(t_theory, y_fit, '-', color=COLORS['fit'], linewidth=2,
                    label=f"Fit (τ={df['tau']:.1f}s)")
    
    ax1.set_xlabel('Dopamine Delay (s)')
    ax1.set_ylabel('Eligibility at Dopamine')
    ax1.set_title('A. Eligibility Trace Decay', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-2, max(delays) + 5)
    ax1.set_ylim(0, 1.1)
    
    # Panel B: Synaptic strength
    ax2 = axes[1]
    
    means = [summary[d]['strength']['mean'] for d in delays]
    sems = [summary[d]['strength']['sem'] for d in delays]
    
    ax2.errorbar(delays, means, yerr=sems, fmt='s-', color=COLORS['classical'],
                 capsize=3, markersize=8, linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Dopamine Delay (s)')
    ax2.set_ylabel('Final Synaptic Strength')
    ax2.set_title('B. Plasticity Depends on Delay', fontweight='bold')
    ax2.set_xlim(-2, max(delays) + 5)
    
    plt.suptitle('Temporal Credit Assignment: Eligibility Decay', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 4. ISOTOPE COMPARISON FIGURE
# =============================================================================

def plot_isotope_comparison(result: ExperimentResult, save_path: Optional[str] = None):
    """
    Plot P31 vs P32 comparison - the definitive quantum test
    """
    
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['double'])
    
    summary = result.summary_stats
    
    # Panel A: Eligibility decay comparison
    ax1 = axes[0]
    
    for isotope, color, marker in [('P31', COLORS['P31'], 'o'), ('P32', COLORS['P32'], 's')]:
        if isotope in summary:
            delays = sorted(summary[isotope].keys())
            means = [summary[isotope][d]['pre_da_eligibility']['mean'] for d in delays]
            sems = [summary[isotope][d]['pre_da_eligibility']['sem'] for d in delays]
            
            ax1.errorbar(delays, means, yerr=sems, fmt=f'{marker}-', color=color,
                        capsize=3, markersize=8, linewidth=2, label=isotope)
    
    # Theoretical curves
    t = np.linspace(0, 70, 100)
    ax1.plot(t, np.exp(-t / 67), '--', color=COLORS['P31'], alpha=0.5, linewidth=1.5, label='T₂=67s (P31)')
    ax1.plot(t, np.exp(-t / 0.3), '--', color=COLORS['P32'], alpha=0.5, linewidth=1.5, label='T₂=0.3s (P32)')
    
    ax1.set_xlabel('Dopamine Delay (s)')
    ax1.set_ylabel('Eligibility')
    ax1.set_title('A. Isotope Effect on Coherence', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-1, 70)
    ax1.set_ylim(0, 1.1)
    
    # Panel B: Synaptic strength comparison
    ax2 = axes[1]
    
    for isotope, color, marker in [('P31', COLORS['P31'], 'o'), ('P32', COLORS['P32'], 's')]:
        if isotope in summary:
            delays = sorted(summary[isotope].keys())
            means = [summary[isotope][d]['strength']['mean'] for d in delays]
            sems = [summary[isotope][d]['strength']['sem'] for d in delays]
            
            ax2.errorbar(delays, means, yerr=sems, fmt=f'{marker}-', color=color,
                        capsize=3, markersize=8, linewidth=2, label=isotope)
    
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Dopamine Delay (s)')
    ax2.set_ylabel('Final Synaptic Strength')
    ax2.set_title('B. Functional Consequence', fontweight='bold')
    ax2.legend(loc='upper right')
    
    plt.suptitle('Isotope Substitution: The Definitive Quantum Test', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 5. PHARMACOLOGICAL DISSECTION FIGURE
# =============================================================================

def plot_pharmacology(result: ExperimentResult, save_path: Optional[str] = None):
    """
    Plot pharmacological dissection showing what each drug blocks
    """
    
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE['triple'])
    
    summary = result.summary_stats
    conditions = list(summary.keys())
    
    # Color coding
    colors = []
    for c in conditions:
        if c == 'Control':
            colors.append(COLORS['control'])
        elif 'APV' in c or 'Nocodazole' in c:
            colors.append(COLORS['blocked'])
        else:
            colors.append(COLORS['partial'])
    
    # Panel A: Q1 (EM Field)
    ax1 = axes[0]
    means = [summary[c]['field']['mean'] for c in conditions]
    sems = [summary[c]['field']['sem'] for c in conditions]
    
    bars = ax1.bar(range(len(conditions)), means, yerr=sems, color=colors, 
                   capsize=3, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels([c[:10] for c in conditions], rotation=45, ha='right')
    ax1.set_ylabel('Peak EM Field (kT)')
    ax1.set_title('A. Quantum System 1\n(Tryptophan)', fontweight='bold', color=COLORS['Q1'])
    ax1.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel B: Q2 (Dimers)
    ax2 = axes[1]
    means = [summary[c]['dimers']['mean'] for c in conditions]
    sems = [summary[c]['dimers']['sem'] for c in conditions]
    
    ax2.bar(range(len(conditions)), means, yerr=sems, color=colors,
            capsize=3, edgecolor='black', linewidth=1)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels([c[:10] for c in conditions], rotation=45, ha='right')
    ax2.set_ylabel('Peak Dimer Count')
    ax2.set_title('B. Quantum System 2\n(Dimers)', fontweight='bold', color=COLORS['Q2'])
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # Panel C: Classical Output
    ax3 = axes[2]
    means = [summary[c]['strength']['mean'] for c in conditions]
    sems = [summary[c]['strength']['sem'] for c in conditions]
    
    ax3.bar(range(len(conditions)), means, yerr=sems, color=colors,
            capsize=3, edgecolor='black', linewidth=1)
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels([c[:10] for c in conditions], rotation=45, ha='right')
    ax3.set_ylabel('Synaptic Strength')
    ax3.set_title('C. Classical Output\n(Plasticity)', fontweight='bold', color=COLORS['classical'])
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['control'], label='Control'),
        mpatches.Patch(color=COLORS['blocked'], label='Blocked'),
        mpatches.Patch(color=COLORS['partial'], label='Partial block'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Pharmacological Dissection of the Cascade', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 6. DOSE-RESPONSE FIGURE (Anesthetic)
# =============================================================================

def plot_dose_response(result: ExperimentResult, save_path: Optional[str] = None):
    """
    Plot anesthetic dose-response curve with IC50 fit
    """
    
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE['double'])
    
    summary = result.summary_stats
    concentrations = sorted([c for c in summary.keys() if isinstance(c, float)])
    
    # Panel A: Field vs concentration
    ax1 = axes[0]
    
    means = [summary[c]['field']['mean'] for c in concentrations]
    sems = [summary[c]['field']['sem'] for c in concentrations]
    x_pct = [c * 100 for c in concentrations]
    
    ax1.errorbar(x_pct, means, yerr=sems, fmt='o-', color=COLORS['Q1'],
                 capsize=3, markersize=8, linewidth=2)
    
    ax1.set_xlabel('Isoflurane Block (%)')
    ax1.set_ylabel('Peak EM Field (kT)')
    ax1.set_title('A. Q1 Suppression', fontweight='bold')
    ax1.set_xlim(-5, 105)
    
    # Panel B: Strength vs concentration (with IC50)
    ax2 = axes[1]
    
    means = [summary[c]['strength']['mean'] for c in concentrations]
    sems = [summary[c]['strength']['sem'] for c in concentrations]
    
    ax2.errorbar(x_pct, means, yerr=sems, fmt='s-', color=COLORS['classical'],
                 capsize=3, markersize=8, linewidth=2, label='Data')
    
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Isoflurane Block (%)')
    ax2.set_ylabel('Synaptic Strength')
    ax2.set_title('B. Plasticity Output', fontweight='bold')
    ax2.set_xlim(-5, 105)
    ax2.legend()
    
    plt.suptitle('Anesthetic Dose-Response: Selective Q1 Blockade', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 7. CASCADE FLOW / TIME COURSE FIGURE
# =============================================================================

def plot_cascade_flow(traces: List[SystemState], title: str = "", 
                      save_path: Optional[str] = None):
    """
    Plot time evolution of complete cascade
    
    Shows Q1 → Q2 → Classical flow over time
    """
    
    setup_style()
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    time = [s.time for s in traces]
    
    # Panel 1: Q1 (Tryptophan)
    ax = axes[0]
    ax.plot(time, [s.collective_field_kT for s in traces], '-', 
            color=COLORS['Q1'], linewidth=2, label='Collective Field')
    ax.plot(time, [s.k_enhancement for s in traces], '--',
            color=COLORS['Q1'], linewidth=1.5, alpha=0.7, label='k Enhancement')
    ax.set_ylabel('Q1: Tryptophan')
    ax.legend(loc='upper right')
    ax.set_title('Quantum System 1: Tryptophan Superradiance', fontweight='bold', color=COLORS['Q1'])
    ax.fill_between(time, 0, [s.collective_field_kT for s in traces], alpha=0.2, color=COLORS['Q1'])
    
    # Panel 2: Q2 (Dimers)
    ax = axes[1]
    ax.plot(time, [s.dimer_count for s in traces], '-',
            color=COLORS['Q2'], linewidth=2, label='Dimer Count')
    ax.plot(time, [s.eligibility * 50 for s in traces], '--',
            color=COLORS['Q2'], linewidth=1.5, alpha=0.7, label='Eligibility (×50)')
    ax.set_ylabel('Q2: Dimers')
    ax.legend(loc='upper right')
    ax.set_title('Quantum System 2: Calcium Phosphate Dimers', fontweight='bold', color=COLORS['Q2'])
    ax.fill_between(time, 0, [s.dimer_count for s in traces], alpha=0.2, color=COLORS['Q2'])
    
    # Panel 3: Plasticity Gate
    ax = axes[2]
    ax.plot(time, [s.committed_level for s in traces], '-',
            color='orange', linewidth=2, label='Committed Level')
    ax.plot(time, [s.camkii_pT286 for s in traces], '--',
            color='darkorange', linewidth=1.5, label='CaMKII pT286')
    
    # Highlight gate open periods
    gate_open = [1 if s.plasticity_gate else 0 for s in traces]
    ax.fill_between(time, 0, gate_open, alpha=0.2, color='yellow', label='Gate Open')
    
    ax.set_ylabel('Classical Bridge')
    ax.legend(loc='upper right')
    ax.set_title('Plasticity Gate & CaMKII', fontweight='bold', color='orange')
    ax.set_ylim(0, 1.1)
    
    # Panel 4: Classical Output
    ax = axes[3]
    ax.plot(time, [s.synaptic_strength for s in traces], '-',
            color=COLORS['classical'], linewidth=3, label='Synaptic Strength')
    ax.plot(time, [s.spine_volume for s in traces], '--',
            color=COLORS['classical'], linewidth=1.5, alpha=0.7, label='Spine Volume')
    
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Classical Output')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.set_title('Structural Plasticity: THE FUNCTIONAL OUTPUT', fontweight='bold', color=COLORS['classical'])
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# 8. SUMMARY DASHBOARD
# =============================================================================

def plot_summary_dashboard(all_results: Dict[str, ExperimentResult], 
                           save_path: Optional[str] = None):
    """
    Create comprehensive summary dashboard showing all experiments
    """
    
    setup_style()
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.35)
    
    # Row 1: Q1 Effects
    # A1: MT Invasion
    if 'mt_invasion' in all_results:
        ax = fig.add_subplot(gs[0, 0])
        summary = all_results['mt_invasion'].summary_stats
        labels = ['MT-', 'MT+']
        values = [summary[l]['field']['mean'] for l in labels]
        errors = [summary[l]['field']['sem'] for l in labels]
        ax.bar(labels, values, yerr=errors, color=[COLORS['blocked'], COLORS['control']], capsize=3)
        ax.set_ylabel('Peak Field (kT)')
        ax.set_title('Q1: MT Invasion', fontweight='bold')
    
    # A2: Anesthetic
    if 'anesthetic' in all_results:
        ax = fig.add_subplot(gs[0, 1])
        summary = all_results['anesthetic'].summary_stats
        concs = sorted([c for c in summary.keys() if isinstance(c, (int, float))])
        x = [c * 100 for c in concs]
        y = [summary[c]['field']['mean'] for c in concs]
        ax.plot(x, y, 'o-', color=COLORS['Q1'], markersize=6)
        ax.set_xlabel('Anesthetic (%)')
        ax.set_ylabel('Peak Field (kT)')
        ax.set_title('Q1: Anesthetic Block', fontweight='bold')
    
    # A3: UV Wavelength
    if 'uv_wavelength' in all_results:
        ax = fig.add_subplot(gs[0, 2])
        summary = all_results['uv_wavelength'].summary_stats
        labels = list(summary.keys())[:5]
        values = [summary[l]['field']['mean'] for l in labels]
        ax.bar(range(len(labels)), values, color=COLORS['Q1'])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l[:8] for l in labels], rotation=45, ha='right')
        ax.set_ylabel('Peak Field (kT)')
        ax.set_title('Q1: UV Wavelength', fontweight='bold')
    
    # A4: Temperature
    if 'temperature' in all_results:
        ax = fig.add_subplot(gs[0, 3])
        summary = all_results['temperature'].summary_stats
        temps = sorted(summary.keys())
        y = [summary[t]['coherence']['mean'] for t in temps]
        ax.plot(temps, y, 's-', color=COLORS['Q2'], markersize=6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Peak Coherence')
        ax.set_title('Q2: Temperature', fontweight='bold')
    
    # Row 2: Q2 Effects
    # B1: Network threshold
    if 'network_threshold' in all_results:
        ax = fig.add_subplot(gs[1, 0])
        summary = all_results['network_threshold'].summary_stats
        n_vals = sorted(summary.keys())
        y = [summary[n]['dimers']['mean'] for n in n_vals]
        ax.plot(n_vals, y, 'o-', color=COLORS['Q2'], markersize=6)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('N Synapses')
        ax.set_ylabel('Peak Dimers')
        ax.set_title('Q2: Network Threshold', fontweight='bold')
    
    # B2: Stim intensity
    if 'stim_intensity' in all_results:
        ax = fig.add_subplot(gs[1, 1])
        summary = all_results['stim_intensity'].summary_stats
        voltages = sorted(summary.keys())
        y = [summary[v]['dimers']['mean'] for v in voltages]
        ax.plot(voltages, y, 's-', color=COLORS['Q2'], markersize=6)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Peak Dimers')
        ax.set_title('Q2: Stim Intensity', fontweight='bold')
    
    # B3: Dopamine timing
    if 'dopamine_timing' in all_results:
        ax = fig.add_subplot(gs[1, 2])
        summary = all_results['dopamine_timing'].summary_stats
        delays = sorted(summary.keys())
        y = [summary[d]['pre_da_eligibility']['mean'] for d in delays]
        ax.plot(delays, y, 'o-', color=COLORS['Q2'], markersize=6, label='Data')
        # Theory
        t = np.linspace(0, max(delays), 50)
        ax.plot(t, np.exp(-t/67), '--', color='gray', alpha=0.5, label='T₂=67s')
        ax.set_xlabel('DA Delay (s)')
        ax.set_ylabel('Eligibility')
        ax.set_title('Q2: Eligibility Decay', fontweight='bold')
        ax.legend(fontsize=8)
    
    # B4: Isotope
    if 'isotope' in all_results:
        ax = fig.add_subplot(gs[1, 3])
        summary = all_results['isotope'].summary_stats
        for iso, color in [('P31', COLORS['P31']), ('P32', COLORS['P32'])]:
            if iso in summary:
                delays = sorted(summary[iso].keys())
                y = [summary[iso][d]['pre_da_eligibility']['mean'] for d in delays]
                ax.plot(delays, y, 'o-', color=color, markersize=6, label=iso)
        ax.set_xlabel('DA Delay (s)')
        ax.set_ylabel('Eligibility')
        ax.set_title('Isotope Effect', fontweight='bold')
        ax.legend(fontsize=8)
    
    # Row 3: Classical Output
    # C1: Network → Strength
    if 'network_threshold' in all_results:
        ax = fig.add_subplot(gs[2, 0])
        summary = all_results['network_threshold'].summary_stats
        n_vals = sorted(summary.keys())
        y = [summary[n]['strength']['mean'] for n in n_vals]
        ax.plot(n_vals, y, 'o-', color=COLORS['classical'], markersize=6)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('N Synapses')
        ax.set_ylabel('Synaptic Strength')
        ax.set_title('Classical: Network→Strength', fontweight='bold')
    
    # C2: Timing → Strength
    if 'dopamine_timing' in all_results:
        ax = fig.add_subplot(gs[2, 1])
        summary = all_results['dopamine_timing'].summary_stats
        delays = sorted(summary.keys())
        y = [summary[d]['strength']['mean'] for d in delays]
        ax.plot(delays, y, 's-', color=COLORS['classical'], markersize=6)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('DA Delay (s)')
        ax.set_ylabel('Synaptic Strength')
        ax.set_title('Classical: Timing→Strength', fontweight='bold')
    
    # C3: Pharmacology
    if 'pharmacology' in all_results:
        ax = fig.add_subplot(gs[2, 2])
        summary = all_results['pharmacology'].summary_stats
        conds = list(summary.keys())
        values = [summary[c]['strength']['mean'] for c in conds]
        colors = [COLORS['control'] if c == 'Control' else COLORS['blocked'] for c in conds]
        ax.bar(range(len(conds)), values, color=colors)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[:8] for c in conds], rotation=45, ha='right')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('Synaptic Strength')
        ax.set_title('Classical: Pharmacology', fontweight='bold')
    
    # C4: Three-factor gate
    if 'three_factor_gate' in all_results:
        ax = fig.add_subplot(gs[2, 3])
        summary = all_results['three_factor_gate'].summary_stats
        conds = list(summary.keys())
        values = [summary[c]['strength']['mean'] for c in conds]
        colors = [COLORS['control'] if 'Full' in c else COLORS['blocked'] for c in conds]
        ax.bar(range(len(conds)), values, color=colors)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c[:12] for c in conds], rotation=45, ha='right')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel('Synaptic Strength')
        ax.set_title('Three-Factor Gate', fontweight='bold')
    
    plt.suptitle('Quantum-Classical Cascade: Complete Characterization\n'
                 'Row 1: Q1 (Tryptophan) | Row 2: Q2 (Dimers) | Row 3: Classical Output',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================

def generate_all_figures(all_results: Dict[str, ExperimentResult],
                         output_dir: str = "figures"):
    """
    Generate all figures for the experiment suite
    """
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating figures...")
    
    # 1. Architecture diagram
    plot_system_architecture(path / f"01_architecture_{timestamp}.png")
    
    # 2. Individual experiment figures
    if 'network_threshold' in all_results:
        plot_network_threshold(all_results['network_threshold'], 
                              path / f"02_network_threshold_{timestamp}.png")
    
    if 'dopamine_timing' in all_results:
        plot_dopamine_timing(all_results['dopamine_timing'],
                            path / f"03_dopamine_timing_{timestamp}.png")
    
    if 'isotope' in all_results:
        plot_isotope_comparison(all_results['isotope'],
                               path / f"04_isotope_{timestamp}.png")
    
    if 'pharmacology' in all_results:
        plot_pharmacology(all_results['pharmacology'],
                         path / f"05_pharmacology_{timestamp}.png")
    
    if 'anesthetic' in all_results:
        plot_dose_response(all_results['anesthetic'],
                          path / f"06_dose_response_{timestamp}.png")
    
    # 3. Summary dashboard
    plot_summary_dashboard(all_results, path / f"07_summary_dashboard_{timestamp}.png")
    
    print(f"\nAll figures saved to: {output_dir}/")
    
    return path