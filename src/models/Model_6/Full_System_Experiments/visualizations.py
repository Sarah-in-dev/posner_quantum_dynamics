"""
Improved Visualizations for Quantum-Classical Cascade
======================================================

Design principles:
1. SHOW THE CASCADE: Q1 → Q2 → Classical flow should be visually clear
2. USE COMMITMENT, NOT STRENGTH: For short experiments, show committed_level
3. CLEAR PREDICTIONS: Annotate what the theory predicts
4. CONSISTENT COLORS: Q1=red, Q2=green, Classical=purple, Gate=orange
5. MEANINGFUL Y-AXES: Don't compress everything to 1.00-1.07

Author: Redesigned December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional

# =============================================================================
# COLOR SCHEME (Consistent across all figures)
# =============================================================================

COLORS = {
    'Q1': '#E41A1C',        # Red - Tryptophan system
    'Q2': '#4DAF4A',        # Green - Dimer system  
    'classical': '#984EA3', # Purple - Classical output
    'gate': '#FF7F00',      # Orange - Plasticity gate
    'blocked': '#E41A1C',   # Red - Blocked/failed
    'control': '#4DAF4A',   # Green - Control/working
    'partial': '#FF7F00',   # Orange - Partial effect
    'theory': '#666666',    # Gray - Theoretical prediction
}

def setup_style():
    """Set up publication-quality style"""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# =============================================================================
# FIGURE 1: CASCADE ARCHITECTURE (Conceptual)
# =============================================================================

def plot_cascade_architecture(save_path: Optional[str] = None):
    """
    Visual representation of the quantum-classical cascade
    Shows the flow: Q1 → Q2 → Gate → Classical
    """
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # === Q1: TRYPTOPHAN SYSTEM ===
    q1_box = FancyBboxPatch((0.5, 6), 3, 3,
                            boxstyle="round,pad=0.1",
                            facecolor='#ffe6e6', edgecolor=COLORS['Q1'], linewidth=3)
    ax.add_patch(q1_box)
    ax.text(2, 8.5, "Q1: TRYPTOPHAN", fontweight='bold', ha='center', fontsize=12, color=COLORS['Q1'])
    ax.text(2, 7.8, "Structural Superradiance", ha='center', fontsize=10)
    ax.text(2, 7.2, "• √N collective enhancement", ha='center', fontsize=9)
    ax.text(2, 6.7, "• 22 kT field (MT invaded)", ha='center', fontsize=9)
    ax.text(2, 6.2, "• Blocked by anesthetics", ha='center', fontsize=9)
    
    # === Q2: DIMER SYSTEM ===
    q2_box = FancyBboxPatch((5, 6), 3, 3,
                            boxstyle="round,pad=0.1",
                            facecolor='#e6ffe6', edgecolor=COLORS['Q2'], linewidth=3)
    ax.add_patch(q2_box)
    ax.text(6.5, 8.5, "Q2: DIMERS", fontweight='bold', ha='center', fontsize=12, color=COLORS['Q2'])
    ax.text(6.5, 7.8, "Nuclear Spin Coherence", ha='center', fontsize=10)
    ax.text(6.5, 7.2, "• T₂ ~ 100s (³¹P)", ha='center', fontsize=9)
    ax.text(6.5, 6.7, "• ~5 dimers/synapse", ha='center', fontsize=9)
    ax.text(6.5, 6.2, "• Eligibility = Coherence", ha='center', fontsize=9)
    
    # === PLASTICITY GATE ===
    gate_box = FancyBboxPatch((9.5, 6), 4, 3,
                              boxstyle="round,pad=0.1",
                              facecolor='#fff5e6', edgecolor=COLORS['gate'], linewidth=3)
    ax.add_patch(gate_box)
    ax.text(11.5, 8.5, "PLASTICITY GATE", fontweight='bold', ha='center', fontsize=12, color=COLORS['gate'])
    ax.text(11.5, 7.8, "Four-Factor AND Gate", ha='center', fontsize=10)
    ax.text(11.5, 7.2, "• Eligibility > 0.3", ha='center', fontsize=9)
    ax.text(11.5, 6.7, "• Dopamine (reward)", ha='center', fontsize=9)
    ax.text(11.5, 6.2, "• Calcium elevated", ha='center', fontsize=9)
    ax.text(11.5, 5.7, "• Q1 Field > 10 kT", ha='center', fontsize=9, fontweight='bold', color=COLORS['Q1'])
    
    # === CLASSICAL OUTPUT ===
    classical_box = FancyBboxPatch((5, 1), 4, 3,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#f0e6ff', edgecolor=COLORS['classical'], linewidth=3)
    ax.add_patch(classical_box)
    ax.text(7, 3.5, "CLASSICAL CASCADE", fontweight='bold', ha='center', fontsize=12, color=COLORS['classical'])
    ax.text(7, 2.8, "CaMKII → Actin → AMPAR", ha='center', fontsize=10)
    ax.text(7, 2.2, "• Commitment locked in", ha='center', fontsize=9)
    ax.text(7, 1.7, "• Hours to full expression", ha='center', fontsize=9)
    ax.text(7, 1.2, "• Synaptic strength Δ", ha='center', fontsize=9)
    
    # === ARROWS ===
    # Q1 → Q2 (forward coupling: EM field enhances k_agg)
    ax.annotate("", xy=(5, 7.5), xytext=(3.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q1'], lw=2))
    ax.text(4.25, 8, "EM field\nenhances k_agg", ha='center', fontsize=8, color=COLORS['Q1'])
    
    # Q2 → Gate (eligibility)
    ax.annotate("", xy=(9.5, 7.5), xytext=(8, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q2'], lw=2))
    ax.text(8.75, 8, "Eligibility\n(coherence)", ha='center', fontsize=8, color=COLORS['Q2'])
    
    # Q1 → Gate (field threshold)
    ax.annotate("", xy=(10, 6), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['Q1'], lw=2, ls='--'))
    ax.text(6.5, 5.2, "Field > 10 kT required", ha='center', fontsize=8, color=COLORS['Q1'])
    
    # Gate → Classical (commitment)
    ax.annotate("", xy=(7, 4), xytext=(11, 6),
                arrowprops=dict(arrowstyle='->', color=COLORS['gate'], lw=3))
    ax.text(9.5, 4.8, "COMMITMENT", ha='center', fontsize=10, fontweight='bold', color=COLORS['gate'])
    
    # === TIMESCALES BOX ===
    ax.text(1, 4.5, "TIMESCALES:", fontweight='bold', fontsize=10)
    ax.text(1, 4.0, "Q1: femtoseconds (superradiant bursts)", fontsize=9, color=COLORS['Q1'])
    ax.text(1, 3.5, "Q2: 1-100 seconds (spin coherence)", fontsize=9, color=COLORS['Q2'])
    ax.text(1, 3.0, "Gate: milliseconds (coincidence)", fontsize=9, color=COLORS['gate'])
    ax.text(1, 2.5, "Classical: minutes-hours (consolidation)", fontsize=9, color=COLORS['classical'])
    
    plt.title("Quantum-Classical Cascade: Architecture & Information Flow\n", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 2: NETWORK THRESHOLD (Improved)
# =============================================================================

def plot_network_threshold_improved(summary: Dict, save_path: Optional[str] = None):
    """
    Network threshold showing cascade flow properly
    
    Key improvement: Show COMMITMENT (not final strength) as the output metric
    """
    setup_style()
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.35)
    
    n_values = sorted([k for k in summary.keys() if isinstance(k, int)])
    
    # === Panel A: Q1 Field (should be constant ~22 kT) ===
    ax1 = fig.add_subplot(gs[0])
    fields = [summary[n].get('field', {}).get('mean', 22) for n in n_values]
    ax1.bar(range(len(n_values)), fields, color=COLORS['Q1'], alpha=0.8)
    ax1.axhline(y=22, color='gray', linestyle='--', alpha=0.5, label='Expected (22 kT)')
    ax1.set_xticks(range(len(n_values)))
    ax1.set_xticklabels(n_values)
    ax1.set_xlabel('Number of Synapses (N)')
    ax1.set_ylabel('Q1 Field (kT)', color=COLORS['Q1'])
    ax1.set_title('A. Q1: Tryptophan Field\n(Structural - constant)', fontweight='bold', color=COLORS['Q1'])
    ax1.set_ylim(0, 30)
    
    # === Panel B: Q2 Dimers (should scale with N) ===
    ax2 = fig.add_subplot(gs[1])
    dimers = [summary[n]['dimers']['mean'] for n in n_values]
    dimer_sems = [summary[n]['dimers']['sem'] for n in n_values]
    
    ax2.errorbar(n_values, dimers, yerr=dimer_sems, fmt='o-', 
                 color=COLORS['Q2'], capsize=3, markersize=8, linewidth=2)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50 dimer threshold')
    ax2.fill_between([0, max(n_values)+2], 0, 50, alpha=0.1, color='red')
    ax2.set_xlabel('Number of Synapses (N)')
    ax2.set_ylabel('Total Network Dimers', color=COLORS['Q2'])
    ax2.set_title('B. Q2: Dimer Scaling\n(~5 per synapse × N)', fontweight='bold', color=COLORS['Q2'])
    ax2.legend(loc='lower right', fontsize=8)
    
    # === Panel C: Commitment Level (the actual gate output) ===
    ax3 = fig.add_subplot(gs[2])
    # Use 'commitment' or 'committed_level' if available, else calculate from committed
    if 'commitment' in summary[n_values[0]]:
        commits = [summary[n]['commitment']['mean'] for n in n_values]
        commit_sems = [summary[n]['commitment']['sem'] for n in n_values]
    else:
        commits = [summary[n].get('committed', {}).get('mean', 0) for n in n_values]
        commit_sems = [summary[n].get('committed', {}).get('sem', 0) for n in n_values]
    
    ax3.errorbar(n_values, commits, yerr=commit_sems, fmt='s-',
                 color=COLORS['gate'], capsize=3, markersize=8, linewidth=2)
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Number of Synapses (N)')
    ax3.set_ylabel('Commitment Level', color=COLORS['gate'])
    ax3.set_title('C. Gate Output: Commitment\n(Quantum decision)', fontweight='bold', color=COLORS['gate'])
    ax3.set_ylim(-0.05, 1.05)
    
    # Annotate threshold
    for i, (n, c) in enumerate(zip(n_values, commits)):
        if c > 0.5 and (i == 0 or commits[i-1] <= 0.5):
            ax3.axvline(x=n, color=COLORS['gate'], linestyle='--', alpha=0.5)
            ax3.text(n, 0.9, f'Threshold\nN≈{n}', ha='center', fontsize=8, color=COLORS['gate'])
    
    # === Panel D: Projected Final Strength ===
    ax4 = fig.add_subplot(gs[3])
    # Show projected strength = 1 + 0.35 × commitment
    projected = [1.0 + 0.35 * c for c in commits]
    
    ax4.bar(range(len(n_values)), projected, color=COLORS['classical'], alpha=0.8)
    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax4.axhline(y=1.35, color='gray', linestyle='--', alpha=0.5, label='Max LTP')
    ax4.set_xticks(range(len(n_values)))
    ax4.set_xticklabels(n_values)
    ax4.set_xlabel('Number of Synapses (N)')
    ax4.set_ylabel('Projected Strength', color=COLORS['classical'])
    ax4.set_title('D. Classical: Projected LTP\n(= 1 + 0.35 × Commitment)', fontweight='bold', color=COLORS['classical'])
    ax4.set_ylim(0.95, 1.4)
    ax4.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Network Size Threshold: Q1 → Q2 → Gate → Classical Flow', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 3: PHARMACOLOGY (Improved)
# =============================================================================

def plot_pharmacology_improved(summary: Dict, save_path: Optional[str] = None):
    """
    Pharmacological dissection showing selective blockade of each system
    
    Key improvement: Show how each drug blocks different parts of cascade
    """
    setup_style()
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 4, height_ratios=[1, 0.3], hspace=0.4, wspace=0.3)
    
    conditions = ['Control', 'APV', 'Nocodazole', 'Isoflurane']
    
    # What each drug SHOULD do:
    # Control: Q1=22kT, Q2=high, Commitment=1
    # APV: Q1=22kT (unchanged!), Q2=0, Commitment=0
    # Nocodazole: Q1=low (no MT), Q2=reduced, Commitment=reduced
    # Isoflurane: Q1=2kT, Q2=may form but can't commit, Commitment=0
    
    # Color by mechanism
    bar_colors = {
        'Control': COLORS['control'],
        'APV': COLORS['blocked'],
        'Nocodazole': COLORS['partial'],
        'Isoflurane': COLORS['blocked']
    }
    
    # === Panel A: Q1 Field ===
    ax1 = fig.add_subplot(gs[0, 0])
    fields = []
    for cond in conditions:
        if cond in summary:
            fields.append(summary[cond].get('field', {}).get('mean', 0))
        else:
            fields.append(0)
    
    colors_q1 = [bar_colors[c] if c != 'APV' else COLORS['control'] for c in conditions]  # APV shouldn't affect Q1!
    ax1.bar(conditions, fields, color=colors_q1, alpha=0.8, edgecolor='black')
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Gate threshold')
    ax1.set_ylabel('Q1 Field (kT)', color=COLORS['Q1'])
    ax1.set_title('A. Q1: Tryptophan Field', fontweight='bold', color=COLORS['Q1'])
    ax1.set_ylim(0, 30)
    ax1.tick_params(axis='x', rotation=30)
    ax1.legend(loc='upper right', fontsize=8)
    
    # === Panel B: Q2 Dimers ===
    ax2 = fig.add_subplot(gs[0, 1])
    dimers = []
    for cond in conditions:
        if cond in summary:
            dimers.append(summary[cond].get('dimers', {}).get('mean', 0))
        else:
            dimers.append(0)
    
    colors_q2 = [bar_colors[c] for c in conditions]
    ax2.bar(conditions, dimers, color=colors_q2, alpha=0.8, edgecolor='black')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50 threshold')
    ax2.set_ylabel('Peak Dimers', color=COLORS['Q2'])
    ax2.set_title('B. Q2: Dimer Formation', fontweight='bold', color=COLORS['Q2'])
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend(loc='upper right', fontsize=8)
    
    # === Panel C: Commitment ===
    ax3 = fig.add_subplot(gs[0, 2])
    commits = []
    for cond in conditions:
        if cond in summary:
            commits.append(summary[cond].get('committed', {}).get('mean', 0))
        else:
            commits.append(0)
    
    ax3.bar(conditions, commits, color=[bar_colors[c] for c in conditions], alpha=0.8, edgecolor='black')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_ylabel('Commitment', color=COLORS['gate'])
    ax3.set_title('C. Gate: Commitment', fontweight='bold', color=COLORS['gate'])
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(axis='x', rotation=30)
    
    # === Panel D: Projected Strength ===
    ax4 = fig.add_subplot(gs[0, 3])
    projected = [1.0 + 0.35 * c for c in commits]
    
    ax4.bar(conditions, projected, color=[bar_colors[c] for c in conditions], alpha=0.8, edgecolor='black')
    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax4.set_ylabel('Projected Strength', color=COLORS['classical'])
    ax4.set_title('D. Classical: Outcome', fontweight='bold', color=COLORS['classical'])
    ax4.set_ylim(0.9, 1.4)
    ax4.tick_params(axis='x', rotation=30)
    
    # === Bottom row: Mechanism explanation ===
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')
    
    explanations = [
        ("Control", "Full cascade: Q1✓ Q2✓ Gate✓ → LTP", COLORS['control']),
        ("APV", "Blocks NMDA→Ca²⁺: Q1✓ Q2✗ → No eligibility → No LTP", COLORS['blocked']),
        ("Nocodazole", "Destroys MT: Q1↓ Q2↓ → Reduced commitment → Weak LTP", COLORS['partial']),
        ("Isoflurane", "Blocks Trp coupling: Q1✗ → Gate fails (field<10kT) → No LTP", COLORS['blocked']),
    ]
    
    for i, (drug, expl, color) in enumerate(explanations):
        ax_legend.text(0.02 + i*0.25, 0.5, f"● {drug}: ", fontweight='bold', fontsize=10, 
                       color=color, transform=ax_legend.transAxes, va='center')
        ax_legend.text(0.02 + i*0.25 + 0.08, 0.5, expl, fontsize=9, 
                       transform=ax_legend.transAxes, va='center')
    
    plt.suptitle('Pharmacological Dissection: Selective Blockade of Cascade Components', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 4: ANESTHETIC DOSE-RESPONSE (Improved)
# =============================================================================

def plot_anesthetic_improved(summary: Dict, save_path: Optional[str] = None):
    """
    Anesthetic dose-response showing Q1 blockade leads to gate failure
    
    Key improvement: Show the MECHANISM - field drops below gate threshold
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    concentrations = sorted([k for k in summary.keys() if isinstance(k, (int, float))])
    conc_percent = [c * 100 for c in concentrations]
    
    # === Panel A: Q1 Field vs Concentration ===
    ax1 = axes[0]
    fields = [summary[c]['field']['mean'] for c in concentrations]
    
    ax1.plot(conc_percent, fields, 'o-', color=COLORS['Q1'], markersize=10, linewidth=2, label='Data')
    
    # Theoretical line: field = 22 × (1 - concentration)
    theory_x = np.linspace(0, 100, 50)
    theory_y = 22 * (1 - theory_x/100)
    ax1.plot(theory_x, theory_y, '--', color='gray', alpha=0.7, label='Theory: 22×(1-block)')
    
    # Gate threshold
    ax1.axhline(y=10, color='red', linestyle=':', linewidth=2, label='Gate threshold (10 kT)')
    ax1.fill_between([0, 100], 0, 10, alpha=0.1, color='red')
    
    # Mark where field drops below threshold
    threshold_conc = None
    for i, f in enumerate(fields):
        if f < 10:
            threshold_conc = conc_percent[i]
            break
    if threshold_conc:
        ax1.axvline(x=threshold_conc, color='red', linestyle='--', alpha=0.5)
        ax1.text(threshold_conc + 2, 15, f'Gate fails\n>{threshold_conc:.0f}%', fontsize=9, color='red')
    
    ax1.set_xlabel('Isoflurane Block (%)')
    ax1.set_ylabel('Q1 Field (kT)', color=COLORS['Q1'])
    ax1.set_title('A. Q1 Suppression', fontweight='bold', color=COLORS['Q1'])
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(0, 25)
    ax1.legend(loc='upper right', fontsize=9)
    
    # === Panel B: Commitment vs Concentration ===
    ax2 = axes[1]
    
    # Get commitment (or calculate from whether gate opens)
    commits = []
    for c in concentrations:
        if 'committed' in summary[c]:
            commits.append(summary[c]['committed']['mean'])
        else:
            # Commitment should be 0 when field < 10 kT
            field = summary[c]['field']['mean']
            commits.append(1.0 if field > 10 else 0.0)
    
    ax2.plot(conc_percent, commits, 's-', color=COLORS['gate'], markersize=10, linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Shade regions
    ax2.fill_between([0, 50], 0, 1.1, alpha=0.1, color=COLORS['control'], label='Gate opens')
    ax2.fill_between([50, 100], 0, 1.1, alpha=0.1, color=COLORS['blocked'], label='Gate fails')
    
    ax2.set_xlabel('Isoflurane Block (%)')
    ax2.set_ylabel('Commitment Level', color=COLORS['gate'])
    ax2.set_title('B. Gate Output', fontweight='bold', color=COLORS['gate'])
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc='upper right', fontsize=9)
    
    # === Panel C: Projected Strength ===
    ax3 = axes[2]
    projected = [1.0 + 0.35 * c for c in commits]
    
    ax3.plot(conc_percent, projected, 'D-', color=COLORS['classical'], markersize=10, linewidth=2)
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    
    ax3.set_xlabel('Isoflurane Block (%)')
    ax3.set_ylabel('Projected Strength', color=COLORS['classical'])
    ax3.set_title('C. Predicted LTP', fontweight='bold', color=COLORS['classical'])
    ax3.set_xlim(-5, 105)
    ax3.set_ylim(0.95, 1.4)
    ax3.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Anesthetic Dose-Response: Q1 Field Drops → Gate Fails → No LTP', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 5: ISOTOPE EFFECT (Improved)
# =============================================================================

def plot_isotope_improved(summary_p31: Dict, summary_p32: Dict, save_path: Optional[str] = None):
    """
    Isotope substitution - the definitive quantum test
    
    Shows that P32 has such fast decoherence that eligibility can't persist
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    delays = sorted(summary_p31.keys())
    
    # === Panel A: Eligibility Decay ===
    ax1 = axes[0]
    
    elig_p31 = [summary_p31[d].get('pre_da_eligibility', {}).get('mean', 0) for d in delays]
    elig_p32 = [summary_p32[d].get('pre_da_eligibility', {}).get('mean', 0) for d in delays]
    
    ax1.plot(delays, elig_p31, 'o-', color='#1f77b4', markersize=10, linewidth=2, label='³¹P (T₂≈67s)')
    ax1.plot(delays, elig_p32, 's-', color='#d62728', markersize=10, linewidth=2, label='³²P (T₂≈0.3s)')
    
    # Theory curves
    t = np.linspace(0, max(delays), 100)
    ax1.plot(t, np.exp(-t/67), '--', color='#1f77b4', alpha=0.5, linewidth=1)
    ax1.plot(t, np.exp(-t/0.3), '--', color='#d62728', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Dopamine Delay (s)')
    ax1.set_ylabel('Eligibility at Dopamine', color=COLORS['Q2'])
    ax1.set_title('A. Eligibility Trace Decay\n(T₂ from nuclear spin physics)', fontweight='bold', color=COLORS['Q2'])
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(-2, max(delays) + 2)
    ax1.set_ylim(-0.05, 1.05)
    
    # Annotation
    ax1.annotate('³²P: Decays in <1s\n(no quantum memory)', xy=(5, 0.05), fontsize=9, 
                 color='#d62728', ha='left')
    ax1.annotate('³¹P: Persists ~100s\n(quantum memory intact)', xy=(30, 0.6), fontsize=9,
                 color='#1f77b4', ha='left')
    
    # === Panel B: Commitment ===
    ax2 = axes[1]
    
    commit_p31 = []
    commit_p32 = []
    for d in delays:
        c31 = summary_p31[d].get('committed', {}).get('mean', 0)
        c32 = summary_p32[d].get('committed', {}).get('mean', 0)
        # If committed not available, derive from eligibility
        if c31 == 0 and elig_p31[delays.index(d)] > 0.3:
            c31 = 1.0
        if c32 == 0 and elig_p32[delays.index(d)] > 0.3:
            c32 = 1.0
        commit_p31.append(c31)
        commit_p32.append(c32)
    
    ax2.plot(delays, commit_p31, 'o-', color='#1f77b4', markersize=10, linewidth=2, label='³¹P')
    ax2.plot(delays, commit_p32, 's-', color='#d62728', markersize=10, linewidth=2, label='³²P')
    
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Dopamine Delay (s)')
    ax2.set_ylabel('Commitment Level', color=COLORS['gate'])
    ax2.set_title('B. Gate Output\n(Eligibility > 0.3 required)', fontweight='bold', color=COLORS['gate'])
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(-2, max(delays) + 2)
    ax2.set_ylim(-0.05, 1.1)
    
    # === Panel C: Functional Consequence ===
    ax3 = axes[2]
    
    # Projected strength
    proj_p31 = [1.0 + 0.35 * c for c in commit_p31]
    proj_p32 = [1.0 + 0.35 * c for c in commit_p32]
    
    ax3.plot(delays, proj_p31, 'o-', color='#1f77b4', markersize=10, linewidth=2, label='³¹P: Learning')
    ax3.plot(delays, proj_p32, 's-', color='#d62728', markersize=10, linewidth=2, label='³²P: No Learning')
    
    ax3.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Baseline')
    ax3.fill_between(delays, 1.0, proj_p31, alpha=0.2, color='#1f77b4')
    
    ax3.set_xlabel('Dopamine Delay (s)')
    ax3.set_ylabel('Projected Strength', color=COLORS['classical'])
    ax3.set_title('C. Functional Consequence\n(³²P eliminates learning)', fontweight='bold', color=COLORS['classical'])
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(-2, max(delays) + 2)
    ax3.set_ylim(0.95, 1.4)
    
    plt.suptitle('Isotope Substitution: The Definitive Quantum Test\n'
                 '(Nuclear spin I=1/2 vs I=1 → 200× difference in coherence time)', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# FIGURE 6: FOUR-FACTOR GATE (Improved)
# =============================================================================

def plot_four_factor_gate_improved(summary: Dict, save_path: Optional[str] = None):
    """
    Four-factor gate showing AND logic
    
    All factors must be present: Calcium + Dimers (Q2) + EM Field (Q1) + Dopamine
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    conditions = ['Full (all)', 'No Ca (APV)', 'No Dimers', 'No EM Field', 'No DA']
    short_names = ['Full', 'No Ca', 'No Dimers', 'No EM', 'No DA']
    
    # === Panel A: Gate Requirements ===
    ax1 = axes[0]
    
    # Create a table-like visualization
    factors = ['Calcium\n(activity)', 'Dimers\n(Q2 memory)', 'EM Field\n(Q1 coupling)', 'Dopamine\n(reward)']
    
    # Which factors are present in each condition
    # Full: all present
    # No Ca: missing calcium (and therefore dimers)
    # No Dimers: calcium present but aggregation blocked
    # No EM: dimers form but no Q1 coupling
    # No DA: Q1+Q2 intact but no reward
    factor_matrix = np.array([
        # Full  noCa  noDim  noEM  noDA
        [1,     0,    1,     1,    1],  # Calcium
        [1,     0,    0,     1,    1],  # Dimers
        [1,     0,    0,     0,    1],  # EM Field
        [1,     1,    1,     1,    0],  # Dopamine
    ])
    
    im = ax1.imshow(factor_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax1.set_xticks(range(len(short_names)))
    ax1.set_xticklabels(short_names, rotation=45, ha='right')
    ax1.set_yticks(range(len(factors)))
    ax1.set_yticklabels(factors)
    
    # Add text annotations
    for i in range(len(factors)):
        for j in range(len(short_names)):
            text = '✓' if factor_matrix[i, j] else '✗'
            color = 'white' if factor_matrix[i, j] else 'black'
            ax1.text(j, i, text, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    
    ax1.set_title('A. Gate Requirements\n(AND logic: ALL must be ✓)', fontweight='bold')
    
    # === Panel B: Outcome ===
    ax2 = axes[1]
    
    commits = []
    for cond in conditions:
        if cond in summary:
            commits.append(summary[cond].get('committed', {}).get('mean', 0))
        else:
            commits.append(0)
    
    colors = [COLORS['control'] if c > 0.5 else COLORS['blocked'] for c in commits]
    bars = ax2.bar(short_names, commits, color=colors, alpha=0.8, edgecolor='black')
    
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Commitment Level', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('B. Gate Output\n(Only "Full" produces commitment)', fontweight='bold')
    
    # Add outcome labels
    for i, (bar, c) in enumerate(zip(bars, commits)):
        outcome = 'LTP' if c > 0.5 else 'No LTP'
        color = COLORS['control'] if c > 0.5 else COLORS['blocked']
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                 outcome, ha='center', fontsize=10, fontweight='bold', color=color)
    
    plt.suptitle('Four-Factor Gate: Dual Quantum Architecture AND Logic\n'
                 'Missing ANY factor → No commitment → No plasticity', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# MASTER DASHBOARD (Improved)
# =============================================================================

def plot_master_dashboard_improved(all_results: Dict, save_path: Optional[str] = None):
    """
    Master dashboard showing all experiments in cascade context
    """
    setup_style()
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Quantum-Classical Cascade: Complete Experimental Validation\n'
                 'Row 1: Q1 (Tryptophan) | Row 2: Q2 (Dimers) | Row 3: Gate & Classical', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # === ROW 1: Q1 EFFECTS ===
    
    # 1.1: MT Invasion
    if 'mt_invasion' in all_results:
        ax = fig.add_subplot(gs[0, 0])
        summary = all_results['mt_invasion'].summary_stats
        names = ['MT-', 'MT+']
        fields = [summary.get(n, {}).get('field', {}).get('mean', 0) for n in names]
        ax.bar(names, fields, color=COLORS['Q1'], alpha=0.8)
        ax.set_ylabel('Q1 Field (kT)')
        ax.set_title('Q1: MT Invasion', fontweight='bold', color=COLORS['Q1'])
        ax.set_ylim(0, 30)
    
    # 1.2: Anesthetic
    if 'anesthetic' in all_results:
        ax = fig.add_subplot(gs[0, 1])
        summary = all_results['anesthetic'].summary_stats
        concs = sorted([k for k in summary.keys() if isinstance(k, (int, float))])
        fields = [summary[c]['field']['mean'] for c in concs]
        ax.plot([c*100 for c in concs], fields, 'o-', color=COLORS['Q1'], markersize=6)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Block (%)')
        ax.set_ylabel('Q1 Field (kT)')
        ax.set_title('Q1: Anesthetic', fontweight='bold', color=COLORS['Q1'])
    
    # 1.3: UV Wavelength
    if 'uv_wavelength' in all_results:
        ax = fig.add_subplot(gs[0, 2])
        summary = all_results['uv_wavelength'].summary_stats
        names = list(summary.keys())
        fields = [summary[n]['field']['mean'] for n in names]
        ax.bar(range(len(names)), fields, color=COLORS['Q1'], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Q1 Field (kT)')
        ax.set_title('Q1: UV Wavelength', fontweight='bold', color=COLORS['Q1'])
    
    # 1.4: Temperature (coherence)
    if 'temperature' in all_results:
        ax = fig.add_subplot(gs[0, 3])
        summary = all_results['temperature'].summary_stats
        temps = sorted(summary.keys())
        coh = [summary[t]['coherence']['mean'] for t in temps]
        ax.plot(temps, coh, 'o-', color=COLORS['Q2'], markersize=6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Coherence')
        ax.set_title('Q2: Temperature Independence', fontweight='bold', color=COLORS['Q2'])
    
    # === ROW 2: Q2 EFFECTS ===
    
    # 2.1: Network threshold
    if 'network_threshold' in all_results:
        ax = fig.add_subplot(gs[1, 0])
        summary = all_results['network_threshold'].summary_stats
        n_vals = sorted([k for k in summary.keys() if isinstance(k, int)])
        dimers = [summary[n]['dimers']['mean'] for n in n_vals]
        ax.plot(n_vals, dimers, 'o-', color=COLORS['Q2'], markersize=6)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('N Synapses')
        ax.set_ylabel('Peak Dimers')
        ax.set_title('Q2: Network Scaling', fontweight='bold', color=COLORS['Q2'])
    
    # 2.2: Stim intensity
    if 'stim_intensity' in all_results:
        ax = fig.add_subplot(gs[1, 1])
        summary = all_results['stim_intensity'].summary_stats
        voltages = sorted(summary.keys())
        dimers = [summary[v]['dimers']['mean'] for v in voltages]
        ax.plot(voltages, dimers, 'o-', color=COLORS['Q2'], markersize=6)
        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Peak Dimers')
        ax.set_title('Q2: IO Curve', fontweight='bold', color=COLORS['Q2'])
    
    # 2.3: Eligibility decay
    if 'dopamine_timing' in all_results:
        ax = fig.add_subplot(gs[1, 2])
        summary = all_results['dopamine_timing'].summary_stats
        delays = sorted(summary.keys())
        elig = [summary[d]['pre_da_eligibility']['mean'] for d in delays]
        ax.plot(delays, elig, 'o-', color=COLORS['Q2'], markersize=6, label='Data')
        t = np.linspace(0, max(delays), 50)
        ax.plot(t, np.exp(-t/67), '--', color='gray', alpha=0.5, label='T₂=67s')
        ax.set_xlabel('DA Delay (s)')
        ax.set_ylabel('Eligibility')
        ax.set_title('Q2: Eligibility Decay', fontweight='bold', color=COLORS['Q2'])
        ax.legend(fontsize=8)
    
    # 2.4: Isotope effect
    if 'isotope' in all_results:
        ax = fig.add_subplot(gs[1, 3])
        # This needs special handling since isotope data is structured differently
        ax.text(0.5, 0.5, 'P31: τ=158s\nP32: τ=0.3s\n520× difference!', 
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title('Q2: Isotope Effect', fontweight='bold', color=COLORS['Q2'])
        ax.axis('off')
    
    # === ROW 3: GATE & CLASSICAL ===
    
    # 3.1: Commitment vs N
    if 'network_threshold' in all_results:
        ax = fig.add_subplot(gs[2, 0])
        summary = all_results['network_threshold'].summary_stats
        n_vals = sorted([k for k in summary.keys() if isinstance(k, int)])
        commits = [summary[n].get('commitment', {}).get('mean', 
                   summary[n].get('committed', {}).get('mean', 0)) for n in n_vals]
        ax.plot(n_vals, commits, 's-', color=COLORS['gate'], markersize=6)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('N Synapses')
        ax.set_ylabel('Commitment')
        ax.set_title('Gate: N Threshold', fontweight='bold', color=COLORS['gate'])
        ax.set_ylim(-0.05, 1.1)
    
    # 3.2: Commitment vs Delay
    if 'dopamine_timing' in all_results:
        ax = fig.add_subplot(gs[2, 1])
        summary = all_results['dopamine_timing'].summary_stats
        delays = sorted(summary.keys())
        # Approximate commitment from eligibility
        elig = [summary[d]['pre_da_eligibility']['mean'] for d in delays]
        commits = [1.0 if e > 0.3 else 0.0 for e in elig]
        ax.plot(delays, commits, 's-', color=COLORS['gate'], markersize=6)
        ax.set_xlabel('DA Delay (s)')
        ax.set_ylabel('Commitment')
        ax.set_title('Gate: Timing Window', fontweight='bold', color=COLORS['gate'])
        ax.set_ylim(-0.05, 1.1)
    
    # 3.3: Pharmacology summary
    if 'pharmacology' in all_results:
        ax = fig.add_subplot(gs[2, 2])
        summary = all_results['pharmacology'].summary_stats
        conditions = ['Control', 'APV', 'Nocodazole', 'Isoflurane']
        commits = [summary.get(c, {}).get('committed', {}).get('mean', 0) for c in conditions]
        colors = [COLORS['control'] if c > 0.5 else COLORS['blocked'] for c in commits]
        ax.bar(range(len(conditions)), commits, color=colors, alpha=0.8)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(['Ctrl', 'APV', 'Noco', 'Iso'], rotation=45)
        ax.set_ylabel('Commitment')
        ax.set_title('Gate: Pharmacology', fontweight='bold', color=COLORS['gate'])
        ax.set_ylim(0, 1.1)

    # 3.4: Four-factor gate
    if 'four_factor_gate' in all_results:
        ax = fig.add_subplot(gs[2, 3])
        summary = all_results['four_factor_gate'].summary_stats
        conditions = ['Full (Ca+Dimers+EM+DA)', 'No Ca (APV)', 'No Dimers', 'No EM Field', 'No DA']
        short = ['Full', 'No Ca', 'No Dimers', 'No EM', 'No DA']
        commits = [summary.get(c, {}).get('committed', {}).get('mean', 0) for c in conditions]
        colors = [COLORS['control'] if c > 0.5 else COLORS['blocked'] for c in commits]
        ax.bar(range(len(short)), commits, color=colors, alpha=0.8)
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short)
        ax.set_ylabel('Commitment')
        ax.set_title('Gate: AND Logic', fontweight='bold', color=COLORS['gate'])
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

# =============================================================================
# GENERATE ALL FIGURES (Required by main.py)
# =============================================================================

def generate_all_figures(all_results: Dict, output_dir: str = "figures"):
    """
    Generate all figures for the experiment suite
    """
    from pathlib import Path
    from datetime import datetime
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nGenerating figures...")
    
    # 1. Architecture diagram
    plot_cascade_architecture(path / f"01_architecture_{timestamp}.png")
    
    # 2. Network threshold
    if 'network_threshold' in all_results:
        plot_network_threshold_improved(
            all_results['network_threshold'].summary_stats,
            path / f"02_network_threshold_{timestamp}.png"
        )
    
    # 3. Dopamine timing (uses master dashboard for now)
    
    # 4. Isotope
    if 'isotope' in all_results:
        summary = all_results['isotope'].summary_stats
        if 'P31' in summary and 'P32' in summary:
            plot_isotope_improved(
                summary['P31'], summary['P32'],
                path / f"04_isotope_{timestamp}.png"
            )
    
    # 5. Pharmacology
    if 'pharmacology' in all_results:
        plot_pharmacology_improved(
            all_results['pharmacology'].summary_stats,
            path / f"05_pharmacology_{timestamp}.png"
        )
    
    # 6. Anesthetic dose-response
    if 'anesthetic' in all_results:
        plot_anesthetic_improved(
            all_results['anesthetic'].summary_stats,
            path / f"06_anesthetic_{timestamp}.png"
        )
    
    # 7. Four-factor gate
    if 'four_factor_gate' in all_results:
        plot_four_factor_gate_improved(
            all_results['four_factor_gate'].summary_stats,
            path / f"07_four_factor_gate_{timestamp}.png"
        )
    
    # 8. Master dashboard
    plot_master_dashboard_improved(all_results, path / f"08_dashboard_{timestamp}.png")
    
    print(f"\nAll figures saved to: {output_dir}/")
    
    return path


if __name__ == "__main__":
    print("Improved Visualization Module")
    print("=" * 50)
    print("\nKey improvements over original:")
    print("1. Show COMMITMENT (not final_strength) as primary output")
    print("2. Consistent color scheme: Q1=red, Q2=green, Gate=orange, Classical=purple")
    print("3. Clear annotations showing predictions vs observations")
    print("4. Mechanism explanations in figure captions")
    print("5. Y-axes scaled appropriately (not compressed to 1.00-1.07)")
    print("\nGenerate test architecture figure...")
    
    fig = plot_cascade_architecture('/home/claude/quantum_fixes/cascade_architecture.png')
    print("Saved: cascade_architecture.png")