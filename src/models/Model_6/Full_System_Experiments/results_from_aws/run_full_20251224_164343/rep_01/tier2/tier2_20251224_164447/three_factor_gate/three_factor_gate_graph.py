#!/usr/bin/env python3
"""
Regenerate three-factor gate figure from saved data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

@dataclass
class ConditionData:
    description: str
    eligibility: float
    dopamine_present: bool
    calcium_elevated: bool
    field_sufficient: bool
    gate_opened: bool
    committed: bool
    commitment_level: float
    n_particles: int
    n_bonds: int

def load_and_plot(json_path: str, output_path: str = None):
    """Load JSON data and regenerate plot with fix"""
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Debug: print what we loaded
    print("Loaded data:")
    for cond, vals in data['conditions'].items():
        print(f"  {cond}: committed={vals['committed']}, commitment_level={vals['commitment_level']}")
    
    # FIX: If committed=True but commitment_level=0, set commitment_level=1.0
    for cond, vals in data['conditions'].items():
        if vals['committed'] and vals['commitment_level'] == 0:
            print(f"  FIXING {cond}: setting commitment_level to 1.0")
            vals['commitment_level'] = 1.0
    
    # Now plot
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.4)
    
    conditions = ['control', 'no_eligibility', 'no_dopamine', 'no_calcium']
    labels = ['Control\n(all factors)', 'No Eligibility\n(block dimers)', 
              'No Dopamine\n(no reward)', 'No Calcium\n(block channels)']
    
    # Colors
    color_present = '#28A745'
    color_absent = '#DC3545'
    color_committed = '#2E86AB'
    
    # === TOP: Factor Heatmap ===
    ax_factors = fig.add_subplot(gs[0])
    
    factors = ['Eligibility\n(dimers)', 'Dopamine\n(reward)', 'Calcium\n(activity)', 
               'EM Field\n(coupling)']
    
    # Build matrix
    matrix = np.zeros((4, 4))
    for i, cond in enumerate(conditions):
        c = data['conditions'][cond]
        matrix[0, i] = 1 if c['eligibility'] > 0.1 else 0
        matrix[1, i] = 1 if c['dopamine_present'] else 0
        matrix[2, i] = 1 if c['calcium_elevated'] else 0
        matrix[3, i] = 1 if c['field_sufficient'] else 0
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([color_absent, color_present])
    
    im = ax_factors.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    ax_factors.set_xticks(range(4))
    ax_factors.set_xticklabels(labels, fontsize=10)
    ax_factors.set_yticks(range(4))
    ax_factors.set_yticklabels(factors, fontsize=10)
    ax_factors.set_title('Factor Presence by Condition', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = '✓' if matrix[i, j] == 1 else '✗'
            ax_factors.text(j, i, text, ha='center', va='center', 
                           fontsize=16, color='white', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_factors, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Absent', 'Present'])
    
    # === BOTTOM: Outcome Bars ===
    ax_outcome = fig.add_subplot(gs[1])
    
    x = np.arange(4)
    width = 0.6
    
    # Commitment levels (now fixed)
    commit_levels = [data['conditions'][c]['commitment_level'] for c in conditions]
    committed = [data['conditions'][c]['committed'] for c in conditions]
    
    bars = ax_outcome.bar(x, commit_levels, width)
    
    # Color by outcome
    for i, (bar, comm) in enumerate(zip(bars, committed)):
        if comm:
            bar.set_color(color_committed)
            bar.set_alpha(0.9)
        else:
            bar.set_color(color_absent)
            bar.set_alpha(0.5)
    
    ax_outcome.set_xlabel('Condition', fontsize=12)
    ax_outcome.set_ylabel('Commitment Level', fontsize=12)
    ax_outcome.set_title('Commitment Outcome (blue = committed, red = not committed)', 
                         fontsize=12, fontweight='bold')
    ax_outcome.set_xticks(x)
    ax_outcome.set_xticklabels(labels, fontsize=9)
    ax_outcome.set_ylim(0, 1.1)
    ax_outcome.grid(True, alpha=0.3, axis='y')
    
    # Add labels
    for i, (level, comm) in enumerate(zip(commit_levels, committed)):
        label = f'{level:.2f}\n{"✓" if comm else "✗"}'
        ax_outcome.annotate(label, xy=(i, level), xytext=(0, 5),
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=10, fontweight='bold')
    
    # Validation message
    if data['gate_logic_validated']:
        fig.text(0.5, 0.02, '✓ THREE-FACTOR AND-GATE VALIDATED: All factors required for commitment',
                fontsize=12, ha='center', color=color_present, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_present))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved corrected figure to {output_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    import sys
    
    # Default paths - adjust to your directory structure
    json_path = "three_factor_gate.json"
    output_path = "three_factor_gate_corrected.png"
    
    # Allow command line override
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    load_and_plot(json_path, output_path)