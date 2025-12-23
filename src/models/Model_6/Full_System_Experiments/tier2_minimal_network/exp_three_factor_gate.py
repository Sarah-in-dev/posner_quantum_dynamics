"""
Tier 2 Experiment: Three-Factor Gate
=====================================

Tests that all three factors are necessary for synaptic commitment:
1. Eligibility (quantum coherence of dimers)
2. Dopamine (reward signal)  
3. Calcium (activity marker)

Protocol:
- Control: All three factors present → commitment
- No eligibility: Block dimer formation → no commitment
- No dopamine: Omit reward signal → no commitment
- No calcium: Block calcium channels → no commitment

This validates the AND-gate logic of the plasticity mechanism.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


@dataclass
class GateConditionResult:
    """Results for single gate condition"""
    condition_name: str
    description: str
    
    # State at gate check
    eligibility: float = 0.0
    dopamine_present: bool = False
    calcium_elevated: bool = False
    field_sufficient: bool = False
    
    # Outcome
    gate_opened: bool = False
    committed: bool = False
    commitment_level: float = 0.0
    
    # Supporting metrics
    n_particles: int = 0
    n_bonds: int = 0
    mean_singlet_prob: float = 0.0
    peak_calcium_uM: float = 0.0
    peak_dopamine_nM: float = 0.0


@dataclass
class ThreeFactorResult:
    """Complete results from three-factor gate experiment"""
    conditions: Dict[str, GateConditionResult] = field(default_factory=dict)
    gate_logic_validated: bool = False


def run_condition(condition: str, n_synapses: int = 5, verbose: bool = True) -> GateConditionResult:
    """
    Run single gate condition
    
    Parameters
    ----------
    condition : str
        One of: 'control', 'no_eligibility', 'no_dopamine', 'no_calcium'
    verbose : bool
        Print progress
    """
    descriptions = {
        'control': 'All factors present (should commit)',
        'no_eligibility': 'Block dimer formation (no quantum memory)',
        'no_dopamine': 'Omit reward signal',
        'no_calcium': 'Block calcium influx'
    }
    
    result = GateConditionResult(
        condition_name=condition,
        description=descriptions.get(condition, '')
    )
    
    if verbose:
        print(f"\n--- Condition: {condition} ---")
        print(f"    {result.description}")
    
    # Configure model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0
    
    if condition == 'no_dopamine':
        params.dopamine = None  # Completely disable dopamine system
    
    # Apply condition-specific modifications
    if condition == 'no_calcium':
        params.calcium.nmda_blocked = True  # Block NMDA receptors (APV)
    
    # Create multiple synapses to reach dimer threshold
    models = [Model6QuantumSynapse(params) for _ in range(n_synapses)]
    
    # For no_eligibility, prevent dimer formation in all synapses
    if condition == 'no_eligibility':
        for model in models:
            model.ca_phosphate.dimerization.k_base = 0.0
    
    dt = 0.001  # 1 ms
    
    if verbose:
        print(f"    Using {n_synapses} synapses")
    
    # Track peak values
    peak_calcium = 0.0
    peak_dopamine = 0.0
    
    # === PHASE 1: BASELINE ===
    if verbose:
        print("    Phase 1: Baseline (100ms)")
    
    for _ in range(100):
        for model in models:
            model.step(dt, {'voltage': -70e-3, 'reward': False})
    
    # === PHASE 2: THETA-BURST STIMULATION ===
    if verbose:
        print("    Phase 2: Theta-burst stimulation")

    # Give dopamine DURING stimulation (except for no_dopamine condition)
    give_dopamine = (condition != 'no_dopamine')

    for burst in range(5):
        for spike in range(4):
            for _ in range(2):
                for model in models:
                    model.step(dt, {'voltage': -10e-3, 'reward': give_dopamine})
                ca = np.max(models[0].calcium.get_concentration()) * 1e6
                peak_calcium = max(peak_calcium, ca)
                # Track dopamine during stimulation
                if hasattr(models[0], 'get_experimental_metrics'):
                    metrics = models[0].get_experimental_metrics()
                    dopa = metrics.get('dopamine_max_nM', 0.0)
                    peak_dopamine = max(peak_dopamine, dopa)
            for _ in range(8):
                for model in models:
                    model.step(dt, {'voltage': -70e-3, 'reward': give_dopamine})  # Changed!
        for _ in range(160):
            for model in models:
                model.step(dt, {'voltage': -70e-3, 'reward': give_dopamine})  # Changed!

    # Record post-stim eligibility (average across models)
    post_stim_elig = np.mean([m.get_eligibility() for m in models])

    # === PHASE 3: DOPAMINE CONTINUATION ===
    if verbose:
        print("    Phase 3: Dopamine continuation (300ms)")

    for _ in range(300):
        for model in models:
            model.step(dt, {'voltage': -70e-3, 'reward': give_dopamine})
        
    # === PHASE 4: CONSOLIDATION ===
    if verbose:
        print("    Phase 4: Consolidation (1s)")

    dt_consol = 0.01  # 10ms
    for _ in range(100):  # 1 second - enough for gate test
        for model in models:
            model.step(dt_consol, {'voltage': -70e-3, 'reward': False})

    # === COLLECT RESULTS (aggregate across all synapses) ===
    total_particles = 0
    total_bonds = 0
    all_singlet_probs = []
    
    for model in models:
        pm = model.dimer_particles.get_network_metrics()
        total_particles += pm['n_dimers']
        total_bonds += pm['n_bonds']
        for d in model.dimer_particles.dimers:
            all_singlet_probs.append(d.singlet_probability)
    
    result.n_particles = total_particles
    result.n_bonds = total_bonds
    
    if all_singlet_probs:
        result.mean_singlet_prob = float(np.mean(all_singlet_probs))
    
    result.peak_calcium_uM = peak_calcium
    result.peak_dopamine_nM = peak_dopamine
    
    # Gate factors
    result.eligibility = post_stim_elig
    result.dopamine_present = peak_dopamine > 50  # Read threshold
    result.calcium_elevated = peak_calcium > 0.5  # µM threshold
    result.field_sufficient = getattr(model, '_collective_field_kT', 0) > 10
    
    # Gate and commitment (any synapse committed = success)
    result.gate_opened = any(getattr(m, '_plasticity_gate', False) for m in models)
    result.committed = any(getattr(m, '_camkii_committed', False) for m in models)
    result.commitment_level = max(getattr(m, '_committed_memory_level', 0.0) for m in models)
    
    if verbose:
        print(f"    Results:")
        print(f"      Particles: {result.n_particles}, Bonds: {result.n_bonds}")
        print(f"      Eligibility: {result.eligibility:.3f}")
        print(f"      Dopamine present: {result.dopamine_present}")
        print(f"      Calcium elevated: {result.calcium_elevated}")
        print(f"      Gate opened: {result.gate_opened}")
        print(f"      Committed: {result.committed}")
    
    return result


def run(verbose: bool = True) -> ThreeFactorResult:
    """
    Run complete three-factor gate experiment
    """
    result = ThreeFactorResult()
    
    if verbose:
        print("="*70)
        print("THREE-FACTOR GATE EXPERIMENT")
        print("="*70)
        print("\nTesting: Control, No Eligibility, No Dopamine, No Calcium")
    
    conditions = ['control', 'no_eligibility', 'no_dopamine', 'no_calcium']
    
    for cond in conditions:
        cond_result = run_condition(cond, n_synapses=10, verbose=verbose)
        result.conditions[cond] = cond_result
    
    # Validate gate logic
    control_committed = result.conditions['control'].committed
    others_not_committed = all(
        not result.conditions[c].committed 
        for c in ['no_eligibility', 'no_dopamine', 'no_calcium']
    )
    
    result.gate_logic_validated = control_committed and others_not_committed
    
    return result


def plot(result: ThreeFactorResult, output_dir: Path = None) -> plt.Figure:
    """
    Generate publication-quality figure
    
    Layout:
    - Top: Factor presence for each condition
    - Bottom: Outcome summary
    """
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
        c = result.conditions[cond]
        matrix[0, i] = 1 if c.eligibility > 0.1 else 0
        matrix[1, i] = 1 if c.dopamine_present else 0
        matrix[2, i] = 1 if c.calcium_elevated else 0
        matrix[3, i] = 1 if c.field_sufficient else 0
    
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
            color = 'white' if matrix[i, j] == 1 else 'white'
            ax_factors.text(j, i, text, ha='center', va='center', 
                           fontsize=16, color=color, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_factors, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Absent', 'Present'])
    
    # === BOTTOM: Outcome Bars ===
    ax_outcome = fig.add_subplot(gs[1])
    
    x = np.arange(4)
    width = 0.6
    
    # Commitment levels
    commit_levels = [result.conditions[c].commitment_level for c in conditions]
    committed = [result.conditions[c].committed for c in conditions]
    
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
    if result.gate_logic_validated:
        fig.text(0.5, 0.02, '✓ THREE-FACTOR AND-GATE VALIDATED: All factors required for commitment',
                fontsize=12, ha='center', color=color_present, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_present))
    else:
        fig.text(0.5, 0.02, '✗ Gate logic not validated - check parameters',
                fontsize=12, ha='center', color=color_absent, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = output_dir / 'three_factor_gate.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        
        # Save data
        data = {
            'experiment': 'three_factor_gate',
            'timestamp': datetime.now().isoformat(),
            'conditions': {
                cond: {
                    'description': c.description,
                    'eligibility': float(c.eligibility),
                    'dopamine_present': bool(c.dopamine_present),
                    'calcium_elevated': bool(c.calcium_elevated),
                    'field_sufficient': bool(c.field_sufficient),
                    'gate_opened': bool(c.gate_opened),
                    'committed': bool(c.committed),
                    'commitment_level': float(c.commitment_level),
                    'n_particles': int(c.n_particles),
                    'n_bonds': int(c.n_bonds)
                }
                for cond, c in result.conditions.items()
            },
            'gate_logic_validated': bool(result.gate_logic_validated)
        }
        
        json_path = output_dir / 'three_factor_gate.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {json_path}")
    
    return fig


def print_summary(result: ThreeFactorResult):
    """Print summary of results"""
    print("\n" + "="*70)
    print("THREE-FACTOR GATE SUMMARY")
    print("="*70)
    
    print(f"\n{'Condition':<20} {'Elig':<8} {'Dopa':<8} {'Ca':<8} {'Committed':<12}")
    print("-"*60)
    
    for cond in ['control', 'no_eligibility', 'no_dopamine', 'no_calcium']:
        c = result.conditions[cond]
        elig = f"{c.eligibility:.2f}" if c.eligibility > 0 else "✗"
        dopa = "✓" if c.dopamine_present else "✗"
        ca = "✓" if c.calcium_elevated else "✗"
        comm = "✓ YES" if c.committed else "✗ no"
        print(f"{cond:<20} {elig:<8} {dopa:<8} {ca:<8} {comm:<12}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if result.gate_logic_validated:
        print("""
✓ THREE-FACTOR AND-GATE VALIDATED

The plasticity mechanism requires ALL THREE factors:
1. ELIGIBILITY: Quantum coherent dimer network (memory trace)
2. DOPAMINE: Reward signal (teaching signal)
3. CALCIUM: Synaptic activity (specificity marker)

Removing ANY factor prevents commitment.
This matches the theoretical prediction and neo-Hebbian plasticity rules.
""")
    else:
        ctrl = result.conditions['control']
        print(f"""
Gate logic not fully validated:
- Control committed: {ctrl.committed}
- Need to check why other conditions may have committed
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run three-factor gate experiment')
    parser.add_argument('--output', type=str, default='results/tier2',
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    result = run(verbose=not args.quiet)
    
    print_summary(result)
    
    fig = plot(result, output_dir=args.output)
    
    plt.show()