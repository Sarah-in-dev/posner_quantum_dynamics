"""
Visualization Script - Generate Publication-Quality Plots

Creates comprehensive visualizations of quantum vs classical learning

Author: Sarah Davidson
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats


def load_results(filepath='results/experiment_data.npz'):
    """Load experiment results"""
    data = np.load(filepath, allow_pickle=True)
    return (
        data['results_quantum'].item(),
        data['results_classical'].item(),
        data['analysis'].item()
    )


def plot_full_analysis(results_quantum, results_classical, analysis,
                      save_path='results/experiment_results.png'):
    """Create comprehensive analysis figure"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Color scheme
    color_classical = '#3498db'  # Blue
    color_quantum = '#e74c3c'    # Red
    color_switch = '#95a5a6'     # Gray
    
    # ========================================================================
    # PANEL 1: Learning Curves (Main Result)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    trials_c = results_classical['trial']
    acc_c = results_classical['accuracy']
    trials_q = results_quantum['trial']
    acc_q = results_quantum['accuracy']
    
    ax1.plot(trials_c, acc_c, color=color_classical, linewidth=2.5, 
             label='Classical Network', alpha=0.8)
    ax1.plot(trials_q, acc_q, color=color_quantum, linewidth=2.5,
             label='Quantum-Coherent Network', alpha=0.8)
    
    # Mark context switches
    for switch in analysis['switch_points']:
        ax1.axvline(x=switch, color=color_switch, linestyle='--', 
                   alpha=0.4, linewidth=1.5)
    
    ax1.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, 
               linewidth=2, label='80% Criterion')
    
    ax1.set_xlabel('Trial Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (Rolling Average)', fontsize=14, fontweight='bold')
    ax1.set_title('Learning Performance: Quantum vs Classical Processing', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0, max(trials_c)])
    
    # Add annotation
    if analysis['speedup_factor']:
        ax1.text(0.02, 0.95, f'Speedup: {analysis["speedup_factor"]:.2f}×',
                transform=ax1.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    
    # ========================================================================
    # PANEL 2: Adaptation Speed Comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    quantum_times = analysis['quantum_adaptation_times']
    classical_times = analysis['classical_adaptation_times']
    
    if quantum_times and classical_times:
        positions = [1, 2]
        box_data = [classical_times, quantum_times]
        
        bp = ax2.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, labels=['Classical', 'Quantum'],
                        showmeans=True, meanline=True)
        
        bp['boxes'][0].set_facecolor(color_classical)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(color_quantum)
        bp['boxes'][1].set_alpha(0.6)
        
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5)
        for cap in bp['caps']:
            cap.set(linewidth=1.5)
        
        ax2.set_ylabel('Trials to 80% Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title(f'Adaptation Speed After Context Switch\n(Speedup: {analysis["speedup_factor"]:.2f}×)',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Statistical test
        if len(quantum_times) >= 3 and len(classical_times) >= 3:
            t_stat, p_value = stats.ttest_ind(classical_times, quantum_times)
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            y_pos = max(max(classical_times), max(quantum_times)) * 1.1
            ax2.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax2.text(1.5, y_pos * 1.02, f'p={p_value:.4f} {sig}',
                    ha='center', fontsize=10, fontweight='bold')
    
    # ========================================================================
    # PANEL 3: Post-Switch Recovery Curves
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1:])
    
    if analysis['quantum_post_switch_curves'] and analysis['classical_post_switch_curves']:
        # Average across all switches
        quantum_avg = np.mean(analysis['quantum_post_switch_curves'], axis=0)
        classical_avg = np.mean(analysis['classical_post_switch_curves'], axis=0)
        
        # Standard error
        quantum_sem = stats.sem(analysis['quantum_post_switch_curves'], axis=0)
        classical_sem = stats.sem(analysis['classical_post_switch_curves'], axis=0)
        
        x_trials = np.arange(len(quantum_avg))
        
        # Plot with error bands
        ax3.plot(x_trials, classical_avg, color=color_classical, linewidth=2.5,
                label='Classical', alpha=0.8)
        ax3.fill_between(x_trials, classical_avg - classical_sem, 
                        classical_avg + classical_sem,
                        color=color_classical, alpha=0.2)
        
        ax3.plot(x_trials, quantum_avg, color=color_quantum, linewidth=2.5,
                label='Quantum', alpha=0.8)
        ax3.fill_between(x_trials, quantum_avg - quantum_sem,
                        quantum_avg + quantum_sem,
                        color=color_quantum, alpha=0.2)
        
        ax3.axhline(y=0.8, color='green', linestyle=':', alpha=0.5,
                   linewidth=2, label='80% Criterion')
        
        ax3.set_xlabel('Trials After Context Switch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Average Recovery Dynamics', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])
    
    # ========================================================================
    # PANEL 4: Quantum Coherence Dynamics
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    if 'coherence_time' in results_quantum and results_quantum['coherence_time']:
        coherence_times = results_quantum['coherence_time']
        
        ax4.plot(trials_q, coherence_times, color='purple', linewidth=2, alpha=0.7)
        
        # Mark mode transitions
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5, 
                   linewidth=2, label='Dimer Mode (T₂=100)')
        ax4.axhline(y=2, color='blue', linestyle='--', alpha=0.5,
                   linewidth=2, label='Trimer Mode (T₂=2)')
        
        # Mark context switches
        for switch in analysis['switch_points']:
            ax4.axvline(x=switch, color=color_switch, linestyle='--',
                       alpha=0.3, linewidth=1.5)
        
        ax4.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Coherence Time (steps)', fontsize=11, fontweight='bold')
        ax4.set_title('Quantum Coherence Dynamics', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # ========================================================================
    # PANEL 5: Active Quantum Paths
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    if 'n_active_paths' in results_quantum and results_quantum['n_active_paths']:
        active_paths = results_quantum['n_active_paths']
        
        ax5.plot(trials_q, active_paths, color='orange', linewidth=2, alpha=0.7)
        
        # Mark context switches
        for switch in analysis['switch_points']:
            ax5.axvline(x=switch, color=color_switch, linestyle='--',
                       alpha=0.3, linewidth=1.5)
        
        ax5.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Number of Active Paths', fontsize=11, fontweight='bold')
        ax5.set_title('Quantum Path Exploration', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 8])
    
    # ========================================================================
    # PANEL 6: Summary Statistics
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Compile summary text
    summary = f"""
EXPERIMENT SUMMARY
{'='*40}

Total Trials: {len(results_quantum['trial'])}
Context Switches: {len(analysis['switch_points'])}

CLASSICAL NETWORK
  Final Accuracy: {results_classical['accuracy'][-1]:.3f}
  Avg Adaptation: {analysis['classical_mean']:.1f} ± {analysis['classical_std']:.1f} trials
  
QUANTUM NETWORK
  Final Accuracy: {results_quantum['accuracy'][-1]:.3f}
  Avg Adaptation: {analysis['quantum_mean']:.1f} ± {analysis['quantum_std']:.1f} trials
  
PERFORMANCE COMPARISON
  Speedup Factor: {analysis['speedup_factor']:.2f}×
  Thesis Prediction: 2.4×
  
  Status: {'✓ CONFIRMED' if analysis['speedup_factor'] >= 2.0 else '⚠ PARTIAL' if analysis['speedup_factor'] >= 1.5 else '✗ NOT MET'}

QUANTUM MECHANISMS
  T₂ (dimer): 100 steps
  T₂ (trimer): 2 steps
  J-coupling: 20 Hz
  Parallel paths: 8
    """
    
    ax6.text(0.05, 0.95, summary, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax6.transAxes)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """Load and visualize results"""
    print("="*70)
    print("EXPERIMENT VISUALIZATION")
    print("="*70)
    
    # Load results
    print("\nLoading experiment data...")
    results_q, results_c, analysis = load_results()
    
    print("Creating visualizations...")
    plot_full_analysis(results_q, results_c, analysis)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()