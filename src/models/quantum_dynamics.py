#!/usr/bin/env python
"""
Run the validated Posner quantum dynamics model
This script demonstrates the refined model with proper parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.models.synaptic_posner import SynapticPosnerModel
from src.models.quantum_dynamics import PosnerQuantumDynamics, QuantumEnhancedPlasticity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def run_complete_analysis():
    """Run the complete validated analysis"""
    
    print("="*60)
    print("POSNER QUANTUM DYNAMICS - VALIDATED MODEL")
    print("="*60)
    
    # 1. Chemical dynamics
    print("\n1. CHEMICAL DYNAMICS")
    print("-"*30)
    
    chemical_model = SynapticPosnerModel()
    
    # Single spike
    single_spike = chemical_model.simulate_single_spike(duration=0.5)
    print(f"Single spike - Peak [Posner]: {np.max(single_spike['posner'])*1e9:.1f} nM")
    
    # Spike train
    spike_train = chemical_model.simulate_spike_train(n_spikes=20, isi=0.1)
    print(f"Spike train - Peak [Posner]: {np.max(spike_train['posner'])*1e9:.1f} nM")
    
    # 2. Quantum dynamics
    print("\n2. QUANTUM DYNAMICS")
    print("-"*30)
    
    # Both isotopes
    qd_p31 = PosnerQuantumDynamics(isotope='P31')
    qd_p32 = PosnerQuantumDynamics(isotope='P32')
    
    # Calculate coherence at peak Posner
    peak_posner = np.max(spike_train['posner'])
    
    coherence_p31 = qd_p31.calculate_coherence_time(peak_posner)
    coherence_p32 = qd_p32.calculate_coherence_time(peak_posner)
    
    print(f"\n³¹P at {peak_posner*1e9:.1f} nM Posner:")
    print(f"  T₂ = {coherence_p31['T2']:.3f} s ({coherence_p31['T2']*1000:.0f} ms)")
    
    print(f"\n³²P at {peak_posner*1e9:.1f} nM Posner:")
    print(f"  T₂ = {coherence_p32['T2']:.3f} s ({coherence_p32['T2']*1000:.0f} ms)")
    
    print(f"\nIsotope ratio: {coherence_p31['T2']/coherence_p32['T2']:.1f}×")
    
    # 3. Enhancement calculation
    print("\n3. LEARNING ENHANCEMENT")
    print("-"*30)
    
    enhancement_p31 = qd_p31.predict_enhancement_factor(peak_posner)
    enhancement_p32 = qd_p32.predict_enhancement_factor(peak_posner)
    
    print(f"³¹P enhancement over classical: {enhancement_p31:.1f}×")
    print(f"³²P enhancement over classical: {enhancement_p32:.1f}×")
    print(f"Relative advantage of ³¹P: {enhancement_p31/enhancement_p32:.1f}×")
    
    # Create figure
    create_summary_figure(chemical_model, qd_p31, qd_p32, spike_train, 
                         enhancement_p31, enhancement_p32)
    
    # 4. Key predictions
    print("\n4. KEY EXPERIMENTAL PREDICTIONS")
    print("-"*30)
    print("• Posner forms transiently at 10-100 nM during synaptic activity")
    print("• ³¹P maintains quantum coherence ~10× longer than ³²P")
    print("• This enables ~10× faster learning in BCI paradigms")
    print("• Effect strongest at 10-20 Hz stimulation")
    print("• Temperature dependence: Q₁₀ < 1.2 for quantum, > 2 for classical")
    
    print("\n✅ Analysis complete!")

def create_summary_figure(chemical_model, qd_p31, qd_p32, spike_train_results,
                         enhancement_p31, enhancement_p32):
    """Create publication-quality summary figure"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Validated Model: Quantum Enhancement of Synaptic Plasticity', 
                 fontsize=16, fontweight='bold')
    
    # 1. Posner dynamics
    ax1.plot(spike_train_results['time'], spike_train_results['posner']*1e9, 
             'g-', linewidth=2)
    for st in spike_train_results['spike_times'][:10]:
        ax1.axvline(st, color='red', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('[Posner] (nM)')
    ax1.set_title('A. Posner Formation During 10 Hz Stimulation')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2)
    
    # 2. Coherence comparison
    conc_range = np.logspace(-10, -7, 50)  # 0.1 to 100 nM
    t2_p31 = [qd_p31.calculate_coherence_time(c)['T2'] for c in conc_range]
    t2_p32 = [qd_p32.calculate_coherence_time(c)['T2'] for c in conc_range]
    
    ax2.loglog(conc_range*1e9, t2_p31, 'b-', label='³¹P', linewidth=2)
    ax2.loglog(conc_range*1e9, t2_p32, 'r--', label='³²P', linewidth=2)
    ax2.axhline(0.020, color='gray', linestyle=':', label='STDP (20 ms)')
    ax2.axvline(np.max(spike_train_results['posner'])*1e9, 
                color='green', linestyle=':', alpha=0.5)
    ax2.set_xlabel('[Posner] (nM)')
    ax2.set_ylabel('T₂ (s)')
    ax2.set_title('B. Isotope-Dependent Coherence Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Enhancement factors
    isotopes = ['Classical', '³²P', '³¹P']
    enhancements = [1.0, enhancement_p32, enhancement_p31]
    colors = ['gray', 'red', 'blue']
    
    bars = ax3.bar(isotopes, enhancements, color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=2)
    for bar, val in zip(bars, enhancements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}×', ha='center', fontweight='bold')
    ax3.set_ylabel('Enhancement Factor')
    ax3.set_title('C. Learning Enhancement')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. BCI learning curves
    qep = QuantumEnhancedPlasticity(qd_p31)
    t_learn = np.linspace(0, 120, 200)
    
    perf_classical = qep.predict_learning_curve(t_learn, 1.0) * 100
    perf_p31 = qep.predict_learning_curve(t_learn, enhancement_p31) * 100
    perf_p32 = qep.predict_learning_curve(t_learn, enhancement_p32) * 100
    
    ax4.plot(t_learn, perf_classical, 'k--', label='Classical', linewidth=2)
    ax4.plot(t_learn, perf_p32, 'r-', label='³²P', linewidth=2)
    ax4.plot(t_learn, perf_p31, 'b-', label='³¹P', linewidth=2)
    
    ax4.axvline(60, color='gray', linestyle=':', alpha=0.5)
    ax4.text(62, 50, '60s', rotation=90)
    ax4.set_xlabel('Training Time (s)')
    ax4.set_ylabel('Performance (%)')
    ax4.set_title('D. Predicted BCI Learning')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 120)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('figures/validated_model_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_parameter_sensitivity():
    """Test sensitivity to key parameters"""
    print("\n\nPARAMETER SENSITIVITY ANALYSIS")
    print("="*40)
    
    model = SynapticPosnerModel()
    
    # Test pH sensitivity
    print("\npH sensitivity:")
    ph_values = np.linspace(6.8, 7.6, 9)
    ph_scan = model.parameter_scan('pH', ph_values)
    
    for ph, posner in zip(ph_values, ph_scan['results']):
        print(f"  pH {ph:.1f}: {posner*1e9:.1f} nM")
    
    # Test temperature sensitivity
    print("\nTemperature sensitivity:")
    temp_values = np.linspace(305, 315, 5)  # 32-42°C
    temp_scan = model.parameter_scan('temperature', temp_values)
    
    for temp, posner in zip(temp_values, temp_scan['results']):
        print(f"  {temp-273:.0f}°C: {posner*1e9:.1f} nM")

if __name__ == "__main__":
    run_complete_analysis()
    test_parameter_sensitivity()