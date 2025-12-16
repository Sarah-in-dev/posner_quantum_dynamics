#!/usr/bin/env python3
"""
Cascade Diagnostic: Trace Where MT Invasion Enhancement Breaks Down
====================================================================

The cascade should be:
  Q1 field → k_enhancement → dimers → coherence → eligibility → commitment → strength

We see Q1 working (field enhancement 12.7 → 22.0 kT), but strength = 1.01 for both.
This script traces each step to find the bottleneck.
"""

import sys
import os
import numpy as np

# Add model path
sys.path.insert(0, '/mnt/user-data/uploads/src/models/Model_6')
sys.path.insert(0, '/mnt/user-data/uploads/src/models/Model_6/Full_System_Experiments')

print("=" * 80)
print("CASCADE DIAGNOSTIC: MT INVASION EXPERIMENT")
print("=" * 80)

try:
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    print("✓ Model imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease ensure model files are available.")
    sys.exit(1)


def run_cascade_diagnostic(mt_invaded: bool, verbose: bool = True):
    """
    Run a single synapse through the cascade and report all metrics
    """
    label = "MT+" if mt_invaded else "MT-"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CONDITION: {label}")
        print(f"{'='*60}")
    
    # Initialize model
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = False  # Single synapse for clarity
    
    model = Model6QuantumSynapse(params)
    model.set_microtubule_invasion(mt_invaded)
    
    dt = 0.001  # 1 ms timestep
    
    # Track cascade metrics over time
    cascade_metrics = {
        'time': [],
        'calcium_uM': [],
        'n_tryptophans': [],
        'em_field_kT': [],
        'k_enhancement': [],
        'dimer_count': [],
        'coherence': [],
        'eligibility': [],
        'plasticity_gate': [],
        'dopamine_nM': [],
        'committed': [],
        'committed_level': [],
        'strength': [],
    }
    
    # =========================================================================
    # PHASE 1: BASELINE (100 ms)
    # =========================================================================
    if verbose:
        print(f"\n--- Phase 1: Baseline (100ms) ---")
    
    for i in range(100):
        model.step(dt, {'voltage': -70e-3, 'reward': False})
    
    metrics = model.get_experimental_metrics()
    if verbose:
        print(f"  Calcium: {metrics.get('calcium_peak_uM', 0):.2f} μM")
        print(f"  Dimers: {getattr(model, '_previous_dimer_count', 0):.1f}")
    
    # =========================================================================
    # PHASE 2: STIMULATION (200 ms at -10 mV)
    # =========================================================================
    if verbose:
        print(f"\n--- Phase 2: Stimulation (200ms at -10mV) ---")
    
    for i in range(200):
        model.step(dt, {'voltage': -10e-3, 'reward': False, 'activity_level': 0.8})
        
        # Record every 10ms
        if i % 10 == 0:
            metrics = model.get_experimental_metrics()
            cascade_metrics['time'].append(100 + i)
            cascade_metrics['calcium_uM'].append(metrics.get('calcium_peak_uM', 0))
            cascade_metrics['n_tryptophans'].append(model.n_tryptophans)
            cascade_metrics['em_field_kT'].append(metrics.get('collective_field_kT', 0))
            cascade_metrics['k_enhancement'].append(metrics.get('em_formation_enhancement', 1))
            
            # VALIDATED DIMER CALCULATION: concentration * 0.006
            dimer_conc_nM = metrics.get('dimer_peak_nM_ct', 0)
            dimer_count = dimer_conc_nM * 0.006  # Validated conversion factor
            cascade_metrics['dimer_count'].append(dimer_count)
            
            cascade_metrics['coherence'].append(metrics.get('coherence_dimer_mean', 0))
            cascade_metrics['eligibility'].append(model.eligibility.get_eligibility())
            cascade_metrics['plasticity_gate'].append(getattr(model, '_plasticity_gate', False))
            cascade_metrics['committed'].append(getattr(model, '_camkii_committed', False))
            cascade_metrics['committed_level'].append(getattr(model, '_committed_memory_level', 0))
            cascade_metrics['strength'].append(model.spine_plasticity.get_synaptic_strength())
            
            # Dopamine - read from metrics
            cascade_metrics['dopamine_nM'].append(metrics.get('dopamine_max_nM', 0))
    
    # Post-stim snapshot
    metrics = model.get_experimental_metrics()
    if verbose:
        dimer_conc_nM = metrics.get('dimer_peak_nM_ct', 0)
        dimer_count = dimer_conc_nM * 0.006  # Validated conversion
        print(f"  Peak Calcium: {metrics.get('calcium_peak_uM', 0):.2f} μM")
        print(f"  Q1 Field: {metrics.get('collective_field_kT', 0):.1f} kT")
        print(f"  k_enhancement: {metrics.get('em_formation_enhancement', 1):.2f}×")
        print(f"  Dimer conc: {dimer_conc_nM:.1f} nM → {dimer_count:.1f} dimers")
        print(f"  Coherence: {metrics.get('coherence_dimer_mean', 0):.3f}")
        print(f"  Eligibility: {model.eligibility.get_eligibility():.3f}")
    
    # =========================================================================
    # PHASE 3: DOPAMINE BURST (300 ms)
    # =========================================================================
    if verbose:
        print(f"\n--- Phase 3: Dopamine Burst (300ms) ---")
    
    for i in range(300):
        model.step(dt, {'voltage': -70e-3, 'reward': True})
        
        # Record every 10ms
        if i % 10 == 0:
            metrics = model.get_experimental_metrics()
            cascade_metrics['time'].append(300 + i)
            cascade_metrics['calcium_uM'].append(metrics.get('calcium_peak_uM', 0))
            cascade_metrics['n_tryptophans'].append(model.n_tryptophans)
            cascade_metrics['em_field_kT'].append(metrics.get('collective_field_kT', 0))
            cascade_metrics['k_enhancement'].append(metrics.get('em_formation_enhancement', 1))
            
            # VALIDATED DIMER CALCULATION: concentration * 0.006
            dimer_conc_nM = metrics.get('dimer_peak_nM_ct', 0)
            dimer_count = dimer_conc_nM * 0.006  # Validated conversion factor
            cascade_metrics['dimer_count'].append(dimer_count)
            
            cascade_metrics['coherence'].append(metrics.get('coherence_dimer_mean', 0))
            cascade_metrics['eligibility'].append(model.eligibility.get_eligibility())
            cascade_metrics['plasticity_gate'].append(getattr(model, '_plasticity_gate', False))
            cascade_metrics['committed'].append(getattr(model, '_camkii_committed', False))
            cascade_metrics['committed_level'].append(getattr(model, '_committed_memory_level', 0))
            cascade_metrics['strength'].append(model.spine_plasticity.get_synaptic_strength())
            
            # Dopamine - read from metrics
            cascade_metrics['dopamine_nM'].append(metrics.get('dopamine_max_nM', 0))
    
    if verbose:
        print(f"  Dopamine peak: {max(cascade_metrics['dopamine_nM']):.1f} nM")
        print(f"  Plasticity gate opened: {any(cascade_metrics['plasticity_gate'])}")
        print(f"  Committed: {any(cascade_metrics['committed'])}")
        print(f"  Committed level: {max(cascade_metrics['committed_level']):.3f}")
    
    # =========================================================================
    # PHASE 4: CONSOLIDATION (1000 ms)
    # =========================================================================
    if verbose:
        print(f"\n--- Phase 4: Consolidation (1000ms) ---")
    
    for i in range(1000):
        model.step(dt, {'voltage': -70e-3, 'reward': False})
    
    final_strength = model.spine_plasticity.get_synaptic_strength()
    
    if verbose:
        print(f"  Final strength: {final_strength:.3f}×")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    summary = {
        'label': label,
        'n_tryptophans': model.n_tryptophans,
        'peak_field_kT': max(cascade_metrics['em_field_kT']),
        'peak_k_enhancement': max(cascade_metrics['k_enhancement']),
        'peak_dimers': max(cascade_metrics['dimer_count']),
        'peak_coherence': max(cascade_metrics['coherence']),
        'peak_eligibility': max(cascade_metrics['eligibility']),
        'peak_dopamine_nM': max(cascade_metrics['dopamine_nM']),
        'gate_opened': any(cascade_metrics['plasticity_gate']),
        'committed': any(cascade_metrics['committed']),
        'committed_level': max(cascade_metrics['committed_level']),
        'final_strength': final_strength,
    }
    
    return summary, cascade_metrics


def check_gate_conditions(summary):
    """
    Check which gate conditions are met/failed
    """
    print("\n" + "=" * 60)
    print("GATE CONDITION ANALYSIS")
    print("=" * 60)
    
    # Thresholds from model6_core.py
    eligibility_threshold = 0.3
    calcium_threshold_uM = 0.5
    field_threshold_kT = 10.0
    dimer_threshold = 30
    dopamine_read_threshold = 50.0  # nM
    
    print(f"\n{'Condition':<25} {'Value':<15} {'Threshold':<15} {'Status':<10}")
    print("-" * 65)
    
    # Eligibility
    elig = summary['peak_eligibility']
    status = "✓ PASS" if elig > eligibility_threshold else "✗ FAIL"
    print(f"{'Eligibility':<25} {elig:<15.3f} {eligibility_threshold:<15.1f} {status}")
    
    # Field
    field = summary['peak_field_kT']
    status = "✓ PASS" if field > field_threshold_kT else "✗ FAIL"
    print(f"{'Q1 Field (kT)':<25} {field:<15.1f} {field_threshold_kT:<15.1f} {status}")
    
    # Dopamine
    da = summary['peak_dopamine_nM']
    status = "✓ PASS" if da > dopamine_read_threshold else "✗ FAIL"
    print(f"{'Dopamine (nM)':<25} {da:<15.1f} {dopamine_read_threshold:<15.1f} {status}")
    
    # Dimers (not directly in gate, but influences eligibility)
    dimers = summary['peak_dimers']
    status = "✓ PASS" if dimers >= 3 else "✗ FAIL"  # dimer_threshold_min
    print(f"{'Dimer count':<25} {dimers:<15.1f} {'3.0':<15} {status}")
    
    # Coherence (not directly in gate, but influences eligibility)
    coh = summary['peak_coherence']
    status = "✓ PASS" if coh > 0.5 else "✗ FAIL"  # coherence_threshold
    print(f"{'Coherence':<25} {coh:<15.3f} {'0.5':<15} {status}")
    
    print("\n" + "-" * 65)
    print(f"{'Gate Opened:':<25} {summary['gate_opened']}")
    print(f"{'Committed:':<25} {summary['committed']}")
    print(f"{'Final Strength:':<25} {summary['final_strength']:.3f}×")


def main():
    """Run diagnostic for both MT+ and MT- conditions"""
    
    # Run MT- condition
    summary_minus, metrics_minus = run_cascade_diagnostic(mt_invaded=False)
    
    # Run MT+ condition  
    summary_plus, metrics_plus = run_cascade_diagnostic(mt_invaded=True)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("CASCADE COMPARISON: MT- vs MT+")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'MT-':<15} {'MT+':<15} {'Ratio':<10}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('N tryptophans', 'n_tryptophans'),
        ('Peak Q1 Field (kT)', 'peak_field_kT'),
        ('Peak k_enhancement', 'peak_k_enhancement'),
        ('Peak Dimers', 'peak_dimers'),
        ('Peak Coherence', 'peak_coherence'),
        ('Peak Eligibility', 'peak_eligibility'),
        ('Peak Dopamine (nM)', 'peak_dopamine_nM'),
        ('Committed Level', 'committed_level'),
        ('Final Strength', 'final_strength'),
    ]
    
    for display_name, key in metrics_to_compare:
        val_minus = summary_minus[key]
        val_plus = summary_plus[key]
        
        if isinstance(val_minus, bool):
            print(f"{display_name:<25} {str(val_minus):<15} {str(val_plus):<15} {'--':<10}")
        elif val_minus > 0:
            ratio = val_plus / val_minus
            print(f"{display_name:<25} {val_minus:<15.3f} {val_plus:<15.3f} {ratio:<10.2f}×")
        else:
            print(f"{display_name:<25} {val_minus:<15.3f} {val_plus:<15.3f} {'--':<10}")
    
    # Boolean comparisons
    print(f"\n{'Gate Opened':<25} {str(summary_minus['gate_opened']):<15} {str(summary_plus['gate_opened']):<15}")
    print(f"{'Committed':<25} {str(summary_minus['committed']):<15} {str(summary_plus['committed']):<15}")
    
    # =========================================================================
    # BOTTLENECK IDENTIFICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 80)
    
    # Check each condition for MT+ (should be the "working" case)
    check_gate_conditions(summary_plus)
    
    # Identify the cascade break point
    print("\n" + "=" * 80)
    print("CASCADE FLOW ANALYSIS")
    print("=" * 80)
    
    cascade_steps = [
        ("1. Tryptophans", summary_plus['n_tryptophans'] > 400, 
         f"{summary_plus['n_tryptophans']} (need >400 for MT+)"),
        ("2. Q1 Field", summary_plus['peak_field_kT'] > 10, 
         f"{summary_plus['peak_field_kT']:.1f} kT (need >10)"),
        ("3. k_enhancement", summary_plus['peak_k_enhancement'] > 1.5, 
         f"{summary_plus['peak_k_enhancement']:.2f}× (need >1.5)"),
        ("4. Dimer formation", summary_plus['peak_dimers'] >= 3, 
         f"{summary_plus['peak_dimers']:.1f} (need ≥3)"),
        ("5. Coherence", summary_plus['peak_coherence'] > 0.5, 
         f"{summary_plus['peak_coherence']:.3f} (need >0.5)"),
        ("6. Eligibility", summary_plus['peak_eligibility'] > 0.3, 
         f"{summary_plus['peak_eligibility']:.3f} (need >0.3)"),
        ("7. Dopamine read", summary_plus['peak_dopamine_nM'] > 50, 
         f"{summary_plus['peak_dopamine_nM']:.1f} nM (need >50)"),
        ("8. Gate opens", summary_plus['gate_opened'], 
         f"{summary_plus['gate_opened']}"),
        ("9. Commitment", summary_plus['committed'], 
         f"{summary_plus['committed']} (level: {summary_plus['committed_level']:.3f})"),
        ("10. Strength change", summary_plus['final_strength'] > 1.05, 
         f"{summary_plus['final_strength']:.3f}× (need >1.05)"),
    ]
    
    print(f"\n{'Step':<25} {'Status':<10} {'Value':<40}")
    print("-" * 75)
    
    first_failure = None
    for step_name, passed, value in cascade_steps:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{step_name:<25} {status:<10} {value:<40}")
        
        if not passed and first_failure is None:
            first_failure = step_name
    
    if first_failure:
        print(f"\n>>> BOTTLENECK IDENTIFIED: {first_failure}")
        print("    The cascade breaks at this step!")
    else:
        print(f"\n>>> All steps passed - cascade should be working!")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if summary_plus['peak_eligibility'] <= 0.3:
        print("""
ELIGIBILITY TOO LOW:
- Eligibility is derived from coherence in quantum_coherence.py
- Check if coherence is being calculated correctly
- Verify dimer count conversion (741 nM → 4-5 dimers)
- The eligibility threshold (0.3) may be too high for the current dimer count
""")
    
    if summary_plus['peak_dimers'] < 3:
        print("""
DIMER COUNT TOO LOW:
- Check dimer_particles.py formation rate
- Verify template enhancement is active
- Confirm calcium is reaching templates
- Check k_agg_for_next_step is being used
""")
    
    if not summary_plus['gate_opened']:
        print("""
GATE NOT OPENING:
- All four conditions must be met simultaneously:
  1. eligibility > 0.3
  2. dopamine > 50 nM (read threshold)
  3. calcium > 0.5 μM
  4. field > 10 kT
- Check timing: dopamine must arrive while other conditions are met
""")
    
    if summary_plus['gate_opened'] and not summary_plus['committed']:
        print("""
GATE OPENS BUT NO COMMITMENT:
- Check _camkii_committed flag setting in model6_core.py
- Verify the commitment logic in PHASE 10
- Check if committed_memory_level is being set
""")
    
    return summary_minus, summary_plus


if __name__ == "__main__":
    main()