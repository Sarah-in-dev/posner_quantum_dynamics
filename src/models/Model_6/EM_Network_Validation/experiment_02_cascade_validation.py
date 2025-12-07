"""
Experiment 02: EM Network Cascade Validation
=============================================

UPDATED: December 7, 2025 - Based on validated cascade architecture

Tests the physics-derived cascade predictions:
1. ISOTOPE EFFECT: P31 vs P32 coherence and cascade activation
2. SYNAPSE THRESHOLD: Sharp transition at N=8-10 synapses
3. ANESTHETIC DISRUPTION: Blocks tryptophan coupling
4. FEEDBACK QUANTIFICATION: EM enhancement of dimer formation

Key Validated Predictions (from cascade_architecture_documentation.md):
-----------------------------------------------------------------------
- Dimers per synapse: 4-6 (we get 5.3)
- Network modulation threshold: 5.0 (we get 10.0 at N=10)
- Collective field: 20-24 kT (we get 22.0)
- Feedback enhancement: 1.3-1.5× dimer increase (we get 1.36×)
- P31 T2: >50s, P32 T2: <5s (nuclear spin difference)

What This Experiment Tests:
---------------------------
1. Does P32 isotope disrupt cascade activation? (Definitive quantum test)
2. Does threshold emerge at N=8-10 synapses? (Network coordination)
3. Does anesthetic block the cascade? (Tryptophan involvement)
4. Is feedback loop functioning? (EM → chemistry coupling)

Author: Sarah Davidson
Date: December 7, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

EXPERIMENT_NAME = "EM Network Cascade Validation"
EXPERIMENT_NUMBER = "02"

# Focused experimental design (not full factorial - targeted tests)
EXPERIMENTS = {
    'A': {
        'name': 'Isotope Comparison',
        'description': 'P31 vs P32 - the definitive quantum test',
        'variables': {
            'isotope': ['P31', 'P32'],
            'n_synapses': [10],  # At threshold
        },
        'n_replicates': 5,
    },
    'B': {
        'name': 'Synapse Threshold Sweep',
        'description': 'Find activation threshold at N=8-10',
        'variables': {
            'n_synapses': [1, 3, 5, 7, 8, 9, 10, 12, 15, 20],
            'isotope': ['P31'],  # Quantum-enabled
        },
        'n_replicates': 3,
    },
    'C': {
        'name': 'Anesthetic Disruption',
        'description': 'Block tryptophan coupling',
        'variables': {
            'anesthetic': [False, True],
            'isotope': ['P31', 'P32'],
            'n_synapses': [10],
        },
        'n_replicates': 3,
    },
    'D': {
        'name': 'Feedback Quantification',
        'description': 'EM ON vs OFF dimer enhancement',
        'variables': {
            'em_enabled': [False, True],
            'n_synapses': [10],
            'isotope': ['P31'],
        },
        'n_replicates': 5,
    },
}

# Burst protocol (validated in cascade tests)
BURST_PROTOCOL = {
    'n_bursts': 20,
    'burst_duration_ms': 30,
    'inter_burst_interval_ms': 150,
    'baseline_steps': 20,
    'recovery_steps': 300,
}

# Success criteria (from validated cascade)
SUCCESS_CRITERIA = {
    'dimers_per_synapse': (4, 8),          # Target: 4-6, allow 4-8
    'network_modulation_threshold': 5.0,    # Must exceed for activation
    'collective_field_kT': (18, 28),        # Target: 20-24, allow 18-28
    'T2_P31_min': 50,                       # P31 should have T2 > 50s
    'T2_P32_max': 10,                       # P32 should have T2 < 10s
    'feedback_enhancement_min': 1.1,        # At least 10% more dimers with EM
    'anesthetic_reduction': 0.5,            # Should reduce field by >50%
    'threshold_n_synapses': (7, 12),        # Threshold should be in this range
}

# Output setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / "results" / f"exp02_cascade_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"EXPERIMENT {EXPERIMENT_NUMBER}: {EXPERIMENT_NAME.upper()}")
print("="*80)
print(f"\nBased on validated cascade architecture (Dec 7, 2025)")
print(f"\nExperiments:")
for key, exp in EXPERIMENTS.items():
    n_conditions = np.prod([len(v) for v in exp['variables'].values()])
    n_runs = n_conditions * exp['n_replicates']
    print(f"  {key}. {exp['name']}: {n_conditions} conditions × {exp['n_replicates']} reps = {n_runs} runs")

total_runs = sum(
    np.prod([len(v) for v in exp['variables'].values()]) * exp['n_replicates']
    for exp in EXPERIMENTS.values()
)
print(f"\nTotal runs: {total_runs}")
print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# PARAMETER CONFIGURATION
# =============================================================================

def configure_parameters(
    n_synapses: int = 10,
    isotope: str = 'P31',
    anesthetic: bool = False,
    em_enabled: bool = True,
    temperature: float = 310.0,
) -> Model6Parameters:
    """
    Configure model parameters for experimental condition.
    
    Args:
        n_synapses: Number of active synapses (1-20)
        isotope: 'P31' or 'P32'
        anesthetic: True to apply isoflurane (blocks tryptophan)
        em_enabled: True to enable EM coupling
        temperature: Temperature in Kelvin
        
    Returns:
        Configured Model6Parameters
    """
    params = Model6Parameters()
    
    # === TIMESTEP ===
    params.simulation.dt_diffusion = 1e-3  # 1 ms
    
    # === EM COUPLING ===
    params.em_coupling_enabled = em_enabled
    params.multi_synapse_enabled = True
    params.multi_synapse.n_synapses_default = n_synapses
    
    # === ISOTOPE ===
    if isotope == 'P31':
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
    elif isotope == 'P32':
        params.environment.fraction_P31 = 0.0
        params.environment.fraction_P32 = 1.0
    else:
        raise ValueError(f"Unknown isotope: {isotope}")
    
    # === ANESTHETIC ===
    if anesthetic:
        params.tryptophan.anesthetic_applied = True
        params.tryptophan.anesthetic_type = 'isoflurane'
        params.tryptophan.anesthetic_blocking_factor = 0.9  # 90% block
    
    # === TEMPERATURE ===
    params.environment.T = temperature
    
    return params

# =============================================================================
# RUN SINGLE CONDITION
# =============================================================================

def run_condition(params: Model6Parameters, condition_name: str = "") -> Dict:
    """
    Run one experimental condition with burst protocol.
    
    Returns dict with all relevant metrics.
    """
    # Initialize model
    model = Model6QuantumSynapse(params)
    
    dt = model.dt
    
    # === BASELINE ===
    for _ in range(BURST_PROTOCOL['baseline_steps']):
        model.step(dt, stimulus={'voltage': -70e-3})
    
    # === BURST PROTOCOL ===
    for burst_num in range(BURST_PROTOCOL['n_bursts']):
        # Active phase (depolarized)
        for _ in range(BURST_PROTOCOL['burst_duration_ms']):
            model.step(dt, stimulus={'voltage': -20e-3, 'activity_level': 0.8})
        
        # Rest phase
        if burst_num < BURST_PROTOCOL['n_bursts'] - 1:
            for _ in range(BURST_PROTOCOL['inter_burst_interval_ms']):
                model.step(dt, stimulus={'voltage': -70e-3})
    
    # === RECOVERY ===
    for _ in range(BURST_PROTOCOL['recovery_steps']):
        model.step(dt, stimulus={'voltage': -70e-3})
    
    # === COLLECT METRICS ===
    metrics = model.get_experimental_metrics()
    
    # Core cascade metrics
    result = {
        'condition': condition_name,
        'dimer_peak_nM': metrics.get('dimer_peak_nM_ct', 0),
        'dimer_mean_nM': metrics.get('dimer_mean_nM_ct', 0),
        'T2_dimer_s': metrics.get('T2_dimer_s', 0),
        'coherence_mean': metrics.get('coherence_dimer_mean', 0),
    }
    
    # EM-specific metrics (if enabled)
    if params.em_coupling_enabled:
        result['network_modulation'] = getattr(model, '_network_modulation', 0)
        result['collective_field_kT'] = getattr(model, '_collective_field_kT', 0)
        result['above_threshold'] = result['network_modulation'] >= 5.0
        
        # k_enhancement from history
        if hasattr(model, '_k_enhancement_history') and len(model._k_enhancement_history) > 0:
            result['k_enhancement_mean'] = float(np.mean(model._k_enhancement_history))
            result['k_enhancement_max'] = float(np.max(model._k_enhancement_history))
        else:
            result['k_enhancement_mean'] = 1.0
            result['k_enhancement_max'] = 1.0
    else:
        result['network_modulation'] = 0
        result['collective_field_kT'] = 0
        result['above_threshold'] = False
        result['k_enhancement_mean'] = 1.0
        result['k_enhancement_max'] = 1.0
    
    # Calculate dimers per synapse
    n_syn = params.multi_synapse.n_synapses_default
    result['dimers_per_synapse'] = result['dimer_peak_nM'] * 0.006 / n_syn
    
    return result

# =============================================================================
# EXPERIMENT A: ISOTOPE COMPARISON
# =============================================================================

def run_experiment_A() -> Dict:
    """
    P31 vs P32 isotope comparison - the definitive quantum test.
    
    Prediction: P31 maintains coherence (T2 > 50s), P32 loses it (T2 < 5s)
    This should affect cascade activation.
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: ISOTOPE COMPARISON (P31 vs P32)")
    print("="*80)
    print("\nThis is the DEFINITIVE test for quantum effects.")
    print("P31 (spin-1/2) should maintain coherence, P32 (spin-1) should not.")
    
    results = {'P31': [], 'P32': []}
    n_reps = EXPERIMENTS['A']['n_replicates']
    
    for isotope in ['P31', 'P32']:
        print(f"\n  Running {isotope} ({n_reps} replicates)...")
        
        for rep in range(n_reps):
            # Different seed per replicate, same across isotopes for pair comparison
            np.random.seed(42 + rep)
            
            params = configure_parameters(
                n_synapses=10,
                isotope=isotope,
                em_enabled=True,
            )
            
            result = run_condition(params, f"{isotope}_rep{rep}")
            result['isotope'] = isotope
            result['replicate'] = rep
            results[isotope].append(result)
            
            print(f"    Rep {rep+1}: T2={result['T2_dimer_s']:.1f}s, "
                  f"field={result['collective_field_kT']:.1f}kT, "
                  f"dimers/syn={result['dimers_per_synapse']:.1f}")
    
    # === ANALYSIS ===
    print("\n--- Analysis ---")
    
    p31_T2 = np.mean([r['T2_dimer_s'] for r in results['P31']])
    p32_T2 = np.mean([r['T2_dimer_s'] for r in results['P32']])
    p31_field = np.mean([r['collective_field_kT'] for r in results['P31']])
    p32_field = np.mean([r['collective_field_kT'] for r in results['P32']])
    p31_dimers = np.mean([r['dimers_per_synapse'] for r in results['P31']])
    p32_dimers = np.mean([r['dimers_per_synapse'] for r in results['P32']])
    
    print(f"\n  P31: T2={p31_T2:.1f}s, field={p31_field:.1f}kT, dimers/syn={p31_dimers:.1f}")
    print(f"  P32: T2={p32_T2:.1f}s, field={p32_field:.1f}kT, dimers/syn={p32_dimers:.1f}")
    
    # Success criteria
    t2_pass = p31_T2 > SUCCESS_CRITERIA['T2_P31_min'] and p32_T2 < SUCCESS_CRITERIA['T2_P32_max']
    
    if t2_pass:
        print(f"\n  ✓ PASS: P31 T2 ({p31_T2:.1f}s) > {SUCCESS_CRITERIA['T2_P31_min']}s")
        print(f"  ✓ PASS: P32 T2 ({p32_T2:.1f}s) < {SUCCESS_CRITERIA['T2_P32_max']}s")
    else:
        if p31_T2 <= SUCCESS_CRITERIA['T2_P31_min']:
            print(f"  ✗ FAIL: P31 T2 ({p31_T2:.1f}s) should be > {SUCCESS_CRITERIA['T2_P31_min']}s")
        if p32_T2 >= SUCCESS_CRITERIA['T2_P32_max']:
            print(f"  ✗ FAIL: P32 T2 ({p32_T2:.1f}s) should be < {SUCCESS_CRITERIA['T2_P32_max']}s")
    
    return {
        'results': results,
        'summary': {
            'p31_T2_mean': p31_T2,
            'p32_T2_mean': p32_T2,
            'p31_field_mean': p31_field,
            'p32_field_mean': p32_field,
            'p31_dimers_mean': p31_dimers,
            'p32_dimers_mean': p32_dimers,
            't2_test_passed': t2_pass,
        }
    }

# =============================================================================
# EXPERIMENT B: SYNAPSE THRESHOLD SWEEP
# =============================================================================

def run_experiment_B() -> Dict:
    """
    Sweep N_synapses from 1-20 to find activation threshold.
    
    Prediction: Sharp transition at N=8-10 synapses where network_modulation
    crosses threshold (5.0) and collective field activates (~20 kT).
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: SYNAPSE THRESHOLD SWEEP")
    print("="*80)
    print("\nPrediction: Threshold at N=8-10 synapses (network_modulation >= 5.0)")
    
    n_synapses_list = EXPERIMENTS['B']['variables']['n_synapses']
    n_reps = EXPERIMENTS['B']['n_replicates']
    
    results = []
    
    for n_syn in n_synapses_list:
        print(f"\n  N={n_syn} synapses ({n_reps} replicates)...")
        
        syn_results = []
        for rep in range(n_reps):
            np.random.seed(42 + rep)
            
            params = configure_parameters(
                n_synapses=n_syn,
                isotope='P31',
                em_enabled=True,
            )
            
            result = run_condition(params, f"N{n_syn}_rep{rep}")
            result['n_synapses'] = n_syn
            result['replicate'] = rep
            syn_results.append(result)
        
        # Average across replicates
        avg_modulation = np.mean([r['network_modulation'] for r in syn_results])
        avg_field = np.mean([r['collective_field_kT'] for r in syn_results])
        above = avg_modulation >= 5.0
        
        print(f"    → modulation={avg_modulation:.2f}, field={avg_field:.1f}kT, "
              f"above_threshold={'✓' if above else '✗'}")
        
        results.append({
            'n_synapses': n_syn,
            'network_modulation_mean': avg_modulation,
            'network_modulation_std': np.std([r['network_modulation'] for r in syn_results]),
            'collective_field_mean': avg_field,
            'collective_field_std': np.std([r['collective_field_kT'] for r in syn_results]),
            'above_threshold': above,
            'raw_results': syn_results,
        })
    
    # === FIND THRESHOLD ===
    print("\n--- Threshold Analysis ---")
    
    # Find first N where above_threshold = True
    threshold_n = None
    for r in results:
        if r['above_threshold']:
            threshold_n = r['n_synapses']
            break
    
    if threshold_n:
        print(f"\n  Threshold detected at N={threshold_n} synapses")
        
        expected_range = SUCCESS_CRITERIA['threshold_n_synapses']
        if expected_range[0] <= threshold_n <= expected_range[1]:
            print(f"  ✓ PASS: Within expected range ({expected_range[0]}-{expected_range[1]})")
        else:
            print(f"  ✗ FAIL: Outside expected range ({expected_range[0]}-{expected_range[1]})")
    else:
        print(f"\n  ✗ FAIL: No threshold detected (cascade never activated)")
        threshold_n = -1
    
    return {
        'results': results,
        'threshold_n': threshold_n,
    }

# =============================================================================
# EXPERIMENT C: ANESTHETIC DISRUPTION
# =============================================================================

def run_experiment_C() -> Dict:
    """
    Test anesthetic disruption of tryptophan coupling.
    
    Prediction: Anesthetic blocks cascade activation, eliminating P31/P32 difference.
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: ANESTHETIC DISRUPTION")
    print("="*80)
    print("\nPrediction: Anesthetic blocks tryptophan coupling, prevents cascade activation")
    
    n_reps = EXPERIMENTS['C']['n_replicates']
    
    results = {
        'P31_control': [],
        'P32_control': [],
        'P31_anesthetic': [],
        'P32_anesthetic': [],
    }
    
    for isotope in ['P31', 'P32']:
        for anesthetic in [False, True]:
            key = f"{isotope}_{'anesthetic' if anesthetic else 'control'}"
            print(f"\n  {key} ({n_reps} replicates)...")
            
            for rep in range(n_reps):
                np.random.seed(42 + rep)
                
                params = configure_parameters(
                    n_synapses=10,
                    isotope=isotope,
                    anesthetic=anesthetic,
                    em_enabled=True,
                )
                
                result = run_condition(params, f"{key}_rep{rep}")
                result['isotope'] = isotope
                result['anesthetic'] = anesthetic
                result['replicate'] = rep
                results[key].append(result)
            
            avg_field = np.mean([r['collective_field_kT'] for r in results[key]])
            print(f"    → avg collective_field = {avg_field:.1f} kT")
    
    # === ANALYSIS ===
    print("\n--- Analysis ---")
    
    p31_ctrl_field = np.mean([r['collective_field_kT'] for r in results['P31_control']])
    p31_anes_field = np.mean([r['collective_field_kT'] for r in results['P31_anesthetic']])
    p32_ctrl_field = np.mean([r['collective_field_kT'] for r in results['P32_control']])
    p32_anes_field = np.mean([r['collective_field_kT'] for r in results['P32_anesthetic']])
    
    print(f"\n  Control:    P31={p31_ctrl_field:.1f}kT, P32={p32_ctrl_field:.1f}kT")
    print(f"  Anesthetic: P31={p31_anes_field:.1f}kT, P32={p32_anes_field:.1f}kT")
    
    # Reduction ratio
    if p31_ctrl_field > 0:
        reduction = 1.0 - (p31_anes_field / p31_ctrl_field)
        print(f"\n  P31 field reduction: {reduction*100:.0f}%")
        
        if reduction >= SUCCESS_CRITERIA['anesthetic_reduction']:
            print(f"  ✓ PASS: Reduction >= {SUCCESS_CRITERIA['anesthetic_reduction']*100:.0f}%")
        else:
            print(f"  ✗ FAIL: Reduction < {SUCCESS_CRITERIA['anesthetic_reduction']*100:.0f}%")
    else:
        reduction = 0
        print("  ✗ FAIL: Control field was 0")
    
    return {
        'results': results,
        'summary': {
            'p31_control_field': p31_ctrl_field,
            'p31_anesthetic_field': p31_anes_field,
            'p32_control_field': p32_ctrl_field,
            'p32_anesthetic_field': p32_anes_field,
            'reduction_fraction': reduction,
        }
    }

# =============================================================================
# EXPERIMENT D: FEEDBACK QUANTIFICATION
# =============================================================================

def run_experiment_D() -> Dict:
    """
    Quantify EM feedback loop: compare dimer counts with EM ON vs OFF.
    
    Runs PAIRED comparisons (OFF then ON for each rep) to match test_feedback_loop.py
    """
    print("\n" + "="*80)
    print("EXPERIMENT D: FEEDBACK QUANTIFICATION")
    print("="*80)
    print("\nPrediction: EM coupling enhances dimer formation by 1.3-1.5×")
    
    n_reps = EXPERIMENTS['D']['n_replicates']
    
    results = {'em_off': [], 'em_on': []}
    
    for rep in range(n_reps):
        seed = 42 + rep
        print(f"\n  Pair {rep+1}/{n_reps} (seed={seed})...")
        
        # --- EM OFF ---
        np.random.seed(seed)
        
        params_off = configure_parameters(
            n_synapses=10,
            isotope='P31',
            em_enabled=False,
        )
        
        result_off = run_condition(params_off, f"em_off_rep{rep}")
        result_off['em_enabled'] = False
        result_off['replicate'] = rep
        results['em_off'].append(result_off)
        
        # --- EM ON (don't reset seed - continues from EM OFF) ---
        params_on = configure_parameters(
            n_synapses=10,
            isotope='P31',
            em_enabled=True,
        )
        
        result_on = run_condition(params_on, f"em_on_rep{rep}")
        result_on['em_enabled'] = True
        result_on['replicate'] = rep
        results['em_on'].append(result_on)
        
        # Report paired result
        pair_enh = result_on['dimer_peak_nM'] / result_off['dimer_peak_nM'] if result_off['dimer_peak_nM'] > 0 else 0
        print(f"    OFF: {result_off['dimer_peak_nM']:.0f}nM, ON: {result_on['dimer_peak_nM']:.0f}nM, "
              f"Enhancement: {pair_enh:.2f}×, k_enh: {result_on.get('k_enhancement_mean', 1.0):.2f}×")
    
    # === ANALYSIS ===
    print("\n--- Analysis ---")
    
    off_dimers = np.mean([r['dimer_peak_nM'] for r in results['em_off']])
    on_dimers = np.mean([r['dimer_peak_nM'] for r in results['em_on']])
    off_std = np.std([r['dimer_peak_nM'] for r in results['em_off']])
    on_std = np.std([r['dimer_peak_nM'] for r in results['em_on']])
    
    enhancement = on_dimers / off_dimers if off_dimers > 0 else 0
    k_enh_mean = np.mean([r.get('k_enhancement_mean', 1.0) for r in results['em_on']])
    
    print(f"\n  EM OFF: {off_dimers:.0f} ± {off_std:.0f} nM")
    print(f"  EM ON:  {on_dimers:.0f} ± {on_std:.0f} nM")
    print(f"  Enhancement: {enhancement:.2f}×")
    print(f"  k_enhancement (calculated): {k_enh_mean:.2f}×")
    
    if enhancement >= SUCCESS_CRITERIA['feedback_enhancement_min']:
        print(f"\n  ✓ PASS: Enhancement >= {SUCCESS_CRITERIA['feedback_enhancement_min']}×")
    else:
        print(f"\n  ✗ FAIL: Enhancement < {SUCCESS_CRITERIA['feedback_enhancement_min']}×")
    
    return {
        'results': results,
        'summary': {
            'em_off_dimers_mean': float(off_dimers),
            'em_off_dimers_std': float(off_std),
            'em_on_dimers_mean': float(on_dimers),
            'em_on_dimers_std': float(on_std),
            'dimer_enhancement': float(enhancement),
            'k_enhancement_mean': float(k_enh_mean),
        }
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_summary_figure(exp_a: Dict, exp_b: Dict, exp_c: Dict, exp_d: Dict):
    """Create 2×2 summary figure with all experiment results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Experiment 02: EM Network Cascade Validation\n(December 7, 2025)',
                 fontsize=14, fontweight='bold')
    
    # === PANEL A: Isotope Comparison ===
    ax = axes[0, 0]
    
    isotopes = ['P31', 'P32']
    T2_means = [exp_a['summary']['p31_T2_mean'], exp_a['summary']['p32_T2_mean']]
    field_means = [exp_a['summary']['p31_field_mean'], exp_a['summary']['p32_field_mean']]
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, T2_means, width, label='T2 (s)', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, field_means, width, label='Field (kT)', color='green', alpha=0.7)
    
    ax.axhline(SUCCESS_CRITERIA['T2_P31_min'], color='blue', linestyle='--', alpha=0.5)
    ax.set_ylabel('T2 Coherence Time (s)', color='blue')
    ax2.set_ylabel('Collective Field (kT)', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(['P31\n(spin-1/2)', 'P32\n(spin-1)'])
    ax.set_title('A. Isotope Comparison', fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # === PANEL B: Synapse Threshold ===
    ax = axes[0, 1]
    
    n_syn = [r['n_synapses'] for r in exp_b['results']]
    modulation = [r['network_modulation_mean'] for r in exp_b['results']]
    field = [r['collective_field_mean'] for r in exp_b['results']]
    
    ax.plot(n_syn, modulation, 'o-', color='purple', linewidth=2, markersize=8, label='Network Modulation')
    ax.axhline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (5.0)')
    ax.axvspan(7, 12, alpha=0.2, color='green', label='Expected threshold zone')
    
    ax.set_xlabel('Number of Active Synapses')
    ax.set_ylabel('Network Modulation')
    ax.set_title('B. Synapse Threshold Sweep', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # === PANEL C: Anesthetic Disruption ===
    ax = axes[1, 0]
    
    conditions = ['P31\nControl', 'P31\nAnesthetic', 'P32\nControl', 'P32\nAnesthetic']
    fields = [
        exp_c['summary']['p31_control_field'],
        exp_c['summary']['p31_anesthetic_field'],
        exp_c['summary']['p32_control_field'],
        exp_c['summary']['p32_anesthetic_field'],
    ]
    colors = ['blue', 'lightblue', 'orange', 'moccasin']
    
    ax.bar(range(4), fields, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(20, color='red', linestyle='--', alpha=0.5, label='Functional threshold')
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Collective Field (kT)')
    ax.set_title('C. Anesthetic Disruption', fontweight='bold')
    ax.legend()
    
    # === PANEL D: Feedback Enhancement ===
    ax = axes[1, 1]
    
    conditions = ['EM OFF\n(Baseline)', 'EM ON\n(Enhanced)']
    dimers = [exp_d['summary']['em_off_dimers_mean'], exp_d['summary']['em_on_dimers_mean']]
    errors = [exp_d['summary']['em_off_dimers_std'], exp_d['summary']['em_on_dimers_std']]
    colors = ['gray', 'green']
    
    ax.bar(range(2), dimers, yerr=errors, color=colors, edgecolor='black', 
           linewidth=1.5, capsize=5, alpha=0.7)
    
    # Add enhancement annotation
    enh = exp_d['summary']['dimer_enhancement']
    ax.annotate(f'{enh:.2f}×', xy=(0.5, max(dimers)), fontsize=14, fontweight='bold',
                ha='center', va='bottom')
    
    ax.set_xticks(range(2))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Peak Dimer Concentration (nM)')
    ax.set_title('D. Feedback Enhancement', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_DIR / 'experiment_02_summary.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved: {fig_path}")
    
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_experiments():
    """Run all experiments and generate report."""
    
    print("\n" + "="*80)
    print("STARTING ALL EXPERIMENTS")
    print("="*80)
    
    # Run experiments
    exp_a = run_experiment_A()
    exp_b = run_experiment_B()
    exp_c = run_experiment_C()
    exp_d = run_experiment_D()
    
    # Create visualization
    create_summary_figure(exp_a, exp_b, exp_c, exp_d)
    
    # === SUMMARY REPORT ===
    print("\n" + "="*80)
    print("EXPERIMENT 02 SUMMARY")
    print("="*80)
    
    results_summary = []
    
    # A: Isotope
    a_pass = exp_a['summary']['t2_test_passed']
    results_summary.append(('A. Isotope Comparison', a_pass))
    print(f"\n{'✓' if a_pass else '✗'} A. Isotope Comparison: "
          f"P31 T2={exp_a['summary']['p31_T2_mean']:.1f}s, P32 T2={exp_a['summary']['p32_T2_mean']:.1f}s")
    
    # B: Threshold
    threshold_range = SUCCESS_CRITERIA['threshold_n_synapses']
    b_pass = threshold_range[0] <= exp_b['threshold_n'] <= threshold_range[1]
    results_summary.append(('B. Synapse Threshold', b_pass))
    print(f"{'✓' if b_pass else '✗'} B. Synapse Threshold: N={exp_b['threshold_n']} "
          f"(expected {threshold_range[0]}-{threshold_range[1]})")
    
    # C: Anesthetic
    c_pass = exp_c['summary']['reduction_fraction'] >= SUCCESS_CRITERIA['anesthetic_reduction']
    results_summary.append(('C. Anesthetic Disruption', c_pass))
    print(f"{'✓' if c_pass else '✗'} C. Anesthetic Disruption: "
          f"{exp_c['summary']['reduction_fraction']*100:.0f}% reduction "
          f"(need >={SUCCESS_CRITERIA['anesthetic_reduction']*100:.0f}%)")
    
    # D: Feedback
    d_pass = exp_d['summary']['dimer_enhancement'] >= SUCCESS_CRITERIA['feedback_enhancement_min']
    results_summary.append(('D. Feedback Enhancement', d_pass))
    print(f"{'✓' if d_pass else '✗'} D. Feedback Enhancement: "
          f"{exp_d['summary']['dimer_enhancement']:.2f}× "
          f"(need >={SUCCESS_CRITERIA['feedback_enhancement_min']}×)")
    
    # Overall
    n_passed = sum(1 for _, passed in results_summary if passed)
    n_total = len(results_summary)
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {n_passed}/{n_total} experiments passed")
    print(f"{'='*80}")
    
    if n_passed == n_total:
        print("\n✓✓✓ CASCADE ARCHITECTURE FULLY VALIDATED ✓✓✓")
    else:
        print("\n⚠ Some experiments failed - investigate results")
    
    # === SAVE RESULTS ===
    all_results = {
        'experiment_A': {
            'name': 'Isotope Comparison',
            'summary': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in exp_a['summary'].items()},
            'passed': bool(a_pass),
        },
        'experiment_B': {
            'name': 'Synapse Threshold',
            'threshold_n': int(exp_b['threshold_n']),
            'passed': bool(b_pass),
        },
        'experiment_C': {
            'name': 'Anesthetic Disruption',
            'summary': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in exp_c['summary'].items()},
            'passed': bool(c_pass),
        },
        'experiment_D': {
            'name': 'Feedback Enhancement',
            'summary': exp_d['summary'],  # Already converted to float in run_experiment_D
            'passed': bool(d_pass),
        },
        'overall': {
            'passed': int(n_passed),
            'total': int(n_total),
            'success_criteria': {k: list(v) if isinstance(v, tuple) else v 
                                for k, v in SUCCESS_CRITERIA.items()},
        }
    }
    
    results_path = OUTPUT_DIR / 'results_summary.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved: {results_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_experiments()