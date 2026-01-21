"""
Paired EM Comparison Test
=========================

PROBLEM: Panel D shows 0.87× (EM ON produces FEWER dimers than EM OFF)
         But cascade architecture predicts 1.3-1.5× ENHANCEMENT

HYPOTHESIS: Different random seeds mask the enhancement signal
            - EM OFF uses seeds: 42, 43, 44...
            - EM ON uses seeds: 100, 101, 102...
            - Stochastic variation > enhancement signal

THIS TEST: Run EM ON and EM OFF with IDENTICAL seeds
           If enhancement is real, we'll see it clearly

Place this file in: src/models/Model_6/EM_Network_Validation/
Run with: python test_paired_em_comparison.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*70)
print("PAIRED EM COMPARISON TEST")
print("="*70)
print("\nGoal: Determine if EM enhancement is real but masked by seed differences")

# =============================================================================
# CONFIGURATION
# =============================================================================

BURST_PROTOCOL = {
    'n_bursts': 5,
    'burst_duration_ms': 30,
    'inter_burst_interval_ms': 150,
    'baseline_steps': 20,
    'recovery_steps': 300,
}

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def run_simulation(em_enabled: bool, seed: int) -> dict:
    """
    Run a single simulation with specified EM setting and seed.
    
    Parameters:
    -----------
    em_enabled : bool
        Whether EM coupling is enabled
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict with metrics
    """
    # CRITICAL: Set seed BEFORE creating model
    np.random.seed(seed)
    
    # Configure parameters
    params = Model6Parameters()
    params.simulation.dt_diffusion = 1e-3  # 1 ms
    params.em_coupling_enabled = em_enabled
    params.multi_synapse_enabled = True
    params.multi_synapse.n_synapses_default = 10
    params.environment.fraction_P31 = 1.0
    params.environment.fraction_P32 = 0.0
    
    # Create model
    model = Model6QuantumSynapse(params)
    dt = model.dt
    
    # Track enhancement over time
    k_enhancement_values = []
    
    # === BASELINE ===
    for _ in range(BURST_PROTOCOL['baseline_steps']):
        model.step(dt, stimulus={'voltage': -70e-3})
    
    # === BURST PROTOCOL ===
    for burst_num in range(BURST_PROTOCOL['n_bursts']):
        # Active phase
        for _ in range(BURST_PROTOCOL['burst_duration_ms']):
            model.step(dt, stimulus={'voltage': -20e-3, 'activity_level': 0.8})
            # Record k_enhancement if available
            k_enh = getattr(model, '_k_enhancement', 1.0)
            k_enhancement_values.append(k_enh)
        
        # Rest phase
        if burst_num < BURST_PROTOCOL['n_bursts'] - 1:
            for _ in range(BURST_PROTOCOL['inter_burst_interval_ms']):
                model.step(dt, stimulus={'voltage': -70e-3})
    
    # === RECOVERY ===
    for _ in range(BURST_PROTOCOL['recovery_steps']):
        model.step(dt, stimulus={'voltage': -70e-3})
    
    # Get metrics
    metrics = model.get_experimental_metrics()
    
    return {
        'em_enabled': em_enabled,
        'seed': seed,
        'dimer_peak_nM': metrics.get('dimer_peak_nM_ct', 0),
        'collective_field_kT': metrics.get('collective_field_kT', 0),
        'k_enhancement_mean': np.mean(k_enhancement_values) if k_enhancement_values else 1.0,
        'k_enhancement_max': np.max(k_enhancement_values) if k_enhancement_values else 1.0,
        'network_modulation': getattr(model, '_network_modulation', 0),
    }

# =============================================================================
# TEST 1: PAIRED COMPARISON (Same Seed)
# =============================================================================

print("\n" + "="*70)
print("TEST 1: PAIRED COMPARISON (Same Seed)")
print("="*70)
print("\nRunning EM OFF and EM ON with IDENTICAL seed...")

SEED = 42

print(f"\n--- EM OFF (seed={SEED}) ---")
result_off = run_simulation(em_enabled=False, seed=SEED)
print(f"  Dimer peak: {result_off['dimer_peak_nM']:.1f} nM")
print(f"  k_enhancement: {result_off['k_enhancement_mean']:.2f}× (should be 1.0)")

print(f"\n--- EM ON (seed={SEED}) ---")
result_on = run_simulation(em_enabled=True, seed=SEED)
print(f"  Dimer peak: {result_on['dimer_peak_nM']:.1f} nM")
print(f"  k_enhancement: {result_on['k_enhancement_mean']:.2f}× (should be >1.0)")
print(f"  k_enhancement max: {result_on['k_enhancement_max']:.2f}×")
print(f"  Collective field: {result_on['collective_field_kT']:.1f} kT")

# Calculate true enhancement
if result_off['dimer_peak_nM'] > 0:
    true_enhancement = result_on['dimer_peak_nM'] / result_off['dimer_peak_nM']
    print(f"\n>>> TRUE ENHANCEMENT (same seed): {true_enhancement:.2f}×")
else:
    true_enhancement = 0
    print("\n>>> Cannot calculate - EM OFF produced 0 dimers")

# =============================================================================
# TEST 2: UNPAIRED COMPARISON (Different Seeds - Like experiment_02)
# =============================================================================

print("\n" + "="*70)
print("TEST 2: UNPAIRED COMPARISON (Different Seeds)")
print("="*70)
print("\nThis mimics what experiment_02 does...")

print(f"\n--- EM OFF (seed=42) ---")
result_off_42 = run_simulation(em_enabled=False, seed=42)
print(f"  Dimer peak: {result_off_42['dimer_peak_nM']:.1f} nM")

print(f"\n--- EM ON (seed=100) ---")
result_on_100 = run_simulation(em_enabled=True, seed=100)
print(f"  Dimer peak: {result_on_100['dimer_peak_nM']:.1f} nM")

if result_off_42['dimer_peak_nM'] > 0:
    apparent_enhancement = result_on_100['dimer_peak_nM'] / result_off_42['dimer_peak_nM']
    print(f"\n>>> APPARENT ENHANCEMENT (diff seeds): {apparent_enhancement:.2f}×")
else:
    apparent_enhancement = 0

# =============================================================================
# TEST 3: MULTIPLE PAIRS (Statistical Power)
# =============================================================================

print("\n" + "="*70)
print("TEST 3: MULTIPLE PAIRED COMPARISONS")
print("="*70)

n_pairs = 5
paired_enhancements = []

print(f"\nRunning {n_pairs} paired comparisons...")

for i in range(n_pairs):
    seed = 42 + i
    
    # EM OFF
    np.random.seed(seed)
    r_off = run_simulation(em_enabled=False, seed=seed)
    
    # EM ON (SAME seed)
    np.random.seed(seed)
    r_on = run_simulation(em_enabled=True, seed=seed)
    
    if r_off['dimer_peak_nM'] > 0:
        enh = r_on['dimer_peak_nM'] / r_off['dimer_peak_nM']
        paired_enhancements.append(enh)
        print(f"  Pair {i+1} (seed={seed}): OFF={r_off['dimer_peak_nM']:.0f}nM, "
              f"ON={r_on['dimer_peak_nM']:.0f}nM, Enhancement={enh:.2f}×")

if paired_enhancements:
    mean_enh = np.mean(paired_enhancements)
    std_enh = np.std(paired_enhancements)
    print(f"\n>>> MEAN PAIRED ENHANCEMENT: {mean_enh:.2f}× ± {std_enh:.2f}")

# =============================================================================
# DIAGNOSIS
# =============================================================================

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("\nQuestion: Is k_enhancement actually being calculated?")
if result_on['k_enhancement_max'] > 1.1:
    print(f"  ✓ YES - k_enhancement reaches {result_on['k_enhancement_max']:.2f}×")
else:
    print(f"  ✗ NO - k_enhancement is only {result_on['k_enhancement_max']:.2f}×")
    print("    → EM coupling module may not be working correctly")

print("\nQuestion: Does same-seed pairing reveal the enhancement?")
if true_enhancement > 1.1:
    print(f"  ✓ YES - True enhancement is {true_enhancement:.2f}×")
    if apparent_enhancement < 1.0:
        print(f"    → But different seeds show {apparent_enhancement:.2f}× (suppression!)")
        print("    → CONCLUSION: Seed mismatch masks the enhancement")
else:
    print(f"  ✗ NO - Even with same seed, enhancement is only {true_enhancement:.2f}×")
    print("    → Enhancement may not be propagating to dimer formation")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if true_enhancement > 1.1 and apparent_enhancement < true_enhancement:
    print("""
The EM enhancement IS working, but different seeds mask it.

FIX for experiment_02_cascade_validation.py:

In run_experiment_D(), change:

    for em_enabled in [False, True]:
        key = 'em_on' if em_enabled else 'em_off'
        for rep in range(n_reps):
            seed = 42 + rep if not em_enabled else 100 + rep  # ← PROBLEM
            np.random.seed(seed)
            ...

To:

    for rep in range(n_reps):
        seed = 42 + rep  # Same seed for each pair
        
        # EM OFF
        np.random.seed(seed)
        params_off = configure_parameters(em_enabled=False, ...)
        result_off = run_condition(params_off, f"em_off_rep{rep}")
        results['em_off'].append(result_off)
        
        # EM ON (SAME SEED!)
        np.random.seed(seed)
        params_on = configure_parameters(em_enabled=True, ...)
        result_on = run_condition(params_on, f"em_on_rep{rep}")
        results['em_on'].append(result_on)
""")
else:
    print("""
The issue may be deeper than seed mismatch.

Check:
1. Is em_coupling_enabled actually being set?
2. Is k_agg_enhanced being used in ca_phosphate.step()?
3. Is there substrate depletion counteracting the enhancement?
""")

print("="*70)