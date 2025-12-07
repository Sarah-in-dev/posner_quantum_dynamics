"""
Test Feedback Loop Validation
=============================

Validates that the cascade feedback loop works correctly:
1. Tryptophan EM field enhances dimer formation
2. More dimers → stronger network modulation
3. Above threshold → collective field emerges
4. Loop is stable (doesn't run away)

Compares EM-enabled vs EM-disabled runs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
np.random.seed(42)  # Reproducibility

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*70)
print("FEEDBACK LOOP VALIDATION")
print("="*70)

# ============================================================================
# TEST 1: EM DISABLED (Baseline)
# ============================================================================

print("\n" + "="*70)
print("TEST 1: EM COUPLING DISABLED (Baseline)")
print("="*70)

np.random.seed(42)  # Reset seed for fair comparison

params_off = Model6Parameters()
params_off.em_coupling_enabled = False  # ← KEY: EM OFF
params_off.multi_synapse_enabled = True
params_off.multi_synapse.n_synapses_default = 10

model_off = Model6QuantumSynapse(params_off)

print("\nRunning simulation (1000 steps)...")
for i in range(1000):
    model_off.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})

metrics_off = model_off.get_experimental_metrics()
dimer_off = metrics_off.get('dimer_peak_nM_ct', 0)

print(f"\nResults (EM OFF):")
print(f"  Dimer peak: {dimer_off:.1f} nM")
print(f"  Dimers per synapse: {dimer_off * 0.006 / 10:.1f}")

# ============================================================================
# TEST 2: EM ENABLED (With Feedback)
# ============================================================================

print("\n" + "="*70)
print("TEST 2: EM COUPLING ENABLED (With Feedback)")
print("="*70)

# np.random.seed(42)  # Reset seed for fair comparison

params_on = Model6Parameters()
params_on.em_coupling_enabled = True  # ← KEY: EM ON
params_on.multi_synapse_enabled = True
params_on.multi_synapse.n_synapses_default = 10

model_on = Model6QuantumSynapse(params_on)

print("\nRunning simulation (1000 steps)...")
for i in range(1000):
    model_on.step(0.001, {'voltage': -20e-3, 'activity_level': 0.8})

metrics_on = model_on.get_experimental_metrics()
dimer_on = metrics_on.get('dimer_peak_nM_ct', 0)

print(f"\nResults (EM ON):")
print(f"  Dimer peak: {dimer_on:.1f} nM")
print(f"  Dimers per synapse: {dimer_on * 0.006 / 10:.1f}")
print(f"  Network modulation: {model_on._network_modulation:.2f}")
print(f"  Collective field: {model_on._collective_field_kT:.1f} kT")

# ============================================================================
# TEST 3: COMPARE AND VALIDATE
# ============================================================================

print("\n" + "="*70)
print("TEST 3: FEEDBACK LOOP ANALYSIS")
print("="*70)

if dimer_off > 0:
    enhancement = dimer_on / dimer_off
    print(f"\nDimer Formation Enhancement:")
    print(f"  EM OFF: {dimer_off:.1f} nM")
    print(f"  EM ON:  {dimer_on:.1f} nM")
    print(f"  Enhancement: {enhancement:.2f}×")
    
    # Expected enhancement from em_coupling_module: ~2-3×
    if 1.5 <= enhancement <= 4.0:
        print(f"\n  ✓ Enhancement in expected range (1.5-4.0×)")
    elif enhancement > 1.0:
        print(f"\n  ~ Enhancement present but outside expected range")
    else:
        print(f"\n  ✗ No enhancement detected - feedback may not be working")
else:
    print("\n  ✗ Baseline dimer count is zero - cannot calculate enhancement")

# Check stability
print(f"\nFeedback Stability:")
print(f"  Collective field: {model_on._collective_field_kT:.1f} kT")
if 15 <= model_on._collective_field_kT <= 30:
    print(f"  ✓ Field in Goldilocks zone (15-30 kT)")
else:
    print(f"  ✗ Field outside expected range")

# Check that EM module is actually being used
if hasattr(model_on, '_k_enhancement_history') and len(model_on._k_enhancement_history) > 0:
    k_enh_mean = np.mean(model_on._k_enhancement_history)
    k_enh_max = np.max(model_on._k_enhancement_history)
    print(f"\nRate Constant Enhancement (k_enhanced/k_baseline):")
    print(f"  Mean: {k_enh_mean:.2f}×")
    print(f"  Max:  {k_enh_max:.2f}×")
else:
    print(f"\n  (k_enhancement history not available)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FEEDBACK LOOP VALIDATION SUMMARY")
print("="*70)

checks_passed = 0
total_checks = 3

# Check 1: Enhancement exists
if dimer_off > 0 and dimer_on > dimer_off:
    print("✓ Forward path working: EM field enhances dimer formation")
    checks_passed += 1
else:
    print("✗ Forward path issue: No enhancement detected")

# Check 2: Threshold reached
if model_on._network_modulation >= 5.0:
    print("✓ Network threshold reached: Superradiance active")
    checks_passed += 1
else:
    print("✗ Network threshold not reached")

# Check 3: Field in range
if 15 <= model_on._collective_field_kT <= 30:
    print("✓ Collective field stable: In Goldilocks zone")
    checks_passed += 1
else:
    print("✗ Collective field out of range")

print(f"\nResult: {checks_passed}/{total_checks} checks passed")

if checks_passed == total_checks:
    print("\n✓✓✓ FEEDBACK LOOP VALIDATED ✓✓✓")
else:
    print("\n⚠ Some checks failed - investigate feedback mechanism")

print("="*70)