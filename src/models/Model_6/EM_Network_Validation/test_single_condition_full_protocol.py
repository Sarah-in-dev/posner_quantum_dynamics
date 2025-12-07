"""
Single Condition Test - Full Protocol
======================================

Tests ONE condition from experiment_02 with the FULL burst protocol
to diagnose if quick mode was causing the failures.

Compares to test_em_field_fix.py which works correctly.

This should run in ~same time as test_em_field_fix (~1-2 minutes)
and tell us if the full experiment will work.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

print("="*80)
print("SINGLE CONDITION TEST - FULL PROTOCOL")
print("="*80)
print("\nTesting if quick mode was the problem...")
print("Running same condition as test_em_field_fix but using experiment_02 setup")

# =============================================================================
# CONFIGURE PARAMETERS (same as experiment_02)
# =============================================================================

def configure_parameters(n_synapses, isotope, uv_condition, anesthetic, temperature):
    """Same function as experiment_02"""
    params = Model6Parameters()
    
    # SET TIMESTEP
    params.simulation.dt_diffusion = 1e-3  # 1 ms
    
    # ENABLE EM COUPLING
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    
    # MULTI-SYNAPSE
    params.multi_synapse.n_synapses_default = n_synapses
    
    # ISOTOPE
    if isotope == 'P31':
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
    elif isotope == 'P32':
        params.environment.fraction_P31 = 0.0
        params.environment.fraction_P32 = 1.0
    else:
        raise ValueError(f"Unknown isotope: {isotope}")
    
    # UV
    if uv_condition == 'none':
        params.metabolic_uv.external_uv_illumination = False
    elif uv_condition == '280nm':
        params.metabolic_uv.external_uv_illumination = True
        params.metabolic_uv.external_uv_wavelength = 280e-9
        params.metabolic_uv.external_uv_intensity = 1e-3
    elif uv_condition == '220nm':
        params.metabolic_uv.external_uv_illumination = True
        params.metabolic_uv.external_uv_wavelength = 220e-9
        params.metabolic_uv.external_uv_intensity = 1e-3
    else:
        raise ValueError(f"Unknown UV condition: {uv_condition}")
    
    # ANESTHETIC
    if anesthetic:
        params.tryptophan.anesthetic_applied = True
        params.tryptophan.anesthetic_type = 'isoflurane'
        params.tryptophan.anesthetic_blocking_factor = 0.9
    
    # TEMPERATURE
    params.environment.T = temperature
    
    return params

# =============================================================================
# RUN TEST
# =============================================================================

print("\n### TEST CONDITION ###")
print("  N_synapses: 10 (threshold)")
print("  Isotope: P31")
print("  UV: none")
print("  Anesthetic: False")
print("  Temperature: 310 K (37°C)")
print("  Protocol: FULL (5 bursts × 30ms, not quick mode)")

# Configure
params = configure_parameters(
    n_synapses=10,
    isotope='P31',
    uv_condition='none',
    anesthetic=False,
    temperature=310
)

# Initialize
print("\nInitializing model...")
model = Model6QuantumSynapse(params=params)
print(f"✓ Model initialized (dt={model.dt*1e3:.1f} ms)")

# FULL PROTOCOL (same as test_em_field_fix)
BURST_PROTOCOL = {
    'n_bursts': 5,
    'burst_duration_ms': 30,
    'inter_burst_interval_ms': 150
}

print(f"\nRunning FULL burst protocol:")
print(f"  {BURST_PROTOCOL['n_bursts']} bursts × {BURST_PROTOCOL['burst_duration_ms']}ms")
print(f"  Inter-burst: {BURST_PROTOCOL['inter_burst_interval_ms']}ms")

# Baseline
print("\n  Baseline (20 steps)...", end="", flush=True)
for _ in range(20):
    model.step(model.dt, stimulus={'voltage': -70e-3})
print(" done")

# Bursts
for burst_num in range(BURST_PROTOCOL['n_bursts']):
    print(f"  Burst {burst_num+1}/{BURST_PROTOCOL['n_bursts']}...", end="", flush=True)
    
    # Active phase
    for _ in range(BURST_PROTOCOL['burst_duration_ms']):
        model.step(model.dt, stimulus={'voltage': -10e-3})
    
    # Rest phase
    if burst_num < BURST_PROTOCOL['n_bursts'] - 1:
        for _ in range(BURST_PROTOCOL['inter_burst_interval_ms']):
            model.step(model.dt, stimulus={'voltage': -70e-3})
    
    print(" done")

# Final recovery
print("  Recovery (300 steps)...", end="", flush=True)
for _ in range(300):
    model.step(model.dt, stimulus={'voltage': -70e-3})
print(" done")

# =============================================================================
# GET RESULTS
# =============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

metrics = model.get_experimental_metrics()

print("\n### Core Metrics ###")
print(f"  Dimer peak: {metrics.get('dimer_peak_nM_ct', 0):.1f} nM")
print(f"  Coherence: {metrics.get('coherence_dimer_mean', 0):.3f}")
print(f"  T2: {metrics.get('T2_dimer_s', 0):.1f} s")

print("\n### EM Coupling - Forward Path (Trp → Dimers) ###")
print(f"  Tryptophan EM field: {metrics.get('trp_em_field_gv_m', 0):.3f} GV/m")
print(f"  k_enhancement: {metrics.get('em_formation_enhancement', 1.0):.2f}×")

print("\n### EM Coupling - Reverse Path (Dimers → Proteins) ###")
print(f"  Collective field: {metrics.get('collective_field_kT', 0):.1f} kT")

# =============================================================================
# COMPARISON TO EXPECTED
# =============================================================================

print("\n" + "="*80)
print("COMPARISON TO EXPECTED VALUES")
print("="*80)
print("\nFrom test_em_field_fix.py (which works correctly):")
print("  Collective field: ~20-30 kT")
print("  Trp EM field: ~1.0-1.5 GV/m")
print("  Coherence: ~0.98")
print("  T2: ~70-80 s")

print("\n### VALIDATION ###")

collective_field = metrics.get('collective_field_kT', 0)
trp_field = metrics.get('trp_em_field_gv_m', 0)
coherence = metrics.get('coherence_dimer_mean', 0)
t2 = metrics.get('T2_dimer_s', 0)

checks_passed = 0
total_checks = 4

print("\nCheck 1: Collective field")
if collective_field > 15:
    print(f"  ✓ PASS: {collective_field:.1f} kT (expected: 20-30 kT)")
    checks_passed += 1
else:
    print(f"  ✗ FAIL: {collective_field:.1f} kT (expected: 20-30 kT)")

print("\nCheck 2: Tryptophan EM field")
if trp_field > 0.5:
    print(f"  ✓ PASS: {trp_field:.3f} GV/m (expected: 1.0-1.5 GV/m)")
    checks_passed += 1
else:
    print(f"  ✗ FAIL: {trp_field:.3f} GV/m (expected: 1.0-1.5 GV/m)")

print("\nCheck 3: Coherence")
if coherence > 0.9:
    print(f"  ✓ PASS: {coherence:.3f} (expected: ~0.98)")
    checks_passed += 1
else:
    print(f"  ⚠ PARTIAL: {coherence:.3f} (expected: ~0.98)")
    checks_passed += 0.5

print("\nCheck 4: T2 coherence time")
if t2 > 50:
    print(f"  ✓ PASS: {t2:.1f} s (expected: 70-80 s)")
    checks_passed += 1
else:
    print(f"  ⚠ PARTIAL: {t2:.1f} s (expected: 70-80 s)")
    checks_passed += 0.5

# =============================================================================
# DIAGNOSIS
# =============================================================================

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"\nChecks passed: {checks_passed}/{total_checks}")

if checks_passed >= 3.5:
    print("\n✓✓✓ EXCELLENT - Full protocol fixes the problem!")
    print("\nConclusion:")
    print("  • Quick mode (3 × 10ms) was insufficient")
    print("  • Full protocol (5 × 30ms) produces correct EM coupling")
    print("  • experiment_02 code is correct")
    print("  • Run full experiment WITHOUT --quick flag")
    print("\nNext step:")
    print("  python experiment_02_em_network_validation.py")
    print("  (Will take several hours but should work)")
    
elif checks_passed >= 2:
    print("\n⚠ PARTIAL SUCCESS - Some improvement but not perfect")
    print("\nPossible issues:")
    print("  • Protocol duration helps but may need tuning")
    print("  • Some parameters may differ from test_em_field_fix")
    print("  • Check mean vs max collective field issue")
    print("\nNext step:")
    print("  • Compare parameters between this and test_em_field_fix")
    print("  • May need to adjust burst protocol further")
    
else:
    print("\n✗✗✗ STILL BROKEN - Full protocol doesn't fix it")
    print("\nThe problem is NOT the quick mode protocol.")
    print("\nPossible issues:")
    print("  • Parameter configuration differs from test_em_field_fix")
    print("  • Model initialization issue")
    print("  • History tracking problem")
    print("  • Metrics collection timing")
    print("\nNext step:")
    print("  • Line-by-line comparison of this vs test_em_field_fix")
    print("  • Add debug prints in model6_core.py")
    print("  • Check if metrics are being reset between conditions")

print("\n" + "="*80)