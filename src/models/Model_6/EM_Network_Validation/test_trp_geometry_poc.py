"""
Test Script: Geometry-Based Spatial Averaging Validation
=========================================================

Tests the updated _calculate_spatial_averaging() method that uses real
PDB 1JFF tryptophan coordinates instead of fitted spatial_factor = 0.5

DIMER CALCULATION (from October 7, 2025 conversation):
------------------------------------------------------
Active zone volume: ~0.01 μm³ = 10⁻¹⁷ L

dimers = concentration_nM × 10⁻⁹ mol/L × 10⁻¹⁷ L × 6.022×10²³
       = concentration_nM × 0.006

Example: 741 nM × 0.006 = 4.5 dimers per synapse ✓

With N=10 synapses at threshold: ~50 total dimers needed for entanglement

Test Strategy:
1. UNIT TEST: Call _calculate_spatial_averaging() directly, check value
2. COMPONENT TEST: Run TryptophanDimerCoupling with EM field input
3. INTEGRATION TEST: Run full Model 6 burst protocol
4. COMPARISON: Document what changed from fitted → geometry-based

Based on working pattern from test_em_field_fix.py

Author: Sarah Davidson
Date: December 6, 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add Model 6 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("GEOMETRY-BASED SPATIAL AVERAGING VALIDATION")
print("="*80)

# =============================================================================
# TEST 1: UNIT TEST - Direct call to _calculate_spatial_averaging()
# =============================================================================

print("\n" + "="*80)
print("TEST 1: UNIT TEST - _calculate_spatial_averaging()")
print("="*80)

try:
    from trp_coordinates_1jff import get_ring_centers, get_dipole_directions, SUMMARY
    
    print("\n### Tryptophan Coordinate Data ###")
    print(f"  Source: {SUMMARY['source']}")
    print(f"  Resolution: {SUMMARY['resolution']}")
    print(f"  Tryptophans per dimer: {SUMMARY['n_total_per_dimer']}")
    
    # Get coordinates
    trp_positions = get_ring_centers()
    print(f"\n### Raw Coordinates (Ångströms) ###")
    print(f"  Shape: {trp_positions.shape}")
    print(f"  Mean position: {np.mean(trp_positions, axis=0)}")
    print(f"  Std dev: {np.std(trp_positions, axis=0)}")
    
    # Calculate spread (size of tryptophan network within tubulin)
    distances_from_center = np.linalg.norm(
        trp_positions - np.mean(trp_positions, axis=0), axis=1
    )
    print(f"\n### Network Geometry ###")
    print(f"  Min distance from center: {np.min(distances_from_center):.1f} Å")
    print(f"  Max distance from center: {np.max(distances_from_center):.1f} Å")
    print(f"  Mean distance from center: {np.mean(distances_from_center):.1f} Å")
    
    print("\n✓ Tryptophan coordinates loaded successfully")
    
except ImportError as e:
    print(f"\n✗ FAILED to import trp_coordinates_1jff: {e}")
    print("  Make sure trp_coordinates_1jff.py is in the Model 6 directory")
    sys.exit(1)

# Now test the actual _calculate_spatial_averaging method
print("\n### Testing _calculate_spatial_averaging() ###")

try:
    from model6_parameters import Model6Parameters
    from em_coupling_module import TryptophanDimerCoupling
    
    params = Model6Parameters()
    params.em_coupling_enabled = True
    
    # Create the forward coupling object
    forward_coupling = TryptophanDimerCoupling(params)
    
    # Call the method
    spatial_factor = forward_coupling._calculate_spatial_averaging()
    
    print(f"\n  Calculated spatial_factor: {spatial_factor:.4f}")
    print(f"  Previous fitted value: 0.5")
    print(f"  Ratio (new/old): {spatial_factor / 0.5:.2f}x")
    
    if spatial_factor > 0 and spatial_factor < 2.0:
        print(f"\n✓ spatial_factor is in reasonable range")
    else:
        print(f"\n⚠ spatial_factor may be out of expected range")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 2: COMPONENT TEST - Forward Coupling Enhancement
# =============================================================================

print("\n" + "="*80)
print("TEST 2: COMPONENT TEST - Forward Coupling Enhancement")
print("="*80)

try:
    # Test with typical EM field from tryptophan module
    test_em_fields = [0, 1e8, 5e8, 1e9, 1.4e9]  # V/m
    
    print("\n### Enhancement vs EM Field ###")
    print(f"{'EM Field (V/m)':<20} {'Enhancement':<15} {'Spatial Factor':<15}")
    print("-" * 50)
    
    for em_field in test_em_fields:
        result = forward_coupling.calculate_formation_enhancement(em_field)
        print(f"{em_field:<20.2e} {result['enhancement_factor']:<15.3f} {result['spatial_averaged']:<15.4f}")
    
    # Check that enhancement increases with field
    low_result = forward_coupling.calculate_formation_enhancement(1e8)
    high_result = forward_coupling.calculate_formation_enhancement(1.4e9)
    
    if high_result['enhancement_factor'] > low_result['enhancement_factor']:
        print(f"\n✓ Enhancement increases with field strength")
    else:
        print(f"\n✗ Enhancement should increase with field")
        
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 3: COMPONENT TEST - Reverse Coupling (Dimers → Proteins)
# =============================================================================

print("\n" + "="*80)
print("TEST 3: COMPONENT TEST - Reverse Coupling")
print("="*80)

try:
    from em_coupling_module import DimerProteinCoupling
    
    reverse_coupling = DimerProteinCoupling(params)
    
    # Test threshold behavior
    test_n_dimers = [1, 5, 10, 20, 30, 40, 50, 75, 100]
    
    print("\n### Collective Field vs N Dimers ###")
    print(f"{'N Dimers':<12} {'Raw Field (kT)':<18} {'Effective (kT)':<18} {'Regime':<12}")
    print("-" * 60)
    
    for n in test_n_dimers:
        result = reverse_coupling.calculate_collective_field(n)
        print(f"{n:<12} {result['energy_kT_raw']:<18.1f} {result['energy_kT_effective']:<18.1f} {result['regime']:<12}")
    
    # Check threshold at N=50
    result_49 = reverse_coupling.calculate_collective_field(49)
    result_50 = reverse_coupling.calculate_collective_field(50)
    
    if result_50['regime'] == 'strong' and result_49['regime'] != 'strong':
        print(f"\n✓ Threshold behavior at N=50 confirmed")
    else:
        print(f"\n⚠ Threshold behavior may have changed")
        
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# TEST 4: INTEGRATION TEST - Full Model 6 Burst Protocol
# =============================================================================

print("\n" + "="*80)
print("TEST 4: INTEGRATION TEST - Full Model 6")
print("="*80)

try:
    from model6_core import Model6QuantumSynapse
    
    # Configure parameters
    params = Model6Parameters()
    params.simulation.dt_diffusion = 1e-3  # 1 ms
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.multi_synapse.n_synapses_default = 10
    params.environment.fraction_P31 = 1.0
    params.environment.fraction_P32 = 0.0
    params.environment.T = 310.15
    
    print("\n### Configuration ###")
    print(f"  N_synapses: 10")
    print(f"  Isotope: P31")
    print(f"  Temperature: 310 K")
    print(f"  EM coupling: ENABLED")
    
    # Initialize model
    print("\nInitializing Model 6...")
    model = Model6QuantumSynapse(params=params)
    print(f"✓ Model initialized (dt={model.dt*1e3:.1f} ms)")
    
    # Burst protocol (same as test_em_field_fix.py)
    BURST_PROTOCOL = {
        'n_bursts': 5,
        'burst_duration_ms': 30,
        'inter_burst_interval_ms': 150
    }
    
    print(f"\n### Running Burst Protocol ###")
    print(f"  {BURST_PROTOCOL['n_bursts']} bursts × {BURST_PROTOCOL['burst_duration_ms']}ms")
    
    # Baseline
    print("  Baseline (20 steps)...", end="", flush=True)
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
    
    # Recovery
    print("  Recovery (300 steps)...", end="", flush=True)
    for _ in range(300):
        model.step(model.dt, stimulus={'voltage': -70e-3})
    print(" done")
    
    # Get results
    metrics = model.get_experimental_metrics()
    
    print("\n### Results ###")
    
    # Dimer formation
    print("\n## Dimer Formation ##")
    dimer_peak = metrics.get('dimer_peak_nM_ct', 0)
    dimer_mean = metrics.get('dimer_mean_nM_ct', 0)
    print(f"  Peak concentration: {dimer_peak:.1f} nM")
    print(f"  Mean concentration: {dimer_mean:.1f} nM")
    
    # Calculate actual dimer count
    # Formula: dimers = concentration_nM × 10⁻⁹ mol/L × 10⁻¹⁷ L × 6.022×10²³
    # Simplified: dimers = concentration_nM × 0.006
    dimers_from_peak = dimer_peak * 0.006
    print(f"  → Total dimers (from peak): {dimers_from_peak:.1f}")
    print(f"  → Per synapse (N=10): {dimers_from_peak/10:.1f}")
    
    # Quantum coherence
    print("\n## Quantum Coherence ##")
    T2 = metrics.get('T2_dimer_s', 0)
    coherence = metrics.get('coherence_dimer_mean', 0)
    print(f"  T2 time: {T2:.1f} s")
    print(f"  Coherence: {coherence:.3f}")
    
    # EM Coupling - Forward
    print("\n## EM Coupling - Forward (Trp → Dimers) ##")
    trp_field = metrics.get('trp_em_field_gv_m', 0)
    k_enhancement = metrics.get('em_formation_enhancement', 1.0)
    print(f"  Tryptophan field: {trp_field:.3f} GV/m")
    print(f"  k_enhancement: {k_enhancement:.2f}×")
    print(f"  spatial_factor used: {spatial_factor:.4f}")
    
    # EM Coupling - Reverse
    print("\n## EM Coupling - Reverse (Dimers → Proteins) ##")
    collective_field = metrics.get('collective_field_kT', 0)
    print(f"  Collective field: {collective_field:.1f} kT")
    
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# TEST 5: SUCCESS CRITERIA
# =============================================================================

print("\n" + "="*80)
print("TEST 5: SUCCESS CRITERIA")
print("="*80)

# Calculate dimers from concentration
# Formula: dimers = concentration_nM × 10⁻⁹ × 10⁻¹⁷ L × 6.022 × 10²³
# Simplified: dimers = concentration_nM × 0.006
n_synapses = 10
dimers_total = dimer_peak * 0.006
dimers_per_synapse = dimers_total / n_synapses
expected_dimers_per_synapse = (4, 6)  # Target: 4-5 dimers per synapse

print(f"\n### Dimer Calculation ###")
print(f"  Peak concentration: {dimer_peak:.1f} nM")
print(f"  Active zone volume: 0.01 μm³ = 10⁻¹⁷ L")
print(f"  Total dimers: {dimer_peak:.1f} × 0.006 = {dimers_total:.1f}")
print(f"  N_synapses: {n_synapses}")
print(f"  Dimers per synapse: {dimers_per_synapse:.1f}")
print(f"  Target: {expected_dimers_per_synapse[0]}-{expected_dimers_per_synapse[1]} per synapse")

# Define criteria
criteria = {
    'spatial_factor_valid': (0.01 < spatial_factor < 2.0, 
                             f"spatial_factor = {spatial_factor:.4f} in range (0.01, 2.0)"),
    'dimers_per_synapse': (expected_dimers_per_synapse[0] <= dimers_per_synapse <= expected_dimers_per_synapse[1] * 2,
                           f"dimers/synapse = {dimers_per_synapse:.1f} in range ({expected_dimers_per_synapse[0]}, {expected_dimers_per_synapse[1]*2})"),
    'total_dimers_threshold': (dimers_total >= 40,
                               f"total dimers = {dimers_total:.0f} >= 40 (approaching 50 threshold)"),
    'T2_coherence': (T2 > 50,
                     f"T2 = {T2:.1f} s > 50s"),
    'forward_coupling': (trp_field > 0.5,
                         f"trp_field = {trp_field:.3f} GV/m > 0.5"),
    'reverse_coupling': (collective_field > 5,
                         f"collective_field = {collective_field:.1f} kT > 5"),
}

print("\n### Validation Checks ###")
passed = 0
failed = 0

for name, (condition, description) in criteria.items():
    if condition:
        print(f"  ✓ {name}: {description}")
        passed += 1
    else:
        print(f"  ✗ {name}: {description}")
        failed += 1

print(f"\n### Summary ###")
print(f"  Passed: {passed}/{len(criteria)}")
print(f"  Failed: {failed}/{len(criteria)}")

if failed == 0:
    print("\n" + "="*80)
    print("✓✓✓ ALL TESTS PASSED - Geometry-based spatial averaging is working!")
    print("="*80)
else:
    print("\n" + "="*80)
    print(f"⚠ {failed} test(s) failed - review results above")
    print("="*80)

# =============================================================================
# SUMMARY: GEOMETRY vs FITTED COMPARISON
# =============================================================================

print("\n" + "="*80)
print("COMPARISON: Geometry-Based vs Fitted Parameters")
print("="*80)

print("""
Parameter               | Fitted (old) | Geometry (new) | Change
------------------------|--------------|----------------|--------""")
print(f"spatial_factor          | 0.5000       | {spatial_factor:.4f}         | {spatial_factor/0.5:.2f}x")
print(f"k_enhancement           | ~2.5x        | {k_enhancement:.2f}x          | {'similar' if 1.5 < k_enhancement < 4.0 else 'different'}")
print(f"collective_field (kT)   | ~20 kT       | {collective_field:.1f} kT         | {'similar' if 10 < collective_field < 50 else 'different'}")
print(f"dimers/synapse          | 4-5          | {dimers_per_synapse:.1f}           | {'✓ matches' if 3 < dimers_per_synapse < 10 else 'check'}")
print(f"total dimers (N=10)     | ~50          | {dimers_total:.0f}            | {'✓ at threshold' if dimers_total > 40 else 'below threshold'}")

print("""
Interpretation:
- spatial_factor from real geometry replaces arbitrary 0.5
- Dimer calculation: dimers = concentration_nM × 0.006 (for 0.01 μm³ active zone)
- 741 nM = 4.5 dimers per synapse (Fisher prediction)
- N=10 synapses at threshold should give ~50 total dimers
- If collective_field > 20 kT, quantum effects are functional
""")

print("="*80)
print("TEST COMPLETE")
print("="*80)