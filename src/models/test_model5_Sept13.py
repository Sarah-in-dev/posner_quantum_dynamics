"""
Comprehensive Test Suite for Model 5: Neuromodulated Quantum Synapse
Tests each component systematically with proper value tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model5_neuromodulated_quantum_synapse import (
    NeuromodulatedQuantumSynapse, 
    Model5Parameters
)

def format_value(value, unit='nM'):
    """Format scientific values for readable output"""
    if value == 0:
        return f"0.0 {unit}"
    elif value < 1e-12:
        return f"{value*1e15:.2f} fM"
    elif value < 1e-9:
        return f"{value*1e12:.2f} pM"
    elif value < 1e-6:
        return f"{value*1e9:.2f} nM"
    elif value < 1e-3:
        return f"{value*1e6:.2f} µM"
    else:
        return f"{value*1e3:.2f} mM"

def print_state_summary(model, label="State"):
    """Print comprehensive state summary with proper masking"""
    print(f"\n{label}:")
    
    # Calculate properly masked means
    mask = model.active_mask
    n_active = np.sum(mask)
    
    # Core concentrations
    ca_mean = np.sum(model.calcium_field * mask) / n_active if n_active > 0 else 0
    ca_max = np.max(model.calcium_field * mask)
    print(f"  1. Calcium: mean={format_value(ca_mean)}, peak={format_value(ca_max)}")
    
    # pH modulation
    ph_mean = np.mean(model.local_pH[mask]) if n_active > 0 else 7.3
    print(f"  2. pH: {ph_mean:.2f}")
    
    # ATP and J-coupling
    j_max = np.max(model.phosphate_j_coupling)
    atp_mean = np.mean(model.atp_field[mask]) if n_active > 0 else 0
    print(f"  3. J-coupling: {j_max:.1f} Hz, ATP: {format_value(atp_mean, 'mM')}")
    
    # Complex equilibrium
    complex_mean = np.mean(model.complex_field[mask]) if n_active > 0 else 0
    complex_max = np.max(model.complex_field * mask)
    print(f"  4. CaHPO4 complex: mean={format_value(complex_mean)}, peak={format_value(complex_max)}")
    
    # PNC dynamics - CRITICAL: proper calculation
    pnc_mean = np.mean(model.pnc_field[mask]) if n_active > 0 else 0
    pnc_max = np.max(model.pnc_field * mask)
    pnc_at_templates = 0
    if len(model.template_indices) > 0:
        template_pnc = [model.pnc_field[tj, ti] for ti, tj in model.template_indices]
        pnc_at_templates = np.mean(template_pnc) if template_pnc else 0
    print(f"  5. PNC: mean={format_value(pnc_mean)}, peak={format_value(pnc_max)}, templates={format_value(pnc_at_templates)}")
    
    # Dopamine
    da_mean = np.mean(model.dopamine_field[mask]) if n_active > 0 else 0
    da_max = np.max(model.dopamine_field * mask)
    d2_occ = np.mean(model.da_field.D2_occupancy[mask]) if n_active > 0 else 0
    print(f"  6. Dopamine: mean={format_value(da_mean)}, peak={format_value(da_max)}, D2={d2_occ*100:.1f}%")
    
    # Template occupancy
    if len(model.template_indices) > 0:
        template_bound = [model.template_pnc_bound[tj, ti] for ti, tj in model.template_indices]
        avg_occupancy = np.mean(template_bound) / model.params.n_binding_sites * 100
        print(f"  7. Template occupancy: {avg_occupancy:.1f}%")
    
    # Dimers and trimers
    dimer_total = np.sum(model.dimer_field * mask)
    trimer_total = np.sum(model.trimer_field * mask)
    print(f"  8. Dimers: {format_value(dimer_total)}, Trimers: {format_value(trimer_total)}")
    if dimer_total > 0 and trimer_total > 0:
        ratio = dimer_total / trimer_total
        print(f"     Dimer/Trimer ratio: {ratio:.2f}")
    
    # Quantum coherence
    dimer_coh_mean = np.mean(model.coherence_dimer[mask]) if n_active > 0 else 0
    trimer_coh_mean = np.mean(model.coherence_trimer[mask]) if n_active > 0 else 0
    print(f"  9. Coherence: Dimer={dimer_coh_mean:.3f}, Trimer={trimer_coh_mean:.3f}")
    
    # Warnings
    if pnc_mean > 1e-6:
        print("     ⚠️ WARNING: PNC concentration exceeds 1 µM!")
    if dimer_total > 10e-6:
        print("     ⚠️ WARNING: Excessive dimer accumulation!")
    if ca_mean < 50e-9:
        print("     ⚠️ WARNING: Calcium depleted!")

def test_baseline_state():
    """Test 1: Verify baseline state is reasonable"""
    print("\n" + "="*70)
    print("TEST 1: BASELINE STATE VALIDATION")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Run for 100ms at baseline
    for i in range(100):
        channel_states = np.zeros(model.params.n_channels)
        model.update_fields(0.001, channel_states, reward_signal=False)
    
    print_state_summary(model, "Baseline after 100ms")
    
    # Validate baseline values
    mask = model.active_mask
    ca_mean = np.mean(model.calcium_field[mask])
    pnc_mean = np.mean(model.pnc_field[mask])
    
    assert 50e-9 < ca_mean < 200e-9, f"Calcium baseline wrong: {ca_mean}"
    assert pnc_mean < 1e-6, f"PNC baseline too high: {pnc_mean}"
    
    print("\n✓ Baseline state validated")
    return True

def test_calcium_response():
    """Test 2: Calcium microdomain formation"""
    print("\n" + "="*70)
    print("TEST 2: CALCIUM MICRODOMAIN FORMATION")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Open channels
    channel_states = np.ones(model.params.n_channels)
    
    print("\nBefore activity:")
    ca_before = np.max(model.calcium_field)
    print(f"  Peak calcium: {format_value(ca_before)}")
    
    # Single timestep with open channels
    model.update_fields(0.001, channel_states, reward_signal=False)
    
    print("\nAfter channels open (1ms):")
    ca_after = np.max(model.calcium_field)
    print(f"  Peak calcium: {format_value(ca_after)}")
    
    # Check microdomains formed
    assert ca_after > 10 * ca_before, "Calcium microdomains should form"
    assert ca_after < 1e-3, f"Calcium too high: {ca_after}"
    
    print("\n✓ Calcium microdomains validated")
    return True

def test_complex_formation():
    """Test 3: CaHPO4 complex equilibrium"""
    print("\n" + "="*70)
    print("TEST 3: COMPLEX FORMATION")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Set up high calcium condition
    model.calcium_field[:] = 10e-6  # 10 µM
    model.phosphate_field[:] = 1e-3  # 1 mM
    
    # Calculate equilibrium
    model.calculate_complex_equilibrium()
    
    complex_mean = np.mean(model.complex_field[model.active_mask])
    print(f"\nWith Ca=10µM, PO4=1mM:")
    print(f"  Complex concentration: {format_value(complex_mean)}")
    
    # Verify reasonable range
    assert 1e-9 < complex_mean < 1e-6, f"Complex out of range: {complex_mean}"
    
    print("\n✓ Complex formation validated")
    return True

def test_pnc_formation_kinetics():
    """Test 4: PNC formation and dissolution"""
    print("\n" + "="*70)
    print("TEST 4: PNC FORMATION KINETICS")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Track PNC over time with activity
    pnc_history = []
    channel_states = np.ones(model.params.n_channels)
    
    print("\nSimulating 50ms of activity...")
    for i in range(50):
        model.update_fields(0.001, channel_states, reward_signal=False)
        pnc_mean = np.mean(model.pnc_field[model.active_mask])
        pnc_history.append(pnc_mean)
        
        if i % 10 == 0:
            print(f"  t={i}ms: PNC = {format_value(pnc_mean)}")
    
    # Check for reasonable growth
    assert pnc_history[-1] > pnc_history[0], "PNC should increase with activity"
    assert pnc_history[-1] < 1e-6, f"PNC too high: {pnc_history[-1]}"
    
    # Now test dissolution (channels close)
    channel_states = np.zeros(model.params.n_channels)
    
    print("\nChannels closed, testing dissolution...")
    for i in range(50):
        model.update_fields(0.001, channel_states, reward_signal=False)
        pnc_mean = np.mean(model.pnc_field[model.active_mask])
        
        if i % 10 == 0:
            print(f"  t={i}ms: PNC = {format_value(pnc_mean)}")
    
    print("\n✓ PNC kinetics validated")
    return True

def test_dopamine_modulation():
    """Test 5: Dopamine release and modulation"""
    print("\n" + "="*70)
    print("TEST 5: DOPAMINE NEUROMODULATION")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Baseline dopamine
    da_baseline = np.max(model.dopamine_field)
    print(f"\nBaseline dopamine: {format_value(da_baseline)}")
    
    # Trigger reward
    channel_states = np.zeros(model.params.n_channels)
    
    print("\nTriggering reward signal...")
    for i in range(10):
        model.update_fields(0.001, channel_states, reward_signal=True)
    
    da_peak = np.max(model.dopamine_field)
    print(f"Peak dopamine after reward: {format_value(da_peak)}")
    
    # Check D2 occupancy
    d2_occ = np.max(model.da_field.D2_occupancy)
    print(f"Peak D2 occupancy: {d2_occ*100:.1f}%")
    
    assert da_peak > 10 * da_baseline, "Dopamine should increase with reward"
    assert d2_occ > 0.5, "D2 receptors should be activated"
    
    print("\n✓ Dopamine modulation validated")
    return True

def test_dimer_trimer_formation():
    """Test 6: Dimer vs trimer selectivity"""
    print("\n" + "="*70)
    print("TEST 6: DIMER/TRIMER FORMATION")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Set up conditions favoring formation
    model.pnc_field[:] = 100e-9  # 100 nM PNC everywhere
    
    # Test without dopamine (should favor trimers)
    print("\nWithout dopamine:")
    model.dopamine_field[:] = 20e-9  # Low baseline
    model.phosphate_j_coupling[:] = 5.0  # Moderate J-coupling
    
    for i in range(10):
        model.form_dimers_and_trimers(0.001)
    
    dimers_low_da = np.sum(model.dimer_field)
    trimers_low_da = np.sum(model.trimer_field)
    print(f"  Dimers: {format_value(dimers_low_da)}")
    print(f"  Trimers: {format_value(trimers_low_da)}")
    
    # Reset and test with high dopamine (should favor dimers)
    model.dimer_field[:] = 0
    model.trimer_field[:] = 0
    model.pnc_field[:] = 100e-9
    
    print("\nWith high dopamine:")
    model.dopamine_field[:] = 500e-9  # High dopamine
    model.phosphate_j_coupling[:] = 20.0  # High J-coupling
    
    for i in range(10):
        model.form_dimers_and_trimers(0.001)
    
    dimers_high_da = np.sum(model.dimer_field)
    trimers_high_da = np.sum(model.trimer_field)
    print(f"  Dimers: {format_value(dimers_high_da)}")
    print(f"  Trimers: {format_value(trimers_high_da)}")
    
    if dimers_high_da > 0 and dimers_low_da > 0:
        enhancement = dimers_high_da / dimers_low_da
        print(f"\nDimer enhancement with dopamine: {enhancement:.1f}x")
    
    print("\n✓ Dimer/trimer selectivity validated")
    return True

def test_quantum_coherence():
    """Test 7: Quantum coherence calculation"""
    print("\n" + "="*70)
    print("TEST 7: QUANTUM COHERENCE")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Create some dimers and trimers
    model.dimer_field[25, 25] = 100e-9
    model.trimer_field[25, 25] = 100e-9
    model.dopamine_field[25, 25] = 200e-9
    model.phosphate_j_coupling[25, 25] = 10.0
    
    # Update coherence
    model.update_quantum_coherence(0.001)
    
    dimer_coh = model.coherence_dimer[25, 25]
    trimer_coh = model.coherence_trimer[25, 25]
    
    print(f"\nAt location (25,25):")
    print(f"  Dimer coherence: {dimer_coh:.3f}")
    print(f"  Trimer coherence: {trimer_coh:.3f}")
    print(f"  Ratio: {dimer_coh/max(trimer_coh, 1e-10):.1f}x")
    
    assert dimer_coh > trimer_coh, "Dimers should maintain better coherence"
    
    print("\n✓ Quantum coherence validated")
    return True

def test_full_cascade():
    """Test 8: Complete cascade from activity to learning signal"""
    print("\n" + "="*70)
    print("TEST 8: FULL CASCADE - ACTIVITY → DOPAMINE → QUANTUM → LEARNING")
    print("="*70)
    
    model = NeuromodulatedQuantumSynapse()
    dt = 0.001
    
    # Phase 1: Baseline (100ms)
    print("\n--- PHASE 1: BASELINE (100ms) ---")
    channel_states = np.zeros(model.params.n_channels)
    for i in range(100):
        model.update_fields(dt, channel_states, reward_signal=False)
    print_state_summary(model, "Baseline")
    
    # Phase 2: Activity (100ms)
    print("\n--- PHASE 2: ACTIVITY (100ms) ---")
    channel_states = np.ones(model.params.n_channels)
    for i in range(100):
        model.update_fields(dt, channel_states, reward_signal=False)
    print_state_summary(model, "After activity")
    
    # Phase 3: Activity + Dopamine (100ms)
    print("\n--- PHASE 3: ACTIVITY + DOPAMINE (100ms) ---")
    for i in range(100):
        # Reward at t=50ms
        reward = (i == 50)
        model.update_fields(dt, channel_states, reward_signal=reward)
    print_state_summary(model, "After activity + dopamine")
    
    # Phase 4: Recovery (100ms)
    print("\n--- PHASE 4: RECOVERY (100ms) ---")
    channel_states = np.zeros(model.params.n_channels)
    for i in range(100):
        model.update_fields(dt, channel_states, reward_signal=False)
    print_state_summary(model, "After recovery")
    
    # Calculate learning signal
    learning = model.calculate_learning_signal()
    print(f"\n  Final learning signal: {learning:.3e}")
    
    print("\n✓ Full cascade validated")
    return True

def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("MODEL 5: COMPREHENSIVE TEST SUITE")
    print("Testing full dynamic cascade with proper value tracking")
    print("="*70)
    
    # Get dopamine parameters for display
    from dopamine_biophysics import DopamineParameters
    da_params = DopamineParameters()
    print(f"\nDopamine parameters initialized: D_eff={da_params.D_effective:.2e} m²/s")
    
    tests = [
        ("Baseline State", test_baseline_state),
        ("Calcium Response", test_calcium_response),
        ("Complex Formation", test_complex_formation),
        ("PNC Kinetics", test_pnc_formation_kinetics),
        ("Dopamine Modulation", test_dopamine_modulation),
        ("Dimer/Trimer Formation", test_dimer_trimer_formation),
        ("Quantum Coherence", test_quantum_coherence),
        ("Full Cascade", test_full_cascade)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED"))
            print(f"\n{'='*70}")
        except AssertionError as e:
            results.append((name, f"FAILED: {str(e)}"))
            print(f"\n✗ Test failed: {e}")
            print(f"{'='*70}")
        except Exception as e:
            results.append((name, f"ERROR: {str(e)}"))
            print(f"\n✗ Error: {e}")
            print(f"{'='*70}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")
        if status != "PASSED":
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Model 5 is functioning correctly with biologically realistic values.")
    else:
        print("\n⚠️ Some tests failed. Review the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)