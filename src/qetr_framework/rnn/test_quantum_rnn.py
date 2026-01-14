"""
Test Suite for Quantum RNN Framework
=====================================

Run this to validate the system works correctly.

Usage:
    python test_quantum_rnn.py

Expected output: All tests should PASS.

Author: Sarah Davidson
"""

import numpy as np
import sys

# Import the modules
from neuron import RateNeuron, NeuronParameters
from quantum_rnn_synapse import QuantumRNNSynapse, SynapseParameters, ClassicalSynapse
from quantum_rnn import QuantumRNN, ClassicalRNN


def test_neuron_basic():
    """Test that neurons respond to input correctly"""
    print("Test: Neuron basic response...", end=" ")
    
    neuron = RateNeuron()
    dt = 0.01  # 10ms
    
    # Drive neuron with strong input
    for _ in range(100):
        neuron.step(dt, input_current=2.0)
    
    # Should be firing
    assert neuron.firing_rate > 0.9, f"Expected rate > 0.9, got {neuron.firing_rate}"
    
    # Remove input, should decay
    for _ in range(100):
        neuron.step(dt, input_current=0.0)
    
    assert neuron.firing_rate < 0.1, f"Expected rate < 0.1, got {neuron.firing_rate}"
    
    print("PASS")


def test_dimer_formation():
    """Test that dimers form on coincident activity"""
    print("Test: Dimer formation...", end=" ")
    
    syn = QuantumRNNSynapse(seed=42)
    dt = 0.1
    
    assert syn.n_dimers == 0, "Should start with no dimers"
    
    # Coincident activity should form dimers
    for _ in range(5):
        syn.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    
    assert syn.n_dimers > 0, f"Should have dimers, got {syn.n_dimers}"
    assert syn.mean_singlet_probability > 0.9, f"Fresh dimers should have P_S near 1.0"
    
    print(f"PASS (formed {syn.n_dimers} dimers)")


def test_quantum_eligibility_persistence():
    """Test that quantum eligibility persists for ~100s"""
    print("Test: Quantum eligibility persistence...", end=" ")
    
    syn = QuantumRNNSynapse(seed=42)
    dt = 0.1
    
    # Form dimers
    for _ in range(10):
        syn.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    
    initial_P_S = syn.mean_singlet_probability
    assert initial_P_S > 0.9, f"Initial P_S should be > 0.9, got {initial_P_S}"
    
    # Wait 60 seconds
    for _ in range(600):
        syn.step(dt, pre_active=False, post_active=False, dopamine=0.0)
    
    final_P_S = syn.mean_singlet_probability
    
    # Should still be above entanglement threshold (0.5)
    assert final_P_S > 0.5, f"P_S after 60s should be > 0.5, got {final_P_S}"
    assert syn.is_eligible, "Should still be eligible after 60s"
    
    print(f"PASS (P_S: {initial_P_S:.3f} → {final_P_S:.3f} after 60s)")


def test_classical_eligibility_decay():
    """Test that classical eligibility decays with τ"""
    print("Test: Classical eligibility decay...", end=" ")
    
    tau = 5.0
    syn = ClassicalSynapse(tau=tau)
    dt = 0.1
    
    # Form eligibility
    for _ in range(5):
        syn.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    
    initial_elig = syn.eligibility
    assert initial_elig > 0.9, f"Initial eligibility should be > 0.9, got {initial_elig}"
    
    # Wait 30 seconds (6 time constants)
    for _ in range(300):
        syn.step(dt, pre_active=False, post_active=False, dopamine=0.0)
    
    final_elig = syn.eligibility
    expected = initial_elig * np.exp(-30.0 / tau)
    
    assert abs(final_elig - expected) < 0.01, f"Expected {expected:.6f}, got {final_elig:.6f}"
    assert final_elig < 0.01, f"Should be near zero after 6τ"
    
    print(f"PASS (eligibility: {initial_elig:.3f} → {final_elig:.6f})")


def test_three_factor_gate():
    """Test that plasticity requires all three factors"""
    print("Test: Three-factor gate...", end=" ")
    
    dt = 0.1
    
    # Test 1: All factors present → gate opens
    syn1 = QuantumRNNSynapse(seed=42)
    for _ in range(5):
        syn1.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    syn1.step(dt, pre_active=False, post_active=True, dopamine=1.0)
    
    assert syn1.gate_open, "Gate should open with all factors"
    
    # Test 2: Missing dopamine → gate closed
    syn2 = QuantumRNNSynapse(seed=42)
    for _ in range(5):
        syn2.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    syn2.step(dt, pre_active=False, post_active=True, dopamine=0.0)
    
    assert not syn2.gate_open, "Gate should be closed without dopamine"
    
    # Test 3: Missing calcium (post_active) → gate closed
    syn3 = QuantumRNNSynapse(seed=42)
    for _ in range(5):
        syn3.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    syn3.step(dt, pre_active=False, post_active=False, dopamine=1.0)
    
    assert not syn3.gate_open, "Gate should be closed without calcium"
    
    print("PASS")


def test_plasticity_accumulation():
    """Test that weight changes accumulate while gate is open"""
    print("Test: Plasticity accumulation...", end=" ")
    
    syn = QuantumRNNSynapse(seed=42)
    dt = 0.1
    
    # Form dimers
    for _ in range(5):
        syn.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    
    initial_weight = syn.weight
    
    # Open gate for 1 second
    for _ in range(10):
        syn.step(dt, pre_active=False, post_active=True, dopamine=1.0)
    
    final_weight = syn.weight
    
    assert final_weight > initial_weight, f"Weight should increase: {initial_weight} → {final_weight}"
    assert syn.cumulative_plasticity > 0, "Should have cumulative plasticity"
    
    print(f"PASS (weight: {initial_weight:.3f} → {final_weight:.3f})")


def test_credit_assignment_quantum_vs_classical():
    """THE KEY TEST: Credit assignment at 60s delay"""
    print("Test: Credit assignment at 60s delay...", end=" ")
    
    dt = 0.1
    delay_steps = 600  # 60 seconds
    
    results = {}
    
    # Quantum synapse
    syn_q = QuantumRNNSynapse(seed=42)
    for _ in range(5):
        syn_q.step(dt, pre_active=True, post_active=True, dopamine=0.0)
    for _ in range(delay_steps):
        syn_q.step(dt, pre_active=False, post_active=False, dopamine=0.0)
    
    weight_before = syn_q.weight
    for _ in range(5):
        syn_q.step(dt, pre_active=False, post_active=True, dopamine=1.0)
    
    results['Quantum'] = syn_q.weight - weight_before
    
    # Classical synapses
    for tau in [1.0, 5.0, 20.0]:
        syn_c = ClassicalSynapse(tau=tau)
        for _ in range(5):
            syn_c.step(dt, pre_active=True, post_active=True, dopamine=0.0)
        for _ in range(delay_steps):
            syn_c.step(dt, pre_active=False, post_active=False, dopamine=0.0)
        
        weight_before = syn_c.weight
        for _ in range(5):
            syn_c.step(dt, pre_active=False, post_active=True, dopamine=1.0)
        
        results[f'Classical τ={tau}s'] = syn_c.weight - weight_before
    
    # Quantum should learn, classical should not
    assert results['Quantum'] > 0.01, f"Quantum should learn, got Δw={results['Quantum']}"
    assert results['Classical τ=1.0s'] < 0.001, f"Classical τ=1s should not learn"
    assert results['Classical τ=5.0s'] < 0.001, f"Classical τ=5s should not learn"
    
    print("PASS")
    print(f"       Quantum: Δw = {results['Quantum']:.4f}")
    print(f"       Classical τ=1s: Δw = {results['Classical τ=1.0s']:.6f}")
    print(f"       Classical τ=5s: Δw = {results['Classical τ=5.0s']:.6f}")
    print(f"       Classical τ=20s: Δw = {results['Classical τ=20.0s']:.6f}")


def test_network_creation():
    """Test that network creates correctly"""
    print("Test: Network creation...", end=" ")
    
    rnn = QuantumRNN(n_neurons=10, seed=42)
    
    assert rnn.n_neurons == 10
    assert rnn.n_synapses == 90  # 10*9 for all-to-all excluding self
    assert rnn.get_weight_matrix().shape == (10, 10)
    
    print(f"PASS ({rnn.n_neurons} neurons, {rnn.n_synapses} synapses)")


def test_network_dynamics():
    """Test that network responds to input"""
    print("Test: Network dynamics...", end=" ")
    
    rnn = QuantumRNN(n_neurons=10, seed=42)
    dt = 0.01  # Smaller timestep for numerical stability
    
    # Create stimulus
    stimulus = np.zeros(10)
    stimulus[:3] = 2.0
    
    # Run network for 1 second
    for _ in range(100):
        rates = rnn.step(dt, stimulus, dopamine=0.0)
    
    # Stimulated neurons should be active
    assert rates[0] > 0.5, f"Stimulated neuron should be active, got {rates[0]}"
    assert rates[1] > 0.5, f"Stimulated neuron should be active, got {rates[1]}"
    
    # Should have formed dimers
    metrics = rnn.get_network_metrics()
    assert metrics['total_dimers'] > 0, "Should have formed dimers"
    assert metrics['n_eligible_synapses'] > 0, "Should have eligible synapses"
    
    print(f"PASS (dimers: {metrics['total_dimers']}, eligible: {metrics['n_eligible_synapses']})")


def test_network_credit_assignment():
    """Test network-level credit assignment"""
    print("Test: Network credit assignment at 30s delay...", end=" ")
    
    dt = 0.1
    n_neurons = 10
    
    # Quantum RNN
    rnn = QuantumRNN(n_neurons=n_neurons, seed=42)
    
    stimulus = np.zeros(n_neurons)
    stimulus[:3] = 2.0
    
    # Phase 1: Stimulate
    for _ in range(10):
        rnn.step(dt, stimulus, dopamine=0.0)
    
    # Track pattern synapses
    pattern_synapses = [(pre, post) for (pre, post), syn in rnn.synapses.items() if syn.n_dimers > 0]
    n_eligible_start = sum(1 for (pre, post) in pattern_synapses if rnn.synapses[(pre, post)].is_eligible)
    
    # Phase 2: Wait 30s (silence)
    for _ in range(300):
        for syn in rnn.synapses.values():
            syn.step(dt, pre_active=False, post_active=False, dopamine=0.0)
    
    n_eligible_end = sum(1 for (pre, post) in pattern_synapses if rnn.synapses[(pre, post)].is_eligible)
    
    # Phase 3: Reward
    weights_before = {k: rnn.synapses[k].weight for k in pattern_synapses}
    for _ in range(10):
        for (pre, post) in pattern_synapses:
            rnn.synapses[(pre, post)].step(dt, pre_active=False, post_active=True, dopamine=1.0)
    
    total_dw = sum(rnn.synapses[k].weight - weights_before[k] for k in pattern_synapses)
    mean_dw = total_dw / len(pattern_synapses) if pattern_synapses else 0
    
    # Eligibility should persist
    assert n_eligible_end == n_eligible_start, f"Eligibility should persist: {n_eligible_start} → {n_eligible_end}"
    
    # Should have learned
    assert mean_dw > 0.01, f"Should have weight change, got {mean_dw}"
    
    print(f"PASS (eligible: {n_eligible_start}→{n_eligible_end}, mean Δw: {mean_dw:.4f})")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("QUANTUM RNN TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_neuron_basic,
        test_dimer_formation,
        test_quantum_eligibility_persistence,
        test_classical_eligibility_decay,
        test_three_factor_gate,
        test_plasticity_accumulation,
        test_credit_assignment_quantum_vs_classical,
        test_network_creation,
        test_network_dynamics,
        test_network_credit_assignment,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! System is working correctly.\n")
    else:
        print(f"\n✗ {failed} test(s) failed. Check output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()