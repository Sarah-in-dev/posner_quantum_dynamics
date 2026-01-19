"""
Test Cross-Neuron Entanglement Integration
==========================================

Verifies that the CrossNeuronEntanglementTracker is properly
integrated with Model 6's MultiRegionNetwork.

Run from the Model_6 directory:
    python test_cross_neuron_integration.py

Author: Sarah Davidson
Date: January 2026
"""

import numpy as np
import sys

def test_integration():
    """Test the full integration."""
    
    print("=" * 70)
    print("CROSS-NEURON ENTANGLEMENT INTEGRATION TEST")
    print("=" * 70)
    
    # === TEST 1: Import check ===
    print("\n--- Test 1: Import Check ---")
    try:
        from cross_neuron_entanglement import (
            CrossNeuronEntanglementTracker,
            CrossNeuronBond,
            PhotonDimerLink
        )
        print("  ✓ cross_neuron_entanglement imports OK")
    except ImportError as e:
        print(f"  ✗ Failed to import cross_neuron_entanglement: {e}")
        return False
    
    try:
        from multi_region_network import MultiRegionNetwork, Region
        print("  ✓ multi_region_network imports OK")
    except ImportError as e:
        print(f"  ✗ Failed to import multi_region_network: {e}")
        return False
    
    try:
        from photon_emission_module import PhotonPacket, PhotonEmissionTracker
        print("  ✓ photon_emission_module imports OK")
    except ImportError as e:
        print(f"  ✗ Failed to import photon_emission_module: {e}")
        return False
    
    try:
        from photon_receiver_module import PhotonReceiver
        print("  ✓ photon_receiver_module imports OK")
    except ImportError as e:
        print(f"  ✗ Failed to import photon_receiver_module: {e}")
        return False
    
    # === TEST 2: PhotonPacket has new fields ===
    print("\n--- Test 2: PhotonPacket New Fields ---")
    try:
        packet = PhotonPacket(
            id=0,
            emission_time=0.0,
            source_synapse_id=0,
            n_photons=10.0,
            wavelength_m=350e-9,
            source_neuron_id=1,  # NEW field
            dimer_link=None      # NEW field
        )
        print(f"  ✓ PhotonPacket created with source_neuron_id={packet.source_neuron_id}")
        print(f"  ✓ PhotonPacket has dimer_link field: {packet.dimer_link}")
    except Exception as e:
        print(f"  ✗ PhotonPacket creation failed: {e}")
        return False
    
    # === TEST 3: PhotonEmissionTracker has neuron_id ===
    print("\n--- Test 3: PhotonEmissionTracker neuron_id ---")
    try:
        emitter = PhotonEmissionTracker(
            synapse_id=0,
            neuron_id=5  # NEW parameter
        )
        assert emitter.neuron_id == 5, "neuron_id not set correctly"
        print(f"  ✓ PhotonEmissionTracker has neuron_id={emitter.neuron_id}")
    except Exception as e:
        print(f"  ✗ PhotonEmissionTracker failed: {e}")
        return False
    
    # === TEST 4: PhotonReceiver has neuron_id and process_entanglement ===
    print("\n--- Test 4: PhotonReceiver neuron_id and process_entanglement ---")
    try:
        receiver = PhotonReceiver(
            synapse_id=0,
            neuron_id=3  # NEW parameter
        )
        assert receiver.neuron_id == 3, "neuron_id not set correctly"
        assert hasattr(receiver, 'recent_dimer_links'), "missing recent_dimer_links"
        assert hasattr(receiver, 'process_entanglement'), "missing process_entanglement method"
        print(f"  ✓ PhotonReceiver has neuron_id={receiver.neuron_id}")
        print(f"  ✓ PhotonReceiver has recent_dimer_links list")
        print(f"  ✓ PhotonReceiver has process_entanglement method")
    except Exception as e:
        print(f"  ✗ PhotonReceiver failed: {e}")
        return False
    
    # === TEST 5: MultiRegionNetwork has entanglement tracker ===
    print("\n--- Test 5: MultiRegionNetwork Entanglement Tracker ---")
    try:
        network = MultiRegionNetwork()
        assert hasattr(network, 'cross_neuron_entanglement'), "missing cross_neuron_entanglement"
        assert isinstance(network.cross_neuron_entanglement, CrossNeuronEntanglementTracker)
        assert 'cross_neuron_bonds' in network.history, "missing cross_neuron_bonds in history"
        assert 'coordination_factor' in network.history, "missing coordination_factor in history"
        print(f"  ✓ MultiRegionNetwork has cross_neuron_entanglement tracker")
        print(f"  ✓ History includes cross_neuron_bonds and coordination_factor")
    except Exception as e:
        print(f"  ✗ MultiRegionNetwork failed: {e}")
        return False
    
    # === TEST 6: MultiRegionNetwork has new methods ===
    print("\n--- Test 6: MultiRegionNetwork New Methods ---")
    try:
        assert hasattr(network, '_collect_dimer_states'), "missing _collect_dimer_states"
        assert hasattr(network, 'get_network_entanglement_state'), "missing get_network_entanglement_state"
        assert hasattr(network, 'apply_reward'), "missing apply_reward"
        print(f"  ✓ _collect_dimer_states method exists")
        print(f"  ✓ get_network_entanglement_state method exists")
        print(f"  ✓ apply_reward method exists")
    except Exception as e:
        print(f"  ✗ Method check failed: {e}")
        return False
    
    # === TEST 7: Create network and run simulation ===
    print("\n--- Test 7: Run Simulation ---")
    try:
        # Create fresh network
        network = MultiRegionNetwork()
        
        # Add two regions
        network.add_region(region_id=0, n_synapses=2, name="Source")
        network.add_region(region_id=1, n_synapses=2, name="Target")
        print(f"  ✓ Added 2 regions with 2 synapses each")
        
        # Connect them
        network.connect_regions(
            source_id=0,
            target_id=1,
            distance_mm=1.0,
            n_parallel_axons=50
        )
        print(f"  ✓ Connected regions with waveguide")
        
        # Run a few steps with stimulation
        dt = 0.001  # 1ms
        active_stim = {'voltage': -10e-3, 'reward': False}
        
        print(f"  Running 100 steps with stimulation...")
        for i in range(100):
            network.step(dt, stimuli={0: active_stim, 1: active_stim})
        
        print(f"  ✓ Simulation ran for {network.time*1000:.0f} ms")
        
        # Check history was recorded
        assert len(network.history['time']) == 100, "History not recorded correctly"
        assert len(network.history['cross_neuron_bonds']) == 100, "Entanglement history not recorded"
        print(f"  ✓ History recorded correctly")
        
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === TEST 8: Check entanglement state ===
    print("\n--- Test 8: Entanglement State ---")
    try:
        state = network.get_network_entanglement_state()
        print(f"  Cross-neuron bonds: {state['n_bonds']}")
        print(f"  Coordination factor: {state['coordination_factor']:.4f}")
        print(f"  Metrics: {state['metrics']}")
        print(f"  ✓ get_network_entanglement_state works")
    except Exception as e:
        print(f"  ✗ Entanglement state failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === TEST 9: Apply reward ===
    print("\n--- Test 9: Apply Reward ---")
    try:
        result = network.apply_reward(reward=1.0)
        print(f"  Coordination factor: {result['coordination_factor']:.4f}")
        print(f"  Cross-neuron bonds: {result['n_cross_neuron_bonds']}")
        print(f"  Number of updates: {len(result['updates'])}")
        
        # Show a sample update
        if result['updates']:
            sample_key = list(result['updates'].keys())[0]
            sample = result['updates'][sample_key]
            print(f"  Sample update {sample_key}:")
            print(f"    eligibility: {sample['eligibility']:.4f}")
            print(f"    coordination_bonus: {sample['coordination_bonus']:.4f}")
            print(f"    delta_w: {sample['delta_w']:.6f}")
        
        print(f"  ✓ apply_reward works")
    except Exception as e:
        print(f"  ✗ Apply reward failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe cross-neuron entanglement system is integrated correctly.")
    print("\nNext steps:")
    print("  1. Run longer simulations to see entanglement bonds form")
    print("  2. Compare entangled vs non-entangled networks on coordination tasks")
    print("  3. Test with different waveguide distances and configurations")
    
    return True


def test_entanglement_formation():
    """
    Longer test to actually see entanglement bonds form.
    Requires dimers to form at both source and target.
    """
    print("\n" + "=" * 70)
    print("ENTANGLEMENT FORMATION TEST (longer simulation)")
    print("=" * 70)
    
    from multi_region_network import MultiRegionNetwork
    
    # Create network
    network = MultiRegionNetwork()
    network.add_region(region_id=0, n_synapses=3, name="Source")
    network.add_region(region_id=1, n_synapses=3, name="Target")
    network.connect_regions(0, 1, distance_mm=1.0)
    
    # Strong stimulation to form dimers
    dt = 0.001
    active_stim = {'voltage': -10e-3, 'reward': False}
    rest_stim = {'voltage': -70e-3, 'reward': False}
    
    print("\nRunning 2 second simulation with theta-burst stimulation...")
    print(f"{'Time (ms)':<12} {'Src Dimers':<12} {'Tgt Dimers':<12} {'Bonds':<10} {'Coord':<10}")
    print("-" * 56)
    
    for step in range(2000):
        t_ms = step * dt * 1000
        
        # Theta burst pattern: 4 spikes at 100Hz, repeated at 5Hz
        in_burst = (step % 200) < 40  # 40ms burst every 200ms
        in_spike = in_burst and (step % 10) < 2  # 2ms spike every 10ms within burst
        
        if in_spike:
            stim = {0: active_stim, 1: active_stim}
        else:
            stim = {0: rest_stim, 1: rest_stim}
        
        network.step(dt, stimuli=stim)
        
        # Report every 200ms
        if step % 200 == 0:
            src_dimers = network.regions[0].get_mean_dimer_count()
            tgt_dimers = network.regions[1].get_mean_dimer_count()
            n_bonds = len(network.cross_neuron_entanglement.bonds)
            coord = network.cross_neuron_entanglement.get_coordination_factor([0, 1])
            
            print(f"{t_ms:>10.0f}  {src_dimers:>10.2f}  {tgt_dimers:>10.2f}  "
                  f"{n_bonds:>8}  {coord:>8.4f}")
    
    # Final state
    print("\n--- Final State ---")
    state = network.get_network_entanglement_state()
    print(f"Total cross-neuron bonds: {state['n_bonds']}")
    print(f"Coordination factor: {state['coordination_factor']:.4f}")
    
    if state['n_bonds'] > 0:
        print("\n✓ Cross-neuron entanglement bonds formed!")
    else:
        print("\n⚠ No bonds formed - this may be expected if dimers didn't form")
        print("  Check that Model 6 dimer formation is working correctly")
    
    return state['n_bonds'] > 0


if __name__ == "__main__":
    # Run basic integration test
    success = test_integration()
    
    if success:
        # Run longer formation test
        print("\n" + "=" * 70)
        response = input("Run longer entanglement formation test? (y/n): ")
        if response.lower() == 'y':
            test_entanglement_formation()
    else:
        print("\n✗ Integration test failed - fix errors before continuing")
        sys.exit(1)