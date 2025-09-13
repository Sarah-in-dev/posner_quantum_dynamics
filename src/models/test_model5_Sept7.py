"""
Updated test script for Model 5 with biophysical dopamine system
Tests the integrated DopamineField functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from model5_neuromodulated_quantum_synapse import NeuromodulatedQuantumSynapse, Model5Parameters

def test_dopamine_biophysics():
    """Test the biophysical dopamine system"""
    print("\n" + "="*60)
    print("TEST 1: BIOPHYSICAL DOPAMINE SYSTEM")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Check initialization
    print(f"\nDopamine field initialized:")
    print(f"  Tonic level: {model.da_field.params.tonic_concentration * 1e9:.1f} nM")
    print(f"  Release sites: {len(model.da_field.release_sites)}")
    print(f"  D2 Kd: {model.da_field.params.Kd_D2 * 1e9:.1f} nM")
    
    # Test baseline
    print("\n--- Testing baseline (no reward) ---")
    baseline_stats = []
    for i in range(50):
        model.update_dopamine(0.001, reward_signal=False)
        stats = model.da_field.get_statistics()
        baseline_stats.append(stats['max'] * 1e9)
    
    print(f"Baseline: {np.mean(baseline_stats):.1f} ± {np.std(baseline_stats):.1f} nM")
    
    # Test reward response
    print("\n--- Testing reward response ---")
    reward_stats = []
    d2_occupancy = []
    
    for i in range(200):  # 200ms
        reward = (i < 100)  # First 100ms
        model.update_dopamine(0.001, reward_signal=reward)
        stats = model.da_field.get_statistics()
        reward_stats.append(stats['max'] * 1e9)
        d2_occupancy.append(stats['D2_occupancy_mean'])
        
        if i % 50 == 0:
            print(f"  t={i}ms: DA={stats['max']*1e9:.1f} nM, "
                  f"Vesicles released={stats['vesicles_released']}, "
                  f"D2 occupancy={stats['D2_occupancy_mean']:.2f}")
    
    peak_da = max(reward_stats)
    peak_d2 = max(d2_occupancy)
    
    print(f"\nPeak dopamine: {peak_da:.1f} nM")
    print(f"Peak D2 occupancy: {peak_d2:.2f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time_ms = np.arange(len(reward_stats))
    ax1.plot(time_ms, reward_stats)
    ax1.axvline(x=100, color='r', linestyle='--', label='Reward ends')
    ax1.set_ylabel('Dopamine (nM)')
    ax1.set_title('Dopamine Concentration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_ms, d2_occupancy)
    ax2.axvline(x=100, color='r', linestyle='--')
    ax2.axhline(y=0.5, color='k', linestyle=':', label='50% occupancy')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('D2 Receptor Occupancy')
    ax2.set_title('D2 Receptor Activation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return peak_da > 100  # Should reach >100 nM

def test_dopamine_debug():
    """Debug dopamine system"""
    print("\n" + "="*60)
    print("DOPAMINE SYSTEM DEBUG")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    model.debug_dopamine_system()
    
    # Try larger timesteps
    print("\n--- Testing with larger timesteps ---")
    model.da_field.field[:] = model.da_field.params.tonic_concentration  # Reset
    
    for dt in [0.001, 0.01, 0.1]:
        # Reset
        model.da_field.vesicles_released = 0
        model.da_field.field[:] = model.da_field.params.tonic_concentration
        
        # Try 10 steps
        for _ in range(10):
            stats = model.da_field.update(dt=dt, stimulus=False, reward=True)
        
        print(f"\ndt={dt*1000}ms: Released {stats['vesicles_released']} vesicles, "
              f"Max DA={stats['max']*1e9:.1f} nM")
    
    # Check the actual release mechanism
    print("\n--- Checking release mechanism ---")
    
    # Look at the volume calculation
    if hasattr(model.da_field, 'dx'):
        volume = model.da_field.dx ** 2 * 20e-9  # Assuming 20nm cleft
        print(f"Grid spacing dx: {model.da_field.dx*1e9:.1f} nm")
        print(f"Voxel volume: {volume*1e18:.3f} µm³")
        
        # Calculate expected concentration from one vesicle
        molecules = 3000
        moles = molecules / 6.022e23
        expected_conc = moles / (volume * 1000)  # M
        print(f"Expected concentration from 1 vesicle: {expected_conc*1e9:.1f} nM")
    
    return True


def test_j_coupling_with_gradients():
    """Test J-coupling with spatial visualization"""
    print("\n" + "="*60)
    print("TEST 2: J-COUPLING SPATIAL DYNAMICS")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Create strong activity
    channel_states = np.ones(model.params.n_channels)  # All channels open
    
    print("\n--- Strong activity burst ---")
    for i in range(5):
        model.update_fields(0.001, channel_states, reward_signal=False)
    
    # Visualize J-coupling field
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(model.phosphate_j_coupling, cmap='hot', vmin=0, vmax=20)
    ax.set_title('J-coupling Field After Activity (Hz)')
    plt.colorbar(im, ax=ax)
    
    # Mark channel locations
    for idx, (ci, cj) in enumerate(model.channel_indices):
        ax.plot(ci, cj, 'wo', markersize=8, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    max_j = np.max(model.phosphate_j_coupling)
    print(f"Max J-coupling: {max_j:.1f} Hz")
    print(f"Locations above 15 Hz: {np.sum(model.phosphate_j_coupling > 15)}")
    
    return max_j > 15

def test_j_coupling_spatial():
    """Test J-coupling with spatial visualization"""
    print("\n" + "="*60)
    print("TEST 2: J-COUPLING SPATIAL DYNAMICS")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Debug: Check channel structure
    print(f"\nChannel info:")
    print(f"  n_channels: {model.params.n_channels}")
    print(f"  channel_indices: {model.channel_indices[:3]}...")  # First 3
    if hasattr(model, 'channels'):
        print(f"  channels object: {type(model.channels)}")
        print(f"  channels.states shape: {model.channels.states.shape if hasattr(model.channels.states, 'shape') else len(model.channels.states)}")
    
    # Open channels properly
    if hasattr(model, 'channels'):
        model.channels.states[:] = 1  # Open all channels
    
    print("\n--- Strong activity burst ---")


def test_quantum_modulation():
    """Test dopamine's quantum modulation effects"""
    print("\n" + "="*60)
    print("TEST 3: QUANTUM MODULATION")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Test location
    i, j = 25, 25
    
    # Test different dopamine levels
    da_levels = [20e-9, 100e-9, 500e-9, 1000e-9]  # nM
    
    print("\nDopamine modulation of dimer/trimer formation:")
    print("DA (nM) | D2 Occ | Dimer Enh | Trimer Supp | Above Threshold")
    print("-" * 60)
    
    for da in da_levels:
        # Set dopamine level
        model.da_field.field[j, i] = da
        
        # Get modulation
        mod = model.da_field.get_quantum_modulation(i, j)
        
        print(f"{da*1e9:7.0f} | {mod['d2_occupancy']:6.2f} | "
              f"{mod['dimer_enhancement']:9.1f} | {mod['trimer_suppression']:11.2f} | "
              f"{'Yes' if mod['above_quantum_threshold'] else 'No':>15}")
    
    return True


def test_coincidence_detection():
    """Test STDP-like coincidence detection"""
    print("\n" + "="*60)
    print("TEST 4: COINCIDENCE DETECTION")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Simulate calcium spike at t=0
    model.calcium_spike_time = 0.0
    
    # Test coincidence factor at different times
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # seconds
    factors = []
    
    print("\nCoincidence factor vs dopamine-calcium timing:")
    print("Time after Ca (s) | Coincidence Factor")
    print("-" * 35)
    
    for t in times:
        model.current_time = t
        factor = model.da_field.get_coincidence_factor(
            calcium_spike_time=model.calcium_spike_time,
            current_time=model.current_time
        )
        factors.append(factor)
        print(f"{t:17.1f} | {factor:18.2f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(times, factors, 'bo-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimal')
    plt.axhline(y=0.5, color='gray', linestyle=':', label='50%')
    plt.xlabel('Time after calcium spike (s)')
    plt.ylabel('Coincidence Factor')
    plt.title('STDP-like Coincidence Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return True

def diagnose_mass_flow(self):
    """Track where all the phosphate is"""
    # Calculate totals
    total_P_free = np.sum(self.phosphate_field * self.active_mask)
    total_P_complex = np.sum(self.complex_field * self.active_mask)
    total_P_pnc = 4 * np.sum(self.pnc_field * self.active_mask)
    total_P_dimer = 4 * np.sum(self.dimer_field * self.active_mask)
    total_P_trimer = 6 * np.sum(self.trimer_field * self.active_mask)
    
    total_P = total_P_free + total_P_complex + total_P_pnc + total_P_dimer + total_P_trimer
    
    print(f"\nPhosphate distribution:")
    print(f"  Free PO4: {total_P_free*1e3:.2f} mM")
    print(f"  In complex: {total_P_complex*1e6:.2f} µM") 
    print(f"  In PNC: {total_P_pnc*1e6:.2f} µM")
    print(f"  In dimers: {total_P_dimer*1e6:.2f} µM")
    print(f"  In trimers: {total_P_trimer*1e6:.2f} µM")
    print(f"  TOTAL: {total_P*1e3:.2f} mM")
    
    # Check calcium too
    total_Ca_free = np.sum(self.calcium_field * self.active_mask)
    total_Ca_complex = np.sum(self.complex_field * self.active_mask)
    total_Ca_pnc = 6 * np.sum(self.pnc_field * self.active_mask)  # Assume Ca6 in PNC
    total_Ca_dimer = 6 * np.sum(self.dimer_field * self.active_mask)
    total_Ca_trimer = 9 * np.sum(self.trimer_field * self.active_mask)
    
    total_Ca = total_Ca_free + total_Ca_complex + total_Ca_pnc + total_Ca_dimer + total_Ca_trimer
    
    print(f"\nCalcium distribution:")
    print(f"  Free Ca: {total_Ca_free*1e6:.2f} µM")
    print(f"  In dimers: {total_Ca_dimer*1e6:.2f} µM")
    print(f"  TOTAL: {total_Ca*1e6:.2f} µM")


def test_integrated_dynamics():
    """Test full integrated simulation with all components"""
    print("\n" + "="*60)
    print("TEST 5: INTEGRATED DYNAMICS")
    print("="*60)
    
    model = NeuromodulatedQuantumSynapse()
    
    # Simulation protocol
    channel_states = np.zeros(model.params.n_channels)
    
    # Tracking
    metrics = {
        'dopamine': [],
        'j_coupling': [],
        'dimers': [],
        'trimers': [],
        'd2_occupancy': [],
        'learning': []
    }
    
    print("\n--- Running integrated simulation ---")
    
    # Phase 1: Baseline (10ms)
    for t in range(10):
        model.update_fields(0.001, channel_states, reward_signal=False)
        metrics['dopamine'].append(np.max(model.dopamine_field) * 1e9)
        metrics['j_coupling'].append(np.max(model.phosphate_j_coupling))
        metrics['dimers'].append(np.sum(model.dimer_field) * 1e9)
        metrics['trimers'].append(np.sum(model.trimer_field) * 1e9)
        metrics['d2_occupancy'].append(np.mean(model.da_field.D2_occupancy))
        metrics['learning'].append(model.learning_signal)
    
    # Phase 2: Activity + Reward (30ms)
    channel_states[0:3] = 1  # Open channels
    for t in range(30):
        reward = (t >= 5 and t < 15)  # Reward from 5-15ms
        model.update_fields(0.001, channel_states, reward_signal=reward)
        
        # Track metrics
        for key in metrics:
            if key == 'd2_occupancy':
                metrics[key].append(np.mean(model.da_field.D2_occupancy))
            elif key == 'learning':
                metrics[key].append(model.learning_signal)
            elif key == 'dopamine':
                metrics[key].append(np.max(model.dopamine_field) * 1e9)
            elif key == 'j_coupling':
                metrics[key].append(np.max(model.phosphate_j_coupling))
            elif key == 'dimers':
                metrics[key].append(np.sum(model.dimer_field) * 1e9)
            elif key == 'trimers':
                metrics[key].append(np.sum(model.trimer_field) * 1e9)
    
    # Phase 3: Recovery (20ms)
    channel_states[:] = 0
    for t in range(20):
        model.update_fields(0.001, channel_states, reward_signal=False)
        
        # Track metrics
        for key in metrics:
            if key == 'd2_occupancy':
                metrics[key].append(np.mean(model.da_field.D2_occupancy))
            elif key == 'learning':
                metrics[key].append(model.learning_signal)
            elif key == 'dopamine':
                metrics[key].append(np.max(model.dopamine_field) * 1e9)
            elif key == 'j_coupling':
                metrics[key].append(np.max(model.phosphate_j_coupling))
            elif key == 'dimers':
                metrics[key].append(np.sum(model.dimer_field) * 1e9)
            elif key == 'trimers':
                metrics[key].append(np.sum(model.trimer_field) * 1e9)
    
    # Plot comprehensive results
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    time_ms = np.arange(60)
    
    # Dopamine & D2
    ax = axes[0, 0]
    ax.plot(time_ms, metrics['dopamine'], 'b-', label='DA')
    ax2 = ax.twinx()
    ax2.plot(time_ms, metrics['d2_occupancy'], 'r-', label='D2 occ')
    ax.set_ylabel('Dopamine (nM)', color='b')
    ax2.set_ylabel('D2 Occupancy', color='r')
    ax.set_title('Dopamine & D2 Receptor')
    ax.axvspan(10, 20, alpha=0.2, color='yellow', label='Activity')
    ax.axvspan(15, 25, alpha=0.2, color='red', label='Reward')
    
    # J-coupling
    axes[0, 1].plot(time_ms, metrics['j_coupling'])
    axes[0, 1].set_ylabel('J-coupling (Hz)')
    axes[0, 1].set_title('Phosphate J-coupling')
    axes[0, 1].axvspan(10, 20, alpha=0.2, color='yellow')
    
    # Dimers vs Trimers
    axes[1, 0].plot(time_ms, metrics['dimers'], label='Dimers')
    axes[1, 0].plot(time_ms, metrics['trimers'], label='Trimers')
    axes[1, 0].set_ylabel('Concentration (nM)')
    axes[1, 0].set_title('Dimer vs Trimer Formation')
    axes[1, 0].legend()
    axes[1, 0].axvspan(10, 20, alpha=0.2, color='yellow')
    axes[1, 0].axvspan(15, 25, alpha=0.2, color='red')
    
    # Learning signal
    axes[1, 1].plot(time_ms, metrics['learning'])
    axes[1, 1].set_ylabel('Learning Signal')
    axes[1, 1].set_title('Integrated Learning Signal')
    axes[1, 1].axvspan(10, 20, alpha=0.2, color='yellow')
    axes[1, 1].axvspan(15, 25, alpha=0.2, color='red')
    
    # Dopamine gradient
    gradient = model.analyze_dopamine_gradients()
    im = axes[2, 0].imshow(gradient * 1e9, cmap='hot')
    axes[2, 0].set_title('Final DA Gradient (nM/µm)')
    plt.colorbar(im, ax=axes[2, 0])
    
    # Final dopamine field
    im2 = axes[2, 1].imshow(model.dopamine_field * 1e9, cmap='viridis')
    axes[2, 1].set_title('Final DA Field (nM)')
    plt.colorbar(im2, ax=axes[2, 1])
    
    for ax in axes[:2, :].flatten():
        ax.set_xlabel('Time (ms)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\nPeak values achieved:")
    print(f"  Dopamine: {max(metrics['dopamine']):.1f} nM")
    print(f"  D2 occupancy: {max(metrics['d2_occupancy']):.2f}")
    print(f"  J-coupling: {max(metrics['j_coupling']):.1f} Hz")
    print(f"  Total dimers: {max(metrics['dimers']):.1f} nM")
    print(f"  Total trimers: {max(metrics['trimers']):.1f} nM")
    print(f"  Peak learning: {max(metrics['learning']):.3f}")
    
    return True

    # After the simulation completes
    print("\nMass balance check:")
    model.diagnose_mass_flow()



def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("MODEL 5 BIOPHYSICAL TEST SUITE")
    print("="*60)
    
    tests = [
        # ("Dopamine Debug", test_dopamine_debug),
        ("Dopamine Biophysics", test_dopamine_biophysics),
        ("J-coupling Spatial", test_j_coupling_with_gradients),
        ("Quantum Modulation", test_quantum_modulation),
        ("Coincidence Detection", test_coincidence_detection),
        ("Integrated Dynamics", test_integrated_dynamics)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()