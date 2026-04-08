#!/usr/bin/env python3
"""
Diagnostic: Coordinated Decoherence in MultiSynapseNetwork
==========================================================

This script diagnoses the current state of the coordinated decoherence
system to identify where the implementation is breaking down.

Tests:
1. Dimer formation - Are dimers forming at expected rates? (~4-5 per synapse)
2. Entanglement bonds - Are bonds forming between co-active synapses?
3. Correlation matrix - Is the correlation matrix non-zero after stimulation?
4. Eligibility decay - How does eligibility change with delay?
5. Threshold crossing - At what delay is eligibility near threshold?
6. Correlated sampling - Does sampling produce correlated outcomes?
7. Gate evaluation - Does coordination affect commitment patterns?

Author: Sarah Davidson
University of Florida
Date: January 2026
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

# Add Model 6 path
MODEL6_PATH = Path(__file__).parent
# Try common locations - including when run from Full_System_Experiments
for possible_path in [
    MODEL6_PATH.parent,  # If in Full_System_Experiments, go up to Model_6
    MODEL6_PATH,
    MODEL6_PATH / "src" / "models" / "Model_6",
    MODEL6_PATH.parent / "src" / "models" / "Model_6",
    Path("/mnt/user-data/uploads/src/models/Model_6"),
    Path.home() / "posner_quantum_dynamics" / "src" / "models" / "Model_6",
]:
    model6_core_path = possible_path / "model6_core.py"
    if model6_core_path.exists():
        sys.path.insert(0, str(possible_path))
        sys.path.insert(0, str(possible_path.parent.parent.parent))  # For imports
        print(f"Found Model 6 at: {possible_path}")
        break
else:
    # Also try direct parent relationships
    current = MODEL6_PATH
    for _ in range(5):
        if (current / "model6_core.py").exists():
            sys.path.insert(0, str(current))
            print(f"Found Model 6 at: {current}")
            break
        current = current.parent
    else:
        print("WARNING: Could not find Model 6 path. Please set MODEL6_PATH manually.")
        print("Attempting import anyway...")

try:
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    from multi_synapse_network import MultiSynapseNetwork, NetworkState
    print("✓ Successfully imported Model 6 components")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease ensure the Model 6 code is accessible.")
    print("You may need to run this script from the Model 6 directory.")
    sys.exit(1)


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_subheader(title: str):
    print(f"\n--- {title} ---")

def print_result(label: str, value, expected=None, unit=""):
    if expected is not None:
        status = "✓" if abs(value - expected) < expected * 0.5 else "✗"
        print(f"  {status} {label}: {value:.4f} {unit} (expected: ~{expected})")
    else:
        print(f"  • {label}: {value:.4f} {unit}")


# =============================================================================
# DIAGNOSTIC 1: DIMER FORMATION
# =============================================================================

def diagnose_dimer_formation(n_synapses: int = 4, verbose: bool = True) -> Dict:
    """
    Test: Are dimers forming at expected rates during stimulation?
    Expected: ~4-5 dimers per synapse after brief stimulation
    
    FAST VERSION: Uses 10ms timesteps and shorter stimulation
    """
    if verbose:
        print_header("DIAGNOSTIC 1: DIMER FORMATION")
    
    # Create network
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0  # P31 isotope
    
    network = MultiSynapseNetwork(
        n_synapses=n_synapses,
        params=params,
        pattern='clustered',
        spacing_um=2.0
    )
    network.initialize(Model6QuantumSynapse, params)
    network.set_microtubule_invasion(True)
    
    if verbose:
        print(f"\nCreated network with {n_synapses} synapses")
    
    # FAST: Use 10ms timesteps and 500ms stimulation
    dt = 0.01  # 10ms timestep (10x faster)
    stim_duration = 0.5  # 500ms (4x shorter)
    n_steps = int(stim_duration / dt)
    
    # Track dimer formation
    dimer_counts_over_time = []
    
    if verbose:
        print(f"Stimulating for {stim_duration}s at dt={dt*1000}ms ({n_steps} steps)...")
    
    start_time = time.time()
    
    for step in range(n_steps):
        # Depolarize all synapses
        state = network.step(dt, {'voltage': -10e-3, 'reward': False})
        
        # Track every 100ms
        if step % 10 == 0:
            counts = []
            for syn in network.synapses:
                if hasattr(syn, 'dimer_particles'):
                    counts.append(len(syn.dimer_particles.dimers))
                else:
                    counts.append(0)
            dimer_counts_over_time.append(counts)
            
            if verbose and step % 20 == 0:
                print(f"    Step {step}/{n_steps}: {sum(counts)} total dimers")
    
    elapsed = time.time() - start_time
    
    # Get final dimer counts
    final_counts = []
    for i, syn in enumerate(network.synapses):
        if hasattr(syn, 'dimer_particles'):
            count = len(syn.dimer_particles.dimers)
        else:
            count = 0
        final_counts.append(count)
        if verbose:
            print(f"  Synapse {i}: {count} dimers")
    
    mean_dimers = np.mean(final_counts)
    total_dimers = sum(final_counts)
    
    if verbose:
        print(f"\n  Mean dimers per synapse: {mean_dimers:.1f} (expected: 2-10)")
        print(f"  Total dimers: {total_dimers}")
        print(f"  Runtime: {elapsed:.2f}s")
    
    result = {
        'n_synapses': n_synapses,
        'final_counts': final_counts,
        'mean_dimers': mean_dimers,
        'total_dimers': total_dimers,
        'counts_over_time': dimer_counts_over_time,
        'network': network,
        'success': mean_dimers > 0.5  # At least some dimers
    }
    
    if verbose:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status}: Dimer formation {'detected' if result['success'] else 'NOT detected'}")
    
    return result


# =============================================================================
# DIAGNOSTIC 2: ENTANGLEMENT BONDS
# =============================================================================

def diagnose_entanglement_bonds(network: MultiSynapseNetwork, verbose: bool = True) -> Dict:
    """
    Test: Are entanglement bonds forming between co-active synapses?
    """
    if verbose:
        print_header("DIAGNOSTIC 2: ENTANGLEMENT BONDS")
    
    tracker = network.entanglement_tracker
    
    # Collect current dimers
    tracker.collect_dimers(network.synapses, network.positions)
    
    # Count bonds
    total_bonds = len(tracker.entanglement_bonds)
    
    # Count within-synapse vs cross-synapse bonds
    within_bonds = 0
    cross_bonds = 0
    
    if tracker.all_dimers:
        dimer_to_synapse = {d['global_id']: d['synapse_idx'] for d in tracker.all_dimers}
        
        for bond in tracker.entanglement_bonds:
            id_i, id_j = bond
            syn_i = dimer_to_synapse.get(id_i, -1)
            syn_j = dimer_to_synapse.get(id_j, -1)
            
            if syn_i >= 0 and syn_j >= 0:
                if syn_i == syn_j:
                    within_bonds += 1
                else:
                    cross_bonds += 1
    
    if verbose:
        print(f"\n  Total dimers tracked: {len(tracker.all_dimers) if tracker.all_dimers else 0}")
        print(f"  Total entanglement bonds: {total_bonds}")
        print(f"    - Within-synapse bonds: {within_bonds}")
        print(f"    - Cross-synapse bonds: {cross_bonds}")
    
    # Check coordination factor
    coord_factor = tracker.get_coordination_factor(network.synapses)
    
    if verbose:
        print(f"  Coordination factor: {coord_factor:.4f}")
    
    result = {
        'total_bonds': total_bonds,
        'within_bonds': within_bonds,
        'cross_bonds': cross_bonds,
        'coordination_factor': coord_factor,
        'success': cross_bonds > 0
    }
    
    if verbose:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status}: Cross-synapse bonds {'detected' if result['success'] else 'NOT detected'}")
    
    return result


# =============================================================================
# DIAGNOSTIC 3: CORRELATION MATRIX
# =============================================================================

def diagnose_correlation_matrix(network: MultiSynapseNetwork, verbose: bool = True) -> Dict:
    """
    Test: Is the correlation matrix non-zero after stimulation?
    """
    if verbose:
        print_header("DIAGNOSTIC 3: CORRELATION MATRIX")
    
    tracker = network.entanglement_tracker
    
    # Get correlation matrix
    tracker.collect_dimers(network.synapses, network.positions)
    C = tracker.get_synapse_correlation_matrix(network.synapses)
    
    n = len(network.synapses)
    
    if verbose:
        print(f"\nCorrelation matrix ({n}x{n}):")
        print("      ", end="")
        for j in range(n):
            print(f"  S{j}  ", end="")
        print()
        
        for i in range(n):
            print(f"  S{i}: ", end="")
            for j in range(n):
                if i == j:
                    print("  --  ", end="")
                else:
                    print(f" {C[i,j]:.3f}", end="")
            print()
    
    # Calculate statistics
    off_diag = C[np.triu_indices(n, k=1)]
    mean_corr = np.mean(off_diag) if len(off_diag) > 0 else 0
    max_corr = np.max(off_diag) if len(off_diag) > 0 else 0
    nonzero_pairs = np.sum(off_diag > 0.01)
    
    if verbose:
        print(f"\n  Mean off-diagonal correlation: {mean_corr:.4f}")
        print(f"  Max correlation: {max_corr:.4f}")
        print(f"  Non-zero pairs (>0.01): {nonzero_pairs}/{len(off_diag)}")
    
    result = {
        'correlation_matrix': C,
        'mean_correlation': mean_corr,
        'max_correlation': max_corr,
        'nonzero_pairs': nonzero_pairs,
        'success': max_corr > 0.01
    }
    
    if verbose:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status}: Correlation matrix {'has' if result['success'] else 'has NO'} non-zero entries")
    
    return result


# =============================================================================
# DIAGNOSTIC 4: ELIGIBILITY DECAY
# =============================================================================

def diagnose_eligibility_decay(network: MultiSynapseNetwork, 
                                delays: List[float] = [1, 5, 10, 20, 40, 60],
                                verbose: bool = True) -> Dict:
    """
    Test: How does eligibility decay with time?
    FAST VERSION: Uses 1s timesteps for decay
    """
    if verbose:
        print_header("DIAGNOSTIC 4: ELIGIBILITY DECAY")
    
    # Get initial eligibility
    initial_elig = [syn.get_eligibility() for syn in network.synapses]
    mean_initial = np.mean(initial_elig)
    
    if verbose:
        print(f"\nInitial eligibility (post-stimulation):")
        for i, e in enumerate(initial_elig):
            print(f"  Synapse {i}: {e:.4f}")
        print(f"  Mean: {mean_initial:.4f}")
    
    # FAST: Use 1s timesteps for decay
    dt = 1.0  # 1 second timestep
    
    eligibility_at_delay = {}
    
    if verbose:
        print(f"\nDecay over time (threshold = 0.33):")
        print(f"  {'Delay (s)':<12} {'Mean Elig':<12} {'vs Threshold':<15}")
        print(f"  {'-'*40}")
    
    current_time = 0
    for target_delay in delays:
        # Step until we reach this delay
        steps_needed = int((target_delay - current_time) / dt)
        
        for _ in range(steps_needed):
            network.step(dt, {'voltage': -70e-3, 'reward': False})
        
        current_time = target_delay
        
        # Measure eligibility
        elig = [syn.get_eligibility() for syn in network.synapses]
        mean_elig = np.mean(elig)
        eligibility_at_delay[target_delay] = {
            'per_synapse': elig,
            'mean': mean_elig
        }
        
        if verbose:
            vs_thresh = "ABOVE" if mean_elig > 0.33 else "BELOW"
            marker = "→" if 0.25 < mean_elig < 0.45 else " "
            print(f"  {target_delay:<12.1f} {mean_elig:<12.4f} {vs_thresh:<15} {marker}")
    
    # Find delay where eligibility crosses threshold
    threshold = 0.33
    crossing_delay = None
    for delay in sorted(eligibility_at_delay.keys()):
        if eligibility_at_delay[delay]['mean'] < threshold:
            crossing_delay = delay
            break
    
    if verbose:
        print(f"\n  Threshold crossing delay: {crossing_delay}s" if crossing_delay else 
              "\n  Eligibility never dropped below threshold in tested range")
    
    result = {
        'initial_eligibility': mean_initial,
        'eligibility_at_delay': eligibility_at_delay,
        'threshold': threshold,
        'crossing_delay': crossing_delay,
        'success': crossing_delay is not None and crossing_delay <= 60
    }
    
    return result


# =============================================================================
# DIAGNOSTIC 5: CORRELATED SAMPLING
# =============================================================================

def diagnose_correlated_sampling(n_synapses: int = 4, 
                                  n_samples: int = 100,
                                  verbose: bool = True) -> Dict:
    """
    Test: Does correlated sampling produce correlated outcomes?
    
    FAST VERSION: Uses 10ms timesteps and shorter stimulation
    """
    if verbose:
        print_header("DIAGNOSTIC 5: CORRELATED SAMPLING")
    
    # Create and stimulate network
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.environment.fraction_P31 = 1.0
    
    network = MultiSynapseNetwork(
        n_synapses=n_synapses,
        params=params,
        pattern='clustered',
        spacing_um=2.0
    )
    network.initialize(Model6QuantumSynapse, params)
    network.set_microtubule_invasion(True)
    
    # FAST: 10ms timesteps, 500ms stimulation
    dt = 0.01
    for _ in range(50):  # 500ms
        network.step(dt, {'voltage': -10e-3, 'reward': False})
    
    if verbose:
        print(f"\nNetwork stimulated. Collecting {n_samples} samples...")
    
    # Get theoretical correlation matrix
    network.entanglement_tracker.collect_dimers(network.synapses, network.positions)
    C_theoretical = network.entanglement_tracker.get_synapse_correlation_matrix(network.synapses)
    
    # Sample many times
    samples = []
    for _ in range(n_samples):
        s = network.sample_correlated_eligibilities()
        samples.append(s)
    
    samples = np.array(samples)  # Shape: (n_samples, n_synapses)
    
    # Compute empirical correlation from samples
    if samples.shape[0] > 1 and np.std(samples) > 0:
        C_empirical = np.corrcoef(samples.T)
    else:
        C_empirical = np.zeros((n_synapses, n_synapses))
    
    if verbose:
        print(f"\nTheoretical correlation matrix (from entanglement):")
        for i in range(n_synapses):
            print(f"  {C_theoretical[i, :]}")
        
        print(f"\nEmpirical correlation matrix (from {n_samples} samples):")
        for i in range(n_synapses):
            print(f"  {C_empirical[i, :]}")
    
    # Compare off-diagonal elements
    off_diag_idx = np.triu_indices(n_synapses, k=1)
    theo_off = C_theoretical[off_diag_idx]
    emp_off = C_empirical[off_diag_idx]
    
    # Check if empirical correlations are higher when theoretical are higher
    if len(theo_off) > 1 and np.std(theo_off) > 0.01:
        correlation_of_correlations = np.corrcoef(theo_off, emp_off)[0, 1]
    else:
        correlation_of_correlations = np.nan
    
    if verbose:
        print(f"\n  Mean theoretical off-diag: {np.mean(theo_off):.4f}")
        print(f"  Mean empirical off-diag: {np.mean(emp_off):.4f}")
        print(f"  Correlation of correlations: {correlation_of_correlations:.4f}")
    
    # Test threshold crossing correlation
    threshold = 0.33
    passes = samples > threshold  # Shape: (n_samples, n_synapses)
    
    # Count how often pairs pass/fail together
    concordance = np.zeros((n_synapses, n_synapses))
    for i in range(n_synapses):
        for j in range(i+1, n_synapses):
            same = np.sum(passes[:, i] == passes[:, j]) / n_samples
            concordance[i, j] = same
            concordance[j, i] = same
    
    if verbose:
        print(f"\nConcordance matrix (fraction of samples where pair pass/fail together):")
        for i in range(n_synapses):
            print(f"  {concordance[i, :]}")
    
    mean_concordance = np.mean(concordance[off_diag_idx])
    
    if verbose:
        print(f"\n  Mean pair concordance: {mean_concordance:.4f}")
        print(f"  (0.5 = random, 1.0 = perfectly correlated)")
    
    result = {
        'C_theoretical': C_theoretical,
        'C_empirical': C_empirical,
        'correlation_of_correlations': correlation_of_correlations,
        'concordance_matrix': concordance,
        'mean_concordance': mean_concordance,
        'samples': samples,
        'success': mean_concordance > 0.55  # Better than random
    }
    
    if verbose:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status}: Correlated sampling {'shows' if result['success'] else 'does NOT show'} coordination")
    
    return result


# =============================================================================
# DIAGNOSTIC 6: COORDINATED VS INDEPENDENT GATE
# =============================================================================

def diagnose_gate_evaluation(n_trials: int = 10, verbose: bool = True) -> Dict:
    """
    Test: Does coordinated gate evaluation produce different patterns than independent?
    
    FAST VERSION: Fewer trials, larger timesteps
    """
    if verbose:
        print_header("DIAGNOSTIC 6: COORDINATED vs INDEPENDENT GATE")
    
    coordinated_results = []
    independent_results = []
    
    n_synapses = 4
    
    for trial in range(n_trials):
        if verbose and trial % 2 == 0:
            print(f"  Running trial {trial+1}/{n_trials}...")
        
        # Create fresh network for each trial
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.environment.fraction_P31 = 1.0
        
        # === COORDINATED NETWORK ===
        network_coord = MultiSynapseNetwork(
            n_synapses=n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=2.0
        )
        network_coord.initialize(Model6QuantumSynapse, params)
        network_coord.set_microtubule_invasion(True)
        network_coord.use_correlated_sampling = True
        
        # FAST: 10ms timesteps, 500ms stimulation
        dt = 0.01
        for _ in range(50):
            network_coord.step(dt, {'voltage': -10e-3, 'reward': False})
        
        # FAST: 1s timesteps for decay, 30s total
        dt_decay = 1.0
        for _ in range(30):
            network_coord.step(dt_decay, {'voltage': -70e-3, 'reward': False})
        
        # Apply reward
        network_coord.step(0.1, {'voltage': -70e-3, 'reward': True})
        
        # Check commitments
        coord_committed = [getattr(s, '_camkii_committed', False) for s in network_coord.synapses]
        coordinated_results.append(coord_committed)
        
        # === INDEPENDENT NETWORK ===
        network_indep = MultiSynapseNetwork(
            n_synapses=n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=2.0
        )
        network_indep.initialize(Model6QuantumSynapse, params)
        network_indep.set_microtubule_invasion(True)
        network_indep.use_correlated_sampling = False
        
        # Same stimulation
        for _ in range(50):
            network_indep.step(dt, {'voltage': -10e-3, 'reward': False})
        
        # Same decay
        for _ in range(30):
            network_indep.step(dt_decay, {'voltage': -70e-3, 'reward': False})
        
        # Apply reward
        network_indep.step(0.1, {'voltage': -70e-3, 'reward': True})
        
        # Check commitments
        indep_committed = [getattr(s, '_camkii_committed', False) for s in network_indep.synapses]
        independent_results.append(indep_committed)
    
    # Analyze results
    coord_results = np.array(coordinated_results)
    indep_results = np.array(independent_results)
    
    # Count all-committed vs partial vs none
    coord_all = np.sum(np.all(coord_results, axis=1))
    coord_none = np.sum(~np.any(coord_results, axis=1))
    coord_partial = n_trials - coord_all - coord_none
    
    indep_all = np.sum(np.all(indep_results, axis=1))
    indep_none = np.sum(~np.any(indep_results, axis=1))
    indep_partial = n_trials - indep_all - indep_none
    
    if verbose:
        print(f"\nResults over {n_trials} trials:")
        print(f"\n  {'Condition':<15} {'All Commit':<12} {'Partial':<12} {'None':<12}")
        print(f"  {'-'*50}")
        print(f"  {'Coordinated':<15} {coord_all:<12} {coord_partial:<12} {coord_none:<12}")
        print(f"  {'Independent':<15} {indep_all:<12} {indep_partial:<12} {indep_none:<12}")
    
    # The key prediction: coordinated should have more all-or-none patterns
    coord_extreme = coord_all + coord_none
    indep_extreme = indep_all + indep_none
    
    if verbose:
        print(f"\n  Coordinated 'extreme' (all or none): {coord_extreme}/{n_trials}")
        print(f"  Independent 'extreme' (all or none): {indep_extreme}/{n_trials}")
        print(f"\n  Coordinated shows {'MORE' if coord_extreme > indep_extreme else 'LESS or EQUAL'} all-or-none patterns")
    
    result = {
        'coordinated_results': coord_results,
        'independent_results': indep_results,
        'coord_all': coord_all,
        'coord_partial': coord_partial,
        'coord_none': coord_none,
        'indep_all': indep_all,
        'indep_partial': indep_partial,
        'indep_none': indep_none,
        'success': coord_extreme > indep_extreme
    }
    
    if verbose:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"\n{status}: Coordination {'produces' if result['success'] else 'does NOT produce'} expected pattern")
    
    return result


# =============================================================================
# MAIN DIAGNOSTIC RUNNER
# =============================================================================

def run_full_diagnostic():
    """Run all diagnostics and summarize results"""
    
    print("\n" + "=" * 70)
    print("MULTI-SYNAPSE COORDINATED DECOHERENCE DIAGNOSTIC")
    print("=" * 70)
    print("\nThis diagnostic checks each component of the coordinated")
    print("decoherence system to identify where issues may exist.")
    
    results = {}
    
    # Diagnostic 1: Dimer Formation
    dimer_result = diagnose_dimer_formation(n_synapses=4)
    results['dimer_formation'] = dimer_result
    
    if not dimer_result['success']:
        print("\n⚠ STOPPING: Dimer formation failed. Fix this first.")
        return results
    
    # Use the network from diagnostic 1 for subsequent tests
    network = dimer_result['network']
    
    # Diagnostic 2: Entanglement Bonds
    bond_result = diagnose_entanglement_bonds(network)
    results['entanglement_bonds'] = bond_result
    
    # Diagnostic 3: Correlation Matrix
    corr_result = diagnose_correlation_matrix(network)
    results['correlation_matrix'] = corr_result
    
    # Diagnostic 4: Eligibility Decay
    decay_result = diagnose_eligibility_decay(network)
    results['eligibility_decay'] = decay_result
    
    # Diagnostic 5: Correlated Sampling (fresh network)
    sampling_result = diagnose_correlated_sampling(n_synapses=4, n_samples=100)
    results['correlated_sampling'] = sampling_result
    
    # Diagnostic 6: Gate Evaluation (multiple fresh networks)
    gate_result = diagnose_gate_evaluation(n_trials=20)
    results['gate_evaluation'] = gate_result
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    all_tests = [
        ('Dimer Formation', results['dimer_formation']['success']),
        ('Entanglement Bonds', results['entanglement_bonds']['success']),
        ('Correlation Matrix', results['correlation_matrix']['success']),
        ('Eligibility Decay', results['eligibility_decay']['success']),
        ('Correlated Sampling', results['correlated_sampling']['success']),
        ('Gate Evaluation', results['gate_evaluation']['success']),
    ]
    
    print("\n  Test                    Status")
    print("  " + "-" * 40)
    for name, passed in all_tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<24} {status}")
    
    n_passed = sum(1 for _, p in all_tests if p)
    n_total = len(all_tests)
    
    print(f"\n  Overall: {n_passed}/{n_total} tests passed")
    
    if n_passed < n_total:
        print("\n  RECOMMENDATIONS:")
        if not results['dimer_formation']['success']:
            print("  • Check calcium dynamics and ATP hydrolysis parameters")
        if not results['entanglement_bonds']['success']:
            print("  • Check bond formation criteria in entanglement tracker")
            print("  • Verify dimers have P_S > 0.5 (coherence threshold)")
        if not results['correlation_matrix']['success']:
            print("  • Verify bond counting in get_synapse_correlation_matrix()")
        if not results['eligibility_decay']['success']:
            print("  • Adjust delay times to bring eligibility near threshold")
        if not results['correlated_sampling']['success']:
            print("  • Check covariance matrix construction in sample_correlated_eligibilities()")
        if not results['gate_evaluation']['success']:
            print("  • Verify use_correlated_sampling flag is respected")
            print("  • Check that eligibility is near threshold at reward time")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_full_diagnostic()