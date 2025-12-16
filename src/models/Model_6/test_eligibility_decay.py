"""
Test Eligibility Decay Fix
===========================
Validates that eligibility persists during delay period with correct T2.

The fix: Formed dimers use intrinsic J-coupling (~15 Hz) for coherence decay,
not the external ATP-driven J-field (which drops to 0.2 Hz during delay).

Expected results:
- P31: Eligibility at 15s delay should be ~0.8 (T2 ~ 67-100s)
- P32: Eligibility at 0.5s delay should be ~0.2 (T2 ~ 0.3s)

Before fix: P31 eligibility crashed to 0 by 15s (T2 ~ 0.8s)
After fix: P31 eligibility should persist (T2 ~ 67-100s)
"""

import sys
import os
import numpy as np

# Add Model_6 to path - adjust this to your local path
sys.path.insert(0, os.path.expanduser('~/posner_quantum_dynamics/src/models/Model_6'))

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse


def run_single_test(isotope='P31', delay_seconds=15.0, verbose=True):
    """
    Run stimulation → delay → measure eligibility
    
    Args:
        isotope: 'P31' or 'P32'
        delay_seconds: Duration of delay phase
        verbose: Print progress
        
    Returns:
        dict with results
    """
    # Configure parameters
    params = Model6Parameters()
    
    if isotope == 'P31':
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
    else:
        params.environment.fraction_P31 = 0.0
        params.environment.fraction_P32 = 1.0
    
    # Initialize model
    model = Model6QuantumSynapse(params)
    
    dt = 0.001  # 1 ms timestep
    
    # Track metrics
    results = {
        'isotope': isotope,
        'delay_seconds': delay_seconds,
        'timeline': [],
    }
    
    # === PHASE 1: BASELINE (100 ms) ===
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {isotope} with {delay_seconds}s delay")
        print(f"{'='*60}")
        print("Phase 1: Baseline (100 ms)")
    
    for i in range(100):
        model.step(dt, {'voltage': -70e-3, 'reward': False})
    
    baseline_metrics = model.quantum.get_experimental_metrics()
    if verbose:
        print(f"  Coherence: {baseline_metrics['coherence_mean']:.4f}")
        print(f"  T2: {baseline_metrics['T2_dimer_s']:.1f} s")
    
    # === PHASE 2: STIMULATION (200 ms) ===
    if verbose:
        print("Phase 2: Stimulation (200 ms)")
    
    for i in range(200):
        model.step(dt, {'voltage': -10e-3, 'reward': False})
    
    post_stim_metrics = model.quantum.get_experimental_metrics()
    post_stim_eligibility = model.eligibility.get_eligibility()
    
    if verbose:
        print(f"  Coherence: {post_stim_metrics['coherence_mean']:.4f}")
        print(f"  Eligibility: {post_stim_eligibility:.4f}")
        print(f"  T2: {post_stim_metrics['T2_dimer_s']:.1f} s")
        print(f"  Dimers: {post_stim_metrics['dimer_peak_nM']:.1f} nM")
    
    results['post_stim_coherence'] = post_stim_metrics['coherence_mean']
    results['post_stim_eligibility'] = post_stim_eligibility
    results['post_stim_T2'] = post_stim_metrics['T2_dimer_s']
    
    # === PHASE 3: DELAY ===
    if verbose:
        print(f"Phase 3: Delay ({delay_seconds}s)")
    
    # Use coarser timestep for long delays
    if delay_seconds > 1:
        dt_delay = 0.01  # 10 ms
    else:
        dt_delay = 0.001  # 1 ms
    
    n_steps = int(delay_seconds / dt_delay)
    
    # Record at intervals
    record_interval = max(1, n_steps // 10)
    
    for i in range(n_steps):
        model.step(dt_delay, {'voltage': -70e-3, 'reward': False})
        
        if i % record_interval == 0:
            metrics = model.quantum.get_experimental_metrics()
            elig = model.eligibility.get_eligibility()
            t = (i + 1) * dt_delay
            results['timeline'].append({
                'time': t,
                'coherence': metrics['coherence_mean'],
                'eligibility': elig,
                'T2': metrics['T2_dimer_s'],
            })
            if verbose and i % (record_interval * 2) == 0:
                print(f"  t={t:.1f}s: coherence={metrics['coherence_mean']:.4f}, "
                      f"elig={elig:.4f}, T2={metrics['T2_dimer_s']:.1f}s")
    
    # Final measurement
    final_metrics = model.quantum.get_experimental_metrics()
    final_eligibility = model.eligibility.get_eligibility()
    
    results['final_coherence'] = final_metrics['coherence_mean']
    results['final_eligibility'] = final_eligibility
    results['final_T2'] = final_metrics['T2_dimer_s']
    
    if verbose:
        print(f"\n  FINAL: coherence={final_metrics['coherence_mean']:.4f}, "
              f"eligibility={final_eligibility:.4f}")
    
    # Calculate effective decay
    if results['post_stim_eligibility'] > 0.01:
        # Fit: eligibility(t) = eligibility(0) * exp(-t/tau)
        # tau = -t / ln(eligibility(t) / eligibility(0))
        ratio = results['final_eligibility'] / results['post_stim_eligibility']
        if ratio > 0.001:
            tau_fitted = -delay_seconds / np.log(ratio)
            results['fitted_tau'] = tau_fitted
            if verbose:
                print(f"  Fitted tau: {tau_fitted:.1f} s")
        else:
            results['fitted_tau'] = 0.0
            if verbose:
                print(f"  Fitted tau: ~0 (eligibility crashed)")
    else:
        results['fitted_tau'] = None
    
    return results


def run_validation_suite():
    """Run full validation comparing P31 vs P32"""
    
    print("\n" + "="*70)
    print("ELIGIBILITY DECAY VALIDATION")
    print("Testing intrinsic J-coupling fix")
    print("="*70)
    
    # Test P31 at multiple delays
    print("\n" + "-"*70)
    print("P31 TESTS (Expected T2 ~ 67-100s)")
    print("-"*70)
    
    p31_results = []
    for delay in [0, 5, 15, 30]:
        result = run_single_test('P31', delay_seconds=delay, verbose=True)
        p31_results.append(result)
    
    # Test P32 at short delays
    print("\n" + "-"*70)
    print("P32 TESTS (Expected T2 ~ 0.3s)")
    print("-"*70)
    
    p32_results = []
    for delay in [0, 0.5, 1.0, 2.0]:
        result = run_single_test('P32', delay_seconds=delay, verbose=True)
        p32_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nP31 Results:")
    print(f"  {'Delay (s)':<12} {'Eligibility':<15} {'Fitted tau (s)':<15}")
    for r in p31_results:
        tau_str = f"{r['fitted_tau']:.1f}" if r['fitted_tau'] else "N/A"
        print(f"  {r['delay_seconds']:<12} {r['final_eligibility']:<15.4f} {tau_str:<15}")
    
    print("\nP32 Results:")
    print(f"  {'Delay (s)':<12} {'Eligibility':<15} {'Fitted tau (s)':<15}")
    for r in p32_results:
        tau_str = f"{r['fitted_tau']:.1f}" if r['fitted_tau'] else "N/A"
        print(f"  {r['delay_seconds']:<12} {r['final_eligibility']:<15.4f} {tau_str:<15}")
    
    # Validation checks
    print("\n" + "-"*70)
    print("VALIDATION")
    print("-"*70)
    
    # P31 at 15s should have eligibility > 0.5 (was 0 before fix)
    p31_15s = next((r for r in p31_results if r['delay_seconds'] == 15), None)
    if p31_15s:
        if p31_15s['final_eligibility'] > 0.5:
            print(f"✓ P31 at 15s: eligibility = {p31_15s['final_eligibility']:.3f} (expected > 0.5)")
        else:
            print(f"✗ P31 at 15s: eligibility = {p31_15s['final_eligibility']:.3f} (expected > 0.5) - FIX NOT WORKING")
    
    # P32 at 0.5s should have eligibility < 0.3 (rapid decay)
    p32_05s = next((r for r in p32_results if r['delay_seconds'] == 0.5), None)
    if p32_05s:
        if p32_05s['final_eligibility'] < 0.3:
            print(f"✓ P32 at 0.5s: eligibility = {p32_05s['final_eligibility']:.3f} (expected < 0.3)")
        else:
            print(f"✗ P32 at 0.5s: eligibility = {p32_05s['final_eligibility']:.3f} (expected < 0.3)")
    
    # Isotope ratio check
    p31_0s = next((r for r in p31_results if r['delay_seconds'] == 0), None)
    p32_0s = next((r for r in p32_results if r['delay_seconds'] == 0), None)
    
    if p31_0s and p32_0s and p32_0s['post_stim_eligibility'] > 0:
        ratio = p31_0s['post_stim_eligibility'] / p32_0s['post_stim_eligibility']
        print(f"\nIsotope effect at t=0: P31/P32 eligibility ratio = {ratio:.1f}x")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_validation_suite()