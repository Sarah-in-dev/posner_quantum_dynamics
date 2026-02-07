#!/usr/bin/env python3
"""
Parallel Runner: Dopamine Timing Experiment
=============================================

Run on EC2 c5.9xlarge (36 vCPU) with:
    nohup python3 -u run_dopamine_parallel.py > log_dopamine.txt 2>&1 &
    echo "PID: $!"

Monitor:
    tail -f log_dopamine.txt

Estimated runtime:
- 12 main delays × 10 trials = 120 stim+dopamine trials
- 6 control delays × 5 trials × 2 conditions = 60 control trials
- Total: 180 trials
- Long delays (200s) take ~10 min each
- With 30 workers: ~30-40 minutes total

Author: Sarah Davidson
Date: February 2026
"""

import logging
logging.disable(logging.INFO)

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_single_trial(args):
    """Worker function for parallel execution"""
    import logging
    logging.disable(logging.INFO)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    condition_type, delay, trial_id, n_synapses = args
    
    try:
        from model6_core import Model6QuantumSynapse
        from model6_parameters import Model6Parameters
        from multi_synapse_network import MultiSynapseNetwork
        
        # Create network
        params = Model6Parameters()
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        network = MultiSynapseNetwork(
            n_synapses=n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        network.set_coordination_mode(True)
        
        dt = 0.001
        
        do_stimulate = condition_type in ('stim_plus_dopamine', 'stim_no_dopamine')
        do_dopamine = condition_type in ('stim_plus_dopamine', 'dopamine_only')
        
        start = time.time()
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst, NO DOPAMINE) ===
        if do_stimulate:
            for burst in range(5):
                for spike in range(4):
                    for _ in range(2):
                        network.step(dt, {"voltage": -10e-3, "reward": False})
                    for _ in range(8):
                        network.step(dt, {"voltage": -70e-3, "reward": False})
                for _ in range(160):
                    network.step(dt, {"voltage": -70e-3, "reward": False})
        else:
            # dopamine_only: rest for equivalent duration
            for _ in range(1000):
                network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # Record post-stim state
        eligibility_post_stim = float(np.mean([s.get_eligibility() for s in network.synapses]))
        dimers_post_stim = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        peak_calcium_uM = float(max(
            getattr(s, '_peak_calcium_uM', 0.0) for s in network.synapses
        ))
        peak_em_field_kT = float(max(
            getattr(s, '_collective_field_kT', 0.0) for s in network.synapses
        ))
        
        # === PHASE 3: DELAY ===
        if delay > 0:
            dt_delay = 0.1 if delay > 5 else 0.05
            n_delay = int(delay / dt_delay)
            for _ in range(n_delay):
                network.step(dt_delay, {'voltage': -70e-3, 'reward': False})
        
        # === PHASE 4: MEASURE AT READOUT ONSET ===
        eligibility_at_readout = float(np.mean([s.get_eligibility() for s in network.synapses]))
        
        all_ps = []
        total_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                total_dimers += len(s.dimer_particles.dimers)
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        mean_singlet_prob = float(np.mean(all_ps)) if all_ps else 0.0
        
        # === PHASE 5: DOPAMINE READOUT (300ms) ===
        if do_dopamine:
            for _ in range(300):
                network.step_with_coordination(dt, {"voltage": -70e-3, "reward": True})
        else:
            for _ in range(300):
                network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 6: CONSOLIDATION (1s) ===
        dt_consol = 0.01
        for _ in range(100):
            network.step(dt_consol, {"voltage": -70e-3, "reward": False})
        
        # === FINAL MEASUREMENTS ===
        committed = bool(network.network_committed)
        commitment_level = float(network.network_commitment_level)
        n_syn_committed = sum(
            1 for s in network.synapses if getattr(s, '_camkii_committed', False)
        )
        
        final_strength = 1.0 + 0.5 * commitment_level if committed else 1.0
        delta_strength = final_strength - 1.0
        
        elapsed = time.time() - start
        
        return {
            'condition_type': condition_type,
            'delay': delay,
            'trial_id': trial_id,
            'n_synapses': n_synapses,
            'eligibility_post_stim': eligibility_post_stim,
            'dimers_post_stim': dimers_post_stim,
            'peak_calcium_uM': peak_calcium_uM,
            'peak_em_field_kT': peak_em_field_kT,
            'eligibility_at_readout': eligibility_at_readout,
            'mean_singlet_prob': mean_singlet_prob,
            'dimer_count': total_dimers,
            'committed': committed,
            'commitment_level': commitment_level,
            'n_synapses_committed': n_syn_committed,
            'final_strength': final_strength,
            'delta_strength': delta_strength,
            'runtime_s': elapsed
        }
        
    except Exception as e:
        return {
            'condition_type': condition_type,
            'delay': delay,
            'trial_id': trial_id,
            'n_synapses': n_synapses,
            'error': str(e),
            'delta_strength': 0.0,
            'eligibility_at_readout': 0.0,
            'committed': False,
            'runtime_s': 0.0
        }


if __name__ == '__main__':
    N_WORKERS = 30  # Leave 6 cores free for OS
    N_SYNAPSES = 10
    N_TRIALS_MAIN = 10
    N_TRIALS_CONTROL = 5
    
    # Delays
    MAIN_DELAYS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200]
    CONTROL_DELAYS = [0, 15, 30, 60, 120, 200]
    
    print("=" * 70, flush=True)
    print("DOPAMINE TIMING PARALLEL RUNNER", flush=True)
    print("=" * 70, flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print(f"Synapses: {N_SYNAPSES}", flush=True)
    print(f"Main delays: {MAIN_DELAYS}", flush=True)
    print(f"Control delays: {CONTROL_DELAYS}", flush=True)
    print(f"Trials (main/control): {N_TRIALS_MAIN}/{N_TRIALS_CONTROL}", flush=True)
    
    # Build job list
    jobs = []
    
    # Main condition: stim + dopamine at all delays
    for delay in MAIN_DELAYS:
        for trial_id in range(N_TRIALS_MAIN):
            jobs.append(('stim_plus_dopamine', delay, trial_id, N_SYNAPSES))
    
    # Control 1: stim only (no dopamine)
    for delay in CONTROL_DELAYS:
        for trial_id in range(N_TRIALS_CONTROL):
            jobs.append(('stim_no_dopamine', delay, trial_id, N_SYNAPSES))
    
    # Control 2: dopamine only (no prior activity)
    for delay in CONTROL_DELAYS:
        for trial_id in range(N_TRIALS_CONTROL):
            jobs.append(('dopamine_only', delay, trial_id, N_SYNAPSES))
    
    total = len(jobs)
    print(f"\nTotal trials: {total}", flush=True)
    print(f"  stim+dopamine: {len(MAIN_DELAYS) * N_TRIALS_MAIN}", flush=True)
    print(f"  stim_only:     {len(CONTROL_DELAYS) * N_TRIALS_CONTROL}", flush=True)
    print(f"  dopamine_only: {len(CONTROL_DELAYS) * N_TRIALS_CONTROL}", flush=True)
    print(f"\nStarting at {time.strftime('%H:%M:%S')}...\n", flush=True)
    
    results = []
    completed = 0
    errors = 0
    t0 = time.time()
    
    with mp.Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_single_trial, jobs):
            completed += 1
            
            if 'error' in r:
                errors += 1
                print(f"  [{completed}/{total}] ERROR {r['condition_type']} "
                      f"delay={r['delay']}s trial={r['trial_id']}: {r['error']}", flush=True)
            else:
                results.append(r)
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                
                print(f"  [{completed}/{total}] {r['condition_type']:<22} "
                      f"delay={r['delay']:>5.1f}s  "
                      f"Δstr={r['delta_strength']:.3f}  "
                      f"elig={r['eligibility_at_readout']:.3f}  "
                      f"commit={r['committed']}  "
                      f"({r['runtime_s']:.1f}s)  "
                      f"ETA: {eta/60:.1f}min", flush=True)
    
    total_time = time.time() - t0
    
    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPLETE: {len(results)} results in {total_time/60:.1f} min "
          f"({errors} errors)", flush=True)
    print(f"{'=' * 70}", flush=True)
    
    # Save raw results
    Path('results').mkdir(exist_ok=True)
    raw_path = 'results/dopamine_timing_parallel_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {raw_path}", flush=True)
    
    # Compute and save summary
    summary = {}
    for ctype in ['stim_plus_dopamine', 'stim_no_dopamine', 'dopamine_only']:
        ctype_results = [r for r in results if r['condition_type'] == ctype]
        delays = sorted(set(r['delay'] for r in ctype_results))
        
        summary[ctype] = {}
        for delay in delays:
            d_results = [r for r in ctype_results if r['delay'] == delay]
            ds = [r['delta_strength'] for r in d_results]
            eligs = [r['eligibility_at_readout'] for r in d_results]
            commits = [1 if r['committed'] else 0 for r in d_results]
            
            summary[ctype][str(delay)] = {
                'delta_strength_mean': float(np.mean(ds)),
                'delta_strength_std': float(np.std(ds)),
                'eligibility_mean': float(np.mean(eligs)),
                'eligibility_std': float(np.std(eligs)),
                'commit_rate': float(np.mean(commits)),
                'n_trials': len(d_results)
            }
    
    # Print quick summary
    print("\n--- QUICK SUMMARY ---", flush=True)
    print(f"{'Condition':<24} {'Delay':<8} {'Δ Strength':<16} {'Commit%':<10}", flush=True)
    print("-" * 60, flush=True)
    
    for ctype in ['stim_plus_dopamine', 'stim_no_dopamine', 'dopamine_only']:
        for delay_str in sorted(summary[ctype].keys(), key=float):
            s = summary[ctype][delay_str]
            print(f"{ctype:<24} {delay_str:>6}s  "
                  f"{s['delta_strength_mean']:.3f} ± {s['delta_strength_std']:.3f}   "
                  f"{s['commit_rate']:.0%}", flush=True)
    
    # Fit decay from stim_plus_dopamine
    from scipy.optimize import curve_fit
    
    spd = summary['stim_plus_dopamine']
    delays_fit = sorted(spd.keys(), key=float)
    x = np.array([float(d) for d in delays_fit])
    y = np.array([spd[d]['delta_strength_mean'] for d in delays_fit])
    
    try:
        def decay_func(t, A, T2):
            return A * np.exp(-t / T2)
        
        popt, _ = curve_fit(decay_func, x, y, p0=[max(y), 100.0],
                           bounds=([0, 1], [2.0, 1000]), maxfev=5000)
        
        y_pred = decay_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nFITTED DECAY: T₂ = {popt[1]:.1f} s, A = {popt[0]:.3f}, R² = {r_sq:.3f}", flush=True)
        
        summary['fit'] = {
            'fitted_T2': float(popt[1]),
            'fitted_A': float(popt[0]),
            'r_squared': float(r_sq)
        }
    except Exception as e:
        print(f"\nFit failed: {e}", flush=True)
    
    summary_path = 'results/dopamine_timing_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)