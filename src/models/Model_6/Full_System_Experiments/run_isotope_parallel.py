#!/usr/bin/env python3
"""
Parallel Runner: Isotope Comparison Experiment (P31 vs P32)
============================================================

The definitive test of quantum coherence necessity in synaptic plasticity.

P31 (I=1/2): T2 ≈ 225s → eligibility persists for minutes
P32 (I=1):   T2 ≈ 0.39s → eligibility decays in <1s

Protocol identical to dopamine timing:
  1. Baseline (100ms)
  2. Theta-burst stimulation (NO dopamine)
  3. Variable delay
  4. Measure eligibility at readout onset
  5. Dopamine readout (300ms) with coordination mode
  6. Consolidation (1s)

Run on EC2 c5.9xlarge (36 vCPU) with:
    nohup python3 -u run_isotope_parallel.py > log_isotope.txt 2>&1 &
    echo "PID: $!"

Monitor:
    tail -f log_isotope.txt

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

    isotope, delay, trial_id, n_synapses = args

    try:
        from model6_core import Model6QuantumSynapse
        from model6_parameters import Model6Parameters
        from multi_synapse_network import MultiSynapseNetwork

        # Create network with isotope setting
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True

        if isotope == 'P31':
            params.environment.fraction_P31 = 1.0
            params.environment.fraction_P32 = 0.0
        else:  # P32
            params.environment.fraction_P31 = 0.0
            params.environment.fraction_P32 = 1.0

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
        start = time.time()

        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})

        # === PHASE 2: STIMULATION (theta-burst, NO DOPAMINE) ===
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):
                    network.step(dt, {"voltage": -10e-3, "reward": False})
                for _ in range(8):
                    network.step(dt, {"voltage": -70e-3, "reward": False})
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": False})

        # Record post-stim state
        eligibility_post_stim = float(np.mean([
            s.get_eligibility() for s in network.synapses
        ]))
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
        eligibility_at_readout = float(np.mean([
            s.get_eligibility() for s in network.synapses
        ]))

        all_ps = []
        total_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                total_dimers += len(s.dimer_particles.dimers)
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        mean_singlet_prob = float(np.mean(all_ps)) if all_ps else 0.0

        # === PHASE 5: DOPAMINE READOUT (300ms) ===
        for _ in range(300):
            network.step_with_coordination(dt, {"voltage": -70e-3, "reward": True})

        # === PHASE 6: CONSOLIDATION (1s) ===
        dt_consol = 0.01
        for _ in range(100):
            network.step(dt_consol, {"voltage": -70e-3, "reward": False})

        # === FINAL MEASUREMENTS ===
        syn_committed = [getattr(s, '_camkii_committed', False) for s in network.synapses]
        syn_levels = [getattr(s, '_committed_memory_level', 0.0) for s in network.synapses]

        committed = any(syn_committed)
        commitment_level = eligibility_at_readout if committed else 0.0
        n_syn_committed = sum(
            1 for s in network.synapses if getattr(s, '_camkii_committed', False)
        )

        final_strength = 1.0 + 0.5 * commitment_level if committed else 1.0
        delta_strength = final_strength - 1.0

        elapsed = time.time() - start

        return {
            'isotope': isotope,
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
            'isotope': isotope,
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
    N_WORKERS = 30
    N_SYNAPSES = 10
    N_TRIALS = 20  # Publication quality

    # P31: full delay series (same as dopamine timing)
    P31_DELAYS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200]

    # P32: short delays only (coherence gone in <1s)
    P32_DELAYS = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    print("=" * 70, flush=True)
    print("ISOTOPE COMPARISON PARALLEL RUNNER", flush=True)
    print("=" * 70, flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print(f"Synapses: {N_SYNAPSES}", flush=True)
    print(f"Trials per condition: {N_TRIALS}", flush=True)
    print(f"P31 delays: {P31_DELAYS}", flush=True)
    print(f"P32 delays: {P32_DELAYS}", flush=True)

    # Build job list
    jobs = []

    for delay in P31_DELAYS:
        for trial_id in range(N_TRIALS):
            jobs.append(('P31', delay, trial_id, N_SYNAPSES))

    for delay in P32_DELAYS:
        for trial_id in range(N_TRIALS):
            jobs.append(('P32', delay, trial_id, N_SYNAPSES))

    total = len(jobs)
    n_p31 = len(P31_DELAYS) * N_TRIALS
    n_p32 = len(P32_DELAYS) * N_TRIALS

    print(f"\nTotal trials: {total}", flush=True)
    print(f"  P31: {n_p31} ({len(P31_DELAYS)} delays × {N_TRIALS} trials)", flush=True)
    print(f"  P32: {n_p32} ({len(P32_DELAYS)} delays × {N_TRIALS} trials)", flush=True)
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
                print(f"  [{completed}/{total}] ERROR {r['isotope']} "
                      f"delay={r['delay']}s trial={r['trial_id']}: {r['error']}", flush=True)
            else:
                results.append(r)
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0

                print(f"  [{completed}/{total}] {r['isotope']} "
                      f"delay={r['delay']:>6.2f}s  "
                      f"elig={r['eligibility_at_readout']:.3f}  "
                      f"P_S={r['mean_singlet_prob']:.3f}  "
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
    raw_path = 'results/isotope_parallel_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {raw_path}", flush=True)

    # Compute and save summary
    summary = {}
    for isotope, delays in [('P31', P31_DELAYS), ('P32', P32_DELAYS)]:
        summary[isotope] = {}
        for delay in delays:
            d_results = [r for r in results
                         if r['isotope'] == isotope and r['delay'] == delay]
            if not d_results:
                continue

            eligs = [r['eligibility_at_readout'] for r in d_results]
            commits = [1 if r['committed'] else 0 for r in d_results]
            strengths = [r['delta_strength'] for r in d_results]
            singlets = [r['mean_singlet_prob'] for r in d_results]

            summary[isotope][str(delay)] = {
                'eligibility_mean': float(np.mean(eligs)),
                'eligibility_std': float(np.std(eligs)),
                'commit_rate': float(np.mean(commits)),
                'delta_strength_mean': float(np.mean(strengths)),
                'delta_strength_std': float(np.std(strengths)),
                'mean_singlet_prob': float(np.mean(singlets)),
                'n_trials': len(d_results)
            }

    # Fit T2 for each isotope
    for isotope, delays in [('P31', P31_DELAYS), ('P32', P32_DELAYS)]:
        x = np.array(delays, dtype=float)
        y = np.array([
            summary[isotope].get(str(d), {}).get('eligibility_mean', 0)
            for d in delays
        ])

        valid = y > 0.01
        if np.sum(valid) >= 2:
            try:
                log_y = np.log(y[valid])
                coeffs = np.polyfit(x[valid], log_y, 1)
                tau = -1.0 / coeffs[0] if coeffs[0] != 0 else 0
                A = np.exp(coeffs[1])

                y_pred = A * np.exp(-x[valid] / tau) if tau > 0 else np.zeros_like(y[valid])
                ss_res = np.sum((y[valid] - y_pred) ** 2)
                ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                summary[f'{isotope}_fitted_T2'] = float(max(0, tau))
                summary[f'{isotope}_r_squared'] = float(r_sq)
            except Exception:
                summary[f'{isotope}_fitted_T2'] = 0.0
                summary[f'{isotope}_r_squared'] = 0.0

    # T2 ratio
    p31_t2 = summary.get('P31_fitted_T2', 0)
    p32_t2 = summary.get('P32_fitted_T2', 0)
    summary['t2_ratio'] = p31_t2 / p32_t2 if p32_t2 > 0 else float('inf')

    summary_path = 'results/isotope_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)

    # Print quick summary
    print(f"\n{'=' * 70}", flush=True)
    print("QUICK SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    for isotope, delays in [('P31', P31_DELAYS), ('P32', P32_DELAYS)]:
        print(f"\n{isotope}:", flush=True)
        for delay in delays:
            s = summary[isotope].get(str(delay), {})
            if s:
                print(f"  delay={delay:>6.2f}s: "
                      f"elig={s['eligibility_mean']:.3f}±{s['eligibility_std']:.3f}  "
                      f"commit={s['commit_rate']:.0%}  "
                      f"P_S={s['mean_singlet_prob']:.3f}  "
                      f"(n={s['n_trials']})", flush=True)

    print(f"\nFitted T2:", flush=True)
    print(f"  P31: {summary.get('P31_fitted_T2', 0):.1f}s (R²={summary.get('P31_r_squared', 0):.3f})", flush=True)
    print(f"  P32: {summary.get('P32_fitted_T2', 0):.2f}s (R²={summary.get('P32_r_squared', 0):.3f})", flush=True)
    print(f"  Ratio: {summary.get('t2_ratio', 0):.0f}×", flush=True)
    print(f"\nTotal runtime: {total_time/60:.1f} minutes", flush=True)
    print("DONE", flush=True)