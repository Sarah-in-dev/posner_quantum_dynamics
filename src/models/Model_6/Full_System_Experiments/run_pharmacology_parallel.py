#!/usr/bin/env python3
"""
Parallel Runner: Pharmacological Dissection Experiment
=======================================================

Run on EC2 c5.9xlarge (36 vCPU) with:
    nohup python3 -u run_pharmacology_parallel.py > log_pharmacology.txt 2>&1 &
    echo "PID: $!"

Monitor:
    tail -f log_pharmacology.txt

7 conditions x 20 trials = 140 trials
No long delays - each trial ~30s wall time
With 30 workers: ~5-10 minutes total

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

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_trial(args):
    """Worker function for parallel execution"""
    import logging
    logging.disable(logging.INFO)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    condition_name, nmda_blocked, mt_invaded, isoflurane_pct, trial_id, n_synapses = args

    try:
        from model6_core import Model6QuantumSynapse
        from model6_parameters import Model6Parameters
        from multi_synapse_network import MultiSynapseNetwork

        # Create network with pharmacological condition
        params = Model6Parameters()
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True

        if nmda_blocked:
            params.calcium.nmda_blocked = True

        if isoflurane_pct > 0:
            params.tryptophan.anesthetic_applied = True
            params.tryptophan.anesthetic_type = 'isoflurane'
            params.tryptophan.anesthetic_blocking_factor = isoflurane_pct / 100.0

        network = MultiSynapseNetwork(
            n_synapses=n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(mt_invaded)

        dt = 0.001
        start = time.time()

        # Track peaks
        peak_field = 0.0
        peak_dimers = 0

        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})

        # === PHASE 2: STIMULATION + DOPAMINE (theta-burst) ===
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):  # 2ms depolarization
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):  # 8ms rest
                    network.step(dt, {"voltage": -70e-3, "reward": True})
            for _ in range(160):  # 160ms inter-burst interval
                network.step(dt, {"voltage": -70e-3, "reward": True})

            # Track peaks
            metrics = network.get_experimental_metrics()
            if metrics.get('mean_field_kT', 0) > peak_field:
                peak_field = metrics.get('mean_field_kT', 0)

            current_dimers = sum(
                len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                for s in network.synapses
            )
            peak_dimers = max(peak_dimers, current_dimers)

        # Peak calcium across all synapses
        peak_calcium = float(max(getattr(s, '_peak_calcium_uM', 0.0) for s in network.synapses))

        # === PHASE 3: CONSOLIDATION (1s) ===
        n_consol = int(1.0 / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})

        # === FINAL MEASUREMENTS ===
        # Dimer singlet probabilities
        all_ps = []
        total_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                total_dimers += len(s.dimer_particles.dimers)
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        mean_singlet_prob = float(np.mean(all_ps)) if all_ps else 0.0

        # Eligibility
        eligibility = float(np.mean([s.get_eligibility() for s in network.synapses]))

        # Commitment (auto-commitment mode, read network-level flags)
        committed = network.network_committed
        commitment_level = float(network.network_commitment_level)
        final_strength = 1.0 + 0.5 * commitment_level if committed else 1.0

        elapsed = time.time() - start

        return {
            'condition': condition_name,
            'nmda_blocked': nmda_blocked,
            'mt_invaded': mt_invaded,
            'isoflurane_pct': isoflurane_pct,
            'trial_id': trial_id,
            'n_synapses': n_synapses,
            'peak_calcium_uM': peak_calcium,
            'peak_em_field_kT': peak_field,
            'peak_dimers': peak_dimers,
            'final_dimers': total_dimers,
            'mean_singlet_prob': mean_singlet_prob,
            'eligibility': eligibility,
            'committed': committed,
            'commitment_level': commitment_level,
            'final_strength': final_strength,
            'runtime_s': elapsed
        }

    except Exception as e:
        return {
            'condition': condition_name,
            'nmda_blocked': nmda_blocked,
            'mt_invaded': mt_invaded,
            'isoflurane_pct': isoflurane_pct,
            'trial_id': trial_id,
            'n_synapses': n_synapses,
            'error': str(e),
            'peak_calcium_uM': 0.0,
            'peak_em_field_kT': 0.0,
            'peak_dimers': 0,
            'final_dimers': 0,
            'mean_singlet_prob': 0.0,
            'eligibility': 0.0,
            'committed': False,
            'commitment_level': 0.0,
            'final_strength': 1.0,
            'runtime_s': 0.0
        }


if __name__ == '__main__':
    N_WORKERS = 30
    N_SYNAPSES = 10
    N_TRIALS = 20

    # Define conditions: (name, nmda_blocked, mt_invaded, isoflurane_pct)
    CONDITIONS = [
        ('Control',        False, True,  0),
        ('APV',            True,  True,  0),
        ('Nocodazole',     False, False, 0),
        ('Isoflurane_25%', False, True,  25),
        ('Isoflurane_50%', False, True,  50),
        ('Isoflurane_75%', False, True,  75),
        ('Isoflurane_100%',False, True,  100),
    ]

    print("=" * 70, flush=True)
    print("PHARMACOLOGY PARALLEL RUNNER", flush=True)
    print("=" * 70, flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)
    print(f"Synapses: {N_SYNAPSES}", flush=True)
    print(f"Trials per condition: {N_TRIALS}", flush=True)
    print(f"Conditions: {len(CONDITIONS)}", flush=True)
    for name, nmda, mt, iso in CONDITIONS:
        print(f"  {name}: nmda_blocked={nmda}, mt_invaded={mt}, iso={iso}%", flush=True)

    # Build job list
    jobs = []
    for name, nmda_blocked, mt_invaded, iso_pct in CONDITIONS:
        for trial_id in range(N_TRIALS):
            jobs.append((name, nmda_blocked, mt_invaded, iso_pct, trial_id, N_SYNAPSES))

    total = len(jobs)
    print(f"\nTotal trials: {total}", flush=True)
    print(f"Starting at {time.strftime('%H:%M:%S')}...\n", flush=True)

    results = []
    completed = 0
    errors = 0
    t0 = time.time()

    with mp.Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_single_trial, jobs):
            completed += 1

            if 'error' in r:
                errors += 1
                print(f"  [{completed}/{total}] ERROR {r['condition']} "
                      f"trial={r['trial_id']}: {r['error']}", flush=True)
            else:
                results.append(r)
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0

                print(f"  [{completed}/{total}] {r['condition']:<20} "
                      f"Ca={r['peak_calcium_uM']:>6.1f}uM  "
                      f"Q1={r['peak_em_field_kT']:>5.1f}kT  "
                      f"dim={r['peak_dimers']:>4d}  "
                      f"elig={r['eligibility']:.3f}  "
                      f"commit={r['committed']}  "
                      f"str={r['final_strength']:.3f}  "
                      f"({r['runtime_s']:.1f}s)  "
                      f"ETA: {eta/60:.1f}min", flush=True)

    total_time = time.time() - t0

    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPLETE: {len(results)} results in {total_time/60:.1f} min "
          f"({errors} errors)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Save raw results
    Path('results').mkdir(exist_ok=True)
    raw_path = 'results/pharmacology_parallel_raw.json'
    with open(raw_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {raw_path}", flush=True)

    # Compute and save summary
    summary = {}
    for name, _, _, _ in CONDITIONS:
        cond_results = [r for r in results if r['condition'] == name]
        if not cond_results:
            continue

        summary[name] = {
            'n_trials': len(cond_results),
            'calcium_mean': float(np.mean([r['peak_calcium_uM'] for r in cond_results])),
            'calcium_std': float(np.std([r['peak_calcium_uM'] for r in cond_results])),
            'calcium_sem': float(np.std([r['peak_calcium_uM'] for r in cond_results]) / np.sqrt(len(cond_results))),
            'em_field_mean': float(np.mean([r['peak_em_field_kT'] for r in cond_results])),
            'em_field_std': float(np.std([r['peak_em_field_kT'] for r in cond_results])),
            'em_field_sem': float(np.std([r['peak_em_field_kT'] for r in cond_results]) / np.sqrt(len(cond_results))),
            'dimers_mean': float(np.mean([r['peak_dimers'] for r in cond_results])),
            'dimers_std': float(np.std([r['peak_dimers'] for r in cond_results])),
            'dimers_sem': float(np.std([r['peak_dimers'] for r in cond_results]) / np.sqrt(len(cond_results))),
            'eligibility_mean': float(np.mean([r['eligibility'] for r in cond_results])),
            'eligibility_std': float(np.std([r['eligibility'] for r in cond_results])),
            'eligibility_sem': float(np.std([r['eligibility'] for r in cond_results]) / np.sqrt(len(cond_results))),
            'commit_rate': float(np.mean([1 if r['committed'] else 0 for r in cond_results])),
            'strength_mean': float(np.mean([r['final_strength'] for r in cond_results])),
            'strength_std': float(np.std([r['final_strength'] for r in cond_results])),
            'strength_sem': float(np.std([r['final_strength'] for r in cond_results]) / np.sqrt(len(cond_results))),
        }

    summary_path = 'results/pharmacology_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)

    # Print summary table
    print(f"\n{'=' * 90}", flush=True)
    print(f"{'Condition':<20} {'Ca (uM)':<12} {'Q1 (kT)':<12} {'Dimers':<10} {'Elig':<10} {'Commit':<10} {'Strength':<10}", flush=True)
    print(f"{'-' * 90}", flush=True)
    for name, _, _, _ in CONDITIONS:
        s = summary.get(name, {})
        print(f"{name:<20} "
              f"{s.get('calcium_mean',0):>6.1f}±{s.get('calcium_sem',0):<4.1f} "
              f"{s.get('em_field_mean',0):>6.1f}±{s.get('em_field_sem',0):<4.1f} "
              f"{s.get('dimers_mean',0):>5.0f}±{s.get('dimers_sem',0):<3.0f} "
              f"{s.get('eligibility_mean',0):>6.3f}    "
              f"{s.get('commit_rate',0):>5.0%}     "
              f"{s.get('strength_mean',1):>5.3f}", flush=True)
    print(f"{'=' * 90}", flush=True)