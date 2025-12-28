#!/usr/bin/env python3
"""
Master Experiment Runner
=========================

Runs complete Tier 1, 2, 3 experiment suites with:
- Proper output organization
- Multiple replicates for statistical power
- Full logging of terminal output
- Comprehensive JSON summaries

Usage:
    # Single full run (default)
    python run_all_tiers.py
    
    # Quick validation
    python run_all_tiers.py --quick
    
    # Multiple replicates for publication
    python run_all_tiers.py --replicates 5
    
    # Specific tier only
    python run_all_tiers.py --tier 3

Output Structure:
    results/
    └── run_full_YYYYMMDD_HHMMSS/
        ├── run_config.json
        ├── run_log.txt              # Complete terminal output
        ├── rep_01/
        │   ├── tier1/
        │   │   ├── tier1_HHMMSS/
        │   │   │   ├── isotope/
        │   │   │   ├── entanglement/
        │   │   │   └── summary.json
        │   ├── tier2/
        │   │   └── tier2_HHMMSS/
        │   │       ├── network_threshold/
        │   │       ├── three_factor_gate/
        │   │       └── summary.json
        │   └── tier3/
        │       └── tier3_HHMMSS/
        │           ├── isotope/
        │           ├── dopamine_timing/
        │           └── tier3_summary.json
        ├── rep_02/
        │   └── ...
        └── aggregate_summary.json   # Stats across all replicates

Author: Sarah Davidson
Date: December 2025
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class TeeLogger:
    """Write to both file and stdout simultaneously"""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def run_tier(tier: int, output_dir: Path, quick: bool = False) -> Dict:
    """
    Run a single tier's experiments
    
    Returns dict with results and timing info
    """
    script_name = f"run_tier{tier}.py"
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"ERROR: {script_name} not found at {script_path}")
        return {'error': f'{script_name} not found', 'success': False}
    
    # Build command
    cmd = [sys.executable, str(script_path), '--output', str(output_dir)]
    if quick:
        cmd.append('--quick')
    cmd.append('--no-plots')  # Don't try to display plots on AWS
    
    print(f"\n{'='*70}")
    print(f"RUNNING TIER {tier}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}")
    print()
    
    start_time = time.time()
    
    # Run and capture output
    result = subprocess.run(
        cmd,
        capture_output=False,  # Let output flow to our logger
        text=True,
        cwd=Path(__file__).parent
    )
    
    elapsed = time.time() - start_time
    
    # Load the summary file that the tier script created
    # The tier scripts create tier{N}_{timestamp}/ subdirs, we need to find it
    tier_dirs = list(output_dir.glob(f"tier{tier}_*"))
    summary_data = {}
    
    if tier_dirs:
        # Get most recent
        tier_dir = sorted(tier_dirs)[-1]
        summary_file = tier_dir / f"tier{tier}_summary.json"
        if not summary_file.exists():
            summary_file = tier_dir / "summary.json"
        
        if summary_file.exists():
            with open(summary_file) as f:
                summary_data = json.load(f)
    
    return {
        'success': result.returncode == 0,
        'returncode': result.returncode,
        'elapsed_seconds': elapsed,
        'output_dir': str(tier_dirs[-1]) if tier_dirs else None,
        'summary': summary_data
    }


def run_replicate(rep_id: int, rep_dir: Path, tiers: List[int], quick: bool) -> Dict:
    """Run all specified tiers for one replicate"""
    
    print(f"\n{'#'*70}")
    print(f"# REPLICATE {rep_id}")
    print(f"# Output: {rep_dir}")
    print(f"{'#'*70}")
    
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    rep_results = {
        'replicate_id': rep_id,
        'start_time': datetime.now().isoformat(),
        'tiers': {}
    }
    
    for tier in tiers:
        tier_output = rep_dir / f"tier{tier}"
        tier_output.mkdir(parents=True, exist_ok=True)
        
        tier_result = run_tier(tier, tier_output, quick=quick)
        rep_results['tiers'][f'tier{tier}'] = tier_result
        
        if not tier_result['success']:
            print(f"\nWARNING: Tier {tier} failed with code {tier_result['returncode']}")
    
    rep_results['end_time'] = datetime.now().isoformat()
    
    # Save replicate summary
    rep_summary_path = rep_dir / 'replicate_summary.json'
    with open(rep_summary_path, 'w') as f:
        json.dump(rep_results, f, indent=2, default=str)
    
    return rep_results


def aggregate_results(run_dir: Path, all_rep_results: List[Dict]) -> Dict:
    """Aggregate results across all replicates"""
    
    aggregate = {
        'n_replicates': len(all_rep_results),
        'all_successful': all(
            all(t['success'] for t in rep['tiers'].values())
            for rep in all_rep_results
        ),
        'tiers': {}
    }
    
    # Collect per-tier statistics
    tier_names = set()
    for rep in all_rep_results:
        tier_names.update(rep['tiers'].keys())
    
    for tier_name in sorted(tier_names):
        tier_results = []
        for rep in all_rep_results:
            if tier_name in rep['tiers']:
                tier_results.append(rep['tiers'][tier_name])
        
        # Compute timing stats
        times = [t['elapsed_seconds'] for t in tier_results if t.get('elapsed_seconds')]
        
        aggregate['tiers'][tier_name] = {
            'n_runs': len(tier_results),
            'n_successful': sum(1 for t in tier_results if t['success']),
            'timing': {
                'mean_seconds': sum(times) / len(times) if times else 0,
                'min_seconds': min(times) if times else 0,
                'max_seconds': max(times) if times else 0,
            }
        }
        
        # Try to aggregate experiment-level results
        experiment_results = {}
        for t in tier_results:
            summary = t.get('summary', {})
            results = summary.get('results', {})
            for exp_name, exp_data in results.items():
                if exp_name not in experiment_results:
                    experiment_results[exp_name] = []
                experiment_results[exp_name].append(exp_data)
        
        if experiment_results:
            aggregate['tiers'][tier_name]['experiments'] = {}
            for exp_name, exp_runs in experiment_results.items():
                aggregate['tiers'][tier_name]['experiments'][exp_name] = {
                    'n_runs': len(exp_runs),
                    'runs': exp_runs
                }
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(
        description='Master experiment runner for all tiers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_tiers.py                    # Single full run, all tiers
    python run_all_tiers.py --quick            # Quick validation
    python run_all_tiers.py --replicates 5     # 5 full runs for statistics
    python run_all_tiers.py --tier 3           # Only tier 3
    python run_all_tiers.py --tier 1 --tier 2  # Tiers 1 and 2 only
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer trials, shorter durations)')
    parser.add_argument('--replicates', type=int, default=1,
                        help='Number of complete replicates to run (default: 1)')
    parser.add_argument('--tier', type=int, action='append', dest='tiers',
                        choices=[1, 2, 3],
                        help='Specific tier(s) to run (default: all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Base output directory (default: results)')
    
    args = parser.parse_args()
    
    # Default to all tiers
    tiers = args.tiers if args.tiers else [1, 2, 3]
    tiers = sorted(set(tiers))
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "quick" if args.quick else "full"
    run_dir = Path(args.output) / f"run_{mode_str}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_path = run_dir / 'run_log.txt'
    logger = TeeLogger(log_path)
    sys.stdout = logger
    
    print(f"{'='*70}")
    print(f"MASTER EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"\nStart time: {datetime.now().isoformat()}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Tiers: {tiers}")
    print(f"Replicates: {args.replicates}")
    print(f"Output directory: {run_dir}")
    print()
    
    # Save run configuration
    config = {
        'timestamp': timestamp,
        'mode': mode_str,
        'tiers': tiers,
        'n_replicates': args.replicates,
        'output_dir': str(run_dir),
        'python_version': sys.version,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    }
    
    config_path = run_dir / 'run_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run all replicates
    all_rep_results = []
    total_start = time.time()
    
    for rep_id in range(1, args.replicates + 1):
        rep_dir = run_dir / f"rep_{rep_id:02d}"
        rep_result = run_replicate(rep_id, rep_dir, tiers, args.quick)
        all_rep_results.append(rep_result)
    
    total_elapsed = time.time() - total_start
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATING RESULTS")
    print(f"{'='*70}")
    
    aggregate = aggregate_results(run_dir, all_rep_results)
    aggregate['total_runtime_seconds'] = total_elapsed
    aggregate['total_runtime_human'] = f"{total_elapsed/60:.1f} minutes"
    
    aggregate_path = run_dir / 'aggregate_summary.json'
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("RUN COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Replicates completed: {len(all_rep_results)}")
    print(f"All successful: {aggregate['all_successful']}")
    print(f"\nResults saved to: {run_dir}")
    print(f"  - run_config.json")
    print(f"  - run_log.txt")
    for rep_id in range(1, args.replicates + 1):
        print(f"  - rep_{rep_id:02d}/")
    print(f"  - aggregate_summary.json")
    
    # Per-tier summary
    print(f"\n--- TIER SUMMARY ---")
    for tier_name, tier_data in aggregate['tiers'].items():
        timing = tier_data['timing']
        print(f"\n{tier_name.upper()}:")
        print(f"  Successful: {tier_data['n_successful']}/{tier_data['n_runs']}")
        print(f"  Time: {timing['mean_seconds']/60:.1f} min (range: {timing['min_seconds']/60:.1f}-{timing['max_seconds']/60:.1f})")
    
    # Restore stdout and close log
    sys.stdout = logger.terminal
    logger.close()
    
    print(f"\nLog saved to: {log_path}")
    
    return 0 if aggregate['all_successful'] else 1


if __name__ == "__main__":
    sys.exit(main())