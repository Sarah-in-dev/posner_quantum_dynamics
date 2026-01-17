#!/usr/bin/env python3
"""
Run Model 6 Network Credit Assignment Experiment
=================================================

This script runs the proper credit assignment experiment using
full Model 6 physics for each synapse.

USAGE:
------
    python run_network_experiment.py --quick          # Quick test (5 trials, 4 delays)
    python run_network_experiment.py                  # Standard (10 trials, 6 delays)
    python run_network_experiment.py --full           # Full (20 trials, 8 delays)
    python run_network_experiment.py --aws            # AWS submission

OUTPUTS:
--------
    results/
        network_credit_assignment.png    # Main figure
        network_results.json             # Raw data
        network_experiment.log           # Detailed log

Author: Sarah Davidson
University of Florida
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from model6_recurrent_network import Model6RecurrentNetwork, NetworkConfig
from model6_credit_assignment_experiment import (
    ExperimentConfig, run_experiment, plot_results, print_summary, save_results
)
    
if __name__ == "__main__":
    # Configure experiment
    config = ExperimentConfig(
        n_neurons=10,           # 10 neurons = 90 synapses
        delays=[1, 5, 10, 20, 30, 60],
        n_trials=10,
        conditions=['P31', 'P32']
    )
        
    # Run experiment
    results = run_experiment(
        config=config,
        Model6Class=Model6QuantumSynapse,
        Model6Params=Model6Parameters,
        verbose=True
    )
        
    # Output
    output_dir = Path(__file__).parent / 'results'
        
    plot_results(results, output_dir / 'network_credit_assignment.png')
    save_results(results, output_dir / 'network_results.json')
    print_summary(results)

def setup_logging(output_dir: Path):
    """Configure logging"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'network_experiment.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run Model 6 Network Credit Assignment Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (5 trials, 4 delays)')
    parser.add_argument('--full', action='store_true', 
                       help='Full experiment (20 trials, 8 delays)')
    parser.add_argument('--aws', action='store_true',
                       help='Generate AWS batch submission')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (for headless servers)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed base')
    parser.add_argument('--n-neurons', type=int, default=10,
                       help='Number of neurons')
    
    args = parser.parse_args()
    
    # Setup output
    output_dir = Path(args.output)
    logger = setup_logging(output_dir)
    
    logger.info("="*70)
    logger.info("MODEL 6 NETWORK CREDIT ASSIGNMENT EXPERIMENT")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    
    # Try to import Model 6
    try:
        from model6_core import Model6QuantumSynapse
        from model6_parameters import Model6Parameters
        logger.info("Successfully imported Model 6 classes")
        MODEL6_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Could not import Model 6: {e}")
        logger.warning("Running in DEMO mode with mock classes")
        MODEL6_AVAILABLE = False
        
        # Create mock classes for testing
        class MockSynapse:
            def __init__(self, params=None):
                self._committed_memory_level = 0.0
                self._current_eligibility = 0.0
                self._camkii_committed = False
                self._collective_field_kT = 0.0
                self._time = 0.0
                self.dimer_particles = type('obj', (object,), {'dimers': []})()
                
            def step(self, dt, stimulus):
                self._time += dt
                # Mock dimer formation
                if stimulus.get('voltage', -70e-3) > -30e-3:
                    # Form dimers
                    for _ in range(2):
                        self.dimer_particles.dimers.append({'P_S': 1.0})
                
                # Mock eligibility decay
                if self.dimer_particles.dimers:
                    self._current_eligibility = len(self.dimer_particles.dimers) / 10.0
                    self._current_eligibility = min(1.0, self._current_eligibility)
                
                # Mock commitment on reward
                if stimulus.get('reward', False) and self._current_eligibility > 0.3:
                    self._committed_memory_level += 0.01 * self._current_eligibility
                    self._camkii_committed = True
                    
            def set_microtubule_invasion(self, invaded):
                pass
                
            def get_eligibility(self):
                return self._current_eligibility
                
            def get_mean_singlet_probability(self):
                if self.dimer_particles.dimers:
                    return sum(d.get('P_S', 0.25) for d in self.dimer_particles.dimers) / len(self.dimer_particles.dimers)
                return 0.25
        
        class MockParams:
            def __init__(self):
                self.environment = type('obj', (object,), {'fraction_P31': 1.0})()
                self.em_coupling_enabled = True
        
        Model6QuantumSynapse = MockSynapse
        Model6Parameters = MockParams
    
    # Import experiment framework
    from model6_recurrent_network import Model6RecurrentNetwork, NetworkConfig
    from model6_credit_assignment_experiment import (
        ExperimentConfig, run_experiment, plot_results, print_summary, save_results
    )
    
    # Configure based on mode
    if args.quick:
        config = ExperimentConfig(
            n_neurons=args.n_neurons,
            delays=[1, 10, 30, 60],
            n_trials=5,
            conditions=['P31', 'P32'],
            seed_base=args.seed
        )
        logger.info("Running in QUICK mode")
    elif args.full:
        config = ExperimentConfig(
            n_neurons=args.n_neurons,
            delays=[1, 5, 10, 20, 30, 60, 90, 120],
            n_trials=20,
            conditions=['P31', 'P32'],
            seed_base=args.seed
        )
        logger.info("Running in FULL mode")
    else:
        config = ExperimentConfig(
            n_neurons=args.n_neurons,
            delays=[1, 5, 10, 20, 30, 60],
            n_trials=10,
            conditions=['P31', 'P32'],
            seed_base=args.seed
        )
        logger.info("Running in STANDARD mode")
    
    logger.info(f"Configuration:")
    logger.info(f"  Neurons: {config.n_neurons}")
    logger.info(f"  Delays: {config.delays}")
    logger.info(f"  Trials: {config.n_trials}")
    logger.info(f"  Conditions: {config.conditions}")
    
    if args.aws:
        # Generate AWS batch submission script
        logger.info("Generating AWS batch submission...")
        generate_aws_submission(config, output_dir)
        return
    
    # Run experiment
    start_time = time.time()
    
    try:
        results = run_experiment(
            config=config,
            Model6Class=Model6QuantumSynapse,
            Model6Params=Model6Parameters,
            verbose=True
        )
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"\nExperiment completed in {elapsed/60:.1f} minutes")
    
    # Save results
    save_results(results, output_dir / 'network_results.json')
    
    # Plot
    if not args.no_plot:
        try:
            plot_results(
                results, 
                output_path=output_dir / 'network_credit_assignment.png',
                show=False
            )
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")
    
    # Print summary
    print_summary(results)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("="*70)


def generate_aws_submission(config, output_dir: Path):
    """Generate AWS batch submission script"""
    
    script = '''#!/bin/bash
#
# AWS Batch submission for Model 6 Network Experiment
# Generated: {timestamp}
#

# Configuration
INSTANCE_TYPE="c5.4xlarge"  # 16 vCPUs, 32 GB RAM
TIMEOUT_HOURS=48

# Create job array - one job per (condition × delay)
CONDITIONS=({conditions})
DELAYS=({delays})
N_TRIALS={n_trials}

for CONDITION in "${{CONDITIONS[@]}}"; do
    for DELAY in "${{DELAYS[@]}}"; do
        echo "Submitting: $CONDITION, delay=$DELAY"
        
        aws batch submit-job \\
            --job-name "model6-net-${{CONDITION}}-${{DELAY}}" \\
            --job-queue "model6-queue" \\
            --job-definition "model6-network-experiment" \\
            --container-overrides "{{
                \\"command\\": [
                    \\"python\\", \\"run_single_condition.py\\",
                    \\"--condition\\", \\"$CONDITION\\",
                    \\"--delay\\", \\"$DELAY\\",
                    \\"--n-trials\\", \\"$N_TRIALS\\"
                ]
            }}"
    done
done

echo "All jobs submitted. Monitor with: aws batch list-jobs --job-queue model6-queue"
'''.format(
        timestamp=datetime.now().isoformat(),
        conditions=' '.join(config.conditions),
        delays=' '.join(str(d) for d in config.delays),
        n_trials=config.n_trials
    )
    
    script_path = output_dir / 'submit_aws.sh'
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script)
    
    print(f"AWS submission script written to: {script_path}")
    print("\nTo submit:")
    print(f"  chmod +x {script_path}")
    print(f"  {script_path}")


if __name__ == "__main__":
    main()