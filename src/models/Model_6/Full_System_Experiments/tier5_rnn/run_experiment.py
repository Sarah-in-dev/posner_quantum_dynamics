#!/usr/bin/env python3
"""
run_experiment.py - Actually runs the credit assignment experiment
"""

import sys
from pathlib import Path

# Get the Model_6 directory (two levels up from tier5_rnn)
MODEL_6_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(MODEL_6_DIR))

# Now these imports should work
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from run_credit_assignment import CreditAssignmentExperiment, ExperimentConfig

def main():
    print("=" * 70)
    print("RUNNING CREDIT ASSIGNMENT EXPERIMENT")
    print("=" * 70)
    
    # Start with a minimal config for debugging
    config = ExperimentConfig(
        n_neurons=5,
        delays=[1, 5, 10],
        n_trials=2
    )
    
    print(f"\nConfig:")
    print(f"  n_neurons: {config.n_neurons}")
    print(f"  delays: {config.delays}")
    print(f"  n_trials: {config.n_trials}")
    
    print("\nCreating experiment...")
    exp = CreditAssignmentExperiment(
        config=config,
        SynapseClass=Model6QuantumSynapse,
        SynapseParams=Model6Parameters
    )
    
    print("\nRunning...")
    results = exp.run()
    
    exp.print_summary(results)
    exp.plot_results(results)

if __name__ == "__main__":
    main()