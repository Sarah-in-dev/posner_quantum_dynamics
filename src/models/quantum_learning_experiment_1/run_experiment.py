"""
Main Experiment: Quantum vs Classical Rapid Adaptation

Tests whether quantum-coherent processing provides faster adaptation
when task context changes unexpectedly.

Prediction from thesis: 2.4× speedup with quantum processing

Author: Sarah Davidson
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / 'models'))

from quantum_layer import QuantumInspiredNetwork
from classical_network import ClassicalNetwork
from task import ContextSwitchTask


def run_experiment(model, task, n_trials: int = 300, learning_rate: float = 0.001,
                   use_quantum: bool = False, verbose: bool = True) -> Dict:
    """
    Run complete learning experiment
    
    Args:
        model: Neural network (quantum or classical)
        task: Task instance
        n_trials: Total number of trials
        learning_rate: Adam learning rate
        use_quantum: Whether using quantum model
        verbose: Print progress
        
    Returns:
        results: Dictionary with metrics for analysis
    """
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking
    results = {
        'trial': [],
        'accuracy': [],
        'loss': [],
        'reward': [],
        'context': [],
        'predictions': [],
        'targets': []
    }
    
    # Quantum-specific tracking
    if use_quantum:
        results['coherence_time'] = []
        results['n_active_paths'] = []
        results['dopamine_level'] = []
        results['mode'] = []
    
    # Running statistics
    recent_rewards = []
    window_size = 10
    
    # Reset task
    task.reset()
    
    if verbose:
        print(f"\nRunning {'QUANTUM' if use_quantum else 'CLASSICAL'} experiment...")
        print(f"Total trials: {n_trials}")
        print("-" * 60)
    
    for trial in range(n_trials):
        # Generate trial
        x, target, switched = task.generate_trial()
        
        if switched and verbose:
            print(f"\n*** CONTEXT SWITCH at trial {trial} → context {task.get_context()} ***\n")
        
        # Compute dopamine signal from recent performance
        if use_quantum and len(recent_rewards) > 0:
            # High recent performance → high dopamine → dimer mode
            dopamine = np.mean(recent_rewards[-20:])
        else:
            dopamine = None
        
        # Forward pass
        if use_quantum:
            output, diagnostics = model(x, reward_signal=dopamine)
        else:
            output = model(x)
            diagnostics = None
        
        # Reshape output for loss
        target_tensor = torch.tensor([target], dtype=torch.long)
        
        # Compute loss
        loss = criterion(output, target_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        prediction = torch.argmax(output, dim=1).item()
        correct = (prediction == target)
        reward = task.compute_reward(prediction, target)
        
        # Update recent rewards
        recent_rewards.append(reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
        
        # Record metrics
        accuracy = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
        
        results['trial'].append(trial)
        results['accuracy'].append(accuracy)
        results['loss'].append(loss.item())
        results['reward'].append(reward)
        results['context'].append(task.get_context())
        results['predictions'].append(prediction)
        results['targets'].append(target)
        
        # Quantum-specific metrics
        if use_quantum and diagnostics:
            results['coherence_time'].append(diagnostics[0]['coherence_time'])
            results['n_active_paths'].append(diagnostics[0]['n_active_paths'])
            results['dopamine_level'].append(diagnostics[0]['dopamine_level'])
            results['mode'].append(diagnostics[0]['mode'])
        
        # Progress printing
        if verbose and (trial + 1) % 50 == 0:
            print(f"Trial {trial+1:3d}: Accuracy={accuracy:.3f}, Loss={loss.item():.3f}")
    
    if verbose:
        print("-" * 60)
        print(f"Experiment complete: Final accuracy = {results['accuracy'][-1]:.3f}\n")
    
    return results


def analyze_adaptation_speed(results_quantum: Dict, results_classical: Dict,
                             switch_points: List[int] = [50, 100, 150, 200, 250]) -> Dict:
    """
    Analyze how quickly each model adapts after context switches
    
    Key metric: trials to recover 80% accuracy after switch
    
    Args:
        results_quantum: Results from quantum model
        results_classical: Results from classical model
        switch_points: Trial numbers where context switched
        
    Returns:
        analysis: Dictionary with adaptation metrics
    """
    analysis = {
        'quantum_adaptation_times': [],
        'classical_adaptation_times': [],
        'quantum_post_switch_curves': [],
        'classical_post_switch_curves': [],
        'switch_points': switch_points
    }
    
    threshold = 0.8  # 80% accuracy threshold
    window = 30  # Look 30 trials after each switch
    
    for switch_trial in switch_points:
        if switch_trial >= len(results_quantum['accuracy']):
            continue
        
        # Extract post-switch accuracy curves
        quantum_acc = results_quantum['accuracy'][switch_trial:switch_trial+window]
        classical_acc = results_classical['accuracy'][switch_trial:switch_trial+window]
        
        # Find first trial reaching threshold
        quantum_adapt = next((i for i, acc in enumerate(quantum_acc) if acc >= threshold), None)
        classical_adapt = next((i for i, acc in enumerate(classical_acc) if acc >= threshold), None)
        
        # Record
        if quantum_adapt is not None:
            analysis['quantum_adaptation_times'].append(quantum_adapt)
            analysis['quantum_post_switch_curves'].append(quantum_acc)
        
        if classical_adapt is not None:
            analysis['classical_adaptation_times'].append(classical_adapt)
            analysis['classical_post_switch_curves'].append(classical_acc)
    
    # Compute speedup factor
    if analysis['quantum_adaptation_times'] and analysis['classical_adaptation_times']:
        quantum_mean = np.mean(analysis['quantum_adaptation_times'])
        classical_mean = np.mean(analysis['classical_adaptation_times'])
        analysis['speedup_factor'] = classical_mean / quantum_mean
        analysis['quantum_mean'] = quantum_mean
        analysis['classical_mean'] = classical_mean
        analysis['quantum_std'] = np.std(analysis['quantum_adaptation_times'])
        analysis['classical_std'] = np.std(analysis['classical_adaptation_times'])
    else:
        analysis['speedup_factor'] = None
    
    return analysis


def print_results(analysis: Dict):
    """Print formatted results"""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS")
    print("="*70)
    
    if analysis['speedup_factor'] is None:
        print("\n⚠ Insufficient data for analysis")
        return
    
    print(f"\nAdaptation Speed After Context Switch:")
    print(f"  Classical: {analysis['classical_mean']:.1f} ± {analysis['classical_std']:.1f} trials")
    print(f"  Quantum:   {analysis['quantum_mean']:.1f} ± {analysis['quantum_std']:.1f} trials")
    
    print(f"\nSpeedup Factor: {analysis['speedup_factor']:.2f}×")
    print(f"Prediction (from thesis): 2.4×")
    
    if analysis['speedup_factor'] >= 2.0:
        print("\n✓✓✓ PREDICTION CONFIRMED ✓✓✓")
    elif analysis['speedup_factor'] >= 1.5:
        print("\n✓ PARTIAL CONFIRMATION (1.5-2.0×)")
    else:
        print("\n⚠ Prediction not met (<1.5×)")
    
    print("\n" + "="*70)


def main():
    """Run complete experiment"""
    print("="*70)
    print("QUANTUM-COHERENT VS CLASSICAL LEARNING EXPERIMENT")
    print("="*70)
    print("\nTesting prediction: Quantum processing enables 2.4× faster")
    print("adaptation when task context changes")
    print("\nBased on:")
    print("  • T2 coherence time: 100s (dimers)")
    print("  • J-coupling: 20 Hz (ATP-driven)")
    print("  • Dopamine modulation: exploration ↔ exploitation")
    print("="*70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    input_dim = 20
    hidden_dim = 64
    output_dim = 2
    n_trials = 300
    learning_rate = 0.001
    
    # Create tasks (separate instances for each model)
    task_classical = ContextSwitchTask(input_dim=input_dim, n_contexts=3, switch_every=50)
    task_quantum = ContextSwitchTask(input_dim=input_dim, n_contexts=3, switch_every=50)
    
    # Create models
    print("\n1. Creating models...")
    classical_model = ClassicalNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=2
    )
    
    quantum_model = QuantumInspiredNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_quantum_layers=2,
        coherence_steps=100
    )
    
    print(f"   Classical network: {sum(p.numel() for p in classical_model.parameters())} parameters")
    print(f"   Quantum network: {sum(p.numel() for p in quantum_model.parameters())} parameters")
    
    # Run experiments
    print("\n2. Running experiments...")
    
    results_classical = run_experiment(
        classical_model, task_classical,
        n_trials=n_trials,
        learning_rate=learning_rate,
        use_quantum=False,
        verbose=True
    )
    
    results_quantum = run_experiment(
        quantum_model, task_quantum,
        n_trials=n_trials,
        learning_rate=learning_rate,
        use_quantum=True,
        verbose=True
    )
    
    # Analyze
    print("3. Analyzing adaptation speed...")
    analysis = analyze_adaptation_speed(results_quantum, results_classical)
    
    # Print results
    print_results(analysis)
    
    # Save results
    print("\n4. Saving results...")
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    np.savez(
        results_dir / 'experiment_data.npz',
        results_classical=results_classical,
        results_quantum=results_quantum,
        analysis=analysis
    )
    print(f"   Saved to: {results_dir / 'experiment_data.npz'}")
    
    print("\n✓ Experiment complete!")
    print("\nNext steps:")
    print("  • Run visualization script to generate plots")
    print("  • Analyze quantum coherence dynamics")
    print("  • Test different hyperparameters")
    
    return results_quantum, results_classical, analysis


if __name__ == "__main__":
    results_q, results_c, analysis = main()