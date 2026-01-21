"""
Quantum RNN Module
==================

Recurrent Neural Network with Model 6 Quantum Synapses.

The key innovation: each synapse is a full Model6QuantumSynapse,
so the eligibility trace timescale EMERGES from quantum physics
rather than being a hyperparameter.

Components:
-----------
- RateNeuron: Simple leaky integrator neuron
- QuantumRNN: RNN where each synapse is Model6QuantumSynapse
- ClassicalRNN: Baseline with exponential eligibility decay
- CreditAssignmentExperiment: The key demonstration

Usage:
------
    from model6_core import Model6QuantumSynapse
    from model6_parameters import Model6Parameters
    from rnn import QuantumRNN, RNNConfig, CreditAssignmentExperiment
    
    # Create network
    config = RNNConfig(n_neurons=10, isotope='P31')
    rnn = QuantumRNN(config, Model6QuantumSynapse, Model6Parameters)
    
    # Run credit assignment experiment
    exp = CreditAssignmentExperiment(...)
    results = exp.run()

Author: Sarah Davidson
University of Florida
"""

from .neuron import RateNeuron, NeuronConfig
from .quantum_rnn import (
    QuantumRNN,
    ClassicalRNN,
    RNNConfig,
    NetworkState,
    create_quantum_rnn
)
from .credit_assignment_experiment import (
    CreditAssignmentExperiment,
    ExperimentConfig,
    TrialResult,
    ConditionResult
)

__all__ = [
    # Neuron
    'RateNeuron',
    'NeuronConfig',
    
    # RNN
    'QuantumRNN',
    'ClassicalRNN', 
    'RNNConfig',
    'NetworkState',
    'create_quantum_rnn',
    
    # Experiment
    'CreditAssignmentExperiment',
    'ExperimentConfig',
    'TrialResult',
    'ConditionResult',
]