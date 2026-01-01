"""
Quantum Eligibility Trace Framework
====================================

Physics-derived reinforcement learning based on nuclear spin coherence.

Core components:
- physics: T2 derivation from first principles
- synapse: Single quantum synapse with eligibility trace
- network: Multi-synapse with entanglement for coordination

The key insight: eligibility trace parameters (decay constant, threshold)
are derived from nuclear spin physics, not tuned as hyperparameters.

Author: Sarah Davidson
"""

from .physics import (
    QuantumEligibilityPhysics,
    Isotope,
    NuclearSpinParameters,
    JCouplingParameters,
    DimerStructure,
    compare_isotopes,
    get_T2_ratio,
)

from .synapse import (
    QuantumSynapse,
    SynapseState,
    SynapseParameters,
    SynapsePopulation,
)

from ..synapse.network import (
    QuantumSynapseNetwork,
    EntanglementParameters,
    NetworkState,
)

__all__ = [
    # Physics
    'QuantumEligibilityPhysics',
    'Isotope',
    'NuclearSpinParameters',
    'JCouplingParameters',
    'DimerStructure',
    'compare_isotopes',
    'get_T2_ratio',
    
    # Synapse
    'QuantumSynapse',
    'SynapseState',
    'SynapseParameters',
    'SynapsePopulation',
    
    # Network
    'QuantumSynapseNetwork',
    'EntanglementParameters',
    'NetworkState',
]

__version__ = '0.1.0'