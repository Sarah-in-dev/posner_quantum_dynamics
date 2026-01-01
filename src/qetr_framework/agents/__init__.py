"""
QETR Agents
============

Reinforcement learning agents using quantum eligibility traces.

- QETRAgent: Main agent with physics-derived eligibility
- QETRNetworkAgent: Multi-agent with entanglement
- TDLambdaAgent: Classical baseline
- IndependentQLearner: Coordination baseline

Author: Sarah Davidson
"""

from .qetr_agent import (
    QETRAgent,
    QETRConfig,
    QETRNetworkAgent,
    TDLambdaAgent,
    IndependentQLearner,
    BaseAgent,
)

__all__ = [
    'QETRAgent',
    'QETRConfig',
    'QETRNetworkAgent',
    'TDLambdaAgent',
    'IndependentQLearner',
    'BaseAgent',
]