"""
QETR Agents
============

Reinforcement learning agents using quantum eligibility traces.

Author: Sarah Davidson
"""

from .qetr_agent import (
    QETRAgent,
    TDLambdaAgent,
    BaseRLAgent,
)

__all__ = [
    'QETRAgent',
    'TDLambdaAgent',
    'BaseRLAgent',
]