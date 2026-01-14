"""
QETR Environments
==================

Environments for testing quantum eligibility trace learning.

- delayed_reward: Temporal credit assignment benchmark
- coordination: Multi-agent coordination without backprop

Author: Sarah Davidson
"""

from .delayed_reward import (
    DelayedRewardEnvironment,
    DelayedRewardConfig,
    make_very_short_delay_task,
    make_short_delay_task,
    make_medium_delay_task,
    make_long_delay_task,
    make_extreme_delay_task,
)

from .coordination import (
    CoordinationEnvironment,
    CoordinationConfig,
    make_easy_coordination,
    make_medium_coordination,
    make_hard_coordination,
    make_scaling_test,
)

__all__ = [
    # Delayed reward
    'DelayedRewardEnvironment',
    'DelayedRewardConfig',
    'make_very_short_delay_task',
    'make_short_delay_task',
    'make_medium_delay_task',
    'make_long_delay_task',
    'make_extreme_delay_task',
    
    # Coordination
    'CoordinationEnvironment',
    'CoordinationConfig',
    'make_easy_coordination',
    'make_medium_coordination',
    'make_hard_coordination',
    'make_scaling_test',
]