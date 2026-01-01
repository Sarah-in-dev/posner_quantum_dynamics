"""
QETR Experiments
=================

Experiments validating quantum eligibility trace learning.

- exp_temporal: Temporal credit assignment + isotope ablation
- exp_coordination: Multi-agent coordination (coming soon)

Author: Sarah Davidson
"""

from .exp_temporal import (
    run_isotope_ablation,
    run_delay_scaling,
    run_td_comparison,
    ExperimentConfig,
)

__all__ = [
    'run_isotope_ablation',
    'run_delay_scaling',
    'run_td_comparison',
    'ExperimentConfig',
]