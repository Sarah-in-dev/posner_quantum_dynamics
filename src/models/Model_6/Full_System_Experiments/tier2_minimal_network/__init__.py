"""
Tier 2: Minimal Multi-Synapse Tests
====================================

Medium-speed experiments (minutes to run) that test network-level effects.

Experiments:
1. network_threshold - How synapse count affects entanglement and commitment
2. three_factor_gate - Verify all factors (eligibility, dopamine, calcium) required

Usage:
    python run_tier2.py              # Run all
    python run_tier2.py --quick      # Quick validation
    python -m tier2_minimal_network.exp_network_threshold  # Single experiment
"""

from .exp_network_threshold import run as run_network_threshold, plot as plot_network_threshold
from .exp_three_factor_gate import run as run_three_factor_gate, plot as plot_three_factor_gate

__all__ = [
    'run_network_threshold', 'plot_network_threshold',
    'run_three_factor_gate', 'plot_three_factor_gate'
]