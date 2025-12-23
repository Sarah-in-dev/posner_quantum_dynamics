"""
Tier 1: Single-Synapse Core Mechanism Tests
============================================

Fast experiments (seconds to run) that demonstrate fundamental physics.

Experiments:
1. isotope_comparison - P31 vs P32 network persistence (THE KILLER EXPERIMENT)
2. entanglement_network - EM-mediated bond formation through quantum bus
3. theta_burst - Multi-spike calcium/dimer integration
4. coherence_decay - Singlet probability validation (Agarwal physics)

Usage:
    python run_tier1.py              # Run all
    python run_tier1.py --quick      # Quick validation
    python -m tier1_single_synapse.exp_isotope_comparison  # Single experiment
"""

from .exp_isotope_comparison import run as run_isotope, plot as plot_isotope
from .exp_entanglement_network import run as run_entanglement, plot as plot_entanglement
from .exp_theta_burst import run as run_theta_burst, plot as plot_theta_burst
from .exp_coherence_decay import run as run_coherence_decay, plot as plot_coherence_decay

__all__ = [
    'run_isotope', 'plot_isotope',
    'run_entanglement', 'plot_entanglement', 
    'run_theta_burst', 'plot_theta_burst',
    'run_coherence_decay', 'plot_coherence_decay'
]