"""
Quantum Cascade Complete Experimental Framework
=================================================

A comprehensive experimental suite for characterizing the integrated
quantum-classical cascade system.

SYSTEM ARCHITECTURE:
    Q1 (Tryptophan) → Q2 (Dimers) → Classical (CaMKII → Plasticity)

This framework includes:
- 10+ experimental paradigms
- Spatial clustering analysis
- Statistical analysis with multiple trials
- Dose-response curve fitting
- Comprehensive visualization suite
- Cloud deployment support (AWS/GCP)

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
import sys
import os
import json
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time

# Optional imports for enhanced functionality
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy import stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add Model 6 to path
MODEL_PATH = Path(__file__).parent
sys.path.insert(0, str(MODEL_PATH))

# Suppress excessive logging
import logging
for logger_name in ['model6_core', 'calcium_system', 'atp_system', 
                    'ca_triphosphate_complex', 'quantum_coherence', 'pH_dynamics',
                    'dopamine_system', 'em_tryptophan_module', 'em_coupling_module',
                    'local_dimer_tubulin_coupling', 'camkii_module', 
                    'spine_plasticity_module', 'model6_parameters']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Global experiment configuration"""
    
    # Parallelization
    n_workers: int = 4  # Number of parallel workers
    use_multiprocessing: bool = True
    
    # Statistical power
    n_trials: int = 5  # Trials per condition
    confidence_level: float = 0.95
    
    # Timing
    dt: float = 1e-3  # 1 ms timestep
    baseline_duration: float = 0.5
    stim_duration: float = 0.2
    dopamine_duration: float = 0.3
    consolidation_duration: float = 60.0
    
    # Recording
    record_interval: int = 10  # Record every N steps
    
    # Output
    output_dir: str = "results"
    save_raw_data: bool = True
    save_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 150
    
    # Quick mode for testing
    quick_mode: bool = False
    
    def __post_init__(self):
        if self.quick_mode:
            self.n_trials = 2
            self.consolidation_duration = 10.0
            self.record_interval = 50


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SystemState:
    """Complete snapshot of the quantum-classical system"""
    
    # Time
    time: float = 0.0
    phase: str = ""
    
    # === QUANTUM SYSTEM 1: Tryptophan Superradiance ===
    n_tryptophans: int = 0
    em_field_trp: float = 0.0
    collective_field_kT: float = 0.0
    superradiance_active: bool = False
    
    # === QUANTUM SYSTEM 2: Calcium Phosphate Dimers ===
    dimer_count: float = 0.0
    dimer_coherence: float = 0.0
    eligibility: float = 0.0
    network_modulation: float = 0.0
    
    # === COUPLING METRICS ===
    k_agg_enhanced: float = 0.0
    k_enhancement: float = 1.0
    forward_coupling_active: bool = False
    reverse_coupling_active: bool = False
    
    # === PLASTICITY GATE ===
    plasticity_gate: bool = False
    gate_eligibility: bool = False
    gate_dopamine: bool = False
    gate_calcium: bool = False
    committed: bool = False
    committed_level: float = 0.0
    
    # === CLASSICAL BRIDGE: CaMKII ===
    camkii_pT286: float = 0.0
    camkii_active: float = 0.0
    molecular_memory: float = 0.0
    
    # === CLASSICAL OUTPUT: Structural Plasticity ===
    spine_volume: float = 1.0
    AMPAR_count: float = 80.0
    synaptic_strength: float = 1.0
    plasticity_phase: str = "baseline"
    
    # === FEEDBACK ===
    n_templates: int = 3
    template_feedback_active: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentCondition:
    """Defines a single experimental condition"""
    
    name: str
    description: str = ""
    
    # === NETWORK ARCHITECTURE ===
    n_synapses: int = 10
    synapse_spacing_um: float = 1.0  # Microns between synapses
    spatial_pattern: str = "clustered"  # "clustered", "distributed", "linear"
    
    # === QUANTUM SYSTEM 1: Tryptophan ===
    mt_invaded: bool = True
    uv_wavelength_nm: Optional[float] = None  # None = metabolic only
    uv_intensity_mW: float = 0.0
    anesthetic_concentration: float = 0.0  # 0-1 fraction block
    
    # === STIMULATION ===
    stim_voltage_mV: float = -10  # Depolarized
    stim_duration_s: float = 0.2
    stim_pattern: str = "single"  # "single", "burst", "theta"
    
    # === DOPAMINE ===
    dopamine_delay_s: float = 0.0
    dopamine_duration_s: float = 0.3
    dopamine_concentration_nM: float = 500.0  # Peak phasic
    
    # === PHARMACOLOGY ===
    apv_applied: bool = False  # NMDA blocker
    nocodazole_applied: bool = False  # MT disruptor
    
    # === ENVIRONMENT ===
    temperature_C: float = 37.0
    isotope: str = "P31"  # P31 or P32
    
    # === PROTOCOL ===
    consolidation_duration_s: float = 60.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrialResult:
    """Results from a single trial"""
    
    condition: ExperimentCondition
    trial_id: int
    
    # Time traces
    traces: List[SystemState] = field(default_factory=list)
    
    # Key timepoints
    baseline: Optional[SystemState] = None
    post_stim: Optional[SystemState] = None
    pre_dopamine: Optional[SystemState] = None
    post_dopamine: Optional[SystemState] = None
    final: Optional[SystemState] = None
    
    # Summary metrics
    peak_values: Dict = field(default_factory=dict)
    
    # Timing
    runtime_seconds: float = 0.0


@dataclass
class ExperimentResult:
    """Aggregated results across trials"""
    
    experiment_name: str
    conditions: List[ExperimentCondition]
    
    # All trial results
    trials: List[TrialResult] = field(default_factory=list)
    
    # Statistical summaries (computed after trials)
    summary_stats: Dict = field(default_factory=dict)
    
    # Fitted parameters
    fitted_params: Dict = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    config: Optional[ExperimentConfig] = None
    runtime_seconds: float = 0.0


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_summary_statistics(values: List[float], confidence: float = 0.95) -> Dict:
    """Compute mean, std, SEM, and confidence interval"""
    
    n = len(values)
    if n == 0:
        return {'mean': 0, 'std': 0, 'sem': 0, 'ci_low': 0, 'ci_high': 0, 'n': 0}
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 0 else 0
    
    # Confidence interval
    if HAS_SCIPY and n > 1:
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_low = mean - t_crit * sem
        ci_high = mean + t_crit * sem
    else:
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
    
    return {
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n': n
    }


def fit_exponential_decay(x: np.ndarray, y: np.ndarray) -> Dict:
    """Fit y = A * exp(-x/tau) + baseline"""
    
    if not HAS_SCIPY or len(x) < 3:
        return {'A': 0, 'tau': 0, 'baseline': 0, 'r_squared': 0}
    
    def exp_decay(t, A, tau, baseline):
        return A * np.exp(-t / tau) + baseline
    
    try:
        # Initial guesses
        A0 = np.max(y) - np.min(y)
        tau0 = np.mean(x)
        baseline0 = np.min(y)
        
        popt, pcov = curve_fit(exp_decay, x, y, 
                               p0=[A0, tau0, baseline0],
                               bounds=([0, 0.1, -1], [10, 1000, 1]),
                               maxfev=5000)
        
        # R-squared
        y_pred = exp_decay(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'A': float(popt[0]),
            'tau': float(popt[1]),
            'baseline': float(popt[2]),
            'r_squared': float(r_squared)
        }
    except Exception:
        return {'A': 0, 'tau': 0, 'baseline': 0, 'r_squared': 0}


def fit_hill_equation(x: np.ndarray, y: np.ndarray) -> Dict:
    """Fit y = bottom + (top - bottom) / (1 + (EC50/x)^n)"""
    
    if not HAS_SCIPY or len(x) < 4:
        return {'EC50': 0, 'hill_n': 0, 'bottom': 0, 'top': 0, 'r_squared': 0}
    
    def hill(x, bottom, top, EC50, n):
        return bottom + (top - bottom) / (1 + (EC50 / (x + 1e-10)) ** n)
    
    try:
        bottom0 = np.min(y)
        top0 = np.max(y)
        EC50_0 = np.median(x)
        
        popt, pcov = curve_fit(hill, x, y,
                               p0=[bottom0, top0, EC50_0, 1.0],
                               bounds=([0, 0, 0.01, 0.1], [2, 2, 1000, 10]),
                               maxfev=5000)
        
        y_pred = hill(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'bottom': float(popt[0]),
            'top': float(popt[1]),
            'EC50': float(popt[2]),
            'hill_n': float(popt[3]),
            'r_squared': float(r_squared)
        }
    except Exception:
        return {'EC50': 0, 'hill_n': 0, 'bottom': 0, 'top': 0, 'r_squared': 0}


def compute_cooperativity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cooperativity coefficient
    
    Cooperativity > 1 means superlinear (collective effects)
    Cooperativity < 1 means sublinear (saturation)
    Cooperativity = 1 means linear
    """
    if len(x) < 2 or x[0] == 0:
        return 1.0
    
    # Compare actual scaling to linear expectation
    # For linear: y(N) / y(1) = N
    # For cooperative: y(N) / y(1) > N
    
    x = np.array(x)
    y = np.array(y)
    
    # Normalize
    x_norm = x / x[0]
    y_norm = y / (y[0] + 1e-10)
    
    # Fit power law: y = x^c
    # log(y) = c * log(x)
    valid = (x_norm > 0) & (y_norm > 0)
    if np.sum(valid) < 2:
        return 1.0
    
    try:
        slope, _, _, _, _ = stats.linregress(np.log(x_norm[valid]), np.log(y_norm[valid]))
        return float(slope)
    except:
        return 1.0


# =============================================================================
# SPATIAL CLUSTERING IMPLEMENTATION
# =============================================================================

def generate_synapse_positions(n_synapses: int, pattern: str, 
                               spacing_um: float = 1.0) -> np.ndarray:
    """
    Generate synapse positions based on spatial pattern
    
    Args:
        n_synapses: Number of synapses
        pattern: "clustered", "distributed", "linear"
        spacing_um: Base spacing in microns
    
    Returns:
        Array of (x, y, z) positions in microns
    """
    
    if pattern == "clustered":
        # All synapses within a small dendritic segment
        # Gaussian distribution around center
        positions = np.random.randn(n_synapses, 3) * spacing_um * 0.5
        
    elif pattern == "distributed":
        # Synapses spread across multiple branches
        # Uniform in a larger volume
        positions = np.random.uniform(-spacing_um * 5, spacing_um * 5, (n_synapses, 3))
        
    elif pattern == "linear":
        # Synapses along a single dendrite
        positions = np.zeros((n_synapses, 3))
        positions[:, 0] = np.arange(n_synapses) * spacing_um
        # Small jitter
        positions += np.random.randn(n_synapses, 3) * 0.1
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return positions


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between synapses"""
    n = len(positions)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = d
            distances[j, i] = d
    
    return distances


def compute_network_coupling_factor(distances: np.ndarray, 
                                    length_constant_um: float = 2.0) -> float:
    """
    Compute effective network coupling based on spatial arrangement
    
    Coupling decays exponentially with distance
    Clustered synapses have stronger coupling
    """
    n = len(distances)
    if n <= 1:
        return 1.0
    
    # Sum of all pairwise couplings
    total_coupling = 0.0
    n_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            d = distances[i, j]
            # Exponential decay with distance
            coupling = np.exp(-d / length_constant_um)
            total_coupling += coupling
            n_pairs += 1
    
    # Normalize by expected coupling for N independent synapses
    if n_pairs > 0:
        mean_coupling = total_coupling / n_pairs
    else:
        mean_coupling = 1.0
    
    return mean_coupling


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_progress_bar(iterable, desc: str = "", total: int = None):
    """Create progress bar if tqdm available, otherwise return iterable"""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    else:
        print(f"  {desc}...")
        return iterable


def save_results(result: ExperimentResult, output_dir: str):
    """Save experiment results to disk"""
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary as JSON
    summary = {
        'experiment_name': result.experiment_name,
        'timestamp': result.timestamp,
        'runtime_seconds': result.runtime_seconds,
        'n_conditions': len(result.conditions),
        'n_trials': len(result.trials),
        'summary_stats': result.summary_stats,
        'fitted_params': result.fitted_params,
        'conditions': [c.to_dict() for c in result.conditions]
    }
    
    json_path = path / f"{result.experiment_name}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  Saved summary to: {json_path}")
    
    return json_path


def print_section_header(title: str, width: int = 70):
    """Print formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection(title: str, width: int = 70):
    """Print formatted subsection header"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)