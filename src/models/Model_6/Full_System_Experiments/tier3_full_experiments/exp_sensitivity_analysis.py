#!/usr/bin/env python3
"""
Experiment: Parameter Sensitivity Analysis
==========================================

Tests robustness of key predictions to parameter uncertainty by running
actual Model 6 simulations with varied parameters.

THIS VERSION USES THE ACTUAL MODEL 6 SYSTEM - not analytical approximations.

SCIENTIFIC BASIS:
All model parameters have literature uncertainty ranges.
Key predictions should be ROBUST to reasonable parameter variation:
1. Isotope discrimination ratio > 100× (P31 vs P32)
2. Coherence time > 60s (behavioral learning window)
3. Dimer count 4-6 per synapse (Fisher prediction)

PARAMETERS TESTED:
1. quantum.T_singlet_dimer - Dimer singlet lifetime (±50%)
2. quantum.J_protection_strength - J-coupling protection (±50%)
3. calcium.ca_spine_peak - Peak calcium concentration (±40%)
4. coupling.k_agg_baseline - Dimer aggregation rate (±50%)

PROTOCOL:
For each parameter value:
1. Run P31 isotope simulation 
2. Run P32 isotope simulation
3. Extract fitted T2 and isotope ratio
4. Assess robustness

Author: Sarah Davidson
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy
import sys
import time
from scipy.optimize import curve_fit

# Add parent to path for Model 6 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# REQUIRE Model 6 imports
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a parameter to vary"""
    name: str              # Path in params (e.g., 'quantum.T_singlet_dimer')
    display_name: str      # For plots
    baseline: float        # Default value
    unit: str              # Unit string
    variation_pct: float   # ±% to vary
    citation: str = ""     # Literature source
    
    @property
    def low(self) -> float:
        return self.baseline * (1 - self.variation_pct / 100)
    
    @property
    def high(self) -> float:
        return self.baseline * (1 + self.variation_pct / 100)


@dataclass
class SingleSimResult:
    """Result from a single simulation"""
    parameter_value: float
    isotope: str
    
    # Key measurements
    fitted_tau: float = 0.0
    eligibility_at_60s: float = 0.0
    peak_dimers: int = 0
    fit_r2: float = 0.0
    
    runtime_s: float = 0.0


@dataclass
class ParameterSweepResult:
    """Results from sweeping one parameter"""
    parameter: ParameterSpec
    values: np.ndarray
    
    # Results at each value
    tau_p31: np.ndarray = None
    tau_p32: np.ndarray = None
    isotope_ratio: np.ndarray = None
    eligibility_60s: np.ndarray = None
    dimer_counts: np.ndarray = None
    
    # Robustness
    tau_always_above_60s: bool = False
    ratio_always_above_100: bool = False


@dataclass
class SensitivityResult:
    """Complete sensitivity analysis"""
    parameters: List[ParameterSpec] = field(default_factory=list)
    sweeps: List[ParameterSweepResult] = field(default_factory=list)
    
    timestamp: str = ""
    runtime_s: float = 0.0
    
    robust_count: int = 0
    sensitive_count: int = 0


# =============================================================================
# PARAMETER ACCESS
# =============================================================================

def get_parameter_specs(params: Model6Parameters) -> List[ParameterSpec]:
    """
    Define parameters to test with their baseline values from params
    """
    specs = []
    
    # 1. Singlet lifetime
    if hasattr(params, 'quantum') and hasattr(params.quantum, 'T_singlet_dimer'):
        specs.append(ParameterSpec(
            name='quantum.T_singlet_dimer',
            display_name='Singlet lifetime',
            baseline=params.quantum.T_singlet_dimer,
            unit='s',
            variation_pct=50,
            citation='Agarwal 2023'
        ))
    
    # 2. J-coupling protection
    if hasattr(params, 'quantum') and hasattr(params.quantum, 'J_protection_strength'):
        specs.append(ParameterSpec(
            name='quantum.J_protection_strength',
            display_name='J-coupling protection',
            baseline=params.quantum.J_protection_strength,
            unit='×',
            variation_pct=50,
            citation='Fisher 2015'
        ))
    
    # 3. Calcium peak (need to find actual attribute name)
    if hasattr(params, 'calcium'):
        ca_attr = None
        for attr in ['ca_spine_peak', 'peak_ca_uM', 'ca_peak']:
            if hasattr(params.calcium, attr):
                ca_attr = attr
                break
        
        if ca_attr:
            specs.append(ParameterSpec(
                name=f'calcium.{ca_attr}',
                display_name='Peak [Ca²⁺]',
                baseline=getattr(params.calcium, ca_attr),
                unit='μM',
                variation_pct=40,
                citation='Sabatini 2000'
            ))
    
    # 4. Aggregation rate
    if hasattr(params, 'coupling') and hasattr(params.coupling, 'k_agg_baseline'):
        specs.append(ParameterSpec(
            name='coupling.k_agg_baseline',
            display_name='Aggregation rate',
            baseline=params.coupling.k_agg_baseline,
            unit='s⁻¹',
            variation_pct=50,
            citation='McDonogh 2024'
        ))
    
    return specs


def set_parameter(params: Any, path: str, value: float) -> None:
    """Set a nested parameter using dot notation"""
    parts = path.split('.')
    obj = params
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_parameter(params: Any, path: str) -> float:
    """Get a nested parameter using dot notation"""
    parts = path.split('.')
    obj = params
    for part in parts:
        obj = getattr(obj, part)
    return obj


# =============================================================================
# SIMULATION
# =============================================================================

def exponential_decay(t, A, tau):
    """Exponential: A * exp(-t/tau)"""
    return A * np.exp(-t / tau)


def fit_decay(times: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """Fit exponential decay, return (tau, r2)"""
    mask = values > 0.01
    if np.sum(mask) < 3:
        return 50.0, 0.0
    
    try:
        popt, _ = curve_fit(exponential_decay, times[mask], values[mask],
                           p0=[values[mask][0], 50.0],
                           bounds=([0.01, 0.5], [2.0, 500]),
                           maxfev=2000)
        A, tau = popt
        
        pred = exponential_decay(times[mask], A, tau)
        ss_res = np.sum((values[mask] - pred)**2)
        ss_tot = np.sum((values[mask] - np.mean(values[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return tau, r2
    except:
        return 50.0, 0.0


def run_single_condition(params: Model6Parameters,
                         isotope: str,
                         n_synapses: int = 5,
                         decay_duration_s: float = 60.0,
                         verbose: bool = False) -> SingleSimResult:
    """
    Run a single Model 6 simulation for sensitivity analysis
    """
    start_time = time.time()
    
    # Set isotope
    p = deepcopy(params)
    if isotope == 'P31':
        p.environment.fraction_P31 = 1.0
        p.environment.fraction_P32 = 0.0
    else:
        p.environment.fraction_P31 = 0.0
        p.environment.fraction_P32 = 1.0
    
    # Create network
    network = MultiSynapseNetwork(
        n_synapses=n_synapses,
        params=p,
        pattern='clustered',
        spacing_um=1.0
    )
    network.initialize(Model6QuantumSynapse, p)
    network.set_microtubule_invasion(True)
    
    dt = 0.001
    
    # Theta-burst stimulation
    for burst in range(5):
        for spike in range(4):
            for _ in range(2):
                network.step(dt, {'voltage': -10e-3, 'reward': False})
            for _ in range(8):
                network.step(dt, {'voltage': -70e-3, 'reward': False})
        for _ in range(160):
            network.step(dt, {'voltage': -70e-3, 'reward': False})
    
    # Get peak dimers
    peak_dimers = sum(len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                      for s in network.synapses)
    
    # Decay phase - record eligibility
    dt_decay = 0.1
    n_steps = int(decay_duration_s / dt_decay)
    
    times = []
    eligibilities = []
    
    for step in range(n_steps):
        state = network.step(dt_decay, {'voltage': -70e-3, 'reward': False})
        times.append(step * dt_decay)
        eligibilities.append(state.mean_eligibility)
    
    times = np.array(times)
    eligibilities = np.array(eligibilities)
    
    # Fit decay
    tau, r2 = fit_decay(times, eligibilities)
    
    # Get eligibility at 60s
    idx_60 = min(int(60 / dt_decay), len(eligibilities) - 1)
    elig_60s = eligibilities[idx_60]
    
    return SingleSimResult(
        parameter_value=0,  # Set by caller
        isotope=isotope,
        fitted_tau=tau,
        eligibility_at_60s=elig_60s,
        peak_dimers=peak_dimers,
        fit_r2=r2,
        runtime_s=time.time() - start_time
    )


def run_parameter_sweep(spec: ParameterSpec,
                        base_params: Model6Parameters,
                        n_values: int = 5,
                        n_synapses: int = 5,
                        decay_duration_s: float = 60.0,
                        verbose: bool = True) -> ParameterSweepResult:
    """
    Sweep one parameter across its range
    """
    if verbose:
        print(f"\n  Sweeping: {spec.display_name}")
        print(f"    Range: {spec.low:.3g} - {spec.high:.3g} {spec.unit}")
    
    values = np.linspace(spec.low, spec.high, n_values)
    
    tau_p31 = []
    tau_p32 = []
    elig_60s = []
    dimer_counts = []
    
    for i, val in enumerate(values):
        if verbose:
            print(f"    [{i+1}/{n_values}] {spec.display_name} = {val:.3g} {spec.unit}", end=" ")
        
        # Create params with this value
        params = deepcopy(base_params)
        set_parameter(params, spec.name, val)
        
        # Run P31
        result_p31 = run_single_condition(
            params, 'P31', n_synapses, decay_duration_s, verbose=False
        )
        
        # Run P32 (shorter - it decays fast)
        result_p32 = run_single_condition(
            params, 'P32', n_synapses, min(10.0, decay_duration_s), verbose=False
        )
        
        tau_p31.append(result_p31.fitted_tau)
        tau_p32.append(result_p32.fitted_tau)
        elig_60s.append(result_p31.eligibility_at_60s)
        dimer_counts.append(result_p31.peak_dimers)
        
        if verbose:
            print(f"→ τ_P31={result_p31.fitted_tau:.1f}s, τ_P32={result_p32.fitted_tau:.1f}s")
    
    tau_p31 = np.array(tau_p31)
    tau_p32 = np.array(tau_p32)
    
    # Calculate isotope ratio
    isotope_ratio = np.where(tau_p32 > 0.1, tau_p31 / tau_p32, 1000)
    
    return ParameterSweepResult(
        parameter=spec,
        values=values,
        tau_p31=tau_p31,
        tau_p32=tau_p32,
        isotope_ratio=isotope_ratio,
        eligibility_60s=np.array(elig_60s),
        dimer_counts=np.array(dimer_counts),
        tau_always_above_60s=bool(np.all(tau_p31 > 60)),
        ratio_always_above_100=bool(np.all(isotope_ratio > 100))
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_sensitivity_figures(result: SensitivityResult, output_path: Path) -> None:
    """Create sensitivity analysis figures"""
    
    sweeps = result.sweeps
    n_params = len(sweeps)
    
    if n_params == 0:
        print("  No parameters to plot")
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    # Figure: 4-panel summary
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Panel A: Coherence time vs parameter variation
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, sweep in enumerate(sweeps):
        x_norm = (sweep.values - sweep.parameter.baseline) / sweep.parameter.baseline * 100
        ax1.plot(x_norm, sweep.tau_p31, '-o', color=colors[i], linewidth=1.5,
                 markersize=4, label=sweep.parameter.display_name[:15])
    
    ax1.axhline(60, color='red', linestyle='--', linewidth=2, label='60s threshold')
    ax1.axhspan(60, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 60 else 300, alpha=0.1, color='green')
    ax1.set_xlabel('Parameter variation (%)')
    ax1.set_ylabel('Fitted τ for ³¹P (s)')
    ax1.set_title('A. Coherence Time Robustness')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Isotope ratio
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, sweep in enumerate(sweeps):
        x_norm = (sweep.values - sweep.parameter.baseline) / sweep.parameter.baseline * 100
        ax2.semilogy(x_norm, sweep.isotope_ratio, '-o', color=colors[i], linewidth=1.5,
                     markersize=4)
    
    ax2.axhline(100, color='red', linestyle='--', linewidth=2, label='100× threshold')
    ax2.set_xlabel('Parameter variation (%)')
    ax2.set_ylabel('Isotope ratio τ_P31/τ_P32')
    ax2.set_title('B. Isotope Discrimination')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Eligibility at 60s
    ax3 = fig.add_subplot(gs[1, 0])
    
    for i, sweep in enumerate(sweeps):
        x_norm = (sweep.values - sweep.parameter.baseline) / sweep.parameter.baseline * 100
        ax3.plot(x_norm, sweep.eligibility_60s, '-o', color=colors[i], linewidth=1.5,
                 markersize=4, label=sweep.parameter.display_name[:15])
    
    ax3.axhline(0.3, color='green', linestyle='--', linewidth=2, label='30% threshold')
    ax3.set_xlabel('Parameter variation (%)')
    ax3.set_ylabel('Eligibility at 60s')
    ax3.set_title('C. Learning Window Eligibility')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = []
    for sweep in sweeps:
        pct_tau_ok = np.mean(sweep.tau_p31 > 60) * 100
        pct_ratio_ok = np.mean(sweep.isotope_ratio > 100) * 100
        min_ratio = np.min(sweep.isotope_ratio)
        
        table_data.append([
            sweep.parameter.display_name[:18],
            f'{pct_tau_ok:.0f}%',
            f'{pct_ratio_ok:.0f}%',
            f'{min_ratio:.0f}×'
        ])
    
    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=['Parameter', 'τ>60s', 'Ratio>100×', 'Min Ratio'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color cells
        for i in range(len(table_data)):
            cell_tau = table[(i+1, 1)]
            if float(table_data[i][1].replace('%', '')) == 100:
                cell_tau.set_facecolor('#d4edda')
            else:
                cell_tau.set_facecolor('#f8d7da')
            
            cell_ratio = table[(i+1, 2)]
            if float(table_data[i][2].replace('%', '')) == 100:
                cell_ratio.set_facecolor('#d4edda')
            else:
                cell_ratio.set_facecolor('#f8d7da')
    
    ax4.set_title('D. Robustness Summary', pad=20)
    
    plt.suptitle('Parameter Sensitivity Analysis (Model 6)\n'
                 'Testing robustness across ±50% parameter variation',
                 fontsize=12, y=0.98)
    
    plt.savefig(output_path / 'sensitivity_summary.png')
    plt.savefig(output_path / 'sensitivity_summary.pdf')
    plt.close()
    
    print(f"  Saved: sensitivity_summary.png")


def save_results(result: SensitivityResult, output_path: Path) -> Dict:
    """Save results to JSON"""
    summary = {
        'experiment': 'Parameter Sensitivity Analysis (Model 6)',
        'timestamp': result.timestamp,
        'runtime_s': result.runtime_s,
        'n_parameters': len(result.parameters),
        'robust_count': result.robust_count,
        'sensitive_count': result.sensitive_count,
        'parameters': []
    }
    
    for sweep in result.sweeps:
        param_data = {
            'name': sweep.parameter.name,
            'display_name': sweep.parameter.display_name,
            'baseline': float(sweep.parameter.baseline),
            'unit': sweep.parameter.unit,
            'results': {
                'tau_p31_mean': float(np.mean(sweep.tau_p31)),
                'tau_p31_min': float(np.min(sweep.tau_p31)),
                'tau_p31_max': float(np.max(sweep.tau_p31)),
                'isotope_ratio_min': float(np.min(sweep.isotope_ratio)),
            },
            'robust': {
                'tau_above_60s': sweep.tau_always_above_60s,
                'ratio_above_100': sweep.ratio_always_above_100,
            }
        }
        summary['parameters'].append(param_data)
    
    with open(output_path / 'sensitivity_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def run_sensitivity_analysis(output_dir: str = None,
                             n_values: int = 5,
                             n_synapses: int = 5,
                             decay_duration_s: float = 60.0,
                             quick: bool = False) -> SensitivityResult:
    """
    Run complete parameter sensitivity analysis using Model 6
    """
    print("=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("Using Model 6 - Actual simulations")
    print("=" * 70)
    
    start_time = time.time()
    
    # Setup output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"sensitivity_{timestamp}")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path}")
    
    # Quick mode
    if quick:
        n_values = 3
        n_synapses = 3
        decay_duration_s = 30.0
        print("Quick mode: reduced values, synapses, duration")
    
    # Get baseline parameters
    base_params = Model6Parameters()
    base_params.em_coupling_enabled = True
    base_params.multi_synapse_enabled = True
    
    # Get parameters to test
    specs = get_parameter_specs(base_params)
    
    if not specs:
        print("WARNING: No parameters found to test!")
        print("  Check that Model6Parameters has expected attributes")
        return SensitivityResult()
    
    print(f"\nParameters to test: {len(specs)}")
    for spec in specs:
        print(f"  - {spec.display_name}: {spec.baseline:.3g} {spec.unit} (±{spec.variation_pct}%)")
    
    # Run sweeps
    sweeps = []
    for spec in specs:
        sweep = run_parameter_sweep(
            spec, base_params,
            n_values=n_values,
            n_synapses=n_synapses,
            decay_duration_s=decay_duration_s,
            verbose=True
        )
        sweeps.append(sweep)
    
    # Create result
    robust_count = sum(1 for s in sweeps if s.tau_always_above_60s and s.ratio_always_above_100)
    
    result = SensitivityResult(
        parameters=specs,
        sweeps=sweeps,
        timestamp=datetime.now().isoformat(),
        runtime_s=time.time() - start_time,
        robust_count=robust_count,
        sensitive_count=len(sweeps) - robust_count
    )
    
    # Generate outputs
    print("\nGenerating figures...")
    create_sensitivity_figures(result, output_path)
    save_results(result, output_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nParameters tested: {len(specs)}")
    print(f"Robust: {robust_count}/{len(specs)}")
    
    for sweep in sweeps:
        status = "✓" if sweep.tau_always_above_60s and sweep.ratio_always_above_100 else "⚠"
        print(f"  {status} {sweep.parameter.display_name}: "
              f"τ={np.mean(sweep.tau_p31):.1f}s "
              f"(range: {np.min(sweep.tau_p31):.1f}-{np.max(sweep.tau_p31):.1f}s)")
    
    print(f"\nTotal runtime: {result.runtime_s:.1f}s")
    print("=" * 70)
    
    return result


class SensitivityExperiment:
    """Wrapper for run_tier3.py integration"""
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.result = None
    
    def run(self, output_dir: Path = None) -> SensitivityResult:
        """Run the experiment"""
        self.result = run_sensitivity_analysis(
            output_dir=str(output_dir) if output_dir else None,
            quick=self.quick_mode
        )
        return self.result
    
    def print_summary(self, result: SensitivityResult) -> None:
        """Print summary"""
        print(f"\nRobust: {result.robust_count}/{len(result.parameters)} parameters")
    
    def plot(self, result: SensitivityResult, output_dir: Path = None) -> None:
        """Plots saved during run"""
        pass
    
    def save_results(self, result: SensitivityResult, path: Path) -> None:
        """Save summary"""
        summary = {
            'robust_count': result.robust_count,
            'sensitive_count': result.sensitive_count,
            'runtime_s': result.runtime_s
        }
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis (Model 6)')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--values', '-n', type=int, default=5)
    parser.add_argument('--synapses', '-s', type=int, default=5)
    parser.add_argument('--duration', '-d', type=float, default=60.0)
    parser.add_argument('--quick', action='store_true')
    
    args = parser.parse_args()
    
    run_sensitivity_analysis(
        output_dir=args.output,
        n_values=args.values,
        n_synapses=args.synapses,
        decay_duration_s=args.duration,
        quick=args.quick
    )