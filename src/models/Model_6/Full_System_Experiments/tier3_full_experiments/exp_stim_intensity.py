#!/usr/bin/env python3
"""
Experiment: Stimulation Intensity (Input-Output Curves)
========================================================

Tests how stimulation strength affects cascade output.

Scientific basis:
- Synaptic plasticity exhibits sigmoid IO curves
- Threshold behavior emerges from cooperative mechanisms
- Hill coefficient indicates cooperativity

Predictions:
- Below threshold: No commitment (insufficient dimers/coherence)
- Sharp transition at threshold voltage
- Above threshold: Saturating response
- Hill coefficient > 1 indicates cooperativity (quantum effects)

Protocol:
1. Vary depolarization voltage during theta-burst
2. Keep timing/pattern identical
3. Measure dimer formation, coherence, commitment

Success criteria:
- Clear sigmoid IO curve
- Threshold near -30 to -20 mV (moderate depolarization)
- Hill coefficient > 1 (cooperative behavior)
- Validates cascade as threshold detector

References:
- Bhalla & Bhalla (2009): Synaptic plasticity models
- Shouval et al. (2002): Unified model of LTP/LTD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


@dataclass
class StimCondition:
    """Single experimental condition"""
    voltage_mV: float  # Peak depolarization voltage
    n_synapses: int = 10
    
    @property
    def voltage_V(self) -> float:
        return self.voltage_mV * 1e-3
    
    @property
    def name(self) -> str:
        return f"{self.voltage_mV:.0f}mV"


@dataclass 
class StimTrialResult:
    """Results from single trial"""
    condition: StimCondition
    trial_id: int
    
    # Calcium response
    peak_calcium_uM: float = 0.0
    calcium_integral: float = 0.0
    
    # Dimer formation
    peak_dimers: int = 0
    final_dimers: int = 0
    
    # Quantum metrics
    mean_singlet_prob: float = 0.0
    eligibility: float = 0.0
    peak_em_field_kT: float = 0.0
    
    # Output
    committed: bool = False
    commitment_level: float = 0.0
    final_strength: float = 1.0
    
    runtime_s: float = 0.0


@dataclass
class StimResult:
    """Complete experiment results"""
    conditions: List[StimCondition]
    trials: List[StimTrialResult]
    
    # Summary by voltage
    summary: Dict = field(default_factory=dict)
    
    # Fitted Hill parameters: strength = S_max / (1 + (K_half/V)^n)
    hill_V_half: float = 0.0  # Half-maximal voltage
    hill_n: float = 0.0  # Hill coefficient (cooperativity)
    hill_S_max: float = 0.0  # Maximum strength
    hill_r_squared: float = 0.0
    
    # Threshold detection
    threshold_voltage_mV: float = 0.0  # 50% commitment
    
    timestamp: str = ""
    runtime_s: float = 0.0


class StimIntensityExperiment:
    """
    Stimulation intensity experiment
    
    Characterizes input-output relationship of the cascade.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Voltage range (mV) - from subthreshold to fully depolarized
        if quick_mode:
            self.voltages_mV = [-60, -40, -20, 0]
            self.n_trials = 2
            self.n_synapses = 5
            self.consolidation_s = 0.5
        else:
            self.voltages_mV = [-70, -60, -50, -40, -30, -20, -10, 0, 10]
            self.n_trials = 5
            self.n_synapses = 10
            self.consolidation_s = 1.0
    
    def _create_network(self) -> MultiSynapseNetwork:
        """Create standard network"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_trial(self, condition: StimCondition, trial_id: int) -> StimTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = StimTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        # Create network
        network = self._create_network()
        
        dt = 0.001  # 1 ms timestep
        
        # Track metrics
        calcium_trace = []
        dimer_trace = []
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst at specified voltage) ===
        stim_voltage = condition.voltage_V
        
        for burst in range(5):
            for spike in range(4):
                # Depolarization at test voltage
                for _ in range(2):
                    state = network.step(dt, {"voltage": stim_voltage, "reward": True})
                    mean_ca = np.mean([s.calcium_peak_uM for s in state.synapse_states]) if state.synapse_states else 0
                    calcium_trace.append(mean_ca)  # Already in μM
                
                # Return to rest
                for _ in range(8):
                    state = network.step(dt, {"voltage": -70e-3, "reward": True})
                    mean_ca = np.mean([s.calcium_peak_uM for s in state.synapse_states]) if state.synapse_states else 0
                    calcium_trace.append(mean_ca)

            # Inter-burst interval
            for _ in range(160):
                state = network.step(dt, {"voltage": -70e-3, "reward": True})
            
            # Record dimers after each burst
            dimers = sum(
                len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                for s in network.synapses
            )
            dimer_trace.append(dimers)
        
        # Record peak calcium and integral
        result.peak_calcium_uM = max(calcium_trace) if calcium_trace else 0
        result.calcium_integral = sum(calcium_trace) * dt if calcium_trace else 0
        result.peak_dimers = max(dimer_trace) if dimer_trace else 0
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # === FINAL MEASUREMENTS ===
        metrics = network.get_experimental_metrics()
        result.peak_em_field_kT = metrics.get('mean_field_kT', 0)
        
        # Final dimer count
        result.final_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Coherence
        all_ps = []
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.25
        
        # Eligibility and commitment
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        result.committed = network.network_committed
        result.commitment_level = network.network_commitment_level
        result.final_strength = 1.0 + 0.5 * result.commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> StimResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning stimulation intensity experiment...")
            print(f"  Voltages: {self.voltages_mV} mV")
            print(f"  Trials per condition: {self.n_trials}")
        
        # Build conditions
        conditions = [StimCondition(
            voltage_mV=V,
            n_synapses=self.n_synapses
        ) for V in self.voltages_mV]
        
        # Run trials
        trials = []
        
        for cond in conditions:
            if self.verbose:
                print(f"  {cond.name}: ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                cond_trials = [t for t in trials if t.condition.voltage_mV == cond.voltage_mV]
                commit_rate = np.mean([1 if t.committed else 0 for t in cond_trials])
                strength = np.mean([t.final_strength for t in cond_trials])
                print(f" commit={commit_rate:.0%}, strength={strength:.2f}×")
        
        # Build result
        result = StimResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Fit Hill curve
        self._fit_hill_curve(result)
        
        # Find threshold
        self._find_threshold(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[StimTrialResult]) -> Dict:
        """Compute summary statistics by voltage"""
        summary = {}
        
        for V in self.voltages_mV:
            cond_trials = [t for t in trials if t.condition.voltage_mV == V]
            
            if cond_trials:
                summary[V] = {
                    'calcium_mean': np.mean([t.peak_calcium_uM for t in cond_trials]),
                    'calcium_std': np.std([t.peak_calcium_uM for t in cond_trials]),
                    'dimers_mean': np.mean([t.peak_dimers for t in cond_trials]),
                    'dimers_std': np.std([t.peak_dimers for t in cond_trials]),
                    'em_field_mean': np.mean([t.peak_em_field_kT for t in cond_trials]),
                    'eligibility_mean': np.mean([t.eligibility for t in cond_trials]),
                    'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                    'commitment_mean': np.mean([t.commitment_level for t in cond_trials]),
                    'strength_mean': np.mean([t.final_strength for t in cond_trials]),
                    'strength_std': np.std([t.final_strength for t in cond_trials]),
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def _fit_hill_curve(self, result: StimResult):
        """Fit Hill equation to strength vs voltage"""
        
        def hill(x, S_max, K_half, n):
            """Hill equation: S = S_max / (1 + (K_half/x)^n)"""
            x_shifted = x + 80  # Shift so all values positive
            K_shifted = K_half + 80
            return S_max / (1 + (K_shifted / np.maximum(x_shifted, 0.1)) ** n)
        
        x = np.array(self.voltages_mV, dtype=float)
        y = np.array([result.summary.get(V, {}).get('strength_mean', 1) for V in self.voltages_mV])
        
        try:
            # Initial guesses
            S_max_0 = np.max(y)
            K_half_0 = np.median(x)
            n_0 = 2.0
            
            popt, pcov = curve_fit(
                hill, x, y,
                p0=[S_max_0, K_half_0, n_0],
                bounds=([1.0, -80, 0.5], [3.0, 20, 10]),
                maxfev=5000
            )
            
            result.hill_S_max = popt[0]
            result.hill_V_half = popt[1]
            result.hill_n = popt[2]
            
            # R-squared
            y_pred = hill(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result.hill_r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Hill fit failed: {e}")
    
    def _find_threshold(self, result: StimResult):
        """Find voltage at 50% commitment"""
        for i, V in enumerate(self.voltages_mV):
            commit = result.summary.get(V, {}).get('commit_rate', 0)
            if commit >= 0.5:
                result.threshold_voltage_mV = V
                break
        else:
            result.threshold_voltage_mV = self.voltages_mV[-1]
    
    def print_summary(self, result: StimResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("STIMULATION INTENSITY RESULTS (IO CURVE)")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Voltage':<10} {'Ca²⁺ (μM)':<12} {'Dimers':<10} {'Commit':<10} {'Strength':<12}")
        print("-"*70)
        
        for V in self.voltages_mV:
            stats = result.summary.get(V, {})
            ca = stats.get('calcium_mean', 0)
            dimers = stats.get('dimers_mean', 0)
            commit = stats.get('commit_rate', 0)
            strength = stats.get('strength_mean', 1)
            strength_std = stats.get('strength_std', 0)
            
            print(f"{V:>6} mV   {ca:<12.2f} {dimers:<10.0f} {commit:<10.0%} {strength:.2f} ± {strength_std:.2f}×")
        
        print("\n" + "="*70)
        print("HILL FIT PARAMETERS")
        print("="*70)
        
        print(f"\n  S_max (max strength):     {result.hill_S_max:.2f}×")
        print(f"  V_half (half-max voltage): {result.hill_V_half:.1f} mV")
        print(f"  n (Hill coefficient):      {result.hill_n:.2f}")
        print(f"  R²:                        {result.hill_r_squared:.3f}")
        
        print(f"\n  Threshold voltage (50% commit): {result.threshold_voltage_mV} mV")
        
        if result.hill_n > 1.5:
            print("\n  ✓ COOPERATIVE BEHAVIOR DETECTED (n > 1.5)")
            print("    Consistent with collective quantum effects")
        elif result.hill_n > 1.0:
            print("\n  ~ MILD COOPERATIVITY (1 < n < 1.5)")
        else:
            print("\n  ⚠ NO COOPERATIVITY (n ≤ 1)")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: StimResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        voltages = self.voltages_mV
        
        # === Panel A: Calcium and Dimers vs Voltage ===
        ax1 = axes[0]
        
        ca_vals = [result.summary.get(V, {}).get('calcium_mean', 0) for V in voltages]
        ca_std = [result.summary.get(V, {}).get('calcium_std', 0) for V in voltages]
        dimer_vals = [result.summary.get(V, {}).get('dimers_mean', 0) for V in voltages]
        
        ax1.errorbar(voltages, ca_vals, yerr=ca_std, fmt='o-', color='#1f77b4',
                    markersize=8, capsize=4, linewidth=2, label='Ca²⁺ (μM)')
        ax1.set_ylabel('Peak Ca²⁺ (μM)', fontsize=11, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(voltages, dimer_vals, 's-', color='#ff7f0e', 
                     markersize=8, linewidth=2, label='Dimers')
        ax1_twin.set_ylabel('Peak Dimers', fontsize=11, color='#ff7f0e')
        ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
        
        ax1.set_xlabel('Depolarization (mV)', fontsize=11)
        ax1.set_title('A. Calcium & Dimer Response', fontweight='bold')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        # === Panel B: IO Curve with Hill Fit ===
        ax2 = axes[1]
        
        strength_vals = [result.summary.get(V, {}).get('strength_mean', 1) for V in voltages]
        strength_std = [result.summary.get(V, {}).get('strength_std', 0) for V in voltages]
        
        ax2.errorbar(voltages, strength_vals, yerr=strength_std, fmt='o', color='#2ca02c',
                    markersize=10, capsize=4, label='Data')
        
        # Plot Hill fit
        if result.hill_r_squared > 0.5:
            v_fit = np.linspace(min(voltages), max(voltages), 100)
            
            def hill(x, S_max, K_half, n):
                x_shifted = x + 80
                K_shifted = K_half + 80
                return S_max / (1 + (K_shifted / np.maximum(x_shifted, 0.1)) ** n)
            
            y_fit = hill(v_fit, result.hill_S_max, result.hill_V_half, result.hill_n)
            ax2.plot(v_fit, y_fit, '-', color='#2ca02c', linewidth=2, alpha=0.7,
                    label=f'Hill fit (n={result.hill_n:.1f})')
        
        # Mark half-max
        ax2.axvline(x=result.hill_V_half, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=(1 + result.hill_S_max)/2, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Depolarization (mV)', fontsize=11)
        ax2.set_ylabel('Synaptic Strength (×)', fontsize=11)
        ax2.set_title('B. Input-Output Curve', fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_ylim(0.9, max(strength_vals) * 1.1)
        
        # === Panel C: Commitment Rate ===
        ax3 = axes[2]
        
        commit_vals = [result.summary.get(V, {}).get('commit_rate', 0) for V in voltages]
        
        ax3.plot(voltages, commit_vals, 's-', color='#9467bd', markersize=10, linewidth=2)
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
        ax3.axvline(x=result.threshold_voltage_mV, color='red', linestyle='--', alpha=0.5,
                   label=f'Threshold: {result.threshold_voltage_mV} mV')
        
        ax3.set_xlabel('Depolarization (mV)', fontsize=11)
        ax3.set_ylabel('Commitment Rate', fontsize=11)
        ax3.set_title('C. Commitment Probability', fontweight='bold')
        ax3.set_ylim(-0.05, 1.1)
        ax3.legend(loc='upper left', fontsize=9)
        
        plt.suptitle('Stimulation Intensity: Input-Output Characteristics', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'stim_intensity.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: StimResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'stim_intensity',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'hill_S_max': result.hill_S_max,
            'hill_V_half': result.hill_V_half,
            'hill_n': result.hill_n,
            'hill_r_squared': result.hill_r_squared,
            'threshold_voltage_mV': result.threshold_voltage_mV,
            'summary': result.summary,
            'voltages_tested': self.voltages_mV,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: StimResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'hill_S_max': result.hill_S_max,
            'hill_V_half': result.hill_V_half,
            'hill_n': result.hill_n,
            'hill_r_squared': result.hill_r_squared,
            'threshold_voltage_mV': result.threshold_voltage_mV,
            'cooperative': result.hill_n > 1.5,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running stimulation intensity experiment (quick mode)...")
    exp = StimIntensityExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()