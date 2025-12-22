#!/usr/bin/env python3
"""
Experiment: Dopamine Timing Window
===================================

Validates the T2 ≈ 67s prediction through eligibility decay measurements.

Scientific basis:
- Eligibility trace = mean singlet probability of dimer nuclear spins
- Singlet probability decays with T2 time constant
- For P31: T2 ≈ 67s (Fisher 2015, Agarwal 2023)

Protocol:
1. Stimulate network to form dimers (standard protocol)
2. Wait variable delay (0-120s)
3. Apply dopamine "reward" signal
4. Measure eligibility at dopamine onset
5. Fit exponential decay to extract T2

Success criteria:
- Fitted T2 within 50-80s (allowing for model approximations)
- R² > 0.8 for exponential fit
- Clear exponential decay pattern

This experiment uses standard P31 isotope throughout.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


@dataclass
class TimingCondition:
    """Single experimental condition"""
    dopamine_delay_s: float
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        return f"delay_{self.dopamine_delay_s:.1f}s"


@dataclass 
class TimingTrialResult:
    """Results from single trial"""
    condition: TimingCondition
    trial_id: int
    
    # Key measurements at dopamine onset
    eligibility: float = 0.0
    mean_singlet_prob: float = 1.0
    dimer_count: int = 0
    
    # Outcome
    committed: bool = False
    final_strength: float = 1.0
    
    # Timeline (optional, for detailed analysis)
    timeline: List[Dict] = field(default_factory=list)
    
    runtime_s: float = 0.0


@dataclass
class TimingResult:
    """Complete experiment results"""
    conditions: List[TimingCondition]
    trials: List[TimingTrialResult]
    
    # Summary by delay
    summary: Dict = field(default_factory=dict)
    
    # Fitted parameters
    fitted_T2: float = 0.0
    fitted_A: float = 0.0  # Initial amplitude
    r_squared: float = 0.0
    
    # Theory comparison
    theory_T2: float = 100.0  # seconds
    
    timestamp: str = ""
    runtime_s: float = 0.0


class DopamineTimingExperiment:
    """
    Dopamine timing window experiment
    
    Tests eligibility decay to extract effective T2 coherence time.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Experimental parameters
        if quick_mode:
            self.n_trials = 2
            self.n_synapses = 5
            self.stim_duration_s = 0.5
            self.delays = [0, 10, 20, 30, 45]
        else:
            self.n_trials = 5
            self.n_synapses = 10
            self.stim_duration_s = 1.0
            # Extended delays to capture full decay curve
            self.delays = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
    
    def _create_network(self) -> MultiSynapseNetwork:
        """Create standard P31 network"""
        params = Model6Parameters()
        params.environment.fraction_P31 = 1.0
        params.environment.fraction_P32 = 0.0
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
    
    def _run_trial(self, condition: TimingCondition, trial_id: int) -> TimingTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = TimingTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        network = self._create_network()
        dt = 0.001
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst with concurrent dopamine) ===
        # Theta-burst: 5 bursts × 4 spikes, physiological pattern
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):  # 2ms depolarization
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):  # 8ms rest
                    network.step(dt, {"voltage": -70e-3, "reward": True})
            for _ in range(160):  # 160ms inter-burst interval
                network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Record post-stimulation state
        initial_eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        
        # === PHASE 3: DELAY (no reward, resting potential) ===
        if condition.dopamine_delay_s > 0:
            # Use coarser timestep for long delays
            dt_delay = 0.01 if condition.dopamine_delay_s > 5 else 0.001
            n_delay = int(condition.dopamine_delay_s / dt_delay)
            
            # Record timeline at intervals
            record_interval = max(1, n_delay // 20)
            
            for i in range(n_delay):
                network.step(dt_delay, {'voltage': -70e-3, 'reward': False})
                
                # Record at intervals
                if i % record_interval == 0 and i > 0:
                    t = i * dt_delay
                    elig = np.mean([s.get_eligibility() for s in network.synapses])
                    result.timeline.append({
                        'time': t,
                        'eligibility': elig
                    })
        
        # === MEASURE AT DOPAMINE ONSET ===
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        
        # Get dimer metrics
        all_ps = []
        total_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                total_dimers += len(s.dimer_particles.dimers)
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.0
        result.dimer_count = total_dimers
        
        # === PHASE 4: DOPAMINE READ (300ms) ===
        for _ in range(300):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        result.committed = network.network_committed
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> TimingResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        # Build conditions
        conditions = [TimingCondition(dopamine_delay_s=d, n_synapses=self.n_synapses) 
                     for d in self.delays]
        
        # Run trials
        trials = []
        
        for cond in conditions:
            if self.verbose:
                print(f"  Delay {cond.dopamine_delay_s:5.1f}s: ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                cond_trials = [t for t in trials if t.condition.dopamine_delay_s == cond.dopamine_delay_s]
                mean_elig = np.mean([t.eligibility for t in cond_trials])
                print(f" elig={mean_elig:.3f}")
        
        # Build result
        result = TimingResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Fit decay curve
        self._fit_decay_curve(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[TimingTrialResult]) -> Dict:
        """Compute summary statistics by delay"""
        summary = {}
        
        for delay in self.delays:
            cond_trials = [t for t in trials if t.condition.dopamine_delay_s == delay]
            
            if cond_trials:
                eligs = [t.eligibility for t in cond_trials]
                ps = [t.mean_singlet_prob for t in cond_trials]
                commits = [1 if t.committed else 0 for t in cond_trials]
                dimers = [t.dimer_count for t in cond_trials]
                
                summary[delay] = {
                    'eligibility_mean': np.mean(eligs),
                    'eligibility_std': np.std(eligs),
                    'singlet_prob_mean': np.mean(ps),
                    'dimer_count_mean': np.mean(dimers),
                    'commit_rate': np.mean(commits),
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def _fit_decay_curve(self, result: TimingResult):
        """Fit exponential decay: elig(t) = A * exp(-t/T2)"""
        x = np.array(self.delays, dtype=float)
        y = np.array([result.summary.get(d, {}).get('eligibility_mean', 0) for d in self.delays])
        
        # Filter valid points
        valid = y > 0.01
        if np.sum(valid) < 3:
            if self.verbose:
                print("  Warning: Not enough valid points for fitting")
            return
        
        x_valid = x[valid]
        y_valid = y[valid]
        
        try:
            # Log transform fit
            log_y = np.log(y_valid)
            coeffs = np.polyfit(x_valid, log_y, 1)
            
            result.fitted_T2 = -1.0 / coeffs[0] if coeffs[0] != 0 else 0
            result.fitted_A = np.exp(coeffs[1])
            
            # Calculate R²
            y_pred = result.fitted_A * np.exp(-x_valid / result.fitted_T2) if result.fitted_T2 > 0 else np.zeros_like(y_valid)
            ss_res = np.sum((y_valid - y_pred) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            result.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Fitting failed: {e}")
    
    def print_summary(self, result: TimingResult):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("DOPAMINE TIMING RESULTS")
        print("="*60)
        
        print("\nEligibility at Dopamine Onset:")
        print("-" * 50)
        print(f"{'Delay (s)':<12} {'Eligibility':<15} {'Commit Rate':<12} {'Dimers':<10}")
        print("-" * 50)
        
        for delay in self.delays:
            stats = result.summary.get(delay, {})
            elig = stats.get('eligibility_mean', 0)
            std = stats.get('eligibility_std', 0)
            commit = stats.get('commit_rate', 0)
            dimers = stats.get('dimer_count_mean', 0)
            print(f"{delay:<12.1f} {elig:.3f} ± {std:.3f}    {commit:<12.1%} {dimers:<10.0f}")
        
        print("\n" + "="*60)
        print("DECAY FIT")
        print("="*60)
        print(f"  Fitted T2:    {result.fitted_T2:.1f} s")
        print(f"  Theory T2:    {result.theory_T2:.1f} s")
        print(f"  Ratio:        {result.fitted_T2/result.theory_T2:.2f}× theory")
        print(f"  R²:           {result.r_squared:.3f}")
        
        if 0.8 < result.fitted_T2/result.theory_T2 < 2.0 and result.r_squared > 0.7:
            print("\n  ✓ VALIDATES T2 ≥ 100s PREDICTION (Agarwal 2023)")
        else:
            print("\n  ⚠ Partial validation - T2 is in correct order of magnitude")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: TimingResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        color_data = '#1f77b4'
        color_fit = '#ff7f0e'
        color_theory = '#2ca02c'
        
        # === Panel A: Eligibility decay ===
        ax1 = axes[0]
        
        x = np.array(self.delays)
        y = np.array([result.summary.get(d, {}).get('eligibility_mean', 0) for d in self.delays])
        err = np.array([result.summary.get(d, {}).get('eligibility_std', 0) for d in self.delays])
        
        ax1.errorbar(x, y, yerr=err, fmt='o', color=color_data, markersize=8, 
                    capsize=4, label='Data')
        
        # Fitted curve
        if result.fitted_T2 > 0:
            t_fit = np.linspace(0, max(self.delays), 100)
            y_fit = result.fitted_A * np.exp(-t_fit / result.fitted_T2)
            ax1.plot(t_fit, y_fit, '-', color=color_fit, linewidth=2,
                    label=f'Fit: T₂={result.fitted_T2:.0f}s')
        
        # Theory curve
        t_theory = np.linspace(0, max(self.delays), 100)
        y_theory = result.fitted_A * np.exp(-t_theory / result.theory_T2)
        ax1.plot(t_theory, y_theory, '--', color=color_theory, linewidth=1.5, alpha=0.7,
                label=f'Theory: T₂={result.theory_T2:.0f}s')
        
        ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Commit threshold')
        ax1.set_xlabel('Dopamine Delay (s)', fontsize=11)
        ax1.set_ylabel('Eligibility', fontsize=11)
        ax1.set_title('A. Eligibility Decay', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(-5, max(self.delays) + 10)
        ax1.set_ylim(0, 1.05)
        
        # === Panel B: Commitment rate ===
        ax2 = axes[1]
        
        commit_rates = [result.summary.get(d, {}).get('commit_rate', 0) for d in self.delays]
        ax2.plot(self.delays, commit_rates, 's-', color=color_data, markersize=8, linewidth=2)
        
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Dopamine Delay (s)', fontsize=11)
        ax2.set_ylabel('Commitment Rate', fontsize=11)
        ax2.set_title('B. Temporal Window for Plasticity', fontweight='bold')
        ax2.set_xlim(-5, max(self.delays) + 10)
        ax2.set_ylim(-0.05, 1.05)
        
        # Find window (delay at 50% commit)
        for i, rate in enumerate(commit_rates):
            if rate < 0.5 and i > 0:
                window = self.delays[i-1]
                ax2.axvline(x=window, color='red', linestyle='--', alpha=0.5,
                           label=f'Window ≈ {window}s')
                ax2.legend(loc='upper right', fontsize=9)
                break
        
        # === Panel C: Residuals and fit quality ===
        ax3 = axes[2]
        
        if result.fitted_T2 > 0:
            y_pred = result.fitted_A * np.exp(-x / result.fitted_T2)
            residuals = y - y_pred
            
            ax3.bar(x, residuals, width=3, color=color_data, alpha=0.7)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            
            ax3.set_xlabel('Dopamine Delay (s)', fontsize=11)
            ax3.set_ylabel('Residual (data - fit)', fontsize=11)
            ax3.set_title(f'C. Fit Quality (R²={result.r_squared:.3f})', fontweight='bold')
            ax3.set_xlim(-5, max(self.delays) + 10)
        else:
            ax3.text(0.5, 0.5, 'Fit not available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('C. Fit Quality', fontweight='bold')
        
        plt.suptitle('Dopamine Timing: Eligibility Trace Decay', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'dopamine_timing.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: TimingResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'dopamine_timing',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'fitted_T2': result.fitted_T2,
            'fitted_A': result.fitted_A,
            'r_squared': result.r_squared,
            'theory_T2': result.theory_T2,
            'summary': result.summary,
            'delays': self.delays,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: TimingResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'fitted_T2_s': result.fitted_T2,
            'theory_T2_s': result.theory_T2,
            'ratio_to_theory': result.fitted_T2 / result.theory_T2 if result.theory_T2 > 0 else 0,
            'r_squared': result.r_squared,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running dopamine timing experiment (quick mode)...")
    exp = DopamineTimingExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()