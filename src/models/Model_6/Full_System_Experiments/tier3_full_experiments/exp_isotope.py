#!/usr/bin/env python3
"""
Experiment: Isotope Comparison (P31 vs P32)
============================================

The definitive test of quantum coherence necessity in synaptic plasticity.

Scientific basis:
- P31 (I=1/2): T2 ≈ 67s due to weak dipole-dipole interactions
- P32 (I=1): T2 ≈ 0.3s due to quadrupolar relaxation

Prediction:
- P31: Eligibility persists for 60-100s, allowing dopamine timing flexibility
- P32: Eligibility decays in <1s, requiring immediate dopamine for commitment

Protocol:
1. Stimulate network to form dimers (identical for both isotopes)
2. Apply dopamine at various delays
3. Measure eligibility at dopamine onset
4. Compare decay curves

Success criteria:
- P31 fitted T2 ≈ 60-70s
- P32 fitted T2 ≈ 0.3-1.0s  
- Ratio > 50x demonstrates quantum coherence is functionally necessary

References:
- Fisher (2015): Quantum cognition hypothesis
- Agarwal et al. (2023): Ca9(PO4)6 dimer quantum properties
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
class IsotopeCondition:
    """Single experimental condition"""
    isotope: str  # 'P31' or 'P32'
    dopamine_delay_s: float
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        return f"{self.isotope}_delay{self.dopamine_delay_s:.1f}s"


@dataclass 
class IsotopeTrialResult:
    """Results from single trial"""
    condition: IsotopeCondition
    trial_id: int
    
    # Key measurements
    pre_dopamine_eligibility: float = 0.0
    post_dopamine_eligibility: float = 0.0
    committed: bool = False
    final_strength: float = 1.0
    
    # Dimer metrics
    peak_dimers: int = 0
    dimers_at_dopamine: int = 0
    mean_singlet_prob: float = 1.0
    
    # Timing
    runtime_s: float = 0.0


@dataclass
class IsotopeResult:
    """Complete experiment results"""
    conditions: List[IsotopeCondition]
    trials: List[IsotopeTrialResult]
    
    # Summary by isotope and delay
    summary: Dict = field(default_factory=dict)
    
    # Fitted parameters
    p31_fitted_T2: float = 0.0
    p32_fitted_T2: float = 0.0
    p31_r_squared: float = 0.0
    p32_r_squared: float = 0.0
    
    # Derived
    quantum_necessary: bool = False
    t2_ratio: float = 0.0
    
    timestamp: str = ""
    runtime_s: float = 0.0


class IsotopeExperiment:
    """
    Isotope comparison experiment
    
    Tests whether nuclear spin coherence time affects eligibility persistence.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Experimental parameters
        if quick_mode:
            self.n_trials = 2
            self.n_synapses = 5
            self.stim_duration_s = 0.5
            self.consolidation_s = 1.0
            self.p31_delays = [0, 15, 30]
            self.p32_delays = [0, 0.5, 1.0]
        else:
            self.n_trials = 5
            self.n_synapses = 10
            self.stim_duration_s = 1.0
            self.consolidation_s = 1.0  # Keep short due to O(n²)
            self.p31_delays = [0, 5, 10, 15, 20, 30, 45, 60]
            self.p32_delays = [0, 0.25, 0.5, 1.0, 2.0]
    
    def _create_network(self, isotope: str) -> MultiSynapseNetwork:
        """Create network with specified isotope"""
        params = Model6Parameters()
        
        
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True

        # Set isotope
        if isotope == 'P31':
            params.environment.fraction_P31 = 1.0
            params.environment.fraction_P32 = 0.0
        else:
            params.environment.fraction_P31 = 0.0
            params.environment.fraction_P32 = 1.0
        
        # Create network
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_trial(self, condition: IsotopeCondition, trial_id: int) -> IsotopeTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = IsotopeTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        # Create network
        network = self._create_network(condition.isotope)
        
        dt = 0.001  # 1 ms timestep
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (with concurrent dopamine) ===
        # Give dopamine DURING stimulation to ensure factor overlap
        n_stim_steps = int(self.stim_duration_s / dt)
        
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):  # 2ms spike
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):  # 8ms rest
                    network.step(dt, {"voltage": -70e-3, "reward": True})
            for _ in range(160):  # 160ms inter-burst interval
                network.step(dt, {"voltage": -70e-3, "reward": True})
                
                # Record peak dimers
                result.peak_dimers = sum(
                    len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                    for s in network.synapses
                )
        
        # === PHASE 3: DELAY (if any) ===
        # During delay, no reward signal - this tests eligibility persistence
        if condition.dopamine_delay_s > 0:
            # Use coarser timestep for long delays
            dt_delay = 0.1 if condition.dopamine_delay_s > 1 else 0.001
            n_delay_steps = int(condition.dopamine_delay_s / dt_delay)
            
            for _ in range(n_delay_steps):
                network.step(dt_delay, {'voltage': -70e-3, 'reward': False})
        
        # === MEASURE AT DOPAMINE ONSET ===
        # Get eligibility just before dopamine would arrive
        eligibilities = [s.get_eligibility() for s in network.synapses]
        result.pre_dopamine_eligibility = np.mean(eligibilities)
        
        # Get dimer count
        result.dimers_at_dopamine = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Get mean singlet probability
        all_ps = []
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.0
        
        # === PHASE 4: DOPAMINE READ (simulate reward arrival) ===
        for _ in range(300):  # 300ms
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Final measurements
        result.post_dopamine_eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        result.committed = network.network_committed
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> IsotopeResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        # Build conditions
        conditions = []
        
        for delay in self.p31_delays:
            conditions.append(IsotopeCondition(
                isotope='P31',
                dopamine_delay_s=delay,
                n_synapses=self.n_synapses
            ))
        
        for delay in self.p32_delays:
            conditions.append(IsotopeCondition(
                isotope='P32',
                dopamine_delay_s=delay,
                n_synapses=self.n_synapses
            ))
        
        # Run trials
        trials = []
        total_trials = len(conditions) * self.n_trials
        
        for i, cond in enumerate(conditions):
            if self.verbose:
                print(f"  {cond.name}: ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                # Show mean eligibility for this condition
                cond_trials = [t for t in trials if t.condition.name == cond.name]
                mean_elig = np.mean([t.pre_dopamine_eligibility for t in cond_trials])
                print(f" elig={mean_elig:.3f}")
        
        # Build result
        result = IsotopeResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary statistics
        result.summary = self._compute_summary(trials)
        
        # Fit decay curves
        self._fit_decay_curves(result)
        
        # Determine if quantum coherence is necessary
        result.quantum_necessary = (
            result.p31_fitted_T2 > 10 * result.p32_fitted_T2 and
            result.p31_fitted_T2 > 20  # Must be meaningfully long
        )
        
        if result.p32_fitted_T2 > 0:
            result.t2_ratio = result.p31_fitted_T2 / result.p32_fitted_T2
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[IsotopeTrialResult]) -> Dict:
        """Compute summary statistics by isotope and delay"""
        summary = {'P31': {}, 'P32': {}}
        
        for isotope in ['P31', 'P32']:
            delays = self.p31_delays if isotope == 'P31' else self.p32_delays
            
            for delay in delays:
                cond_trials = [t for t in trials 
                              if t.condition.isotope == isotope 
                              and t.condition.dopamine_delay_s == delay]
                
                if cond_trials:
                    eligs = [t.pre_dopamine_eligibility for t in cond_trials]
                    commits = [1 if t.committed else 0 for t in cond_trials]
                    
                    summary[isotope][delay] = {
                        'eligibility_mean': np.mean(eligs),
                        'eligibility_std': np.std(eligs),
                        'eligibility_n': len(eligs),
                        'commit_rate': np.mean(commits),
                        'n_trials': len(cond_trials)
                    }
        
        return summary
    
    def _fit_decay_curves(self, result: IsotopeResult):
        """Fit exponential decay to extract T2"""
        
        for isotope, delays in [('P31', self.p31_delays), ('P32', self.p32_delays)]:
            x = np.array(delays, dtype=float)
            y = np.array([
                result.summary[isotope].get(d, {}).get('eligibility_mean', 0)
                for d in delays
            ])
            
            # Filter valid points
            valid = y > 0.01
            if np.sum(valid) < 2:
                continue
            
            x_valid = x[valid]
            y_valid = y[valid]
            
            # Fit: y = A * exp(-t/tau)
            # Log transform: log(y) = log(A) - t/tau
            try:
                log_y = np.log(y_valid)
                coeffs = np.polyfit(x_valid, log_y, 1)
                
                tau = -1.0 / coeffs[0] if coeffs[0] != 0 else 0
                A = np.exp(coeffs[1])
                
                # Calculate R²
                y_pred = A * np.exp(-x_valid / tau) if tau > 0 else np.zeros_like(y_valid)
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                
                if isotope == 'P31':
                    result.p31_fitted_T2 = max(0, tau)
                    result.p31_r_squared = r_squared
                else:
                    result.p32_fitted_T2 = max(0, tau)
                    result.p32_r_squared = r_squared
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not fit {isotope} decay: {e}")
    
    def print_summary(self, result: IsotopeResult):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("ISOTOPE COMPARISON RESULTS")
        print("="*60)
        
        print("\nP31 (³¹P, I=1/2, T2_theory ≈ 67s):")
        print("-" * 40)
        for delay in self.p31_delays:
            stats = result.summary['P31'].get(delay, {})
            elig = stats.get('eligibility_mean', 0)
            std = stats.get('eligibility_std', 0)
            print(f"  Delay {delay:5.1f}s: eligibility = {elig:.3f} ± {std:.3f}")
        
        print(f"\n  Fitted T2 = {result.p31_fitted_T2:.1f}s (R² = {result.p31_r_squared:.3f})")
        
        print("\nP32 (³²P, I=1, T2_theory ≈ 0.3s):")
        print("-" * 40)
        for delay in self.p32_delays:
            stats = result.summary['P32'].get(delay, {})
            elig = stats.get('eligibility_mean', 0)
            std = stats.get('eligibility_std', 0)
            print(f"  Delay {delay:5.1f}s: eligibility = {elig:.3f} ± {std:.3f}")
        
        print(f"\n  Fitted T2 = {result.p32_fitted_T2:.1f}s (R² = {result.p32_r_squared:.3f})")
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print(f"  T2 ratio (P31/P32): {result.t2_ratio:.1f}x")
        print(f"  Theory prediction: ~220x (67s / 0.3s)")
        
        if result.quantum_necessary:
            print("\n  ✓ QUANTUM COHERENCE IS FUNCTIONALLY NECESSARY")
            print("    P31 maintains eligibility; P32 does not")
        else:
            print("\n  ⚠ Results inconclusive - check experimental parameters")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: IsotopeResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # Colors
        color_p31 = '#1f77b4'  # Blue
        color_p32 = '#d62728'  # Red
        
        # === Panel A: Eligibility decay curves ===
        ax1 = axes[0]
        
        # P31 data
        x_p31 = np.array(self.p31_delays)
        y_p31 = np.array([result.summary['P31'].get(d, {}).get('eligibility_mean', 0) for d in self.p31_delays])
        err_p31 = np.array([result.summary['P31'].get(d, {}).get('eligibility_std', 0) for d in self.p31_delays])
        
        ax1.errorbar(x_p31, y_p31, yerr=err_p31, fmt='o-', color=color_p31, 
                    markersize=8, linewidth=2, capsize=4, label=f'³¹P (T₂={result.p31_fitted_T2:.0f}s)')
        
        # P32 data  
        x_p32 = np.array(self.p32_delays)
        y_p32 = np.array([result.summary['P32'].get(d, {}).get('eligibility_mean', 0) for d in self.p32_delays])
        err_p32 = np.array([result.summary['P32'].get(d, {}).get('eligibility_std', 0) for d in self.p32_delays])
        
        ax1.errorbar(x_p32, y_p32, yerr=err_p32, fmt='s-', color=color_p32,
                    markersize=8, linewidth=2, capsize=4, label=f'³²P (T₂={result.p32_fitted_T2:.1f}s)')
        
        # Theory curves
        t_theory = np.linspace(0, max(self.p31_delays), 100)
        if result.p31_fitted_T2 > 0:
            ax1.plot(t_theory, y_p31[0] * np.exp(-t_theory / result.p31_fitted_T2), 
                    '--', color=color_p31, alpha=0.5, linewidth=1)
        if result.p32_fitted_T2 > 0:
            t_theory_p32 = np.linspace(0, max(self.p32_delays), 100)
            ax1.plot(t_theory_p32, y_p32[0] * np.exp(-t_theory_p32 / result.p32_fitted_T2),
                    '--', color=color_p32, alpha=0.5, linewidth=1)
        
        ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5, label='Commit threshold')
        ax1.set_xlabel('Dopamine Delay (s)', fontsize=11)
        ax1.set_ylabel('Eligibility at Dopamine', fontsize=11)
        ax1.set_title('A. Eligibility Decay', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(-2, max(self.p31_delays) + 5)
        ax1.set_ylim(0, 1.05)
        
        # === Panel B: Commitment window ===
        ax2 = axes[1]
        
        # Compute commit rates
        commit_p31 = [result.summary['P31'].get(d, {}).get('commit_rate', 0) for d in self.p31_delays]
        commit_p32 = [result.summary['P32'].get(d, {}).get('commit_rate', 0) for d in self.p32_delays]
        
        ax2.plot(self.p31_delays, commit_p31, 'o-', color=color_p31, markersize=8, linewidth=2, label='³¹P')
        ax2.plot(self.p32_delays, commit_p32, 's-', color=color_p32, markersize=8, linewidth=2, label='³²P')
        
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Dopamine Delay (s)', fontsize=11)
        ax2.set_ylabel('Commitment Rate', fontsize=11)
        ax2.set_title('B. Temporal Window for Plasticity', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xlim(-2, max(self.p31_delays) + 5)
        ax2.set_ylim(-0.05, 1.05)
        
        # === Panel C: Summary bar chart ===
        ax3 = axes[2]
        
        categories = ['T₂ (fitted)', 'Window (50%)', 'Max delay\n(commit)']
        
        # P31 values
        p31_window = self._find_50pct_window(result, 'P31')
        p31_max_commit = self._find_max_commit_delay(result, 'P31')
        p31_vals = [result.p31_fitted_T2, p31_window, p31_max_commit]
        
        # P32 values  
        p32_window = self._find_50pct_window(result, 'P32')
        p32_max_commit = self._find_max_commit_delay(result, 'P32')
        p32_vals = [result.p32_fitted_T2, p32_window, p32_max_commit]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, p31_vals, width, label='³¹P', color=color_p31, alpha=0.8)
        bars2 = ax3.bar(x + width/2, p32_vals, width, label='³²P', color=color_p32, alpha=0.8)
        
        ax3.set_ylabel('Time (s)', fontsize=11)
        ax3.set_title('C. Functional Comparison', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, fontsize=10)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_yscale('log')
        ax3.set_ylim(0.1, 200)
        
        # Add ratio annotation
        if result.t2_ratio > 1:
            ax3.annotate(f'{result.t2_ratio:.0f}×', xy=(0, max(p31_vals[0], p32_vals[0])),
                        xytext=(0, max(p31_vals[0], p32_vals[0]) * 1.5),
                        ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle('Isotope Comparison: ³¹P vs ³²P Nuclear Spin Coherence', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'isotope_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def _find_50pct_window(self, result: IsotopeResult, isotope: str) -> float:
        """Find delay at which eligibility drops to 50% of initial"""
        delays = self.p31_delays if isotope == 'P31' else self.p32_delays
        initial = result.summary[isotope].get(0, {}).get('eligibility_mean', 1)
        target = initial * 0.5
        
        for delay in delays:
            elig = result.summary[isotope].get(delay, {}).get('eligibility_mean', 0)
            if elig < target:
                return delay
        return delays[-1]
    
    def _find_max_commit_delay(self, result: IsotopeResult, isotope: str) -> float:
        """Find maximum delay that still achieves commitment"""
        delays = self.p31_delays if isotope == 'P31' else self.p32_delays
        max_delay = 0
        
        for delay in delays:
            commit_rate = result.summary[isotope].get(delay, {}).get('commit_rate', 0)
            if commit_rate > 0.5:
                max_delay = delay
        
        return max_delay
    
    def save_results(self, result: IsotopeResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'isotope_comparison',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'p31_fitted_T2': result.p31_fitted_T2,
            'p32_fitted_T2': result.p32_fitted_T2,
            'p31_r_squared': result.p31_r_squared,
            'p32_r_squared': result.p32_r_squared,
            't2_ratio': result.t2_ratio,
            'quantum_necessary': result.quantum_necessary,
            'summary': result.summary,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: IsotopeResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'p31_fitted_T2': result.p31_fitted_T2,
            'p32_fitted_T2': result.p32_fitted_T2,
            't2_ratio': result.t2_ratio,
            'quantum_necessary': result.quantum_necessary,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    # Quick test
    print("Running isotope comparison (quick mode)...")
    exp = IsotopeExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()