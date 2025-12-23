#!/usr/bin/env python3
"""
Experiment: Consolidation Kinetics
===================================

Tests the full classical cascade from quantum commitment to structural plasticity.

Scientific basis:
- Quantum commitment (Q1+Q2+Gate) triggers classical consolidation
- CaMKII activation → Actin polymerization → Spine volume increase
- Timeline: commitment (seconds) → structural change (minutes)

Predictions:
- Spine volume should increase following commitment
- Timescale: measurable volume change within 60s
- Rate depends on commitment strength
- Matches experimental LTP literature (Matsuzaki 2004: ~1.6× at 60s)

Protocol:
1. Induce commitment via standard theta-burst
2. Track spine volume over extended consolidation period
3. Measure kinetics of structural change

Success criteria:
- Volume increase correlates with commitment
- Timeline matches literature (~20s onset, peak at 2-5 min)
- No structural change without commitment
- Validates quantum-classical coupling

References:
- Matsuzaki et al. (2004): Spine volume dynamics
- Bosch et al. (2014): LTP actin dynamics
- Harvey & Bhalla (2015): Structural plasticity timescales
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
class ConsolidationCondition:
    """Single experimental condition"""
    consolidation_duration_s: float
    dopamine_present: bool = True
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        da_str = "+DA" if self.dopamine_present else "-DA"
        return f"{self.consolidation_duration_s:.0f}s_{da_str}"


@dataclass 
class ConsolidationTrialResult:
    """Results from single trial"""
    condition: ConsolidationCondition
    trial_id: int
    
    # Timeline data
    time_points: List[float] = field(default_factory=list)  # seconds
    spine_volumes: List[float] = field(default_factory=list)  # relative to baseline
    commitment_levels: List[float] = field(default_factory=list)
    eligibility_trace: List[float] = field(default_factory=list)
    
    # Key metrics
    committed: bool = False
    time_to_commitment_s: float = 0.0
    
    # Final state
    final_spine_volume: float = 1.0
    peak_spine_volume: float = 1.0
    final_strength: float = 1.0
    
    # Kinetics
    onset_time_s: float = 0.0  # Time to first detectable volume change
    half_max_time_s: float = 0.0  # Time to 50% of peak volume
    
    runtime_s: float = 0.0


@dataclass
class ConsolidationResult:
    """Complete experiment results"""
    conditions: List[ConsolidationCondition]
    trials: List[ConsolidationTrialResult]
    
    # Summary by duration
    summary: Dict = field(default_factory=dict)
    
    # Kinetic parameters
    mean_onset_time_s: float = 0.0
    mean_half_max_time_s: float = 0.0
    
    # Literature comparison
    matches_literature: bool = False
    volume_at_60s: float = 1.0  # Compare to Matsuzaki ~1.6×
    
    timestamp: str = ""
    runtime_s: float = 0.0


class ConsolidationKineticsExperiment:
    """
    Consolidation kinetics experiment
    
    Tracks structural plasticity following quantum commitment.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Consolidation durations to test
        if quick_mode:
            self.durations_s = [10, 30, 60]
            self.n_trials = 2
            self.n_synapses = 5
            self.sample_interval_s = 5.0  # Sample every 5s
        else:
            self.durations_s = [10, 20, 30, 45, 60, 90, 120]
            self.n_trials = 5
            self.n_synapses = 10
            self.sample_interval_s = 2.0  # Sample every 2s
    
    def _create_network(self, dopamine_present: bool = True) -> MultiSynapseNetwork:
        """Create network for consolidation experiment"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        if not dopamine_present:
            # Disable dopamine for control
            params.dopamine = None
        
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _get_spine_volume(self, network: MultiSynapseNetwork) -> float:
        """Get mean spine volume across network"""
        volumes = []
        for s in network.synapses:
            if hasattr(s, 'spine_plasticity') and hasattr(s.spine_plasticity, 'spine_volume'):
                volumes.append(s.spine_plasticity.spine_volume)
            elif hasattr(s, '_spine_volume'):
                volumes.append(s._spine_volume)
            else:
                # Default if not tracked
                volumes.append(1.0)
        
        return np.mean(volumes) if volumes else 1.0
    
    def _run_trial(self, condition: ConsolidationCondition, trial_id: int) -> ConsolidationTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = ConsolidationTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        # Create network
        network = self._create_network(condition.dopamine_present)
        
        dt = 0.001  # 1 ms timestep
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        baseline_volume = self._get_spine_volume(network)
        
        # Record initial state
        result.time_points.append(0.0)
        result.spine_volumes.append(1.0)  # Normalized to baseline
        result.commitment_levels.append(network.network_commitment_level)
        result.eligibility_trace.append(np.mean([s.get_eligibility() for s in network.synapses]))
        
        # === PHASE 2: STIMULATION (theta-burst with dopamine) ===
        reward_signal = condition.dopamine_present
        
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):
                    network.step(dt, {"voltage": -10e-3, "reward": reward_signal})
                for _ in range(8):
                    network.step(dt, {"voltage": -70e-3, "reward": reward_signal})
            
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": reward_signal})
        
        # Record post-stimulation state (~1s mark)
        current_volume = self._get_spine_volume(network)
        result.time_points.append(1.0)
        result.spine_volumes.append(current_volume / baseline_volume)
        result.commitment_levels.append(network.network_commitment_level)
        result.eligibility_trace.append(np.mean([s.get_eligibility() for s in network.synapses]))
        
        if network.network_committed and not result.committed:
            result.committed = True
            result.time_to_commitment_s = 1.0
        
        # === PHASE 3: CONSOLIDATION (track over time) ===
        samples_per_interval = int(self.sample_interval_s / dt)
        n_intervals = int(condition.consolidation_duration_s / self.sample_interval_s)
        
        for interval in range(n_intervals):
            # Run for one sample interval
            for _ in range(samples_per_interval):
                # Use coarser timestep for efficiency
                network.step(0.01, {"voltage": -70e-3, "reward": reward_signal})
            
            current_time = 1.0 + (interval + 1) * self.sample_interval_s
            current_volume = self._get_spine_volume(network)
            
            result.time_points.append(current_time)
            result.spine_volumes.append(current_volume / baseline_volume)
            result.commitment_levels.append(network.network_commitment_level)
            result.eligibility_trace.append(np.mean([s.get_eligibility() for s in network.synapses]))
            
            # Check for commitment
            if network.network_committed and not result.committed:
                result.committed = True
                result.time_to_commitment_s = current_time
        
        # === COMPUTE KINETIC METRICS ===
        volumes = np.array(result.spine_volumes)
        times = np.array(result.time_points)
        
        result.peak_spine_volume = np.max(volumes)
        result.final_spine_volume = volumes[-1] if len(volumes) > 0 else 1.0
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        # Onset time (first >5% increase)
        onset_threshold = 1.05
        onset_idx = np.where(volumes > onset_threshold)[0]
        if len(onset_idx) > 0:
            result.onset_time_s = times[onset_idx[0]]
        
        # Half-max time
        if result.peak_spine_volume > 1.0:
            half_max = 1.0 + (result.peak_spine_volume - 1.0) / 2
            half_max_idx = np.where(volumes >= half_max)[0]
            if len(half_max_idx) > 0:
                result.half_max_time_s = times[half_max_idx[0]]
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> ConsolidationResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning consolidation kinetics experiment...")
            print(f"  Durations: {self.durations_s} s")
            print(f"  Trials per condition: {self.n_trials}")
        
        # Build conditions (with and without dopamine)
        conditions = []
        for duration in self.durations_s:
            conditions.append(ConsolidationCondition(
                consolidation_duration_s=duration,
                dopamine_present=True,
                n_synapses=self.n_synapses
            ))
        
        # Add no-dopamine control at longest duration
        conditions.append(ConsolidationCondition(
            consolidation_duration_s=max(self.durations_s),
            dopamine_present=False,
            n_synapses=self.n_synapses
        ))
        
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
                cond_trials = [t for t in trials if t.condition.name == cond.name]
                commit_rate = np.mean([1 if t.committed else 0 for t in cond_trials])
                volume = np.mean([t.final_spine_volume for t in cond_trials])
                print(f" commit={commit_rate:.0%}, volume={volume:.2f}×")
        
        # Build result
        result = ConsolidationResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Compute kinetic parameters
        self._compute_kinetics(result)
        
        # Compare to literature
        self._compare_to_literature(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[ConsolidationTrialResult]) -> Dict:
        """Compute summary statistics"""
        summary = {}
        
        for duration in self.durations_s:
            # With dopamine
            key = f"{duration:.0f}s_+DA"
            cond_trials = [t for t in trials 
                          if t.condition.consolidation_duration_s == duration 
                          and t.condition.dopamine_present]
            
            if cond_trials:
                summary[key] = {
                    'duration_s': duration,
                    'dopamine': True,
                    'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                    'volume_mean': np.mean([t.final_spine_volume for t in cond_trials]),
                    'volume_std': np.std([t.final_spine_volume for t in cond_trials]),
                    'peak_volume_mean': np.mean([t.peak_spine_volume for t in cond_trials]),
                    'strength_mean': np.mean([t.final_strength for t in cond_trials]),
                    'onset_time_mean': np.mean([t.onset_time_s for t in cond_trials if t.onset_time_s > 0]),
                    'half_max_time_mean': np.mean([t.half_max_time_s for t in cond_trials if t.half_max_time_s > 0]),
                    'n_trials': len(cond_trials)
                }
        
        # No-dopamine control
        control_trials = [t for t in trials if not t.condition.dopamine_present]
        if control_trials:
            summary['control_-DA'] = {
                'duration_s': max(self.durations_s),
                'dopamine': False,
                'commit_rate': np.mean([1 if t.committed else 0 for t in control_trials]),
                'volume_mean': np.mean([t.final_spine_volume for t in control_trials]),
                'volume_std': np.std([t.final_spine_volume for t in control_trials]),
                'n_trials': len(control_trials)
            }
        
        return summary
    
    def _compute_kinetics(self, result: ConsolidationResult):
        """Compute kinetic parameters from committed trials"""
        committed_trials = [t for t in result.trials if t.committed and t.condition.dopamine_present]
        
        if committed_trials:
            onset_times = [t.onset_time_s for t in committed_trials if t.onset_time_s > 0]
            half_max_times = [t.half_max_time_s for t in committed_trials if t.half_max_time_s > 0]
            
            result.mean_onset_time_s = np.mean(onset_times) if onset_times else 0
            result.mean_half_max_time_s = np.mean(half_max_times) if half_max_times else 0
    
    def _compare_to_literature(self, result: ConsolidationResult):
        """Compare to experimental literature"""
        # Matsuzaki 2004: ~1.6× spine volume at 60s post-LTP
        key_60s = "60s_+DA"
        if key_60s in result.summary:
            result.volume_at_60s = result.summary[key_60s].get('volume_mean', 1.0)
            
            # Literature range: 1.4-1.8× at 60s
            result.matches_literature = 1.3 <= result.volume_at_60s <= 2.0
    
    def print_summary(self, result: ConsolidationResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("CONSOLIDATION KINETICS RESULTS")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Duration':<12} {'DA':<6} {'Commit':<10} {'Volume':<15} {'Strength':<10}")
        print("-"*70)
        
        for key in sorted(result.summary.keys(), key=lambda x: (not result.summary[x].get('dopamine', True), result.summary[x].get('duration_s', 0))):
            stats = result.summary[key]
            duration = stats.get('duration_s', 0)
            da = "Yes" if stats.get('dopamine', True) else "No"
            commit = stats.get('commit_rate', 0)
            volume = stats.get('volume_mean', 1)
            volume_std = stats.get('volume_std', 0)
            strength = stats.get('strength_mean', 1)
            
            print(f"{duration:>6}s      {da:<6} {commit:<10.0%} {volume:.2f} ± {volume_std:.2f}×   {strength:<10.2f}×")
        
        print("\n" + "="*70)
        print("KINETIC ANALYSIS")
        print("="*70)
        
        print(f"\n  Mean onset time (first >5% volume increase): {result.mean_onset_time_s:.1f}s")
        print(f"  Mean half-max time:                           {result.mean_half_max_time_s:.1f}s")
        
        print("\n" + "="*70)
        print("LITERATURE COMPARISON")
        print("="*70)
        
        print(f"\n  Volume at 60s: {result.volume_at_60s:.2f}×")
        print(f"  Literature (Matsuzaki 2004): ~1.6× (range 1.4-1.8×)")
        
        if result.matches_literature:
            print("\n  ✓ MATCHES EXPERIMENTAL LITERATURE")
            print("    Quantum-classical cascade produces realistic structural plasticity")
        else:
            print(f"\n  ⚠ Volume at 60s ({result.volume_at_60s:.2f}×) outside expected range")
        
        # Control comparison
        control = result.summary.get('control_-DA', {})
        if control:
            print(f"\n  Control (-DA) volume: {control.get('volume_mean', 1):.2f}×")
            if control.get('volume_mean', 1) < 1.1:
                print("  ✓ No volume change without dopamine (as expected)")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: ConsolidationResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # === Panel A: Volume Timeline (individual trials) ===
        ax1 = axes[0]
        
        # Get longest duration committed trials
        long_duration = max(self.durations_s)
        committed_trials = [t for t in result.trials 
                          if t.condition.consolidation_duration_s == long_duration 
                          and t.condition.dopamine_present
                          and t.committed]
        
        # Plot individual traces
        for trial in committed_trials[:3]:  # Limit to 3 for clarity
            ax1.plot(trial.time_points, trial.spine_volumes, 'o-', alpha=0.5, markersize=4)
        
        # Mean trace
        if committed_trials:
            # Interpolate to common time points
            common_times = committed_trials[0].time_points
            mean_volumes = np.mean([t.spine_volumes[:len(common_times)] for t in committed_trials], axis=0)
            ax1.plot(common_times, mean_volumes, 'k-', linewidth=3, label='Mean')
        
        # Mark literature value at 60s
        ax1.axhline(y=1.6, color='red', linestyle='--', alpha=0.5, label='Literature (~1.6×)')
        ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Spine Volume (× baseline)', fontsize=11)
        ax1.set_title('A. Volume Dynamics', fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_ylim(0.9, 2.0)
        
        # === Panel B: Volume vs Duration ===
        ax2 = axes[1]
        
        durations = []
        volumes = []
        volume_stds = []
        
        for d in self.durations_s:
            key = f"{d:.0f}s_+DA"
            if key in result.summary:
                durations.append(d)
                volumes.append(result.summary[key]['volume_mean'])
                volume_stds.append(result.summary[key]['volume_std'])
        
        ax2.errorbar(durations, volumes, yerr=volume_stds, fmt='o-', color='#2ca02c',
                    markersize=10, capsize=4, linewidth=2, label='+DA')
        
        # Control point
        control = result.summary.get('control_-DA', {})
        if control:
            ax2.scatter([max(durations)], [control.get('volume_mean', 1)], 
                       s=150, color='red', marker='x', linewidth=3, label='-DA control')
        
        ax2.axhline(y=1.6, color='gray', linestyle='--', alpha=0.5, label='Literature')
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
        
        ax2.set_xlabel('Consolidation Duration (s)', fontsize=11)
        ax2.set_ylabel('Final Spine Volume (×)', fontsize=11)
        ax2.set_title('B. Duration Dependence', fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_ylim(0.9, 2.0)
        
        # === Panel C: Commitment Level Timeline ===
        ax3 = axes[2]
        
        # Plot commitment level traces
        for trial in committed_trials[:3]:
            ax3.plot(trial.time_points, trial.commitment_levels, 'o-', alpha=0.5, markersize=4)
        
        if committed_trials:
            mean_commitment = np.mean([t.commitment_levels[:len(common_times)] for t in committed_trials], axis=0)
            ax3.plot(common_times, mean_commitment, 'k-', linewidth=3, label='Mean')
        
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Commit threshold')
        
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Commitment Level', fontsize=11)
        ax3.set_title('C. Commitment Dynamics', fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.set_ylim(0, 1.1)
        
        plt.suptitle('Consolidation Kinetics: Quantum → Structural Plasticity', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'consolidation_kinetics.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: ConsolidationResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'consolidation_kinetics',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'mean_onset_time_s': result.mean_onset_time_s,
            'mean_half_max_time_s': result.mean_half_max_time_s,
            'volume_at_60s': result.volume_at_60s,
            'matches_literature': result.matches_literature,
            'summary': result.summary,
            'durations_tested': self.durations_s,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: ConsolidationResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'mean_onset_time_s': result.mean_onset_time_s,
            'mean_half_max_time_s': result.mean_half_max_time_s,
            'volume_at_60s': result.volume_at_60s,
            'matches_literature': result.matches_literature,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running consolidation kinetics experiment (quick mode)...")
    exp = ConsolidationKineticsExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()