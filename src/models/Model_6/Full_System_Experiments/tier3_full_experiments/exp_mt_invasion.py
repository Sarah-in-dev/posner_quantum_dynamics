#!/usr/bin/env python3
"""
Experiment: MT Invasion Requirement
====================================

Compares MT+ (invaded) vs MT- (naive) synapses to validate tryptophan network
requirement for the Q1 layer of the cascade.

Scientific basis:
- Microtubules can invade dendritic spines during plasticity
- MT invasion brings tryptophan-rich tubulin into the spine
- Tryptophan networks enable superradiance (Q1 layer)
- Without MT invasion, synapses lack the quantum "operating system"

Conditions:
1. MT+ (invaded): Full tryptophan network → strong EM field → enhanced dimer formation
2. MT- (naive): No tryptophan network → weak EM field → reduced enhancement

Predictions:
- MT+: Strong Q1 field (>20 kT), enhanced dimer formation, full commitment
- MT-: Weak Q1 field (<5 kT), baseline dimer formation, reduced commitment

This tests whether the tryptophan superradiance layer is functionally necessary.
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

from model6_parameters import Model6Parameters
from multi_synapse_network import MultiSynapseNetwork


@dataclass
class MTCondition:
    """Single experimental condition"""
    name: str
    mt_invaded: bool
    n_synapses: int = 10
    
    @property
    def label(self) -> str:
        return "MT+" if self.mt_invaded else "MT-"


@dataclass 
class MTTrialResult:
    """Results from single trial"""
    condition: MTCondition
    trial_id: int
    
    # Q1 metrics
    peak_em_field_kT: float = 0.0
    mean_em_field_kT: float = 0.0
    
    # Q2 metrics
    peak_dimers: int = 0
    final_dimers: int = 0
    mean_singlet_prob: float = 1.0
    eligibility: float = 0.0
    
    # Enhancement ratio
    formation_enhancement: float = 1.0
    
    # Gate output
    committed: bool = False
    commitment_level: float = 0.0
    final_strength: float = 1.0
    
    # Timeline for plotting
    timeline: List[Dict] = field(default_factory=list)
    
    runtime_s: float = 0.0


@dataclass
class MTResult:
    """Complete experiment results"""
    conditions: List[MTCondition]
    trials: List[MTTrialResult]
    
    # Summary by condition
    summary: Dict = field(default_factory=dict)
    
    # Key comparisons
    mt_plus_field: float = 0.0
    mt_minus_field: float = 0.0
    field_ratio: float = 0.0
    
    mt_plus_commit: float = 0.0
    mt_minus_commit: float = 0.0
    
    # Validation
    mt_required: bool = False
    
    timestamp: str = ""
    runtime_s: float = 0.0


class MTInvasionExperiment:
    """
    MT invasion requirement experiment
    
    Tests whether tryptophan network (via MT invasion) is necessary for full cascade.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Experimental parameters
        if quick_mode:
            self.n_trials = 3
            self.n_synapses = 5
            self.stim_duration_s = 0.5
            self.consolidation_s = 0.5
        else:
            self.n_trials = 5
            self.n_synapses = 10
            self.stim_duration_s = 1.0
            self.consolidation_s = 1.0
    
    def _build_conditions(self) -> List[MTCondition]:
        """Build MT+ and MT- conditions"""
        return [
            MTCondition(name='MT+', mt_invaded=True, n_synapses=self.n_synapses),
            MTCondition(name='MT-', mt_invaded=False, n_synapses=self.n_synapses),
        ]
    
    def _create_network(self, mt_invaded: bool) -> MultiSynapseNetwork:
        """Create network with specified MT status"""
        params = Model6Parameters()
        
        network = MultiSynapseNetwork(
            n_synapses=self.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0,
            mt_invaded=mt_invaded
        )
        
        return network
    
    def _run_trial(self, condition: MTCondition, trial_id: int) -> MTTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = MTTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        network = self._create_network(condition.mt_invaded)
        dt = 0.001
        
        # Track values over time
        em_fields = []
        dimer_counts = []
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION + DOPAMINE ===
        n_stim = int(self.stim_duration_s / dt)
        
        for i in range(n_stim):
            network.step(dt, {"voltage": -10e-3, "reward": True})
            
            # Sample every 10ms
            if i % 10 == 0:
                metrics = network.get_network_metrics()
                em_field = metrics.get('collective_field_kT', 0)
                em_fields.append(em_field)
                
                dimers = sum(
                    len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                    for s in network.synapses
                )
                dimer_counts.append(dimers)
                
                # Record timeline
                result.timeline.append({
                    'time': i * dt,
                    'em_field': em_field,
                    'dimers': dimers
                })
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # === FINAL MEASUREMENTS ===
        result.peak_em_field_kT = max(em_fields) if em_fields else 0
        result.mean_em_field_kT = np.mean(em_fields) if em_fields else 0
        result.peak_dimers = max(dimer_counts) if dimer_counts else 0
        
        # Final dimer count
        result.final_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Singlet probability
        all_ps = []
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.0
        
        # Eligibility
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        
        # Formation enhancement (from first synapse if available)
        if network.synapses:
            s = network.synapses[0]
            if hasattr(s, 'get_experimental_metrics'):
                metrics = s.get_experimental_metrics()
                result.formation_enhancement = metrics.get('em_formation_enhancement', 1.0)
        
        # Commitment
        result.committed = network.network_committed
        result.commitment_level = network.network_commitment_level
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> MTResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        conditions = self._build_conditions()
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
                mean_field = np.mean([t.peak_em_field_kT for t in cond_trials])
                mean_commit = np.mean([t.committed for t in cond_trials])
                print(f" Q1={mean_field:.1f}kT, commit={mean_commit:.0%}")
        
        # Build result
        result = MTResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Extract key comparisons
        mt_plus = result.summary.get('MT+', {})
        mt_minus = result.summary.get('MT-', {})
        
        result.mt_plus_field = mt_plus.get('em_field_mean', 0)
        result.mt_minus_field = mt_minus.get('em_field_mean', 0)
        result.field_ratio = result.mt_plus_field / result.mt_minus_field if result.mt_minus_field > 0 else float('inf')
        
        result.mt_plus_commit = mt_plus.get('commit_rate', 0)
        result.mt_minus_commit = mt_minus.get('commit_rate', 0)
        
        # Validate: MT+ should have significantly higher field and commitment
        result.mt_required = (
            result.mt_plus_field > 2 * result.mt_minus_field and
            result.mt_plus_commit > result.mt_minus_commit + 0.2
        )
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[MTTrialResult]) -> Dict:
        """Compute summary statistics by condition"""
        summary = {}
        
        for name in ['MT+', 'MT-']:
            cond_trials = [t for t in trials if t.condition.name == name]
            
            if cond_trials:
                summary[name] = {
                    'em_field_mean': np.mean([t.peak_em_field_kT for t in cond_trials]),
                    'em_field_std': np.std([t.peak_em_field_kT for t in cond_trials]),
                    'dimers_mean': np.mean([t.peak_dimers for t in cond_trials]),
                    'dimers_std': np.std([t.peak_dimers for t in cond_trials]),
                    'eligibility_mean': np.mean([t.eligibility for t in cond_trials]),
                    'enhancement_mean': np.mean([t.formation_enhancement for t in cond_trials]),
                    'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                    'strength_mean': np.mean([t.final_strength for t in cond_trials]),
                    'strength_std': np.std([t.final_strength for t in cond_trials]),
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def print_summary(self, result: MTResult):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("MT INVASION EXPERIMENT RESULTS")
        print("="*60)
        
        print("\nComparison: MT+ (invaded) vs MT- (naive)")
        print("-" * 50)
        
        mt_plus = result.summary.get('MT+', {})
        mt_minus = result.summary.get('MT-', {})
        
        print(f"\n{'Metric':<25} {'MT+':<15} {'MT-':<15} {'Ratio':<10}")
        print("-" * 65)
        
        # Q1 Field
        plus_field = mt_plus.get('em_field_mean', 0)
        minus_field = mt_minus.get('em_field_mean', 0)
        ratio = plus_field / minus_field if minus_field > 0 else float('inf')
        print(f"{'Q1 Field (kT)':<25} {plus_field:<15.1f} {minus_field:<15.1f} {ratio:<10.1f}×")
        
        # Dimer count
        plus_dim = mt_plus.get('dimers_mean', 0)
        minus_dim = mt_minus.get('dimers_mean', 0)
        ratio = plus_dim / minus_dim if minus_dim > 0 else float('inf')
        print(f"{'Dimers':<25} {plus_dim:<15.0f} {minus_dim:<15.0f} {ratio:<10.1f}×")
        
        # Formation enhancement
        plus_enh = mt_plus.get('enhancement_mean', 1)
        minus_enh = mt_minus.get('enhancement_mean', 1)
        ratio = plus_enh / minus_enh if minus_enh > 0 else float('inf')
        print(f"{'EM Enhancement':<25} {plus_enh:<15.0f}× {minus_enh:<15.0f}× {ratio:<10.1f}×")
        
        # Commitment
        plus_commit = mt_plus.get('commit_rate', 0)
        minus_commit = mt_minus.get('commit_rate', 0)
        print(f"{'Commitment Rate':<25} {plus_commit:<15.0%} {minus_commit:<15.0%}")
        
        # Strength
        plus_str = mt_plus.get('strength_mean', 1)
        minus_str = mt_minus.get('strength_mean', 1)
        print(f"{'Final Strength':<25} {plus_str:<15.2f}× {minus_str:<15.2f}×")
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        
        if result.mt_required:
            print("\n  ✓ MT INVASION IS FUNCTIONALLY REQUIRED")
            print("    Tryptophan network provides essential Q1 layer")
            print(f"    Q1 field ratio: {result.field_ratio:.1f}×")
        else:
            print("\n  ⚠ Results inconclusive")
            print(f"    Q1 field ratio: {result.field_ratio:.1f}× (expected >2×)")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: MTResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        color_plus = '#2ca02c'   # Green for MT+
        color_minus = '#d62728'  # Red for MT-
        
        # === Panel A: Cascade comparison ===
        ax1 = axes[0]
        
        metrics = ['Q1 Field\n(kT)', 'Dimers', 'Commit\n(%)']
        mt_plus = result.summary.get('MT+', {})
        mt_minus = result.summary.get('MT-', {})
        
        plus_vals = [
            mt_plus.get('em_field_mean', 0),
            mt_plus.get('dimers_mean', 0),
            mt_plus.get('commit_rate', 0) * 100
        ]
        minus_vals = [
            mt_minus.get('em_field_mean', 0),
            mt_minus.get('dimers_mean', 0),
            mt_minus.get('commit_rate', 0) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, plus_vals, width, label='MT+ (invaded)', color=color_plus, alpha=0.8)
        ax1.bar(x + width/2, minus_vals, width, label='MT- (naive)', color=color_minus, alpha=0.8)
        
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('A. Cascade Components', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        
        # === Panel B: Timeline comparison ===
        ax2 = axes[1]
        
        # Get representative trials
        plus_trials = [t for t in result.trials if t.condition.name == 'MT+']
        minus_trials = [t for t in result.trials if t.condition.name == 'MT-']
        
        if plus_trials and plus_trials[0].timeline:
            times = [p['time'] for p in plus_trials[0].timeline]
            plus_fields = [p['em_field'] for p in plus_trials[0].timeline]
            ax2.plot(times, plus_fields, '-', color=color_plus, linewidth=2, label='MT+ Q1 field')
        
        if minus_trials and minus_trials[0].timeline:
            times = [p['time'] for p in minus_trials[0].timeline]
            minus_fields = [p['em_field'] for p in minus_trials[0].timeline]
            ax2.plot(times, minus_fields, '-', color=color_minus, linewidth=2, label='MT- Q1 field')
        
        ax2.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Threshold (20 kT)')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('EM Field (kT)', fontsize=11)
        ax2.set_title('B. Q1 Field Dynamics', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        
        # === Panel C: Enhancement ratio ===
        ax3 = axes[2]
        
        # Show the enhancement effect
        categories = ['Baseline\nFormation', 'EM\nEnhanced']
        
        # Baseline is 1x for both
        plus_baseline = 1.0
        minus_baseline = 1.0
        plus_enhanced = mt_plus.get('enhancement_mean', 1)
        minus_enhanced = mt_minus.get('enhancement_mean', 1)
        
        x = np.arange(len(categories))
        
        ax3.plot(x, [plus_baseline, plus_enhanced], 'o-', color=color_plus, 
                markersize=10, linewidth=2, label='MT+')
        ax3.plot(x, [minus_baseline, minus_enhanced], 's-', color=color_minus,
                markersize=10, linewidth=2, label='MT-')
        
        ax3.set_ylabel('Formation Rate (×)', fontsize=11)
        ax3.set_title('C. EM Field Enhancement', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, fontsize=10)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.set_ylim(0, max(plus_enhanced, minus_enhanced) * 1.2)
        
        # Annotate ratio
        if plus_enhanced > 1 and minus_enhanced > 0:
            ratio = plus_enhanced / minus_enhanced
            ax3.annotate(f'{ratio:.0f}× difference', 
                        xy=(1, (plus_enhanced + minus_enhanced)/2),
                        xytext=(1.3, (plus_enhanced + minus_enhanced)/2),
                        fontsize=10, ha='left',
                        arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.suptitle('MT Invasion Requirement: Tryptophan Network as Q1 Layer', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'mt_invasion.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: MTResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'mt_invasion',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'mt_plus_field': result.mt_plus_field,
            'mt_minus_field': result.mt_minus_field,
            'field_ratio': result.field_ratio,
            'mt_plus_commit': result.mt_plus_commit,
            'mt_minus_commit': result.mt_minus_commit,
            'mt_required': result.mt_required,
            'summary': result.summary,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: MTResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'mt_plus_field_kT': result.mt_plus_field,
            'mt_minus_field_kT': result.mt_minus_field,
            'field_ratio': result.field_ratio,
            'mt_plus_commit': result.mt_plus_commit,
            'mt_minus_commit': result.mt_minus_commit,
            'mt_required': result.mt_required,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running MT invasion experiment (quick mode)...")
    exp = MTInvasionExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()