#!/usr/bin/env python3
"""
Experiment: Spatial Clustering Effects
=======================================

Tests how synapse geometry affects quantum cooperativity.

Scientific basis:
- EM coupling strength decays with distance (exponential)
- Clustered synapses should have stronger collective effects
- Distributed synapses may not reach critical coupling threshold

Predictions:
- Clustered synapses: Strong EM coupling, faster commitment
- Linear arrangement: Intermediate coupling
- Distributed synapses: Weak coupling, may fail to commit

Protocol:
1. Create networks with identical synapse count but different geometries
2. Apply identical stimulation protocol
3. Measure EM field coherence, coupling strength, commitment

Success criteria:
- Clustered > Linear > Distributed for field strength
- Coupling decay matches expected exponential
- Validates collective quantum effects require proximity

References:
- Harvey & Bhalla (2015): Clustered plasticity
- Murakoshi & Bhalla (2012): Spine clustering
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
class ClusterCondition:
    """Single experimental condition"""
    pattern: str  # 'clustered', 'linear', 'distributed', 'random'
    spacing_um: float  # Characteristic spacing
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        return f"{self.pattern}_{self.spacing_um:.1f}µm"


@dataclass 
class ClusterTrialResult:
    """Results from single trial"""
    condition: ClusterCondition
    trial_id: int
    
    # Geometry metrics
    mean_distance_um: float = 0.0
    max_distance_um: float = 0.0
    mean_coupling_strength: float = 0.0
    
    # Q1 metrics
    peak_em_field_kT: float = 0.0
    field_coherence: float = 0.0  # How synchronized across synapses
    
    # Q2 metrics
    peak_dimers: int = 0
    entanglement_fraction: float = 0.0
    
    # Output
    eligibility: float = 0.0
    committed: bool = False
    commitment_time_s: float = 0.0  # Time to commitment
    final_strength: float = 1.0
    
    runtime_s: float = 0.0


@dataclass
class ClusterResult:
    """Complete experiment results"""
    conditions: List[ClusterCondition]
    trials: List[ClusterTrialResult]
    
    # Summary by pattern
    summary: Dict = field(default_factory=dict)
    
    # Ranking
    best_pattern: str = ""
    worst_pattern: str = ""
    
    # Coupling analysis
    coupling_decay_constant: float = 0.0  # μm
    
    timestamp: str = ""
    runtime_s: float = 0.0


class SpatialClusteringExperiment:
    """
    Spatial clustering experiment
    
    Tests how synapse geometry affects quantum cooperativity.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Patterns and spacings
        if quick_mode:
            self.patterns = ['clustered', 'distributed']
            self.spacings_um = [0.5, 2.0]
            self.n_trials = 2
            self.n_synapses = 5
            self.consolidation_s = 0.5
        else:
            self.patterns = ['clustered', 'linear', 'distributed', 'random']
            self.spacings_um = [0.5, 1.0, 2.0, 5.0]
            self.n_trials = 5
            self.n_synapses = 10
            self.consolidation_s = 1.0
    
    def _create_network(self, condition: ClusterCondition) -> MultiSynapseNetwork:
        """Create network with specified geometry"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        # Set coupling length constant
        if hasattr(params, 'em_coupling'):
            params.em_coupling.coupling_length_um = 2.0  # Standard value
        
        network = MultiSynapseNetwork(
            n_synapses=condition.n_synapses,
            params=params,
            pattern=condition.pattern,
            spacing_um=condition.spacing_um
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_trial(self, condition: ClusterCondition, trial_id: int) -> ClusterTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = ClusterTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        # Create network
        network = self._create_network(condition)
        
        # Record geometry metrics
        distances = network.distances[np.triu_indices(condition.n_synapses, k=1)]
        result.mean_distance_um = np.mean(distances)
        result.max_distance_um = np.max(distances)
        
        couplings = network.coupling_weights[np.triu_indices(condition.n_synapses, k=1)]
        result.mean_coupling_strength = np.mean(couplings)
        
        dt = 0.001  # 1 ms timestep
        
        # Track metrics over time
        em_fields = []
        per_synapse_fields = []
        commitment_time = None
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst with dopamine) ===
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):
                    state = network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):
                    state = network.step(dt, {"voltage": -70e-3, "reward": True})
            
            # Record EM field
            metrics = network.get_experimental_metrics()
            em_fields.append(metrics.get('mean_field_kT', 0))
            
            # Record per-synapse fields for coherence calculation
            synapse_fields = []
            for s in network.synapses:
                if hasattr(s, '_em_field_trp'):
                    synapse_fields.append(s._em_field_trp)
            if synapse_fields:
                per_synapse_fields.append(synapse_fields)
            
            # Check for commitment
            if network.network_committed and commitment_time is None:
                commitment_time = burst * 0.2  # Approximate time in seconds
            
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Record peak EM field
        result.peak_em_field_kT = max(em_fields) if em_fields else 0
        
        # Calculate field coherence (how synchronized are synapses)
        if per_synapse_fields:
            # Coherence = 1 - coefficient of variation
            cvs = []
            for fields in per_synapse_fields:
                if len(fields) > 1 and np.mean(fields) > 0:
                    cv = np.std(fields) / np.mean(fields)
                    cvs.append(cv)
            result.field_coherence = 1 - np.mean(cvs) if cvs else 0
        
        # Record dimers
        result.peak_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Entanglement fraction
        total_dimers = 0
        entangled_dimers = 0
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                for d in s.dimer_particles.dimers:
                    total_dimers += 1
                    if hasattr(d, 'is_entangled') and d.is_entangled:
                        entangled_dimers += 1
        result.entanglement_fraction = entangled_dimers / total_dimers if total_dimers > 0 else 0
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for step in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
            
            # Check for commitment
            if network.network_committed and commitment_time is None:
                commitment_time = 1.0 + step * dt  # Time after stimulation
        
        # === FINAL MEASUREMENTS ===
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        result.committed = network.network_committed
        result.commitment_time_s = commitment_time if commitment_time else 0
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> ClusterResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning spatial clustering experiment...")
            print(f"  Patterns: {self.patterns}")
            print(f"  Spacings: {self.spacings_um} µm")
            print(f"  Trials per condition: {self.n_trials}")
        
        # Build conditions
        conditions = []
        for pattern in self.patterns:
            for spacing in self.spacings_um:
                conditions.append(ClusterCondition(
                    pattern=pattern,
                    spacing_um=spacing,
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
                field = np.mean([t.peak_em_field_kT for t in cond_trials])
                commit = np.mean([1 if t.committed else 0 for t in cond_trials])
                print(f" Q1={field:.1f}kT, commit={commit:.0%}")
        
        # Build result
        result = ClusterResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Find best/worst patterns
        self._rank_patterns(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[ClusterTrialResult]) -> Dict:
        """Compute summary statistics"""
        summary = {}
        
        for pattern in self.patterns:
            for spacing in self.spacings_um:
                key = f"{pattern}_{spacing:.1f}µm"
                cond_trials = [t for t in trials 
                              if t.condition.pattern == pattern 
                              and t.condition.spacing_um == spacing]
                
                if cond_trials:
                    summary[key] = {
                        'pattern': pattern,
                        'spacing_um': spacing,
                        'mean_distance_um': np.mean([t.mean_distance_um for t in cond_trials]),
                        'mean_coupling': np.mean([t.mean_coupling_strength for t in cond_trials]),
                        'em_field_mean': np.mean([t.peak_em_field_kT for t in cond_trials]),
                        'em_field_std': np.std([t.peak_em_field_kT for t in cond_trials]),
                        'field_coherence_mean': np.mean([t.field_coherence for t in cond_trials]),
                        'dimers_mean': np.mean([t.peak_dimers for t in cond_trials]),
                        'entanglement_mean': np.mean([t.entanglement_fraction for t in cond_trials]),
                        'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                        'commit_time_mean': np.mean([t.commitment_time_s for t in cond_trials if t.committed]),
                        'strength_mean': np.mean([t.final_strength for t in cond_trials]),
                        'n_trials': len(cond_trials)
                    }
        
        return summary
    
    def _rank_patterns(self, result: ClusterResult):
        """Rank patterns by performance"""
        pattern_scores = {}
        
        for pattern in self.patterns:
            pattern_trials = [t for t in result.trials if t.condition.pattern == pattern]
            if pattern_trials:
                # Score based on field strength and commitment
                field = np.mean([t.peak_em_field_kT for t in pattern_trials])
                commit = np.mean([1 if t.committed else 0 for t in pattern_trials])
                pattern_scores[pattern] = field * (1 + commit)  # Combined score
        
        if pattern_scores:
            result.best_pattern = max(pattern_scores, key=pattern_scores.get)
            result.worst_pattern = min(pattern_scores, key=pattern_scores.get)
    
    def print_summary(self, result: ClusterResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("SPATIAL CLUSTERING RESULTS")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Pattern':<20} {'Spacing':<10} {'Distance':<10} {'Coupling':<10} {'Q1 Field':<10} {'Commit':<10}")
        print("-"*70)
        
        for key in sorted(result.summary.keys()):
            stats = result.summary[key]
            pattern = stats['pattern']
            spacing = stats['spacing_um']
            distance = stats['mean_distance_um']
            coupling = stats['mean_coupling']
            field = stats['em_field_mean']
            commit = stats['commit_rate']
            
            print(f"{pattern:<20} {spacing:<10.1f} {distance:<10.2f} {coupling:<10.3f} {field:<10.1f} {commit:<10.0%}")
        
        print("\n" + "="*70)
        print("PATTERN RANKING")
        print("="*70)
        
        print(f"\n  Best pattern:  {result.best_pattern}")
        print(f"  Worst pattern: {result.worst_pattern}")
        
        # Summary by pattern only
        print("\n  By pattern (averaged over spacings):")
        for pattern in self.patterns:
            pattern_trials = [t for t in result.trials if t.condition.pattern == pattern]
            if pattern_trials:
                field = np.mean([t.peak_em_field_kT for t in pattern_trials])
                commit = np.mean([1 if t.committed else 0 for t in pattern_trials])
                print(f"    {pattern:<15}: Q1={field:.1f}kT, commit={commit:.0%}")
        
        if result.best_pattern == 'clustered':
            print("\n  ✓ CLUSTERED GEOMETRY OPTIMAL")
            print("    Validates proximity requirement for quantum cooperativity")
        else:
            print(f"\n  ⚠ Best pattern is {result.best_pattern}, not clustered")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: ClusterResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # Colors by pattern
        colors = {
            'clustered': '#2ca02c',
            'linear': '#1f77b4',
            'distributed': '#ff7f0e',
            'random': '#9467bd'
        }
        
        # === Panel A: EM Field vs Mean Distance ===
        ax1 = axes[0]
        
        for key, stats in result.summary.items():
            pattern = stats['pattern']
            distance = stats['mean_distance_um']
            field = stats['em_field_mean']
            field_std = stats['em_field_std']
            
            ax1.errorbar(distance, field, yerr=field_std, fmt='o', 
                        color=colors.get(pattern, 'gray'), markersize=10,
                        capsize=4, label=pattern if key.endswith('0.5µm') or key.endswith('1.0µm') else '')
        
        ax1.set_xlabel('Mean Inter-synapse Distance (µm)', fontsize=11)
        ax1.set_ylabel('Q1 EM Field (kT)', fontsize=11)
        ax1.set_title('A. Field vs Distance', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        
        # === Panel B: Coupling Strength vs Field ===
        ax2 = axes[1]
        
        for key, stats in result.summary.items():
            pattern = stats['pattern']
            coupling = stats['mean_coupling']
            field = stats['em_field_mean']
            
            ax2.scatter(coupling, field, s=100, color=colors.get(pattern, 'gray'),
                       label=pattern if key.endswith('0.5µm') else '', alpha=0.7)
        
        ax2.set_xlabel('Mean Coupling Strength', fontsize=11)
        ax2.set_ylabel('Q1 EM Field (kT)', fontsize=11)
        ax2.set_title('B. Field vs Coupling', fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        
        # === Panel C: Commitment Rate by Pattern ===
        ax3 = axes[2]
        
        x = np.arange(len(self.patterns))
        width = 0.8 / len(self.spacings_um)
        
        for i, spacing in enumerate(self.spacings_um):
            commits = []
            for pattern in self.patterns:
                key = f"{pattern}_{spacing:.1f}µm"
                commits.append(result.summary.get(key, {}).get('commit_rate', 0))
            
            offset = (i - len(self.spacings_um)/2 + 0.5) * width
            bars = ax3.bar(x + offset, commits, width, label=f'{spacing:.1f}µm', alpha=0.8)
        
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Pattern', fontsize=11)
        ax3.set_ylabel('Commitment Rate', fontsize=11)
        ax3.set_title('C. Commitment by Geometry', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.patterns, rotation=20)
        ax3.set_ylim(0, 1.1)
        ax3.legend(title='Spacing', loc='upper right', fontsize=8)
        
        plt.suptitle('Spatial Clustering: Geometry Effects on Quantum Cooperativity', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'spatial_clustering.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: ClusterResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'spatial_clustering',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'best_pattern': result.best_pattern,
            'worst_pattern': result.worst_pattern,
            'summary': result.summary,
            'patterns_tested': self.patterns,
            'spacings_tested': self.spacings_um,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: ClusterResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'best_pattern': result.best_pattern,
            'worst_pattern': result.worst_pattern,
            'clustered_optimal': result.best_pattern == 'clustered',
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running spatial clustering experiment (quick mode)...")
    exp = SpatialClusteringExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()