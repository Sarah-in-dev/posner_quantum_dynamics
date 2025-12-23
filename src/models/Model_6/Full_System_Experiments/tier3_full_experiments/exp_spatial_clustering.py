#!/usr/bin/env python3
"""
Experiment: Spatial Clustering Effects
=======================================

Tests how synapse geometry affects quantum cooperativity.

Scientific basis:
- Cross-synapse entanglement requires proximity
- Clustered synapses form unified quantum networks
- Distributed synapses have fragmented networks

Predictions:
- Clustered: Few clusters, many cross-synapse bonds, high coverage
- Distributed: Many clusters, few cross-synapse bonds, low coverage

Protocol:
1. Create networks with identical synapse count but different geometries
2. Apply identical stimulation protocol
3. Measure cross-synapse entanglement, cluster count, network coverage

Success criteria:
- Clustered has fewer clusters (unified network)
- Clustered has more cross-synapse bonds
- Validates collective quantum effects require proximity
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
    
    # Spatial entanglement metrics (KEY METRICS)
    n_clusters: int = 0
    cross_synapse_bonds: int = 0
    within_synapse_bonds: int = 0
    network_coverage: float = 0.0
    n_entangled_network: int = 0
    
    # Field metrics
    q2_field_kT: float = 0.0
    q1_field_kT: float = 0.0
    field_coherence: float = 0.0
    
    # Dimer metrics
    peak_dimers: int = 0
    entanglement_fraction: float = 0.0
    
    # Output
    eligibility: float = 0.0
    committed: bool = False
    commitment_time_s: float = 0.0
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
        q2_fields = []
        commitment_time = None
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION (theta-burst with dopamine) ===
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):
                    network.step(dt, {"voltage": -70e-3, "reward": True})
            
            # Record Q2 field
            metrics = network.get_experimental_metrics()
            q2_fields.append(metrics.get('q2_field_kT', 0))
            
            # Check for commitment
            if network.network_committed and commitment_time is None:
                commitment_time = burst * 0.2
            
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for step in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
            if network.network_committed and commitment_time is None:
                commitment_time = 1.0 + step * dt
        
        # === FINAL MEASUREMENTS ===
        metrics = network.get_experimental_metrics()
        
        # Q2 field and Q1 field
        result.q2_field_kT = max(q2_fields) if q2_fields else 0
        result.q1_field_kT = metrics.get('mean_field_kT', 0)
        
        # Cross-synapse entanglement
        result.cross_synapse_bonds = metrics.get('cross_synapse_bonds', 0)
        result.within_synapse_bonds = metrics.get('within_synapse_bonds', 0)
        result.n_entangled_network = metrics.get('n_entangled_network', 0)
        
        # Count clusters using union-find
        tracker = network.entanglement_tracker
        if tracker.all_dimers:
            parent = {d['global_id']: d['global_id'] for d in tracker.all_dimers}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            for bond in tracker.entanglement_bonds:
                if bond[0] in parent and bond[1] in parent:
                    px, py = find(bond[0]), find(bond[1])
                    if px != py:
                        parent[px] = py
            
            roots = set(find(d['global_id']) for d in tracker.all_dimers)
            result.n_clusters = len(roots)
            
            # Largest cluster coverage
            clusters = {}
            for d in tracker.all_dimers:
                root = find(d['global_id'])
                clusters[root] = clusters.get(root, 0) + 1
            largest = max(clusters.values()) if clusters else 0
            result.network_coverage = largest / len(tracker.all_dimers)
        else:
            result.n_clusters = 0
            result.network_coverage = 0.0
        
        # Dimer metrics
        result.peak_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # Commitment
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
                clusters = np.mean([t.n_clusters for t in cond_trials])
                cross = np.mean([t.cross_synapse_bonds for t in cond_trials])
                coverage = np.mean([t.network_coverage for t in cond_trials])
                commit = np.mean([1 if t.committed else 0 for t in cond_trials])
                print(f" clusters={clusters:.0f}, cross={cross:.0f}, coverage={coverage:.0%}, commit={commit:.0%}")
        
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
                        'n_clusters_mean': np.mean([t.n_clusters for t in cond_trials]),
                        'cross_bonds_mean': np.mean([t.cross_synapse_bonds for t in cond_trials]),
                        'within_bonds_mean': np.mean([t.within_synapse_bonds for t in cond_trials]),
                        'coverage_mean': np.mean([t.network_coverage for t in cond_trials]),
                        'q2_field_mean': np.mean([t.q2_field_kT for t in cond_trials]),
                        'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                        'n_trials': len(cond_trials)
                    }
        
        return summary
    
    def _rank_patterns(self, result: ClusterResult):
        """Rank patterns by performance (fewer clusters = better connectivity)"""
        pattern_scores = {}
        
        for pattern in self.patterns:
            pattern_trials = [t for t in result.trials if t.condition.pattern == pattern]
            if pattern_trials:
                # Score: more cross-bonds and fewer clusters = better
                cross = np.mean([t.cross_synapse_bonds for t in pattern_trials])
                clusters = np.mean([t.n_clusters for t in pattern_trials])
                coverage = np.mean([t.network_coverage for t in pattern_trials])
                # Higher cross-bonds, lower clusters, higher coverage = better
                pattern_scores[pattern] = cross * coverage / max(clusters, 1)
        
        if pattern_scores:
            result.best_pattern = max(pattern_scores, key=pattern_scores.get)
            result.worst_pattern = min(pattern_scores, key=pattern_scores.get)
    
    def print_summary(self, result: ClusterResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("SPATIAL CLUSTERING RESULTS")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Pattern':<12} {'Spacing':<8} {'Clusters':<10} {'CrossBonds':<12} {'Coverage':<10} {'Commit':<8}")
        print("-"*70)
        
        for key in sorted(result.summary.keys()):
            stats = result.summary[key]
            print(f"{stats['pattern']:<12} {stats['spacing_um']:<8.1f} "
                  f"{stats['n_clusters_mean']:<10.0f} {stats['cross_bonds_mean']:<12.0f} "
                  f"{stats['coverage_mean']:<10.0%} {stats['commit_rate']:<8.0%}")
        
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
                clusters = np.mean([t.n_clusters for t in pattern_trials])
                cross = np.mean([t.cross_synapse_bonds for t in pattern_trials])
                coverage = np.mean([t.network_coverage for t in pattern_trials])
                print(f"    {pattern:<15}: clusters={clusters:.0f}, cross_bonds={cross:.0f}, coverage={coverage:.0%}")
        
        if result.best_pattern == 'clustered':
            print("\n  ✓ CLUSTERED GEOMETRY OPTIMAL")
            print("    Validates proximity requirement for quantum cooperativity")
        else:
            print(f"\n  ⚠ Best pattern is {result.best_pattern}, not clustered")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: ClusterResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # Collect data for plotting
        distances_c = []
        distances_d = []
        couplings_c = []
        couplings_d = []
        clusters_c = []
        clusters_d = []
        cross_c = []
        cross_d = []
        
        for spacing in self.spacings_um:
            key_c = f"clustered_{spacing:.1f}µm"
            key_d = f"distributed_{spacing:.1f}µm"
            
            if key_c in result.summary:
                distances_c.append(result.summary[key_c]['mean_distance_um'])
                couplings_c.append(result.summary[key_c]['mean_coupling'])
                clusters_c.append(result.summary[key_c]['n_clusters_mean'])
                cross_c.append(result.summary[key_c]['cross_bonds_mean'])
            
            if key_d in result.summary:
                distances_d.append(result.summary[key_d]['mean_distance_um'])
                couplings_d.append(result.summary[key_d]['mean_coupling'])
                clusters_d.append(result.summary[key_d]['n_clusters_mean'])
                cross_d.append(result.summary[key_d]['cross_bonds_mean'])
        
        # === Panel A: Clusters vs Distance ===
        ax1 = axes[0]
        if distances_c and clusters_c:
            ax1.scatter(distances_c, clusters_c, c='green', s=100, label='clustered')
            ax1.errorbar(distances_c, clusters_c, fmt='none', c='green', alpha=0.5)
        if distances_d and clusters_d:
            ax1.scatter(distances_d, clusters_d, c='orange', s=100, label='distributed')
            ax1.errorbar(distances_d, clusters_d, fmt='none', c='orange', alpha=0.5)
        
        ax1.set_xlabel('Mean Inter-synapse Distance (µm)', fontsize=11)
        ax1.set_ylabel('Number of Clusters', fontsize=11)
        ax1.set_title('A. Network Fragmentation', fontweight='bold')
        ax1.legend(loc='upper left')
        
        # === Panel B: Cross-synapse bonds vs Coupling ===
        ax2 = axes[1]
        if couplings_c and cross_c:
            ax2.scatter(couplings_c, cross_c, c='green', s=100, label='clustered')
        if couplings_d and cross_d:
            ax2.scatter(couplings_d, cross_d, c='orange', s=100, label='distributed')
        
        ax2.set_xlabel('Mean Coupling Strength', fontsize=11)
        ax2.set_ylabel('Cross-Synapse Bonds', fontsize=11)
        ax2.set_title('B. Cross-Synapse Entanglement', fontweight='bold')
        ax2.legend(loc='upper left')
        
        # === Panel C: Network Coverage by Pattern ===
        ax3 = axes[2]
        
        x = np.arange(len(self.patterns))
        width = 0.8 / len(self.spacings_um)
        
        for i, spacing in enumerate(self.spacings_um):
            coverages = []
            for pattern in self.patterns:
                key = f"{pattern}_{spacing:.1f}µm"
                coverages.append(result.summary.get(key, {}).get('coverage_mean', 0))
            
            offset = (i - len(self.spacings_um)/2 + 0.5) * width
            ax3.bar(x + offset, coverages, width, label=f'{spacing:.1f}µm', alpha=0.8)
        
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Pattern', fontsize=11)
        ax3.set_ylabel('Network Coverage', fontsize=11)
        ax3.set_title('C. Unified Network Coverage', fontweight='bold')
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