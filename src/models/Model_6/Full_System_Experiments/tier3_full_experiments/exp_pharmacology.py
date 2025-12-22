#!/usr/bin/env python3
"""
Experiment: Pharmacological Dissection
=======================================

Separates Q1 (tryptophan/EM), Q2 (dimer coherence), and classical contributions.

Interventions:
1. Control - Full cascade intact
2. APV (NMDA blocker) - Blocks calcium → no dimers → no Q2 → no classical
3. Nocodazole (MT disruptor) - Blocks Q1 (tryptophan network) → reduced EM enhancement
4. Isoflurane (anesthetic) - Blocks tryptophan superradiance → reduced Q1 field

Predictions:
- Control: Full commitment, strength ~1.5×
- APV: No commitment (no calcium → no dimers)
- Nocodazole: Reduced commitment (no EM enhancement)
- Isoflurane: Dose-dependent reduction in Q1 field

This experiment validates that both quantum layers (Q1 and Q2) are necessary
for full plasticity, and that classical mechanisms alone are insufficient.
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
class PharmCondition:
    """Single pharmacological condition"""
    name: str
    description: str
    
    # Intervention parameters
    nmda_blocked: bool = False      # APV
    mt_invaded: bool = True         # Nocodazole sets to False
    isoflurane_pct: float = 0.0     # 0-100%
    
    n_synapses: int = 10


@dataclass 
class PharmTrialResult:
    """Results from single trial"""
    condition: PharmCondition
    trial_id: int
    
    # Q1 metrics (tryptophan/EM)
    peak_em_field_kT: float = 0.0
    
    # Q2 metrics (dimer coherence)
    peak_dimers: int = 0
    mean_singlet_prob: float = 1.0
    eligibility: float = 0.0
    
    # Gate output
    committed: bool = False
    commitment_level: float = 0.0
    
    # Classical outcome
    final_strength: float = 1.0
    
    runtime_s: float = 0.0


@dataclass
class PharmResult:
    """Complete experiment results"""
    conditions: List[PharmCondition]
    trials: List[PharmTrialResult]
    
    # Summary by condition
    summary: Dict = field(default_factory=dict)
    
    # Validation flags
    apv_blocked: bool = False
    nocodazole_reduced: bool = False
    isoflurane_dose_dependent: bool = False
    dissection_valid: bool = False
    
    timestamp: str = ""
    runtime_s: float = 0.0


class PharmacologyExperiment:
    """
    Pharmacological dissection experiment
    
    Tests that blocking specific cascade components produces expected effects.
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        # Experimental parameters
        if quick_mode:
            self.n_trials = 2
            self.n_synapses = 5
            self.stim_duration_s = 0.5
            self.consolidation_s = 0.5
            self.isoflurane_doses = [0, 50, 100]
        else:
            self.n_trials = 5
            self.n_synapses = 10
            self.stim_duration_s = 1.0
            self.consolidation_s = 1.0
            self.isoflurane_doses = [0, 25, 50, 75, 100]
    
    def _build_conditions(self) -> List[PharmCondition]:
        """Build list of pharmacological conditions"""
        conditions = []
        
        # 1. Control - full cascade
        conditions.append(PharmCondition(
            name='Control',
            description='Full cascade intact',
            nmda_blocked=False,
            mt_invaded=True,
            isoflurane_pct=0.0,
            n_synapses=self.n_synapses
        ))
        
        # 2. APV - NMDA blocker
        conditions.append(PharmCondition(
            name='APV',
            description='NMDA blocked → no calcium → no dimers',
            nmda_blocked=True,
            mt_invaded=True,
            isoflurane_pct=0.0,
            n_synapses=self.n_synapses
        ))
        
        # 3. Nocodazole - MT disruptor
        conditions.append(PharmCondition(
            name='Nocodazole',
            description='MT disrupted → no tryptophan network → reduced Q1',
            nmda_blocked=False,
            mt_invaded=False,  # MT- naive synapses
            isoflurane_pct=0.0,
            n_synapses=self.n_synapses
        ))
        
        # 4. Isoflurane doses
        for dose in self.isoflurane_doses:
            if dose > 0:  # Skip 0 (that's control)
                conditions.append(PharmCondition(
                    name=f'Isoflurane_{dose}%',
                    description=f'Isoflurane {dose}% → reduced tryptophan superradiance',
                    nmda_blocked=False,
                    mt_invaded=True,
                    isoflurane_pct=dose,
                    n_synapses=self.n_synapses
                ))
        
        return conditions
    
    def _create_network(self, condition: PharmCondition) -> MultiSynapseNetwork:
        """Create network with specified pharmacological condition"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        # APV blocks NMDA
        if condition.nmda_blocked:
            params.calcium.nmda_blocked = True
        
        # Isoflurane reduces tryptophan quantum yield
        if condition.isoflurane_pct > 0:
            # Scale tryptophan quantum yield by (1 - dose/100)
            reduction = 1.0 - condition.isoflurane_pct / 100.0
            params.tryptophan.quantum_yield_free *= reduction
        
        # Create network
        network = MultiSynapseNetwork(
            n_synapses=condition.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(condition.mt_invaded)
        
        return network
    
    def _run_trial(self, condition: PharmCondition, trial_id: int) -> PharmTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = PharmTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        network = self._create_network(condition)
        dt = 0.001
        
        # Track peak values
        peak_field = 0.0
        peak_dimers = 0
        
        # === PHASE 1: BASELINE (100ms) ===
        for _ in range(100):
            network.step(dt, {"voltage": -70e-3, "reward": False})
        
        # === PHASE 2: STIMULATION + DOPAMINE (theta-burst for gate) ===
        for burst in range(5):
            for spike in range(4):
                for _ in range(2):  # 2ms depolarization
                    network.step(dt, {"voltage": -10e-3, "reward": True})
                for _ in range(8):  # 8ms rest
                    network.step(dt, {"voltage": -70e-3, "reward": True})
            for _ in range(160):  # 160ms inter-burst interval
                network.step(dt, {"voltage": -70e-3, "reward": True})
                
            # Track peaks during stimulation
            metrics = network.get_experimental_metrics()
            if metrics.get('mean_field_kT', 0) > peak_field:
                peak_field = metrics.get('mean_field_kT', 0)
            
            current_dimers = sum(
                len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
                for s in network.synapses
            )
            peak_dimers = max(peak_dimers, current_dimers)
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # === FINAL MEASUREMENTS ===
        result.peak_em_field_kT = peak_field
        result.peak_dimers = peak_dimers
        
        # Dimer metrics
        all_ps = []
        for s in network.synapses:
            if hasattr(s, 'dimer_particles'):
                for d in s.dimer_particles.dimers:
                    all_ps.append(d.singlet_probability)
        result.mean_singlet_prob = np.mean(all_ps) if all_ps else 0.0
        
        # Eligibility
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        
        # Commitment
        result.committed = network.network_committed
        result.commitment_level = network.network_commitment_level
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> PharmResult:
        """Execute complete experiment"""
        start_time = time.time()
        
        conditions = self._build_conditions()
        trials = []
        
        for cond in conditions:
            if self.verbose:
                print(f"  {cond.name:<20}: ", end='', flush=True)
            
            for trial_id in range(self.n_trials):
                trial_result = self._run_trial(cond, trial_id)
                trials.append(trial_result)
                
                if self.verbose:
                    print(".", end='', flush=True)
            
            if self.verbose:
                cond_trials = [t for t in trials if t.condition.name == cond.name]
                mean_commit = np.mean([t.committed for t in cond_trials])
                mean_field = np.mean([t.peak_em_field_kT for t in cond_trials])
                print(f" commit={mean_commit:.0%}, Q1={mean_field:.1f}kT")
        
        # Build result
        result = PharmResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Compute summary
        result.summary = self._compute_summary(trials)
        
        # Validate predictions
        self._validate_predictions(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[PharmTrialResult]) -> Dict:
        """Compute summary statistics by condition"""
        summary = {}
        
        condition_names = set(t.condition.name for t in trials)
        
        for name in condition_names:
            cond_trials = [t for t in trials if t.condition.name == name]
            
            if cond_trials:
                summary[name] = {
                    'em_field_mean': np.mean([t.peak_em_field_kT for t in cond_trials]),
                    'em_field_std': np.std([t.peak_em_field_kT for t in cond_trials]),
                    'dimers_mean': np.mean([t.peak_dimers for t in cond_trials]),
                    'eligibility_mean': np.mean([t.eligibility for t in cond_trials]),
                    'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                    'strength_mean': np.mean([t.final_strength for t in cond_trials]),
                    'strength_std': np.std([t.final_strength for t in cond_trials]),
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def _validate_predictions(self, result: PharmResult):
        """Check if predictions are met"""
        ctrl = result.summary.get('Control', {})
        apv = result.summary.get('APV', {})
        noco = result.summary.get('Nocodazole', {})
        
        # APV should block completely
        result.apv_blocked = (
            apv.get('commit_rate', 1) < 0.2 and
            ctrl.get('commit_rate', 0) > 0.5
        )
        
        # Nocodazole should reduce (but may not fully block)
        result.nocodazole_reduced = (
            noco.get('em_field_mean', 100) < ctrl.get('em_field_mean', 0) * 0.7
        )
        
        # Isoflurane dose-response
        iso_names = [n for n in result.summary.keys() if 'Isoflurane' in n]
        if len(iso_names) >= 2:
            doses = sorted(iso_names)
            fields = [result.summary[d]['em_field_mean'] for d in doses]
            # Check monotonic decrease
            result.isoflurane_dose_dependent = all(
                fields[i] >= fields[i+1] for i in range(len(fields)-1)
            )
        
        # Overall validation
        result.dissection_valid = (
            result.apv_blocked and 
            (result.nocodazole_reduced or result.isoflurane_dose_dependent)
        )
    
    def print_summary(self, result: PharmResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("PHARMACOLOGICAL DISSECTION RESULTS")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Condition':<20} {'Q1 Field':<12} {'Dimers':<10} {'Commit':<10} {'Strength':<10}")
        print("-"*70)
        
        for name in ['Control', 'APV', 'Nocodazole'] + [n for n in result.summary.keys() if 'Isoflurane' in n]:
            stats = result.summary.get(name, {})
            field = stats.get('em_field_mean', 0)
            dimers = stats.get('dimers_mean', 0)
            commit = stats.get('commit_rate', 0)
            strength = stats.get('strength_mean', 1)
            
            print(f"{name:<20} {field:<12.1f} {dimers:<10.0f} {commit:<10.1%} {strength:<10.2f}×")
        
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        print(f"\n  APV blocks cascade:        {'✓' if result.apv_blocked else '✗'}")
        print(f"  Nocodazole reduces Q1:     {'✓' if result.nocodazole_reduced else '✗'}")
        print(f"  Isoflurane dose-response:  {'✓' if result.isoflurane_dose_dependent else '✗'}")
        
        if result.dissection_valid:
            print("\n  ✓ PHARMACOLOGICAL DISSECTION VALIDATES CASCADE ARCHITECTURE")
            print("    Both Q1 (tryptophan) and Q2 (dimer) layers are necessary")
        else:
            print("\n  ⚠ Partial validation - check individual conditions")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: PharmResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        # Colors
        colors = {
            'Control': '#2ca02c',
            'APV': '#d62728',
            'Nocodazole': '#ff7f0e',
        }
        for dose in self.isoflurane_doses:
            if dose > 0:
                colors[f'Isoflurane_{dose}%'] = plt.cm.Purples(0.3 + 0.7 * dose / 100)
        
        # === Panel A: Cascade component comparison ===
        ax1 = axes[0]
        
        main_conditions = ['Control', 'APV', 'Nocodazole']
        x = np.arange(len(main_conditions))
        width = 0.25
        
        fields = [result.summary.get(c, {}).get('em_field_mean', 0) for c in main_conditions]
        dimers = [result.summary.get(c, {}).get('dimers_mean', 0) for c in main_conditions]
        commits = [result.summary.get(c, {}).get('commit_rate', 0) * 100 for c in main_conditions]  # Scale for visibility
        
        ax1.bar(x - width, fields, width, label='Q1 Field (kT)', color='#1f77b4', alpha=0.8)
        ax1.bar(x, dimers, width, label='Dimers (count)', color='#ff7f0e', alpha=0.8)
        ax1.bar(x + width, commits, width, label='Commit (%)', color='#2ca02c', alpha=0.8)
        
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('A. Cascade Components', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(main_conditions, fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        
        # === Panel B: Commitment rates ===
        ax2 = axes[1]
        
        all_conditions = ['Control', 'APV', 'Nocodazole'] + sorted([n for n in result.summary.keys() if 'Isoflurane' in n])
        commit_rates = [result.summary.get(c, {}).get('commit_rate', 0) for c in all_conditions]
        bar_colors = [colors.get(c, '#888888') for c in all_conditions]
        
        bars = ax2.bar(range(len(all_conditions)), commit_rates, color=bar_colors, alpha=0.8)
        
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Commitment Rate', fontsize=11)
        ax2.set_title('B. Pharmacological Block', fontweight='bold')
        ax2.set_xticks(range(len(all_conditions)))
        ax2.set_xticklabels([c.replace('Isoflurane_', 'Iso ') for c in all_conditions], 
                          rotation=45, ha='right', fontsize=9)
        ax2.set_ylim(0, 1.1)
        
        # === Panel C: Isoflurane dose-response ===
        ax3 = axes[2]
        
        iso_conditions = sorted([n for n in result.summary.keys() if 'Isoflurane' in n])
        if iso_conditions:
            doses = [0] + [int(n.split('_')[1].replace('%', '')) for n in iso_conditions]
            fields = [result.summary['Control']['em_field_mean']] + [result.summary[n]['em_field_mean'] for n in iso_conditions]
            commits = [result.summary['Control']['commit_rate']] + [result.summary[n]['commit_rate'] for n in iso_conditions]
            
            ax3.plot(doses, fields, 'o-', color='#1f77b4', markersize=8, linewidth=2, label='Q1 Field (kT)')
            ax3.set_ylabel('Q1 Field (kT)', fontsize=11, color='#1f77b4')
            ax3.tick_params(axis='y', labelcolor='#1f77b4')
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(doses, commits, 's-', color='#2ca02c', markersize=8, linewidth=2, label='Commit Rate')
            ax3_twin.set_ylabel('Commit Rate', fontsize=11, color='#2ca02c')
            ax3_twin.tick_params(axis='y', labelcolor='#2ca02c')
            ax3_twin.set_ylim(-0.05, 1.1)
            
            ax3.set_xlabel('Isoflurane (%)', fontsize=11)
            ax3.set_title('C. Anesthetic Dose-Response', fontweight='bold')
            
            # Combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Isoflurane data\nnot available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
        
        plt.suptitle('Pharmacological Dissection: Q1 + Q2 + Classical Cascade', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'pharmacology.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: PharmResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'pharmacology',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'apv_blocked': result.apv_blocked,
            'nocodazole_reduced': result.nocodazole_reduced,
            'isoflurane_dose_dependent': result.isoflurane_dose_dependent,
            'dissection_valid': result.dissection_valid,
            'summary': result.summary,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: PharmResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'apv_blocked': result.apv_blocked,
            'nocodazole_reduced': result.nocodazole_reduced,
            'isoflurane_dose_dependent': result.isoflurane_dose_dependent,
            'dissection_valid': result.dissection_valid,
            'control_commit_rate': result.summary.get('Control', {}).get('commit_rate', 0),
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running pharmacology experiment (quick mode)...")
    exp = PharmacologyExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()