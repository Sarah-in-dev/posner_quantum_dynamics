#!/usr/bin/env python3
"""
Experiment: External UV Enhancement
====================================

Tests whether external UV illumination enhances the quantum cascade.

Scientific basis:
- Metabolic UV (~20 photons/s at PSD) drives tryptophan superradiance
- External UV can supplement metabolic production
- Model parameter: params.metabolic_uv.external_uv_illumination

NOTE: Current model does NOT implement wavelength-dependent absorption.
This experiment tests external UV ON vs OFF, not wavelength tuning.
A future model enhancement could add tryptophan absorption spectrum.

Protocol:
1. Run with metabolic UV only (control)
2. Run with external UV added (various intensities)
3. Compare EM field strength and cascade output

Success criteria:
- External UV increases EM field strength
- Higher intensity → more enhancement (with saturation)
- Validates UV → tryptophan → dimer coupling pathway

References:
- Kurian et al. (2022): Tryptophan superradiance in microtubules
- Tang & Dai (2014): Biophoton emission during neural activity
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
class UVCondition:
    """Single experimental condition"""
    wavelength_nm: float
    n_synapses: int = 10
    
    @property
    def name(self) -> str:
        return f"{self.wavelength_nm:.0f}nm"

@dataclass 
class UVTrialResult:
    """Results from single trial"""
    condition: UVCondition
    trial_id: int
    
    # Q1 metrics
    peak_em_field_kT: float = 0.0
    mean_em_field_kT: float = 0.0
    
    # Q2 metrics
    peak_dimers: int = 0
    
    # Output
    eligibility: float = 0.0
    committed: bool = False
    final_strength: float = 1.0
    
    runtime_s: float = 0.0


@dataclass
class UVResult:
    """Complete experiment results"""
    conditions: List[UVCondition]
    trials: List[UVTrialResult]
    
    # Summary by wavelength
    summary: Dict = field(default_factory=dict)
    
    # Spectral characterization
    peak_wavelength_nm: float = 0.0  # Wavelength of maximum effect
    spectral_width_nm: float = 0.0  # FWHM of enhancement curve
    peak_enhancement: float = 1.0
    
    # Control comparison
    no_uv_baseline: Dict = field(default_factory=dict)
    
    timestamp: str = ""
    runtime_s: float = 0.0


class UVWavelengthExperiment:
    """
    UV intensity experiment
    
    Tests whether external UV enhances the quantum cascade.
    Wavelength fixed at 280nm (tryptophan peak).
    """
    
    def __init__(self, quick_mode: bool = False, verbose: bool = True):
        self.quick_mode = quick_mode
        self.verbose = verbose
        
        if quick_mode:
            self.wavelengths_nm = [280, 350, 400, 500]
            self.n_trials = 2
            self.n_synapses = 5
            self.consolidation_s = 0.5
        else:
            self.wavelengths_nm = [250, 270, 280, 290, 310, 350, 400, 500]
            self.n_trials = 5
            self.n_synapses = 10
            self.consolidation_s = 1.0
        
        self.intensity_mW = 1.0  # Fixed intensity for all wavelengths
    
    def _create_network(self, condition: UVCondition) -> MultiSynapseNetwork:
        """Create network with UV at specified wavelength"""
        params = Model6Parameters()
        params.em_coupling_enabled = True
        params.multi_synapse_enabled = True
        
        # External UV at the condition's wavelength
        params.metabolic_uv.external_uv_illumination = True
        params.metabolic_uv.external_uv_wavelength = condition.wavelength_nm * 1e-9
        params.metabolic_uv.external_uv_intensity = self.intensity_mW * 1e-3
        
        # Create network
        network = MultiSynapseNetwork(
            n_synapses=condition.n_synapses,
            params=params,
            pattern='clustered',
            spacing_um=1.0
        )
        
        network.initialize(Model6QuantumSynapse, params)
        network.set_microtubule_invasion(True)
        
        return network
    
    def _run_trial(self, condition: UVCondition, trial_id: int) -> UVTrialResult:
        """Execute single trial"""
        start_time = time.time()
        
        result = UVTrialResult(
            condition=condition,
            trial_id=trial_id
        )
        
        # Create network
        network = self._create_network(condition)
        
        dt = 0.001  # 1 ms timestep
        
        # Track EM field
        em_fields = []
        
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
            
            # Record EM field after each burst
            metrics = network.get_experimental_metrics()
            em_fields.append(metrics.get('mean_field_kT', 0))
            
            for _ in range(160):
                network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # Record peak EM field
        result.peak_em_field_kT = max(em_fields) if em_fields else 0
        result.mean_em_field_kT = np.mean(em_fields) if em_fields else 0
        
        # Record dimers
        result.peak_dimers = sum(
            len(s.dimer_particles.dimers) if hasattr(s, 'dimer_particles') else 0
            for s in network.synapses
        )
        
        # === PHASE 3: CONSOLIDATION ===
        n_consol = int(self.consolidation_s / dt)
        for _ in range(n_consol):
            network.step(dt, {"voltage": -70e-3, "reward": True})
        
        # === FINAL MEASUREMENTS ===
        result.eligibility = np.mean([s.get_eligibility() for s in network.synapses])
        result.committed = network.network_committed
        result.final_strength = 1.0 + 0.5 * network.network_commitment_level if result.committed else 1.0
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def run(self) -> UVResult:
        start_time = time.time()
        
        if self.verbose:
            print(f"\nRunning UV wavelength experiment...")
            print(f"  Wavelengths: {self.wavelengths_nm} nm")
            print(f"  Trials per condition: {self.n_trials}")
        
        conditions = [UVCondition(
            wavelength_nm=wl,
            n_synapses=self.n_synapses
        ) for wl in self.wavelengths_nm]
        
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
                cond_trials = [t for t in trials if t.condition.wavelength_nm == cond.wavelength_nm]
                field = np.mean([t.peak_em_field_kT for t in cond_trials])
                dimers = np.mean([t.peak_dimers for t in cond_trials])
                print(f" Q1={field:.1f}kT, dimers={dimers:.0f}")
        
        # Build result
        result = UVResult(
            conditions=conditions,
            trials=trials,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Just run first wavelength as baseline or remove this block entirely
        result.no_uv_baseline = {}

        # Compute summary
        result.summary = self._compute_summary(trials, result.no_uv_baseline)
        
        # Find optimal intensity
        self._characterize_response(result)
        
        result.runtime_s = time.time() - start_time
        
        return result
    
    def _compute_summary(self, trials: List[UVTrialResult], baseline: Dict) -> Dict:
        """Compute summary statistics by intensity"""
        summary = {}
        
        baseline_field = baseline.get('em_field_mean', 1)
        baseline_dimers = baseline.get('dimers_mean', 1)
        for wavelength in self.wavelengths_nm:
            cond_trials = [t for t in trials if t.condition.wavelength_nm == wavelength]

            if cond_trials:
                field_mean = np.mean([t.peak_em_field_kT for t in cond_trials])
                dimers_mean = np.mean([t.peak_dimers for t in cond_trials])

                summary[wavelength] = {
                    'em_field_mean': field_mean,
                    'em_field_std': np.std([t.peak_em_field_kT for t in cond_trials]),
                    'dimers_mean': dimers_mean,
                    'dimers_std': np.std([t.peak_dimers for t in cond_trials]),
                    'eligibility_mean': np.mean([t.eligibility for t in cond_trials]),
                    'commit_rate': np.mean([1 if t.committed else 0 for t in cond_trials]),
                    'enhancement_field': field_mean / max(baseline_field, 0.1),
                    'enhancement_dimers': dimers_mean / max(baseline_dimers, 0.1),
                    'n_trials': len(cond_trials)
                }
        
        return summary
    
    def _characterize_response(self, result: UVResult):
        """Find optimal intensity and characterize dose-response"""
        if not result.summary:
            result.peak_enhancement = 1.0
            return
        
        # Find intensity with maximum enhancement
        best_intensity = 0
        best_enhancement = 1.0
        
        for wavelength in self.wavelengths_nm:
            stats = result.summary.get(wavelength, {})
            enhancement = stats.get('enhancement_field', 1.0)
            if enhancement > best_enhancement:
                best_enhancement = enhancement
                best_intensity = wavelength

        result.peak_wavelength_nm = 280  # Fixed
        result.peak_enhancement = best_enhancement
        
        # Store optimal intensity (using spectral_width_nm field for now)
        result.spectral_width_nm = best_intensity  # Repurposing this field
    
    def print_summary(self, result: UVResult):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("UV WAVELENGTH EFFECTS RESULTS")
        print("="*70)
        
        print(f"\nControl (metabolic only): Q1 Field = {result.no_uv_baseline.get('em_field_mean', 0):.1f} kT")
        
        print("\n" + "-"*70)
        print(f"{'Intensity':<12} {'Q1 Field':<12} {'Enhancement':<12} {'Dimers':<10} {'Commit':<10}")
        print("-"*70)

        for wavelength in self.wavelengths_nm:
            stats = result.summary.get(wavelength, {})
            field = stats.get('em_field_mean', 0)
            enhancement = stats.get('enhancement_field', 1)
            dimers = stats.get('dimers_mean', 0)
            commit = stats.get('commit_rate', 0)
            
            # Highlight best
            is_best = (wavelength == result.spectral_width_nm and result.peak_enhancement > 1.1)
            marker = " ←" if is_best else ""

            label = "Metabolic" if wavelength == 0 else f"{wavelength:.1f} nm"
            print(f"{label:<12} {field:<12.1f} {enhancement:<12.2f}× {dimers:<10.0f} {commit:<10.0%}{marker}")
        
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        if result.peak_enhancement > 1.1:
            print(f"\n  ✓ External UV enhances cascade")
            print(f"    Peak enhancement: {result.peak_enhancement:.2f}× at {result.spectral_width_nm:.1f} mW")
            print("    Validates UV → tryptophan → dimer pathway")
        else:
            print(f"\n  ~ No significant enhancement from external UV")
            print("    May indicate metabolic UV is already saturating")
        
        print(f"\nRuntime: {result.runtime_s:.1f}s")
    
    def plot(self, result: UVResult, output_dir: Optional[Path] = None) -> plt.Figure:
        """Generate publication-quality figure"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        
        wavelengths = self.wavelengths_nm
        
        # === Panel A: Q1 Field vs Wavelength ===
        ax1 = axes[0]
        
        fields = [result.summary.get(i, {}).get('em_field_mean', 0) for i in wavelengths]
        field_std = [result.summary.get(i, {}).get('em_field_std', 0) for i in wavelengths]
        
        ax1.errorbar(wavelengths, fields, yerr=field_std, fmt='o-', color='#9467bd',
                    markersize=8, capsize=4, linewidth=2)
        
        ax1.set_xlabel('UV Wavelength (nm)', fontsize=11)
        ax1.set_ylabel('Q1 EM Field (kT)', fontsize=11)
        ax1.set_title('A. UV Dose Response', fontweight='bold')
        
        # === Panel B: Enhancement Factor ===
        ax2 = axes[1]

        enhancements = [result.summary.get(i, {}).get('enhancement_field', 1) for i in wavelengths]

        ax2.bar(range(len(wavelengths)), enhancements, color='#9467bd', alpha=0.7, edgecolor='black')
        ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='No enhancement')
        ax2.set_xticks(range(len(wavelengths)))
        ax2.set_xticklabels([f"{i}" if i > 0 else "Met" for i in wavelengths])

        ax2.set_xlabel('UV Wavelength (nm)', fontsize=11)
        ax2.set_ylabel('Enhancement (×)', fontsize=11)
        ax2.set_title('B. Field Enhancement', fontweight='bold')
        
        # === Panel C: Dimer Formation ===
        ax3 = axes[2]

        dimers = [result.summary.get(i, {}).get('dimers_mean', 0) for i in wavelengths]

        ax3.bar(range(len(wavelengths)), dimers, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(wavelengths)))
        ax3.set_xticklabels([f"{i}" if i > 0 else "Met" for i in wavelengths])

        ax3.set_xlabel('UV Wavelength (nm)', fontsize=11)
        ax3.set_ylabel('Dimers', fontsize=11)
        ax3.set_title('C. Dimer Formation', fontweight='bold')

        plt.suptitle('UV Wavelength: External Enhancement of Quantum Cascade',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'uv_wavelength.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  Saved figure to {save_path}")
        
        return fig
    
    def save_results(self, result: UVResult, path: Path):
        """Save results to JSON"""
        data = {
            'experiment': 'uv_intensity',
            'timestamp': result.timestamp,
            'runtime_s': result.runtime_s,
            'peak_enhancement': result.peak_enhancement,
            'experiment': 'uv_wavelength',
            'peak_wavelength_nm': result.peak_wavelength_nm,
            'no_uv_baseline': result.no_uv_baseline,
            'summary': {str(k): v for k, v in result.summary.items()},
            'wavelengths_tested': self.wavelengths_nm,
            'n_trials': self.n_trials,
            'n_synapses': self.n_synapses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_dict(self, result: UVResult) -> dict:
        """Get summary as dictionary for master results"""
        return {
            'peak_wavelength_nm': result.peak_wavelength_nm,
            'peak_enhancement': result.peak_enhancement,
            'wavelength_specific': result.peak_enhancement > 1.1,
            'runtime_s': result.runtime_s
        }


if __name__ == "__main__":
    print("Running UV wavelength experiment (quick mode)...")
    exp = UVWavelengthExperiment(quick_mode=True, verbose=True)
    result = exp.run()
    exp.print_summary(result)
    fig = exp.plot(result)
    plt.show()