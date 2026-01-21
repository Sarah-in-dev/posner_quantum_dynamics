"""
Two-Region Cross-Brain Quantum Coupling POC
============================================

PROOF OF CONCEPT:
Shows that quantum coherent emission from one brain region can
modulate dimer formation in a distant region via myelin waveguides.

ARCHITECTURE:
------------
Region A (Source)                    Region B (Target)
┌─────────────────┐                  ┌─────────────────┐
│  Synapses       │                  │  Synapses       │
│  ↓              │                  │  ↑              │
│  Tryptophan SR  │──── Myelin ────→│  Photon         │
│  ↓              │    Waveguide     │  Reception      │
│  Photon         │                  │  ↓              │
│  Emission       │                  │  Enhanced       │
│                 │                  │  Dimer Formation│
└─────────────────┘                  └─────────────────┘

EXPERIMENT:
----------
1. Source region receives theta-burst stimulation
2. Tryptophan superradiance generates photons
3. Photons propagate through myelin waveguide (5mm, ~100 nodes)
4. Target region receives photons
5. TARGET DIMER FORMATION IS ENHANCED even without direct stimulation

This demonstrates the core mechanism for cross-brain quantum coupling.

PREDICTIONS:
-----------
1. Target dimer count increases when source is stimulated
2. Effect depends on waveguide transmission (blocked by demyelination)
3. Effect shows delay consistent with propagation + integration time
4. Magnitude scales with source activity level

Author: Sarah Davidson
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
from pathlib import Path

# Local modules
from photon_emission_module import PhotonEmissionTracker, PhotonEmissionParameters
from myelin_waveguide_module import (
    MyelinWaveguide, WaveguideConnection, WaveguideNetwork, MyelinWaveguideParameters
)
from photon_receiver_module import PhotonReceiver, PhotonReceiverParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLIFIED SYNAPSE MODEL (for POC - would use full Model6 in integration)
# =============================================================================

@dataclass
class SimplifiedSynapseState:
    """Minimal synapse state for POC"""
    n_tryptophans: int = 400
    em_field_kT: float = 5.0
    dimer_count: float = 0.0
    dimer_rate: float = 1.0  # Relative rate (1.0 = baseline)
    calcium_active: bool = False


class SimplifiedSynapse:
    """
    Simplified synapse model for POC
    
    Captures essential physics without full Model 6 complexity:
    - Tryptophan count increases with MT invasion during activity
    - EM field scales with sqrt(N_trp)
    - Dimer formation rate depends on calcium and EM field
    - Dimer count accumulates during stimulation
    """
    
    def __init__(self, synapse_id: int, region_id: int):
        self.synapse_id = synapse_id
        self.region_id = region_id
        
        # State
        self.state = SimplifiedSynapseState()
        self.time = 0.0
        
        # Parameters (from Model 6)
        self.n_trp_baseline = 400
        self.n_trp_active = 1200
        self.em_field_baseline_kT = 5.0
        self.em_field_active_kT = 22.0
        
        # Dimer formation
        self.k_dimer_base = 0.1  # dimers/s baseline
        self.dimer_decay = 0.001  # 1/s (very slow)
        
        # External modulation
        self.external_trp_multiplier = 1.0
        self.external_k_multiplier = 1.0
        
        # History
        self.history = {
            'time': [],
            'n_tryptophans': [],
            'em_field_kT': [],
            'dimer_count': [],
            'dimer_rate': [],
            'calcium_active': []
        }
    
    def set_external_modulation(self, trp_mult: float, k_mult: float):
        """Set modulation from external photon input"""
        self.external_trp_multiplier = trp_mult
        self.external_k_multiplier = k_mult
    
    def step(self, dt: float, calcium_active: bool = False) -> SimplifiedSynapseState:
        """
        Advance synapse state
        
        Parameters
        ----------
        dt : float
            Time step (s)
        calcium_active : bool
            Whether calcium influx is occurring (stimulation)
        """
        self.time += dt
        self.state.calcium_active = calcium_active
        
        # === TRYPTOPHAN COUNT ===
        if calcium_active:
            # MT invasion increases tryptophans
            self.state.n_tryptophans = int(self.n_trp_active * self.external_trp_multiplier)
        else:
            # Baseline
            self.state.n_tryptophans = int(self.n_trp_baseline * self.external_trp_multiplier)
        
        # === EM FIELD ===
        # Scales as sqrt(N) for partially coherent system
        n_eff = self.state.n_tryptophans
        self.state.em_field_kT = self.em_field_baseline_kT * np.sqrt(n_eff / self.n_trp_baseline)
        
        if calcium_active:
            # Additional boost during activity
            self.state.em_field_kT = min(30.0, self.state.em_field_kT * 1.5)
        
        # === DIMER FORMATION ===
        # Rate depends on calcium and external modulation
        if calcium_active:
            self.state.dimer_rate = self.k_dimer_base * 10.0 * self.external_k_multiplier
        else:
            self.state.dimer_rate = self.k_dimer_base * self.external_k_multiplier
        
        # Accumulate dimers
        self.state.dimer_count += self.state.dimer_rate * dt
        
        # Slow decay
        self.state.dimer_count *= (1.0 - self.dimer_decay * dt)
        
        # Record
        self.history['time'].append(self.time)
        self.history['n_tryptophans'].append(self.state.n_tryptophans)
        self.history['em_field_kT'].append(self.state.em_field_kT)
        self.history['dimer_count'].append(self.state.dimer_count)
        self.history['dimer_rate'].append(self.state.dimer_rate)
        self.history['calcium_active'].append(calcium_active)
        
        return self.state
    
    def get_tryptophan_state(self) -> Dict:
        """Get state dict compatible with PhotonEmissionTracker"""
        return {
            'collective': {
                'n_tryptophans': self.state.n_tryptophans,
                'enhancement_factor': np.sqrt(self.state.n_tryptophans / self.n_trp_baseline)
            },
            'output': {
                'collective_field_kT': self.state.em_field_kT
            }
        }


# =============================================================================
# TWO-REGION NETWORK
# =============================================================================

@dataclass
class RegionState:
    """State of a brain region"""
    region_id: int
    n_synapses: int
    mean_dimer_count: float = 0.0
    mean_em_field_kT: float = 0.0
    total_photons_emitted: float = 0.0
    total_photons_received: float = 0.0


class TwoRegionNetwork:
    """
    Two connected brain regions for cross-region coupling POC
    """
    
    def __init__(self,
                 n_synapses_per_region: int = 5,
                 connection_distance_mm: float = 5.0,
                 n_parallel_axons: int = 100):
        """
        Initialize two-region network
        
        Parameters
        ----------
        n_synapses_per_region : int
            Number of synapses in each region
        connection_distance_mm : float
            Distance between regions (mm)
        n_parallel_axons : int
            Number of axons in connecting bundle
        """
        self.n_synapses = n_synapses_per_region
        self.time = 0.0
        
        # === REGION A (SOURCE) ===
        self.region_a_synapses = [
            SimplifiedSynapse(synapse_id=i, region_id=0)
            for i in range(n_synapses_per_region)
        ]
        
        # Photon emission trackers for source region
        self.region_a_emitters = [
            PhotonEmissionTracker(synapse_id=i, mt_alignment=0.7)
            for i in range(n_synapses_per_region)
        ]
        
        # === REGION B (TARGET) ===
        self.region_b_synapses = [
            SimplifiedSynapse(synapse_id=i + n_synapses_per_region, region_id=1)
            for i in range(n_synapses_per_region)
        ]
        
        # Photon receivers for target region
        self.region_b_receivers = [
            PhotonReceiver(synapse_id=i + n_synapses_per_region)
            for i in range(n_synapses_per_region)
        ]
        
        # === WAVEGUIDE CONNECTION ===
        self.waveguide_network = WaveguideNetwork()
        self.waveguide = self.waveguide_network.add_connection(
            source_id=0,
            target_id=1,
            distance_mm=connection_distance_mm,
            axon_radius_um=1.0,
            n_myelin_layers=20,
            n_parallel_axons=n_parallel_axons
        )
        
        # State tracking
        self.region_a_state = RegionState(region_id=0, n_synapses=n_synapses_per_region)
        self.region_b_state = RegionState(region_id=1, n_synapses=n_synapses_per_region)
        
        # History
        self.history = {
            'time': [],
            'region_a_dimers': [],
            'region_b_dimers': [],
            'region_a_em_field': [],
            'region_b_em_field': [],
            'photons_in_transit': [],
            'photons_delivered': []
        }
        
        logger.info("TwoRegionNetwork initialized")
        logger.info(f"  Synapses per region: {n_synapses_per_region}")
        logger.info(f"  Connection distance: {connection_distance_mm} mm")
        logger.info(f"  Parallel axons: {n_parallel_axons}")
    
    def step(self, 
             dt: float,
             region_a_active: bool = False,
             region_b_active: bool = False) -> Dict:
        """
        Advance network by one timestep
        
        Parameters
        ----------
        dt : float
            Time step (s)
        region_a_active : bool
            Whether region A (source) has calcium activity
        region_b_active : bool
            Whether region B (target) has direct calcium activity
        """
        self.time += dt
        
        # === REGION A: SOURCE ===
        region_a_photons = []
        
        for synapse, emitter in zip(self.region_a_synapses, self.region_a_emitters):
            # Step synapse
            synapse.step(dt, calcium_active=region_a_active)
            
            # Step emission tracker
            trp_state = synapse.get_tryptophan_state()
            emission_result = emitter.step(dt, trp_state, ca_spike_active=region_a_active)
            
            # Collect emitted packets
            region_a_photons.extend(emission_result['new_packets'])
        
        # === WAVEGUIDE PROPAGATION ===
        # Inject new packets
        if region_a_photons:
            self.waveguide_network.inject_packets(
                source_id=0,
                packets=region_a_photons,
                current_time=self.time
            )
        
        # Propagate existing packets
        deliveries = self.waveguide_network.step(self.time)
        
        # Get deliveries for region B
        region_b_deliveries = deliveries.get(1, [])
        
        # === REGION B: TARGET ===
        # Distribute deliveries across receivers (round-robin for simplicity)
        deliveries_per_receiver = [[] for _ in range(self.n_synapses)]
        for i, delivery in enumerate(region_b_deliveries):
            deliveries_per_receiver[i % self.n_synapses].append(delivery)
        
        for i, (synapse, receiver) in enumerate(zip(self.region_b_synapses, 
                                                     self.region_b_receivers)):
            # Receive photons
            reception = receiver.receive_photons(deliveries_per_receiver[i], self.time)
            
            # Apply modulation to synapse
            modulation = receiver.get_modulation_factors()
            synapse.set_external_modulation(
                trp_mult=modulation['trp_n_effective_multiplier'],
                k_mult=modulation['k_agg_multiplier']
            )
            
            # Step synapse (with or without direct activity)
            synapse.step(dt, calcium_active=region_b_active)
        
        # === AGGREGATE STATES ===
        self.region_a_state.mean_dimer_count = np.mean(
            [s.state.dimer_count for s in self.region_a_synapses]
        )
        self.region_a_state.mean_em_field_kT = np.mean(
            [s.state.em_field_kT for s in self.region_a_synapses]
        )
        
        self.region_b_state.mean_dimer_count = np.mean(
            [s.state.dimer_count for s in self.region_b_synapses]
        )
        self.region_b_state.mean_em_field_kT = np.mean(
            [s.state.em_field_kT for s in self.region_b_synapses]
        )
        
        # Record history
        self.history['time'].append(self.time)
        self.history['region_a_dimers'].append(self.region_a_state.mean_dimer_count)
        self.history['region_b_dimers'].append(self.region_b_state.mean_dimer_count)
        self.history['region_a_em_field'].append(self.region_a_state.mean_em_field_kT)
        self.history['region_b_em_field'].append(self.region_b_state.mean_em_field_kT)
        self.history['photons_in_transit'].append(len(self.waveguide.packets_in_transit))
        self.history['photons_delivered'].append(len(region_b_deliveries))
        
        return {
            'region_a': self.region_a_state,
            'region_b': self.region_b_state,
            'photons_delivered': len(region_b_deliveries)
        }


# =============================================================================
# POC EXPERIMENT
# =============================================================================

@dataclass
class POCResult:
    """Results from POC experiment"""
    
    # Conditions
    source_stimulated: bool
    target_stimulated: bool
    connection_intact: bool
    
    # Timing
    duration_s: float
    stim_start_s: float
    stim_end_s: float
    
    # Outcomes
    source_final_dimers: float = 0.0
    target_final_dimers: float = 0.0
    target_baseline_dimers: float = 0.0
    target_dimer_enhancement: float = 1.0  # Ratio vs baseline
    
    # Photon statistics
    total_photons_emitted: float = 0.0
    total_photons_delivered: float = 0.0
    
    # Time series
    times: List[float] = field(default_factory=list)
    source_dimers: List[float] = field(default_factory=list)
    target_dimers: List[float] = field(default_factory=list)


def run_poc_experiment(
    duration_s: float = 2.0,
    stim_start_s: float = 0.5,
    stim_duration_s: float = 1.0,
    stimulate_source: bool = True,
    stimulate_target: bool = False,
    connection_intact: bool = True,
    n_parallel_axons: int = 100,
    verbose: bool = True
) -> POCResult:
    """
    Run POC experiment
    
    Parameters
    ----------
    duration_s : float
        Total experiment duration
    stim_start_s : float
        When stimulation begins
    stim_duration_s : float
        Duration of stimulation
    stimulate_source : bool
        Whether to stimulate source region
    stimulate_target : bool
        Whether to directly stimulate target region
    connection_intact : bool
        If False, simulates demyelination (no photon propagation)
    n_parallel_axons : int
        Number of axons (0 = no connection)
    verbose : bool
        Print progress
    """
    
    if verbose:
        print("\n" + "="*70)
        print("TWO-REGION QUANTUM COUPLING POC")
        print("="*70)
        print(f"\nConditions:")
        print(f"  Source stimulated: {stimulate_source}")
        print(f"  Target stimulated: {stimulate_target}")
        print(f"  Connection intact: {connection_intact}")
        print(f"  Parallel axons: {n_parallel_axons}")
    
    # Create network
    axons = n_parallel_axons if connection_intact else 0
    network = TwoRegionNetwork(
        n_synapses_per_region=5,
        connection_distance_mm=5.0,
        n_parallel_axons=axons
    )
    
    # Run simulation
    dt = 0.001  # 1 ms timestep
    n_steps = int(duration_s / dt)
    stim_start_step = int(stim_start_s / dt)
    stim_end_step = int((stim_start_s + stim_duration_s) / dt)
    
    if verbose:
        print(f"\nSimulating {duration_s}s ({n_steps} steps)...")
        print(f"Stimulation: {stim_start_s}s - {stim_start_s + stim_duration_s}s")
    
    # Record baseline (before stimulation)
    baseline_samples = []
    
    for step in range(n_steps):
        # Determine stimulation state
        in_stim_window = stim_start_step <= step < stim_end_step
        
        source_active = stimulate_source and in_stim_window
        target_active = stimulate_target and in_stim_window
        
        # Step network
        result = network.step(dt, 
                             region_a_active=source_active,
                             region_b_active=target_active)
        
        # Record baseline
        if step < stim_start_step:
            baseline_samples.append(result['region_b'].mean_dimer_count)
        
        # Progress
        if verbose and step % 500 == 0:
            t = step * dt
            print(f"  t={t:.2f}s: Source dimers={result['region_a'].mean_dimer_count:.2f}, "
                  f"Target dimers={result['region_b'].mean_dimer_count:.3f}")
    
    # Compile results
    baseline = np.mean(baseline_samples) if baseline_samples else 0.0
    final_target = network.history['region_b_dimers'][-1]
    
    result = POCResult(
        source_stimulated=stimulate_source,
        target_stimulated=stimulate_target,
        connection_intact=connection_intact,
        duration_s=duration_s,
        stim_start_s=stim_start_s,
        stim_end_s=stim_start_s + stim_duration_s,
        source_final_dimers=network.history['region_a_dimers'][-1],
        target_final_dimers=final_target,
        target_baseline_dimers=baseline,
        target_dimer_enhancement=final_target / baseline if baseline > 0 else 1.0,
        total_photons_emitted=sum(
            e.get_emission_summary()['total_coupled'] 
            for e in network.region_a_emitters
        ),
        total_photons_delivered=network.waveguide.stats['total_photons_delivered'],
        times=network.history['time'],
        source_dimers=network.history['region_a_dimers'],
        target_dimers=network.history['region_b_dimers']
    )
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nSource region (A):")
        print(f"  Final dimer count: {result.source_final_dimers:.2f}")
        print(f"  Photons coupled to waveguide: {result.total_photons_emitted:.2f}")
        
        print(f"\nTarget region (B):")
        print(f"  Baseline dimer count: {result.target_baseline_dimers:.4f}")
        print(f"  Final dimer count: {result.target_final_dimers:.4f}")
        print(f"  Enhancement factor: {result.target_dimer_enhancement:.2f}x")
        print(f"  Photons received: {result.total_photons_delivered:.2f}")
    
    return result


def run_comparison_experiment(verbose: bool = True) -> Dict[str, POCResult]:
    """
    Run comparison between conditions
    
    Conditions:
    1. Source only (test cross-region effect)
    2. No stimulation (baseline control)
    3. Connection severed (demyelination control)
    4. Target only (direct stimulation control)
    """
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON EXPERIMENT: CROSS-REGION QUANTUM COUPLING")
        print("="*70)
    
    results = {}
    
    # Condition 1: Source stimulation only
    if verbose:
        print("\n>>> CONDITION 1: Source Stimulation Only <<<")
    results['source_only'] = run_poc_experiment(
        stimulate_source=True,
        stimulate_target=False,
        connection_intact=True,
        verbose=verbose
    )
    
    # Condition 2: No stimulation (baseline)
    if verbose:
        print("\n>>> CONDITION 2: No Stimulation (Baseline) <<<")
    results['baseline'] = run_poc_experiment(
        stimulate_source=False,
        stimulate_target=False,
        connection_intact=True,
        verbose=verbose
    )
    
    # Condition 3: Connection severed (demyelination)
    if verbose:
        print("\n>>> CONDITION 3: Demyelinated (No Connection) <<<")
    results['demyelinated'] = run_poc_experiment(
        stimulate_source=True,
        stimulate_target=False,
        connection_intact=False,
        verbose=verbose
    )
    
    # Condition 4: Direct target stimulation
    if verbose:
        print("\n>>> CONDITION 4: Direct Target Stimulation <<<")
    results['target_direct'] = run_poc_experiment(
        stimulate_source=False,
        stimulate_target=True,
        connection_intact=True,
        verbose=verbose
    )
    
    # Summary comparison
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY COMPARISON")
        print("="*70)
        print(f"\n{'Condition':<25} {'Target Dimers':<15} {'Enhancement':<15} {'Photons Delivered':<15}")
        print("-"*70)
        
        for name, res in results.items():
            print(f"{name:<25} {res.target_final_dimers:<15.4f} "
                  f"{res.target_dimer_enhancement:<15.2f}x "
                  f"{res.total_photons_delivered:<15.1f}")
        
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        
        source_effect = results['source_only'].target_final_dimers - results['baseline'].target_final_dimers
        demyelin_effect = results['demyelinated'].target_final_dimers - results['baseline'].target_final_dimers
        
        print(f"\n1. Cross-region effect (source stim → target dimers):")
        print(f"   Target dimer increase: {source_effect:.4f}")
        print(f"   This represents {results['source_only'].target_dimer_enhancement:.1f}x baseline")
        
        print(f"\n2. Demyelination control:")
        print(f"   Without connection: {demyelin_effect:.4f} change")
        print(f"   Waveguide is {'REQUIRED' if source_effect > demyelin_effect else 'NOT required'}")
        
        print(f"\n3. Direct vs indirect stimulation:")
        direct = results['target_direct'].target_final_dimers
        indirect = results['source_only'].target_final_dimers
        print(f"   Direct stimulation: {direct:.4f} dimers")
        print(f"   Cross-region: {indirect:.4f} dimers ({indirect/direct*100:.1f}% of direct)")
    
    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(results: Dict[str, POCResult], output_dir: Optional[Path] = None):
    """Generate plots for POC results"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1 = axes[0, 0]
    for name, res in results.items():
        ax1.plot(res.times, res.target_dimers, label=name, alpha=0.8)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Stim start')
    ax1.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, label='Stim end')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Target Region Dimer Count')
    ax1.set_title('Target Dimer Dynamics Across Conditions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Source dimers (for reference)
    ax2 = axes[0, 1]
    for name, res in results.items():
        ax2.plot(res.times, res.source_dimers, label=name, alpha=0.8)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Source Region Dimer Count')
    ax2.set_title('Source Dimer Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final dimer comparison bar chart
    ax3 = axes[1, 0]
    names = list(results.keys())
    target_dimers = [results[n].target_final_dimers for n in names]
    colors = ['green' if n == 'source_only' else 'gray' for n in names]
    bars = ax3.bar(names, target_dimers, color=colors, alpha=0.7)
    ax3.set_ylabel('Final Target Dimer Count')
    ax3.set_title('Cross-Region Effect Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Annotate enhancement
    baseline_val = results['baseline'].target_final_dimers
    for i, (name, val) in enumerate(zip(names, target_dimers)):
        if baseline_val > 0:
            enhancement = val / baseline_val
            ax3.annotate(f'{enhancement:.1f}x', 
                        (i, val + 0.001), 
                        ha='center', fontsize=9)
    
    # Plot 4: Photon delivery
    ax4 = axes[1, 1]
    delivered = [results[n].total_photons_delivered for n in names]
    ax4.bar(names, delivered, color='blue', alpha=0.7)
    ax4.set_ylabel('Photons Delivered to Target')
    ax4.set_title('Waveguide Transmission')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'cross_region_poc_results.png', dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_dir / 'cross_region_poc_results.png'}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CROSS-REGION QUANTUM COUPLING - PROOF OF CONCEPT")
    print("="*70)
    print("""
This experiment demonstrates that quantum coherent photon emission 
from one brain region can modulate dimer formation in a distant 
region connected via myelinated axons.

The key prediction: TARGET REGION DIMER FORMATION IS ENHANCED 
when SOURCE REGION is stimulated, even without direct target stimulation.
This effect requires intact myelin waveguides.
""")
    
    # Run comparison experiment
    results = run_comparison_experiment(verbose=True)
    
    # Try to plot
    try:
        fig = plot_results(results, output_dir=Path(__file__).parent / 'output')
        if fig:
            print("\n✓ Plot saved to output directory")
    except Exception as e:
        print(f"\nNote: Could not generate plot: {e}")
    
    # Save results as JSON
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to serializable format
    results_dict = {}
    for name, res in results.items():
        results_dict[name] = {
            'source_stimulated': res.source_stimulated,
            'target_stimulated': res.target_stimulated,
            'connection_intact': res.connection_intact,
            'source_final_dimers': res.source_final_dimers,
            'target_final_dimers': res.target_final_dimers,
            'target_baseline_dimers': res.target_baseline_dimers,
            'target_dimer_enhancement': res.target_dimer_enhancement,
            'total_photons_emitted': res.total_photons_emitted,
            'total_photons_delivered': res.total_photons_delivered
        }
    
    json_path = output_dir / 'poc_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")
    
    print("\n" + "="*70)
    print("POC COMPLETE")
    print("="*70)