"""
Multi-Region Network Orchestrator
==================================

Runs cross-region quantum coupling experiments using full Model6QuantumSynapse.

Usage:
    python multi_region_network.py

Author: Sarah Davidson  
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging

# Model 6 imports
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters

# Waveguide network
from myelin_waveguide_module import WaveguideConnection, WaveguideNetwork

# After existing imports, add:
from cross_neuron_entanglement import (
    CrossNeuronEntanglementTracker,
    CrossNeuronEntanglementParameters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# REGION CLASS
# =============================================================================

@dataclass
class RegionConfig:
    """Configuration for a brain region"""
    region_id: int
    n_synapses: int = 5
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Region_{self.region_id}"


class Region:
    """
    A brain region containing multiple Model6 synapses
    """
    
    def __init__(self, config: RegionConfig, params: Optional[Model6Parameters] = None):
        self.config = config
        self.region_id = config.region_id
        self.name = config.name
        
        # Create synapses with cross-region enabled
        if params is None:
            params = Model6Parameters()
        params.em_coupling_enabled = True
        params.cross_region_enabled = True
        
        self.synapses: List[Model6QuantumSynapse] = []
        for i in range(config.n_synapses):
            synapse = Model6QuantumSynapse(params=params)
            self.synapses.append(synapse)
        
        logger.info(f"Region '{self.name}' initialized with {config.n_synapses} synapses")
    
    def step(self, dt: float, stimulus: Optional[Dict] = None):
        """Step all synapses in the region"""
        for synapse in self.synapses:
            synapse.step(dt, stimulus)
    
    def get_emitted_packets(self) -> List:
        """Collect all emitted packets from synapses"""
        all_packets = []
        for synapse in self.synapses:
            packets = synapse.get_emitted_packets()
            all_packets.extend(packets)
        return all_packets
    
    def receive_photons(self, delivered_packets: List, current_time: float):
        """Distribute received photons across synapses"""
        if not delivered_packets:
            return
        
        # Round-robin distribution
        for i, packet in enumerate(delivered_packets):
            synapse_idx = i % len(self.synapses)
            self.synapses[synapse_idx].receive_photons([packet], current_time)
    
    def get_mean_dimer_count(self) -> float:
        """Get mean dimer count across synapses"""
        counts = []
        for synapse in self.synapses:
            if hasattr(synapse, 'dimer_particles'):
                counts.append(len(synapse.dimer_particles.dimers))
            else:
                counts.append(synapse.ca_phosphate.get_dimer_concentration().sum())
        return np.mean(counts) if counts else 0.0
    
    def get_mean_singlet_probability(self) -> float:
        """Get mean singlet probability across synapses"""
        probs = []
        for synapse in self.synapses:
            if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                for d in synapse.dimer_particles.dimers:
                    probs.append(d.singlet_probability)
        return np.mean(probs) if probs else 0.25
    
    def get_mean_em_field(self) -> float:
        """Get mean EM field across synapses"""
        fields = []
        for synapse in self.synapses:
            if hasattr(synapse, '_collective_field_kT'):
                fields.append(synapse._collective_field_kT)
        return np.mean(fields) if fields else 0.0


# =============================================================================
# MULTI-REGION NETWORK
# =============================================================================

class MultiRegionNetwork:
    """
    Network of brain regions connected by myelin waveguides
    """
    
    def __init__(self):
        self.regions: Dict[int, Region] = {}
        self.waveguide_network = WaveguideNetwork()
        self.time = 0.0
        
        # NEW: Cross-neuron entanglement tracker
        self.cross_neuron_entanglement = CrossNeuronEntanglementTracker()
        
        # History tracking
        self.history = {
            'time': [],
            'region_dimers': {},
            'region_em_fields': {},
            'photons_transmitted': {},
            'cross_neuron_bonds': [],      # NEW
            'coordination_factor': []       # NEW
        }

    def add_region(self, 
                   region_id: int, 
                   n_synapses: int = 5,
                   name: str = "",
                   params: Optional[Model6Parameters] = None) -> Region:
        """Add a region to the network"""
        config = RegionConfig(region_id=region_id, n_synapses=n_synapses, name=name)
        region = Region(config, params)
        self.regions[region_id] = region
        
        # Initialize history for this region
        self.history['region_dimers'][region_id] = []
        self.history['region_em_fields'][region_id] = []
        
        return region
    
    def connect_regions(self,
                        source_id: int,
                        target_id: int,
                        distance_mm: float = 5.0,
                        n_parallel_axons: int = 100,
                        axon_radius_um: float = 1.0,
                        n_myelin_layers: int = 20):
        """Add waveguide connection between regions"""
        
        self.waveguide_network.add_connection(
            source_id=source_id,
            target_id=target_id,
            distance_mm=distance_mm,
            axon_radius_um=axon_radius_um,
            n_myelin_layers=n_myelin_layers,
            n_parallel_axons=n_parallel_axons
        )
        
        # Track transmission history
        key = (source_id, target_id)
        self.history['photons_transmitted'][key] = []
        
        logger.info(f"Connected region {source_id} → {target_id} "
                   f"({distance_mm}mm, {n_parallel_axons} axons)")
    
    def step(self, dt: float, stimuli: Optional[Dict[int, Dict]] = None):
        """
        Advance the network by one timestep
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        stimuli : dict
            Map of region_id → stimulus dict for that region
        """
        self.time += dt
        stimuli = stimuli or {}
        
        # === STEP 1: Run all regions ===
        for region_id, region in self.regions.items():
            stimulus = stimuli.get(region_id, {'voltage': -70e-3, 'reward': False})
            region.step(dt, stimulus)
        
        # === STEP 2: Collect emitted photons and TAG WITH DIMER INFO ===
        emitted_by_region = {}
        for region_id, region in self.regions.items():
            packets = region.get_emitted_packets()
            emitted_by_region[region_id] = packets
            
            # NEW: Tag packets with source dimer information
            for packet in packets:
                synapse_idx = packet.source_synapse_id if hasattr(packet, 'source_synapse_id') else 0
                if synapse_idx < len(region.synapses):
                    synapse = region.synapses[synapse_idx]
                    if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                        packet.dimer_link = self.cross_neuron_entanglement.tag_packet_with_dimers(
                            packet_id=packet.id,
                            emission_time=self.time,
                            source_dimers=synapse.dimer_particles.dimers,
                            source_neuron_id=region_id,
                            source_synapse_id=synapse_idx
                        )
                        packet.source_neuron_id = region_id
            
            # Inject into waveguides
            if packets:
                self.waveguide_network.inject_packets(
                    source_id=region_id,
                    packets=packets,
                    current_time=self.time
                )
        
        # === STEP 3: Propagate through waveguides ===
        deliveries = self.waveguide_network.step(self.time)
        
        # === STEP 4: Deliver photons and PROCESS ENTANGLEMENT ===
        for target_id, packets in deliveries.items():
            if target_id in self.regions:
                region = self.regions[target_id]
                region.receive_photons(packets, self.time)
                
                # NEW: Process cross-neuron entanglement
                for packet_record in packets:
                    packet = packet_record.get('packet')
                    if packet and hasattr(packet, 'dimer_link') and packet.dimer_link is not None:
                        # Find which synapse received this packet
                        for syn_idx, synapse in enumerate(region.synapses):
                            if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                                self.cross_neuron_entanglement.process_arrival(
                                    dimer_link=packet.dimer_link,
                                    n_photons_delivered=packet_record.get('delivered_photons', 0),
                                    target_dimers=synapse.dimer_particles.dimers,
                                    target_neuron_id=target_id,
                                    target_synapse_id=syn_idx,
                                    current_time=self.time
                                )
        
        # === STEP 5: Update cross-neuron entanglement decay ===
        dimer_states = self._collect_dimer_states()
        self.cross_neuron_entanglement.step(dt, dimer_states, self.time)
        
        # === STEP 6: Record history ===
        self.history['time'].append(self.time)
        
        for region_id, region in self.regions.items():
            self.history['region_dimers'][region_id].append(
                region.get_mean_dimer_count()
            )
            self.history['region_em_fields'][region_id].append(
                region.get_mean_em_field()
            )
        
        for key in self.history['photons_transmitted']:
            src, tgt = key
            delivered = len(deliveries.get(tgt, []))
            self.history['photons_transmitted'][key].append(delivered)
        
        # NEW: Record entanglement history
        self.history['cross_neuron_bonds'].append(
            len(self.cross_neuron_entanglement.bonds)
        )
        self.history['coordination_factor'].append(
            self.cross_neuron_entanglement.get_coordination_factor(
                list(self.regions.keys())
            )
        )

    def _collect_dimer_states(self) -> Dict:
        """
        Collect current singlet probabilities of all dimers across network.
        
        Returns dict mapping (neuron_id, synapse_id, dimer_id) -> P_S
        """
        states = {}
        
        for region_id, region in self.regions.items():
            for syn_idx, synapse in enumerate(region.synapses):
                if hasattr(synapse, 'dimer_particles'):
                    for dimer in synapse.dimer_particles.dimers:
                        key = (region_id, syn_idx, dimer.id)
                        states[key] = dimer.singlet_probability
        
        return states

    def get_network_entanglement_state(self) -> Dict:
        """
        Get current state of cross-neuron entanglement.
        
        Used for coordination experiments and reward distribution.
        """
        neuron_ids = list(self.regions.keys())
        
        return {
            'n_bonds': len(self.cross_neuron_entanglement.bonds),
            'coordination_factor': self.cross_neuron_entanglement.get_coordination_factor(neuron_ids),
            'entanglement_matrix': self.cross_neuron_entanglement.get_entanglement_matrix(neuron_ids),
            'metrics': self.cross_neuron_entanglement.get_network_metrics()
        }

    def apply_reward(self, reward: float) -> Dict:
        """
        Apply reward to network with entanglement-based coordination.
        
        Neurons with cross-neuron entanglement update coherently.
        This is the key mechanism for coordination without backprop.
        
        Parameters
        ----------
        reward : float
            Scalar reward signal (like dopamine)
            
        Returns
        -------
        dict with update information
        """
        ent_state = self.get_network_entanglement_state()
        coordination = ent_state['coordination_factor']
        
        updates = {}
        
        for region_id, region in self.regions.items():
            for syn_idx, synapse in enumerate(region.synapses):
                # Get local eligibility from dimer coherence
                if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                    mean_P_S = np.mean([d.singlet_probability for d in synapse.dimer_particles.dimers])
                    eligibility = (mean_P_S - 0.25) / 0.75  # Map to 0-1
                else:
                    eligibility = 0.0
                
                # Get cross-neuron bonds involving this region
                bonds = self.cross_neuron_entanglement.get_bonds_for_neuron(region_id)
                bond_strength = sum(b.strength for b in bonds) / max(1, len(bonds)) if bonds else 0.0
                
                # Coordination bonus: stronger update for more entangled synapses
                coordination_bonus = 1.0 + coordination * bond_strength
                
                # Weight update
                learning_rate = 0.01
                delta_w = reward * eligibility * coordination_bonus * learning_rate
                
                updates[(region_id, syn_idx)] = {
                    'eligibility': eligibility,
                    'bond_strength': bond_strength,
                    'coordination_bonus': coordination_bonus,
                    'delta_w': delta_w
                }
        
        return {
            'updates': updates,
            'coordination_factor': coordination,
            'n_cross_neuron_bonds': ent_state['n_bonds']
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_two_region_experiment(
    duration_s: float = 2.0,
    stim_start_s: float = 0.5,
    stim_duration_s: float = 1.0,
    stimulate_source: bool = True,
    stimulate_target: bool = False,
    connection_intact: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run two-region cross-coupling experiment with full Model6 synapses
    """
    
    if verbose:
        print("\n" + "="*70)
        print("CROSS-REGION EXPERIMENT (Full Model 6)")
        print("="*70)
        print(f"\nConditions:")
        print(f"  Source stimulated: {stimulate_source}")
        print(f"  Target stimulated: {stimulate_target}")
        print(f"  Connection intact: {connection_intact}")
    
    # Create network
    network = MultiRegionNetwork()
    
    # Add regions
    network.add_region(region_id=0, n_synapses=3, name="Source")
    network.add_region(region_id=1, n_synapses=3, name="Target")
    
    # Connect if intact
    if connection_intact:
        network.connect_regions(
            source_id=0,
            target_id=1,
            distance_mm=5.0,
            n_parallel_axons=100
        )
    
    # Run simulation
    dt = 0.001  # 1 ms
    n_steps = int(duration_s / dt)
    stim_start_step = int(stim_start_s / dt)
    stim_end_step = int((stim_start_s + stim_duration_s) / dt)
    
    if verbose:
        print(f"\nSimulating {duration_s}s ({n_steps} steps)...")
    
    # Stimulus definitions
    active_stimulus = {'voltage': -10e-3, 'reward': False}
    rest_stimulus = {'voltage': -70e-3, 'reward': False}
    
    for step in range(n_steps):
        in_stim_window = stim_start_step <= step < stim_end_step
        
        stimuli = {}
        
        # Source stimulation
        if stimulate_source and in_stim_window:
            stimuli[0] = active_stimulus
        else:
            stimuli[0] = rest_stimulus
        
        # Target stimulation
        if stimulate_target and in_stim_window:
            stimuli[1] = active_stimulus
        else:
            stimuli[1] = rest_stimulus
        
        network.step(dt, stimuli)
        
        # Progress report
        if verbose and step % 500 == 0:
            t = step * dt
            src_dimers = network.regions[0].get_mean_dimer_count()
            tgt_dimers = network.regions[1].get_mean_dimer_count()
            print(f"  t={t:.2f}s: Source={src_dimers:.2f}, Target={tgt_dimers:.2f}")
    
    # Compile results
    results = {
        'duration_s': duration_s,
        'stimulate_source': stimulate_source,
        'stimulate_target': stimulate_target,
        'connection_intact': connection_intact,
        'source_final_dimers': network.history['region_dimers'][0][-1],
        'target_final_dimers': network.history['region_dimers'][1][-1],
        'history': network.history
    }
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Source final dimers: {results['source_final_dimers']:.2f}")
        print(f"Target final dimers: {results['target_final_dimers']:.2f}")
    
    return results


def run_comparison_experiment(verbose: bool = True) -> Dict[str, Dict]:
    """
    Run all four conditions for comparison
    """
    
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON EXPERIMENT: CROSS-REGION QUANTUM COUPLING")
        print("="*70)
    
    results = {}
    
    # Condition 1: Source only
    if verbose:
        print("\n>>> CONDITION 1: Source Stimulation Only <<<")
    results['source_only'] = run_two_region_experiment(
        stimulate_source=True,
        stimulate_target=False,
        connection_intact=True,
        verbose=verbose
    )
    
    # Condition 2: Baseline
    if verbose:
        print("\n>>> CONDITION 2: No Stimulation (Baseline) <<<")
    results['baseline'] = run_two_region_experiment(
        stimulate_source=False,
        stimulate_target=False,
        connection_intact=True,
        verbose=verbose
    )
    
    # Condition 3: Demyelinated
    if verbose:
        print("\n>>> CONDITION 3: Demyelinated <<<")
    results['demyelinated'] = run_two_region_experiment(
        stimulate_source=True,
        stimulate_target=False,
        connection_intact=False,
        verbose=verbose
    )
    
    # Condition 4: Direct target
    if verbose:
        print("\n>>> CONDITION 4: Direct Target Stimulation <<<")
    results['target_direct'] = run_two_region_experiment(
        stimulate_source=False,
        stimulate_target=True,
        connection_intact=True,
        verbose=verbose
    )
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Condition':<20} {'Source Dimers':<15} {'Target Dimers':<15}")
        print("-"*50)
        for name, res in results.items():
            print(f"{name:<20} {res['source_final_dimers']:<15.2f} "
                  f"{res['target_final_dimers']:<15.2f}")
        
        # Cross-region effect
        source_effect = (results['source_only']['target_final_dimers'] - 
                        results['baseline']['target_final_dimers'])
        demyelin_effect = (results['demyelinated']['target_final_dimers'] - 
                          results['baseline']['target_final_dimers'])
        
        print(f"\nCross-region effect: {source_effect:.4f} dimers")
        print(f"Demyelinated control: {demyelin_effect:.4f} dimers")
        
        if source_effect > demyelin_effect + 0.001:
            print("\n✓ WAVEGUIDE EFFECT CONFIRMED")
        else:
            print("\n✗ No significant waveguide effect detected")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_comparison_experiment(verbose=True)
    
    # Save results
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Convert history arrays to lists for JSON
    json_results = {}
    for name, res in results.items():
        json_results[name] = {
            'stimulate_source': res['stimulate_source'],
            'stimulate_target': res['stimulate_target'],
            'connection_intact': res['connection_intact'],
            'source_final_dimers': res['source_final_dimers'],
            'target_final_dimers': res['target_final_dimers']
        }
    
    json_path = output_dir / 'model6_cross_region_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {json_path}")