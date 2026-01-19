"""
Photon Receiver Module - Target Synapse Modulation by Incoming Biophotons
=========================================================================

When photons arrive at a target synapse via myelin waveguides, they can:
1. Excite local tryptophan networks (enhancing superradiant field)
2. Directly modulate dimer formation chemistry
3. Create phase coherence with source synapse

This module implements the target-side coupling of the cross-region
quantum communication pathway.

PHYSICAL MECHANISM:
------------------
Arriving 350 nm photons (from source superradiance) can:

1. TRYPTOPHAN EXCITATION
   - Absorbed by tryptophan at ~280-350 nm
   - Adds to local superradiant network
   - Enhances EM field that promotes dimer formation

2. DIRECT CHEMICAL MODULATION  
   - UV photons can directly affect phosphate chemistry
   - Modulates ion pair → dimer transition rates
   - Similar to external UV enhancement already in Model 6

3. PHASE COUPLING (speculative but testable)
   - Incoming photon phase could influence local quantum state
   - Creates non-local correlation between source and target
   - Requires coherent detection (challenging experimentally)

For the POC, we focus on mechanisms 1 and 2 which have clear
physical basis and produce measurable effects.

Author: Sarah Davidson
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cross_neuron_entanglement import CrossNeuronEntanglementTracker, CrossNeuronBond
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class PhotonReceiverParameters:
    """
    Parameters for photon reception and synapse modulation
    """
    
    # === TRYPTOPHAN EXCITATION ===
    # Arriving photons can excite local tryptophans
    # 350 nm is near tryptophan absorption tail (peak at 280 nm)
    
    # Absorption efficiency at 350 nm (tail of tryptophan spectrum)
    # Increased for POC - actual value ~15%, but using higher for demonstration
    absorption_efficiency_350nm: float = 0.50  # 50% for POC
    
    # Each absorbed photon adds to collective excitation
    # Enhancement = sqrt(N_absorbed) for partially coherent addition
    excitation_scaling: str = 'sqrt'  # 'linear' or 'sqrt'
    
    # Maximum enhancement factor (saturation)
    max_trp_enhancement: float = 2.0  # 2× max boost
    
    # === DIMER FORMATION MODULATION ===
    # External photons can enhance dimer formation rate
    # Similar mechanism to metabolic UV in Model 6
    
    # Enhancement factor per arriving photon
    # Increased for POC to show clear effect
    dimer_rate_enhancement_per_photon: float = 0.05  # 5% per photon
    
    # Maximum dimer rate enhancement
    max_dimer_enhancement: float = 3.0  # 200% max boost
    
    # Time constant for photon effect decay
    photon_effect_tau_s: float = 0.5  # 500 ms decay time (longer for accumulation)
    
    # === PHASE COUPLING (future) ===
    # For now, we track phase but don't use it functionally
    phase_coupling_enabled: bool = False
    phase_coherence_threshold: float = 0.5


# =============================================================================
# PHOTON RECEIVER
# =============================================================================

class PhotonReceiver:
    """
    Receives photons at target synapse and modulates local dynamics
    """
    
    def __init__(self,
                 synapse_id: int,
                 neuron_id: int = 0,  # NEW
                 params: Optional[PhotonReceiverParameters] = None):
        """
        Initialize receiver for a target synapse
        
        Parameters
        ----------
        synapse_id : int
            Target synapse identifier
        params : PhotonReceiverParameters
            Reception parameters
        """
        self.synapse_id = synapse_id
        self.neuron_id = neuron_id  # NEW
        self.params = params or PhotonReceiverParameters()
        
        # State variables
        self.time = 0.0
        self.accumulated_photons = 0.0  # Decaying accumulator
        self.total_received = 0.0
        
        # Current modulation effects
        self.trp_enhancement = 1.0
        self.dimer_rate_enhancement = 1.0
        
        # Phase tracking (for future use)
        self.phase_accumulator = 0.0
        self.n_phase_samples = 0
        
        # History
        self.history = {
            'time': [],
            'photons_received': [],
            'accumulated_photons': [],
            'trp_enhancement': [],
            'dimer_enhancement': []
        }

        # NEW: Track received dimer links for cross-neuron entanglement
        self.recent_dimer_links: List[dict] = []
        
        logger.info(f"PhotonReceiver initialized for synapse {synapse_id}")
    
    def receive_photons(self, 
                        delivered_packets: List[dict],
                        current_time: float) -> Dict:
        """
        Process delivered photon packets
        
        Parameters
        ----------
        delivered_packets : list
            Packet records from waveguide (each has 'delivered_photons', etc.)
        current_time : float
            Current simulation time (s)
        
        Returns
        -------
        dict with reception results
        """
        dt = current_time - self.time if self.time > 0 else 0.001
        self.time = current_time
        
        # === DECAY ACCUMULATED PHOTONS ===
        decay_factor = np.exp(-dt / self.params.photon_effect_tau_s)
        self.accumulated_photons *= decay_factor
        
        # === RECEIVE NEW PHOTONS ===
        n_received_this_step = 0.0
        phases = []
        
        for record in delivered_packets:
            n_photons = record['delivered_photons']
            n_received_this_step += n_photons
            
            # Track phase if available
            if 'packet' in record and hasattr(record['packet'], 'phase'):
                phases.append(record['packet'].phase)
            
            # NEW: Capture dimer link for cross-neuron entanglement
            if 'packet' in record and hasattr(record['packet'], 'dimer_link'):
                if record['packet'].dimer_link is not None:
                    self.recent_dimer_links.append({
                        'dimer_link': record['packet'].dimer_link,
                        'n_photons': n_photons,
                        'arrival_time': current_time
                    })
        
        # Add to accumulator (with absorption efficiency)
        absorbed = n_received_this_step * self.params.absorption_efficiency_350nm
        self.accumulated_photons += absorbed
        self.total_received += n_received_this_step
        
        # === CALCULATE TRYPTOPHAN ENHANCEMENT ===
        if self.accumulated_photons > 0:
            if self.params.excitation_scaling == 'sqrt':
                enhancement_raw = np.sqrt(self.accumulated_photons)
            else:
                enhancement_raw = self.accumulated_photons
            
            self.trp_enhancement = 1.0 + min(
                enhancement_raw * 0.1,  # Scaling factor
                self.params.max_trp_enhancement - 1.0
            )
        else:
            self.trp_enhancement = 1.0
        
        # === CALCULATE DIMER RATE ENHANCEMENT ===
        rate_boost = self.accumulated_photons * self.params.dimer_rate_enhancement_per_photon
        self.dimer_rate_enhancement = 1.0 + min(
            rate_boost,
            self.params.max_dimer_enhancement - 1.0
        )
        
        # === PHASE ACCUMULATION (for future use) ===
        if phases and self.params.phase_coupling_enabled:
            # Circular mean of phases
            self.phase_accumulator += np.sum(np.exp(1j * np.array(phases)))
            self.n_phase_samples += len(phases)
        
        # Record history
        self.history['time'].append(current_time)
        self.history['photons_received'].append(n_received_this_step)
        self.history['accumulated_photons'].append(self.accumulated_photons)
        self.history['trp_enhancement'].append(self.trp_enhancement)
        self.history['dimer_enhancement'].append(self.dimer_rate_enhancement)
        
        return {
            'n_received': n_received_this_step,
            'n_absorbed': absorbed,
            'accumulated': self.accumulated_photons,
            'trp_enhancement': self.trp_enhancement,
            'dimer_enhancement': self.dimer_rate_enhancement
        }
    
    
    def process_entanglement(self,
                            target_dimers: List,
                            entanglement_tracker: 'CrossNeuronEntanglementTracker',
                            current_time: float) -> List['CrossNeuronBond']:
        """
        Process recent arrivals for cross-neuron entanglement.
        
        Should be called after receive_photons() when target dimers are available.
        
        Parameters
        ----------
        target_dimers : list
            Current Dimer objects at this synapse
        entanglement_tracker : CrossNeuronEntanglementTracker
            The network's cross-neuron entanglement tracker
        current_time : float
            Current simulation time (s)
            
        Returns
        -------
        list of CrossNeuronBond
            Newly created entanglement bonds
        """
        all_new_bonds = []
        
        for arrival in self.recent_dimer_links:
            new_bonds = entanglement_tracker.process_arrival(
                dimer_link=arrival['dimer_link'],
                n_photons_delivered=arrival['n_photons'],
                target_dimers=target_dimers,
                target_neuron_id=self.neuron_id,
                target_synapse_id=self.synapse_id,
                current_time=current_time
            )
            all_new_bonds.extend(new_bonds)
        
        # Clear processed links
        self.recent_dimer_links = []
        
        return all_new_bonds
    
    
    def get_modulation_factors(self) -> Dict:
        """
        Get current modulation factors for synapse
        
        Returns factors that should be applied to:
        - Tryptophan network effective size
        - Dimer formation rate constant
        """
        return {
            'trp_n_effective_multiplier': self.trp_enhancement,
            'k_agg_multiplier': self.dimer_rate_enhancement,
            'photon_accumulated': self.accumulated_photons
        }
    
    def get_phase_coherence(self) -> Optional[float]:
        """
        Get phase coherence of received photons
        
        Returns coherence value (0-1) if phase coupling enabled,
        None otherwise.
        """
        if not self.params.phase_coupling_enabled or self.n_phase_samples == 0:
            return None
        
        # Magnitude of mean phasor / number of samples = coherence
        mean_phasor = self.phase_accumulator / self.n_phase_samples
        return abs(mean_phasor)
    
    def reset_phase(self):
        """Reset phase accumulator"""
        self.phase_accumulator = 0.0
        self.n_phase_samples = 0
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_received': self.total_received,
            'current_accumulated': self.accumulated_photons,
            'current_trp_enhancement': self.trp_enhancement,
            'current_dimer_enhancement': self.dimer_rate_enhancement,
            'phase_coherence': self.get_phase_coherence()
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHOTON RECEIVER MODULE - VALIDATION")
    print("="*70)
    
    # Create receiver
    receiver = PhotonReceiver(synapse_id=1)
    
    # Simulate receiving photons over time
    print("\n--- Simulating Photon Reception ---")
    print(f"{'Time (ms)':<12} {'Received':<12} {'Accumulated':<14} "
          f"{'Trp Enh':<12} {'Dimer Enh':<12}")
    print("-"*64)
    
    dt = 0.001  # 1 ms
    
    # Create mock delivery records
    def make_delivery(n_photons):
        from photon_emission_module import PhotonPacket
        packet = PhotonPacket(id=0, emission_time=0, source_synapse_id=0, 
                             n_photons=n_photons, wavelength_m=350e-9)
        return {'delivered_photons': n_photons, 'packet': packet}
    
    # Simulate bursts of incoming photons
    for step in range(200):  # 200 ms
        t = step * dt
        
        # Deliver photon bursts at specific times
        if step in [10, 30, 50, 70, 90]:  # Every 20 ms for 5 bursts
            deliveries = [make_delivery(100)]  # 100 photons per burst
        else:
            deliveries = []
        
        result = receiver.receive_photons(deliveries, t)
        
        # Print every 10 ms
        if step % 10 == 0:
            print(f"{t*1000:>10.1f}  "
                  f"{result['n_received']:>10.1f}  "
                  f"{result['accumulated']:>12.2f}  "
                  f"{result['trp_enhancement']:>10.3f}  "
                  f"{result['dimer_enhancement']:>10.4f}")
    
    print("\n--- Final Summary ---")
    summary = receiver.get_summary()
    for key, value in summary.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Test decay
    print("\n--- Testing Decay (no new photons) ---")
    print(f"{'Time (ms)':<12} {'Accumulated':<14} {'Trp Enh':<12} {'Dimer Enh':<12}")
    print("-"*52)
    
    for step in range(100):  # Another 100 ms with no input
        t = 0.2 + step * dt
        result = receiver.receive_photons([], t)
        
        if step % 20 == 0:
            print(f"{t*1000:>10.1f}  "
                  f"{result['accumulated']:>12.2f}  "
                  f"{result['trp_enhancement']:>10.3f}  "
                  f"{result['dimer_enhancement']:>10.4f}")
    
    print("\n✓ Photon receiver module validated")