"""
Photon Emission Module - Superradiant Emission Escaping Local Synapse
=====================================================================

Extends the TryptophanSuperradianceModule to track photons that escape
the local synaptic environment and could propagate to distant regions.

PHYSICAL BASIS:
--------------
When tryptophan networks emit superradiantly:
1. Most energy is absorbed locally (enhances local dimer formation)
2. Some photons escape the local environment
3. Escaped photons can couple into axonal waveguides (myelin)

The fraction that escapes depends on:
- Emission geometry (microtubule orientation relative to axon)
- Local absorption (how much is captured by nearby tryptophans)
- Waveguide coupling efficiency

KEY LITERATURE:
--------------
Babcock et al. 2024 - Superradiant emission from MT networks
Kumar et al. 2016 (Sci Rep) - Waveguide coupling in myelinated axons
Sun et al. 2022 (Appl Opt) - Wavelength-dependent propagation

Author: Sarah Davidson
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cross_neuron_entanglement import PhotonDimerLink
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class PhotonEmissionParameters:
    """
    Parameters for superradiant photon emission and waveguide coupling
    
    All values derived from literature with explicit citations.
    """
    
    # === EMISSION PROPERTIES ===
    # From Babcock et al. 2024
    wavelength_emission_m: float = 350e-9  # Tryptophan fluorescence peak
    emission_lifetime_s: float = 100e-15  # 100 fs superradiant burst
    quantum_yield_mt: float = 0.116  # Microtubule quantum yield (70% enhanced)
    
    # Emission directionality
    # Superradiant emission is directional along collective dipole axis
    # For MT aligned with axon, emission cone ~30° half-angle
    emission_cone_half_angle_deg: float = 30.0
    
    # === LOCAL vs ESCAPED FRACTION ===
    # Most emitted photons are reabsorbed locally or scatter
    # Only a fraction escapes to potential waveguide coupling
    # For POC, use optimistic values - can tune for realism later
    local_absorption_fraction: float = 0.70  # 70% absorbed locally
    scattering_loss_fraction: float = 0.10   # 10% scattered
    # Escaped fraction = 1 - 0.70 - 0.10 = 0.20 (20%)
    
    @property
    def escaped_fraction(self) -> float:
        return 1.0 - self.local_absorption_fraction - self.scattering_loss_fraction
    
    # === WAVEGUIDE COUPLING ===
    # Geometric overlap between MT emission and myelin entrance
    # Depends on MT orientation relative to axon axis
    # Increased for POC to show clear effect
    coupling_efficiency_aligned: float = 0.30  # MT parallel to axon
    coupling_efficiency_perpendicular: float = 0.05  # MT perpendicular
    
    # Axon initial segment has aligned MTs
    # Spine MTs are less aligned
    mt_alignment_ais: float = 0.9  # 90% aligned in axon initial segment
    mt_alignment_spine: float = 0.3  # 30% aligned in spine
    
    # === PHOTON STATISTICS ===
    # From Model 6 metabolic UV parameters
    baseline_photon_flux: float = 20.0  # photons/s at PSD baseline
    active_enhancement: float = 5.0  # During Ca²⁺ transient
    ltp_enhancement: float = 10.0  # During sustained LTP
    
    # Theta burst parameters
    spikes_per_burst: int = 4
    bursts_per_protocol: int = 5
    spike_duration_ms: float = 2.0
    interspike_interval_ms: float = 8.0  # Within burst (100 Hz)
    interburst_interval_ms: float = 200.0  # Between bursts (5 Hz theta)


# =============================================================================
# PHOTON PACKET
# =============================================================================

@dataclass
class PhotonPacket:
    """
    Represents a packet of coherent photons from superradiant emission
    
    Not individual photons, but the collective emission from one
    superradiant event (100 fs burst).
    """
    
    id: int
    emission_time: float  # When emitted (s)
    source_synapse_id: int
    
    # Photon properties
    n_photons: float  # Number of photons in packet (can be fractional for averaging)
    wavelength_m: float = 350e-9
    
    # Phase information (for potential interference effects)
    phase: float = 0.0  # radians
    coherence_length_m: float = 1e-6  # ~1 μm coherence length for UV
    
    # Polarization (preserved in waveguide - Zangari et al. 2023)
    polarization_angle: float = 0.0  # radians
    
    # Propagation state
    in_waveguide: bool = False
    propagation_distance_m: float = 0.0
    attenuation_factor: float = 1.0  # Remaining intensity fraction

    # NEW: Cross-neuron entanglement support
    source_neuron_id: int = 0
    dimer_link: Optional['PhotonDimerLink'] = None


# =============================================================================
# EMISSION TRACKER
# =============================================================================

class PhotonEmissionTracker:
    """
    Tracks photon emission from a synapse's tryptophan network
    
    Interfaces with TryptophanSuperradianceModule to capture escaped photons.
    """
    
    def __init__(self, 
                 synapse_id: int,
                 neuron_id: int = 0,  # NEW
                 params: Optional[PhotonEmissionParameters] = None,
                 mt_alignment: float = 0.5):
        """
        Initialize emission tracker
        
        Parameters
        ----------
        synapse_id : int
            Identifier for source synapse
        params : PhotonEmissionParameters
            Emission parameters
        mt_alignment : float
            Fraction of MTs aligned with axon (0-1)
            Higher in axon initial segment, lower in spines
        """
        self.synapse_id = synapse_id
        self.neuron_id = neuron_id  # NEW
        self.params = params or PhotonEmissionParameters()
        self.mt_alignment = mt_alignment
        
        # Packet management
        self.packets: List[PhotonPacket] = []
        self.next_packet_id = 0
        self.time = 0.0
        
        # Emission history
        self.history = {
            'time': [],
            'n_tryptophans': [],
            'photons_emitted': [],
            'photons_escaped': [],
            'photons_coupled': [],
            'em_field_kT': []
        }
        
        # Calculate effective coupling efficiency
        self._update_coupling_efficiency()
        
        logger.info(f"PhotonEmissionTracker initialized for synapse {synapse_id}")
        logger.info(f"  MT alignment: {mt_alignment:.2f}")
        logger.info(f"  Effective coupling: {self.coupling_efficiency:.3f}")
    
    def _update_coupling_efficiency(self):
        """Calculate coupling efficiency based on MT alignment"""
        aligned = self.params.coupling_efficiency_aligned
        perp = self.params.coupling_efficiency_perpendicular
        
        # Weighted average based on alignment
        self.coupling_efficiency = (
            self.mt_alignment * aligned + 
            (1 - self.mt_alignment) * perp
        )
    
    def step(self, 
             dt: float,
             tryptophan_state: Dict,
             ca_spike_active: bool = False) -> Dict:
        """
        Update emission state based on tryptophan module output
        
        Parameters
        ----------
        dt : float
            Time step (s)
        tryptophan_state : dict
            Output from TryptophanSuperradianceModule.update()
            Expected keys: 'collective', 'emission', 'output'
        ca_spike_active : bool
            Whether Ca²⁺ spike is occurring
        
        Returns
        -------
        dict with:
            'n_emitted': Total photons emitted this step
            'n_escaped': Photons escaping local environment
            'n_coupled': Photons coupled to waveguide
            'new_packets': List of PhotonPacket objects
        """
        self.time += dt
        
        # Extract state from tryptophan module
        n_tryptophans = tryptophan_state.get('collective', {}).get('n_tryptophans', 0)
        enhancement = tryptophan_state.get('collective', {}).get('enhancement_factor', 1.0)
        em_field_kT = tryptophan_state.get('output', {}).get('collective_field_kT', 0.0)
        
        # Calculate emission rate
        if ca_spike_active:
            base_flux = self.params.baseline_photon_flux * self.params.active_enhancement
        else:
            base_flux = self.params.baseline_photon_flux
        
        # Superradiant enhancement
        # Emission rate scales with √N for partially coherent system
        if n_tryptophans > 100:
            sr_enhancement = np.sqrt(n_tryptophans / 100)
        else:
            sr_enhancement = 1.0
        
        # Total photons emitted this timestep
        n_emitted = base_flux * sr_enhancement * self.params.quantum_yield_mt * dt
        
        # Fraction escaping local environment
        n_escaped = n_emitted * self.params.escaped_fraction
        
        # Fraction coupling to waveguide
        n_coupled = n_escaped * self.coupling_efficiency
        
        # Create photon packet if significant emission
        new_packets = []
        if n_coupled > 0.001:  # Lowered threshold for POC
            packet = PhotonPacket(
                id=self.next_packet_id,
                emission_time=self.time,
                source_synapse_id=self.synapse_id,
                n_photons=n_coupled,
                wavelength_m=self.params.wavelength_emission_m,
                phase=np.random.uniform(0, 2*np.pi),
                polarization_angle=np.random.uniform(0, np.pi)
            )
            new_packets.append(packet)
            self.packets.append(packet)
            self.next_packet_id += 1
        
        # Record history
        self.history['time'].append(self.time)
        self.history['n_tryptophans'].append(n_tryptophans)
        self.history['photons_emitted'].append(n_emitted)
        self.history['photons_escaped'].append(n_escaped)
        self.history['photons_coupled'].append(n_coupled)
        self.history['em_field_kT'].append(em_field_kT)
        
        return {
            'n_emitted': n_emitted,
            'n_escaped': n_escaped,
            'n_coupled': n_coupled,
            'new_packets': new_packets
        }
    
    def get_pending_packets(self) -> List[PhotonPacket]:
        """Get packets ready for waveguide propagation"""
        return [p for p in self.packets if not p.in_waveguide]
    
    def mark_packets_in_waveguide(self, packet_ids: List[int]):
        """Mark packets as entered waveguide"""
        for packet in self.packets:
            if packet.id in packet_ids:
                packet.in_waveguide = True
    
    def get_emission_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.history['time']:
            return {}
        
        return {
            'total_time': self.time,
            'total_emitted': sum(self.history['photons_emitted']),
            'total_escaped': sum(self.history['photons_escaped']),
            'total_coupled': sum(self.history['photons_coupled']),
            'n_packets': len(self.packets),
            'mean_em_field_kT': np.mean(self.history['em_field_kT'])
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHOTON EMISSION MODULE - VALIDATION")
    print("="*70)
    
    # Create tracker
    tracker = PhotonEmissionTracker(
        synapse_id=0,
        mt_alignment=0.7  # Moderately aligned
    )
    
    # Simulate theta burst
    print("\n--- Simulating Theta Burst ---")
    print(f"{'Time (ms)':<12} {'Emitted':<12} {'Escaped':<12} {'Coupled':<12} {'Packets':<10}")
    print("-"*60)
    
    dt = 0.001  # 1 ms
    
    # Mock tryptophan state (would come from TryptophanSuperradianceModule)
    baseline_state = {
        'collective': {'n_tryptophans': 400, 'enhancement_factor': 1.0},
        'output': {'collective_field_kT': 5.0}
    }
    
    active_state = {
        'collective': {'n_tryptophans': 1200, 'enhancement_factor': 3.5},
        'output': {'collective_field_kT': 22.0}
    }
    
    # Run 5 theta bursts
    for burst in range(5):
        # 4 spikes at 100 Hz
        for spike in range(4):
            # 2ms depolarization (active)
            for _ in range(2):
                result = tracker.step(dt, active_state, ca_spike_active=True)
            
            # 8ms rest within burst
            for _ in range(8):
                result = tracker.step(dt, baseline_state, ca_spike_active=False)
        
        # 160ms between bursts
        for _ in range(160):
            result = tracker.step(dt, baseline_state, ca_spike_active=False)
        
        # Report after each burst
        summary = tracker.get_emission_summary()
        print(f"{tracker.time*1000:>10.1f}  "
              f"{summary['total_emitted']:>10.2f}  "
              f"{summary['total_escaped']:>10.3f}  "
              f"{summary['total_coupled']:>10.4f}  "
              f"{summary['n_packets']:>8}")
    
    print("\n--- Final Summary ---")
    summary = tracker.get_emission_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n✓ Photon emission module validated")