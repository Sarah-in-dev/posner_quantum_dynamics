"""
Cross-Neuron Entanglement Tracker
==================================

Extends dimer entanglement across neurons via photon transport through
myelin waveguides. This is the key mechanism for network-level coordination.

PHYSICAL MECHANISM:
------------------
Within a single dendrite, dimers become entangled through:
1. Birth correlation (shared pyrophosphate origin)
2. Shared EM field environment (tryptophan superradiance)

Across neurons, the same mechanisms apply but are mediated by photons:
1. Source synapse emits photons during dimer formation
2. Photons carry "birth correlation" information (timing, coherence)
3. Target synapse receives photons during its own dimer formation
4. Target dimers become entangled with source dimers (correlated birth)

The photon acts as a "quantum courier" - extending the correlated EM 
environment across the waveguide connection.

KEY PHYSICS:
-----------
- Entanglement requires temporal coincidence (~0.5s window from ATP burst)
- Entanglement strength depends on:
  * Source dimer coherence (P_S at emission time)
  * Photon transmission efficiency
  * Target dimer coherence (P_S at formation)
  * Timing coincidence factor
- Entanglement decays with T2 of weakest partner

INTEGRATION:
-----------
This module coordinates between:
- PhotonEmissionTracker: Tags packets with source dimer info
- MyelinWaveguide: Transports packets (preserves timing info)
- PhotonReceiver: Reports arrivals for entanglement creation
- DimerParticleSystem: Holds the actual dimers and bonds

Author: Sarah Davidson
Model 6 - Cross-Neuron Extension
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DimerSnapshot:
    """
    Snapshot of a dimer's state at a specific time.
    Used to record source dimer state when photon is emitted.
    """
    dimer_id: int
    synapse_id: int
    neuron_id: int
    singlet_probability: float
    birth_time: float
    snapshot_time: float
    
    @property
    def is_coherent(self) -> bool:
        """Still above entanglement threshold"""
        return self.singlet_probability > 0.5
    
    def __hash__(self):
        return hash((self.neuron_id, self.synapse_id, self.dimer_id))


@dataclass
class CrossNeuronBond:
    """
    Entanglement bond between dimers on different neurons.
    
    Similar to EntanglementBond in dimer_particles.py but tracks
    cross-neuron information.
    """
    # Source dimer info
    source_neuron_id: int
    source_synapse_id: int
    source_dimer_id: int
    
    # Target dimer info
    target_neuron_id: int
    target_synapse_id: int
    target_dimer_id: int
    
    # Bond properties
    strength: float = 0.0           # Entanglement strength [0, 1]
    formation_time: float = 0.0
    
    # For decay tracking
    source_P_S_at_formation: float = 1.0
    target_P_S_at_formation: float = 1.0
    
    def __hash__(self):
        # Canonical ordering for symmetric lookup
        source = (self.source_neuron_id, self.source_synapse_id, self.source_dimer_id)
        target = (self.target_neuron_id, self.target_synapse_id, self.target_dimer_id)
        if source < target:
            return hash((source, target))
        else:
            return hash((target, source))
    
    def __eq__(self, other):
        if not isinstance(other, CrossNeuronBond):
            return False
        return hash(self) == hash(other)
    
    @property 
    def bond_key(self) -> Tuple:
        """Canonical key for lookup"""
        source = (self.source_neuron_id, self.source_synapse_id, self.source_dimer_id)
        target = (self.target_neuron_id, self.target_synapse_id, self.target_dimer_id)
        return (min(source, target), max(source, target))


@dataclass
class PhotonDimerLink:
    """
    Links a photon packet to the source dimers that were coherent at emission.
    Attached to PhotonPacket for transport through waveguide.
    """
    packet_id: int
    emission_time: float
    source_neuron_id: int
    source_synapse_id: int
    
    # Snapshots of coherent dimers at emission time
    dimer_snapshots: List[DimerSnapshot] = field(default_factory=list)
    
    @property
    def n_coherent_dimers(self) -> int:
        return len([d for d in self.dimer_snapshots if d.is_coherent])
    
    @property
    def mean_coherence(self) -> float:
        if not self.dimer_snapshots:
            return 0.0
        return np.mean([d.singlet_probability for d in self.dimer_snapshots])


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass 
class CrossNeuronEntanglementParameters:
    """
    Parameters for cross-neuron entanglement formation.
    
    Based on same physics as within-dendrite entanglement but
    accounting for photon transport.
    """
    
    # === COINCIDENCE WINDOW ===
    # Time window for "birth correlation" via photon
    # Same as ATP hydrolysis burst window
    coincidence_window_s: float = 0.5  # 500ms
    
    # === ENTANGLEMENT FORMATION ===
    # Base rate for attempting entanglement when photon arrives
    k_entangle_base: float = 1.0  # 1/s
    
    # Minimum coherence for participating in entanglement
    coherence_threshold: float = 0.5  # P_S > 0.5 (Agarwal threshold)
    
    # Minimum photons for significant coupling
    min_photons_for_coupling: float = 0.01  # At least 0.01 photons

    # === STRENGTH CALCULATION ===
    # Entanglement strength = base × source_coherence × target_coherence × timing × photon_factor
    
    # How photon count affects strength (sqrt scaling like superradiance)
    photon_scaling: str = 'sqrt'  # 'linear' or 'sqrt'
    
    # Maximum strength achievable
    max_strength: float = 1.0
    
    # === DECAY ===
    # Cross-neuron bonds decay based on weakest partner's T2
    # Additional decay from lack of shared EM environment
    extra_decay_rate: float = 0.01  # 1/s additional decay
    
    # === COLLECTIVE EFFECTS ===
    # Minimum cross-neuron bonds for network effects
    n_bonds_threshold: int = 5


# =============================================================================
# MAIN TRACKER CLASS
# =============================================================================

class CrossNeuronEntanglementTracker:
    """
    Tracks and manages entanglement between dimers across neurons.
    
    USAGE:
    ------
    1. When emitting photons, call tag_packet_with_dimers() to attach
       source dimer information to the photon packet.
    
    2. When photons arrive at target, call process_arrival() with the
       packet and current target dimer state.
    
    3. Call step() each timestep to decay bonds and update statistics.
    
    4. Query get_entangled_pairs() for current cross-neuron entanglement.
    
    EXAMPLE:
    --------
    tracker = CrossNeuronEntanglementTracker()
    
    # At source, when emitting:
    link = tracker.tag_packet_with_dimers(
        packet=photon_packet,
        source_dimers=synapse.dimer_particles.dimers,
        source_neuron_id=0,
        source_synapse_id=0
    )
    packet.dimer_link = link  # Attach to packet
    
    # At target, when receiving:
    new_bonds = tracker.process_arrival(
        packet=delivered_packet,
        target_dimers=target_synapse.dimer_particles.dimers,
        target_neuron_id=1,
        target_synapse_id=0,
        current_time=t
    )
    
    # Each timestep:
    tracker.step(dt, all_dimers_by_location)
    """
    
    def __init__(self, params: Optional[CrossNeuronEntanglementParameters] = None):
        """
        Initialize cross-neuron entanglement tracker.
        
        Args:
            params: Entanglement parameters (uses defaults if None)
        """
        self.params = params or CrossNeuronEntanglementParameters()
        
        # Active cross-neuron bonds
        self.bonds: Set[CrossNeuronBond] = set()
        self._bond_lookup: Dict[Tuple, CrossNeuronBond] = {}
        
        # Statistics
        self.time = 0.0
        self.total_bonds_formed = 0
        self.total_bonds_broken = 0
        
        # History
        self.history = {
            'time': [],
            'n_bonds': [],
            'mean_strength': [],
            'n_neuron_pairs': []
        }
        
        logger.info("CrossNeuronEntanglementTracker initialized")
        logger.info(f"  Coincidence window: {self.params.coincidence_window_s}s")
        logger.info(f"  Coherence threshold: {self.params.coherence_threshold}")
    
    # =========================================================================
    # PHOTON TAGGING (called at source)
    # =========================================================================
    
    def tag_packet_with_dimers(self,
                            packet_id: int,
                            emission_time: float,
                            source_dimers: List,
                            source_neuron_id: int,
                            source_synapse_id: int) -> PhotonDimerLink:
        # Create snapshots of RECENTLY FORMED coherent dimers only
        # (dimers born within coincidence window of emission)
        snapshots = []
        for dimer in source_dimers:
            # Must be coherent
            if dimer.singlet_probability <= self.params.coherence_threshold:
                continue
            
            # Must have formed recently (within coincidence window)
            dimer_age = emission_time - dimer.birth_time
            if dimer_age > self.params.coincidence_window_s or dimer_age < 0:
                continue
            
            snapshot = DimerSnapshot(
                dimer_id=dimer.id,
                synapse_id=source_synapse_id,
                neuron_id=source_neuron_id,
                singlet_probability=dimer.singlet_probability,
                birth_time=dimer.birth_time,
                snapshot_time=emission_time
            )
            snapshots.append(snapshot)
        
        link = PhotonDimerLink(
            packet_id=packet_id,
            emission_time=emission_time,
            source_neuron_id=source_neuron_id,
            source_synapse_id=source_synapse_id,
            dimer_snapshots=snapshots
        )
        
        logger.debug(f"Tagged packet {packet_id} with {len(snapshots)} coherent dimers")
        
        return link
    
    # =========================================================================
    # ARRIVAL PROCESSING (called at target)
    # =========================================================================
    
    def process_arrival(self,
                        dimer_link: PhotonDimerLink,
                        n_photons_delivered: float,
                        target_dimers: List,  # List of Dimer objects
                        target_neuron_id: int,
                        target_synapse_id: int,
                        current_time: float) -> List[CrossNeuronBond]:
        """
        Process photon arrival and create cross-neuron entanglement.
        
        When photons arrive at target during dimer formation, creates
        entanglement bonds between source and target dimers based on
        timing coincidence.
        
        Args:
            dimer_link: The PhotonDimerLink attached to the packet
            n_photons_delivered: Number of photons that made it through
            target_dimers: Current dimers at target synapse
            target_neuron_id: Which neuron is receiving
            target_synapse_id: Which synapse on that neuron
            current_time: Current simulation time
            
        Returns:
            List of newly created CrossNeuronBond objects
        """
        if dimer_link is None:
            return []
        
        if n_photons_delivered < self.params.min_photons_for_coupling:
            return []
        
        new_bonds = []
        p = self.params
        
        # Photon factor (how much the photon count contributes)
        if p.photon_scaling == 'sqrt':
            photon_factor = np.sqrt(n_photons_delivered / p.min_photons_for_coupling)
        else:
            photon_factor = n_photons_delivered / p.min_photons_for_coupling
        photon_factor = min(photon_factor, 2.0)  # Cap at 2x
        
        # Check each source dimer against each target dimer
        for source_snapshot in dimer_link.dimer_snapshots:
            if not source_snapshot.is_coherent:
                continue
            
            for target_dimer in target_dimers:
                if target_dimer.singlet_probability < p.coherence_threshold:
                    continue
                
                # === TIMING COINCIDENCE ===
                # Target dimer should have formed recently (within coincidence window)
                time_since_target_birth = current_time - target_dimer.birth_time
                
                if time_since_target_birth > p.coincidence_window_s:
                    continue  # Target dimer too old
                
                if time_since_target_birth < 0:
                    continue  # Shouldn't happen
                
                # Timing factor: recent formation = stronger entanglement
                timing_factor = 1.0 - (time_since_target_birth / p.coincidence_window_s)
                
                # === CHECK IF BOND ALREADY EXISTS ===
                bond_key = self._make_bond_key(
                    source_snapshot.neuron_id, source_snapshot.synapse_id, source_snapshot.dimer_id,
                    target_neuron_id, target_synapse_id, target_dimer.id
                )
                
                if bond_key in self._bond_lookup:
                    # Strengthen existing bond
                    existing = self._bond_lookup[bond_key]
                    strength_boost = (
                        p.k_entangle_base * 
                        source_snapshot.singlet_probability *
                        target_dimer.singlet_probability *
                        timing_factor *
                        photon_factor *
                        0.1  # Incremental boost
                    )
                    existing.strength = min(p.max_strength, existing.strength + strength_boost)
                    continue
                
                # === CREATE NEW BOND ===
                strength = (
                    source_snapshot.singlet_probability *
                    target_dimer.singlet_probability *
                    timing_factor *
                    photon_factor
                )
                strength = min(strength, p.max_strength)
                
                if strength < 0.1:
                    continue  # Too weak to bother
                
                bond = CrossNeuronBond(
                    source_neuron_id=source_snapshot.neuron_id,
                    source_synapse_id=source_snapshot.synapse_id,
                    source_dimer_id=source_snapshot.dimer_id,
                    target_neuron_id=target_neuron_id,
                    target_synapse_id=target_synapse_id,
                    target_dimer_id=target_dimer.id,
                    strength=strength,
                    formation_time=current_time,
                    source_P_S_at_formation=source_snapshot.singlet_probability,
                    target_P_S_at_formation=target_dimer.singlet_probability
                )
                
                self.bonds.add(bond)
                self._bond_lookup[bond_key] = bond
                self.total_bonds_formed += 1
                new_bonds.append(bond)
                
                logger.debug(
                    f"Created cross-neuron bond: "
                    f"N{source_snapshot.neuron_id}S{source_snapshot.synapse_id}D{source_snapshot.dimer_id} "
                    f"↔ N{target_neuron_id}S{target_synapse_id}D{target_dimer.id} "
                    f"(strength={strength:.3f})"
                )
        
        return new_bonds
    
    def _make_bond_key(self, n1, s1, d1, n2, s2, d2) -> Tuple:
        """Create canonical bond key for lookup"""
        loc1 = (n1, s1, d1)
        loc2 = (n2, s2, d2)
        return (min(loc1, loc2), max(loc1, loc2))
    
    # =========================================================================
    # TIME EVOLUTION
    # =========================================================================
    
    def step(self, 
             dt: float,
             dimer_states: Dict[Tuple[int, int, int], float],
             current_time: float):
        """
        Update cross-neuron entanglement (decay, remove broken bonds).
        
        Args:
            dt: Time step
            dimer_states: Dict mapping (neuron_id, synapse_id, dimer_id) -> P_S
                          For looking up current coherence of bonded dimers
            current_time: Current simulation time
        """
        self.time = current_time
        
        bonds_to_remove = []
        
        for bond in self.bonds:
            # Look up current coherence of both partners
            source_key = (bond.source_neuron_id, bond.source_synapse_id, bond.source_dimer_id)
            target_key = (bond.target_neuron_id, bond.target_synapse_id, bond.target_dimer_id)
            
            source_P_S = dimer_states.get(source_key, 0.0)
            target_P_S = dimer_states.get(target_key, 0.0)
            
            # Bond breaks if either partner loses coherence
            if source_P_S < self.params.coherence_threshold or \
               target_P_S < self.params.coherence_threshold:
                bonds_to_remove.append(bond)
                continue
            
            # Decay bond strength
            # Rate depends on weakest partner + extra decay from spatial separation
            min_P_S = min(source_P_S, target_P_S)
            coherence_decay_rate = (1.0 - min_P_S) * 0.1  # Weaker coherence = faster decay
            total_decay_rate = coherence_decay_rate + self.params.extra_decay_rate
            
            bond.strength *= np.exp(-total_decay_rate * dt)
            
            # Remove if strength too low
            if bond.strength < 0.05:
                bonds_to_remove.append(bond)
        
        # Remove broken bonds
        for bond in bonds_to_remove:
            self.bonds.discard(bond)
            bond_key = bond.bond_key
            if bond_key in self._bond_lookup:
                del self._bond_lookup[bond_key]
            self.total_bonds_broken += 1
        
        # Record history
        self.history['time'].append(current_time)
        self.history['n_bonds'].append(len(self.bonds))
        self.history['mean_strength'].append(self._mean_strength())
        self.history['n_neuron_pairs'].append(self._count_neuron_pairs())
    
    def _mean_strength(self) -> float:
        if not self.bonds:
            return 0.0
        return np.mean([b.strength for b in self.bonds])
    
    def _count_neuron_pairs(self) -> int:
        """Count unique neuron pairs with entanglement"""
        pairs = set()
        for bond in self.bonds:
            pair = (min(bond.source_neuron_id, bond.target_neuron_id),
                    max(bond.source_neuron_id, bond.target_neuron_id))
            pairs.add(pair)
        return len(pairs)
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_bonds_for_neuron(self, neuron_id: int) -> List[CrossNeuronBond]:
        """Get all cross-neuron bonds involving a specific neuron"""
        return [b for b in self.bonds 
                if b.source_neuron_id == neuron_id or b.target_neuron_id == neuron_id]
    
    def get_bonds_between_neurons(self, neuron_a: int, neuron_b: int) -> List[CrossNeuronBond]:
        """Get bonds between two specific neurons"""
        return [b for b in self.bonds
                if (b.source_neuron_id == neuron_a and b.target_neuron_id == neuron_b) or
                   (b.source_neuron_id == neuron_b and b.target_neuron_id == neuron_a)]
    
    def get_entanglement_matrix(self, neuron_ids: List[int]) -> np.ndarray:
        """
        Get neuron-level entanglement matrix.
        
        Returns NxN matrix where entry (i,j) is total entanglement
        strength between neurons i and j.
        """
        n = len(neuron_ids)
        id_to_idx = {nid: i for i, nid in enumerate(neuron_ids)}
        
        matrix = np.zeros((n, n))
        
        for bond in self.bonds:
            if bond.source_neuron_id in id_to_idx and bond.target_neuron_id in id_to_idx:
                i = id_to_idx[bond.source_neuron_id]
                j = id_to_idx[bond.target_neuron_id]
                matrix[i, j] += bond.strength
                matrix[j, i] += bond.strength  # Symmetric
        
        return matrix
    
    def get_network_metrics(self) -> Dict:
        """Get summary metrics for cross-neuron entanglement"""
        return {
            'n_bonds': len(self.bonds),
            'mean_strength': self._mean_strength(),
            'n_neuron_pairs': self._count_neuron_pairs(),
            'total_formed': self.total_bonds_formed,
            'total_broken': self.total_bonds_broken,
            'above_threshold': len(self.bonds) >= self.params.n_bonds_threshold
        }
    
    def get_coordination_factor(self, neuron_ids: List[int]) -> float:
        """
        Calculate coordination factor for reward distribution.
        
        This is used when applying reward - neurons with cross-neuron
        entanglement should update coherently.
        
        Returns value in [0, 1] indicating degree of network coordination.
        """
        if len(neuron_ids) < 2:
            return 0.0
        
        matrix = self.get_entanglement_matrix(neuron_ids)
        
        # Coordination = normalized sum of off-diagonal elements
        n = len(neuron_ids)
        off_diagonal_sum = np.sum(matrix) - np.trace(matrix)
        max_possible = n * (n - 1)  # If all pairs had strength 1.0
        
        if max_possible == 0:
            return 0.0
        
        return min(1.0, off_diagonal_sum / max_possible)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CROSS-NEURON ENTANGLEMENT TRACKER VALIDATION")
    print("=" * 70)
    
    # Mock dimer class for testing
    @dataclass
    class MockDimer:
        id: int
        birth_time: float
        singlet_probability: float = 0.9
    
    tracker = CrossNeuronEntanglementTracker()
    
    # Test 1: Tag packet with dimers
    print("\n--- Test 1: Tag Packet with Dimers ---")
    source_dimers = [
        MockDimer(id=0, birth_time=0.0, singlet_probability=0.95),
        MockDimer(id=1, birth_time=0.1, singlet_probability=0.90),
        MockDimer(id=2, birth_time=0.2, singlet_probability=0.40),  # Below threshold
    ]
    
    link = tracker.tag_packet_with_dimers(
        packet_id=0,
        emission_time=0.5,
        source_dimers=source_dimers,
        source_neuron_id=0,
        source_synapse_id=0
    )
    
    print(f"  Source dimers: {len(source_dimers)}")
    print(f"  Coherent snapshots: {len(link.dimer_snapshots)}")
    print(f"  Mean coherence: {link.mean_coherence:.3f}")
    assert len(link.dimer_snapshots) == 2, "Should only snapshot coherent dimers"
    print("  ✓ Correct filtering of coherent dimers")
    
    # Test 2: Process arrival and create bonds
    print("\n--- Test 2: Process Arrival ---")
    target_dimers = [
        MockDimer(id=100, birth_time=0.55, singlet_probability=0.85),  # Recent
        MockDimer(id=101, birth_time=0.60, singlet_probability=0.80),  # Recent
        MockDimer(id=102, birth_time=0.01, singlet_probability=0.70),  # Old
    ]
    
    new_bonds = tracker.process_arrival(
        dimer_link=link,
        n_photons_delivered=50.0,
        target_dimers=target_dimers,
        target_neuron_id=1,
        target_synapse_id=0,
        current_time=0.6
    )
    
    print(f"  New bonds created: {len(new_bonds)}")
    print(f"  Total bonds: {len(tracker.bonds)}")
    
    for bond in new_bonds:
        print(f"    D{bond.source_dimer_id} ↔ D{bond.target_dimer_id}: {bond.strength:.3f}")
    
    # Should create bonds: 2 source × 2 recent target = 4 bonds
    # (Old target dimer 102 should be excluded)
    assert len(new_bonds) == 4, f"Expected 4 bonds, got {len(new_bonds)}"
    print("  ✓ Correct bond creation with timing filter")
    
    # Test 3: Bond decay
    print("\n--- Test 3: Bond Decay ---")
    initial_strength = list(tracker.bonds)[0].strength
    
    # Create dimer state dict
    dimer_states = {
        (0, 0, 0): 0.90,  # Source dimer 0
        (0, 0, 1): 0.85,  # Source dimer 1
        (1, 0, 100): 0.80,  # Target dimer 100
        (1, 0, 101): 0.75,  # Target dimer 101
    }
    
    # Step forward 10 seconds
    for _ in range(100):
        tracker.step(dt=0.1, dimer_states=dimer_states, current_time=tracker.time + 0.1)
    
    final_strength = list(tracker.bonds)[0].strength if tracker.bonds else 0
    print(f"  Initial strength: {initial_strength:.3f}")
    print(f"  After 10s: {final_strength:.3f}")
    print(f"  Remaining bonds: {len(tracker.bonds)}")
    assert final_strength < initial_strength, "Bond should decay"
    print("  ✓ Bonds decay over time")
    
    # Test 4: Bond breaks when dimer loses coherence
    print("\n--- Test 4: Bond Breaking ---")
    # Set one dimer below threshold
    dimer_states[(1, 0, 100)] = 0.3  # Below 0.5 threshold
    
    tracker.step(dt=0.1, dimer_states=dimer_states, current_time=tracker.time + 0.1)
    
    print(f"  Bonds after coherence loss: {len(tracker.bonds)}")
    print(f"  Total bonds broken: {tracker.total_bonds_broken}")
    print("  ✓ Bonds break when partner loses coherence")
    
    # Test 5: Entanglement matrix
    print("\n--- Test 5: Entanglement Matrix ---")
    matrix = tracker.get_entanglement_matrix([0, 1])
    print(f"  Matrix shape: {matrix.shape}")
    print(f"  N0-N1 entanglement: {matrix[0, 1]:.3f}")
    
    # Test 6: Coordination factor
    print("\n--- Test 6: Coordination Factor ---")
    coord = tracker.get_coordination_factor([0, 1])
    print(f"  Coordination factor: {coord:.3f}")
    
    # Test 7: Network metrics
    print("\n--- Test 7: Network Metrics ---")
    metrics = tracker.get_network_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)