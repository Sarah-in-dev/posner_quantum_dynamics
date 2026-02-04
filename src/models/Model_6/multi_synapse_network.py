"""
Multi-Synapse Network Module
============================

Implements realistic multi-synapse architecture where:
- Each synapse is an independent Model6 instance with its own grid
- Synapses are positioned along a dendritic segment
- Quantum fields couple through the shared microtubule network
- Network-level threshold determines commitment

ARCHITECTURE:
------------
```
Dendrite shaft (with microtubules running through)
    │
    ├── Spine 1 (Model6 instance, own 100x100 grid) ─┐
    │       └── ~5 dimers, own calcium dynamics      │
    ├── Spine 2 (Model6 instance, own 100x100 grid)  │
    │       └── ~5 dimers, own calcium dynamics      ├── Within 20µm
    ├── ...                                          │   Fields couple via MT
    ├── Spine N (Model6 instance, own 100x100 grid) ─┘
    │       └── ~5 dimers, own calcium dynamics
```

KEY PHYSICS:
-----------
1. Each synapse independently produces ~4-6 dimers (stochastic)
2. Dimers create local quantum fields
3. Fields propagate through dendritic microtubules (decay with distance)
4. Network field = sum of all synapse contributions (with distance weighting)
5. Commitment occurs when network field exceeds threshold (~50 dimers worth)

This replaces the "multiply by N" hack with actual multi-synapse physics.

"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynapseState:
    """State of a single synapse in the network"""
    position_um: np.ndarray  # (x, y, z) position in microns
    dimer_count: float = 0.0
    coherence: float = 0.0
    collective_field_kT: float = 0.0
    eligibility: float = 0.0
    committed: bool = False
    committed_level: float = 0.0
    calcium_peak_uM: float = 0.0


@dataclass 
class NetworkState:
    """Aggregate state of the multi-synapse network"""
    n_synapses: int
    total_dimers: float = 0.0
    network_field_kT: float = 0.0
    mean_coherence: float = 0.0
    mean_eligibility: float = 0.0
    n_committed: int = 0
    network_committed: bool = False
    network_commitment_level: float = 0.0
    synapse_states: List[SynapseState] = field(default_factory=list)

class NetworkEntanglementTracker:
    """
    Tracks entanglement ACROSS synapses in a multi-synapse network
    
    Physics basis (Fisher 2015):
    - Entanglement from shared ATP hydrolysis pool
    - Nearby synapses share dendritic J-coupling environment
    - Coupling decays with inter-synapse distance
    
    This replaces the naive sum of per-synapse entangled counts
    with true network-level entanglement tracking.
    """
    
    def __init__(self, coupling_length_um: float = 5.0, 
                 j_coupling_threshold: float = 5.0,
                 coherence_threshold: float = 0.3):
        self.coupling_length_um = coupling_length_um
        self.j_coupling_threshold = j_coupling_threshold
        self.coherence_threshold = coherence_threshold
        
        # Network state
        self.all_dimers = []  # List of (dimer, synapse_idx, synapse_pos)
        self.entanglement_bonds = set()  # (global_id_i, global_id_j)
        
    def collect_dimers(self, synapses: List, positions: np.ndarray):
        """
        Collect all dimers from all synapses with position info
        
        Parameters
        ----------
        synapses : List[Model6QuantumSynapse]
            All synapse models in network
        positions : np.ndarray
            Synapse positions in microns, shape (n_synapses, 3)
        """
        self.all_dimers = []
        global_id = 0
        
        for syn_idx, synapse in enumerate(synapses):
            if not hasattr(synapse, 'dimer_particles'):
                continue
                
            particle_system = synapse.dimer_particles
            synapse_pos = positions[syn_idx]
            
            for dimer in particle_system.dimers:
                self.all_dimers.append({
                    'global_id': global_id,
                    'dimer': dimer,
                    'synapse_idx': syn_idx,
                    'synapse_pos_um': synapse_pos,
                    'local_j': dimer.local_j_coupling,
                    'coherence': dimer.coherence,
                    'P_S': dimer.singlet_probability  # <-- ADD THIS LINE
                })
                global_id += 1
    
    def step(self, dt: float, synapses: List, positions: np.ndarray) -> dict:
        """
        Update network-level entanglement
        
        Returns
        -------
        dict with network metrics
        """
        # Collect current dimers from all synapses
        self.collect_dimers(synapses, positions)
        
        n = len(self.all_dimers)
        if n < 2:
            return {
                'n_total_dimers': n,
                'n_entangled_network': n if n == 1 else 0,
                'n_bonds': 0,
                'f_entangled': 0.0
            }
        
        # Update entanglement bonds
        self._update_entanglement(dt)
        
        # Find largest connected cluster
        largest_cluster = self._find_largest_cluster()
        
        # Compute fraction
        entangled_ids = set()
        for bond in self.entanglement_bonds:
            entangled_ids.add(bond[0])
            entangled_ids.add(bond[1])
        
        f_entangled = len(entangled_ids) / n if n > 0 else 0.0
        
        return {
            'n_total_dimers': n,
            'n_entangled_network': len(largest_cluster),
            'n_bonds': len(self.entanglement_bonds),
            'f_entangled': f_entangled
        }
    
    def _update_entanglement(self, dt: float):
        """
        Update pairwise entanglement with cross-synapse coupling
        """
        k_entangle = 0.5  # Base rate
        
        # Clean up bonds for dimers that no longer exist
        current_ids = {d['global_id'] for d in self.all_dimers}
        self.entanglement_bonds = {b for b in self.entanglement_bonds 
                                    if b[0] in current_ids and b[1] in current_ids}
        
        n = len(self.all_dimers)
        
        for i in range(n):
            for j in range(i + 1, n):
                d_i = self.all_dimers[i]
                d_j = self.all_dimers[j]
                
                # Skip if either below coherence threshold
                if (d_i['coherence'] < self.coherence_threshold or
                    d_j['coherence'] < self.coherence_threshold):
                    self._remove_bond(d_i['global_id'], d_j['global_id'])
                    continue
                
                # Calculate coupling strength
                j_coupling_factor = self._calculate_coupling(d_i, d_j)
                
                if j_coupling_factor < 0.1:
                    # Too weak to maintain entanglement
                    self._remove_bond(d_i['global_id'], d_j['global_id'])
                    continue
                
                # Coherence factor
                coherence_factor = d_i['coherence'] * d_j['coherence']
                
                # Effective rate
                k_eff = k_entangle * j_coupling_factor * coherence_factor
                
                # Check current bond status
                bond_exists = self._has_bond(d_i['global_id'], d_j['global_id'])
                
                if not bond_exists:
                    # Try to form bond
                    p_entangle = 1 - np.exp(-k_eff * dt)
                    if np.random.random() < p_entangle:
                        self._add_bond(d_i['global_id'], d_j['global_id'])
                else:
                    # Check for disentanglement
                    k_disentangle = 0.1 * (1 - j_coupling_factor * coherence_factor)
                    p_disentangle = 1 - np.exp(-k_disentangle * dt)
                    if np.random.random() < p_disentangle:
                        self._remove_bond(d_i['global_id'], d_j['global_id'])
    
    def _calculate_coupling(self, d_i: dict, d_j: dict) -> float:
        """
        Calculate effective coupling between two dimers
        
        Same synapse: use J-coupling directly
        Different synapses: J-coupling × distance decay
        """
        j_local_i = d_i['local_j']
        j_local_j = d_j['local_j']
        
        # Both need sufficient J-coupling
        if j_local_i < self.j_coupling_threshold or j_local_j < self.j_coupling_threshold:
            return 0.0
        
        j_factor = min(j_local_i, j_local_j) / 20.0  # Normalize
        
        if d_i['synapse_idx'] == d_j['synapse_idx']:
            # Same synapse - full coupling
            return j_factor
        else:
            # Different synapses - distance-dependent
            dist_um = np.linalg.norm(d_i['synapse_pos_um'] - d_j['synapse_pos_um'])
            
            # Exponential decay with distance
            # At coupling_length_um, factor = 1/e ≈ 0.37
            spatial_factor = np.exp(-dist_um / self.coupling_length_um)
            
            return j_factor * spatial_factor
    
    def get_synapse_correlation_matrix(self, synapses: List) -> np.ndarray:
        """
        Compute correlation matrix between synapses based on entanglement.
        
        Physics: C_ij = E_ij × P_S_i × P_S_j

        Quantum derivation: For two Werner states with singlet probabilities P_S_i, P_S_j,
        the inter-dimer bond fidelity undergoes independent decoherence in each environment.
        Measurement correlation scales linearly with bond fidelity = P_S_i × P_S_j.
        
        Parameters
        ----------
        synapses : List[Model6QuantumSynapse]
            All synapse models in network
            
        Returns
        -------
        np.ndarray : Correlation matrix (n_synapses × n_synapses)
        """
        n_syn = len(synapses)
        C = np.zeros((n_syn, n_syn))
        
        if not self.all_dimers:
            return C
        
        # Get mean P_singlet per synapse
        P_singlet = []
        for syn in synapses:
            if hasattr(syn, 'dimer_particles') and syn.dimer_particles.dimers:
                mean_ps = np.mean([d.singlet_probability for d in syn.dimer_particles.dimers])
            else:
                mean_ps = 0.25  # Thermal
            P_singlet.append(mean_ps)
        P_singlet = np.array(P_singlet)
        
        # Count cross-synapse bonds
        # Build lookup: global_id -> synapse_idx
        dimer_to_synapse = {d['global_id']: d['synapse_idx'] for d in self.all_dimers}
        
        # Count bonds between each synapse pair
        bond_counts = np.zeros((n_syn, n_syn))
        for id_i, id_j in self.entanglement_bonds:
            syn_i = dimer_to_synapse.get(id_i, -1)
            syn_j = dimer_to_synapse.get(id_j, -1)
            if syn_i >= 0 and syn_j >= 0 and syn_i != syn_j:
                bond_counts[syn_i, syn_j] += 1
                bond_counts[syn_j, syn_i] += 1
        
        # Count dimers per synapse for normalization
        dimers_per_synapse = np.zeros(n_syn)
        for d in self.all_dimers:
            dimers_per_synapse[d['synapse_idx']] += 1
        
        # Build correlation matrix
        for i in range(n_syn):
            for j in range(i+1, n_syn):
                # Skip if either synapse has no coherent dimers
                if P_singlet[i] < 0.5 or P_singlet[j] < 0.5:
                    continue
                
                # E_ij = normalized bond count
                max_bonds = min(dimers_per_synapse[i], dimers_per_synapse[j])
                if max_bonds > 0:
                    E_ij = bond_counts[i, j] / max_bonds
                else:
                    E_ij = 0.0
                
                # C_ij = E_ij × P_S_i × P_S_j (product form - quantum mechanically derived)
                coherence_factor = P_singlet[i] * P_singlet[j]
                C[i, j] = E_ij * coherence_factor
                C[j, i] = C[i, j]
        
        # Clamp to valid correlation range [0, 1]
        np.fill_diagonal(C, 0.0)  # No self-correlation
        C = np.clip(C, 0.0, 1.0)

        # PSD safeguard for multivariate normal sampling
        # (rarely needed - only if bond topology is inconsistent)
        if n_syn > 1:
            eigenvalues = np.linalg.eigvalsh(C + np.eye(n_syn))  # Add identity for full correlation matrix
            if np.min(eigenvalues) < 1e-10:
                # Project to nearest PSD: clip negative eigenvalues
                eigenvalues_full, eigenvectors = np.linalg.eigh(C + np.eye(n_syn))
                eigenvalues_full = np.maximum(eigenvalues_full, 1e-10)
                C_full = eigenvectors @ np.diag(eigenvalues_full) @ eigenvectors.T
                C = C_full - np.eye(n_syn)  # Remove identity to get off-diagonal
                C = np.clip(C, 0.0, 1.0)
                np.fill_diagonal(C, 0.0)

        return C


    def get_coordination_factor(self, synapses: List) -> float:
        """
        Get overall coordination factor for network.
        
        Returns value in [0, 1] indicating degree of cross-synapse entanglement.
        """
        C = self.get_synapse_correlation_matrix(synapses)
        n = len(synapses)
        if n < 2:
            return 0.0
        
        # Sum of off-diagonal normalized by maximum possible
        off_diag_sum = np.sum(C) - np.trace(C)
        max_possible = n * (n - 1)
        
        return min(1.0, off_diag_sum / max_possible)
    
    
    
    def _find_largest_cluster(self) -> set:
        """
        Find largest connected component using union-find
        """
        if not self.all_dimers:
            return set()
        
        # Union-find
        parent = {d['global_id']: d['global_id'] for d in self.all_dimers}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union all bonded pairs
        for id_i, id_j in self.entanglement_bonds:
            if id_i in parent and id_j in parent:
                union(id_i, id_j)
        
        # Group by root
        clusters = {}
        for d in self.all_dimers:
            root = find(d['global_id'])
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(d['global_id'])
        
        # Return largest
        if not clusters:
            return set()
        return max(clusters.values(), key=len)
    
    def _has_bond(self, id_i: int, id_j: int) -> bool:
        key = (min(id_i, id_j), max(id_i, id_j))
        return key in self.entanglement_bonds
    
    def _add_bond(self, id_i: int, id_j: int):
        key = (min(id_i, id_j), max(id_i, id_j))
        self.entanglement_bonds.add(key)
    
    def _remove_bond(self, id_i: int, id_j: int):
        key = (min(id_i, id_j), max(id_i, id_j))
        self.entanglement_bonds.discard(key)


    def perform_quantum_measurement(self, synapses: List) -> np.ndarray:
        """
        Only the ENTANGLED NETWORK determines eligibility.
        Unbonded dimers don't count.
        """
        
        # Find all dimers in the entangled network (those with bonds)
        bonded_dimer_ids = set()
        for id_i, id_j in self.entanglement_bonds:
            bonded_dimer_ids.add(id_i)
            bonded_dimer_ids.add(id_j)
        
        # Average P_S of the entangled network
        network_P_S_values = [d['P_S'] for d in self.all_dimers 
                            if d['global_id'] in bonded_dimer_ids]
        
        if not network_P_S_values:
            # No entangled network → no quantum eligibility
            return np.zeros(len(synapses))
        
        network_P_S = np.mean(network_P_S_values)
        
        # THE NETWORK COLLAPSES AS A UNIT
        # Single roll for the entire entangled network
        network_outcome = np.random.random() < network_P_S  # True = singlet
        
        # All bonded dimers get the SAME outcome
        dimer_outcomes = {}
        for gid in bonded_dimer_ids:
            dimer_outcomes[gid] = network_outcome
        
        # Unbonded dimers DON'T CONTRIBUTE to eligibility
        # (or collapse independently but don't count)
        
        # Synapse eligibility = fraction of its dimers in the network that are singlet
        # If a synapse has no bonded dimers, eligibility = 0
        n_syn = len(synapses)
        eligibilities = np.zeros(n_syn)
        
        for d in self.all_dimers:
            if d['global_id'] in bonded_dimer_ids:
                syn_idx = d['synapse_idx']
                if network_outcome:
                    eligibilities[syn_idx] += 1  # Count bonded singlets
        
        # Normalize by total bonded dimers per synapse
        bonded_per_synapse = np.zeros(n_syn)
        for d in self.all_dimers:
            if d['global_id'] in bonded_dimer_ids:
                bonded_per_synapse[d['synapse_idx']] += 1
        
        for i in range(n_syn):
            if bonded_per_synapse[i] > 0:
                eligibilities[i] /= bonded_per_synapse[i]
        
        return eligibilities

    def perform_independent_measurement(self, synapses: List) -> np.ndarray:
        """
        Control condition: Measure all dimers INDEPENDENTLY (no bond correlation).
        
        Same physics for individual dimers, but bonds don't cause joint collapse.
        This is the classical control - what happens without quantum coordination.
        
        Returns
        -------
        np.ndarray : Eligibility for each synapse [0, 1]
        """
        n_syn = len(synapses)
        
        if not self.all_dimers:
            return np.zeros(n_syn)
        
        # Build P_S lookup
        dimer_P_S = {}
        for d in self.all_dimers:
            dimer_P_S[d['global_id']] = d.get('P_S', 0.25)
        
        # Measure ALL dimers independently (bonds ignored)
        dimer_outcomes = {}
        for d in self.all_dimers:
            gid = d['global_id']
            P_S = dimer_P_S.get(gid, 0.25)
            dimer_outcomes[gid] = np.random.random() < P_S
        
        # Compute eligibilities
        eligible_count = np.zeros(n_syn)
        total_count = np.zeros(n_syn)
        
        for d in self.all_dimers:
            syn_idx = d['synapse_idx']
            if syn_idx < n_syn:
                total_count[syn_idx] += 1
                if dimer_outcomes.get(d['global_id'], False):
                    eligible_count[syn_idx] += 1
        
        eligibilities = np.zeros(n_syn)
        for i in range(n_syn):
            if total_count[i] > 0:
                eligibilities[i] = eligible_count[i] / total_count[i]
        
        # Store for diagnostics
        self._last_measurement = {
            'total_dimers': int(np.sum(total_count)),
            'total_bonds': len(self.entanglement_bonds),
            'singlet_outcomes': int(np.sum(eligible_count)),
            'eligibilities': eligibilities.copy()
        }
        
        return eligibilities


class MultiSynapseNetwork:
    """
    Manages N independent synapses with realistic spatial coupling
    
    Each synapse is a full Model6 instance. They couple through:
    1. Shared dendritic voltage (electrical coupling)
    2. Quantum field summation through microtubules
    3. Network-level commitment threshold
    
    Parameters
    ----------
    n_synapses : int
        Number of synapses in the network
    spacing_um : float
        Average spacing between synapses in microns
    pattern : str
        Spatial arrangement: 'linear', 'clustered', 'distributed'
    coupling_length_um : float
        Length constant for field coupling (default 5 µm)
    commitment_threshold_dimers : float
        Network dimer threshold for commitment (default 25)
    """
    
    def __init__(self,
                 n_synapses: int = 10,
                 spacing_um: float = 2.0,
                 pattern: str = 'linear',
                 coupling_length_um: float = 5.0,
                 field_threshold_kT: float = 20.0,
                 params=None,
                 use_correlated_sampling: bool = True):  # ADD THIS
    
        self.n_synapses = n_synapses
        self.spacing_um = spacing_um
        self.pattern = pattern
        self.coupling_length_um = coupling_length_um
        self.field_threshold_kT = field_threshold_kT
        self.params = params
        self.use_correlated_sampling = use_correlated_sampling
        self.disable_auto_commitment = False 

        # Generate synapse positions
        self.positions = self._generate_positions()
        
        # Compute distance matrix and coupling weights
        self.distances = self._compute_distances()
        self.coupling_weights = self._compute_coupling_weights()
        
        # Create individual synapse models (lazy initialization)
        self.synapses: List = []  # Will hold Model6 instances
        self._initialized = False
        
        # Network state
        self.network_committed = False
        self.network_commitment_level = 0.0
        self.time = 0.0
        
        # Network-level entanglement tracking
        self.entanglement_tracker = NetworkEntanglementTracker(
            coupling_length_um=coupling_length_um
        )
        
        # History
        self.history = {
            'time': [],
            'total_dimers': [],
            'network_field': [],
            'n_committed': [],
            'synapse_dimers': []  # List of lists
        }
        
        logger.info(f"MultiSynapseNetwork created: {n_synapses} synapses, "
                   f"{pattern} pattern, {spacing_um}µm spacing")
    
    def _generate_positions(self) -> np.ndarray:
        """Generate synapse positions based on pattern"""
        
        if self.pattern == 'linear':
            # Synapses along a straight dendrite
            positions = np.zeros((self.n_synapses, 3))
            positions[:, 0] = np.arange(self.n_synapses) * self.spacing_um
            # Small perpendicular jitter (spines don't align perfectly)
            positions[:, 1] = np.random.randn(self.n_synapses) * 0.2
            positions[:, 2] = np.random.randn(self.n_synapses) * 0.2
            
        elif self.pattern == 'clustered':
            # Synapses clustered in a small region
            positions = np.random.randn(self.n_synapses, 3) * self.spacing_um * 0.5
            
        elif self.pattern == 'distributed':
            # Synapses spread across multiple branches
            positions = np.random.uniform(
                -self.spacing_um * 3, 
                self.spacing_um * 3, 
                (self.n_synapses, 3)
            )
            
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        return positions
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise distances between synapses"""
        n = self.n_synapses
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                distances[i, j] = d
                distances[j, i] = d
        
        return distances
    
    def _compute_coupling_weights(self) -> np.ndarray:
        """
        Compute coupling weights based on distance
        
        Quantum fields decay exponentially along dendritic microtubules.
        Weight_ij = exp(-d_ij / λ) where λ is coupling length constant
        """
        weights = np.exp(-self.distances / self.coupling_length_um)
        # Self-coupling is 1.0
        np.fill_diagonal(weights, 1.0)
        return weights
    
    
    def initialize(self, ModelClass, base_params=None):
        """
        Initialize individual synapse models
        
        Parameters
        ----------
        ModelClass : class
            The Model6QuantumSynapse class to instantiate
        base_params : Model6Parameters, optional
            Base parameters (will be copied for each synapse)
        """
        from copy import deepcopy
        
        self.synapses = []
        
        for i in range(self.n_synapses):
            # Each synapse gets its own parameters (for independent stochasticity)
            if base_params is not None:
                params = deepcopy(base_params)
            else:
                params = None
            
            # Create model instance
            model = ModelClass(params=params)
            model._network_controlled = True 
            
            # Store position info
            model._network_position = self.positions[i]
            model._network_index = i
            
            self.synapses.append(model)
        
        self._initialized = True
        logger.info(f"Initialized {self.n_synapses} Model6 instances")
    
    def disable_network_commitment(self):
        """Disable network-level field commitment for coordination experiments"""
        self.field_threshold_kT = float('inf')
    
    def configure_all(self, **kwargs):
        """Apply configuration to all synapses"""
        for synapse in self.synapses:
            for key, value in kwargs.items():
                if hasattr(synapse, key):
                    setattr(synapse, key, value)
                elif hasattr(synapse, f'set_{key}'):
                    getattr(synapse, f'set_{key}')(value)
    
    def set_microtubule_invasion(self, invaded: bool):
        """Set MT invasion state for all synapses"""
        for synapse in self.synapses:
            synapse.set_microtubule_invasion(invaded)
    
    def step(self, dt: float, stimulus: Dict) -> NetworkState:
        """
        Step all synapses and compute network state
        
        Parameters
        ----------
        dt : float
            Time step in seconds
        stimulus : dict
            Stimulus parameters (applied to all synapses)
            
        Returns
        -------
        NetworkState
            Aggregate network state
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized. Call initialize() first.")
        
        # Step each synapse independently
        synapse_states = []
        
        for i, synapse in enumerate(self.synapses):
            # Each synapse gets the same stimulus
            # (In future, could have synapse-specific stimuli)
            synapse.step(dt, stimulus)
            
            # Get dimer count directly from particle system (fast)
            if hasattr(synapse, 'dimer_particles'):
                dimer_count = len(synapse.dimer_particles.dimers)
            else:
                dimer_count = 0

            state = SynapseState(
                position_um=self.positions[i],
                dimer_count=dimer_count,
                coherence=synapse.get_mean_singlet_probability() if hasattr(synapse, 'get_mean_singlet_probability') else 0.0,
                collective_field_kT=getattr(synapse, '_collective_field_kT', 0.0),
                eligibility=getattr(synapse, '_current_eligibility', 0.0),
                committed=getattr(synapse, '_camkii_committed', False),
                committed_level=getattr(synapse, '_committed_memory_level', 0.0),
                calcium_peak_uM=np.max(synapse.calcium.get_concentration()) * 1e6,
            )
            synapse_states.append(state)
        
        # Update network-level entanglement
        # Only update entanglement every 10 steps (expensive O(n²) operation)
        if not hasattr(self, '_entanglement_step_counter'):
            self._entanglement_step_counter = 0
        self._entanglement_step_counter += 1

        if self._entanglement_step_counter % 10 == 0:
            self._network_entanglement = self.entanglement_tracker.step(
                dt, self.synapses, self.positions
            )
        
        # =========================================================================
        # GAP 3: COORDINATED THREE-FACTOR GATE EVALUATION
        # =========================================================================
        # When reward is present, evaluate gates with correlated sampling.
        # This ensures entangled synapses commit/fail together.
        if stimulus.get('reward', False):
            if self.use_correlated_sampling:
                print(f"[DEBUG] reward={stimulus.get('reward')}, use_correlated_sampling={self.use_correlated_sampling}, type={type(self.use_correlated_sampling)}")
                self._evaluate_coordinated_gate(stimulus)
            else:
                self._evaluate_independent_gate(stimulus)
        # =========================================================================
        
        # Compute network-level quantities
        network_state = self._compute_network_state(synapse_states)
        
        # Check for network commitment every step
        # (commitment is irreversible once field exceeds threshold with eligibility)
        # SKIP if disable_auto_commitment is True (for coordination experiments)
        if not self.disable_auto_commitment:
            self._check_network_commitment(network_state)
        
        # Update time
        self.time += dt
        
        # Record history
        self._record_history(network_state)
        
        return network_state
    
    def _compute_network_state(self, synapse_states: List[SynapseState]) -> NetworkState:
        """
        Compute aggregate network state with proper field coupling
        
        The key physics:
        - Each synapse contributes its dimers to the network
        - Fields couple through microtubules with distance-dependent decay
        - Total network field determines commitment
        """
        
        # Raw totals
        dimer_counts = np.array([s.dimer_count for s in synapse_states])
        coherences = np.array([s.coherence for s in synapse_states])
        eligibilities = np.array([s.eligibility for s in synapse_states])
        
        # Simple sum of dimers (each synapse contributes independently)
        total_dimers = np.sum(dimer_counts)
        
        # Mean coherence (weighted by dimer count)
        if total_dimers > 0:
            mean_coherence = np.sum(coherences * dimer_counts) / total_dimers
        else:
            mean_coherence = 0.0
        
        # Mean eligibility
        mean_eligibility = np.mean(eligibilities)
        
        # Network field from physics (emergent, not prescribed)
        #
        # Physics basis:
        # - Single coherent dimer: U_single = 6.6 kT (Fisher 2015)
        # - Entanglement fraction: f_ent = 0.3 (only 30% form coherent network)
        # - Coherent summation: √N_ent enhancement
        # - CaMKII is local to dimers (no spatial reduction)
        #
        # U_network = U_single × √(f_ent × N_coherent)
        # For N=50, coherence=0.85: √(0.3 × 42.5) × 6.6 = 23.6 kT ✓
        
        U_single_kT = 6.6  # From Fisher 2015
    
        # Get TRUE entangled network from cross-synapse tracker
        if hasattr(self, '_network_entanglement') and self._network_entanglement:
            n_entangled = self._network_entanglement['n_entangled_network']
            n_bonds = self._network_entanglement['n_bonds']
        else:
            n_entangled = 0
            n_bonds = 0
        
        # === Q2 FIELD: From entangled dimers ===
        N_collective_threshold = 35.0
        
        if n_entangled >= N_collective_threshold:
            n_possible_pairs = n_entangled * (n_entangled - 1) // 2
            if n_possible_pairs > 0:
                f_ent_emergent = n_bonds / n_possible_pairs
            else:
                f_ent_emergent = 0.0
            effective_N = f_ent_emergent * n_entangled
            q2_field = U_single_kT * np.sqrt(effective_N)
        else:
            q2_field = 0.0
        
        # === Q1 FIELD: From tryptophan networks ===
        # Each synapse has its own Q1 field; network field is the mean
        q1_fields = [s.collective_field_kT for s in synapse_states]
        q1_field = np.mean(q1_fields) if q1_fields else 0.0
        
        # === TOTAL NETWORK FIELD ===
        # Q1 and Q2 are coupled systems - use the dominant contribution
        network_field = max(q1_field, q2_field)
        
        # Count committed synapses
        n_committed = sum(1 for s in synapse_states if s.committed)
        
        return NetworkState(
            n_synapses=self.n_synapses,
            total_dimers=total_dimers,
            network_field_kT=network_field,
            mean_coherence=mean_coherence,
            mean_eligibility=mean_eligibility,
            n_committed=n_committed,
            network_committed=self.network_committed,
            network_commitment_level=self.network_commitment_level,
            synapse_states=synapse_states
        )
    
    
    def sample_correlated_eligibilities(self, rng: np.random.Generator = None) -> np.ndarray:
        """
        Sample eligibilities with quantum correlations for reward application.
        
        This is the "measurement" - entangled synapses get correlated outcomes.
        
        Returns
        -------
        np.ndarray : Sampled eligibilities for each synapse
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Get mean eligibilities from each synapse
        mean_elig = np.array([s.get_eligibility() for s in self.synapses])
        
        # Get correlation matrix from entanglement
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        C = self.entanglement_tracker.get_synapse_correlation_matrix(self.synapses)
        
        # Build covariance matrix
        base_var = 0.2  # Variance scale for quantum fluctuations
        cov = np.eye(self.n_synapses) * base_var
        
        for i in range(self.n_synapses):
            for j in range(i+1, self.n_synapses):
                cov[i, j] = C[i, j] * base_var
                cov[j, i] = cov[i, j]
        
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < 0:
            cov += np.eye(self.n_synapses) * (-np.min(eigvals) + 1e-6)
        
        # Sample from multivariate normal
        sampled = rng.multivariate_normal(mean_elig, cov)
        
        return np.clip(sampled, 0, 1)


    def apply_reward_correlated(self, reward: float, learning_rate: float = 0.05) -> np.ndarray:
        """
        Apply reward with entanglement-based coordination.
        
        Correlated synapses get similar eligibility samples, producing
        coordinated weight updates without backpropagation.
        
        Parameters
        ----------
        reward : float
            Scalar reward signal (can be negative)
        learning_rate : float
            Learning rate for weight updates
            
        Returns
        -------
        np.ndarray : Weight changes for each synapse
        """
        # Sample eligibilities with correlations
        sampled_elig = self.sample_correlated_eligibilities()
        
        # Compute weight changes
        delta_w = learning_rate * reward * sampled_elig
        
        # Apply to synapses (need to implement weight storage)
        # This hooks into the CaMKII commitment mechanism
        for i, syn in enumerate(self.synapses):
            if hasattr(syn, '_committed_memory_level'):
                syn._committed_memory_level += delta_w[i]
                syn._committed_memory_level = np.clip(syn._committed_memory_level, 0, 1)
        
        return delta_w
    
    
    def _check_network_commitment(self, state: NetworkState):
        """
        Check if network crosses commitment threshold
        
        EMERGENT from physics:
        - Network field must exceed thermal noise (~20 kT)
        - This naturally requires ~50 coherent dimers
        - No fitted parameters!
        """
        if self.network_committed:
            return  # Already committed, can't uncommit
        
        # Physics-based threshold: field must overcome thermal fluctuations
        if state.network_field_kT >= self.field_threshold_kT:
            if state.mean_eligibility > 0.3:
                self.network_committed = True
                # Commitment level scales with field strength above threshold
                excess = state.network_field_kT / self.field_threshold_kT
                self.network_commitment_level = min(1.0, state.mean_eligibility * excess)
                
                logger.info(f"Network COMMITTED: field={state.network_field_kT:.1f} kT "
                          f"(threshold={self.field_threshold_kT} kT), "
                          f"dimers={state.total_dimers:.1f}, "
                          f"level={self.network_commitment_level:.2f}")
    
    def step_with_coordination(self, dt: float, stimulus: dict) -> NetworkState:
        """
        Step network with coordinated three-factor gate evaluation.
        
        When reward is present, use correlated sampling to decide which
        synapses open their plasticity gates. Entangled synapses tend to
        commit together or fail together.
        
        This replaces independent gate evaluation with coordinated "measurement".
        """
        reward_present = stimulus.get('reward', False)
        
        # Step individual synapses (calcium, dimers, etc.)
        for synapse in self.synapses:
            synapse.step(dt, stimulus)
        
        # Update network entanglement tracking
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        ent_metrics = self.entanglement_tracker.step(dt, self.synapses, self.positions)
        self._network_entanglement = ent_metrics
        
        # === COORDINATED THREE-FACTOR GATE ===
        if reward_present:
            self._evaluate_coordinated_gate(stimulus)
        
        # Update network state
        self.time += dt
        self._record_history()
        
        return self._get_network_state()


    def _evaluate_coordinated_gate(self, stimulus: dict):
        """
        Evaluate three-factor gate with QUANTUM MEASUREMENT.
        
        Physics: When reward arrives, the quantum state is "measured".
        - Bonded dimers collapse TOGETHER (perfect correlation)
        - Unbonded dimers collapse independently
        - Synapse eligibility = fraction of dimers that collapsed to singlet
        
        The gate has three factors:
        1. Eligibility: majority of dimers collapsed to singlet (quantum)
        2. Dopamine: reward signal present (classical)
        3. Calcium: postsynaptic activity (classical)
        """
        dopamine_present = stimulus.get('reward', False)
        if not dopamine_present:
            return
        
        # Ensure dimer registry is current
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        
        # === QUANTUM MEASUREMENT ===
        # Bonded dimers collapse together; unbonded collapse independently
        measured_eligibilities = self.entanglement_tracker.perform_quantum_measurement(
            self.synapses
        )
        
        # Diagnostic output
        measurement = getattr(self.entanglement_tracker, '_last_measurement', {})
        n_bonds = measurement.get('total_bonds', 0)
        n_dimers = measurement.get('total_dimers', 0)
        n_singlet = measurement.get('singlet_outcomes', 0)
        
        logger.info(f"QUANTUM MEASUREMENT: {n_dimers} dimers, {n_bonds} bonds, "
                   f"{n_singlet} singlet outcomes")
        logger.info(f"  Measured eligibilities: {[f'{e:.2f}' for e in measured_eligibilities]}")
        
        # Gate threshold: eligibility > 0.5 means majority collapsed to singlet
        elig_threshold = 0.5
        
        # === THREE-FACTOR GATE EVALUATION ===
        for i, syn in enumerate(self.synapses):
            # Calcium factor
            calcium_uM = getattr(syn, '_peak_calcium_uM', 0.0)
            calcium_elevated = calcium_uM > 0.5  # µM threshold
            
            # Three-factor AND gate
            gate_open = (
                measured_eligibilities[i] > elig_threshold and  # Quantum eligibility
                dopamine_present and                             # Reward signal
                calcium_elevated                                 # Postsynaptic activity
            )
            
            syn._plasticity_gate = gate_open
            
            # Commitment
            if gate_open and not getattr(syn, '_camkii_committed', False):
                syn._camkii_committed = True
                syn._commitment_time = self.time
                
                # Commitment level scales with measured eligibility
                syn._committed_memory_level = measured_eligibilities[i]
        
        # === POST-MEASUREMENT: Entanglement is consumed ===
        # Measurement collapses the quantum state
        if hasattr(self, '_apply_measurement_collapse'):
            self._apply_measurement_collapse()


    def _apply_measurement_collapse(self):
        """
        Apply measurement collapse to entanglement network.
        
        Physics: Quantum measurement destroys superposition.
        After reward-triggered "collapse", cross-synapse entanglement
        bonds are weakened or broken.
        
        This is a one-time effect when reward is applied.
        """
        # Reduce bond strengths significantly (measurement effect)
        # Don't completely destroy - some residual correlation remains
        collapse_factor = 0.3  # Bonds reduced to 30% strength
        
        tracker = self.entanglement_tracker
        
        # For bonds, we can't easily modify strength (they're in a set)
        # Instead, we accelerate decay by marking measurement time
        self._last_measurement_time = self.time
        
        # Alternative: remove some bonds probabilistically
        # Bonds between synapses that BOTH committed remain (correlated collapse)
        # Bonds where one committed and one didn't break (decoherence)
        
        committed_synapses = set()
        for i, syn in enumerate(self.synapses):
            if getattr(syn, '_camkii_committed', False):
                committed_synapses.add(i)
        
        if not hasattr(tracker, 'entanglement_bonds'):
            return
        
        # Build dimer-to-synapse lookup
        if not tracker.all_dimers:
            return
        
        dimer_to_synapse = {d['global_id']: d['synapse_idx'] for d in tracker.all_dimers}
        
        # Find bonds to break
        bonds_to_remove = set()
        for bond in tracker.entanglement_bonds:
            id_i, id_j = bond
            syn_i = dimer_to_synapse.get(id_i, -1)
            syn_j = dimer_to_synapse.get(id_j, -1)
            
            if syn_i < 0 or syn_j < 0:
                continue
            
            # Break bond if commitment was discordant (one yes, one no)
            i_committed = syn_i in committed_synapses
            j_committed = syn_j in committed_synapses
            
            if i_committed != j_committed:
                # Discordant - bond breaks with high probability
                if np.random.random() < 0.8:
                    bonds_to_remove.add(bond)
            else:
                # Concordant - bond weakens but persists
                # (handled by accelerated decay)
                pass
        
        # Remove discordant bonds
        for bond in bonds_to_remove:
            tracker.entanglement_bonds.discard(bond)
    
    
    def _evaluate_independent_gate(self, stimulus: dict):
        """
        Control condition: Measure all dimers INDEPENDENTLY.
        
        Same quantum physics for individual dimers (P_S determines collapse probability),
        but bonds DON'T cause joint collapse. This is what happens without entanglement-
        based coordination.
        
        Key difference from coordinated gate:
        - Coordinated: Bonded dimers collapse together (correlation = 1.0)
        - Independent: All dimers collapse independently (correlation = 0)
        """
        dopamine_present = stimulus.get('reward', False)
        if not dopamine_present:
            return
        
        # Ensure dimer registry is current
        self.entanglement_tracker.collect_dimers(self.synapses, self.positions)
        
        # === INDEPENDENT MEASUREMENT ===
        # All dimers collapse independently (bonds ignored)
        measured_eligibilities = self.entanglement_tracker.perform_independent_measurement(
            self.synapses
        )
        
        # Diagnostic output
        measurement = getattr(self.entanglement_tracker, '_last_measurement', {})
        n_dimers = measurement.get('total_dimers', 0)
        n_singlet = measurement.get('singlet_outcomes', 0)
        
        logger.info(f"INDEPENDENT MEASUREMENT: {n_dimers} dimers, "
                   f"{n_singlet} singlet outcomes (bonds ignored)")
        logger.info(f"  Measured eligibilities: {[f'{e:.2f}' for e in measured_eligibilities]}")
        
        # Gate threshold
        elig_threshold = 0.5
        
        # === THREE-FACTOR GATE EVALUATION ===
        for i, syn in enumerate(self.synapses):
            # Calcium factor
            calcium_uM = getattr(syn, '_peak_calcium_uM', 0.0)
            calcium_elevated = calcium_uM > 0.5
            
            # Three-factor AND gate
            gate_open = (
                measured_eligibilities[i] > elig_threshold and
                dopamine_present and
                calcium_elevated
            )
            
            syn._plasticity_gate = gate_open
            
            # Commitment
            if gate_open and not getattr(syn, '_camkii_committed', False):
                syn._camkii_committed = True
                syn._commitment_time = self.time
                syn._committed_memory_level = measured_eligibilities[i]
    
    def _record_history(self, state: NetworkState):
        """Record network state to history"""
        self.history['time'].append(self.time)
        self.history['total_dimers'].append(state.total_dimers)
        self.history['network_field'].append(state.network_field_kT)
        self.history['n_committed'].append(state.n_committed)
        self.history['synapse_dimers'].append(
            [s.dimer_count for s in state.synapse_states]
        )
    
    def get_experimental_metrics(self) -> Dict:
        """Get metrics for experimental comparison"""
        
        if not self.synapses:
            return {}
        
        # Collect from all synapses
        dimer_counts = [getattr(s, '_previous_dimer_count', 0) for s in self.synapses]
        coherences = [getattr(s, '_previous_coherence', 0) for s in self.synapses]
        fields = [getattr(s, '_collective_field_kT', 0) for s in self.synapses]
        
        # Cross-synapse entanglement metrics (THE KEY SPATIAL EFFECT)
        ent = self._network_entanglement if hasattr(self, '_network_entanglement') and self._network_entanglement else {}
        n_entangled_network = ent.get('n_entangled_network', 0)
        n_total_bonds = ent.get('n_bonds', 0)
        
        # Count within vs cross-synapse bonds
        within_bonds = 0
        cross_bonds = 0
        if hasattr(self, 'entanglement_tracker') and self.entanglement_tracker.all_dimers:
            dimer_synapse = {d['global_id']: d['synapse_idx'] for d in self.entanglement_tracker.all_dimers}
            for bond in self.entanglement_tracker.entanglement_bonds:
                id1, id2 = bond
                if dimer_synapse.get(id1) == dimer_synapse.get(id2):
                    within_bonds += 1
                else:
                    cross_bonds += 1
        
        # Q2 field from entangled dimer network
        U_single_kT = 6.6
        q2_field = U_single_kT * np.sqrt(n_entangled_network) if n_entangled_network > 0 else 0.0
        
        return {
            'n_synapses': self.n_synapses,
            'total_dimers': sum(dimer_counts),
            'mean_dimers_per_synapse': np.mean(dimer_counts),
            'std_dimers_per_synapse': np.std(dimer_counts),
            'mean_coherence': np.mean(coherences),
            'mean_field_kT': np.mean(fields),  # Q1 field (local tryptophan)
            'q2_field_kT': q2_field,  # Q2 field (entangled dimer network)
            'n_entangled_network': n_entangled_network,
            'within_synapse_bonds': within_bonds,
            'cross_synapse_bonds': cross_bonds,
            'total_bonds': n_total_bonds,
            'network_committed': self.network_committed,
            'network_commitment_level': self.network_commitment_level,
            'pattern': self.pattern,
            'spacing_um': self.spacing_um,
            'coupling_length_um': self.coupling_length_um
        }
    
    
    def set_coordination_mode(self, enable: bool = True):
        """
        Configure network for coordination experiments.
        
        When enabled:
        - Disables automatic network commitment (field threshold check)
        - Commitment only happens via three-factor gate at reward time
        - Correlated sampling enabled for entangled synapses
        
        When disabled:
        - Normal operation with field-threshold commitment
        
        Parameters
        ----------
        enable : bool
            True for coordination experiments, False for normal operation
        """
        self.disable_auto_commitment = enable
        self.use_correlated_sampling = enable
        
        if enable:
            logger.info("Coordination mode ENABLED: commitment only via reward gate")
        else:
            logger.info("Coordination mode DISABLED: normal operation")
    
    
    def reset(self):
        """Reset network state (but keep synapses and coordination mode)"""
        self.network_committed = False
        self.network_commitment_level = 0.0
        self.time = 0.0
        # Note: disable_auto_commitment and use_correlated_sampling are preserved
        self.history = {
            'time': [],
            'total_dimers': [],
            'network_field': [],
            'n_committed': [],
            'synapse_dimers': []
        }
        
        # Reset individual synapses
        for synapse in self.synapses:
            synapse._camkii_committed = False
            synapse._committed_memory_level = 0.0


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def create_network_from_condition(condition, ModelClass, params=None):
    """
    Factory function to create MultiSynapseNetwork from ExperimentCondition
    
    Parameters
    ----------
    condition : ExperimentCondition
        Experiment condition with n_synapses, spatial_pattern, etc.
    ModelClass : class
        Model6QuantumSynapse class
    params : Model6Parameters, optional
        Base parameters
        
    Returns
    -------
    MultiSynapseNetwork
        Configured network
    """
    network = MultiSynapseNetwork(
        n_synapses=condition.n_synapses,
        spacing_um=condition.synapse_spacing_um,
        pattern=condition.spatial_pattern,
        commitment_threshold_dimers=25.0  # From theory
    )
    
    network.initialize(ModelClass, params)
    
    # Apply condition-specific settings
    network.set_microtubule_invasion(condition.mt_invaded)
    
    return network


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-SYNAPSE NETWORK VALIDATION")
    print("=" * 70)
    
    # Test network creation
    print("\n1. Creating network with 10 synapses...")
    network = MultiSynapseNetwork(
        n_synapses=10,
        spacing_um=2.0,
        pattern='linear'
    )
    
    print(f"   Positions shape: {network.positions.shape}")
    print(f"   Distance matrix shape: {network.distances.shape}")
    print(f"   Coupling weights shape: {network.coupling_weights.shape}")
    
    print("\n2. Synapse positions (µm):")
    for i, pos in enumerate(network.positions):
        print(f"   Synapse {i}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print("\n3. Distance statistics:")
    distances = network.distances[np.triu_indices(10, k=1)]
    print(f"   Mean distance: {np.mean(distances):.2f} µm")
    print(f"   Min distance: {np.min(distances):.2f} µm")
    print(f"   Max distance: {np.max(distances):.2f} µm")
    
    print("\n4. Coupling statistics:")
    weights = network.coupling_weights[np.triu_indices(10, k=1)]
    print(f"   Mean coupling: {np.mean(weights):.3f}")
    print(f"   Min coupling: {np.min(weights):.3f}")
    print(f"   Max coupling: {np.max(weights):.3f}")
    
    print("\n5. Testing different patterns:")
    for pattern in ['linear', 'clustered', 'distributed']:
        net = MultiSynapseNetwork(n_synapses=10, pattern=pattern)
        dist = net.distances[np.triu_indices(10, k=1)]
        coup = net.coupling_weights[np.triu_indices(10, k=1)]
        print(f"   {pattern:12s}: mean_dist={np.mean(dist):.2f}µm, "
              f"mean_coupling={np.mean(coup):.3f}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nTo use with Model6:")
    print("  from multi_synapse_network import MultiSynapseNetwork")
    print("  from model6_core import Model6QuantumSynapse")
    print("  ")
    print("  network = MultiSynapseNetwork(n_synapses=10)")
    print("  network.initialize(Model6QuantumSynapse, params)")
    print("  network.set_microtubule_invasion(True)")
    print("  ")
    print("  for t in range(n_steps):")
    print("      state = network.step(dt, {'voltage': -0.01, 'reward': False})")
    print("      print(f'Total dimers: {state.total_dimers:.1f}')")