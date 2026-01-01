"""
Quantum Synapse Network
========================

Multi-synapse network with entanglement for coordination experiments.

The key mechanism: synapses that fire together form entangled quantum states.
When reward "measures" the system, all correlated synapses update consistently.

This provides coordination without explicit error backpropagation -
the quantum state carries correlation information that enables
coherent network-level learning from scalar reward signals.

Physics basis:
- Co-active synapses form dimers in similar time windows
- Shared electromagnetic field from tryptophan superradiance couples dimers
- Entanglement creates non-local correlations
- Reward-triggered measurement collapses all correlated states consistently

Author: Sarah Davidson
Quantum Synapse ML Framework
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set
from enum import Enum

from ..quantum_eligibility.physics import QuantumEligibilityPhysics, Isotope
from ..quantum_eligibility.synapse import QuantumSynapse, SynapseParameters

@dataclass
class EntanglementParameters:
    """
    Parameters for inter-synapse entanglement
    
    Based on Model 6 EM coupling module physics.
    """
    # Coupling strength between co-active synapses
    coupling_strength: float = 0.5
    
    # Time window for co-activation to create entanglement (seconds)
    coincidence_window: float = 0.5
    
    # Decay rate of entanglement (separate from individual coherence)
    entanglement_decay_rate: float = 0.01  # Per second
    
    # Threshold for collective effects
    n_entangled_threshold: int = 3  # Minimum for network effects
    
    # Collective enhancement factor (sqrt(N) scaling from superradiance)
    collective_enhancement: bool = True


@dataclass
class NetworkState:
    """
    Network-level state tracking
    """
    # Entanglement matrix (N x N symmetric, off-diagonal = entanglement strength)
    entanglement_matrix: np.ndarray = None
    
    # Recent activation times for coincidence detection
    last_activation_time: np.ndarray = None
    
    # Collective coherence (network-level)
    collective_coherence: float = 0.0
    
    # Statistics
    n_collective_updates: int = 0


class QuantumSynapseNetwork:
    """
    Network of quantum synapses with entanglement
    
    This is the key innovation for solving the coordination problem.
    
    Mechanism:
    1. Synapses activated together become entangled
    2. Entanglement creates correlations in their quantum states
    3. When reward arrives, correlated synapses update coherently
    4. No explicit per-synapse error signal needed
    
    This is fundamentally different from:
    - Independent learners (no coordination)
    - Backprop (requires per-synapse gradients)
    - Message passing (requires explicit communication)
    
    Usage:
        network = QuantumSynapseNetwork(n_synapses=10)
        network.activate([0, 1, 2], strength=1.0)  # Creates entanglement
        network.step(dt=0.1)
        network.apply_reward(reward=1.0)  # Coordinated update
    """
    
    def __init__(self,
                 n_synapses: int,
                 isotope: Isotope = Isotope.P31,
                 entanglement_params: Optional[EntanglementParameters] = None,
                 synapse_params: Optional[SynapseParameters] = None,
                 seed: Optional[int] = None):
        """
        Initialize network
        
        Args:
            n_synapses: Number of synapses
            isotope: Isotope for all synapses
            entanglement_params: Inter-synapse coupling parameters
            synapse_params: Individual synapse parameters
            seed: Random seed
        """
        self.n_synapses = n_synapses
        self.isotope = isotope
        self.ent_params = entanglement_params or EntanglementParameters()
        self.syn_params = synapse_params or SynapseParameters()
        
        # Random state
        self.rng = np.random.default_rng(seed)
        
        # Create synapses
        seeds = self.rng.integers(0, 2**31, size=n_synapses)
        self.synapses = [
            QuantumSynapse(isotope=isotope, params=self.syn_params, seed=int(s))
            for s in seeds
        ]
        
        # Physics (shared)
        self.physics = self.synapses[0].physics
        self.T2 = self.synapses[0].T2
        
        # Network state
        self.state = NetworkState(
            entanglement_matrix=np.zeros((n_synapses, n_synapses)),
            last_activation_time=np.full(n_synapses, -np.inf),
        )
        
        # Time tracking
        self.time = 0.0
        
        # History
        self._history_enabled = False
        self._history: Dict[str, List] = {}
    
    def activate(self, 
                 indices: List[int], 
                 strength: float = 1.0,
                 strengths: Optional[np.ndarray] = None):
        """
        Activate subset of synapses
        
        Co-activated synapses become entangled.
        
        Args:
            indices: Which synapses to activate
            strength: Uniform activation strength
            strengths: Per-synapse strengths (overrides uniform)
        """
        if strengths is None:
            strengths = np.full(len(indices), strength)
        
        # Activate individual synapses
        for idx, s in zip(indices, strengths):
            self.synapses[idx].activate(s)
            self.state.last_activation_time[idx] = self.time
        
        # Create/strengthen entanglement between co-activated synapses
        self._update_entanglement(indices, strengths)
    
    def _update_entanglement(self, active_indices: List[int], strengths: np.ndarray):
        """
        Update entanglement matrix for co-activated synapses
        
        Synapses activated within coincidence window become entangled.
        Entanglement strength depends on activation strengths.
        """
        ep = self.ent_params
        
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                if idx_i >= idx_j:
                    continue  # Only upper triangle
                
                # Check if within coincidence window
                time_diff = abs(self.state.last_activation_time[idx_i] - 
                               self.state.last_activation_time[idx_j])
                
                if time_diff < ep.coincidence_window:
                    # Create entanglement proportional to geometric mean of strengths
                    ent_strength = ep.coupling_strength * np.sqrt(strengths[i] * strengths[j])
                    
                    # Add to existing entanglement (saturates at 1)
                    current = self.state.entanglement_matrix[idx_i, idx_j]
                    new_ent = min(1.0, current + ent_strength)
                    
                    # Symmetric matrix
                    self.state.entanglement_matrix[idx_i, idx_j] = new_ent
                    self.state.entanglement_matrix[idx_j, idx_i] = new_ent
    
    def step(self, dt: float):
        """
        Time evolution for entire network
        
        Both individual coherence and inter-synapse entanglement decay.
        """
        # Individual synapse evolution
        for syn in self.synapses:
            syn.step(dt)
        
        # Entanglement decay
        decay = np.exp(-dt * self.ent_params.entanglement_decay_rate)
        self.state.entanglement_matrix *= decay
        
        # Update collective coherence
        self._update_collective_coherence()
        
        # Time update
        self.time += dt
        
        # History
        if self._history_enabled:
            if 'collective_coherence' in self._history:
                self._history['collective_coherence'].append(self.state.collective_coherence)
            if 'mean_eligibility' in self._history:
                self._history['mean_eligibility'].append(self.get_mean_eligibility())
    
    def _update_collective_coherence(self):
        """
        Calculate network-level collective coherence
        
        Based on entanglement structure and individual coherences.
        """
        # Sum of entanglement
        total_entanglement = np.sum(self.state.entanglement_matrix) / 2  # Off-diagonal sum
        
        # Mean individual coherence (singlet probability)
        mean_singlet = np.mean([s.singlet_prob for s in self.synapses])
        
        # Collective coherence = entanglement × individual coherence
        # With collective enhancement from superradiance
        if self.ent_params.collective_enhancement:
            n_entangled = self._count_entangled_synapses()
            enhancement = np.sqrt(max(1, n_entangled))
        else:
            enhancement = 1.0
        
        self.state.collective_coherence = (total_entanglement / self.n_synapses) * mean_singlet * enhancement
    
    def _count_entangled_synapses(self) -> int:
        """Count synapses with significant entanglement"""
        # A synapse is "entangled" if it has >0.1 entanglement with any other
        threshold = 0.1
        entangled = np.any(self.state.entanglement_matrix > threshold, axis=0)
        return np.sum(entangled)
    
    def apply_reward(self, reward: float) -> np.ndarray:
        """
        Apply reward to network with entanglement-based coordination
        
        Key mechanism: Correlated synapses update together.
        
        The weight update for synapse i is:
            Δw_i = lr × reward × e_i × (1 + Σ_j ent_ij × e_j)
        
        The second term is the coordination bonus: eligible synapses
        that are entangled with other eligible synapses get enhanced updates.
        
        Args:
            reward: Scalar reward signal
            
        Returns:
            Array of weight changes
        """
        eligibilities = self.get_eligibilities()
        ent_matrix = self.state.entanglement_matrix
        lr = self.syn_params.learning_rate
        
        weight_changes = np.zeros(self.n_synapses)
        
        for i, syn in enumerate(self.synapses):
            if eligibilities[i] < self.syn_params.eligibility_threshold:
                continue
            
            # Base weight change
            base_change = lr * reward * eligibilities[i]
            
            # Coordination bonus from entangled synapses
            coord_bonus = 0.0
            for j in range(self.n_synapses):
                if i != j and eligibilities[j] > self.syn_params.eligibility_threshold:
                    coord_bonus += ent_matrix[i, j] * eligibilities[j]
            
            # Total change (bounded coordination bonus)
            total_change = base_change * (1.0 + min(1.0, coord_bonus))
            
            # Apply to synapse
            old_weight = syn.weight
            syn.apply_reward(reward * (1.0 + min(1.0, coord_bonus)), learning_rate=lr)
            weight_changes[i] = syn.weight - old_weight
        
        # Measurement disturbs entanglement
        self.state.entanglement_matrix *= 0.5
        self.state.n_collective_updates += 1
        
        return weight_changes
    
    def get_eligibilities(self) -> np.ndarray:
        """Get all eligibility values"""
        return np.array([s.eligibility for s in self.synapses])
    
    def get_weights(self) -> np.ndarray:
        """Get all weights"""
        return np.array([s.weight for s in self.synapses])
    
    def set_weights(self, weights: np.ndarray):
        """Set all weights"""
        for syn, w in zip(self.synapses, weights):
            syn.state.weight = w
    
    def get_mean_eligibility(self) -> float:
        """Get mean eligibility across network"""
        return np.mean(self.get_eligibilities())
    
    def get_entanglement_graph(self) -> np.ndarray:
        """Get entanglement matrix (for visualization)"""
        return self.state.entanglement_matrix.copy()
    
    def get_eligible_clusters(self, threshold: float = 0.1) -> List[Set[int]]:
        """
        Find clusters of entangled eligible synapses
        
        Returns:
            List of sets, each set contains indices of one cluster
        """
        eligibilities = self.get_eligibilities()
        ent = self.state.entanglement_matrix
        
        # Build adjacency for eligible synapses
        eligible = eligibilities > self.syn_params.eligibility_threshold
        
        # Simple connected components via BFS
        visited = set()
        clusters = []
        
        for start in range(self.n_synapses):
            if start in visited or not eligible[start]:
                continue
            
            # BFS from start
            cluster = set()
            queue = [start]
            
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                
                # Find connected eligible synapses
                for neighbor in range(self.n_synapses):
                    if neighbor not in visited and eligible[neighbor]:
                        if ent[node, neighbor] > threshold:
                            queue.append(neighbor)
            
            clusters.append(cluster)
        
        return clusters
    
    def reset(self):
        """Reset network state (keep weights)"""
        for syn in self.synapses:
            syn.reset()
        
        self.state.entanglement_matrix.fill(0)
        self.state.last_activation_time.fill(-np.inf)
        self.state.collective_coherence = 0.0
        self.time = 0.0
    
    def enable_history(self, variables: Optional[List[str]] = None):
        """Enable history tracking"""
        self._history_enabled = True
        if variables is None:
            variables = ['collective_coherence', 'mean_eligibility']
        self._history = {v: [] for v in variables}
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """Get recorded history"""
        return {k: np.array(v) for k, v in self._history.items()}
    
    def __repr__(self) -> str:
        return (f"QuantumSynapseNetwork(n={self.n_synapses}, "
                f"isotope={self.isotope.value}, T2={self.T2:.1f}s)")


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM SYNAPSE NETWORK TEST")
    print("=" * 60)
    
    # Test 1: Basic network
    print("\n--- Test 1: Basic Network ---")
    net = QuantumSynapseNetwork(n_synapses=5, isotope=Isotope.P31)
    print(net)
    
    # Test 2: Entanglement creation
    print("\n--- Test 2: Entanglement Creation ---")
    net.activate([0, 1, 2], strength=1.0)  # Co-activate 3 synapses
    
    print("Entanglement matrix after co-activation:")
    print(np.round(net.state.entanglement_matrix, 2))
    
    # Test 3: Coordination effect
    print("\n--- Test 3: Coordination vs Independent ---")
    
    # Network with entanglement
    net_ent = QuantumSynapseNetwork(n_synapses=5, isotope=Isotope.P31, seed=42)
    net_ent.activate([0, 1, 2], strength=1.0)  # Create entanglement
    net_ent.step(dt=1.0)
    
    # Network without entanglement (activate separately)
    net_ind = QuantumSynapseNetwork(n_synapses=5, isotope=Isotope.P31, seed=42)
    net_ind.activate([0], strength=1.0)
    net_ind.step(dt=0.6)  # Break coincidence window
    net_ind.activate([1], strength=1.0)
    net_ind.step(dt=0.6)
    net_ind.activate([2], strength=1.0)
    net_ind.step(dt=1.0 - 1.2)  # Total 1 second
    
    # Apply same reward
    changes_ent = net_ent.apply_reward(reward=1.0)
    changes_ind = net_ind.apply_reward(reward=1.0)
    
    print(f"Entangled network weight changes: {np.round(changes_ent, 4)}")
    print(f"Independent activation changes:   {np.round(changes_ind, 4)}")
    print(f"Coordination bonus: {np.sum(changes_ent) / max(0.001, np.sum(changes_ind)):.2f}x")
    
    # Test 4: Isotope effect on network
    print("\n--- Test 4: Isotope Effect ---")
    
    for isotope in [Isotope.P31, Isotope.P32]:
        net = QuantumSynapseNetwork(n_synapses=10, isotope=isotope, seed=123)
        net.activate(list(range(5)), strength=1.0)
        
        # Wait 30 seconds
        for _ in range(300):
            net.step(dt=0.1)
        
        changes = net.apply_reward(reward=1.0)
        
        print(f"{isotope.value}: Mean weight change = {np.mean(changes):.5f}")
    
    # Test 5: Cluster detection
    print("\n--- Test 5: Cluster Detection ---")
    net = QuantumSynapseNetwork(n_synapses=10, seed=42)
    
    # Create two separate clusters
    net.activate([0, 1, 2], strength=1.0)
    net.step(dt=1.0)  # Break coincidence
    net.activate([5, 6, 7, 8], strength=1.0)
    
    clusters = net.get_eligible_clusters()
    print(f"Found {len(clusters)} clusters: {clusters}")
    
    print("\n--- Done ---")