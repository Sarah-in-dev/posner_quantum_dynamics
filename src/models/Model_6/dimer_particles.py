"""
Particle-based dimer tracking with emergent entanglement
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Dimer:
    """Individual calcium phosphate dimer particle"""
    
    id: int
    position: np.ndarray          # (x, y, z) in nm
    birth_time: float             # When formed
    coherence: float = 1.0        # Quantum coherence [0, 1]
    template_bound: bool = False  # Bound to scaffolding protein
    
    # Local environment (updated each step)
    local_j_coupling: float = 0.0
    local_calcium: float = 0.0
    
    def __hash__(self):
        return self.id


@dataclass 
class EntanglementBond:
    """Pairwise entanglement between two dimers"""
    
    dimer_i: int  # ID
    dimer_j: int  # ID
    strength: float = 0.0      # Entanglement strength [0, 1]
    formation_time: float = 0.0
    
    def __hash__(self):
        return hash((min(self.dimer_i, self.dimer_j), 
                     max(self.dimer_i, self.dimer_j)))


class DimerParticleSystem:
    """
    Particle-based tracking of dimers with emergent entanglement
    
    PHYSICS:
    --------
    1. BIRTH: Dimers form from PNC aggregation
       - Rate proportional to [PNC]² × template_enhancement
       - Position sampled from formation probability field
       
    2. DEATH: Dimers dissolve
       - Base rate k_dissolution ~ 0.001/s (very stable)
       - Template-bound: 10× more stable
       - Enhanced by low calcium (thermodynamic drive)
       
    3. COHERENCE: Each dimer has quantum coherence
       - Born with coherence = 1
       - Decays with T2 ~ 100s (J-coupling dependent)
       - Below threshold (0.3), cannot participate in entanglement
       
    4. ENTANGLEMENT: Pairwise, distance-dependent
       - Coupling J_ij ~ 1/r³ (dipole-dipole)
       - Rate: k_ent × J_ij × coherence_i × coherence_j × J_protection
       - Disentanglement from decoherence of either partner
    """
    
    def __init__(self, params, grid_shape=(100, 100), dx=4e-9):
        self.params = params
        self.grid_shape = grid_shape
        self.dx = dx  # Grid spacing in meters
        self.dx_nm = dx * 1e9  # In nanometers
        
        # Particle storage
        self.dimers: List[Dimer] = []
        self.next_id = 0
        self.time = 0.0
        
        # Entanglement network (sparse - only store active bonds)
        self.entanglement_bonds: Set[EntanglementBond] = set()
        self.entanglement_matrix: Optional[np.ndarray] = None  # Dense, for analysis
        
        # Physics parameters
        self.T2_base = 100.0          # s, base coherence time
        self.k_dissolution = 0.001    # 1/s, dissolution rate
        self.k_entangle = 0.5         # 1/s, entanglement attempt rate
        self.coupling_length = 5.0    # nm, characteristic coupling distance
        self.coherence_threshold = 0.3  # Minimum for entanglement
        self.j_coupling_threshold = 5.0  # Hz, minimum for protection
        
        # Formation tracking
        self.formation_rate_field = np.zeros(grid_shape)
        
        # Diagnostics
        self.history = {
            'time': [],
            'n_dimers': [],
            'n_entangled_pairs': [],
            'largest_cluster': [],
            'mean_coherence': [],
            'f_entangled': []
        }
        
        logger.info("DimerParticleSystem initialized")
        logger.info(f"  Max dimers expected: ~100")
        logger.info(f"  Max pairs: ~5000 (trivial computation)")
    
    # =========================================================================
    # BIRTH/DEATH PROCESSES
    # =========================================================================
    
    def step_formation(self, dt: float, formation_rate_field: np.ndarray,
                       template_field: np.ndarray) -> int:
        """
        Stochastic dimer formation from concentration field
        
        Converts continuous formation rate to discrete birth events
        """
        # Total formation rate (integrate over grid)
        # formation_rate_field is in M/s, need to convert to particles/s
        volume_per_voxel = (self.dx ** 3) * 1000  # Liters
        N_A = 6.022e23
        
        particles_per_s = formation_rate_field * volume_per_voxel * N_A
        total_rate = np.sum(particles_per_s)
        
        # Expected births this timestep
        expected_births = total_rate * dt
        
        # Poisson sampling
        n_births = np.random.poisson(expected_births)
        
        # Cap to avoid explosion
        n_births = min(n_births, 10)
        
        # Sample positions for new dimers
        if n_births > 0 and total_rate > 0:
            # Probability distribution over grid
            prob = particles_per_s.flatten() / total_rate
            
            for _ in range(n_births):
                # Sample grid position
                idx = np.random.choice(len(prob), p=prob)
                grid_pos = np.unravel_index(idx, self.grid_shape)
                
                # Convert to physical position (with jitter within voxel)
                pos_nm = np.array([
                    (grid_pos[0] + np.random.random()) * self.dx_nm,
                    (grid_pos[1] + np.random.random()) * self.dx_nm,
                    np.random.random() * self.dx_nm  # Z within single layer
                ])
                
                # Check template binding
                template_bound = template_field[grid_pos] > 50
                
                # Create dimer
                dimer = Dimer(
                    id=self.next_id,
                    position=pos_nm,
                    birth_time=self.time,
                    coherence=1.0,
                    template_bound=template_bound
                )
                
                self.dimers.append(dimer)
                self.next_id += 1
        
        return n_births
    
    def step_dissolution(self, dt: float, calcium_field: np.ndarray) -> int:
        """
        Stochastic dimer dissolution
        """
        to_remove = []
        
        for dimer in self.dimers:
            # Base dissolution rate
            k = self.k_dissolution
            
            # Template binding protects
            if dimer.template_bound:
                k *= 0.1  # 10× more stable
            
            # Low calcium slightly increases dissolution
            # (thermodynamic drive toward ions)
            grid_pos = self._position_to_grid(dimer.position)
            local_ca = calcium_field[grid_pos]
            if local_ca < 1e-6:  # Below 1 µM
                k *= 1.5
            
            # Stochastic death
            p_dissolve = 1 - np.exp(-k * dt)
            if np.random.random() < p_dissolve:
                to_remove.append(dimer)
        
        # Remove dissolved dimers
        for dimer in to_remove:
            self._remove_dimer(dimer)
        
        return len(to_remove)
    
    def _remove_dimer(self, dimer: Dimer):
        """Remove dimer and its entanglement bonds"""
        self.dimers.remove(dimer)
        
        # Remove all entanglement bonds involving this dimer
        bonds_to_remove = [b for b in self.entanglement_bonds 
                          if b.dimer_i == dimer.id or b.dimer_j == dimer.id]
        for bond in bonds_to_remove:
            self.entanglement_bonds.discard(bond)
    
    # =========================================================================
    # COHERENCE DYNAMICS
    # =========================================================================
    
    def step_coherence(self, dt: float, j_coupling_field: np.ndarray):
        """
        Update coherence of each dimer
        
        T2 depends on:
        - Base T2 (~100s for dimers)
        - J-coupling protection
        - Template binding (reduces tumbling)
        """
        for dimer in self.dimers:
            grid_pos = self._position_to_grid(dimer.position)
            j_coupling = j_coupling_field[grid_pos]
            dimer.local_j_coupling = j_coupling
            
            # Effective T2
            # J-coupling protection: T2_eff = T2_base × (1 + J/J_ref)
            J_ref = 10.0  # Hz
            j_protection = 1.0 + j_coupling / J_ref
            
            # Template binding: additional protection
            template_factor = 1.3 if dimer.template_bound else 1.0
            
            T2_eff = self.T2_base * j_protection * template_factor
            
            # Exponential decay with noise
            decay = np.exp(-dt / T2_eff)
            noise = 1.0 + 0.02 * np.random.randn()  # 2% noise
            
            dimer.coherence *= decay * noise
            dimer.coherence = np.clip(dimer.coherence, 0, 1)
    
    # =========================================================================
    # ENTANGLEMENT DYNAMICS - THE KEY PART
    # =========================================================================
    
    def step_entanglement(self, dt: float):
        """
        Update entanglement network based on SHARED J-COUPLING ENVIRONMENT
        
        Physics basis (Fisher 2015):
        - Entanglement inherited from ATP hydrolysis (pyrophosphate → 2Pi)
        - J-coupling field marks regions where coherent phosphates exist
        - Dimers in same J-coupling environment share entanglement
        """
        n = len(self.dimers)
        if n < 2:
            return
        
        for i in range(n):
            for j in range(i + 1, n):
                dimer_i = self.dimers[i]
                dimer_j = self.dimers[j]
                
                # Skip if either below coherence threshold
                if (dimer_i.coherence < self.coherence_threshold or 
                    dimer_j.coherence < self.coherence_threshold):
                    self._remove_bond(dimer_i.id, dimer_j.id)
                    continue
                
                # SHARED J-COUPLING ENVIRONMENT determines entanglement
                # Both must be in strong J-field (from same ATP pool)
                j_min = min(dimer_i.local_j_coupling, dimer_j.local_j_coupling)
                
                if j_min < self.j_coupling_threshold:
                    # Not in coherent environment
                    continue
                
                # Coupling strength from J-field (not distance)
                # Higher J = stronger coherent coupling
                j_coupling_factor = j_min / 20.0  # Normalize to ~1 at 20 Hz
                
                # Coherence factor
                coherence_factor = dimer_i.coherence * dimer_j.coherence
                
                # Effective coupling rate
                k_eff = self.k_entangle * j_coupling_factor * coherence_factor
                
                bond = self._get_bond(dimer_i.id, dimer_j.id)
                
                if bond is None:
                    p_entangle = 1 - np.exp(-k_eff * dt)
                    if np.random.random() < p_entangle:
                        self._create_bond(dimer_i.id, dimer_j.id, strength=coherence_factor)
                else:
                    # Disentanglement if J-coupling drops or coherence lost
                    k_disentangle = 0.1 * (1 - j_coupling_factor * coherence_factor)
                    p_disentangle = 1 - np.exp(-k_disentangle * dt)
                    
                    if np.random.random() < p_disentangle:
                        self._remove_bond(dimer_i.id, dimer_j.id)
                    else:
                        bond.strength = coherence_factor
    
    def _get_bond(self, id_i: int, id_j: int) -> Optional[EntanglementBond]:
        """Get entanglement bond between two dimers"""
        key = (min(id_i, id_j), max(id_i, id_j))
        for bond in self.entanglement_bonds:
            if (min(bond.dimer_i, bond.dimer_j), max(bond.dimer_i, bond.dimer_j)) == key:
                return bond
        return None
    
    def _create_bond(self, id_i: int, id_j: int, strength: float):
        """Create new entanglement bond"""
        bond = EntanglementBond(
            dimer_i=id_i,
            dimer_j=id_j,
            strength=strength,
            formation_time=self.time
        )
        self.entanglement_bonds.add(bond)
    
    def _remove_bond(self, id_i: int, id_j: int):
        """Remove entanglement bond"""
        bond = self._get_bond(id_i, id_j)
        if bond:
            self.entanglement_bonds.discard(bond)
    
    # =========================================================================
    # NETWORK ANALYSIS - FIND ENTANGLED CLUSTERS
    # =========================================================================
    
    def find_entangled_clusters(self) -> List[Set[int]]:
        """
        Find connected components in entanglement graph
        
        Uses union-find for efficiency
        """
        if not self.dimers:
            return []
        
        # Union-find structure
        parent = {d.id: d.id for d in self.dimers}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union all entangled pairs
        for bond in self.entanglement_bonds:
            if bond.dimer_i in parent and bond.dimer_j in parent:
                union(bond.dimer_i, bond.dimer_j)
        
        # Group by root
        clusters = {}
        for d in self.dimers:
            root = find(d.id)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(d.id)
        
        return list(clusters.values())
    
    def get_largest_cluster_size(self) -> int:
        """Size of largest entangled cluster"""
        clusters = self.find_entangled_clusters()
        if not clusters:
            return 0
        return max(len(c) for c in clusters)
    
    def get_network_metrics(self) -> dict:
        """Compute network-level metrics"""
        n_dimers = len(self.dimers)
        n_bonds = len(self.entanglement_bonds)
        clusters = self.find_entangled_clusters()
        largest = max(len(c) for c in clusters) if clusters else 0
        
        # Fraction of dimers in entangled network
        entangled_dimers = set()
        for bond in self.entanglement_bonds:
            entangled_dimers.add(bond.dimer_i)
            entangled_dimers.add(bond.dimer_j)
        
        f_entangled = len(entangled_dimers) / n_dimers if n_dimers > 0 else 0
        
        # Mean coherence
        mean_coherence = np.mean([d.coherence for d in self.dimers]) if self.dimers else 0
        
        return {
            'n_dimers': n_dimers,
            'n_bonds': n_bonds,
            'n_clusters': len(clusters),
            'largest_cluster': largest,
            'f_entangled': f_entangled,
            'mean_coherence': mean_coherence,
            # THE KEY METRIC - dimers in connected entangled network
            'n_network_dimers': largest  
        }
    
    # =========================================================================
    # MAIN STEP
    # =========================================================================
    
    def step(self, dt: float, 
             formation_rate_field: np.ndarray,
             template_field: np.ndarray,
             calcium_field: np.ndarray,
             j_coupling_field: np.ndarray) -> dict:
        """
        Main simulation step
        """
        # 1. Formation (birth)
        n_born = self.step_formation(dt, formation_rate_field, template_field)
        
        # 2. Dissolution (death)
        n_died = self.step_dissolution(dt, calcium_field)
        
        # 3. Coherence decay
        self.step_coherence(dt, j_coupling_field)
        
        # 4. Entanglement dynamics (the expensive part, but still cheap)
        self.step_entanglement(dt)
        
        # 5. Update time
        self.time += dt
        
        # 6. Metrics
        metrics = self.get_network_metrics()
        
        # Record history
        self.history['time'].append(self.time)
        self.history['n_dimers'].append(metrics['n_dimers'])
        self.history['n_entangled_pairs'].append(metrics['n_bonds'])
        self.history['largest_cluster'].append(metrics['largest_cluster'])
        self.history['mean_coherence'].append(metrics['mean_coherence'])
        self.history['f_entangled'].append(metrics['f_entangled'])
        
        return metrics
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _position_to_grid(self, pos_nm: np.ndarray) -> Tuple[int, int]:
        """Convert nm position to grid indices"""
        idx = (pos_nm[:2] / self.dx_nm).astype(int)
        idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
        return tuple(idx)
    
    def get_concentration_field(self) -> np.ndarray:
        """Convert particles back to concentration field for compatibility"""
        field = np.zeros(self.grid_shape)
        
        for dimer in self.dimers:
            grid_pos = self._position_to_grid(dimer.position)
            field[grid_pos] += 1
        
        # Convert count to concentration (M)
        volume_per_voxel = (self.dx ** 3) * 1000  # Liters
        N_A = 6.022e23
        field = field / (volume_per_voxel * N_A)
        
        return field