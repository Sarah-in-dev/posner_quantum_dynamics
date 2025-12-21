"""
Particle-based dimer tracking with emergent entanglement
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# NEW:
@dataclass
class Dimer:
    """
    Individual calcium phosphate dimer particle
    
    Quantum state tracked via singlet probability (Agarwal et al. 2023)
    rather than phenomenological T2 decay.
    
    Key physics:
    - 4 ³¹P spins per dimer, forming 2 singlet pairs from pyrophosphate
    - P_S = 1.0: pure singlet (maximally entangled)
    - P_S > 0.5: entanglement preserved
    - P_S = 0.25: thermalized (no entanglement)
    """
    
    id: int
    position: np.ndarray          # (x, y, z) in nm
    birth_time: float             # When formed
    template_bound: bool = False  # Bound to scaffolding protein
    
    # === SINGLET STATE (replaces phenomenological coherence) ===
    singlet_probability: float = 1.0  # P_S: 0.25 (thermal) to 1.0 (pure singlet)
    
    # Intra-dimer J-coupling constants (Hz)
    # 6 values for 4 spins: pairs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Initialized from Agarwal DFT distribution: mean ~0.15 Hz, range -0.5 to +0.5 Hz
    j_couplings_intra: np.ndarray = field(default=None)
    
    # Local environment (updated each step)
    local_j_coupling: float = 0.0  # External J-field (ATP-derived)
    local_calcium: float = 0.0
    
    def __post_init__(self):
        if self.j_couplings_intra is None:
            # DFT values from Agarwal Table 11: mean ~0.15 Hz, std ~0.15 Hz
            self.j_couplings_intra = np.random.normal(0.15, 0.15, size=6)
            self.j_couplings_intra = np.clip(self.j_couplings_intra, -0.5, 0.5)
    
    @property
    def is_entangled(self) -> bool:
        """Entanglement preserved when P_S > 0.5 (Agarwal threshold)"""
        return self.singlet_probability > 0.5
    
    @property
    def coherence(self) -> float:
        """Backward compatibility: map singlet probability to 0-1 scale"""
        # P_S = 0.25 → coherence = 0
        # P_S = 1.0 → coherence = 1
        return (self.singlet_probability - 0.25) / 0.75
    
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
        # self.T2_base = 100.0          # s, base coherence time
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
    
    def step_population(self, dt: float, dimer_concentration: np.ndarray,
                        template_field: np.ndarray) -> dict:
        """
        Adjust particle population to track concentration field
        
        Physics:
        - Chemistry determines HOW MANY dimers exist (fast equilibrium)
        - Particle system tracks WHICH ONES and their quantum state
        - Birth/death maintains population, coherence evolves with T2
        - Hysteresis prevents rapid turnover from concentration noise
        """
        # Calculate target particle count from concentration
        # Use active zone volume (0.01 μm³) - matches established dimer calculations
        az_volume_L = 1e-17  # 0.01 μm³ active zone (Bhalla 2004)
        N_A = 6.022e23
        
        # Peak concentration at templates × active zone volume
        peak_conc = np.max(dimer_concentration)
        target_count = int(round(peak_conc * az_volume_L * N_A))
        
        current_count = len(self.dimers)
        n_births = 0
        n_deaths = 0
        
        # Hysteresis: only adjust if difference > 1 particle
        # This prevents rapid turnover from concentration noise
        difference = target_count - current_count
        
        # --- BIRTH: Add particles if significantly below target ---
        if difference >= 2:
            n_to_add = difference - 1  # Leave buffer
            
            conc_weighted = dimer_concentration * (1 + template_field * 10)
            total_weight = np.sum(conc_weighted)
            
            if total_weight > 0:
                prob = conc_weighted.flatten() / total_weight
                
                for _ in range(n_to_add):
                    idx = np.random.choice(len(prob), p=prob)
                    grid_pos = np.unravel_index(idx, self.grid_shape)
                    
                    pos_nm = np.array([
                        (grid_pos[0] + np.random.random()) * self.dx_nm,
                        (grid_pos[1] + np.random.random()) * self.dx_nm,
                        np.random.random() * 20.0
                    ])
                    
                    template_bound = template_field[grid_pos] > 0.5
                    
                    dimer = Dimer(
                        id=self.next_id,
                        position=pos_nm,
                        birth_time=self.time,
                        template_bound=template_bound
                    )
                    
                    self.dimers.append(dimer)
                    self.next_id += 1
                    n_births += 1
                    
                    # INHERITED ENTANGLEMENT (Fisher 2015):
                    # Phosphates from same pyrophosphate hydrolysis are born entangled
                    # Check existing dimers for shared origin
                    if template_bound:
                        birth_window = 0.1  # 100ms - same ATP burst
                        for other in self.dimers[:-1]:  # Exclude just-added dimer
                            if other.template_bound and other.is_entangled:
                                if abs(other.birth_time - dimer.birth_time) < birth_window:
                                    # Shared origin - inherit entanglement
                                    strength = other.singlet_probability * dimer.singlet_probability
                                    self._create_bond(dimer.id, other.id, strength=strength)
        
        # --- DEATH: Remove particles if significantly above target ---
        elif difference <= -2:
            n_to_remove = abs(difference) - 1  # Leave buffer
            
            sorted_dimers = sorted(self.dimers, key=lambda d: d.coherence)
            
            for i in range(min(n_to_remove, len(sorted_dimers))):
                dimer = sorted_dimers[i]
                self.dimers.remove(dimer)
                self._remove_all_bonds_for_dimer(dimer.id)
                n_deaths += 1
        
        return {'n_births': n_births, 'n_deaths': n_deaths}
    
    def _remove_all_bonds_for_dimer(self, dimer_id: int):
        """Remove all entanglement bonds involving a specific dimer"""
        to_remove = [bond for bond in self.entanglement_bonds 
                    if bond.dimer_i == dimer_id or bond.dimer_j == dimer_id]
        for bond in to_remove:
            self.entanglement_bonds.discard(bond)
    
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
    
    # NEW:
    def step_coherence(self, dt: float, j_coupling_field: np.ndarray):
        """
        Update singlet probability for each dimer
        
        Physics from Agarwal et al. 2023:
        - Singlet decay driven by J-coupling frequency spread (destructive interference)
        - Dimers (4 spins) have fewer frequencies → slower decay than trimers (6 spins)
        - Characteristic singlet lifetime: 100-1000s for dimers
        - Entanglement threshold: P_S > 0.5
        """
        # Singlet dynamics parameters (from Agarwal 2023)
        P_S_thermal = 0.25  # Thermalized (maximally mixed) state
        
        # ISOTOPE-SPECIFIC singlet lifetimes
        # P31: T_singlet ~216s (Agarwal - dipolar relaxation only)
        # P32: T_singlet ~0.4s (quadrupolar relaxation dominates)
        T_singlet_P31 = 216.0  # s
        T_singlet_P32 = 0.4    # s
        
        # Get isotope fraction from params
        fraction_P31 = getattr(self.params.environment, 'fraction_P31', 1.0)
        
        for dimer in self.dimers:
            grid_pos = self._position_to_grid(dimer.position)
            j_external = j_coupling_field[grid_pos]
            dimer.local_j_coupling = j_external
            
            # === SINGLET PROBABILITY DECAY ===
            # Isotope-weighted singlet lifetime
            T_singlet_base = fraction_P31 * T_singlet_P31 + (1 - fraction_P31) * T_singlet_P32
            
            # J-coupling spread determines decay rate
            j_spread = np.std(dimer.j_couplings_intra)
            j_mean = np.abs(np.mean(dimer.j_couplings_intra))
            
            # Spread factor: uniform J → slow decay, spread J → faster decay
            spread_factor = 1.0 + 2.0 * j_spread / (j_mean + 0.1)
            
            # Template binding reduces tumbling → slower dipolar relaxation
            template_factor = 0.7 if dimer.template_bound else 1.0
            
            # Effective singlet lifetime
            T_singlet_eff = T_singlet_base / (spread_factor * template_factor)
            T_singlet_eff = max(T_singlet_eff, 0.1)  # Minimum 0.1s for P32
            
            # Decay toward thermal equilibrium
            # P_S(t) = P_thermal + (P_S(0) - P_thermal) * exp(-t/T)
            decay_factor = np.exp(-dt / T_singlet_eff)
            
            # Add small stochastic noise
            noise = 1.0 + 0.01 * np.sqrt(dt) * np.random.randn()
            
            # Update singlet probability
            P_excess = dimer.singlet_probability - P_S_thermal
            dimer.singlet_probability = P_S_thermal + P_excess * decay_factor * noise
            
            # Clamp to valid range
            dimer.singlet_probability = np.clip(dimer.singlet_probability, P_S_thermal, 1.0)
    
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
                
                # Skip if either loses singlet state (P_S > 0.5 required)
                if not dimer_i.is_entangled or not dimer_j.is_entangled:
                    self._remove_bond(dimer_i.id, dimer_j.id)
                    continue
                
                # SHARED ORIGIN determines entanglement (Fisher 2015):
                # Phosphates from same pyrophosphate hydrolysis event
                # 1. Both template-bound (same ATPase/pyrophosphatase region)
                if not (dimer_i.template_bound and dimer_j.template_bound):
                    continue
                
                # 2. Formed within same temporal window (same activity burst)
                birth_window = 0.1  # 100ms - one ATP hydrolysis burst
                if abs(dimer_i.birth_time - dimer_j.birth_time) > birth_window:
                    continue
                
                # Coupling strength from shared origin
                time_factor = 1.0 - abs(dimer_i.birth_time - dimer_j.birth_time) / birth_window
                
                # Coherence factor (singlet probability product)
                coherence_factor = dimer_i.singlet_probability * dimer_j.singlet_probability
                
                # Effective coupling rate
                k_eff = self.k_entangle * time_factor * coherence_factor
                
                bond = self._get_bond(dimer_i.id, dimer_j.id)
                
                if bond is None:
                    p_entangle = 1 - np.exp(-k_eff * dt)
                    if np.random.random() < p_entangle:
                        self._create_bond(dimer_i.id, dimer_j.id, strength=coherence_factor)
                else:
                    # Disentanglement if singlet probability drops
                    k_disentangle = 0.1 * (1 - coherence_factor)
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
             dimer_concentration: np.ndarray,
             template_field: np.ndarray,
             calcium_field: np.ndarray,
             j_coupling_field: np.ndarray) -> dict:
        """
        Main simulation step
        
        Parameters
        ----------
        dimer_concentration : np.ndarray
            Dimer concentration field (M) from ca_triphosphate
        template_field : np.ndarray
            Template enhancement field
        calcium_field : np.ndarray
            Calcium concentration (M) - for future use
        j_coupling_field : np.ndarray
            J-coupling field (Hz) from ATP system
        """
        # 1. Update time first
        self.time += dt
        
        # 2. Population: birth/death to track concentration (FAST chemistry)
        pop_result = self.step_population(dt, dimer_concentration, template_field)
        
        # 3. Coherence: T2 decay for each particle (SLOW quantum)
        self.step_coherence(dt, j_coupling_field)
        
        # 4. Entanglement: update bonds based on coherence and J-coupling (EMERGENT)
        self.step_entanglement(dt)
        
        # 5. Metrics
        metrics = self.get_network_metrics()
        metrics['n_births'] = pop_result['n_births']
        metrics['n_deaths'] = pop_result['n_deaths']
        
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
        cleft_height = 20e-9  # 20nm - Zuber et al. 2005
        volume_per_voxel = (self.dx ** 2) * cleft_height * 1000  # Liters
        N_A = 6.022e23
        field = field / (volume_per_voxel * N_A)
        
        return field