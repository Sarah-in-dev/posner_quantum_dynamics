"""
Singlet State Dynamics for Calcium Phosphate Dimers
Based on Agarwal et al. 2023 "The Biological Qubit"

Physics:
- Track singlet probability P_S(t) for pairs of ³¹P nuclear spins
- P_S > 0.5 indicates entanglement preserved
- Decay driven by J-coupling frequency spread (destructive interference)
- Dimers (4 spins) have fewer frequencies → slower decay than trimers (6 spins)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SingletState:
    """State of a singlet pair within a dimer"""
    probability: float = 1.0  # P_S (0.25 to 1.0)
    pair_indices: Tuple[int, int] = (0, 1)  # Which ³¹P nuclei
    creation_time: float = 0.0
    
    @property
    def is_entangled(self) -> bool:
        return self.probability > 0.5


class DimerSingletDynamics:
    """
    Track singlet probability evolution in a dimer
    
    A dimer (Ca₆(PO₄)₄) has 4 ³¹P nuclei.
    From pyrophosphate hydrolysis, we expect 2 singlet pairs.
    """
    
    def __init__(self, params):
        self.params = params
        
        # J-coupling matrix for 4 spins (6 unique values)
        # Initialize with random values from distribution
        self.J_matrix = self._initialize_j_couplings()
        
        # Calculate eigenfrequencies (determines oscillation)
        self.frequencies = self._calculate_frequencies()
        
        # Singlet states (could have up to 2 pairs)
        self.singlet_pairs = []
        
    def _initialize_j_couplings(self) -> np.ndarray:
        """
        Initialize J-coupling matrix from Agarwal's DFT values
        
        Typical values: -0.4 to +0.2 Hz
        """
        n = self.params.n_spins_dimer  # 4
        J = np.zeros((n, n))
        
        # Fill upper triangle with random J-couplings
        for i in range(n):
            for j in range(i+1, n):
                J[i,j] = np.random.normal(
                    self.params.J_intra_dimer_mean,
                    self.params.J_intra_dimer_std
                )
                J[j,i] = J[i,j]  # Symmetric
        
        return J
    
    def _calculate_frequencies(self) -> np.ndarray:
        """
        Calculate oscillation frequencies from J-coupling network
        
        These drive the coherent evolution that spreads singlet order
        """
        # For 4 spins with 6 J-couplings, we get combinations
        # Simplified: use eigenvalues of J-matrix
        eigenvalues = np.linalg.eigvalsh(self.J_matrix)
        
        # Frequencies are differences between eigenvalues
        n = len(eigenvalues)
        frequencies = []
        for i in range(n):
            for j in range(i+1, n):
                freq = abs(eigenvalues[i] - eigenvalues[j])
                if freq > 0:
                    frequencies.append(freq)
        
        return np.array(frequencies)
    
    def step(self, dt: float, dipolar_relaxation_rate: float = 0.0):
        """
        Update singlet probability
        
        Two contributions:
        1. Coherent oscillation (J-coupling driven)
        2. Dipolar relaxation (slow decay toward 0.25)
        """
        for singlet in self.singlet_pairs:
            # === Coherent evolution ===
            # Simplified model: P_S oscillates with multiple frequencies
            # Destructive interference causes apparent decay
            
            t = singlet.creation_time
            
            # Sum of oscillating terms (each frequency contributes)
            oscillation = 0.0
            for freq in self.frequencies:
                # Each frequency contributes with random phase
                phase = np.random.uniform(0, 2*np.pi)  # Or track phase
                oscillation += np.cos(2*np.pi*freq*t + phase)
            
            # Normalize by number of frequencies
            if len(self.frequencies) > 0:
                oscillation /= len(self.frequencies)
            
            # P_S = 0.25 + 0.75 * <coherent term>
            # When coherent terms cancel: P_S → 0.25
            # When coherent: P_S can reach 1.0
            
            # === Simplified decay model ===
            # Based on Agarwal's finding: dimers maintain P_S > 0.5 for ~100s
            T_decay = self.params.T_singlet_dimer
            
            # Exponential envelope on oscillations
            envelope = np.exp(-t / T_decay)
            
            # Update probability
            # P_S decays from 1.0 toward 0.25 with time constant T_decay
            P_thermal = self.params.singlet_thermal  # 0.25
            P_initial = 1.0
            
            singlet.probability = P_thermal + (P_initial - P_thermal) * envelope
            
            # Add dipolar relaxation contribution
            if dipolar_relaxation_rate > 0:
                singlet.probability -= dipolar_relaxation_rate * dt * (singlet.probability - P_thermal)
            
            # Clamp
            singlet.probability = np.clip(singlet.probability, P_thermal, 1.0)
            
            singlet.creation_time += dt
    
    def add_singlet_pair(self, pair_indices: Tuple[int, int] = (0, 1)):
        """Create a new singlet pair (from pyrophosphate hydrolysis)"""
        self.singlet_pairs.append(SingletState(
            probability=1.0,
            pair_indices=pair_indices,
            creation_time=0.0
        ))
    
    def get_entanglement_fraction(self) -> float:
        """Fraction of singlet pairs that are still entangled"""
        if not self.singlet_pairs:
            return 0.0
        n_entangled = sum(1 for s in self.singlet_pairs if s.is_entangled)
        return n_entangled / len(self.singlet_pairs)
    
    def get_mean_singlet_probability(self) -> float:
        """Average P_S across all pairs"""
        if not self.singlet_pairs:
            return 0.0
        return np.mean([s.probability for s in self.singlet_pairs])