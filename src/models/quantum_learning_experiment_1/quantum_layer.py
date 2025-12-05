"""
Quantum Coherent Layer - Based on Biological Posner Molecule Dynamics

Implements quantum-inspired processing based on:
- T2 coherence time: ~100s for dimers (P31)
- J-coupling: 20 Hz from ATP hydrolysis
- Dopamine modulation: switches between dimer/trimer modes
- Spatial localization: hotspot-based processing

Author: Sarah Davidson
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class QuantumCoherentLayer(nn.Module):
    """
    Quantum-inspired neural layer based on biological Posner molecule dynamics
    
    Key biological parameters from Model 5/6:
    - T2_dimer = 100 seconds (long coherence)
    - T2_trimer = 2 seconds (short coherence)
    - J_coupling = 20 Hz (ATP-driven)
    - Dopamine threshold = 0.5 (switches modes)
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        coherence_steps: Base coherence time in timesteps (default 100)
        n_paths: Number of parallel quantum paths (default 8)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 coherence_steps: int = 100, n_paths: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_paths = n_paths
        
        # Biological parameters from your model
        self.T2_dimer = coherence_steps  # 100 timesteps (like 100s)
        self.T2_trimer = coherence_steps // 50  # 2 timesteps (like 2s)
        
        # J-coupling frequency (20 Hz from ATP)
        self.J_coupling_freq = 20.0
        
        # Standard neural network weights
        self.W_input = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        
        # Multiple parallel pathways (like different dimer configurations)
        self.W_paths = nn.Parameter(
            torch.randn(self.n_paths, hidden_dim, hidden_dim) * 0.1
        )
        
        # J-coupling creates correlations between pathways
        # Initialize with small random values
        self.J_coupling = nn.Parameter(torch.randn(self.n_paths, self.n_paths) * 0.01)
        
        # State buffers (not trained, just tracked)
        self.register_buffer('coherent_state', 
                           torch.zeros(1, self.n_paths, hidden_dim))
        self.register_buffer('coherence_amplitude', 
                           torch.ones(1, self.n_paths))
        self.register_buffer('time_since_formation', 
                           torch.zeros(1))
        
        
        # Mechanism: Continuous dimer/trimer populations
        self.register_buffer('p_dimer', torch.tensor(0.5))
        self.register_buffer('p_trimer', torch.tensor(0.5))
        self.tau_formation = 20  # timesteps (~200ms in biology)
        self.dopamine_sensitivity = 5.0  # Steepness of sigmoid
        
        
        # Current dopamine level
        self.dopamine_level = 0.0
        
    def forward(self, x: torch.Tensor, dopamine_signal: float = None, 
                force_measurement: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through quantum coherent layer
        
        Args:
            x: Input tensor [batch, input_dim]
            dopamine_signal: Reward/dopamine signal (0-1)
            force_measurement: Force collapse of superposition
            
        Returns:
            output: Processed tensor [batch, hidden_dim]
            info: Dictionary with quantum state information
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Update dopamine level
        if dopamine_signal is not None:
            self.dopamine_level = dopamine_signal
        
        # Update dimer/trimer populations (continuous, not binary!)
        self._update_populations()
        
        # Effective coherence is population-weighted average
        T2_effective = float(self.p_dimer * self.T2_dimer + self.p_trimer * self.T2_trimer)
        
        # Mode for diagnostics
        mode = 'dimer' if self.p_dimer > 0.5 else 'trimer'
        
        # ===================================================================
        # QUANTUM-INSPIRED PROCESSING
        # ===================================================================
        
        # 1. SUPERPOSITION: Create multiple parallel computation paths
        base_activation = torch.matmul(x, self.W_input)  # [batch, hidden_dim]
        
        # Initialize superposition tensor
        superposition = torch.zeros(batch_size, self.n_paths, self.hidden_dim, 
                                   device=device)
        
        # Expand coherent state to match batch size
        if self.coherent_state.shape[0] != batch_size:
            self.coherent_state = self.coherent_state.expand(batch_size, -1, -1).clone()
            self.coherence_amplitude = self.coherence_amplitude.expand(batch_size, -1).clone()
        
        # Create superposition across all paths
        for path_idx in range(self.n_paths):
            # Each path transforms the input differently
            path_activation = torch.matmul(base_activation, self.W_paths[path_idx])
            
            # Mix with existing coherent state (temporal coherence)
            alpha = torch.exp(-self.time_since_formation / T2_effective).clamp(0, 1)
            
            superposition[:, path_idx, :] = (
                alpha * self.coherent_state[:, path_idx, :] + 
                (1 - alpha) * path_activation
            )
        
        # 2. INTERFERENCE: J-coupling creates path interactions
        interference_matrix = self._compute_interference()
        
        # Apply interference between paths
        interfered_state = torch.zeros_like(superposition)
        for i in range(self.n_paths):
            for j in range(self.n_paths):
                coupling = interference_matrix[i, j]
                interfered_state[:, i, :] += coupling * superposition[:, j, :]
        
        # Normalize by number of paths
        interfered_state = interfered_state / self.n_paths
        
        # 3. COHERENCE DECAY: Apply decoherence
        coherence_factor = torch.exp(-self.time_since_formation / T2_effective).clamp(0, 1)
        interfered_state = interfered_state * coherence_factor
        
        # 4. MEASUREMENT: Collapse or maintain superposition
        measurement_occurred = False
        
        if force_measurement or (dopamine_signal is not None and dopamine_signal > 0.7):
            # HIGH DOPAMINE: Collapse to single state (measurement)
            # Weight by coherence amplitude (quantum probability)
            path_weights = F.softmax(self.coherence_amplitude, dim=1)  # [batch, n_paths]
            
            # Weighted sum = collapse
            collapsed_state = torch.sum(
                interfered_state * path_weights.unsqueeze(-1),
                dim=1
            )  # [batch, hidden_dim]
            
            output = collapsed_state
            
            # Reset after measurement
            self.coherent_state = interfered_state.detach()
            self.time_since_formation = torch.zeros_like(self.time_since_formation)
            measurement_occurred = True
            
        else:
            # LOW DOPAMINE: Maintain superposition (exploration)
            output = torch.mean(interfered_state, dim=1)  # [batch, hidden_dim]
            
            # Update coherent state
            self.coherent_state = interfered_state.detach()
            self.time_since_formation += 1
            measurement_occurred = False
        
        # Update path amplitudes based on contribution
        with torch.no_grad():
            path_contributions = torch.norm(interfered_state, dim=2)  # [batch, n_paths]
            self.coherence_amplitude = (
                0.9 * self.coherence_amplitude + 
                0.1 * path_contributions.mean(dim=0)
            )
        
        # Collect diagnostic information
        info = {
            'coherence_time': T2_effective,
            'time_since_formation': self.time_since_formation.item(),
            'measurement_occurred': measurement_occurred,
            'mode': mode,
            'p_dimer': float(self.p_dimer),
            'p_trimer': float(self.p_trimer),
            'dopamine_level': self.dopamine_level,
            'path_amplitudes': self.coherence_amplitude[0].detach().cpu().numpy(),
            'n_active_paths': (self.coherence_amplitude[0] > 0.1).sum().item(),
            'coherence_factor': coherence_factor.item()
        }
        
        return output, info
    
    def _compute_interference(self) -> torch.Tensor:
        """
        Compute interference pattern from J-coupling
        
        Mimics ATP-driven J-coupling (20 Hz) creating correlations
        between quantum states
        
        Returns:
            interference_matrix: [n_paths, n_paths]
        """
        # Symmetrize J-coupling matrix
        J_sym = (self.J_coupling + self.J_coupling.t()) / 2
        
        # Phase accumulation: φ = 2π * f * t
        # f = J_coupling_freq (20 Hz)
        # t = time_since_formation (in units where 1 step ~ 1 second)
        phase = (2 * np.pi * self.J_coupling_freq * 
                self.time_since_formation / 1000.0)
        
        # Interference pattern: cos(phase * J_coupling)
        # Paths with stronger coupling interfere more
        interference = torch.cos(phase * J_sym)
        
        # Normalize to valid probability distribution
        interference = F.softmax(interference, dim=1)
        
        return interference
    
    def reset_coherence(self):
        """Reset quantum state (call between episodes/tasks)"""
        self.coherent_state = torch.zeros_like(self.coherent_state)
        self.coherence_amplitude = torch.ones_like(self.coherence_amplitude)
        self.time_since_formation = torch.zeros_like(self.time_since_formation)
        self.dopamine_level = 0.0
        # Reset populations to equilibrium
        self.p_dimer.fill_(0.5)
        self.p_trimer.fill_(0.5)

class QuantumInspiredNetwork(nn.Module):
    """
    Complete network using quantum coherent layers
    
    Architecture:
    - Input layer (classical)
    - n_quantum_layers (quantum coherent)
    - Output layer (classical, forces measurement)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_quantum_layers: int = 2, coherence_steps: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input processing (classical)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Quantum coherent layers
        self.quantum_layers = nn.ModuleList([
            QuantumCoherentLayer(
                hidden_dim if i > 0 else hidden_dim,
                hidden_dim,
                coherence_steps=coherence_steps,
                n_paths=8
            )
            for i in range(n_quantum_layers)
        ])
        
        # Output layer (classical)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def _update_populations(self):
        """
        Update dimer/trimer population fractions based on dopamine
        
        Biology: High dopamine (D2 activation) → reduced Ca²⁺ → favors dimers
        Uses smooth sigmoid transition, not binary switch
        """
        # Target fractions from dopamine level
        # High DA (0.8) → ~70% dimers
        # Low DA (0.2) → ~30% dimers
        target_p_dimer = 1.0 / (1.0 + np.exp(-self.dopamine_sensitivity * (self.dopamine_level - 0.5)))
        target_p_trimer = 1.0 - target_p_dimer
        
        # Smooth transition (first-order kinetics)
        alpha = 1.0 / self.tau_formation
        self.p_dimer = self.p_dimer * (1 - alpha) + target_p_dimer * alpha
        self.p_trimer = self.p_trimer * (1 - alpha) + target_p_trimer * alpha
        
        # Ensure normalization
        total = self.p_dimer + self.p_trimer
        self.p_dimer = self.p_dimer / total
        self.p_trimer = self.p_trimer / total
    
    
    
    
    def forward(self, x: torch.Tensor, reward_signal: float = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass
        
        Args:
            x: Input [batch, input_dim]
            reward_signal: Dopamine-like signal (0-1)
            
        Returns:
            output: Network output [batch, output_dim]
            diagnostics: List of quantum state info dicts
        """
        # Classical input processing
        h = F.relu(self.input_layer(x))
        
        # Quantum coherent processing
        diagnostics = []
        for layer in self.quantum_layers:
            h, info = layer(h, dopamine_signal=reward_signal)
            h = F.relu(h)
            diagnostics.append(info)
        
        # Final output (forces measurement/collapse)
        output = self.output_layer(h)
        
        return output, diagnostics
    
    def reset_quantum_state(self):
        """Reset all quantum layers (call between episodes)"""
        for layer in self.quantum_layers:
            if isinstance(layer, QuantumCoherentLayer):
                layer.reset_coherence()