"""
Classical Neural Network - Baseline for Comparison

Standard feedforward architecture without quantum features
Used as control to measure quantum advantage

Author: Sarah Davidson
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalNetwork(nn.Module):
    """
    Standard feedforward neural network
    
    Architecture matches quantum network but without:
    - Multiple parallel paths
    - Coherence dynamics
    - J-coupling interference
    - Dopamine modulation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass
        
        Args:
            x: Input [batch, input_dim]
            
        Returns:
            output: Network output [batch, output_dim]
        """
        # Input processing
        h = F.relu(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        
        # Output
        output = self.output_layer(h)
        
        return output