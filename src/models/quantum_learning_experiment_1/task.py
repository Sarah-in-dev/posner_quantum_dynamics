"""
Context Switching Task - Tests Rapid Adaptation

Mimics BCI scenario where mapping rules change unexpectedly
and the agent must rapidly detect and adapt.

Tests prediction: quantum systems adapt 2.4× faster

Author: Sarah Davidson
Date: October 2025
"""

import torch
import numpy as np
from typing import Tuple


class ContextSwitchTask:
    """
    Task with changing context/rules that tests rapid adaptation
    
    Scenario:
    - Agent learns input → output mapping
    - Context switches every N trials
    - New mapping must be learned quickly
    
    This mirrors BCI learning where neural patterns must adapt
    to new decoder mappings
    
    Args:
        input_dim: Dimensionality of input patterns
        n_contexts: Number of different contexts to switch between
        switch_every: Number of trials before context switch
    """
    
    def __init__(self, input_dim: int = 20, n_contexts: int = 3, 
                 switch_every: int = 50):
        self.input_dim = input_dim
        self.n_contexts = n_contexts
        self.switch_every = switch_every
        
        # Current state
        self.current_context = 0
        self.trial_num = 0
        
        # Generate random mapping for each context
        # Each context has different input→output transformation
        self.context_weights = [
            torch.randn(input_dim, 2) * 0.5 
            for _ in range(n_contexts)
        ]
        
        # Normalize weights
        for i in range(n_contexts):
            self.context_weights[i] = (
                self.context_weights[i] / 
                torch.norm(self.context_weights[i], dim=0, keepdim=True)
            )
    
    def generate_trial(self) -> Tuple[torch.Tensor, int, bool]:
        """
        Generate input and target for current trial
        
        Returns:
            x: Input pattern [1, input_dim]
            target: Target class (0 or 1)
            switched: Whether context just switched
        """
        # Check for context switch
        switched = False
        if self.trial_num > 0 and self.trial_num % self.switch_every == 0:
            self.current_context = (self.current_context + 1) % self.n_contexts
            switched = True
        
        # Generate random input
        x = torch.randn(1, self.input_dim)
        
        # Compute target based on current context mapping
        logits = torch.matmul(x, self.context_weights[self.current_context])
        target = torch.argmax(logits, dim=1).item()
        
        self.trial_num += 1
        
        return x, target, switched
    
    def compute_reward(self, prediction: int, target: int) -> float:
        """
        Compute reward signal
        
        Args:
            prediction: Model prediction
            target: True target
            
        Returns:
            reward: 1.0 if correct, 0.0 if wrong
        """
        return 1.0 if prediction == target else 0.0
    
    def reset(self):
        """Reset task state"""
        self.current_context = 0
        self.trial_num = 0
    
    def get_context(self) -> int:
        """Get current context index"""
        return self.current_context