"""
ddsc_module.py - Dendritic Delayed Stochastic CaMKII

Implements the instructive signal that triggers structural plasticity
after eligibility gating. Based on Nature 2024 findings that DDSC
peaks 30-40s after BTSP induction.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class DDSCParameters:
    """Parameters for DDSC dynamics"""
    tau_rise: float = 15.0        # seconds - time to rise
    tau_decay: float = 50.0       # seconds - decay time constant
    peak_time: float = 35.0       # seconds - when DDSC peaks
    threshold: float = 0.3        # eligibility threshold to trigger
    max_amplitude: float = 1.0    # maximum DDSC amplitude
    
    # Stochastic component (DDSC is inherently stochastic)
    stochastic: bool = True
    noise_amplitude: float = 0.1  # CV of DDSC amplitude


class DDSCSystem:
    """
    Models Dendritic Delayed Stochastic CaMKII activation.
    
    DDSC is triggered when eligibility exceeds threshold at plateau arrival.
    It then rises over ~15s, peaks at ~35s, and decays over ~50s.
    This serves as the instructive signal for structural plasticity.
    """
    
    def __init__(self, params: Optional[DDSCParameters] = None):
        self.params = params or DDSCParameters()
        self.reset()
    
    def reset(self):
        """Reset DDSC state"""
        self.triggered = False
        self.trigger_time = None
        self.eligibility_at_trigger = 0.0
        self.integrated_ddsc = 0.0  # Accumulated instructive signal
        self.current_ddsc = 0.0
    
    def check_trigger(self, eligibility: float, current_time: float) -> bool:
        """
        Check if plateau should trigger DDSC.
        Call this when plateau potential arrives.
        
        Returns True if DDSC was triggered.
        """
        if self.triggered:
            return False  # Already triggered
            
        if eligibility >= self.params.threshold:
            self.triggered = True
            self.trigger_time = current_time
            self.eligibility_at_trigger = eligibility
            return True
        
        return False
    
    def compute_ddsc(self, current_time: float) -> float:
        """
        Compute current DDSC activation level.
        
        DDSC follows a rise-then-decay profile:
        DDSC(t) = eligibility * (1 - exp(-t/tau_rise)) * exp(-t/tau_decay)
        
        This peaks around tau_rise * ln(1 + tau_decay/tau_rise) ≈ 35s
        """
        if not self.triggered or self.trigger_time is None:
            self.current_ddsc = 0.0
            return 0.0
        
        t = current_time - self.trigger_time
        if t < 0:
            self.current_ddsc = 0.0
            return 0.0
        
        # Rise-decay dynamics
        rise = 1.0 - np.exp(-t / self.params.tau_rise)
        decay = np.exp(-t / self.params.tau_decay)
        
        # Amplitude scales with eligibility at trigger
        amplitude = self.eligibility_at_trigger * self.params.max_amplitude
        
        # Add stochastic component if enabled
        if self.params.stochastic:
            noise = np.random.normal(0, self.params.noise_amplitude)
            amplitude *= (1.0 + noise)
            amplitude = max(0, amplitude)  # No negative DDSC
        
        self.current_ddsc = amplitude * rise * decay
        return self.current_ddsc
    
    def integrate(self, current_time: float, dt: float) -> float:
        """
        Integrate DDSC over time step.
        Returns the accumulated instructive signal.
        """
        ddsc = self.compute_ddsc(current_time)
        self.integrated_ddsc += ddsc * dt
        return self.integrated_ddsc
    
    def get_structural_drive(self) -> float:
        """
        Get the drive signal for structural plasticity.
        This is a saturating function of integrated DDSC.
        """
        # Saturating function - approaches 1.0 asymptotically
        # K_half is the integrated DDSC needed for half-max effect
        K_half = 20.0  # Tune this based on expected integration
        
        return self.integrated_ddsc / (self.integrated_ddsc + K_half)
    
    def get_state(self) -> dict:
        """Return current state for diagnostics"""
        return {
            'triggered': self.triggered,
            'trigger_time': self.trigger_time,
            'eligibility_at_trigger': self.eligibility_at_trigger,
            'current_ddsc': self.current_ddsc,
            'integrated_ddsc': self.integrated_ddsc,
            'structural_drive': self.get_structural_drive()
        }