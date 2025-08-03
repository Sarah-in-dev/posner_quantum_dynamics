"""
Integrated analysis combining chemical and quantum dynamics
UPDATED to use the validated working model
"""

import numpy as np
from typing import Dict, List
import logging

from ..models.working_posner_model import WorkingPosnerModel, IsotopeComparison

logger = logging.getLogger(__name__)


class IntegratedQuantumSynapseModel:
    """
    Wrapper class for backward compatibility
    Now uses the validated WorkingPosnerModel
    """
    
    def __init__(self, isotope='P31'):
        self.isotope = isotope
        self.model = WorkingPosnerModel(isotope=isotope)
        logger.info(f"Initialized integrated model with {isotope} using validated parameters")
        
    def simulate_quantum_learning_window(self, n_spikes=10, frequency=10.0):
        """
        Full simulation: spikes → Posner → quantum coherence
        Maintains the original API for compatibility
        """
        # Use the working model's spike train simulation
        results = self.model.simulate_spike_train(
            n_spikes=n_spikes, 
            frequency=frequency,
            duration=max(3.0, n_spikes/frequency + 1.0)
        )
        
        # Format results to match expected output
        return {
            'times': results['time'],
            'calcium': results['calcium'],
            'posner': results['posner'],
            'coherence': np.exp(-0.1 / results['coherence_time']),  # Simplified coherence metric
            'quantum_window': results['coherence_time'],
            'enhancement_factor': results['enhancement_factor'],
            'spike_times': results['spike_times']
        }
    
    def compare_isotopes(self, n_spikes=20, frequency=10.0):
        """Compare P31 and P32 responses"""
        comparison = IsotopeComparison()
        return comparison.compare_spike_trains(n_spikes, frequency)
    
    def predict_learning(self, duration=120.0):
        """Predict BCI learning curve"""
        return self.model.predict_bci_learning(duration)


# For backward compatibility with notebooks
def create_integrated_model(isotope='P31'):
    """Factory function for creating integrated models"""
    return IntegratedQuantumSynapseModel(isotope)


# Quick test function
def test_integrated_model():
    """Test the integrated model works correctly"""
    model = IntegratedQuantumSynapseModel('P31')
    results = model.simulate_quantum_learning_window()
    
    print(f"Integrated model test:")
    print(f"  Max Posner: {np.max(results['posner'])*1e9:.1f} nM")
    print(f"  Enhancement: {results['enhancement_factor']:.1f}x")
    
    return results


if __name__ == "__main__":
    test_integrated_model()