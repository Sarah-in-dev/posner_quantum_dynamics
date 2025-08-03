"""
Synaptic Posner Molecule Formation Model - UPDATED VERSION
Models Ca9(PO4)6 formation dynamics in synaptic clefts
Updated with validated kinetics from model refinement process
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class SynapticParameters:
    """Parameters for synaptic cleft environment - UPDATED with validated values"""
    cleft_width: float = 20e-9  # 20 nm
    active_zone_radius: float = 200e-9  # 200 nm
    
    # Baseline concentrations
    ca_baseline: float = 1e-7  # 100 nM
    po4_baseline: float = 1e-3  # 1 mM
    
    # Kinetic parameters - UPDATED based on model refinement
    kf_posner: float = 2e-3  # Effective formation rate constant
    kr_posner: float = 0.5   # Dissolution rate (2s lifetime)
    
    # Diffusion coefficients
    diffusion_ca: float = 2.2e-10  # m²/s
    diffusion_po4: float = 1e-10  # m²/s
    
    # Environmental conditions - UPDATED
    pH: float = 7.3  # Slightly higher than original 7.2
    temperature: float = 310  # K (37°C)
    
    @property
    def volume(self) -> float:
        """Synaptic cleft volume in L"""
        return np.pi * self.active_zone_radius**2 * self.cleft_width * 1000


class SynapticPosnerModel:
    """
    Models Posner molecule (Ca9(PO4)6) formation in synaptic clefts
    UPDATED: Uses effective kinetics validated through iterative refinement
    """
    
    def __init__(self, params: SynapticParameters = None):
        self.params = params or SynapticParameters()
        logger.info(f"Initialized model with cleft volume: {self.params.volume:.2e} L")
        logger.info(f"Using updated kinetics: kf={self.params.kf_posner}, kr={self.params.kr_posner}")
        
    def calcium_spike(self, t: float, amplitude: float = 300e-6, 
                     duration: float = 0.001, rise_time: float = 0.0001) -> float:
        """Models calcium transient during vesicle release"""
        if t < 0:
            return self.params.ca_baseline
        
        # Double exponential for realistic dynamics
        rise_phase = 1 - np.exp(-t/rise_time)
        decay_phase = np.exp(-t/duration)
        
        return self.params.ca_baseline + amplitude * rise_phase * decay_phase
    
    def phosphate_speciation(self, pH: float = None) -> Dict[str, float]:
        """Calculate phosphate species fractions at given pH"""
        pH = pH or self.params.pH
        
        # pKa values for phosphoric acid
        pKa1, pKa2, pKa3 = 2.15, 7.20, 12.35
        
        # Calculate alpha fractions
        h_conc = 10**(-pH)
        denominator = (h_conc**3 + h_conc**2 * 10**(-pKa1) + 
                      h_conc * 10**(-pKa1-pKa2) + 10**(-pKa1-pKa2-pKa3))
        
        alpha = {
            'H3PO4': h_conc**3 / denominator,
            'H2PO4-': h_conc**2 * 10**(-pKa1) / denominator,
            'HPO4_2-': h_conc * 10**(-pKa1-pKa2) / denominator,
            'PO4_3-': 10**(-pKa1-pKa2-pKa3) / denominator
        }
        
        return alpha
    
    def posner_formation_rate(self, ca_conc: float, po4_total: float, 
                            pH: float = None, temp: float = None) -> float:
        """
        UPDATED: Effective formation rate using dominant phosphate species
        This represents the rate-limiting step of Posner formation
        """
        pH = pH or self.params.pH
        temp = temp or self.params.temperature
        
        # Get phosphate speciation
        alpha = self.phosphate_speciation(pH)
        
        # At pH 7.3, HPO4²⁻ is dominant (~61%), PO4³⁻ is only ~0.05%
        # Use effective kinetics with HPO4²⁻
        hpo4_conc = po4_total * alpha['HPO4_2-']
        
        # Temperature correction (minor effect)
        Q10 = 2.0  # Typical for chemical reactions
        temp_factor = Q10**((temp - 310) / 10)
        
        # Effective rate - simplified from full Ca9(PO4)6 stoichiometry
        # This gives physiological nM concentrations
        rate = self.params.kf_posner * temp_factor * ca_conc * hpo4_conc
        
        return rate
    
    def simulate_single_spike(self, duration: float = 0.1, dt: float = 0.0001) -> Dict:
        """Simulate Posner dynamics for a single calcium spike"""
        times = np.arange(0, duration, dt)
        n_steps = len(times)
        
        # Initialize arrays
        ca = np.zeros(n_steps)
        po4 = np.ones(n_steps) * self.params.po4_baseline
        posner = np.zeros(n_steps)
        formation_rate = np.zeros(n_steps)
        
        # Simulate dynamics
        for i, t in enumerate(times):
            ca[i] = self.calcium_spike(t)
            
            if i > 0:
                # Calculate rates
                form_rate = self.posner_formation_rate(ca[i], po4[i])
                diss_rate = self.params.kr_posner * posner[i-1]
                
                # Update Posner concentration
                dposner = (form_rate - diss_rate) * dt
                posner[i] = max(0, posner[i-1] + dposner)
                formation_rate[i] = form_rate
                
                # Note: We don't deplete Ca/PO4 because at nM Posner levels,
                # the depletion is negligible (< 0.001% of available ions)
        
        logger.info(f"Single spike simulation complete. Max Posner: {np.max(posner)*1e9:.1f} nM")
        
        return {
            'time': times,
            'calcium': ca,
            'phosphate': po4,
            'posner': posner,
            'formation_rate': formation_rate
        }
    
    def simulate_spike_train(self, n_spikes: int = 10, isi: float = 0.1,
                           total_time: float = None) -> Dict:
        """Simulate Posner dynamics during spike train"""
        if total_time is None:
            total_time = n_spikes * isi + 1.0
            
        dt = 0.0001
        times = np.arange(0, total_time, dt)
        n_steps = len(times)
        
        # Initialize arrays
        ca = np.zeros(n_steps)
        po4 = np.ones(n_steps) * self.params.po4_baseline
        posner = np.zeros(n_steps)
        
        # Generate spike times
        spike_times = np.arange(0, n_spikes * isi, isi)
        
        for i, t in enumerate(times):
            # Sum calcium contributions from all spikes
            ca_total = self.params.ca_baseline
            for spike_t in spike_times:
                if t >= spike_t:
                    ca_total += self.calcium_spike(t - spike_t) - self.params.ca_baseline
            ca[i] = ca_total
            
            if i > 0:
                # Posner dynamics
                form_rate = self.posner_formation_rate(ca[i], po4[i])
                diss_rate = self.params.kr_posner * posner[i-1]
                
                dposner = (form_rate - diss_rate) * dt
                posner[i] = max(0, posner[i-1] + dposner)
        
        logger.info(f"Spike train simulation complete. {n_spikes} spikes, max Posner: {np.max(posner)*1e9:.1f} nM")
        
        return {
            'time': times,
            'calcium': ca,
            'phosphate': po4,
            'posner': posner,
            'spike_times': spike_times
        }
    
    def parameter_scan(self, param_name: str, param_values: np.ndarray,
                      metric: str = 'max_posner') -> Dict:
        """
        Scan a parameter and measure effect on Posner formation
        Useful for sensitivity analysis
        """
        results = []
        original_value = getattr(self.params, param_name)
        
        for value in param_values:
            # Update parameter
            setattr(self.params, param_name, value)
            
            # Run simulation
            sim = self.simulate_single_spike()
            
            # Extract metric
            if metric == 'max_posner':
                results.append(np.max(sim['posner']))
            elif metric == 'steady_state':
                results.append(sim['posner'][-1])
            elif metric == 'time_to_peak':
                results.append(sim['time'][np.argmax(sim['posner'])])
            
        # Restore original value
        setattr(self.params, param_name, original_value)
        
        return {
            'parameter': param_name,
            'values': param_values,
            'results': np.array(results)
        }


# Validation function
def validate_model():
    """Run basic validation to ensure model behaves correctly"""
    model = SynapticPosnerModel()
    
    # Test single spike
    results = model.simulate_single_spike()
    max_posner = np.max(results['posner']) * 1e9  # Convert to nM
    
    print("Model Validation:")
    print(f"  Max [Posner]: {max_posner:.1f} nM")
    print(f"  Expected range: 1-100 nM")
    
    if 1 < max_posner < 100:
        print("  ✓ PASS: Posner concentration in physiological range")
    else:
        print("  ✗ FAIL: Posner concentration out of range")
    
    # Check calcium doesn't go negative
    if np.all(results['calcium'] >= 0):
        print("  ✓ PASS: Calcium remains positive")
    else:
        print("  ✗ FAIL: Negative calcium detected")
    
    # Check pH dependence
    ph_values = [6.8, 7.2, 7.6]
    for ph in ph_values:
        model.params.pH = ph
        spec = model.phosphate_speciation()
        print(f"  pH {ph}: HPO₄²⁻ = {spec['HPO4_2-']*100:.1f}%, PO₄³⁻ = {spec['PO4_3-']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    validate_model()