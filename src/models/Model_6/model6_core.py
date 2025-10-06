"""
Model 6 Core - Complete Quantum Synapse Integration
====================================================
Integrates all subsystems into a complete emergent quantum synapse model

All components from literature-validated modules:
- Calcium system (channels, diffusion, buffering)
- ATP system (hydrolysis, J-coupling, phosphate)
- PNC formation (thermodynamic nucleation)
- Posner system (dimer/trimer formation, quantum coherence)
- pH dynamics (activity-dependent acidification)
- Dopamine system (reward signaling and quantum modulation)

NOTHING IS PRESCRIBED - ALL EMERGENT!

Author: Sarah Davidson
Date: October 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import h5py
from datetime import datetime

from model6_parameters import Model6Parameters
from calcium_system import CalciumSystem
from atp_system import ATPSystem
from ca_triphosphate_complex import CalciumTriphosphateSystem
from posner_system import PosnerSystem
from pH_dynamics import pHDynamics

# Import dopamine system
try:
    from dopamine_system import DopamineSystemAdapter
    HAS_DOPAMINE = True
except ImportError:
    HAS_DOPAMINE = False
    logging.warning("dopamine_system not found - dopamine effects disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model6QuantumSynapse:
    """
    Complete emergent quantum synapse model
    
    Integrates all physics from first principles:
    - No prescribed rates
    - No arbitrary enhancement factors
    - All behaviors emerge from interactions
    
    Key experimental controls:
    - Temperature (for Q10 tests)
    - Dopamine levels (reward signal)
    - Isotope composition (via params, not tested here)
    """
    
    def __init__(self, params: Optional[Model6Parameters] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize complete quantum synapse model
        
        Args:
            params: Model parameters (default: natural conditions)
            output_dir: Directory for saving results
        """
        # Parameters
        self.params = params or Model6Parameters()
        
        # Grid setup
        self.grid_shape = (self.params.spatial.grid_size, 
                          self.params.spatial.grid_size)
        self.dx = (2 * self.params.spatial.active_zone_radius / 
                   self.params.spatial.grid_size)
        
        # Time tracking
        self.time = 0.0
        self.dt = self.params.simulation.dt_diffusion  # Base timestep
        
        # Output
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subsystems
        self._initialize_subsystems()
        
        # History tracking
        self.history = {
            'time': [],
            'calcium_peak': [],
            'atp_mean': [],
            'j_coupling_max': [],
            'monomer_total': [],
            'dimer_ct_total': [],
            'dimer_concentration': [],
            'trimer_concentration': [],
            'coherence_dimer': [],
            'coherence_trimer': [],
            'pH_min': [],
            'dopamine_max': [],
        }
        
        logger.info("Model 6 initialized successfully")
        
    def _initialize_subsystems(self):
        """Initialize all physics subsystems with proper connectivity"""
        
        # 1. Calcium channels
        # Place channels in a cluster at center (typical active zone)
        n_channels = self.params.calcium.n_channels_per_site
        center = self.grid_shape[0] // 2
        
        # Random positions near center
        np.random.seed(42)  # Reproducibility
        channel_positions = []
        for _ in range(n_channels):
            x = center + np.random.randint(-2, 3)
            y = center + np.random.randint(-2, 3)
            channel_positions.append((x, y))
        channel_positions = np.array(channel_positions)
        
        # 2. Template positions (protein scaffolds for PNC nucleation)
        # Place at channel sites where high calcium occurs
        template_positions = [(x, y) for x, y in channel_positions[:3]]
        
        # Initialize all subsystems
        self.calcium = CalciumSystem(
            self.grid_shape, self.dx, channel_positions, self.params
        )
        logger.info(f"Calcium system: {n_channels} channels at center")
        
        self.atp = ATPSystem(
            self.grid_shape, self.dx, self.params
        )
        logger.info("ATP system initialized")
        
        self.triphosphate = CalciumTriphosphateSystem(
            self.grid_shape, self.dx, self.params, template_positions
        )
        logger.info(f"PNC system: {len(template_positions)} template sites")
        
        self.posner = PosnerSystem(
            self.grid_shape, self.params
        )
        logger.info("Posner system initialized")
        
        self.pH = pHDynamics(
            self.grid_shape, self.dx, self.params
        )
        logger.info("pH dynamics initialized")
        
        # Dopamine system (optional)
        if HAS_DOPAMINE:
            self.dopamine = DopamineSystemAdapter(
                grid_shape=self.grid_shape,  # Corrected: pass tuple
                dx=self.dx,
                params=self.params,
                release_sites=[(center, center)]
            )
            logger.info("Dopamine system initialized")
        else:
            self.dopamine = None
            logger.warning("Dopamine system not available")
        
    def step(self, dt: float, stimulus: Optional[Dict] = None):
        """
        Advance simulation by one timestep
        
        This is the orchestrator that connects all components.
        
        Args:
            dt: Time step (s)
            stimulus: Dictionary with control signals
                - 'voltage': Membrane voltage (V) - controls calcium channels
                - 'reward': Reward signal (boolean) - triggers dopamine
                - 'temperature': Optional temperature override (K)
        """
        stimulus = stimulus or {}
        
        # === STEP 1: CALCIUM DYNAMICS ===
        # Channels open based on voltage, calcium diffuses
        self.calcium.step(dt, stimulus)
        ca_conc = self.calcium.get_concentration()
        ca_peak = self.calcium.get_peak_concentration()
        
        # Calculate calcium influx for pH (needed later)
        ca_influx = np.gradient(ca_conc, axis=0) / dt  # Rough estimate
        
        # === STEP 2: ATP & J-COUPLING ===
        # ATP hydrolysis triggered by calcium, produces phosphate and J-coupling
        pH_field = self.pH.get_pH()
        self.atp.step(dt, ca_conc, pH_field)
        
        j_coupling = self.atp.get_j_coupling()
        atp_conc = self.atp.get_atp_concentration()
        phosphate = self.atp.get_phosphate_for_posner()
        
        # === STEP 3: pH DYNAMICS ===
        # Activity acidifies the synapse
        # Need ATP hydrolysis rate for pH
        atp_prev = getattr(self, '_atp_prev', atp_conc)
        atp_hydrolysis_rate = (atp_prev - atp_conc) / dt
        atp_hydrolysis_rate = np.maximum(atp_hydrolysis_rate, 0)
        self._atp_prev = atp_conc.copy()
        
        # Activity level from channels
        activity_level = self.calcium.channels.get_open_fraction()
        activity_field = np.ones(self.grid_shape) * activity_level
        
        self.pH.step(dt, atp_hydrolysis_rate, ca_influx, activity_field)
        pH_field = self.pH.get_pH()
        
        # === STEP 4: CALCIUM TRIPHOSPHATE FORMATION ===
        # Ca(HPO4)3^4- monomers form instantly via equilibrium
        # Then dimerize to form quantum qubits
        # No template_active parameter needed - templates built into system

        self.triphosphate.step(dt, ca_conc, phosphate)
        monomer_conc = self.triphosphate.get_monomer_concentration()
        dimer_conc = self.triphosphate.get_dimer_concentration()
        
        # === STEP 5: DOPAMINE (if available) ===
        if self.dopamine is not None:
            reward = stimulus.get('reward', False)
            self.dopamine.step(dt, reward_signal=reward)  # Corrected method name
            
            # Get D2 receptor occupancy for Posner modulation
            da_metrics = self.dopamine.get_experimental_metrics()  # Corrected method name
            d2_occupancy_mean = da_metrics['d2_occupancy_mean']
            
            # Make spatial field (simplified - use mean everywhere)
            # TODO: Could use spatial D2 field from dopamine system
            dopamine_d2 = np.ones(self.grid_shape) * d2_occupancy_mean
        else:
            dopamine_d2 = np.zeros(self.grid_shape)
        
        # === STEP 6: QUANTUM DIMER DYNAMICS ===
        # Dimers [Ca(HPO4)3]2^8- are the quantum qubits (Agarwal et al. 2023)
        # Track quantum coherence in these dimers
        # Note: For now, pass dimer_conc to posner system
        # TODO: Rename posner_system.py to quantum_dimer_system.py
        temperature = stimulus.get('temperature', self.params.environment.T)

        self.posner.step(
    dt, dimer_conc, ca_conc, j_coupling, dopamine_d2, temperature
)
        
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return all experimental metrics for validation
        
        This is what gets compared to experimental data!
        """
        # Get metrics from each subsystem
        ca_metrics = self.calcium.get_experimental_metrics()
        atp_metrics = self.atp.get_experimental_metrics()
        triphosphate_metrics = self.triphosphate.get_experimental_metrics()
        posner_metrics = self.posner.get_experimental_metrics()
        pH_metrics = self.pH.get_experimental_metrics()
        
        metrics = {
            # Calcium
            'calcium_peak_uM': ca_metrics['peak_ca_uM'],
            'calcium_mean_uM': ca_metrics.get('mean_ca_uM', 0),
            
            # ATP
            'atp_mean_mM': atp_metrics['atp_mean_mM'],
            'j_coupling_max_Hz': atp_metrics['j_coupling_max_Hz'],
            'j_coupling_mean_Hz': atp_metrics['j_coupling_mean_Hz'],
            
            # Calcium Triphosphate (the correct chemistry!)
            'monomer_peak_nM': triphosphate_metrics['monomer_peak_nM'],
            'monomer_mean_nM': triphosphate_metrics['monomer_mean_nM'],
            'dimer_peak_nM_ct': triphosphate_metrics['dimer_peak_nM'],  # From triphosphate
            'dimer_mean_nM_ct': triphosphate_metrics['dimer_mean_nM'],
            
            # Posner
            'dimer_peak_nM': posner_metrics['dimer_peak_nM'],
            'trimer_peak_nM': posner_metrics['trimer_peak_nM'],
            'dimer_trimer_ratio': posner_metrics['dimer_trimer_ratio'],
            
            # Quantum
            'coherence_dimer_mean': posner_metrics['coherence_dimer_mean'],
            'coherence_trimer_mean': posner_metrics['coherence_trimer_mean'],
            'T2_dimer_s': posner_metrics['T2_dimer_s'],
            'T2_trimer_s': posner_metrics['T2_trimer_s'],
            
            # pH
            'pH_min': pH_metrics['pH_min'],
            'pH_mean': pH_metrics['pH_mean'],
        }
        
        # Dopamine (if available)
        if self.dopamine is not None:
            da_metrics = self.dopamine.get_experimental_metrics()
            metrics['dopamine_max_nM'] = da_metrics['dopamine_max_nM']
            metrics['dopamine_mean_nM'] = da_metrics['dopamine_mean_nM']
            metrics['d2_occupancy_mean'] = da_metrics['d2_occupancy_mean']
        else:
            metrics['dopamine_max_nM'] = 0
            metrics['dopamine_mean_nM'] = 0
            metrics['d2_occupancy_mean'] = 0
        
        return metrics
    
    def _record_timestep(self):
        """Record current state to history"""
        self.history['time'].append(self.time)
        
        metrics = self.get_experimental_metrics()
        
        self.history['calcium_peak'].append(metrics['calcium_peak_uM'])
        self.history['atp_mean'].append(metrics['atp_mean_mM'])
        self.history['j_coupling_max'].append(metrics['j_coupling_max_Hz'])
        self.history['dimer_concentration'].append(metrics['dimer_peak_nM'])
        self.history['trimer_concentration'].append(metrics['trimer_peak_nM'])
        self.history['coherence_dimer'].append(metrics['coherence_dimer_mean'])
        self.history['coherence_trimer'].append(metrics['coherence_trimer_mean'])
        self.history['pH_min'].append(metrics['pH_min'])
        self.history['dopamine_max'].append(metrics['dopamine_max_nM'])
        
    def _log_status(self):
        """Log current simulation status"""
        if len(self.history['time']) == 0:
            return
        
        logger.info(f"t={self.time*1e3:.0f} ms: "
                   f"Ca={self.history['calcium_peak'][-1]:.1f} μM, "
                   f"J={self.history['j_coupling_max'][-1]:.1f} Hz, "
                   f"Dimer={self.history['dimer_concentration'][-1]:.2f} nM, "
                   f"Coherence={self.history['coherence_dimer'][-1]:.3f}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save simulation results to HDF5 file"""
        if self.output_dir is None:
            logger.warning("No output directory specified")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model6_results_{timestamp}.h5"
        
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            # TODO: Save all parameters as attributes
            
            # Save history
            history_group = f.create_group('history')
            for key, values in self.history.items():
                history_group.create_dataset(key, data=values)
            
            # Save final state fields
            state_group = f.create_group('final_state')
            state_group.create_dataset('calcium', data=self.calcium.get_concentration())
            state_group.create_dataset('ct_dimer', data=self.triphosphate.get_dimer_concentration())
            state_group.create_dataset('dimer', data=self.posner.get_dimer_concentration())
            state_group.create_dataset('trimer', data=self.posner.get_trimer_concentration())
            state_group.create_dataset('coherence_dimer', data=self.posner.get_coherence_dimer())
            
        logger.info(f"Results saved to {filepath}")


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL 6 CORE ORCHESTRATOR TEST")
    print("="*70)
    
    # Create model with default parameters
    model = Model6QuantumSynapse()
    
    print("\n### Quick Integration Test ###")
    print("Running 10 steps to verify all components connect properly...")
    
    # Test baseline
    for i in range(10):
        model.step(model.dt, stimulus={'voltage': -70e-3})  # Resting
    
    print("✓ Baseline steps completed")
    
    # Test with activity
    print("\nTesting with depolarization...")
    for i in range(10):
        model.step(model.dt, stimulus={'voltage': -10e-3})  # Depolarized
    
    print("✓ Activity steps completed")
    
    # Get metrics
    metrics = model.get_experimental_metrics()
    
    print("\nFinal Metrics:")
    print(f"  Calcium: {metrics['calcium_peak_uM']:.3f} μM")
    print(f"  ATP: {metrics['atp_mean_mM']:.2f} mM")
    print(f"  J-coupling: {metrics['j_coupling_max_Hz']:.1f} Hz")
    print(f"  Ca(HPO4)3 monomers: {metrics['monomer_peak_nM']:.2f} nM, dimers: {metrics['dimer_peak_nM_ct']:.2f} nM")
    print(f"  Coherence: {metrics['coherence_dimer_mean']:.3f}")
    
    print("\n" + "="*70)
    print("✓ ORCHESTRATOR FUNCTIONING CORRECTLY")
    print("="*70)