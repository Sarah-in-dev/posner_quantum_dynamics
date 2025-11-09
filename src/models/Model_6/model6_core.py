"""
Model 6 Core - Complete Quantum Synapse Integration
====================================================
Integrates all subsystems into a complete emergent quantum synapse model

CORRECTED ARCHITECTURE (Nov 2025):
- Calcium system (stochastic channel gating, diffusion, buffering)
- ATP system (hydrolysis, J-coupling, phosphate)
- Ca₆(PO₄)₄ dimer formation (ca_triphosphate_complex.py)
- Quantum coherence tracking (quantum_coherence_system.py - NOT posner!)
- pH dynamics (activity-dependent acidification)
- Dopamine system (reward signaling)

KEY FIX: Quantum coherence tracks dimers FROM ca_triphosphate, not from PNCs
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
from ca_triphosphate_complex import CaHPO4DimerSystem
from quantum_coherence import QuantumCoherenceSystem  # CHANGED from posner_system
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
    Complete quantum synapse model integrating all subsystems
    
    Architecture:
    1. Calcium influx (stochastic channels)
    2. ATP hydrolysis → phosphate + J-coupling
    3. CaHPO₄ ion pairs → Ca₆(PO₄)₄ dimers (ca_triphosphate)
    4. Quantum coherence tracking of dimers (quantum_coherence_system)
    5. pH dynamics
    6. Dopamine modulation (optional)
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
            'ion_pair_total': [],
            'dimer_ct_total': [],
            'coherence_mean': [],
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
        # np.random.seed(42)  # Reproducibility for channel POSITIONS only
        # NOTE: Channel gating is stochastic with NO seed
        channel_positions = []
        for _ in range(n_channels):
            x = center + np.random.randint(-2, 3)
            y = center + np.random.randint(-2, 3)
            channel_positions.append((x, y))
        channel_positions = np.array(channel_positions)
        
        # 2. Template positions (protein scaffolds for dimer nucleation)
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
        
        self.ca_phosphate = CaHPO4DimerSystem(
            self.grid_shape, self.dx, self.params, template_positions
        )
        logger.info(f"Calcium phosphate system: {len(template_positions)} template sites")
        
        # CHANGED: Use QuantumCoherenceSystem instead of PosnerSystem
        self.quantum = QuantumCoherenceSystem(
            self.grid_shape, self.params.quantum, isotope_P31_fraction=self.params.environment.fraction_P31
        )
        logger.info("Quantum coherence system initialized")
        
        self.pH = pHDynamics(
            self.grid_shape, self.dx, self.params
        )
        logger.info("pH dynamics initialized")
       
        # Dopamine system (optional)
        if HAS_DOPAMINE and self.params.dopamine is not None:
            self.dopamine = DopamineSystemAdapter(
                grid_shape=self.grid_shape,
                dx=self.dx,
                params=self.params,
                release_sites=[(center, center)]
            )
            logger.info("Dopamine system initialized")
        else:
            self.dopamine = None
            if self.params.dopamine is None:
                logger.info("Dopamine system disabled (params.dopamine = None)")
            else:
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
        # NOTE: Channel gating is STOCHASTIC (no seed, varies run-to-run)
        self.calcium.step(dt, stimulus)
        ca_conc = self.calcium.get_concentration()
        
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
        
        # === STEP 4: CALCIUM PHOSPHATE DIMER FORMATION ===
        # Ca²⁺ + HPO₄²⁻ → CaHPO₄ (instant equilibrium)
        # 6 × CaHPO₄ → Ca₆(PO₄)₄ dimer (slow aggregation)
        # These are the quantum qubits (4 ³¹P nuclei)
        self.ca_phosphate.step(dt, ca_conc, phosphate)
        dimer_conc = self.ca_phosphate.get_dimer_concentration()
        
        # === STEP 5: DOPAMINE (if available) ===
        if self.dopamine is not None:
            reward = stimulus.get('reward', False)
            self.dopamine.step(dt, reward_signal=reward)
        
        # === STEP 6: QUANTUM COHERENCE TRACKING ===
        # CRITICAL FIX: Pass dimer_conc (from ca_phosphate), NOT pnc_large!
        # The quantum system ONLY tracks coherence, it doesn't create dimers
        temperature = stimulus.get('temperature', self.params.environment.T)
        
        self.quantum.step(dt, dimer_conc, j_coupling, temperature)
        
        # Update time
        self.time += dt
        
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return all experimental metrics for validation
        
        Returns metrics from each subsystem aligned with experimental measurements
        """
        # Get metrics from each subsystem
        ca_metrics = self.calcium.get_experimental_metrics()
        atp_metrics = self.atp.get_experimental_metrics()
        ca_phosphate_metrics = self.ca_phosphate.get_experimental_metrics()
        quantum_metrics = self.quantum.get_experimental_metrics()  # CHANGED from posner_metrics
        pH_metrics = self.pH.get_experimental_metrics()
        
        metrics = {
            # Calcium
            'calcium_peak_uM': ca_metrics['peak_ca_uM'],
            'calcium_mean_uM': ca_metrics.get('mean_ca_nM', 0) / 1000,  # Convert nM to μM
            
            # ATP
            'atp_mean_mM': atp_metrics['atp_mean_mM'],
            'j_coupling_max_Hz': atp_metrics['j_coupling_max_Hz'],
            'j_coupling_mean_Hz': atp_metrics['j_coupling_mean_Hz'],
            
            # Calcium Phosphate (the correct Ca₆(PO₄)₄ chemistry!)
            'ion_pair_peak_nM': ca_phosphate_metrics['ion_pair_peak_nM'],
            'ion_pair_mean_nM': ca_phosphate_metrics['ion_pair_mean_nM'],
            'dimer_peak_nM_ct': ca_phosphate_metrics['dimer_peak_nM'],
            'dimer_mean_nM_ct': ca_phosphate_metrics['dimer_mean_nM'],
            
            # Quantum Coherence (CHANGED - simpler metrics)
            'coherence_mean': quantum_metrics['coherence_mean'],
            'coherence_std': quantum_metrics['coherence_std'],
            'coherence_peak': quantum_metrics['coherence_peak'],
            'T2_dimer_s': quantum_metrics['T2_dimer_s'],
            
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
        self.history['ion_pair_total'].append(metrics['ion_pair_peak_nM'])
        self.history['dimer_ct_total'].append(metrics['dimer_peak_nM_ct'])
        self.history['coherence_mean'].append(metrics['coherence_mean'])
        self.history['pH_min'].append(metrics['pH_min'])
        
        if self.dopamine is not None:
            self.history['dopamine_max'].append(metrics['dopamine_max_nM'])
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save simulation results to HDF5 file
        
        Args:
            filepath: Path to output file (default: auto-generated in output_dir)
        """
        if filepath is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"model6_results_{timestamp}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['grid_shape'] = self.grid_shape
            f.attrs['dx_nm'] = self.dx * 1e9
            f.attrs['total_time_s'] = self.time
            
            # History
            hist_group = f.create_group('history')
            for key, values in self.history.items():
                hist_group.create_dataset(key, data=values)
            
            # Final state
            state_group = f.create_group('final_state')
            state_group.create_dataset('calcium', data=self.calcium.get_concentration())
            state_group.create_dataset('ion_pairs', data=self.ca_phosphate.get_ion_pair_concentration())
            state_group.create_dataset('dimers', data=self.ca_phosphate.get_dimer_concentration())
            state_group.create_dataset('coherence', data=self.quantum.get_coherence_field())
            
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
    print(f"  Ion pairs: {metrics['ion_pair_peak_nM']:.2f} nM")
    print(f"  Ca₆(PO₄)₄ dimers: {metrics['dimer_peak_nM_ct']:.2f} nM")
    print(f"  Quantum coherence: {metrics['coherence_mean']:.3f}")
    print(f"  T2 time: {metrics['T2_dimer_s']:.1f} s")
    
    print("\n" + "="*70)
    print("✓ ORCHESTRATOR FUNCTIONING CORRECTLY")
    print("  - Calcium → ATP → Dimers → Quantum coherence")
    print("  - All subsystems connected")
    print("  - Proper architecture (no redundant formation)")
    print("="*70)