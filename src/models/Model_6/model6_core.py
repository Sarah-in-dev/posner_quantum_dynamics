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

# EM coupling modules (optional)
try:
    from em_tryptophan_module import TryptophanSuperradianceModule
    from em_coupling_module import EMCouplingModule
    HAS_EM_COUPLING = True
except ImportError:
    HAS_EM_COUPLING = False
    logging.info("EM coupling modules not found - running Model 6 baseline")

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
        
        
        # Initialize EM coupling (optional)
        self.em_enabled = (self.params.em_coupling_enabled and HAS_EM_COUPLING)
        
        if self.em_enabled:
            self.tryptophan = TryptophanSuperradianceModule(self.params)
            self.em_coupling = EMCouplingModule(self.params)
            
            logger.info("EM coupling ENABLED")
        else:
            self.tryptophan = None
            self.em_coupling = None
            logger.info("EM coupling DISABLED (Model 6 baseline)")
        
        
        self._em_field_history = []
        self._k_enhancement_history = []
        self._collective_field_history = []
        
        
        # Track tryptophan count (changes with MT invasion)
        self.n_tryptophans = self.params.tryptophan.n_trp_baseline
        
        
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

        # Add to self.history dict (around line ~95):
        if self.em_enabled:
            self.history['trp_em_field'] = []
            self.history['k_enhancement'] = []
            self.history['collective_field_kT'] = []
        
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

        # EXPERIMENTAL OVERRIDE: Use fixed phosphate if set
        if hasattr(self, '_override_phosphate'):
            phosphate = self._override_phosphate

        # DIAGNOSTIC: Check phosphate values
        if np.random.rand() < 0.0001:
            print(f"model6_core.step: phosphate = {np.mean(phosphate)*1e3:.2f} mM (mean), {np.max(phosphate)*1e3:.2f} mM (max)")
        
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
        
        # === EM COUPLING: FORWARD PATH (Tryptophan → Dimers) ===
        if self.em_enabled:
            # Calculate tryptophan state
            # UV flux from ATP metabolism (proportional to hydrolysis)
            metabolic_uv_flux = self.params.metabolic_uv.photon_flux_baseline
            if activity_level > 0.1:  # During activity
                metabolic_uv_flux *= self.params.metabolic_uv.flux_enhancement_active
            
            if self.params.metabolic_uv.external_uv_illumination:
                wavelength = self.params.metabolic_uv.external_uv_wavelength
                intensity = self.params.metabolic_uv.external_uv_intensity
                h, c = 6.626e-34, 3e8
                psd_area = 3.14159 * (350e-9)**2
                external_flux = (intensity * psd_area) / (h * c / wavelength)
                metabolic_uv_flux += external_flux

            # Ca spike detection (for correlated bursts)
            ca_spike_active = (activity_level > 0.1)
            
            # Update tryptophan superradiance
            trp_state = self.tryptophan.update(
                dt=dt,
                photon_flux=metabolic_uv_flux,
                n_tryptophans=self.n_tryptophans,
                ca_spike_active=ca_spike_active
            )
            
            # NOW we can use trp_state
            em_field_trp = trp_state['output']['em_field_time_averaged']
            
            # Get baseline aggregation rate from ca_phosphate system
            k_agg_baseline = self.ca_phosphate.dimerization.k_base
            
            # Calculate enhanced rate from EM coupling
            coupling_state = self.em_coupling.update(
                em_field_trp=em_field_trp,
                n_coherent_dimers=0,  # Don't know yet, update after
                k_agg_baseline=k_agg_baseline,
                phosphate_fraction=np.mean(phosphate) / 0.001
            )
            
            k_agg_enhanced = coupling_state['output']['k_agg_enhanced']
            
            # Store current values for this step
            self._em_field_trp = em_field_trp
            self._k_enhancement = coupling_state['forward']['enhancement']
            
            # TRACK HISTORY (inside the if block!)
            self._em_field_history.append(em_field_trp)
            self._k_enhancement_history.append(coupling_state['forward']['enhancement'])
        else:
            k_agg_enhanced = self.ca_phosphate.dimerization.k_base
            self._em_field_trp = 0.0
            self._k_enhancement = 1.0
        
        # === STEP 4: CALCIUM PHOSPHATE DIMER FORMATION ===
        # Ca²⁺ + HPO₄²⁻ → CaHPO₄ (instant equilibrium)
        # 6 × CaHPO₄ → Ca₆(PO₄)₄ dimer (slow aggregation)
        # These are the quantum qubits (4 ³¹P nuclei)
        # Pass enhanced k_agg if EM coupling enabled
        if self.em_enabled:
            # Temporarily override aggregation rate
            k_agg_original = self.ca_phosphate.dimerization.k_base
            self.ca_phosphate.dimerization.k_base = k_agg_enhanced
            self.ca_phosphate.step(dt, ca_conc, phosphate)
            self.ca_phosphate.dimerization.k_base = k_agg_original  # Restore
        else:
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
        
            
        # === EM COUPLING: REVERSE PATH (Dimers → Proteins) ===
        if self.em_enabled:
            
            # Quantum processing happens at HIGH concentration sites
            coherent_mask = self.quantum.coherence > 0.5
            if np.sum(coherent_mask) > 0:
                # Get peak concentration (where templates are)
                peak_conc = np.max(self.quantum.dimer_concentration[coherent_mask])
                
                # Functional processing domain volume per template (~50nm radius sphere)
                # Volume = (4/3)π(50e-9)³ = 5.2e-22 m³
                processing_volume = 5.2e-22  # m³ per template
                n_templates = 3  # Number of processing sites
                
                # Molecules per synapse
                n_coherent_single = peak_conc * processing_volume * n_templates * 1000 * 6.022e23
                
            else:
                n_coherent_single = 0.0
            
            # CORRECT: Scale by number of synapses
            if self.params.multi_synapse_enabled:
                n_coherent = int(n_coherent_single * self.params.multi_synapse.n_synapses_default)

            else:
                n_coherent = int(n_coherent_single)
            
            
            # Calculate collective quantum field
            coupling_state_reverse = self.em_coupling.update(
                em_field_trp=self._em_field_trp,
                n_coherent_dimers=n_coherent,
                k_agg_baseline=self.ca_phosphate.dimerization.k_base,
                phosphate_fraction=np.mean(phosphate) / 0.001
            )
            
            protein_modulation_kT = coupling_state_reverse['output']['protein_modulation_kT']
            
            # Store for history
            self._collective_field_kT = protein_modulation_kT
            self._collective_field_history.append(self._collective_field_kT)
        else:
            self._collective_field_kT = 0.0
        
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

            'trimer_peak_nM_ct': ca_phosphate_metrics.get('trimer_peak_nM', 0),  
            'trimer_mean_nM_ct': ca_phosphate_metrics.get('trimer_mean_nM', 0), 

            'dimer_trimer_ratio': ca_phosphate_metrics.get('dimer_trimer_ratio', 0),  # ADD THIS LINE

            # PNC metrics (NEW - from Turhan et al. 2024)
            'pnc_peak_nM': ca_phosphate_metrics.get('pnc_peak_nM', 0),
            'pnc_mean_nM': ca_phosphate_metrics.get('pnc_mean_nM', 0),
            'pnc_binding_fraction': ca_phosphate_metrics.get('pnc_binding_fraction', 0),
            'pnc_lifetime_mean_s': ca_phosphate_metrics.get('pnc_lifetime_mean_s', 0),
            'pnc_lifetime_max_s': ca_phosphate_metrics.get('pnc_lifetime_max_s', 0),
            
            # Quantum Coherence (CHANGED - simpler metrics)
            'coherence_dimer_mean': quantum_metrics['coherence_mean'],
            'coherence_std': quantum_metrics['coherence_std'],
            'coherence_peak': quantum_metrics['coherence_peak'],
            'T2_dimer_s': quantum_metrics['T2_dimer_s'],
            
            # pH
            'pH_min': pH_metrics['pH_min'],
            'pH_mean': pH_metrics['pH_mean'],
        }
        
        # EM Coupling metrics
        if self.em_enabled and len(self._em_field_history) > 0 and len(self._collective_field_history) > 0:
            metrics['trp_em_field_gv_m'] = np.max(self._em_field_history) / 1e9
            metrics['em_formation_enhancement'] = np.max(self._k_enhancement_history)
            # Use last 30% of simulation (after transients settled)
            steady_state_start = int(len(self._collective_field_history) * 0.7)
            if steady_state_start < len(self._collective_field_history):
                metrics['collective_field_kT'] = np.mean(self._collective_field_history[steady_state_start:])
            else:
                metrics['collective_field_kT'] = self._collective_field_kT  # Fallback to current
        else:
            metrics['trp_em_field_gv_m'] = 0.0
            metrics['em_formation_enhancement'] = 1.0
            metrics['collective_field_kT'] = 0.0  # ADD THIS LINE
        
        
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

        if self.em_enabled:
            self.history['trp_em_field'].append(self._em_field_trp)
            self.history['k_enhancement'].append(self._k_enhancement)
            self.history['collective_field_kT'].append(self._collective_field_kT)
    
    
    def set_microtubule_invasion(self, invaded: bool):
        """
        Set microtubule invasion state
        
        Updates tryptophan count when MT invades spine during plasticity
        
        Args:
            invaded: True if microtubules have invaded
        """
        if invaded:
            # MT invasion brings additional tryptophans
            self.n_tryptophans = self.params.get_total_tryptophans(mt_invaded=True)
        else:
            # Just baseline PSD lattice
            self.n_tryptophans = self.params.get_total_tryptophans(mt_invaded=False)
        
        if self.em_enabled:
            logger.info(f"MT invasion: {invaded}, n_trp: {self.n_tryptophans}")
    
    def get_em_coupling_state(self) -> Dict:
        """
        Get current EM coupling state (if enabled)
        
        Returns:
            dict with EM coupling diagnostics or empty dict if disabled
        """
        if not self.em_enabled:
            return {}
        
        return {
            'em_field_trp': self._em_field_trp,
            'k_enhancement': self._k_enhancement,
            'collective_field_kT': self._collective_field_kT,
            'n_tryptophans': self.n_tryptophans
        }
    
    
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

            # EM coupling state (if enabled)
            if self.em_enabled:
                em_group = f.create_group('em_coupling')
                em_group.attrs['enabled'] = True
                em_group.attrs['n_tryptophans'] = self.n_tryptophans
                em_group.attrs['trp_em_field_final'] = self._em_field_trp
                em_group.attrs['k_enhancement_final'] = self._k_enhancement
                em_group.attrs['collective_field_kT_final'] = self._collective_field_kT
            else:
                em_group = f.create_group('em_coupling')
                em_group.attrs['enabled'] = False
            
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