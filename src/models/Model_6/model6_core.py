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
- Dopamine system (from dopamine_biophysics.py)

NOTHING IS PRESCRIBED - ALL EMERGENT!

This is the complete implementation for:
"Phosphorus Nuclear Spin Effects on Neural Adaptation: 
 Testing Quantum Mechanisms in Rapid BCI Learning"
 (Davidson 2025)
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
from pnc_formation import PNCFormationSystem
from posner_system import PosnerSystem
from pH_dynamics import pHDynamics

# Import dopamine system adapter
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
    - Isotope composition (P31 vs P32)
    - Temperature (for Q10 tests)
    - Dopamine levels (reward signal)
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
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup subsystems
        self._initialize_subsystems()
        
        # Data storage for analysis
        self.history = {
            'time': [],
            'calcium_peak': [],
            'atp_mean': [],
            'j_coupling_max': [],
            'pnc_total': [],
            'dimer_concentration': [],
            'trimer_concentration': [],
            'coherence_dimer': [],
            'coherence_trimer': [],
            'pH_min': [],
            'dopamine_max': [],
        }
        
        logger.info("="*70)
        logger.info("MODEL 6 QUANTUM SYNAPSE INITIALIZED")
        logger.info("="*70)
        logger.info(f"Grid: {self.grid_shape}, dx={self.dx*1e9:.1f} nm")
        logger.info(f"Isotope: {self.params.environment.fraction_P31*100:.0f}% ³¹P")
        logger.info(f"Temperature: {self.params.environment.T:.1f} K")
        logger.info("="*70)
        
    def _initialize_subsystems(self):
        """Initialize all physics subsystems"""
        
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
        
        # 2. Template positions (protein scaffolds)
        # Place at channel sites
        template_positions = [(x, y) for x, y in channel_positions[:3]]
        
        # Initialize all subsystems
        self.calcium = CalciumSystem(
            self.grid_shape, self.dx, channel_positions, self.params
        )
        
        self.atp = ATPSystem(
            self.grid_shape, self.dx, self.params
        )
        
        self.pnc = PNCFormationSystem(
            self.grid_shape, self.dx, self.params, template_positions
        )
        
        self.posner = PosnerSystem(
            self.grid_shape, self.params
        )
        
        self.pH = pHDynamics(
            self.grid_shape, self.dx, self.params
        )
        
        # Dopamine system (optional)
        if HAS_DOPAMINE:
            self.dopamine = DopamineSystemAdapter(
                grid_size=self.grid_shape[0],
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
        
        Args:
            dt: Time step (s)
            stimulus: Dictionary with control signals
                - 'voltage': Membrane voltage (V)
                - 'reward': Reward signal (boolean)
                - 'temperature': Optional temperature override (K)
        """
        stimulus = stimulus or {}
        
        # === STEP 1: CALCIUM DYNAMICS ===
        self.calcium.step(dt, stimulus)
        ca_conc = self.calcium.get_concentration()
        ca_peak = self.calcium.get_peak_concentration()
        
        # Calculate calcium influx for pH
        ca_influx = np.gradient(ca_conc, axis=0) / dt  # Rough estimate
        
        # === STEP 2: ATP & J-COUPLING ===
        pH_field = self.pH.get_pH()
        self.atp.step(dt, ca_conc, pH_field)
        
        j_coupling = self.atp.get_j_coupling()
        atp_conc = self.atp.get_atp_concentration()
        phosphate = self.atp.get_phosphate_for_posner()
        
        # === STEP 3: pH DYNAMICS ===
        # Need ATP hydrolysis rate
        atp_prev = getattr(self, '_atp_prev', atp_conc)
        atp_hydrolysis_rate = (atp_prev - atp_conc) / dt
        atp_hydrolysis_rate = np.maximum(atp_hydrolysis_rate, 0)
        self._atp_prev = atp_conc.copy()
        
        activity_level = self.calcium.channels.get_open_fraction()
        activity_field = np.ones(self.grid_shape) * activity_level
        
        self.pH.step(dt, atp_hydrolysis_rate, ca_influx, activity_field)
        pH_field = self.pH.get_pH()
        
        # === STEP 4: PNC FORMATION ===
        # Templates active where there are channels
        template_active = None  # Use defaults for now
        
        self.pnc.step(dt, ca_conc, phosphate, template_active)
        pnc_total = self.pnc.get_pnc_concentration()
        pnc_large = self.pnc.get_large_pncs()
        
        # === STEP 5: DOPAMINE (if available) ===
        if self.dopamine is not None:
            reward = stimulus.get('reward', False)
            self.dopamine.update(dt, reward=reward)
            
            # Get D2 receptor occupancy
            da_stats = self.dopamine.get_statistics()
            d2_occupancy_mean = da_stats['D2_occupancy_mean']
            
            # Make spatial field (simplified - use mean everywhere)
            dopamine_d2 = np.ones(self.grid_shape) * d2_occupancy_mean
        else:
            dopamine_d2 = np.zeros(self.grid_shape)
        
        # === STEP 6: POSNER FORMATION & QUANTUM COHERENCE ===
        temperature = stimulus.get('temperature', self.params.environment.T)
        
        self.posner.step(
            dt, pnc_large, ca_conc, j_coupling, dopamine_d2, temperature
        )
        
        # Update time
        self.time += dt
        
    def run_protocol(self, protocol_name: str = "standard", duration: float = 1.0):
        """
        Run a predefined stimulation protocol
        
        Args:
            protocol_name: Name of protocol
                - "standard": Brief calcium spike
                - "learning": Calcium spike + reward
                - "isotope_test": Compare P31 vs P32
                - "temperature_series": Q10 measurement
            duration: Total duration (s)
        """
        logger.info(f"\nRunning protocol: {protocol_name}")
        logger.info(f"Duration: {duration:.2f} s")
        
        if protocol_name == "standard":
            self._run_standard_protocol(duration)
        elif protocol_name == "learning":
            self._run_learning_protocol(duration)
        elif protocol_name == "isotope_test":
            self._run_isotope_test(duration)
        elif protocol_name == "temperature_series":
            self._run_temperature_series(duration)
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")
    
    def _run_standard_protocol(self, duration: float):
        """Standard protocol: calcium spike then recovery"""
        
        dt = self.dt
        n_steps = int(duration / dt)
        
        # Spike for first 100 ms
        spike_duration = 0.1  # s
        spike_steps = int(spike_duration / dt)
        
        for i in range(n_steps):
            # Depolarization during spike
            if i < spike_steps:
                voltage = -0.01  # -10 mV (open channels)
            else:
                voltage = -0.07  # -70 mV (rest)
            
            stimulus = {'voltage': voltage}
            
            self.step(dt, stimulus)
            
            # Record every 1 ms
            if i % int(1e-3 / dt) == 0:
                self._record_timestep()
                
                if i % int(0.1 / dt) == 0:  # Log every 100 ms
                    self._log_status()
    
    def _run_learning_protocol(self, duration: float):
        """Learning protocol: spike + delayed reward"""
        
        dt = self.dt
        n_steps = int(duration / dt)
        
        spike_duration = 0.1  # s
        spike_steps = int(spike_duration / dt)
        
        # Reward 100 ms after spike
        reward_delay = 0.1  # s
        reward_time = spike_duration + reward_delay
        reward_duration = 0.2  # s
        
        for i in range(n_steps):
            t = i * dt
            
            # Voltage
            voltage = -0.01 if i < spike_steps else -0.07
            
            # Reward
            reward = (reward_time <= t < reward_time + reward_duration)
            
            stimulus = {'voltage': voltage, 'reward': reward}
            
            self.step(dt, stimulus)
            
            if i % int(1e-3 / dt) == 0:
                self._record_timestep()
                if i % int(0.1 / dt) == 0:
                    self._log_status()
    
    def _run_isotope_test(self, duration: float):
        """
        Compare natural (P31) vs substituted (P32) isotope
        
        KEY EXPERIMENTAL PREDICTION from thesis!
        """
        logger.info("\n" + "="*70)
        logger.info("ISOTOPE SUBSTITUTION TEST")
        logger.info("="*70)
        
        # Run with natural abundance (P31)
        logger.info("\nCondition 1: Natural Abundance (³¹P)")
        params_natural = Model6Parameters()
        model_natural = Model6QuantumSynapse(params_natural)
        model_natural._run_standard_protocol(duration)
        metrics_natural = model_natural.get_experimental_metrics()
        
        # Run with P32 substitution
        logger.info("\nCondition 2: ³²P Substitution")
        params_P32 = Model6Parameters()
        params_P32.environment.fraction_P31 = 0.0
        params_P32.environment.fraction_P32 = 1.0
        model_P32 = Model6QuantumSynapse(params_P32)
        model_P32._run_standard_protocol(duration)
        metrics_P32 = model_P32.get_experimental_metrics()
        
        # Compare
        logger.info("\n" + "="*70)
        logger.info("ISOTOPE COMPARISON")
        logger.info("="*70)
        
        logger.info(f"\n³¹P (natural):")
        logger.info(f"  T2 dimer: {metrics_natural['T2_dimer_s']:.1f} s")
        logger.info(f"  Peak coherence: {metrics_natural['coherence_dimer_max']:.3f}")
        logger.info(f"  Dimer/trimer ratio: {metrics_natural['dimer_trimer_ratio']:.2f}")
        
        logger.info(f"\n³²P (substituted):")
        logger.info(f"  T2 dimer: {metrics_P32['T2_dimer_s']:.1f} s")
        logger.info(f"  Peak coherence: {metrics_P32['coherence_dimer_max']:.3f}")
        logger.info(f"  Dimer/trimer ratio: {metrics_P32['dimer_trimer_ratio']:.2f}")
        
        # Fold changes
        t2_fold = metrics_natural['T2_dimer_s'] / metrics_P32['T2_dimer_s']
        coherence_fold = (metrics_natural['coherence_dimer_max'] / 
                         metrics_P32['coherence_dimer_max']) if metrics_P32['coherence_dimer_max'] > 0 else np.inf
        
        logger.info(f"\nFold Changes:")
        logger.info(f"  T2: {t2_fold:.2f}x")
        logger.info(f"  Coherence: {coherence_fold:.2f}x")
        logger.info(f"  Thesis prediction: 1.57-10x")
        
        if t2_fold > 5:
            logger.info("\n✓ ISOTOPE EFFECT CONFIRMED!")
        else:
            logger.warning(f"\n⚠ Weak isotope effect (fold change = {t2_fold:.2f}x)")
    
    def _run_temperature_series(self, duration: float):
        """
        Test temperature dependence (Q10 measurement)
        
        KEY TEST: Quantum processes should be temperature-independent!
        """
        logger.info("\n" + "="*70)
        logger.info("TEMPERATURE SERIES (Q10 TEST)")
        logger.info("="*70)
        
        temperatures = [305.15, 310.15, 313.15]  # 32, 37, 40°C
        results = []
        
        for T in temperatures:
            logger.info(f"\nTemperature: {T-273.15:.0f}°C")
            
            params_temp = Model6Parameters()
            params_temp.environment.T = T
            model_temp = Model6QuantumSynapse(params_temp)
            
            # Run protocol
            dt = model_temp.dt
            n_steps = int(duration / dt)
            
            for i in range(n_steps):
                voltage = -0.01 if i < int(0.1/dt) else -0.07
                stimulus = {'voltage': voltage, 'temperature': T}
                model_temp.step(dt, stimulus)
                
                if i % int(1e-3 / dt) == 0:
                    model_temp._record_timestep()
            
            metrics = model_temp.get_experimental_metrics()
            results.append((T, metrics))
            
            logger.info(f"  T2 dimer: {metrics['T2_dimer_s']:.1f} s")
            logger.info(f"  Peak coherence: {metrics['coherence_dimer_max']:.3f}")
        
        # Calculate Q10
        if len(results) >= 2:
            T1, metrics1 = results[0]
            T2, metrics2 = results[-1]
            
            Q10 = (metrics2['T2_dimer_s'] / metrics1['T2_dimer_s']) ** (10 / (T2 - T1))
            
            logger.info(f"\n" + "="*70)
            logger.info(f"Q10 = {Q10:.2f}")
            logger.info(f"Expected (quantum): ~1.0")
            logger.info(f"Expected (classical): >2.0")
            
            if Q10 < 1.2:
                logger.info("\n✓ TEMPERATURE-INDEPENDENT (Quantum signature!)")
            else:
                logger.warning(f"\n⚠ Temperature-dependent (Q10 = {Q10:.2f})")
    
    def _record_timestep(self):
        """Record current state"""
        self.history['time'].append(self.time)
        
        # Get metrics from each subsystem
        ca_metrics = self.calcium.get_experimental_metrics()
        atp_metrics = self.atp.get_experimental_metrics()
        pnc_metrics = self.pnc.get_experimental_metrics()
        posner_metrics = self.posner.get_experimental_metrics()
        pH_metrics = self.pH.get_experimental_metrics()
        
        self.history['calcium_peak'].append(ca_metrics['peak_ca_uM'])
        self.history['atp_mean'].append(atp_metrics['atp_mean_mM'])
        self.history['j_coupling_max'].append(atp_metrics['j_coupling_max_Hz'])
        self.history['pnc_total'].append(pnc_metrics['pnc_peak_nM'])
        self.history['dimer_concentration'].append(posner_metrics['dimer_peak_nM'])
        self.history['trimer_concentration'].append(posner_metrics['trimer_peak_nM'])
        self.history['coherence_dimer'].append(posner_metrics['coherence_dimer_mean'])
        self.history['coherence_trimer'].append(posner_metrics['coherence_trimer_mean'])
        self.history['pH_min'].append(pH_metrics['pH_min'])
        
        if self.dopamine is not None:
            da_stats = self.dopamine.get_statistics()
            self.history['dopamine_max'].append(da_stats['max'] * 1e9)  # nM
        else:
            self.history['dopamine_max'].append(0)
    
    def _log_status(self):
        """Log current simulation status"""
        if len(self.history['time']) == 0:
            return
        
        logger.info(f"t={self.time*1e3:.0f} ms: "
                   f"Ca={self.history['calcium_peak'][-1]:.1f} μM, "
                   f"J={self.history['j_coupling_max'][-1]:.1f} Hz, "
                   f"Dimer={self.history['dimer_concentration'][-1]:.1f} nM, "
                   f"Coherence={self.history['coherence_dimer'][-1]:.3f}")
    
    def get_experimental_metrics(self) -> Dict[str, float]:
        """
        Return all experimental metrics for validation
        
        This is what gets compared to experimental data!
        """
        if len(self.history['time']) == 0:
            logger.warning("No data recorded yet")
            return {}
        
        metrics = {}
        
        # Peak values
        metrics['calcium_peak_uM'] = max(self.history['calcium_peak'])
        metrics['j_coupling_max_Hz'] = max(self.history['j_coupling_max'])
        metrics['dimer_peak_nM'] = max(self.history['dimer_concentration'])
        metrics['trimer_peak_nM'] = max(self.history['trimer_concentration'])
        metrics['coherence_dimer_max'] = max(self.history['coherence_dimer'])
        metrics['coherence_trimer_max'] = max(self.history['coherence_trimer'])
        
        # Ratios
        dimer_total = sum(self.history['dimer_concentration'])
        trimer_total = sum(self.history['trimer_concentration'])
        metrics['dimer_trimer_ratio'] = (dimer_total / trimer_total 
                                         if trimer_total > 0 else np.inf)
        
        # Isotope-dependent
        metrics['T2_dimer_s'] = self.posner.quantum.T2_eff_dimer
        metrics['T2_trimer_s'] = self.posner.quantum.T2_eff_trimer
        metrics['P31_fraction'] = self.params.environment.fraction_P31
        
        # pH
        metrics['pH_min'] = min(self.history['pH_min'])
        metrics['pH_drop'] = 7.35 - metrics['pH_min']
        
        return metrics
    
    def save_results(self, filename: Optional[str] = None):
        """Save simulation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model6_results_{timestamp}.h5"
        
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            params_dict = self.params.to_dict()
            for key, value in params_dict.items():
                if isinstance(value, dict):
                    subgroup = params_group.create_group(key)
                    for k, v in value.items():
                        subgroup.attrs[k] = v
            
            # Save history
            history_group = f.create_group('history')
            for key, values in self.history.items():
                history_group.create_dataset(key, data=np.array(values))
            
            # Save final metrics
            metrics = self.get_experimental_metrics()
            metrics_group = f.create_group('metrics')
            for key, value in metrics.items():
                metrics_group.attrs[key] = value
        
        logger.info(f"Results saved to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL 6: COMPLETE QUANTUM SYNAPSE")
    print("Emergent quantum processing from first principles")
    print("="*70)
    
    # Run standard protocol
    print("\n### TEST 1: Standard Protocol ###")
    model = Model6QuantumSynapse()
    model.run_protocol("standard", duration=0.5)
    
    metrics = model.get_experimental_metrics()
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    model.save_results()
    
    print("\n### TEST 2: Isotope Comparison ###")
    model.run_protocol("isotope_test", duration=0.5)
    
    print("\n### TEST 3: Temperature Series ###")
    model.run_protocol("temperature_series", duration=0.3)
    
    print("\n" + "="*70)
    print("All tests complete!")
    print("="*70)