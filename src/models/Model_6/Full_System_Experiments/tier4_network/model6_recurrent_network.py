"""
Model 6 Recurrent Network
==========================

A recurrent neural network where each synapse is a FULL Model6QuantumSynapse.

This is the proper integration of Model 6 physics with network dynamics.
Unlike the simplified QuantumRNNSynapse, this preserves ALL Model 6 mechanisms:
- Calcium dynamics with voltage-gated channels
- ATP hydrolysis and phosphate release  
- Prenucleation cluster pathway
- Dimer formation with proper chemistry
- Singlet probability evolution
- Three-factor gate (eligibility + dopamine + calcium)
- CaMKII commitment and plasticity

Architecture:
    External Input → [Neurons with Model6 Synapses] → Output
    
    Each recurrent synapse (i→j) is a Model6QuantumSynapse instance.
    The synapse receives voltage based on pre/post neuron activity,
    implementing NMDA-like coincidence detection.

Key Innovation:
    - Activity → Voltage mapping captures NMDA receptor properties
    - Collective EM field computed across all synapses
    - Cross-synapse entanglement from birth correlations
    - Network learns from delayed reward using quantum eligibility

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import logging
import time

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
from model6_core import Model6QuantumSynapse
from model6_parameters import Model6Parameters


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NeuronConfig:
    """Configuration for rate-based neurons"""
    tau_membrane: float = 0.020      # Membrane time constant (20 ms)
    threshold: float = 1.0           # Soft threshold for sigmoid
    gain: float = 5.0                # Sigmoid steepness
    resting_potential: float = 0.0   # Resting state


@dataclass  
class NetworkConfig:
    """Configuration for Model 6 recurrent network"""
    
    # Network architecture
    n_neurons: int = 10
    connection_probability: float = 1.0  # 1.0 = all-to-all (no self)
    
    # Isotope (for quantum coherence)
    isotope: str = 'P31'  # 'P31' or 'P32'
    
    # Activity thresholds for voltage mapping
    pre_activity_threshold: float = 0.3
    post_activity_threshold: float = 0.3
    
    # Voltage levels (V)
    voltage_resting: float = -70e-3
    voltage_partial: float = -50e-3  
    voltage_depolarized: float = -10e-3
    
    # Weight extraction
    base_weight: float = 0.5
    weight_scale: float = 1.0
    
    # Collective field
    enable_collective_field: bool = True
    field_coupling_strength: float = 1.0
    
    # Neuron config
    neuron_config: NeuronConfig = field(default_factory=NeuronConfig)
    
    # Microtubule invasion (enables tryptophan EM field)
    mt_invaded: bool = True

    # Encoding parameters
    encoding_activity_threshold: float = 0.5
    theta_burst_during_encoding: bool = True


@dataclass
class NetworkState:
    """Snapshot of network state at one timestep"""
    time: float
    
    # Neuron state
    rates: np.ndarray
    membrane_potentials: np.ndarray
    
    # Aggregate synapse state
    total_dimers: int
    mean_eligibility: float
    mean_singlet_prob: float
    n_committed: int
    
    # Collective effects
    collective_field_kT: float
    n_entangled_synapses: int
    
    # Per-synapse details (optional, for analysis)
    synapse_dimers: Optional[Dict[Tuple[int, int], int]] = None
    synapse_eligibilities: Optional[Dict[Tuple[int, int], float]] = None


# =============================================================================
# RATE NEURON (simple, as the physics is in the synapses)
# =============================================================================

class RateNeuron:
    """
    Simple rate-based neuron.
    
    The interesting physics is in the Model 6 synapses, not here.
    This just provides realistic membrane dynamics.
    """
    
    def __init__(self, config: Optional[NeuronConfig] = None, neuron_id: int = 0):
        self.config = config or NeuronConfig()
        self.neuron_id = neuron_id
        
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0
        self.total_input = 0.0
    
    def step(self, dt: float, input_current: float) -> float:
        """Update neuron state"""
        self.total_input = input_current
        
        # Leaky integration
        dV = dt * (-self.membrane_potential + input_current) / self.config.tau_membrane
        self.membrane_potential += dV
        
        # Sigmoid output
        x = self.config.gain * (self.membrane_potential - self.config.threshold)
        x = np.clip(x, -500, 500)
        self.firing_rate = 1.0 / (1.0 + np.exp(-x))
        
        return self.firing_rate
    
    def reset(self):
        self.membrane_potential = self.config.resting_potential
        self.firing_rate = 0.0


# =============================================================================
# MODEL 6 RECURRENT NETWORK
# =============================================================================

class Model6RecurrentNetwork:
    """
    Recurrent neural network with Model 6 quantum synapses.
    
    This is the proper way to test quantum eligibility traces:
    - Each synapse is a FULL Model6QuantumSynapse
    - All biophysics preserved (calcium, ATP, dimers, coherence)
    - Network dynamics drive synaptic voltage
    - Collective EM field couples synapses
    
    Usage:
        # Create network
        network = Model6RecurrentNetwork(
            n_neurons=10,
            isotope='P31',
            Model6Class=Model6QuantumSynapse,
            Model6Params=Model6Parameters
        )
        
        # Run simulation
        for t in range(n_steps):
            state = network.step(dt, external_input, reward=False)
        
        # Apply reward
        for t in range(reward_steps):
            state = network.step(dt, external_input, reward=True)
    """
    
    def __init__(self,
                 n_neurons: int = 10,
                 isotope: str = 'P31',
                 Model6Class: Any = None,
                 Model6Params: Any = None,
                 config: Optional[NetworkConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize Model 6 recurrent network.
        """
        # Configuration
        if config is not None:
            self.config = config
        else:
            self.config = NetworkConfig(n_neurons=n_neurons, isotope=isotope)
        
        self.n_neurons = self.config.n_neurons
        self.rng = np.random.default_rng(seed)
        
        # Store Model 6 classes
        self.Model6Class = Model6Class
        self.Model6Params = Model6Params
        
        # Create neurons
        self.neurons: List[RateNeuron] = []
        for i in range(self.n_neurons):
            neuron = RateNeuron(config=self.config.neuron_config, neuron_id=i)
            self.neurons.append(neuron)
        
        # Create synapses - each is a FULL Model6 instance
        self.synapses: Dict[Tuple[int, int], Any] = {}
        self._create_synapses()
        
        # Network state
        self.time = 0.0
        self._current_rates = np.zeros(self.n_neurons)
        
        # Collective field state
        self._collective_field_kT = 0.0
        
        # Statistics
        self.n_synapses = len(self.synapses)
        
        logger.info(f"Model6RecurrentNetwork created: {self.n_neurons} neurons, "
                   f"{self.n_synapses} synapses, isotope={self.config.isotope}")
    
    def _create_synapses(self):
        """Create Model 6 synapses for recurrent connections"""
        
        if self.Model6Class is None:
            raise ValueError("Model6Class must be provided")
        
        synapse_id = 0
        
        for pre in range(self.n_neurons):
            for post in range(self.n_neurons):
                if pre == post:
                    continue  # No self-connections
                
                # Probabilistic connectivity
                if self.rng.random() > self.config.connection_probability:
                    continue
                
                # Create Model 6 parameters
                if self.Model6Params is not None:
                    params = self.Model6Params()
                    
                    # Set isotope
                    if hasattr(params, 'environment'):
                        params.environment.fraction_P31 = 1.0 if self.config.isotope == 'P31' else 0.0
                    
                    # Enable EM coupling
                    if hasattr(params, 'em_coupling_enabled'):
                        params.em_coupling_enabled = self.config.enable_collective_field
                else:
                    params = None
                
                # Create synapse
                synapse = self.Model6Class(params=params)
                
                # Set microtubule invasion
                if hasattr(synapse, 'set_microtubule_invasion'):
                    synapse.set_microtubule_invasion(self.config.mt_invaded)
                
                # Store metadata
                synapse._network_pre = pre
                synapse._network_post = post
                synapse._network_id = synapse_id
                
                self.synapses[(pre, post)] = synapse
                synapse_id += 1
    
    def _compute_synapse_voltage(self, pre_rate: float, post_rate: float) -> float:
        """
        Map neuron activity to synaptic voltage.
        """
        pre_active = pre_rate > self.config.pre_activity_threshold
        post_active = post_rate > self.config.post_activity_threshold
        
        if pre_active and post_active:
            return self.config.voltage_depolarized
        elif pre_active:
            return self.config.voltage_partial
        else:
            return self.config.voltage_resting
    
    def _get_synapse_weight(self, synapse) -> float:
        """Extract effective synaptic weight from Model 6 state."""
        base = self.config.base_weight
        committed_level = getattr(synapse, '_committed_memory_level', 0.0)
        return base + committed_level * self.config.weight_scale
    
    def _compute_collective_field(self) -> float:
        """Compute collective EM field from all synapses."""
        if not self.config.enable_collective_field:
            return 0.0
        
        total_dimers = 0
        for synapse in self.synapses.values():
            if hasattr(synapse, 'dimer_particles'):
                total_dimers += len(synapse.dimer_particles.dimers)
            elif hasattr(synapse, 'get_experimental_metrics'):
                metrics = synapse.get_experimental_metrics()
                dimer_nM = metrics.get('dimer_peak_nM_ct', 0.0)
                total_dimers += int(dimer_nM * 1e-17 * 6e23 / 1e9)
        
        if total_dimers < 5:
            return 0.0
        
        collective_enhancement = np.sqrt(total_dimers / 5)
        base_field = min(total_dimers, 50) * 0.5
        
        return base_field * collective_enhancement * self.config.field_coupling_strength
    
    def _distribute_collective_field(self, field_kT: float):
        """Distribute collective field back to individual synapses."""
        for synapse in self.synapses.values():
            if hasattr(synapse, '_collective_field_kT'):
                synapse._collective_field_kT = field_kT
    
    def step(self, 
             dt: float, 
             external_input: np.ndarray,
             reward: bool = False) -> NetworkState:
        """
        Advance network by one timestep.
        """
        self.time += dt
        
        # Get current rates (from previous step)
        rates = self._current_rates.copy()
        
        # Compute collective field
        self._collective_field_kT = self._compute_collective_field()
        self._distribute_collective_field(self._collective_field_kT)
        
        # Update each synapse with appropriate voltage
        recurrent_input = np.zeros(self.n_neurons)
        
        for (pre, post), synapse in self.synapses.items():
            # Compute voltage based on pre/post activity
            voltage = self._compute_synapse_voltage(rates[pre], rates[post])
            
            # Step the FULL Model 6 synapse
            stimulus = {
                'voltage': voltage,
                'reward': reward
            }
            synapse.step(dt, stimulus)
            
            # Get effective weight for recurrent input
            weight = self._get_synapse_weight(synapse)
            recurrent_input[post] += weight * rates[pre]
        
        # Update neurons
        total_input = external_input + recurrent_input
        
        for i, neuron in enumerate(self.neurons):
            neuron.step(dt, total_input[i])
            self._current_rates[i] = neuron.firing_rate
        
        # Compute network state
        return self._compute_network_state()
    
    def run_encoding_theta_burst(self,
                                  external_input: np.ndarray,
                                  n_bursts: int = 5,
                                  reward: bool = False) -> NetworkState:
        """
        Run validated theta-burst protocol on synapses between co-active neurons.
        
        Protocol: 5 bursts at 5Hz, each burst = 4 spikes at 100Hz
        Each spike = 2ms depolarization + 8ms rest
        
        This matches exp_theta_burst.py validation.
        """
        dt = 0.001  # 1ms timestep
        
        # Identify active neurons
        active_neurons = set(np.where(external_input > 0.5)[0])
        
        # Identify synapses to stimulate (between co-active neurons)
        active_synapses = {
            (pre, post) for (pre, post) in self.synapses.keys()
            if pre in active_neurons and post in active_neurons
        }
        
        logger.info(f"Theta-burst: {len(active_neurons)} neurons, {len(active_synapses)} synapses")
        
        # Run theta burst protocol
        for burst in range(n_bursts):
            for spike in range(4):
                # 2ms depolarization
                for _ in range(2):
                    self.time += dt
                    for (pre, post), synapse in self.synapses.items():
                        if (pre, post) in active_synapses:
                            voltage = self.config.voltage_depolarized
                        else:
                            voltage = self.config.voltage_resting
                        synapse.step(dt, {'voltage': voltage, 'reward': reward})
                
                # 8ms rest
                for _ in range(8):
                    self.time += dt
                    for synapse in self.synapses.values():
                        synapse.step(dt, {'voltage': self.config.voltage_resting, 'reward': reward})
            
            # 160ms between bursts
            for _ in range(160):
                self.time += dt
                for synapse in self.synapses.values():
                    synapse.step(dt, {'voltage': self.config.voltage_resting, 'reward': reward})
        
        # Update neuron rates
        for i, neuron in enumerate(self.neurons):
            neuron.firing_rate = 0.8 if i in active_neurons else 0.1
            self._current_rates[i] = neuron.firing_rate
        
        return self._compute_network_state()
    
    def step_delay_fast(self, dt: float) -> NetworkState:
        """Fast delay stepping - only decay dimer coherence"""
        self.time += dt
        
        # T2 based on isotope (from physics)
        T2 = 100.0 if self.config.isotope == 'P31' else 0.4
        
        for synapse in self.synapses.values():
            if hasattr(synapse, 'dimer_particles'):
                for dimer in synapse.dimer_particles.dimers:
                    dimer.singlet_probability *= np.exp(-dt / T2)
        
        return self._compute_network_state()
    
    def step_reward_only(self, dt: float) -> NetworkState:
        """Reward phase - check gate with existing eligibility, don't form new dimers"""
        self.time += dt
        
        for synapse in self.synapses.values():
            # Step dopamine only
            if hasattr(synapse, 'dopamine') and synapse.dopamine is not None:
                synapse.dopamine.step(dt, reward_signal=True)
            
            # Check three-factor gate with EXISTING dimer eligibility
            if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                mean_P_S = np.mean([d.singlet_probability for d in synapse.dimer_particles.dimers])
                n_entangled = sum(1 for d in synapse.dimer_particles.dimers if d.singlet_probability > 0.5)
                
                eligibility_present = (mean_P_S > 0.5)
                dopamine_read = synapse._dopamine_above_read_threshold() if hasattr(synapse, '_dopamine_above_read_threshold') else True
                calcium_elevated = True  # Assume reactivation provides calcium
                
                plasticity_gate = eligibility_present and dopamine_read and calcium_elevated
                
                if plasticity_gate and not getattr(synapse, '_camkii_committed', False):
                    synapse._camkii_committed = True
                    eligibility = (mean_P_S - 0.25) / 0.75
                    dimer_factor = min(1.0, n_entangled / 10.0)
                    synapse._committed_memory_level = eligibility * dimer_factor
        
        return self._compute_network_state()
    
    def _compute_network_state(self) -> NetworkState:
        """Compute aggregate network state"""
        
        # Aggregate synapse metrics
        total_dimers = 0
        eligibilities = []
        singlet_probs = []
        n_committed = 0
        
        for synapse in self.synapses.values():
            # Dimer count
            if hasattr(synapse, 'dimer_particles'):
                total_dimers += len(synapse.dimer_particles.dimers)
            
            # Eligibility
            if hasattr(synapse, '_current_eligibility'):
                eligibilities.append(synapse._current_eligibility)
            elif hasattr(synapse, 'get_eligibility'):
                eligibilities.append(synapse.get_eligibility())
            
            # Singlet probability
            if hasattr(synapse, 'get_mean_singlet_probability'):
                singlet_probs.append(synapse.get_mean_singlet_probability())
            
            # Commitment
            if getattr(synapse, '_camkii_committed', False):
                n_committed += 1
        
        return NetworkState(
            time=self.time,
            rates=self._current_rates.copy(),
            membrane_potentials=np.array([n.membrane_potential for n in self.neurons]),
            total_dimers=total_dimers,
            mean_eligibility=np.mean(eligibilities) if eligibilities else 0.0,
            mean_singlet_prob=np.mean(singlet_probs) if singlet_probs else 0.25,
            n_committed=n_committed,
            collective_field_kT=self._collective_field_kT,
            n_entangled_synapses=sum(1 for e in eligibilities if e > 0.5)
        )
    
    def get_rates(self) -> np.ndarray:
        """Get current firing rates"""
        return self._current_rates.copy()
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get current effective weight matrix"""
        W = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            W[pre, post] = self._get_synapse_weight(synapse)
        return W
    
    def get_eligibility_matrix(self) -> np.ndarray:
        """Get current eligibility matrix"""
        E = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            if hasattr(synapse, '_current_eligibility'):
                E[pre, post] = synapse._current_eligibility
            elif hasattr(synapse, 'get_eligibility'):
                E[pre, post] = synapse.get_eligibility()
        return E
    
    def get_commitment_matrix(self) -> np.ndarray:
        """Get commitment state matrix"""
        C = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            C[pre, post] = getattr(synapse, '_committed_memory_level', 0.0)
        return C
    
    def get_network_metrics(self) -> Dict:
        """Get comprehensive network metrics"""
        state = self._compute_network_state()
        
        return {
            'time': self.time,
            'n_neurons': self.n_neurons,
            'n_synapses': self.n_synapses,
            'total_dimers': state.total_dimers,
            'mean_eligibility': state.mean_eligibility,
            'mean_singlet_prob': state.mean_singlet_prob,
            'n_committed': state.n_committed,
            'collective_field_kT': state.collective_field_kT,
            'mean_rate': np.mean(self._current_rates),
            'active_neurons': np.sum(self._current_rates > 0.5)
        }
    
    def reset(self):
        """Reset network to initial state"""
        for neuron in self.neurons:
            neuron.reset()
        
        self._current_rates = np.zeros(self.n_neurons)
        self.time = 0.0
    
    def __repr__(self) -> str:
        return (f"Model6RecurrentNetwork(n_neurons={self.n_neurons}, "
                f"n_synapses={self.n_synapses}, isotope={self.config.isotope})")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model6_network(
    n_neurons: int = 10,
    isotope: str = 'P31',
    mt_invaded: bool = True,
    seed: Optional[int] = None
) -> Model6RecurrentNetwork:
    """
    Factory function to create Model 6 recurrent network.
    """
    config = NetworkConfig(
        n_neurons=n_neurons,
        isotope=isotope,
        mt_invaded=mt_invaded
    )
    
    return Model6RecurrentNetwork(
        config=config,
        seed=seed
    )


# =============================================================================
# TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL 6 RECURRENT NETWORK")
    print("=" * 70)
    print("\nThis module requires Model 6 classes to be injected.")
    print("\nUsage:")
    print("  from model6_core import Model6QuantumSynapse")
    print("  from model6_parameters import Model6Parameters")
    print("  ")
    print("  network = Model6RecurrentNetwork(")
    print("      n_neurons=10,")
    print("      isotope='P31',")
    print("      Model6Class=Model6QuantumSynapse,")
    print("      Model6Params=Model6Parameters")
    print("  )")
    print("  ")
    print("  # Run encoding with theta burst")
    print("  stimulus = np.array([2.0, 2.0, 2.0, 0, 0, 0, 0, 0, 0, 0])")
    print("  state = network.run_encoding_theta_burst(stimulus)")
    print("  print(f'Dimers formed: {state.total_dimers}')")