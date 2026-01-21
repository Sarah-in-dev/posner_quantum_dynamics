"""
Quantum RNN - Recurrent Neural Network with Model 6 Quantum Synapses
=====================================================================

A recurrent neural network where EACH synaptic connection is a full
Model6QuantumSynapse with quantum eligibility traces.

Architecture:
    External Input → [Neurons with Quantum Synapses] → Output
    
    - N rate-based neurons (simple leaky integrators)
    - N×(N-1) recurrent synapses (all-to-all, no self-connections)
    - Each synapse is a FULL Model6QuantumSynapse instance

Key Innovation:
    The eligibility trace timescale is NOT a hyperparameter.
    It EMERGES from quantum physics:
    - P31 isotope: T₂ ≈ 100s (enables long-delay credit assignment)
    - P32 isotope: T₂ ≈ 0.4s (control - should fail at long delays)

Credit Assignment Protocol:
    1. ENCODING: Stimulus activates subset of neurons
       → Coincident pre/post activity → Dimers form → Eligibility created
       
    2. DELAY: Stimulus removed, network may maintain activity
       → Dimers persist (P31) or decay (P32)
       → Eligibility maintained by quantum coherence
       
    3. REWARD: Dopamine signal arrives
       → Three-factor gate: eligibility × dopamine × calcium
       → If all present → CaMKII commitment → Weight change

Author: Sarah Davidson
University of Florida
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


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
class RNNConfig:
    """Configuration for Quantum RNN"""
    
    # Network architecture
    n_neurons: int = 10
    connection_probability: float = 1.0  # 1.0 = all-to-all (no self)
    
    # Isotope selection (determines T₂)
    isotope: str = 'P31'  # 'P31' (~100s) or 'P32' (~0.4s)
    
    # Activity → Voltage mapping thresholds
    pre_activity_threshold: float = 0.3
    post_activity_threshold: float = 0.3
    
    # Voltage levels for synaptic drive (Volts)
    voltage_resting: float = -70e-3      # -70 mV
    voltage_partial: float = -50e-3      # -50 mV (pre only)
    voltage_depolarized: float = -10e-3  # -10 mV (pre AND post)
    
    # Weight mapping
    base_weight: float = 0.5             # Baseline synaptic weight
    weight_scale: float = 1.0            # How much commitment affects weight
    
    # Microtubule invasion (enables EM coupling)
    mt_invaded: bool = True
    
    # Neuron config
    neuron_config: NeuronConfig = field(default_factory=NeuronConfig)


@dataclass
class NetworkState:
    """Snapshot of network state at one timestep"""
    time: float
    
    # Neuron state
    rates: np.ndarray                    # [n_neurons] firing rates
    membrane_potentials: np.ndarray      # [n_neurons] membrane voltages
    
    # Aggregate synapse state
    total_dimers: int
    mean_eligibility: float
    mean_singlet_prob: float
    n_committed: int
    
    # Per-synapse (optional, for analysis)
    synapse_dimers: Optional[Dict[Tuple[int, int], int]] = None
    synapse_eligibilities: Optional[Dict[Tuple[int, int], float]] = None


# =============================================================================
# RATE NEURON (inline to avoid import issues)
# =============================================================================

class RateNeuron:
    """Simple rate-based neuron - physics is in the synapses"""
    
    def __init__(self, config: NeuronConfig, neuron_id: int = 0):
        self.config = config
        self.neuron_id = neuron_id
        self.membrane_potential = config.resting_potential
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
        self.total_input = 0.0


# =============================================================================
# QUANTUM RNN
# =============================================================================

class QuantumRNN:
    """
    Recurrent neural network with Model 6 quantum synapses.
    
    Each synapse is a FULL Model6QuantumSynapse, preserving:
    - Calcium dynamics with voltage-gated channels
    - ATP hydrolysis and phosphate release
    - CaHPO₄ → Ca₆(PO₄)₄ dimer formation
    - Singlet probability evolution (quantum coherence)
    - Three-factor gate (eligibility × dopamine × calcium)
    - CaMKII commitment and plasticity
    
    Usage:
        from model6_core import Model6QuantumSynapse
        from model6_parameters import Model6Parameters
        
        rnn = QuantumRNN(
            config=RNNConfig(n_neurons=10, isotope='P31'),
            SynapseClass=Model6QuantumSynapse,
            SynapseParams=Model6Parameters
        )
        
        # Run simulation
        for t in range(n_steps):
            state = rnn.step(dt, external_input, reward=False)
        
        # Apply reward after delay
        state = rnn.step(dt, external_input, reward=True)
    """
    
    def __init__(self,
                 config: RNNConfig,
                 SynapseClass: Any,
                 SynapseParams: Any,
                 seed: Optional[int] = None):
        """
        Initialize Quantum RNN.
        
        Args:
            config: Network configuration
            SynapseClass: Model6QuantumSynapse class
            SynapseParams: Model6Parameters class
            seed: Random seed for reproducibility
        """
        self.config = config
        self.SynapseClass = SynapseClass
        self.SynapseParams = SynapseParams
        self.n_neurons = config.n_neurons
        self.rng = np.random.default_rng(seed)
        
        # Create neurons
        self.neurons: List[RateNeuron] = []
        for i in range(self.n_neurons):
            neuron = RateNeuron(config=config.neuron_config, neuron_id=i)
            self.neurons.append(neuron)
        
        # Create synapses - each is a FULL Model6 instance
        # Key: (pre_idx, post_idx) → synapse
        self.synapses: Dict[Tuple[int, int], Any] = {}
        self._create_synapses(seed)
        
        # Network state
        self.time = 0.0
        self._current_rates = np.zeros(self.n_neurons)
        
        # Statistics
        self.n_synapses = len(self.synapses)
        
        logger.info(f"QuantumRNN initialized: {self.n_neurons} neurons, "
                   f"{self.n_synapses} synapses, isotope={config.isotope}")
    
    def _create_synapses(self, seed: Optional[int] = None):
        """Create Model6 synapses for all recurrent connections"""
        
        synapse_id = 0
        
        for pre in range(self.n_neurons):
            for post in range(self.n_neurons):
                if pre == post:
                    continue  # No self-connections
                
                # Probabilistic connectivity
                if self.rng.random() > self.config.connection_probability:
                    continue
                
                # Create Model6 parameters with correct isotope
                params = self.SynapseParams()
                
                # Set isotope
                if hasattr(params, 'environment'):
                    params.environment.fraction_P31 = 1.0 if self.config.isotope == 'P31' else 0.0
                
                # Enable EM coupling
                if hasattr(params, 'em_coupling_enabled'):
                    params.em_coupling_enabled = True
                
                # Create synapse
                synapse = self.SynapseClass(params=params)
                
                # Set microtubule invasion
                if hasattr(synapse, 'set_microtubule_invasion'):
                    synapse.set_microtubule_invasion(self.config.mt_invaded)
                
                # Store metadata on synapse
                synapse._rnn_pre = pre
                synapse._rnn_post = post
                synapse._rnn_id = synapse_id
                
                self.synapses[(pre, post)] = synapse
                synapse_id += 1
    
    def _compute_synapse_voltage(self, pre_rate: float, post_rate: float) -> float:
        """
        Map neuron firing rates to synaptic voltage.
        
        This implements NMDA-like coincidence detection:
        - Neither active: resting voltage (-70 mV)
        - Pre only: partial depolarization (-50 mV)
        - Pre AND post: full depolarization (-10 mV) → Ca²⁺ influx → dimers
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
        """
        Extract effective synaptic weight from Model6 state.
        
        Weight = base + (commitment_level × scale)
        """
        base = self.config.base_weight
        committed_level = getattr(synapse, '_committed_memory_level', 0.0)
        return base + committed_level * self.config.weight_scale
    
    def step(self, 
             dt: float, 
             external_input: np.ndarray,
             reward: bool = False) -> NetworkState:
        """
        Advance network by one timestep.
        
        Args:
            dt: Timestep in seconds
            external_input: External input current to each neuron [n_neurons]
            reward: Whether reward/dopamine signal is present
            
        Returns:
            NetworkState with current network state
        """
        self.time += dt
        
        # Get current firing rates (from previous step)
        rates = self._current_rates.copy()
        
        # Compute recurrent input for each neuron
        recurrent_input = np.zeros(self.n_neurons)
        
        for (pre, post), synapse in self.synapses.items():
            # Compute voltage based on pre/post activity
            voltage = self._compute_synapse_voltage(rates[pre], rates[post])
            
            # Step the FULL Model6 synapse
            stimulus = {
                'voltage': voltage,
                'reward': reward
            }
            synapse.step(dt, stimulus)
            
            # Compute weighted input to post neuron
            weight = self._get_synapse_weight(synapse)
            recurrent_input[post] += weight * rates[pre]
        
        # Update each neuron
        total_input = external_input + recurrent_input
        
        for i, neuron in enumerate(self.neurons):
            neuron.step(dt, total_input[i])
            self._current_rates[i] = neuron.firing_rate
        
        # Compute and return network state
        return self._compute_network_state()
    
    def _compute_network_state(self) -> NetworkState:
        """Compute aggregate network state"""
        
        total_dimers = 0
        eligibilities = []
        singlet_probs = []
        n_committed = 0
        
        for synapse in self.synapses.values():
            # Dimer count
            if hasattr(synapse, 'dimer_particles'):
                total_dimers += len(synapse.dimer_particles.dimers)
            
            # Eligibility
            if hasattr(synapse, 'get_eligibility'):
                eligibilities.append(synapse.get_eligibility())
            elif hasattr(synapse, '_current_eligibility'):
                eligibilities.append(synapse._current_eligibility)
            
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
            n_committed=n_committed
        )
    
    # =========================================================================
    # CONVENIENCE METHODS FOR EXPERIMENTS
    # =========================================================================
    
    def get_rates(self) -> np.ndarray:
        """Get current firing rates [n_neurons]"""
        return self._current_rates.copy()
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get current effective weight matrix [n_neurons × n_neurons]"""
        W = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            W[pre, post] = self._get_synapse_weight(synapse)
        return W
    
    def get_eligibility_matrix(self) -> np.ndarray:
        """Get current eligibility matrix [n_neurons × n_neurons]"""
        E = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            if hasattr(synapse, 'get_eligibility'):
                E[pre, post] = synapse.get_eligibility()
            elif hasattr(synapse, '_current_eligibility'):
                E[pre, post] = synapse._current_eligibility
        return E
    
    def get_commitment_matrix(self) -> np.ndarray:
        """Get commitment state matrix [n_neurons × n_neurons]"""
        C = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            C[pre, post] = getattr(synapse, '_committed_memory_level', 0.0)
        return C
    
    def get_dimer_matrix(self) -> np.ndarray:
        """Get dimer count matrix [n_neurons × n_neurons]"""
        D = np.zeros((self.n_neurons, self.n_neurons))
        for (pre, post), synapse in self.synapses.items():
            if hasattr(synapse, 'dimer_particles'):
                D[pre, post] = len(synapse.dimer_particles.dimers)
        return D
    
    def reset(self):
        """Reset network to initial state (keeps synapses but resets activity)"""
        for neuron in self.neurons:
            neuron.reset()
        self._current_rates = np.zeros(self.n_neurons)
        self.time = 0.0
    
    def reset_synapses(self):
        """Full reset including synaptic state"""
        self.reset()
        for synapse in self.synapses.values():
            if hasattr(synapse, 'reset'):
                synapse.reset()
    
    def get_network_metrics(self) -> Dict:
        """Get comprehensive network metrics for logging"""
        state = self._compute_network_state()
        
        return {
            'time': self.time,
            'n_neurons': self.n_neurons,
            'n_synapses': self.n_synapses,
            'isotope': self.config.isotope,
            'total_dimers': state.total_dimers,
            'mean_eligibility': state.mean_eligibility,
            'mean_singlet_prob': state.mean_singlet_prob,
            'n_committed': state.n_committed,
            'mean_rate': float(np.mean(self._current_rates)),
            'active_neurons': int(np.sum(self._current_rates > 0.5)),
        }
    
    def __repr__(self) -> str:
        return (f"QuantumRNN(n_neurons={self.n_neurons}, "
                f"n_synapses={self.n_synapses}, "
                f"isotope={self.config.isotope})")


# =============================================================================
# CLASSICAL RNN FOR COMPARISON
# =============================================================================

class ClassicalRNN:
    """
    Classical RNN with exponential eligibility decay.
    
    For comparison - the decay constant τ is a HYPERPARAMETER here,
    unlike QuantumRNN where it emerges from physics.
    """
    
    def __init__(self, n_neurons: int, tau: float = 5.0, seed: Optional[int] = None):
        """
        Args:
            n_neurons: Number of neurons
            tau: Eligibility decay time constant (HYPERPARAMETER)
            seed: Random seed
        """
        self.n_neurons = n_neurons
        self.tau = tau
        self.rng = np.random.default_rng(seed)
        
        # Neurons
        self.membrane_potentials = np.zeros(n_neurons)
        self.rates = np.zeros(n_neurons)
        
        # Synapses (simple: just weights and eligibility)
        self.weights = np.ones((n_neurons, n_neurons)) * 0.5
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
        self.eligibility = np.zeros((n_neurons, n_neurons))
        self.committed = np.zeros((n_neurons, n_neurons), dtype=bool)
        
        self.time = 0.0
        self.n_synapses = n_neurons * (n_neurons - 1)
    
    def step(self, dt: float, external_input: np.ndarray, reward: bool = False) -> Dict:
        """Advance network by one timestep"""
        self.time += dt
        
        # Recurrent input
        recurrent = self.weights @ self.rates
        total_input = external_input + recurrent
        
        # Update neurons (simple leaky integrator)
        tau_m = 0.02
        self.membrane_potentials += dt * (-self.membrane_potentials + total_input) / tau_m
        
        # Sigmoid
        x = 5.0 * (self.membrane_potentials - 1.0)
        x = np.clip(x, -500, 500)
        new_rates = 1.0 / (1.0 + np.exp(-x))
        
        # Update eligibility
        # Form eligibility on coincident activity
        for pre in range(self.n_neurons):
            for post in range(self.n_neurons):
                if pre == post:
                    continue
                    
                pre_active = self.rates[pre] > 0.3
                post_active = self.rates[post] > 0.3
                
                if pre_active and post_active:
                    # Eligibility rises
                    self.eligibility[pre, post] += dt * (1.0 - self.eligibility[pre, post])
                
                # Exponential decay (THIS IS THE HYPERPARAMETER)
                self.eligibility[pre, post] *= np.exp(-dt / self.tau)
        
        # Apply reward
        if reward:
            for pre in range(self.n_neurons):
                for post in range(self.n_neurons):
                    if pre == post:
                        continue
                    if self.eligibility[pre, post] > 0.3 and not self.committed[pre, post]:
                        self.committed[pre, post] = True
                        self.weights[pre, post] += 0.5 * self.eligibility[pre, post]
        
        self.rates = new_rates
        
        return {
            'time': self.time,
            'mean_eligibility': np.mean(self.eligibility),
            'n_committed': np.sum(self.committed),
            'mean_rate': np.mean(self.rates)
        }
    
    def reset(self):
        self.membrane_potentials = np.zeros(self.n_neurons)
        self.rates = np.zeros(self.n_neurons)
        self.eligibility = np.zeros((self.n_neurons, self.n_neurons))
        self.time = 0.0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_quantum_rnn(
    n_neurons: int = 10,
    isotope: str = 'P31',
    SynapseClass: Any = None,
    SynapseParams: Any = None,
    seed: Optional[int] = None
) -> QuantumRNN:
    """
    Factory function to create Quantum RNN.
    
    Args:
        n_neurons: Number of neurons
        isotope: 'P31' (quantum, ~100s) or 'P32' (control, ~0.4s)
        SynapseClass: Model6QuantumSynapse class
        SynapseParams: Model6Parameters class  
        seed: Random seed
        
    Returns:
        Configured QuantumRNN
    """
    config = RNNConfig(
        n_neurons=n_neurons,
        isotope=isotope
    )
    
    return QuantumRNN(
        config=config,
        SynapseClass=SynapseClass,
        SynapseParams=SynapseParams,
        seed=seed
    )


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM RNN")
    print("=" * 70)
    print("\nThis module requires Model6 classes:")
    print()
    print("  from model6_core import Model6QuantumSynapse")
    print("  from model6_parameters import Model6Parameters")
    print()
    print("  rnn = create_quantum_rnn(")
    print("      n_neurons=10,")
    print("      isotope='P31',")
    print("      SynapseClass=Model6QuantumSynapse,")
    print("      SynapseParams=Model6Parameters")
    print("  )")
    print()
    print("  # Run encoding")
    print("  stimulus = np.zeros(10)")
    print("  stimulus[:3] = 2.0  # Activate first 3 neurons")
    print("  ")
    print("  for t in range(100):")
    print("      state = rnn.step(dt=0.01, external_input=stimulus, reward=False)")
    print("  ")
    print("  print(f'Dimers formed: {state.total_dimers}')")
    print("  print(f'Mean eligibility: {state.mean_eligibility:.3f}')")
    print()
    
    # Test classical RNN
    print("\n--- Testing Classical RNN (no Model6 required) ---")
    classical = ClassicalRNN(n_neurons=5, tau=5.0)
    
    stimulus = np.zeros(5)
    stimulus[:2] = 2.0
    
    # Encoding
    for _ in range(100):
        classical.step(0.01, stimulus, reward=False)
    
    print(f"After encoding: eligibility={classical.eligibility.mean():.3f}")
    
    # Delay
    for _ in range(500):
        classical.step(0.01, np.zeros(5), reward=False)
    
    print(f"After 5s delay: eligibility={classical.eligibility.mean():.6f}")
    
    # Reward
    classical.step(0.01, np.zeros(5), reward=True)
    print(f"After reward: committed={classical.committed.sum()}")