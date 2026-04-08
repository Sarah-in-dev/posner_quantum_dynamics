"""
CA1 Multi-Traversal Place Field Scenario
==========================================
Generates biologically realistic CA1 input scenarios for the quantum biology
parameter sweep. Models repeated traversals of a place field with inter-
traversal gaps, reflecting how hippocampal synapses accumulate entanglement
topology across laps.

Canonical CA1 place field pattern:
  - Animal traverses place field → 8-16 theta cycles at 8 Hz
  - Each theta cycle delivers a calcium burst to co-active synapses
  - Inter-traversal interval: 30-90s (animal running laps)
  - Dopamine arrives post-traversal (reward at end of maze)
  - Entanglement topology accumulates across traversals

Usage:
    scenario = ThetaBurstScenario(
        ca_amplitude=1e-5,
        theta_cycles_per_traversal=12,
        n_traversals=6,
        inter_traversal_interval_s=45.0,
        burst_duration_ms=50,
        theta_period_ms=125,
        dopamine_delay_s=1.0,
        silence_duration_s=60.0,
    )
    scenario.run(network, dt=0.001)
    metrics = scenario.extract_metrics(network)
    # metrics["traversal_snapshots"] has per-traversal topology trajectory
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ThetaBurstScenario:
    """
    Multi-traversal CA1 place field scenario for the sweep.

    Parameters correspond directly to STIMULUS_DIMENSIONS in quantum_dimensions.py.
    All timing in seconds internally; constructor accepts ms where noted.
    """

    # Stimulus parameters (match STIMULUS_DIMENSIONS dim_ids)
    ca_amplitude: float = 1e-5              # Peak calcium per burst (M)
    theta_cycles_per_traversal: int = 12    # Theta cycles per place field traversal
    n_traversals: int = 6                   # Number of place field traversals
    inter_traversal_interval_s: float = 45.0  # Gap between traversals (s)
    burst_duration_ms: float = 50.0         # Duration of each calcium burst (ms)
    theta_period_ms: float = 125.0          # Theta cycle period (ms) — 8 Hz canonical
    dopamine_delay_s: float = 1.0           # Dopamine arrival post-traversal (s)
    dopamine_after_traversal: int = -1      # Which traversal triggers dopamine (1-indexed, -1=last)
    silence_duration_s: float = 60.0        # Final silence after last traversal (s)

    # Internal state (populated during run)
    _ran: bool = field(default=False, init=False, repr=False)
    _traversal_snapshots: List[Dict] = field(default_factory=list, init=False, repr=False)
    _commitment_snapshot: Optional[Dict] = field(default=None, init=False, repr=False)

    @property
    def burst_duration_s(self) -> float:
        return self.burst_duration_ms / 1000.0

    @property
    def theta_period_s(self) -> float:
        return self.theta_period_ms / 1000.0

    @property
    def traversal_duration_s(self) -> float:
        return self.theta_cycles_per_traversal * self.theta_period_s

    @property
    def total_duration_s(self) -> float:
        stim = self.n_traversals * self.traversal_duration_s
        gaps = max(0, self.n_traversals - 1) * self.inter_traversal_interval_s
        return stim + gaps + self.silence_duration_s

    def _snapshot_topology(self, network, traversal_idx: int, t: float) -> Dict:
        """Capture entanglement topology metrics at a point in time."""
        net_metrics = network.get_experimental_metrics()
        return {
            "traversal": traversal_idx,
            "time_s": round(t, 4),
            "n_entangled_network": net_metrics.get("n_entangled_network", 0),
            "cross_synapse_bonds": net_metrics.get("cross_synapse_bonds", 0),
            "within_synapse_bonds": net_metrics.get("within_synapse_bonds", 0),
            "total_bonds": net_metrics.get("total_bonds", 0),
            "n_clusters": net_metrics.get("n_clusters", 0),
            "mean_connectivity": net_metrics.get("mean_connectivity", 0.0),
            "total_dimers": net_metrics.get("total_dimers", 0.0),
            "mean_coherence": net_metrics.get("mean_coherence", 0.0),
            "q2_field_kT": net_metrics.get("q2_field_kT", 0.0),
            "ps_mean": float(np.mean([
                s.get_mean_singlet_probability() for s in network.synapses
            ])),
            "committed": net_metrics.get("network_committed", False),
        }

    def _run_epoch(self, network, dt: float, duration_s: float,
                   burst_onsets: List[float], t_offset: float,
                   dopamine_time: Optional[float]) -> float:
        """
        Run a simulation epoch (traversal or silence).

        Calcium enters via voltage-gated channel physics, not direct injection.
        During each theta burst: 4 spikes at 100 Hz (2ms depolarization at
        -10 mV, 8ms rest at -70 mV). Between bursts and during silence:
        resting voltage (-70 mV).

        Args:
            network: MultiSynapseNetwork
            dt: timestep (s)
            duration_s: how long to run (s)
            burst_onsets: list of burst onset times relative to epoch start
            t_offset: global time offset for this epoch
            dopamine_time: global time when dopamine fires (or None)

        Returns:
            Global time after epoch completes.
        """
        n_steps = int(duration_s / dt)

        # Spike timing within a burst: 4 spikes at 100 Hz
        # Each spike = 2ms depolarization + 8ms rest = 10ms
        spike_period = 0.010       # 10ms per spike (100 Hz)
        depol_duration = 0.002     # 2ms depolarization
        spikes_per_burst = 4
        burst_active_duration = spikes_per_burst * spike_period  # 40ms

        for step in range(n_steps):
            t_local = step * dt
            t_global = t_offset + t_local

            # Dopamine signal
            reward = (dopamine_time is not None and
                      abs(t_global - dopamine_time) < dt * 2)

            # Determine voltage: check if we're inside a burst
            voltage = -70e-3  # default: resting
            for onset in burst_onsets:
                if onset <= t_local < onset + burst_active_duration:
                    # Inside burst — determine spike phase
                    t_in_burst = t_local - onset
                    t_in_spike = t_in_burst % spike_period
                    if t_in_spike < depol_duration:
                        voltage = -10e-3  # depolarized
                    break

            network.step(dt, {"voltage": voltage, "reward": reward})

            # Check for commitment event
            if (self._commitment_snapshot is None and
                    network.network_committed):
                self._commitment_snapshot = self._snapshot_topology(
                    network, -1, t_global
                )

        return t_offset + duration_s

    def run(self, network, dt: float = 0.001) -> None:
        """
        Execute multi-traversal place field scenario.

        For each traversal:
          1. Deliver theta_cycles_per_traversal bursts at theta frequency
          2. Snapshot topology after traversal
          3. Silence for inter_traversal_interval_s (except after last)

        Dopamine is delivered ONCE, after the traversal specified by
        dopamine_after_traversal (at traversal_end + dopamine_delay_s).
        This tests how much entanglement accumulation is needed before
        the three-factor gate can open.

        After all traversals: silence for silence_duration_s, then final snapshot.
        """
        self._traversal_snapshots = []
        self._commitment_snapshot = None
        t = 0.0

        # Disable auto-commitment so network_committed is only set
        # through the three-factor gate (requires dopamine + calcium + eligibility)
        network.disable_auto_commitment = True

        # Resolve which traversal triggers dopamine (0-indexed internally)
        if self.dopamine_after_traversal == -1:
            da_trav_idx = self.n_traversals - 1
        else:
            da_trav_idx = min(self.dopamine_after_traversal - 1,
                              self.n_traversals - 1)

        dopamine_time = None  # set when we reach the target traversal

        for trav in range(self.n_traversals):
            # Build burst onsets for this traversal
            burst_onsets = [
                c * self.theta_period_s
                for c in range(self.theta_cycles_per_traversal)
            ]

            # Run traversal (no dopamine during bursts)
            t = self._run_epoch(
                network, dt, self.traversal_duration_s,
                burst_onsets, t, None
            )

            # Snapshot after traversal (before dopamine)
            self._traversal_snapshots.append(
                self._snapshot_topology(network, trav, t)
            )

            # Compute dopamine time when we reach the target traversal
            if trav == da_trav_idx and dopamine_time is None:
                dopamine_time = t + self.dopamine_delay_s

            # Gap after traversal
            if trav < self.n_traversals - 1:
                gap = self.inter_traversal_interval_s
            elif dopamine_time is not None and dopamine_time >= t:
                # Last traversal and dopamine not yet delivered
                gap = self.dopamine_delay_s + 0.01
            else:
                gap = 0

            if gap > 0:
                t = self._run_epoch(
                    network, dt, gap,
                    [], t, dopamine_time
                )

            # Clear dopamine_time once it's in the past
            if dopamine_time is not None and t > dopamine_time:
                dopamine_time = None

        # Final silence period
        if self.silence_duration_s > 0:
            t = self._run_epoch(
                network, dt, self.silence_duration_s,
                [], t, None
            )

        # Final snapshot (after all silence)
        self._traversal_snapshots.append(
            self._snapshot_topology(network, self.n_traversals, t)
        )

        self._ran = True

    def extract_metrics(self, network) -> Dict[str, Any]:
        """
        Extract spatiotemporal output metrics after scenario execution.

        Returns a flat dict suitable for storage as a TestVector result.
        Includes per-traversal topology trajectory and commitment snapshot.
        """
        if not self._ran:
            raise RuntimeError("Call run() before extract_metrics()")

        net_metrics = network.get_experimental_metrics()

        # Per-synapse final state
        dimer_counts = []
        ps_values = []
        po4_remaining = []

        for syn in network.synapses:
            dimer_counts.append(
                float(syn.ca_phosphate.dimerization.dimer_concentration.sum())
            )
            ps_values.append(float(syn.get_mean_singlet_probability()))
            po4_remaining.append(
                float(syn.atp.phosphate.phosphate_structural.mean())
            )

        # Backbone condensation state
        backbone_eta = getattr(network, '_backbone_eta', 0.0)

        return {
            # Q1 outputs
            "mean_field_kT":          net_metrics.get("mean_field_kT", 0.0),
            "backbone_eta":           float(backbone_eta),

            # Q2 outputs
            "dimers_mean_per_syn":    float(np.mean(dimer_counts)),
            "dimers_total":           float(np.sum(dimer_counts)),
            "dimers_std":             float(np.std(dimer_counts)),
            "ps_mean":                float(np.mean(ps_values)),
            "ps_std":                 float(np.std(ps_values)),
            "po4_mean_remaining":     float(np.mean(po4_remaining)),
            "po4_fraction_consumed":  1.0 - float(np.mean(po4_remaining)) /
                                      max(po4_remaining[0] if po4_remaining else 1.0, 1e-12),

            # Network-level coordination
            "committed":              net_metrics.get("committed", False),
            "commit_level":           net_metrics.get("commit_level", 0.0),
            "n_synapses":             len(network.synapses),

            # Entanglement topology (final state)
            "n_entangled_network":    net_metrics.get("n_entangled_network", 0),
            "within_synapse_bonds":   net_metrics.get("within_synapse_bonds", 0),
            "cross_synapse_bonds":    net_metrics.get("cross_synapse_bonds", 0),
            "total_bonds":            net_metrics.get("total_bonds", 0),
            "q2_field_kT":            net_metrics.get("q2_field_kT", 0.0),
            "mean_coherence":         net_metrics.get("mean_coherence", 0.0),
            "total_dimers":           net_metrics.get("total_dimers", 0.0),
            "n_clusters":             net_metrics.get("n_clusters", 0),
            "mean_connectivity":      net_metrics.get("mean_connectivity", 0.0),

            # Traversal trajectory (list of per-traversal snapshots + final)
            "traversal_snapshots":    self._traversal_snapshots,
            "commitment_snapshot":    self._commitment_snapshot,

            # Scenario parameters (for result traceability)
            "ca_amplitude":           self.ca_amplitude,
            "theta_cycles_per_traversal": self.theta_cycles_per_traversal,
            "n_traversals":           self.n_traversals,
            "inter_traversal_interval_s": self.inter_traversal_interval_s,
            "burst_duration_ms":      self.burst_duration_ms,
            "theta_period_ms":        self.theta_period_ms,
            "dopamine_delay_s":       self.dopamine_delay_s,
            "dopamine_after_traversal": self.dopamine_after_traversal,
            "silence_duration_s":     self.silence_duration_s,
        }


def scenario_from_vector(values: Dict[str, Any]) -> ThetaBurstScenario:
    """
    Build a ThetaBurstScenario from a TestVector's values dict.

    Only extracts stimulus dimensions — network/Q1/Q2 parameters are
    applied separately to the model before running.
    """
    return ThetaBurstScenario(
        ca_amplitude=values.get("stim_ca_amplitude", 1e-5),
        theta_cycles_per_traversal=int(values.get("stim_theta_cycles", 12)),
        n_traversals=int(values.get("stim_n_traversals", 6)),
        inter_traversal_interval_s=float(values.get("stim_inter_traversal_s", 45.0)),
        burst_duration_ms=float(values.get("stim_burst_duration_ms", 50.0)),
        theta_period_ms=float(values.get("stim_theta_period_ms", 125.0)),
        dopamine_delay_s=float(values.get("stim_dopamine_delay", 1.0)),
        dopamine_after_traversal=int(values.get("stim_dopamine_after_traversal", -1)),
        silence_duration_s=float(values.get("stim_silence_duration", 60.0)),
    )
