"""
Coherence-Gated Learning — the ABSTRACT primitive, extracted from the Model 6 biophysics.

WHAT THIS IS. Model 6 simulates the molecular substrate (calcium-phosphate dimers, their coherence, CaMKII,
DARPP-32/PP1). That simulation is a research INSTRUMENT: it tells us what the computational primitive is. It is
not, and cannot be, the deliverable — 6 synapses for 65 s of biological time costs ~35 h of compute, because
simulating physics is enormously more expensive than being physics.

This module is the deliverable side: the primitive with the physics removed. It is licensed by a measurement,
not an assumption — the substrate-necessity test (2026-08-24, `results/f3_substrate_test/`) replaced the
emergent dimer-population coherence with a plain decaying scalar of the same time constant and found the two
statistically indistinguishable once a trace existed (0.917 vs 1.000, permutation p = 0.16). The eligibility
trace really is reproducible as a fading number. Per `coherence-gated-learning`: "This is NOT quantum computing
on classical hardware (that's just slow simulation). This is abstracting the computational primitives the
biology reveals and implementing them as a new architecture."

THE PRIMITIVE (each element traced to the Model 6 result that validated it):
  1. LONG GRADED ELIGIBILITY — a per-unit trace set by that unit's own activity, decaying over a horizon far
     longer than the activity itself, and UNREADABLE below a threshold.
     [Model 6: the coherent P_S tag, readable above the Werner floor; delay curve F3-f credits 5-30 s where a
      short classical trace is dead.]
  2. GLOBAL SCALAR REWARD — one number broadcast identically to every unit, carrying NO information about which
     unit deserves it. No gradient, no per-parameter error signal, no stored computation graph.
     [Model 6: dopamine is volume transmission — the same signal reaches every synapse.]
  3. CREDIT LANDS LOCALLY ANYWAY — the product of (2) and (1): only units holding a readable trace can commit.
     [Model 6 F4-a: 8 synapses, 3 driven, one global reward -> precision 1.000, every committer was driven;
      within-run permutation p = 0.0003.]
  4. REWARD IS NECESSARY — activity alone never commits; the reward gates the commitment itself.
     [Model 6: DAPK1 gates CaMKII-GluN2B binding, suppressed only by the reward signal. F4-a no-reward arm:
      0 commitments across all 32 slots. This is what dissolved 'calcium domination'.]
  5. STOCHASTIC, THRESHOLDED COMMITMENT — not proportional; a probabilistic latch.
     [Model 6: DDSC — "dendritic, delayed and STOCHASTIC CaMKII activation" (Jain 2024).]
  6. STRUCTURAL SELF-MAINTAINING MEMORY — once committed, the change persists on its own and is protected;
     it is not held up by continued input.
     [Model 6: the CaMKII-GluN2B complex persists after CaM and pT286 decay (Cell Reports 2024).]
  7. SIGN EMERGES FROM THE REWARD — potentiation vs depression is not imposed; it follows the reward's sign.
     [Model 6: LTP/LTD emerges from PP1 activity, never an imposed +/-1.]

WHAT IS DELIBERATELY NOT HERE: no coherence, no quantum state, no Werner bound as physics. The threshold below
which a trace is unreadable is kept because it is computationally load-bearing (it is what makes the horizon
finite and graded), not because it is quantum. Claims about quantum advantage do NOT transfer to this module —
it is the (A) architecture, classical by construction.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class CGLParams:
    # --- eligibility trace ---
    tau: float = 40.0            # trace time constant, in the same units as `dt` passed to step()
    trace_floor: float = 0.25    # value a fully-decayed trace relaxes to (an unreadable baseline)
    readable_threshold: float = 0.707   # below this the trace carries no credit (finite, graded horizon)
    p_trace_forms: float = 1.0   # probability that activity actually lays a trace (substrate stochasticity;
                                 # 1.0 = deterministic. Model 6 measured ~2/3 at its drive.)
    # --- commitment ---
    commit_gain: float = 3.0     # maps (trace strength x |reward|) onto a commit probability
    commit_threshold: float = 0.5
    # --- learning ---
    lr: float = 1.0              # weight step applied on commitment
    w_min: float = 0.0
    w_max: float = 4.0
    # --- persistence ---
    decay_uncommitted: float = 0.0   # committed weights are structural: they do NOT decay (primitive 6)
    seed: Optional[int] = None


class CoherenceGatedLearner:
    """N units, each with an eligibility trace and a weight. Learns from a delayed GLOBAL scalar reward.

    Usage per timestep:
        learner.activate(active_mask)     # which units were driven this step (lays traces)
        learner.decay(dt)                 # traces fade
        learner.reward(r)                 # a global scalar; may arrive many steps later, or never
    """

    def __init__(self, n_units: int, params: Optional[CGLParams] = None):
        self.n = n_units
        self.p = params or CGLParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.trace = np.full(n_units, self.p.trace_floor, dtype=float)
        self.w = np.zeros(n_units, dtype=float)
        self.committed = np.zeros(n_units, dtype=bool)
        self.n_commits = 0
        self.n_reward_events = 0

    # ---- primitive 1: activity lays a local trace ----
    def activate(self, active: Sequence[bool]):
        active = np.asarray(active, dtype=bool)
        forms = active & (self.rng.random(self.n) < self.p.p_trace_forms)
        self.trace[forms] = 1.0
        return forms

    def decay(self, dt: float = 1.0):
        f = np.exp(-dt / self.p.tau)
        self.trace = self.p.trace_floor + (self.trace - self.p.trace_floor) * f

    # ---- the readable, graded horizon ----
    def readable(self) -> np.ndarray:
        return self.trace > self.p.readable_threshold

    def strength(self) -> np.ndarray:
        """Trace rescaled to [0,1]; 0 at the floor, 1 at a fresh trace."""
        s = (self.trace - self.p.trace_floor) / (1.0 - self.p.trace_floor)
        return np.clip(s, 0.0, 1.0)

    # ---- primitives 2-7: one global scalar decides, locally ----
    def reward(self, r: float) -> np.ndarray:
        """Deliver a GLOBAL scalar reward. Returns the boolean mask of units that committed this event."""
        self.n_reward_events += 1
        if r == 0.0:
            return np.zeros(self.n, dtype=bool)          # primitive 4: no reward, no commitment
        s = self.strength() * self.readable()            # primitive 1+3: only readable traces can be credited
        p_commit = np.clip(self.p.commit_gain * s * abs(r), 0.0, 1.0)
        fired = self.rng.random(self.n) < p_commit       # primitive 5: stochastic, thresholded
        if not fired.any():
            return fired
        self.w[fired] = np.clip(self.w[fired] + self.p.lr * np.sign(r) * s[fired],
                                self.p.w_min, self.p.w_max)   # primitive 7: sign follows the reward
        self.committed |= fired                          # primitive 6: structural latch
        self.n_commits += int(fired.sum())
        self.trace[fired] = self.p.trace_floor           # the trace is consumed by the readout
        return fired

    def reset_traces(self):
        self.trace[:] = self.p.trace_floor
