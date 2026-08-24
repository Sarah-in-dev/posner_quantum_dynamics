#!/usr/bin/env python3
"""
THE NETWORK PRIMITIVE — the three constraints the abstraction was missing, taken from the biological code.

WHY. cgl_primitive.py models ONE synapse's store. Its binding rule binds every co-active unit to every other,
so simultaneous input collapses into a single blob: 22 features presented together return ONE component of
size 22 (verified). That is why the graph was inert in both established benchmarks -- it had nothing to
compute. The biological network (multi_synapse_network.py) does NOT bind indiscriminately, because it is
holding three constraints this abstraction never implemented.

TAKEN DIRECTLY FROM multi_synapse_network.py (values included, so the provenance is checkable):

 1. SPATIAL COUPLING. `spatial_factor = exp(-dist_um / coupling_length_um)`, coupling_length_um = 5.0
    (line 718). Bonds weaken with distance; a unit does not couple equally to everything.

 2. BOUNDED DEGREE -- THE SPIN LEDGER. `_claim_cross_spins` (line 324): a forming bond must claim one 31P
    nucleus at EACH endpoint, and a dimer has FOUR. Intra- and cross-synapse bonds spend the SAME four slots.
    A bond that cannot claim a free spin is REFUSED and counted as FRUSTRATION. This is a hard capacity
    constraint: components must COMPETE for connectivity. Nothing like it exists in the current abstraction.

 3. A GLOBAL GATE. `k_cross = 0.5 * sqrt(eta_i*eta_j) * w_spatial * P_S_i*P_S_j` (line 494). Both endpoints
    need eta > 0, so with the backbone uncondensed NO distributed component can form at all. (This is not
    hypothetical: F4-b measured eta = 0 throughout, which is why cross-synapse bonds have never once been
    observed in this codebase -- see the F4-b log entry.)

 4. FIDELITY THRESHOLD. `F_werner = P_S_i * P_S_j * w_spatial`, bond counts iff F > 0.5 (WERNER_ENTANGLEMENT_BOUND).

 5. DISSOLUTION. k_release = 1/T2 + 1/tau_dimer = 1/216 + 1/200 = 9.63e-3 /s (K_RELEASE_PHYSICAL), so a bond
    dies when its coherence does rather than at an unmoored rate.

HONEST NAMING. `distance` is physical in the biology. Abstract features have no intrinsic geometry, so a
POSITION MUST BE ASSIGNED, and that assignment is a modelling choice, not something the biology dictates.
Random positions make the coupling a random sparse graph -- an architectural prior comparable to random
receptive fields. Said plainly rather than presented as derived.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np


@dataclass
class NetworkCGLParams:
    # --- trace dynamics (unchanged from cgl_primitive) ---
    tau: float = 216.0
    trace_floor: float = 0.25
    # --- 1. spatial coupling ---
    coupling_length: float = 5.0        # exp(-d/lambda); biology: 5.0 um
    # --- 2. bounded degree: 4 31P nuclei per dimer ---
    spin_capacity: int = 4
    # --- 3. global gate ---
    eta: float = 1.0                    # 0 => no bonds can form at all (the measured F4-b regime)
    # --- 4/5. fidelity and dissolution ---
    edge_threshold: float = 0.5         # Werner bound
    k_entangle_base: float = 0.5        # K_ENTANGLE_EM_BASE
    k_release: float = 1.0 / 216.0 + 1.0 / 200.0   # K_RELEASE_PHYSICAL
    bind_window: float = 2.0
    seed: Optional[int] = None


class NetworkCoherenceGatedPrimitive:
    """Units carry traces AND positions. Bonds form stochastically under distance, capacity and a gate."""

    def __init__(self, n_units: int, positions: Optional[np.ndarray] = None,
                 params: Optional[NetworkCGLParams] = None):
        self.n = n_units
        self.p = params or NetworkCGLParams()
        self.rng = np.random.default_rng(self.p.seed)
        if positions is None:                                  # assigned, not derived -- see docstring
            positions = self.rng.uniform(0.0, 10.0, size=(n_units, 2))
        self.pos = np.asarray(positions, dtype=float)
        d = np.linalg.norm(self.pos[:, None, :] - self.pos[None, :, :], axis=-1)
        self.w = np.exp(-d / self.p.coupling_length)            # 1. spatial coupling
        np.fill_diagonal(self.w, 0.0)
        self.trace = np.full(n_units, self.p.trace_floor)
        self.bonded = np.zeros((n_units, n_units), dtype=bool)
        self.last_active = np.full(n_units, -np.inf)
        self.t = 0.0
        self.n_frustrated = 0                                   # 2. refusals for want of a free spin

    # ---------------- capacity ----------------
    def degree(self) -> np.ndarray:
        return self.bonded.sum(axis=1)

    def _has_spin(self, i: int) -> bool:
        return self.degree()[i] < self.p.spin_capacity

    # ---------------- binding ----------------
    def activate(self, active: Sequence[bool], dt: float = 1.0) -> None:
        """Co-active units ATTEMPT to bond. Each attempt is gated by eta, distance and both traces, and is
        refused outright when either endpoint has spent all four of its spins."""
        act = np.asarray(active, dtype=bool)
        idx = np.flatnonzero(act)
        self.trace[idx] = 1.0
        self.last_active[idx] = self.t
        recent = np.flatnonzero((self.t - self.last_active) <= self.p.bind_window)
        cand = np.unique(np.concatenate([idx, recent])) if len(recent) else idx
        if self.p.eta <= 0.0:
            return                                              # 3. gate shut: no distributed component
        pairs = [(int(i), int(j)) for a, i in enumerate(cand) for j in cand[a + 1:]
                 if not self.bonded[i, j]]
        self.rng.shuffle(pairs)                                 # competition order must not be positional
        for i, j in pairs:
            k_cross = (self.p.k_entangle_base * self.p.eta * self.w[i, j]
                       * self.trace[i] * self.trace[j])         # the biological rate law
            if self.rng.random() >= 1.0 - np.exp(-k_cross * dt):
                continue
            if not (self._has_spin(i) and self._has_spin(j)):   # 2. spin ledger: refuse, count frustration
                self.n_frustrated += 1
                continue
            self.bonded[i, j] = self.bonded[j, i] = True

    def decay(self, dt: float = 1.0) -> None:
        self.t += dt
        self.trace = self.p.trace_floor + (self.trace - self.p.trace_floor) * np.exp(-dt / self.p.tau)
        if self.bonded.any():                                   # 5. coherence-limited dissolution
            live = np.argwhere(np.triu(self.bonded))
            if len(live):
                die = self.rng.random(len(live)) < (1.0 - np.exp(-self.p.k_release * dt))
                for (i, j) in live[die]:
                    self.bonded[i, j] = self.bonded[j, i] = False

    # ---------------- components ----------------
    def fidelity(self) -> np.ndarray:
        F = np.outer(self.trace, self.trace) * self.w           # 4. F = P_i * P_j * w_spatial
        np.fill_diagonal(F, 0.0)
        return F * self.bonded

    def components(self) -> List[Set[int]]:
        adj = self.fidelity() > self.p.edge_threshold
        bound = np.sqrt(self.p.edge_threshold)
        seen, comps = np.zeros(self.n, dtype=bool), []
        for s in range(self.n):
            if seen[s]:
                continue
            if not adj[s].any():
                if self.trace[s] > bound:
                    seen[s] = True; comps.append({s})
                continue
            stack, comp = [s], set(); seen[s] = True
            while stack:
                u = stack.pop(); comp.add(u)
                for v in np.flatnonzero(adj[u]):
                    if not seen[v]:
                        seen[v] = True; stack.append(v)
            comps.append(comp)
        return comps

    def clear_episode(self) -> None:
        self.bonded[:] = False
