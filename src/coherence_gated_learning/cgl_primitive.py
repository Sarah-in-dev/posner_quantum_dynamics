#!/usr/bin/env python3
"""
THE CONSOLIDATED PRIMITIVE — every validated mechanism in ONE implementation.

WHY THIS FILE EXISTS. The mechanisms were each discovered and validated in a separate benchmark script, so
they ended up spread across five partial implementations, none holding more than four of the seven:

    mechanism                      graph_cgl  agent  forager  Depress  temporal
    two-timescale binding             yes       -       -        -      partial
    per-component joint collapse      yes      yes     yes      yes        -
    partial reactivation              yes       -      yes      yes       yes
    evidence accumulation              -       yes     yes      yes       yes
    active depression (LTD arm)        -        -       -       yes        -
    commit-probability action          -       yes     yes      yes        -
    trace decay / readability         yes       -       -        -        yes

Benchmarking any one of them measures a SUBSET, and a result would not say which. This file is the single
reference implementation; test_primitive_regression.py checks it still reproduces each original result.

THE SEVEN MECHANISMS, and what each is grounded in (Model 6):

 1. TRACE DECAY WITH A FLOOR. Each unit carries a trace decaying with `tau` toward `trace_floor`, and is
    READABLE only above `read_threshold`. Model 6: dimer singlet probability P_S decaying on T2=216 s toward
    the 0.25 thermal floor, readable until it crosses the Werner floor 1/sqrt(2)=0.7071 (~107 s).

 2. TWO-TIMESCALE BINDING (load-bearing). Bond FORMATION is fast (`bind_window`); bond PERSISTENCE is long
    (`tau`). Model 6 separates these: dimers bond within one calcium event (ms) while the coherence they
    carry lasts ~100 s. Collapsing them makes binding indiscriminate -- every live node binds to every other,
    every trial yields one junk component, nothing is learnable.

 3. PER-COMPONENT JOINT COLLAPSE. ONE coin for a whole connected component: it commits together or not at
    all. This is what makes credit CONJUNCTIVE rather than per-unit, and it is the mechanism a scalar
    eligibility trace provably cannot express.

 4. SINGLETON COMPONENTS. A node with no surviving edges but a readable trace is its own component (one
    synapse = one nanodomain = one component). Bound at sqrt(edge_threshold), the same Werner floor the
    pairwise edges use.

 5. PARTIAL REACTIVATION. An unseen combination is answered by the stored components it OVERLAPS, weighted
    by Jaccard similarity ** sharpness. Without it, components are atomic keys and an unseen combination sits
    at exactly chance (measured). HONEST NAMING: mechanically this is similarity-weighted retrieval; it is
    motivated by the attractor framing, not measured in Model 6.

 6. COMMIT-PROBABILITY ACTION SELECTION. One coin per candidate at p = clip(gain * value). Exactly one fires
    -> exploit; several -> choose among them; none -> uniform. Exploration falls out of how consolidated the
    memory is; there is NO epsilon, temperature, or schedule.

 7. ACTIVE DEPRESSION (the LTD arm). value <- value + rate*(r - value), rate scaled by `ltd_gain` when the
    outcome contradicts. Model 6 does not wait for decay: PP1 strips CaMKII-pThr286 and DAPK1 disrupts the
    GluN2B complex, actively taking a memory apart when reward turns against it. Without this arm the system
    locks in (a novelty-by-count drive was tried FIRST and failed). HONEST NAMING: this update rule is the
    standard incremental/exponentially-weighted rule and is NOT claimed as novel; what is distinctive is the
    conjunctive representation it updates and the commit-probability exploration around it.

TWO WAYS TO SUPPLY STRUCTURE:
  GRAPH MODE    -- call activate()/decay(); components() derives the groups from temporal correlation.
  EXPLICIT MODE -- pass unit sets straight to value()/choose()/learn() when the grouping is given by the
                   data (tabular features). Same store, same collapse, same depression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np


@dataclass
class CGLParams:
    # --- 1. trace dynamics (Model 6: P_S on T2=216 s, thermal floor 0.25, Werner floor 0.7071) ---
    tau: float = 216.0
    trace_floor: float = 0.25
    read_threshold: float = 0.7071
    # --- 2. binding: FAST formation, LONG persistence ---
    bind_window: float = 2.0
    edge_threshold: float = 0.5
    # --- 5. partial reactivation ---
    overlap_generalization: bool = True
    overlap_sharpness: float = 4.0
    # --- 6. commit probability. TWO gains, because they play two different roles and were validated
    #        separately: `commit_gain` scales the READOUT collapse probability in graph mode (validated at
    #        3.0), `select_gain` scales ACTION selection (validated at 1.6). The partial implementations
    #        both called this "gain", which hid the distinction.
    commit_gain: float = 3.0
    select_gain: float = 1.6
    # --- 7. learning: leaky (revisable) vs accumulate (sticky) ---
    update: str = "leaky"          # "leaky" | "accumulate"
    learn_rate: float = 0.1        # leaky mode
    ltd_gain: float = 1.0          # >1 => contradiction bites harder than confirmation
    prior_k: float = 2.0           # accumulate mode: sum / (count + k), a pessimistic prior
    seed: Optional[int] = None


class CoherenceGatedPrimitive:
    """All seven mechanisms. Graph mode and explicit mode share one store."""

    def __init__(self, n_units: int = 0, params: Optional[CGLParams] = None):
        self.n = n_units
        self.p = params or CGLParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.trace = np.full(max(n_units, 0), self.p.trace_floor, dtype=float)
        self.bonded = np.zeros((max(n_units, 0),) * 2, dtype=bool)
        self.last_active = np.full(max(n_units, 0), -np.inf)
        self.t = 0.0
        self.store: dict = {}      # frozenset -> float (leaky) or [sum, count] (accumulate)
        self.index: dict = {}      # unit -> set of keys containing it (keeps partial reactivation sub-linear)
        self.n_commits = 0

    # ---------------- 1 & 2: traces and two-timescale binding ----------------
    def activate(self, active: Sequence[bool]) -> None:
        idx = np.flatnonzero(np.asarray(active, dtype=bool))
        recent = np.flatnonzero((self.t - self.last_active) <= self.p.bind_window)
        for i in idx:
            for j in np.concatenate([idx, recent]):
                if i != j:
                    self.bonded[i, j] = self.bonded[j, i] = True
        self.trace[idx] = 1.0
        self.last_active[idx] = self.t

    def decay(self, dt: float = 1.0) -> None:
        self.t += dt
        f = np.exp(-dt / self.p.tau)
        self.trace = self.p.trace_floor + (self.trace - self.p.trace_floor) * f

    def readable(self) -> np.ndarray:
        return self.trace > self.p.read_threshold

    def fidelity(self) -> np.ndarray:
        F = np.outer(self.trace, self.trace)     # F_ij = t_i * t_j  (Model 6: P_S_i * P_S_j)
        np.fill_diagonal(F, 0.0)
        return F * self.bonded

    # ---------------- 3 & 4: components, including singletons ----------------
    def components(self) -> List[Set[int]]:
        adj = self.fidelity() > self.p.edge_threshold
        singleton_bound = np.sqrt(self.p.edge_threshold)
        seen, comps = np.zeros(self.n, dtype=bool), []
        for s in range(self.n):
            if seen[s]:
                continue
            if not adj[s].any():
                if self.trace[s] > singleton_bound:
                    seen[s] = True
                    comps.append({s})
                continue
            stack, comp = [s], set()
            seen[s] = True
            while stack:
                u = stack.pop(); comp.add(u)
                for v in np.flatnonzero(adj[u]):
                    if not seen[v]:
                        seen[v] = True; stack.append(v)
            comps.append(comp)
        return comps

    # ---------------- 5: value, with partial reactivation ----------------
    def _raw(self, key) -> Optional[float]:
        s = self.store.get(key)
        if s is None:
            return None
        return float(s) if self.p.update == "leaky" else s[0] / (s[1] + self.p.prior_k)

    def value_evidence(self, units: Iterable[int]) -> tuple:
        """(value, has_evidence). `has_evidence` is False when NOTHING is stored for this component and no
        stored component overlaps it -- i.e. the learner is not ignorant-but-guessing-zero, it is genuinely
        ignorant. Callers need that distinction to apply a novelty/optimism bonus only where it is warranted;
        without it a greedy selector cannot tell "known to be worth 0" from "never tried", and deadlocks on
        any action whose true value is 0 (measured, Benchmark 10)."""
        key = frozenset(units)
        if not key:
            return 0.0, False
        exact = self._raw(key)
        if exact is not None:
            return exact, True
        if not self.p.overlap_generalization:
            return 0.0, False
        cand: Set[frozenset] = set()
        for u in key:
            cand |= self.index.get(u, set())
        num = den = 0.0
        for k in cand:
            sim = len(key & k) / len(key | k)
            w = sim ** self.p.overlap_sharpness
            num += w * self._raw(k); den += w
        return (float(num / den), True) if den > 0 else (0.0, False)

    def value(self, units: Iterable[int]) -> float:
        key = frozenset(units)
        if not key:
            return 0.0
        exact = self._raw(key)
        if exact is not None:                    # an exact memory always wins
            return exact
        if not self.p.overlap_generalization:
            return 0.0
        cand: Set[frozenset] = set()
        for u in key:
            cand |= self.index.get(u, set())
        num = den = 0.0
        for k in cand:
            sim = len(key & k) / len(key | k)
            w = sim ** self.p.overlap_sharpness
            num += w * self._raw(k); den += w
        return float(num / den) if den > 0 else 0.0

    # ---------------- 6: commit-probability action selection ----------------
    def choose(self, candidates: Sequence[Iterable[int]]) -> int:
        """One coin per candidate component. No epsilon, no temperature, no schedule."""
        vals = np.array([self.value(c) for c in candidates], dtype=float)
        p = np.clip(self.p.select_gain * vals, 0.0, 1.0)   # negative evidence -> 0 -> never committed
        fired = np.flatnonzero(self.rng.random(len(candidates)) < p)
        if len(fired) == 1:
            return int(fired[0])                            # consolidated: one component collapses
        if len(fired) > 1:
            return int(self.rng.choice(fired))              # several live: pick among them
        return int(self.rng.integers(len(candidates)))       # nothing committed: fully exploratory

    # ---------------- 7: learning, with the active depression arm ----------------
    def learn(self, units: Iterable[int], r: float) -> None:
        key = frozenset(units)
        if not key:
            return
        if key not in self.store:
            self.store[key] = 0.0 if self.p.update == "leaky" else [0.0, 0]
            for u in key:
                self.index.setdefault(u, set()).add(key)
        if self.p.update == "leaky":
            v = self.store[key]
            rate = self.p.learn_rate * (self.p.ltd_gain if r < v else 1.0)
            self.store[key] = v + rate * (r - v)
        else:
            s = self.store[key]; s[0] += r; s[1] += 1

    def reward(self, r: float) -> List[Set[int]]:
        """GRAPH MODE readout: one shared coin per component; committed components consume their traces."""
        if r == 0.0:
            return []                                       # reward is necessary
        fired = []
        for comp in self.components():
            members = sorted(comp)
            p_commit = float(np.clip(self.p.commit_gain * self.trace[members].mean() * abs(r), 0.0, 1.0))
            if self.rng.random() < p_commit:                # ONE coin for the whole component
                self.learn(members, r)
                self.n_commits += 1
                fired.append(comp)
                for m in members:
                    self.trace[m] = self.p.trace_floor      # the readout consumes the trace
        return fired

    def clear_episode(self) -> None:
        """Bonds are per-episode; traces persist."""
        self.bonded[:] = False
