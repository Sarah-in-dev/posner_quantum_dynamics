"""
The ACTUAL primitive: eligibility as an evolving GRAPH, read out by per-component joint collapse.

WHY THIS FILE EXISTS. A first extraction (`cgl.py`) reduced eligibility to a per-unit scalar trace and
committed each unit with an INDEPENDENT draw. That is TD(lambda) — and benchmarking it against a
classical eligibility-trace learner produced a tie, which was uninformative because both arms were the same
architecture. The program says explicitly what the primitive is NOT:

    "In RL: eligibility trace is a decaying scalar attached to each weight.
     In our system: eligibility trace is the TOPOLOGY of an evolving graph.
     The graph carries information no scalar per weight can: temporal ordering, cluster structure,
     relative timing across inputs. The graph structure at decision time IS the credit assignment."

Two mechanisms were dropped in that first pass, and they are the whole point:

  1. ELIGIBILITY IS THE TOPOLOGY. Nodes activate; EDGES form between temporally correlated nodes; edge
     fidelity decays and an edge stops counting below a threshold. Credit assignment is a READ of the graph,
     not a sum over stored scalars. [Model 6: cross-bond fidelity F = P_S_i * P_S_j * w, edge iff F > 0.5 --
     the Werner separability bound, which is physics, not a tunable knob.]

  2. READOUT IS PER-COMPONENT JOINT COLLAPSE. ONE shared stochastic draw per connected component; every node
     in that component commits TOGETHER. Not an independent draw per node.
     [Model 6: `perform_quantum_measurement` collapses connected components jointly -- one coin per
     component, probability = mean P_S of its members.]

The consequence is the claim with no RL analogue: the number of independent decisions equals the number of
connected components, so a FRAGMENTED graph yields many independent samples (exploration) and a CONSOLIDATED
graph yields one correlated decision (exploitation). Exploration is not a hyperparameter -- it falls out of
what has already been learned.

CLASSICAL BY CONSTRUCTION. One shared coin per component is a common-cause correlation, not entanglement.
This is the (A) architecture; no quantum-advantage claim transfers here.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np


@dataclass
class GraphCGLParams:
    tau: float = 216.0             # trace decay constant
    trace_floor: float = 0.25      # thermal-analogue floor
    edge_threshold: float = 0.5    # an edge counts only if fidelity F_ij > this (Werner analogue)
    bind_window: float = 2.0       # TWO TIMESCALES (this is load-bearing). Bond FORMATION is fast and local
                                   # -- only nodes active within this window bind. Bond PERSISTENCE is long
                                   # (tau, above). Model 6 separates these: dimers bond within a single
                                   # calcium event (ms) but the coherence they carry lasts ~100 s. Collapsing
                                   # them into one constant makes binding indiscriminate: every still-live
                                   # node binds to every other, every trial yields one junk component, and no
                                   # structure is ever learnable. (Found 2026-08-24 by instrumenting a
                                   # training run; the first version declared this parameter and ignored it.)
    commit_gain: float = 3.0
    lr: float = 1.0
    # --- PARTIAL REACTIVATION (pattern completion) ---
    # Without this, stored components are ATOMIC KEYS: {a,b} and {a,c} are as unrelated as {a,b} and {x,y},
    # so a combination never observed has no entry and the readout is at exactly chance (measured, B4).
    # With it, an unfamiliar component is answered by the stored components it OVERLAPS, weighted by
    # similarity -- i.e. a partial input partially reactivates the patterns it resembles. This is the
    # attractor/pattern-completion behaviour the program's framing invokes ("the entanglement topology
    # settling into a stable graph structure ... = attractor in graph space"). An EXACT match always wins,
    # so the conjunction and capacity results are preserved unchanged.
    # HONEST NAMING: mechanically this is similarity-weighted retrieval over stored components. It is a
    # design choice motivated by the attractor framing, not something Model 6 measured.
    overlap_generalization: bool = True
    overlap_sharpness: float = 4.0   # exponent on Jaccard similarity; higher = more nearly exact-match only
    seed: Optional[int] = None


class GraphCoherenceGatedLearner:
    """Nodes carry traces; co-active nodes BIND into edges; readout collapses whole components together."""

    def __init__(self, n_units: int, params: Optional[GraphCGLParams] = None):
        self.n = n_units
        self.p = params or GraphCGLParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.trace = np.full(n_units, self.p.trace_floor, dtype=float)
        self.bonded = np.zeros((n_units, n_units), dtype=bool)   # has this pair co-activated recently
        self.t = 0.0                                             # internal clock
        self.last_active = np.full(n_units, -np.inf)             # last activation time per node
        self.group_w = {}                                        # frozenset(component) -> accumulated weight
        self.n_commits = 0

    # ---- primitive 1: activity lays traces AND binds co-active nodes into edges ----
    def activate(self, active: Sequence[bool]):
        active = np.asarray(active, dtype=bool)
        idx = np.flatnonzero(active)
        # Bind only to nodes active within the SHORT formation window (genuine temporal correlation).
        recent = np.flatnonzero((self.t - self.last_active) <= self.p.bind_window)
        for i in idx:
            for j in np.concatenate([idx, recent]):
                if i != j:
                    self.bonded[i, j] = self.bonded[j, i] = True
        self.trace[active] = 1.0
        self.last_active[active] = self.t

    def decay(self, dt: float = 1.0):
        self.t += dt
        f = np.exp(-dt / self.p.tau)
        self.trace = self.p.trace_floor + (self.trace - self.p.trace_floor) * f

    # ---- the graph that actually counts, right now ----
    def fidelity(self) -> np.ndarray:
        F = np.outer(self.trace, self.trace)          # F_ij = t_i * t_j  (Model 6: P_S_i * P_S_j)
        np.fill_diagonal(F, 0.0)
        return F * self.bonded

    def components(self) -> List[Set[int]]:
        """Connected components of the THRESHOLDED graph — the partition that gets read out.

        A node with NO surviving edges but whose OWN trace is still readable forms a SINGLETON component.
        (Defect found 2026-08-24 by diagnosis: the first version skipped edgeless nodes entirely, so a lone
        active unit could never be credited at all. That is wrong against Model 6, where a single synapse's
        dimer cloud is itself one connected component — "one synapse = one nanodomain = one component". The
        singleton bound is sqrt(edge_threshold): a self-pair would need t*t > threshold, i.e. t > 0.707, which
        is the same Werner floor the pairwise edges use.)"""
        F = self.fidelity()
        adj = F > self.p.edge_threshold
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

    # ---- primitive 2: ONE draw per component; the whole component commits together ----
    def reward(self, r: float) -> List[Set[int]]:
        if r == 0.0:
            return []                                  # reward is necessary
        fired = []
        for comp in self.components():
            members = sorted(comp)
            p_commit = float(np.clip(self.p.commit_gain * self.trace[members].mean() * abs(r), 0.0, 1.0))
            if self.rng.random() < p_commit:           # ONE shared coin for the whole component
                key = frozenset(members)
                self.group_w[key] = self.group_w.get(key, 0.0) + self.p.lr * np.sign(r)
                self.n_commits += 1
                fired.append(comp)
                for m in members:                      # the readout consumes the trace
                    self.trace[m] = self.p.trace_floor
        return fired

    def predict_group(self, members) -> float:
        """Value for a component. Exact memory if we have it; otherwise partial reactivation of overlapping
        memories (weighted by Jaccard similarity), which is what lets a never-seen combination be answered."""
        key = frozenset(int(m) for m in members)
        if not key:
            return 0.0
        if key in self.group_w:
            return float(self.group_w[key])
        if not self.p.overlap_generalization:
            return 0.0
        num = den = 0.0
        for k, v in self.group_w.items():
            inter = len(key & k)
            if inter == 0:
                continue
            sim = inter / len(key | k)
            wgt = sim ** self.p.overlap_sharpness
            num += wgt * v; den += wgt
        return float(num / den) if den > 0 else 0.0

    def best_group(self) -> Optional[frozenset]:
        if not self.group_w:
            return None
        return max(self.group_w.items(), key=lambda kv: kv[1])[0]

    def unit_scores(self) -> np.ndarray:
        """Per-unit credit implied by the committed groups (for comparison with scalar learners)."""
        s = np.zeros(self.n)
        for grp, w in self.group_w.items():
            for m in grp:
                s[m] += w
        return s

    def clear_episode(self):
        """End-of-episode reset of the binding graph (traces persist; bonds are per-episode)."""
        self.bonded[:] = False
