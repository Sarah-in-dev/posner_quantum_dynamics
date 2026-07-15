"""
entanglement_topology.py  —  cheap, honest topology of the cross-synapse
entanglement graph.

WHY THIS EXISTS (and why it is deliberately small)
---------------------------------------------------
`entanglement-topology-measurement` proposes Fiedler-value flux and
persistent-homology *vineyards* as the way to see loops/voids in the
entanglement graph over the coherence window.  The TALON self-comprehension
program built exactly that class of machinery (persistent homology early
warning; a connection-Laplacian / Procrustes sheaf) and KILLED most of it as
vacuous or degenerate — and the one topological object that *earned* was the
cheapest one: connected-component gluing (H0) plus a linear-algebra Betti
count over the coboundary, no ripser / gudhi.

So before Model 6 invests in vineyards, this module computes — essentially for
free, every timestep — the two Betti numbers of the 1-skeleton:

    Betti0 = V - rank(delta)   = number of connected components
    Betti1 = E - rank(delta)   = number of independent cycles (loops)

where delta is the (unsigned) graph coboundary / incidence operator.  For a
graph these collapse to the exact combinatorial identities

    Betti0 = c                    (components)
    Betti1 = E - V + c            (cycle rank)

so we compute V, E, c directly by union-find (matching
`NetworkEntanglementTracker._find_all_clusters` bond-for-bond) and OPTIONALLY
cross-check against rank(delta^T delta) via numpy — the same self-test TALON's
betti_instrument uses to prove the combinatorial count equals ker(delta).

Betti1 is the quantity `entanglement-topology-measurement` actually wants
(loops = closed entanglement paths that scalar and spectral metrics miss).
If Betti1 == 0 the graph is a FOREST and every inconsistency / partition is
single-edge localizable — "armed but not firing," in TALON's phrase.  It flips
on the instant the cross-synapse graph grows a genuine cycle.  That is the
signal that tells you whether the expensive persistence machinery has anything
to measure yet.

The component-size list is returned too: its distribution over a drive x
damping sweep is the SOC (self-organized-criticality) avalanche signature —
a power law there is the drive-independent-attractor evidence, and it is a
Betti0 statistic, not something you need vineyards for.

DISCIPLINE
----------
- Edges are built with the SAME Werner threshold as _find_all_clusters
  (intra bonds by bare existence; cross bonds iff fidelity > WERNER bound).
- The graph is treated as SIMPLE: a node pair is one edge even if it appears
  in more than one bond dict, so clique-fill does not fake extra loops.
- This measures the 1-skeleton only. It does NOT claim to see voids (H2+) or
  the persistence/lifetime of a loop — that is the vineyard question, held
  until Betti1 > 0 AND the components are stable over the window (F1/F7).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class BettiReport:
    """One timestep's topology of the Werner-thresholded entanglement graph."""
    n_nodes: int                       # V — dimers participating in >= 1 counted edge
    n_edges: int                       # E — simple (deduped) edges
    betti0: int                        # number of connected components  (== len(clusters))
    betti1: int                        # cycle rank  E - V + c  (independent loops)
    component_sizes: List[int]         # descending; the SOC avalanche distribution
    cycles_per_component: List[int]    # Betti1 localized to each component (same order)
    rank_delta: Optional[int] = None   # set only when the numpy cross-check runs
    crosscheck_ok: Optional[bool] = None

    def as_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "betti0": self.betti0,
            "betti1": self.betti1,
            "component_sizes": self.component_sizes,
            "cycles_per_component": self.cycles_per_component,
            "rank_delta": self.rank_delta,
            "crosscheck_ok": self.crosscheck_ok,
        }


def _normalized_edges(
    node_ids: Set,
    intra_bonds: Dict[Tuple, float],
    cross_bonds: Dict[Tuple, float],
    werner_bound: float,
) -> Set[Tuple]:
    """Simple undirected edge set, Werner-thresholded, matching _find_all_clusters."""
    edges: Set[Tuple] = set()

    def add(a, b):
        if a in node_ids and b in node_ids and a != b:
            edges.add((a, b) if a <= b else (b, a))

    for (i, j) in intra_bonds:            # intra: by bare existence
        add(i, j)
    for (i, j), fidelity in cross_bonds.items():   # cross: Werner bound
        if fidelity > werner_bound:
            add(i, j)
    return edges


def compute_betti(
    all_dimers: List[dict],
    intra_bonds: Dict[Tuple, float],
    cross_bonds: Dict[Tuple, float],
    werner_bound: float = 0.5,
    crosscheck: bool = False,
) -> BettiReport:
    """
    Pure function: (dimers, bond dicts) -> BettiReport.

    Mirrors NetworkEntanglementTracker._find_all_clusters exactly for node/edge
    inclusion, so Betti0 == len(_find_all_clusters()). Only dimers that
    participate in >= 1 counted edge are nodes (unbonded dimers are omitted
    singletons, as in the tracker).
    """
    universe = {d["global_id"] for d in all_dimers}

    # First pass: which dimers actually carry a counted edge -> node set.
    edges_all = _normalized_edges(universe, intra_bonds, cross_bonds, werner_bound)
    node_ids: Set = set()
    for a, b in edges_all:
        node_ids.add(a)
        node_ids.add(b)

    V = len(node_ids)
    E = len(edges_all)

    if V == 0:
        return BettiReport(0, 0, 0, 0, [], [],
                           rank_delta=0 if crosscheck else None,
                           crosscheck_ok=True if crosscheck else None)

    # Union-find for components (same shape as the tracker's).
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in edges_all:
        union(a, b)

    # Group nodes and edges by component root.
    comp_nodes: Dict = {}
    for n in node_ids:
        comp_nodes.setdefault(find(n), set()).add(n)
    comp_edge_count: Dict = {}
    for a, b in edges_all:
        comp_edge_count[find(a)] = comp_edge_count.get(find(a), 0) + 1

    c = len(comp_nodes)
    betti0 = c
    betti1 = E - V + c   # global cycle rank

    # Localize loops to each component: b1_comp = e_comp - v_comp + 1
    roots = sorted(comp_nodes, key=lambda r: len(comp_nodes[r]), reverse=True)
    component_sizes = [len(comp_nodes[r]) for r in roots]
    cycles_per_component = [
        comp_edge_count.get(r, 0) - len(comp_nodes[r]) + 1 for r in roots
    ]

    report = BettiReport(
        n_nodes=V,
        n_edges=E,
        betti0=betti0,
        betti1=betti1,
        component_sizes=component_sizes,
        cycles_per_component=cycles_per_component,
    )

    if crosscheck:
        report.rank_delta, report.crosscheck_ok = _coboundary_crosscheck(
            node_ids, edges_all, betti0, betti1, V, E
        )
    return report


def compute_synapse_quotient_betti(
    all_dimers: List[dict],
    cross_bonds: Dict[Tuple, float],
    werner_bound: float = 0.5,
    crosscheck: bool = False,
) -> BettiReport:
    """
    Topology of the SYNAPSE-quotient graph: nodes = synapses, edge between two
    synapses iff any cross-synapse bond between their dimers clears the Werner
    bound.  THIS is the lens `entanglement-topology-measurement` actually wants:
    a Betti1 > 0 here is a closed entanglement path ACROSS synapses (a genuine
    multi-synapse loop), not intra-spine clique-fill.  The raw whole-graph
    compute_betti() is dominated by within-spine dense blobs (one component per
    spine, huge trivial cycle rank) and must NOT be read as the entanglement
    topology of the network.

    Requires each dimer dict to carry 'global_id' and 'synapse_idx'.
    """
    gid_to_syn = {d["global_id"]: d.get("synapse_idx") for d in all_dimers}

    syn_nodes: Set = set()
    syn_edges: Set[Tuple] = set()
    for (i, j), fidelity in cross_bonds.items():
        if fidelity <= werner_bound:
            continue
        si, sj = gid_to_syn.get(i), gid_to_syn.get(j)
        if si is None or sj is None or si == sj:
            continue
        syn_nodes.add(si)
        syn_nodes.add(sj)
        syn_edges.add((si, sj) if si <= sj else (sj, si))

    # Reuse the same combinatorial+coboundary machinery via a synthetic graph:
    # one "dimer" per synapse node, cross edges as intra bonds.
    fake_dimers = [{"global_id": s} for s in syn_nodes]
    fake_intra = {e: 1.0 for e in syn_edges}
    return compute_betti(fake_dimers, fake_intra, {}, werner_bound=0.0,
                         crosscheck=crosscheck)


def _coboundary_crosscheck(node_ids, edges, betti0, betti1, V, E):
    """
    TALON-style proof that the combinatorial count equals the linear-algebra
    count: build the unsigned incidence matrix delta (E x V), rank via SVD,
    assert Betti0 == V - rank and Betti1 == E - rank.  numpy-only, optional.
    """
    try:
        import numpy as np
    except Exception:
        return None, None
    if E == 0:
        return 0, (betti0 == V and betti1 == 0)
    index = {n: k for k, n in enumerate(sorted(node_ids))}
    delta = np.zeros((E, V), dtype=float)
    for row, (a, b) in enumerate(sorted(edges)):
        delta[row, index[a]] = 1.0
        delta[row, index[b]] = -1.0
    rank = int(np.linalg.matrix_rank(delta))
    ok = (betti0 == V - rank) and (betti1 == E - rank)
    return rank, ok


# ---------------------------------------------------------------------------
# Method shim — paste onto NetworkEntanglementTracker, or call the pure
# function directly. One line in step()/get_metrics wires it in:
#
#     topo = self.compute_entanglement_topology(crosscheck=False)
#     metrics["betti0"] = topo.betti0
#     metrics["betti1"] = topo.betti1
#     metrics["component_sizes"] = topo.component_sizes
#
# def compute_entanglement_topology(self, crosscheck: bool = False) -> BettiReport:
#     return compute_betti(
#         self.all_dimers,
#         self.intra_synapse_bonds_cache,
#         self.cross_synapse_bonds,
#         werner_bound=self.WERNER_ENTANGLEMENT_BOUND,
#         crosscheck=crosscheck,
#     )
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Self-test: the combinatorial count must equal the coboundary rank.
    # Two disjoint triangles {0,1,2} and {3,4,5}: c=2, each triangle has 1 loop.
    dimers = [{"global_id": i} for i in range(6)]
    intra = {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.9,
             (3, 4): 0.9, (4, 5): 0.9, (3, 5): 0.9}
    cross = {}
    r = compute_betti(dimers, intra, cross, crosscheck=True)
    assert r.betti0 == 2, r.betti0
    assert r.betti1 == 2, r.betti1           # one independent cycle per triangle
    assert r.component_sizes == [3, 3], r.component_sizes
    assert r.cycles_per_component == [1, 1], r.cycles_per_component
    assert r.crosscheck_ok, r.as_dict()

    # A forest: path 0-1-2 plus edge 3-4. c=2, no loops -> "armed but not firing".
    r2 = compute_betti(
        [{"global_id": i} for i in range(5)],
        {(0, 1): 0.9, (1, 2): 0.9, (3, 4): 0.9}, {}, crosscheck=True,
    )
    assert r2.betti0 == 2 and r2.betti1 == 0, r2.as_dict()
    assert r2.crosscheck_ok, r2.as_dict()

    # Werner threshold respected: a weak cross bond (F<=0.5) is NOT an edge.
    r3 = compute_betti(
        [{"global_id": i} for i in range(3)],
        {(0, 1): 0.9}, {(1, 2): 0.4}, werner_bound=0.5, crosscheck=True,
    )
    assert r3.n_edges == 1 and r3.betti0 == 1 and r3.n_nodes == 2, r3.as_dict()

    print("entanglement_topology self-test OK:", r.as_dict())
