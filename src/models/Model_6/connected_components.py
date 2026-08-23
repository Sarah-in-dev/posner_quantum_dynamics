"""Exact vectorised connected components — a drop-in replacement for the Python union-find loops.

WHY THIS EXISTS (measured, not assumed). Profiling 20 network steps at 3,554 dimers (115.5 s total) showed the
dominant cost was NOT the O(n^2) cross-synapse pair algebra but the two Python union-find passes over the
near-complete INTRA-synapse bond graph (~1.65M edges), run every step:
    multi_synapse_network._find_all_clusters      33.7 s   (134M `find`, 33M `union` calls)
    dimer_particles.find_entangled_clusters       19.1 s   (30M unions)
Union-find is near-linear in theory; the cost here is ~10^8 Python-level function calls, not the algorithm.

WHAT THIS IS. Hook-and-compress label propagation: every node repeatedly adopts the smallest label among its
neighbours, then pointer-jumps to full path compression, until nothing changes. It returns EXACTLY the same
partition as union-find (connected components are unique — there is no approximation, no tolerance, no
sampling); only the inner loops move from Python into numpy/C. Nothing about the physics changes.
"""
import numpy as np


def connected_component_labels(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Labels for an undirected graph on nodes 0..n-1; labels[i] is the minimal index in i's component.

    src/dst are parallel int arrays of edge endpoints. Self-loops and duplicate edges are harmless.
    """
    labels = np.arange(n, dtype=np.int64)
    if n == 0 or len(src) == 0:
        return labels
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    # symmetrise once, and sort once — the adjacency structure is fixed for this call
    u = np.concatenate((src, dst))
    v = np.concatenate((dst, src))
    order = np.argsort(u, kind="stable")
    u_s, v_s = u[order], v[order]
    starts = np.flatnonzero(np.concatenate(([True], u_s[1:] != u_s[:-1])))
    heads = u_s[starts]

    for _ in range(10000):                      # bounded; converges in O(log n) rounds
        cand = np.minimum.reduceat(labels[v_s], starts)   # min neighbour label per node
        new = labels.copy()
        new[heads] = np.minimum(new[heads], cand)
        while True:                              # pointer jumping -> full path compression
            nxt = new[new]
            if np.array_equal(nxt, new):
                break
            new = nxt
        if np.array_equal(new, labels):
            return labels
        labels = new
    return labels


def components_from_pairs(ids, pairs):
    """Group `ids` into components given an iterable of (id_a, id_b) pairs.

    Returns a list of sets of ids, containing ONLY ids that appear in at least one pair — matching the
    semantics of the union-find implementations this replaces (unbonded singletons are omitted).
    """
    index = {gid: k for k, gid in enumerate(ids)}
    a, b = [], []
    bonded = set()
    for id_i, id_j in pairs:
        ri, rj = index.get(id_i), index.get(id_j)
        if ri is None or rj is None:
            continue
        a.append(ri); b.append(rj)
        bonded.add(id_i); bonded.add(id_j)
    if not bonded:
        return []
    labels = connected_component_labels(len(index), np.asarray(a, dtype=np.int64),
                                        np.asarray(b, dtype=np.int64))
    out = {}
    for gid in bonded:
        out.setdefault(int(labels[index[gid]]), set()).add(gid)
    return list(out.values())
