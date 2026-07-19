#!/usr/bin/env python3
"""
PO-5 UNIT 15 (v2) — RIG de-risk: does shared-event bonding carry INPUT-DEPENDENT partition
BEYOND DENSITY? Pure graph, no physics.

v1 leaned negative but had TWO design flaws that could manufacture a false negative (PO-5
caught them before reporting):
  (1) group event-pools OVERLAPPED (each group independently drew M//G of M events), washing
      out assortativity. FIX: DISJOINT pools — the event space is partitioned into G blocks.
  (2) the AMI metric was NOT chance-adjusted, so it inflated at high fragmentation (many
      singleton components spuriously "agree" with any labeling). FIX: use NEWMAN MODULARITY
      of the input labeling as the primary metric — it subtracts the degree-expected
      (configuration-model) edge probability BY CONSTRUCTION, so Q>0 IS "structure beyond
      density". No separate null run needed; density/degree is controlled analytically.

Q = (1/2m) Σ_ij [A_ij − k_i k_j/2m] δ(g_i,g_j)   over the input group labels g.
  Q ~ 0  -> the graph carries no input-group structure beyond what degree/density forces.
  Q >> 0 -> it does. This is the §8 "beyond density" test, exactly.

Dimer-faithful: K=2 events/dimer (LOCKED: Ca6(PO4)4 dimer = 2 singlet pairs).
"""
import numpy as np
from itertools import combinations
RNG=np.random.default_rng(20260719)
N,K,G=1000,2,5

def edges_from_events(ev):
    by={}
    for i,es in enumerate(ev):
        for e in es: by.setdefault(e,[]).append(i)
    E=set()
    for m in by.values():
        for a,b in combinations(m,2): E.add((a,b) if a<b else (b,a))
    return list(E)

def components(N,edges):
    p=list(range(N))
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    for a,b in edges:
        ra,rb=f(a),f(b)
        if ra!=rb: p[ra]=rb
    lab={}; return np.array([lab.setdefault(f(i),len(lab)) for i in range(N)])

def modularity(N,edges,labels):
    if not edges: return 0.0
    deg=np.zeros(N)
    for a,b in edges: deg[a]+=1; deg[b]+=1
    m=len(edges); Q=0.0
    # sum over edges within a community minus expected
    import collections
    ein=collections.Counter()
    for a,b in edges:
        if labels[a]==labels[b]: ein[labels[a]]+=1
    degsum=collections.Counter()
    for i in range(N): degsum[labels[i]]+=deg[i]
    for c in set(labels):
        Q += ein[c]/m - (degsum[c]/(2*m))**2
    return Q

def build(M,groups,assort,disjoint=True):
    if disjoint:
        blocks=np.array_split(RNG.permutation(M),G)     # DISJOINT pools
        pools=[b for b in blocks]
    else:
        pools=[RNG.choice(M,size=max(2,M//G),replace=False) for _ in range(G)]
    ev=[set() for _ in range(N)]
    for i in range(N):
        g=groups[i]
        while len(ev[i])<K:
            ev[i].add(int(RNG.choice(pools[g])) if RNG.random()<assort else int(RNG.integers(M)))
    return ev

def main():
    groups=RNG.integers(0,G,size=N)
    print("="*96)
    print("PO-5 UNIT 15 v2 — input-dependent partition beyond density? (disjoint pools, modularity)")
    print(f"  N={N} K={K} G={G}   metric: Newman modularity Q of the INPUT labels (Q>0 = beyond density)")
    print("="*96)
    print(f"  {'M':>6s} {'assort':>7s} {'edges':>9s} {'lg_frac':>8s} {'Q(input)':>9s} {'Q(shuffled)':>12s}")
    rows=[]
    for M in (250,500,1000,2000,4000,8000):
        assort=0.9
        ev=build(M,groups,assort,disjoint=True)
        edges=edges_from_events(ev)
        part=components(N,edges)
        Qin=modularity(N,edges,groups)
        Qshuf=modularity(N,edges,RNG.permutation(groups))   # labels shuffled = chance level
        lg=np.bincount(part).max()/N if len(edges) else 0
        rows.append((M,len(edges),lg,Qin,Qshuf))
        print(f"  {M:6d} {assort:7.2f} {len(edges):9d} {lg:8.3f} {Qin:9.4f} {Qshuf:12.4f}")
    print("\n"+"="*96)
    # a usable channel: Q(input) clearly > Q(shuffled) AND graph not saturated
    live=[r for r in rows if r[3]>=0.10 and r[3]>5*abs(r[4]) and 0.05<r[2]<0.98]
    if live:
        print("VERDICT: RIG CARRIES INPUT-DEPENDENT PARTITION BEYOND DENSITY.")
        for r in live:
            print(f"    M={r[0]}: Q(input)={r[3]:.3f} vs Q(shuffled)={r[4]:.3f}, largest_frac={r[2]:.3f}")
        print("  => provenance CAN encode input; the build is worth it, IF the biology realises")
        print("     assortative event-sharing (the spatial-join physics the build must still earn).")
    else:
        print("VERDICT: no input-dependent partition beyond density in any usable regime.")
        print("  Even with disjoint pools and a density-corrected metric, the shared-event")
        print("  partition does not track input above chance where the graph is unsaturated.")
        print("  Provenance is input-blind too; the build is NOT worth days.")
    print("\nLIMITS: abstract; assumes assortative event-sharing. Whether births actually inherit")
    print("input-correlated events is the spatial-join physics a build would have to establish.")

if __name__=="__main__":
    main()
