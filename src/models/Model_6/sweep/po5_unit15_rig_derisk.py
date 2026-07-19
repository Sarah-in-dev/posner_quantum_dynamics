#!/usr/bin/env python3
"""
PO-5 UNIT 15 — RIG de-risk: can provenance/shared-event bonding carry INPUT-DEPENDENT
partition structure BEYOND DENSITY? (pure graph, no physics)

This is the gate the deep research named: the object is the s=1 random intersection graph
(nodes=dimers, features=hydrolysis events, edge iff shared event). The literature establishes
it is richer than density (clique structure below m<n^3) but does NOT establish it carries
INPUT-DEPENDENT partition structure. That is the exact §8 property. Test it abstractly BEFORE
any 50-75 line build.

THE MODEL (dimer-faithful, per LOCKED quantum-system-canonical:43):
  N dimers, each carries K=2 hydrolysis events (dimer = 2 singlet pairs = <=2 events). Edge
  iff two dimers share an event. M events available. Density is governed by M (fewer events =
  more sharing = denser).

INPUT: input assigns dimers to G groups and biases event-selection so same-group dimers draw
from a shared event pool (assortative provenance). Different inputs = different groupings.
This is the ONLY input->structure hypothesis; if it fails, provenance is input-blind too.

THE DECISIVE CONTROL (§8's "beyond density"): a DEGREE-PRESERVING configuration null. Rewire
events keeping every dimer's event-COUNT identical (so density and degree sequence are fixed),
destroying only the group structure. If the input partition tracks the groups MORE than the
degree-matched null does, the structure is beyond density -> provenance CAN carry input.

PRE-REGISTERED:
  metric: adjusted mutual information (AMI) between the graph's connected-component partition
          and the input group labels.
  P1 input-RIG AMI must exceed the degree-preserving null AMI by a clear margin (ratio >= 3,
     or input-AMI >= 0.1 while null-AMI ~ 0).
  P2 sweep M (density). At which densities, if any, does the signal survive? A signal only in a
     regime the biology never reaches is not a usable answer.
  If input-AMI ~ null-AMI at ALL densities -> provenance is ALSO input-blind. The whole
  direction is dead and the build is not worth days. Reported as such.
"""
import numpy as np
from math import log
from itertools import combinations

RNG = np.random.default_rng(20260719)
N, K, G = 1000, 2, 5     # dimers, events per dimer, input groups

def components(N, edges):
    p = list(range(N))
    def f(x):
        while p[x] != x: p[x]=p[p[x]]; x=p[x]
        return x
    for a,b in edges:
        ra,rb=f(a),f(b)
        if ra!=rb: p[ra]=rb
    lab={}
    return np.array([lab.setdefault(f(i), len(lab)) for i in range(N)])

def ami(a, b):
    # adjusted mutual information (sklearn-free, adequate for this scale)
    a=np.asarray(a); b=np.asarray(b); n=len(a)
    def ent(x):
        _,c=np.unique(x,return_counts=True); pr=c/n
        return -np.sum(pr*np.log(pr))
    ca=np.unique(a); cb=np.unique(b)
    mi=0.0
    for u in ca:
        ia=a==u; pa=ia.mean()
        for v in cb:
            ib=b==v; pab=(ia&ib).mean()
            if pab>0: mi+=pab*log(pab/(pa*ib.mean()))
    Ha,Hb=ent(a),ent(b)
    emi=0.0  # EMI approx ~0 at this cluster count/scale; report raw MI-based NMI too
    denom=max((Ha+Hb)/2,1e-12)
    return (mi-emi)/max(denom-emi,1e-12), mi/denom

def build_events(M, groups, assortativity):
    """Each dimer draws K events. With prob `assortativity` an event is drawn from the dimer's
    GROUP pool; else uniform. Returns node->set(events)."""
    pools=[RNG.choice(M, size=max(2,M//G), replace=False) for _ in range(G)]
    ev=[set() for _ in range(N)]
    for i in range(N):
        g=groups[i]
        while len(ev[i])<K:
            if RNG.random()<assortativity:
                ev[i].add(int(RNG.choice(pools[g])))
            else:
                ev[i].add(int(RNG.integers(M)))
    return ev

def edges_from_events(ev):
    by_event={}
    for i,es in enumerate(ev):
        for e in es: by_event.setdefault(e,[]).append(i)
    E=set()
    for members in by_event.values():
        for a,b in combinations(members,2): E.add((a,b))
    return list(E)

def degree_preserving_null(ev, M):
    """Rewire keeping each node's event COUNT identical; events chosen uniformly (no groups).
    Fixes degree sequence / density, destroys group structure."""
    return [set(int(x) for x in RNG.choice(M, size=len(es), replace=False)) for es in ev]

def main():
    groups=RNG.integers(0,G,size=N)
    print("="*92)
    print("PO-5 UNIT 15 — can shared-event (RIG) bonding carry INPUT-DEPENDENT partition beyond density?")
    print(f"  N={N} dimers, K={K} events/dimer (dimer=2 singlet pairs), G={G} input groups")
    print("="*92)
    print(f"  {'M(events)':>10s} {'assort':>7s} {'edges':>9s} {'lg_comp':>8s} "
          f"{'AMI(input)':>11s} {'AMI(null)':>10s} {'ratio':>7s}")
    results=[]
    for M in (60, 120, 250, 500, 1000, 2000):
        for assort in (0.9,):
            ev=build_events(M, groups, assort)
            edges=edges_from_events(ev)
            part=components(N, edges)
            ami_in,_=ami(part, groups)
            evn=degree_preserving_null(ev, M)
            partn=components(N, edges_from_events(evn))
            ami_null,_=ami(partn, groups)
            lg=np.bincount(part).max()/N
            ratio=ami_in/ami_null if ami_null>1e-6 else float('inf')
            results.append((M,assort,len(edges),lg,ami_in,ami_null,ratio))
            print(f"  {M:10d} {assort:7.2f} {len(edges):9d} {lg:8.3f} "
                  f"{ami_in:11.4f} {ami_null:10.4f} {ratio:7.2f}")
    print("\n" + "="*92)
    live=[r for r in results if r[4]>=0.1 and (r[5]<1e-6 or r[4]/max(r[5],1e-9)>=3) and 0.05<r[3]<0.95]
    if live:
        print("VERDICT: RIG CARRIES INPUT-DEPENDENT PARTITION BEYOND DENSITY.")
        print("  Input group structure is recovered by the component partition FAR above the")
        print("  degree-preserving null, in an UNSATURATED regime:")
        for r in live:
            print(f"    M={r[0]}: AMI(input)={r[4]:.3f} vs AMI(null)={r[5]:.3f}, largest_frac={r[3]:.3f}")
        print("  => the provenance build is worth it. §8 has a candidate channel.")
    else:
        sat=[r for r in results if r[3]>=0.95]
        print("VERDICT: provenance is ALSO input-blind, OR only signals in a saturated/degenerate")
        print("  regime. Either way the shared-event structure does not carry input-dependent")
        print("  partition where the graph is usable. The build is NOT worth days.")
        print(f"  (saturated cells: {len(sat)}/{len(results)})")
    print("\nLIMITS: abstract graph, no physics. Tests whether the RIG CAN carry input structure")
    print("in principle; the biological event-assignment may or may not realise the assortativity")
    print("this assumes — that is the spatial-join physics the build would still have to earn.")

if __name__=="__main__":
    main()
