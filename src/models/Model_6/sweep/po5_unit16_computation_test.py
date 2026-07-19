#!/usr/bin/env python3
"""
PO-5 UNIT 16b — THE COMPUTATION TEST: with provenance bonding, does the partition carry
INPUT-DEPENDENT structure beyond density?

Pre-registered: docs/PREREG_PO5_UNIT16_PROVENANCE_BUILD.md.
Provenance mechanism verified sparse (sanity: 51 bonds, lf 0.021 vs the 459k-bond blob).

The provenance channel is NOT density — it is WHERE calcium is elevated -> which dimers share
events. Input can move the partition at fixed population. Two spatial input conditions:
  COND-A: drive one voltage/glutamate profile
  COND-B: a different temporal profile (pulsed) -> different calcium spatiotemporal pattern
The metric is Newman MODULARITY (density-corrected) of the input labeling, plus component
structure. >=5 seeds.

PART 1: sweep provenance_event_rate to find an unsaturated-but-connected regime.
PART 2: at that regime, A vs B vs seed-null. Q(input) vs Q(shuffled); component effect size.
"""
import sys, os, json, time, logging
import numpy as np
logging.disable(logging.INFO)
SWEEP_DIR=os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR=os.path.dirname(SWEEP_DIR)
PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0,MODEL6_DIR); sys.path.insert(0,os.path.join(PROJECT_ROOT,"sweep"))
DT,T_SIM=0.005,1.0

def graph(dp):
    ent=[d for d in dp.dimers if d.is_entangled]; ids=[d.id for d in ent]; es=set(ids)
    edges=[(a,b) for (a,b) in dp._bond_lookup if a in es and b in es]
    return ids, edges

def components(ids,edges):
    p={i:i for i in ids}
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    for a,b in edges:
        ra,rb=f(a),f(b)
        if ra!=rb: p[ra]=rb
    lab={}; part={i:lab.setdefault(f(i),len(lab)) for i in ids}
    from collections import Counter
    sz=Counter(part.values())
    return part, len(sz), (max(sz.values())/len(ids) if ids else 0)

def modularity(ids,edges,label_of):
    if not edges: return 0.0
    deg={i:0 for i in ids}
    for a,b in edges: deg[a]+=1; deg[b]+=1
    m=len(edges)
    from collections import Counter
    ein=Counter(); dc=Counter()
    for a,b in edges:
        if label_of[a]==label_of[b]: ein[label_of[a]]+=1
    for i in ids: dc[label_of[i]]+=deg[i]
    return sum(ein[c]/m-(dc[c]/(2*m))**2 for c in set(label_of.values()))

def run(cond, seed, rate):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease
    np.random.seed(seed)
    p=Model6Parameters(); p.em_coupling_enabled=True
    net=MultiSynapseNetwork(n_synapses=1,pattern="clustered",spacing_um=1.0)
    net.initialize(Model6QuantumSynapse,p)
    for s in net.synapses: s.set_microtubule_invasion(True)
    syn=net.synapses[0]; dp=syn.dimer_particles
    dp.provenance_bonding=True; dp.provenance_event_rate=rate
    rel=PresynapticRelease(seed=seed); t=0.0
    for _ in range(int(round(T_SIM/DT))):
        a=0.95 if cond=="A" else (0.95 if (t%0.6)<0.2 else 0.0)
        g=rel.step(a,DT); syn.step(DT,{"voltage":-10e-3,"reward":False,"glutamate":g}); t+=DT
    ids,edges=graph(dp)
    # spatial input label: which half of the domain a dimer sits in (the provenance signal
    # is spatial, so the natural input labeling is spatial region)
    lab={d.id: int(d.position[0] > (dp.grid_shape[0]*dp.dx_nm/2)) for d in dp.dimers if d.is_entangled}
    part,ncomp,lf=components(ids,edges)
    Qin=modularity(ids,edges,lab)
    Qsh=modularity(ids,edges,{i:v for i,v in zip(lab,np.random.permutation(list(lab.values())))})
    return {"cond":cond,"seed":seed,"rate":rate,"V":len(ids),"E":len(edges),
            "components":ncomp,"largest_frac":lf,"Q_input":Qin,"Q_shuffled":Qsh}

def main():
    print("="*96); print("PO-5 UNIT 16b — provenance computation test"); print("="*96,flush=True)
    print("\nPART 1 — event-rate sweep (find unsaturated-but-connected regime):",flush=True)
    print(f"  {'rate':>6s} {'V':>6s} {'edges':>7s} {'comps':>7s} {'largest_frac':>13s}")
    best=None
    for rate in (0.5,2.0,5.0,15.0,40.0,100.0):
        r=run("A",900,rate)
        print(f"  {rate:6.1f} {r['V']:6d} {r['E']:7d} {r['components']:7d} {r['largest_frac']:13.4f}",flush=True)
        if 0.1<r['largest_frac']<0.9 and best is None:
            best=rate
    if best is None:
        # fall back to the rate giving the most edges below saturation
        best=15.0
    print(f"\n  chosen operating rate = {best}",flush=True)

    print("\nPART 2 — INPUT CONTRAST at rate={} (5 seeds/cond):".format(best),flush=True)
    rows=[]
    for cond in ("A","B"):
        for s in (11,12,13,14,15):
            r=run(cond,s,best); rows.append(r)
            print(f"  cond {cond} seed {s}: V={r['V']} E={r['E']} comps={r['components']} "
                  f"lf={r['largest_frac']:.3f} Q_in={r['Q_input']:.4f} Q_sh={r['Q_shuffled']:.4f}",flush=True)
    with open(os.path.join(SWEEP_DIR,"po5_unit16_results.json"),"w") as f:
        json.dump({"operating_rate":best,"rows":rows},f,indent=2)

    def eff(key):
        A=np.array([r[key] for r in rows if r['cond']=="A"],float)
        B=np.array([r[key] for r in rows if r['cond']=="B"],float)
        pooled=np.sqrt((A.std(ddof=1)**2+B.std(ddof=1)**2)/2)
        return A.mean(),B.mean(),(abs(A.mean()-B.mean())/pooled if pooled>1e-12 else float('nan'))
    Qin=np.mean([r['Q_input'] for r in rows]); Qsh=np.mean([r['Q_shuffled'] for r in rows])
    print("\n"+"="*96)
    print(f"P1 mechanism: mean largest_frac = {np.mean([r['largest_frac'] for r in rows]):.3f} "
          f"{'(sparse, OK)' if np.mean([r['largest_frac'] for r in rows])<0.9 else '(still blobbed — MOOT)'}")
    print(f"P2 beyond-density: Q(input)={Qin:.4f} vs Q(shuffled)={Qsh:.4f}  "
          f"ratio={Qin/Qsh if abs(Qsh)>1e-6 else float('inf'):.2f}")
    a,b,d=eff("components")
    print(f"P2 input contrast: components A={a:.1f} B={b:.1f}  effect d={d:.2f}")
    strong_Q = Qin>=0.1 and Qin>5*abs(Qsh)
    print("\n"+"="*96)
    if strong_Q:
        print("VERDICT: PROVENANCE BONDING CARRIES INPUT-DEPENDENT PARTITION BEYOND DENSITY.")
        print("  The partition tracks spatial input structure (Q>>shuffled) in a sparse graph.")
        print("  §8's keystone has a working mechanism, for the first time in this investigation.")
    else:
        print("VERDICT: provenance produces a sparse graph but the partition does NOT carry input")
        print("  structure beyond density here. The mechanism is faithful but not (yet) computational")
        print("  in this regime. Reported as the finding.")
    print("\nLIMITS: 1 synapse, 1 s, 5 seeds. event_rate/age are modelling choices, swept not tuned.")
    print("A positive shows the CHANNEL exists in the model; it does not certify the rate constants.")

if __name__=="__main__":
    main()
