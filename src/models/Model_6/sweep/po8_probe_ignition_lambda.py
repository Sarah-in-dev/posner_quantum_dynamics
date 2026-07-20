import sys, os, time, math
d='src/models/Model_6/sweep'; sys.path.insert(0,'src/models/Model_6'); sys.path.insert(0,d)
import logging; logging.disable(logging.INFO)
import numpy as np
from po7_unit18_correlation_domains import build_weighted_graph, bounded_dijkstra, connected_components
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease
DT=5e-3; VOLT=-40e-3; ACT=0.95
def summ(tr):
    nodes,adj,edges,maxF=build_weighted_graph(tr); V=len(nodes)
    if not V: return (0,0,0.0,0.0,0)
    S=[sum(math.exp(-x) for x in bounded_dijkstra(u,adj).values()) for u in nodes]
    cross=[1 for k,v in tr.cross_synapse_bonds.items() if float(v)>0.5 and k[0][0]!=k[1][0]]
    return (V,len(edges),float(np.mean(S)),maxF,len(cross))
for lam in (5.0,214.0):
    p=Model6Parameters(); p.em_coupling_enabled=True; p.multi_synapse_enabled=True; p.environment.fraction_P31=1.0
    net=MultiSynapseNetwork(n_synapses=7,pattern='linear',spacing_um=1.0,coupling_length_um=lam)
    net.initialize(Model6QuantumSynapse,p)
    for s in net.synapses:
        s.set_microtubule_invasion(True); s.dimer_particles.provenance_bonding=True; s.dimer_particles.spin_resolved=True
    net.disable_auto_commitment=True; tr=net.entanglement_tracker; rel=PresynapticRelease(None)
    peak=0.0; w0=time.time()
    snaps={int(round(t/DT)):t for t in (10.0,15.0,20.0,25.0)}
    for i in range(1,int(round(25.0/DT))+1):
        g=rel.step(ACT,DT); net.step(DT,{'voltage':VOLT,'reward':False,'glutamate':g})
        peak=max(peak,max((float(getattr(s,'_backbone_eta',0.0)) for s in net.synapses),default=0.0))
        if i in snaps:
            V,E,md,mf,nc=summ(tr)
            print(f'lam={lam:5.0f} t={snaps[i]:5.1f}s wall={time.time()-w0:6.1f}s peak_eta={peak:.3f} V={V:4d} E={E:5d} mean_dom={md:7.1f} maxF={mf:.3f} n_cross={nc}',flush=True)
