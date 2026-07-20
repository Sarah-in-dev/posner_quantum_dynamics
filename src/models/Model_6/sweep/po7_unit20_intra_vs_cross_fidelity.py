#!/usr/bin/env python3
"""
PO-7 UNIT 20 — verify the linchpin of the Unit 18 domain result: intra bonds are near-lossless.

Unit 18 found correlated domains of ~468 dimers (~1.35 synapses), NOT the ~2-5 the p~0.56
chain-picture predicted. The stated cause: intra-synapse bonds run at F->1 (p->1, w->0), so a
whole synapse is one lossless correlated core, and cross-bridges (p~0.6) extend it partway into
neighbours. This confirms that directly. FREE-RUNNING (no seed). 12 s (past ignition).

⚠ TIME CAVEAT: F_intra ~ 1 here because P_S has not decayed at 12 s (T_singlet = 216 s). This is
the WRITE-TIME graph. At READOUT time (dopamine, tens-to-hundreds of s later) P_S decays, intra F
falls toward 0.5, and domains fragment — the advisor's small domains and this large one are the
same graph at different times. The readout-time measurement (dopamine at realistic delay +
corrected release rate) is the real computational output and is the next experiment.
"""
import sys, logging
import numpy as np
logging.disable(logging.INFO)
sys.path.insert(0,'src/models/Model_6'); sys.path.insert(0,'sweep')
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease
p=Model6Parameters(); p.em_coupling_enabled=True; p.multi_synapse_enabled=True; p.environment.fraction_P31=1.0
net=MultiSynapseNetwork(n_synapses=7,pattern="linear",spacing_um=1.0)
net.initialize(Model6QuantumSynapse,p)
for s in net.synapses:
    s.set_microtubule_invasion(True); s.dimer_particles.provenance_bonding=True; s.dimer_particles.spin_resolved=True
net.disable_auto_commitment=True
tr=net.entanglement_tracker; rel=PresynapticRelease(None)
for i in range(int(12.0/0.005)):
    g=rel.step(0.95,0.005)
    net.step(0.005,{"voltage":-40e-3,"reward":False,"glutamate":g})
ns=set(d['global_id'] for d in tr.all_dimers)
intra=np.array([f for k,f in tr.intra_synapse_bonds_cache.items() if k[0] in ns and k[1] in ns])
cross=np.array([f for k,f in tr.cross_synapse_bonds.items() if f>0.5 and k[0][0]!=k[1][0]])
def pc(F): return (4*F-1)/3
if len(intra):
    print(f"INTRA n={len(intra)}: F med={np.median(intra):.4f} max={intra.max():.4f} min={intra.min():.4f} | p_med={pc(np.median(intra)):.4f} | frac F>0.9={100*(intra>0.9).mean():.0f}% | w_med={-np.log(max(pc(np.median(intra)),1e-9)):.4f}")
if len(cross):
    print(f"CROSS n={len(cross)}: F med={np.median(cross):.4f} max={cross.max():.4f} | p_med={pc(np.median(cross)):.4f}")
