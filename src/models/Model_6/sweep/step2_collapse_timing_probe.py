#!/usr/bin/env python3
"""
COLLAPSE-TIMING CHECK — does the quantum measurement (perform_quantum_measurement, fired by
_evaluate_coordinated_gate when _coordinated_measurement_performed flips False->True) read the
CLEAN structured partition, or the blob after the plateau's dimer burst?

Steps the reward phase ONE step at a time, logging per step: dimers, cluster-quotient partition,
and whether the collapse fired THIS step. Answers whether Step-2's negative is a fixable timing
artifact (collapse read the blob) or a real downstream washout (collapse read clean, credit lost later).
"""
import sys, os, logging, types
import numpy as np
M6 = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/src/models/Model_6"
REPO_SWEEP = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/sweep"
sys.path.insert(0, M6); sys.path.insert(0, os.path.join(M6,"sweep")); sys.path.insert(0, REPO_SWEEP)
logging.disable(logging.INFO)
for _n in ['model6_core','multi_synapse_network','dimer_particles','analytical_calcium_system',
           'atp_system','ca_triphosphate_complex','quantum_coherence','pH_dynamics','dopamine_system',
           'em_tryptophan_module','em_coupling_module','local_dimer_tubulin_coupling','camkii_module',
           'spine_plasticity_module','photon_emission_module','photon_receiver_module','ddsc_module',
           'vibrational_cascade_module']:
    logging.getLogger(_n).setLevel(logging.ERROR)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease

PER, N_CLUST = 2, 4
N_SYN = PER*N_CLUST
DT = 5e-3
WRITE_S, GAP2_S = 3.0, 1.0
VOLT, ACT, REST = -40e-3, 0.95, -70e-3
ETA_ON = 0.30
CLUSTERS = {c: list(range(c*PER, c*PER+PER)) for c in range(N_CLUST)}
PAIRSETS = {"pair1": ([0,1],[2,3])}

def burst_gated_eta(self):
    for s in self.synapses:
        s.set_backbone_condensation_eta(ETA_ON if getattr(s,"_active_now",False) else 0.0)

def active_synapses(t):
    first, second = PAIRSETS["pair1"]; act=set()
    if t < WRITE_S:
        for c in first: act |= set(CLUSTERS[c])
    lo=WRITE_S+GAP2_S
    if lo <= t < lo+WRITE_S:
        for c in second: act |= set(CLUSTERS[c])
    return act

def build():
    p=Model6Parameters(); p.em_coupling_enabled=True; p.multi_synapse_enabled=True; p.environment.fraction_P31=1.0
    net=MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                            coupling_length_um=5.0, fidelity_length_um=214.0, use_correlated_sampling=True)
    gap_um=3.0; centers=[i*gap_um for i in range(N_CLUST)]
    xs=[centers[c]+0.5*k for c in range(N_CLUST) for k in range(PER)]
    net.positions=np.array([[x,0.0,0.0] for x in xs])
    net.distances=np.sqrt(((net.positions[:,None,:]-net.positions[None,:,:])**2).sum(-1))
    net.coupling_weights=np.exp(-net.distances/5.0); np.fill_diagonal(net.coupling_weights,1.0)
    net.fidelity_weights=np.exp(-net.distances/214.0); np.fill_diagonal(net.fidelity_weights,1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses: s.set_microtubule_invasion(True)
    net.disable_auto_commitment=True
    net._update_backbone_field=types.MethodType(burst_gated_eta, net)
    return net

def dimers(net): return sum(len(s.dimer_particles.dimers) for s in net.synapses)
def quotient(net):
    tr=net.entanglement_tracker; tr.collect_dimers(net.synapses, net.positions)
    gid_syn={d['global_id']:d['synapse_idx'] for d in tr.all_dimers}
    syn_c={i:c for c in range(N_CLUST) for i in CLUSTERS[c]}
    edges=set()
    for (i,j),F in tr.cross_synapse_bonds.items():
        if F>0.5:
            a,b=gid_syn.get(i),gid_syn.get(j)
            if a is not None and b is not None and a!=b:
                ca,cb=syn_c[a],syn_c[b]
                if ca!=cb: edges.add((min(ca,cb),max(ca,cb)))
    return sorted(edges)

net=build(); rels=[PresynapticRelease(None) for _ in range(N_SYN)]
# WRITE
t=0.0
for _ in range(int(round((2*WRITE_S+GAP2_S)/DT))):
    act=active_synapses(t)
    for i,s in enumerate(net.synapses): s._active_now=(i in act)
    per_syn=[{"voltage":VOLT,"glutamate":rels[i].step(ACT,DT)} if i in act else {"voltage":REST,"glutamate":0.0} for i in range(N_SYN)]
    net.step(DT, {"per_synapse":per_syn,"reward":False}); t+=DT
print(f"END-OF-WRITE: dimers={dimers(net)}  partition={quotient(net)}  (structured target = [(0,1),(2,3)])", flush=True)

# REWARD phase, one step at a time, instrumented
print(f"\n{'step':>4} {'t_ms':>6} {'dimers':>7} {'collapse?':>9} {'partition (cluster edges)'}", flush=True)
collapse_step=None; collapse_dimers=None; collapse_part=None
for k in range(60):   # 60 steps = 0.3s reward
    for s in net.synapses: s._active_now=False
    was = bool(getattr(net,'_coordinated_measurement_performed',False))
    per_syn=[{"voltage":REST,"glutamate":0.0,"reward":True,"plateau_potential":True} for _ in range(N_SYN)]
    net.step(DT, {"per_synapse":per_syn,"reward":True})
    now = bool(getattr(net,'_coordinated_measurement_performed',False))
    fired = (now and not was)
    nd=dimers(net); q=quotient(net)
    if fired: collapse_step, collapse_dimers, collapse_part = k, nd, q
    if k < 12 or fired or k % 10 == 0:
        print(f"{k:>4} {k*DT*1000:6.0f} {nd:>7} {'<-- FIRED' if fired else ('yes' if now else 'no'):>9} {q}", flush=True)

print(f"\n===== RESULT =====", flush=True)
if collapse_step is not None:
    clean = (collapse_part == [(0,1),(2,3)])
    print(f"Collapse fired at reward step {collapse_step} (t={collapse_step*DT*1000:.0f}ms into reward)", flush=True)
    print(f"  dimers at collapse = {collapse_dimers}  (end-of-write was the structured baseline)", flush=True)
    print(f"  partition at collapse = {collapse_part}", flush=True)
    print(f"  -> {'CLEAN {AB}|{CD} at collapse => negative is DOWNSTREAM washout, not timing' if clean else 'BLOBBED at collapse => negative is a fixable TIMING artifact (measure before the burst)'}", flush=True)
else:
    print("Collapse did NOT fire during the 0.3s reward window (unexpected — the gate never opened).", flush=True)
