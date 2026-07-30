#!/usr/bin/env python3
"""
STEP-2 SELECTIVE-CONSOLIDATION PROBE — does the entanglement partition drive SELECTIVE,
durable consolidation, and does the P31->P32 isotope swap kill it?  (Non-circular discrimination.)

Observable (per Sarah's caution): per-cluster durable actin_stable (the consolidation marker),
read POST-GAP — NOT total spine volume (which conflates the isotope-independent calcium growth arm).

Engagement: burst-gated eta (active synapses -> eta ON, inactive -> 0), the transparent form of
Step-1's mechanism that keeps the partition STRUCTURED ({AB}|{CD}) instead of a blob. actin_enlargement
left natural so actin_stable is not confounded.

Per-draw record (PO-11-scorer compatible): dw_cluster = per-cluster Δactin_stable, plus mode(pairing),
order, arm(P31/P32), and regime diagnostics.  Scored by within-condition partial correlation
(po11_valence_score.py): P31 RECOVERS the co-active pairing, P32 at chance.

Usage:
  python step2_probe.py --arm P31 --mode pair1 --order fwd --n 1        # validation
  python step2_probe.py --sweep --draws 10 --out <dir>                  # full sweep (all cells)
"""
import sys, os, json, argparse, types
from datetime import datetime, timezone
import numpy as np
M6 = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/src/models/Model_6"
REPO_SWEEP = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/trusting-heyrovsky-1338e9/sweep"
sys.path.insert(0, M6); sys.path.insert(0, os.path.join(M6,"sweep")); sys.path.insert(0, REPO_SWEEP)
import logging; logging.disable(logging.INFO)
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
from run_theta_burst_45s import analytical_gap

PER, N_CLUST = 2, 4
N_SYN = PER*N_CLUST
DT = 5e-3
WRITE_S, GAP2_S, REWARD_S, INTEG_S, DELAY_S = 3.0, 1.0, 0.3, 6.0, 30.0  # full integ, but stepped WITHOUT the network tracker (measurement already done at reward)
VOLT, ACT, REST = -40e-3, 0.95, -70e-3
ETA_ON = 0.30
CLUSTERS = {c: list(range(c*PER, c*PER+PER)) for c in range(N_CLUST)}
NAME = {0:"A",1:"B",2:"C",3:"D"}
PAIRSETS = {"pair1": ([0,1],[2,3]), "pair2": ([0,2],[1,3])}

def burst_gated_eta(self):
    for s in self.synapses:
        s.set_backbone_condensation_eta(ETA_ON if getattr(s, "_active_now", False) else 0.0)

def active_synapses(t, mode, order):
    first, second = PAIRSETS[mode]
    if order == "rev": first, second = second, first
    act = set()
    if t < WRITE_S:
        for c in first: act |= set(CLUSTERS[c])
    lo = WRITE_S+GAP2_S
    if lo <= t < lo+WRITE_S:
        for c in second: act |= set(CLUSTERS[c])
    return act

def build(frac_p31):
    p = Model6Parameters(); p.em_coupling_enabled=True; p.multi_synapse_enabled=True
    p.environment.fraction_P31 = frac_p31
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=214.0,
                              use_correlated_sampling=True)
    gap_um=3.0; centers=[i*gap_um for i in range(N_CLUST)]
    xs=[centers[c]+0.5*k for c in range(N_CLUST) for k in range(PER)]
    net.positions=np.array([[x,0.0,0.0] for x in xs])
    net.distances=np.sqrt(((net.positions[:,None,:]-net.positions[None,:,:])**2).sum(-1))
    net.coupling_weights=np.exp(-net.distances/5.0); np.fill_diagonal(net.coupling_weights,1.0)
    net.fidelity_weights=np.exp(-net.distances/214.0); np.fill_diagonal(net.fidelity_weights,1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses: s.set_microtubule_invasion(True)
    net.disable_auto_commitment=True
    net._update_backbone_field = types.MethodType(burst_gated_eta, net)   # burst-gated engagement
    return net

def cluster_quotient(net):
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

def one_draw(frac_p31, mode, order):
    net=build(frac_p31); rels=[PresynapticRelease(None) for _ in range(N_SYN)]
    stable_pre=np.array([s.spine_plasticity.actin_stable for s in net.synapses])
    t=0.0; nsteps=int(round((2*WRITE_S+GAP2_S)/DT)); peak_d=0
    for _ in range(nsteps):
        act=active_synapses(t, mode, order)
        for i,s in enumerate(net.synapses): s._active_now = (i in act)
        per_syn=[{"voltage":VOLT,"glutamate":rels[i].step(ACT,DT)} if i in act
                 else {"voltage":REST,"glutamate":0.0} for i in range(N_SYN)]
        net.step(DT, {"per_synapse":per_syn,"reward":False})
        peak_d=max(peak_d, sum(len(s.dimer_particles.dimers) for s in net.synapses)); t+=DT
    quot=cluster_quotient(net)                       # partition at end-of-write
    for _ in range(int(REWARD_S/DT)):
        for s in net.synapses: s._active_now=False
        per_syn=[{"voltage":REST,"glutamate":0.0,"reward":True,"plateau_potential":True} for _ in range(N_SYN)]
        net.step(DT, {"per_synapse":per_syn,"reward":True}); t+=DT
    # CaMKII integration: step synapses DIRECTLY (skip the O(n^2) network tracker — the partition
    # measurement already fired at reward; no new cross-bonds needed here). ~50x cheaper at high dimers.
    for _ in range(int(INTEG_S/DT)):
        for s in net.synapses: s.step(DT, {"voltage":REST,"glutamate":0.0})
        net.time += DT; t+=DT
    comm=[bool(getattr(s,'_camkii_committed',False)) for s in net.synapses]
    analytical_gap(net, DELAY_S, dt_sub=1.0); net.step(DT, {"voltage":REST,"reward":False})
    stable_post=np.array([s.spine_plasticity.actin_stable for s in net.synapses])
    dstab=stable_post-stable_pre
    dw_cluster={NAME[c]: float(dstab[np.array(CLUSTERS[c])].mean()) for c in range(N_CLUST)}
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), arm=("P31" if frac_p31>=0.5 else "P32"),
                fraction_P31=frac_p31, mode=mode, order=order,
                dw_cluster=dw_cluster, partition_edges=quot, n_committed=int(sum(comm)),
                peak_dimers=int(peak_d))

def daemonize(logpath):
    if os.fork() > 0: os._exit(0)
    os.setsid()
    if os.fork() > 0: os._exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    f=open(logpath,"a",buffering=1)
    os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)   # stdout/stderr -> log (raw fds; sys.*.fileno() can be invalid when tool-launched)
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)  # stdin -> /dev/null, best-effort
    except OSError:
        pass

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["P31","P32"]); ap.add_argument("--mode", choices=["pair1","pair2"])
    ap.add_argument("--order", choices=["fwd","rev"], default="fwd"); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--group", action="store_true", help="run P31+P32 for one (mode,order) cell, checkpointed")
    ap.add_argument("--draws", type=int, default=6); ap.add_argument("--out", default=None)
    ap.add_argument("--daemonize", default=None)
    a=ap.parse_args()
    if a.daemonize: daemonize(a.daemonize)
    if a.group:
        import time; t0=time.time()
        outdir=a.out or os.path.join(M6,"results","step2_consolidation"); os.makedirs(outdir, exist_ok=True)
        mode, order = a.mode or "pair1", a.order
        print(f"[0s] GROUP {mode}/{order} ppid={os.getppid()} pid={os.getpid()} draws={a.draws} (P31+P32)", flush=True)
        for frac,armn in [(0.0,"P32"),(1.0,"P31")]:      # P32 first (fast) so partial results arrive early
            fp=os.path.join(outdir, f"{armn}_{mode}_{order}.jsonl")
            with open(fp,"a",buffering=1) as f:
                for k in range(a.draws):
                    r=one_draw(frac, mode, order); f.write(json.dumps(r)+"\n")
                    print(f"[{time.time()-t0:.0f}s] {armn} {mode} {order} #{k} edges={r['partition_edges']} "
                          f"commit={r['n_committed']} dimers={r['peak_dimers']} "
                          f"dw={ {k2:round(v,4) for k2,v in r['dw_cluster'].items()} }", flush=True)
        print(f"[{time.time()-t0:.0f}s] GROUP {mode}/{order} DONE", flush=True)
    else:
        frac = 1.0 if a.arm=="P31" else 0.0
        for _ in range(a.n):
            print(json.dumps(one_draw(frac, a.mode or "pair1", a.order)), flush=True)
