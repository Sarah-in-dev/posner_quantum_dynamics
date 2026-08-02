#!/usr/bin/env python3
"""
F2 CONTROL LADDER — validate the binding-melt measurement trigger (posner_binding) after wiring.

Pre-registered in docs/PREREG_F2_BINDING_AS_MEASUREMENT.md. Adapted from the STEP-2 probe, with the
TWO traps that STEP-2 carried FIXED:
  (1) imports point at the MASTER checkout (where the F2 wiring + posner_binding live), NOT the
      trusting-heyrovsky worktree.  (2) the isotope arm uses F1's environment.dopant ∈ {None,'Li6','Li7'}
      — NOT the deprecated circular environment.fraction_P31.

Conditions:
  C0  undoped, reward PRESENT   — baseline (reproduce STEP-2's calcium-dominated commitment; pipeline works).
  C1  undoped, reward ABSENT    — the DECOUPLING at scale: the measurement must still FIRE with no reward.
  C2  Li6 vs Li7, matched drive — the isotope moves the MEASUREMENT via the coherence window (F1 follow-on:
      delayed readout, spanning the window — NOT edges-on-contact).
  C3  shuffle control           — randomize which synapses are co-active vs the partition; selectivity must vanish.

Per-draw record (PO-11-scorer compatible + F2 instrumentation): dw_cluster (per-cluster Δactin_stable
POST-GAP), partition_edges (end-of-write), measured (did the collapse fire), measured_at_s (when),
partition_at_measure (acceptance-4: is it the clean {AB}|{CD}?), n_committed, peak_dimers.

Usage:
  python f2_control_ladder.py --cond C1 --n 1 --fast          # dry run (reduced steps)
  python f2_control_ladder.py --cond C2 --iso Li7 --n 6 --out <dir>
  python f2_control_ladder.py --group --draws 8 --out <dir> [--daemonize <log>]   # full ladder, checkpointed
"""
import sys, os, json, argparse, types
from datetime import datetime, timezone
import numpy as np

# --- MASTER paths (trap #1 fixed): this checkout, not the worktree ---
M6 = "/Users/sarahdavidson/posner_quantum_dynamics/src/models/Model_6"
REPO_SWEEP = "/Users/sarahdavidson/posner_quantum_dynamics/sweep"
sys.path.insert(0, M6); sys.path.insert(0, os.path.join(M6, "sweep")); sys.path.insert(0, REPO_SWEEP)
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork
from presynaptic_release import PresynapticRelease
from run_theta_burst_45s import analytical_gap

PER, N_CLUST = 2, 4
N_SYN = PER * N_CLUST
DT = 5e-3
WRITE_S, GAP2_S, REWARD_S, INTEG_S, DELAY_S = 3.0, 1.0, 0.3, 6.0, 30.0
VOLT, ACT, REST = -40e-3, 0.95, -70e-3
ETA_ON = 0.30
CLUSTERS = {c: list(range(c * PER, c * PER + PER)) for c in range(N_CLUST)}
NAME = {0: "A", 1: "B", 2: "C", 3: "D"}
PAIRSETS = {"pair1": ([0, 1], [2, 3]), "pair2": ([0, 2], [1, 3])}


def burst_gated_eta(self):
    for s in self.synapses:
        s.set_backbone_condensation_eta(ETA_ON if getattr(s, "_active_now", False) else 0.0)


def active_synapses(t, mode, order, wf):
    first, second = PAIRSETS[mode]
    if order == "rev":
        first, second = second, first
    act = set()
    if t < wf:
        for c in first:
            act |= set(CLUSTERS[c])
    lo = wf + GAP2_S
    if lo <= t < lo + wf:
        for c in second:
            act |= set(CLUSTERS[c])
    return act


def build(dopant):
    p = Model6Parameters(); p.em_coupling_enabled = True; p.multi_synapse_enabled = True
    p.environment.dopant = dopant           # F1 lever (trap #2 fixed): None / 'Li6' / 'Li7'
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=214.0,
                              use_correlated_sampling=True)
    gap_um = 3.0; centers = [i * gap_um for i in range(N_CLUST)]
    xs = [centers[c] + 0.5 * k for c in range(N_CLUST) for k in range(PER)]
    net.positions = np.array([[x, 0.0, 0.0] for x in xs])
    net.distances = np.sqrt(((net.positions[:, None, :] - net.positions[None, :, :]) ** 2).sum(-1))
    net.coupling_weights = np.exp(-net.distances / 5.0); np.fill_diagonal(net.coupling_weights, 1.0)
    net.fidelity_weights = np.exp(-net.distances / 214.0); np.fill_diagonal(net.fidelity_weights, 1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    net._update_backbone_field = types.MethodType(burst_gated_eta, net)
    return net


def cluster_quotient(net):
    tr = net.entanglement_tracker; tr.collect_dimers(net.synapses, net.positions)
    gid_syn = {d['global_id']: d['synapse_idx'] for d in tr.all_dimers}
    syn_c = {i: c for c in range(N_CLUST) for i in CLUSTERS[c]}
    edges = set()
    for (i, j), F in tr.cross_synapse_bonds.items():
        if F > 0.5:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                ca, cb = syn_c[a], syn_c[b]
                if ca != cb:
                    edges.add((min(ca, cb), max(ca, cb)))
    return sorted(edges)


def one_draw(cond, dopant, mode, order, reward_present, shuffle, fast=False):
    wf = 1.0 if fast else WRITE_S
    integ = 1.0 if fast else INTEG_S
    delay = 3.0 if fast else DELAY_S
    rwd = 0.05 if fast else REWARD_S   # fast: cap the plateau reward phase — plateau_potential floods
                                       # calcium→dimers (thousands), and the O(n²) tracker on that is the
                                       # ~47-min cost. Reward-present cells are NOT fast unless this is bounded.
    net = build(dopant); rels = [PresynapticRelease(None) for _ in range(N_SYN)]
    stable_pre = np.array([s.spine_plasticity.actin_stable for s in net.synapses])
    # C3 shuffle: permute which synapse-indices are treated as co-active, breaking the geometry↔partition tie
    perm = list(range(N_SYN))
    if shuffle:
        perm = list(np.random.permutation(N_SYN))
    t = 0.0; nsteps = int(round((2 * wf + GAP2_S) / DT)); peak_d = 0
    measured_at = None; partition_at_measure = None
    for _ in range(nsteps):
        act = active_synapses(t, mode, order, wf)
        act = set(perm[i] for i in act) if shuffle else act
        for i, s in enumerate(net.synapses):
            s._active_now = (i in act)
        per_syn = [{"voltage": VOLT, "glutamate": rels[i].step(ACT, DT)} if i in act
                   else {"voltage": REST, "glutamate": 0.0} for i in range(N_SYN)]
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        # F2 instrumentation: catch the FIRST measurement fire + the partition it read (acceptance-4)
        if measured_at is None and getattr(net, "_coordinated_measurement_performed", False):
            measured_at = round(t, 4); partition_at_measure = cluster_quotient(net)
        peak_d = max(peak_d, sum(len(s.dimer_particles.dimers) for s in net.synapses)); t += DT
    quot = cluster_quotient(net)
    # reward phase (only when reward_present): plateau flood + reward signal (now a LEARNING signal, not the trigger)
    if reward_present:
        for _ in range(int(rwd / DT)):
            for s in net.synapses:
                s._active_now = False
            per_syn = [{"voltage": REST, "glutamate": 0.0, "reward": True, "plateau_potential": True} for _ in range(N_SYN)]
            net.step(DT, {"per_synapse": per_syn, "reward": True})
            if measured_at is None and getattr(net, "_coordinated_measurement_performed", False):
                measured_at = round(t, 4); partition_at_measure = cluster_quotient(net)
            t += DT
    # CaMKII integration (step synapses directly; measurement already fired)
    for _ in range(int(integ / DT)):
        for s in net.synapses:
            s.step(DT, {"voltage": REST, "glutamate": 0.0})
        net.time += DT; t += DT
    comm = [bool(getattr(s, '_camkii_committed', False)) for s in net.synapses]
    analytical_gap(net, delay, dt_sub=1.0); net.step(DT, {"voltage": REST, "reward": False})
    stable_post = np.array([s.spine_plasticity.actin_stable for s in net.synapses])
    dstab = stable_post - stable_pre
    dw_cluster = {NAME[c]: float(dstab[np.array(CLUSTERS[c])].mean()) for c in range(N_CLUST)}
    return dict(timestamp=datetime.now(timezone.utc).isoformat(), cond=cond, dopant=str(dopant),
                mode=mode, order=order, reward_present=reward_present, shuffle=shuffle,
                dw_cluster=dw_cluster, partition_edges=quot,
                measured=measured_at is not None, measured_at_s=measured_at,
                partition_at_measure=partition_at_measure,
                n_committed=int(sum(comm)), peak_dimers=int(peak_d))


# condition table: (dopant, reward_present, shuffle); iso overrides dopant for C2
CONDS = {
    "C0": (None, True, False),
    "C1": (None, False, False),
    "C2": (None, False, False),   # dopant set from --iso
    "C3": (None, False, True),
}


def daemonize(logpath):
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    f = open(logpath, "a", buffering=1)
    os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def run_cell(cond, iso, mode, order, n, outdir, fast, t0, label):
    import time
    dopant, reward_present, shuffle = CONDS[cond]
    if cond == "C2":
        dopant = iso
    fp = os.path.join(outdir, f"{cond}_{iso or 'none'}_{mode}_{order}.jsonl") if outdir else None
    f = open(fp, "a", buffering=1) if fp else None
    for k in range(n):
        r = one_draw(cond, dopant, mode, order, reward_present, shuffle, fast=fast)
        if f:
            f.write(json.dumps(r) + "\n")
        print(f"[{time.time()-t0:.0f}s] {label} {cond} {str(dopant):>4} {mode} {order} #{k} "
              f"measured={r['measured']}@{r['measured_at_s']} part@meas={r['partition_at_measure']} "
              f"edges={r['partition_edges']} commit={r['n_committed']} dimers={r['peak_dimers']} "
              f"dw={ {k2: round(v,4) for k2,v in r['dw_cluster'].items()} }", flush=True)
    if f:
        f.close()


if __name__ == '__main__':
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=list(CONDS))
    ap.add_argument("--iso", choices=["Li6", "Li7"], default=None)
    ap.add_argument("--mode", choices=["pair1", "pair2"], default="pair1")
    ap.add_argument("--order", choices=["fwd", "rev"], default="fwd")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--group", action="store_true", help="run the FULL ladder, checkpointed")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="reduced steps (dry run)")
    ap.add_argument("--daemonize", default=None)
    a = ap.parse_args()
    if a.daemonize:
        daemonize(a.daemonize)
    t0 = time.time()
    outdir = a.out or os.path.join(M6, "results", "f2_control_ladder")
    if a.out or a.group:
        os.makedirs(outdir, exist_ok=True)
    if a.group:
        print(f"[0s] F2 LADDER pid={os.getpid()} draws={a.draws} fast={a.fast}", flush=True)
        # reward-ABSENT cells first (fast, the scientifically key ones: decoupling / isotope / shuffle);
        # reward-present C0 LAST (plateau-flooded, slowest even in --fast) so it never blocks the results.
        for cond in ["C1", "C3"]:
            for mode in ["pair1", "pair2"]:
                run_cell(cond, None, mode, "fwd", a.draws, outdir, a.fast, t0, "grp")
        for iso in ["Li6", "Li7"]:
            for mode in ["pair1", "pair2"]:
                run_cell("C2", iso, mode, "fwd", a.draws, outdir, a.fast, t0, "grp")
        if not a.fast:   # C0 is reward-present (plateau-flooded, ~7.7 min/draw even capped) and in fast
            for mode in ["pair1", "pair2"]:   # mode consolidation truncates to commit=0 — it belongs in
                run_cell("C0", None, mode, "fwd", a.draws, outdir, a.fast, t0, "grp")   # the FULL overnight set
        print(f"[{time.time()-t0:.0f}s] F2 LADDER DONE", flush=True)
    else:
        run_cell(a.cond or "C1", a.iso, a.mode, a.order, a.n, outdir if a.out else None, a.fast, t0, "one")
