#!/usr/bin/env python3
"""
PO-9 UNIT B — the readout-time input-selectivity keystone. NO SEEDING.

Pre-registered: docs/PREREG_PO9_UNIT_B_READOUT_KEYSTONE.md. Read it before changing anything.

QUESTION: does the correlated-domain partition AT READOUT carry WHICH INPUT was presented,
beyond density and geometry, AS A FUNCTION OF lambda_F?

PROTOCOL per draw:  per-synapse WRITE (SYNC or STAGGER, density matched, ALL driven)
                    -> analytical_gap(delay)  (AGE, the confined-niche intrinsic mechanism)
                    -> score the SYNAPSE-level correlation graph vs the input grouping.

SCORING (synapse level; the dimer-level trap is documented in the prereg): the 7-node synapse
graph has edge weight W_ij = sum over cross-bridges (i,j) with F>Werner of p_e=(4F-1)/3 (total
correlation flux between the two synapses). Q_act = Newman modularity of that weighted graph
against the input-group partition {A}{B}{spacer}. NULL BAND = Q against >=200 random equal-size
relabellings; z = (Q_true - mean_null)/std_null. Report z per (condition, lambda_F, delay).

VERDICT DEMONSTRATED FAILING FIRST: run condition SYNCxSYNC (identity removed) and confirm z is
within the null band before trusting any positive on SYNC-vs-STAGGER.

lambda decoupling (L.PO9-1): metabolic length stays 5 um; fidelity_length_um = lambda_F is swept.
Drive via net.step per-synapse ONLY (per-synapse s.step() leaves eta=0 -> nothing ignites).
"""
import sys, os, json, math, argparse
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR); sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))
import logging; logging.disable(logging.INFO)

from presynaptic_release import PresynapticRelease
from run_theta_burst_45s import analytical_gap   # PO-4 surface; CALLED, never edited

N_SYN = 8
DT = 5e-3
WRITE_S = 40.0                       # per-group write window. Was 20 s; raised because condensate
                                     # ignition latency is ~13-15 s, comparable to a 20 s window, so
                                     # partial temporal offsets gave NO CO-IGNITION overlap (cross_w=0
                                     # by artifact, a false "step"). At 40 s, ~25 s remains co-ignitable
                                     # after latency, so the graded-overlap sweep can resolve cross_w
                                     # vs actual co-ignition duration (advisor point 4).
VOLT, ACT = -40e-3, 0.95
REST = -70e-3
WERNER = 0.5
DELAYS = [0.0, 10.0, 20.0, 30.0, 40.0, 60.0]
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "po9_unitB")

# TWO SEPARATED CLUSTERS of 4 (PO-9). WHY this geometry, not a linear array:
#   - Ignition needs a spatial QUORUM: a tight cluster of 4 ignites (peak η ~0.19); 3-4 spread on a
#     linear array do NOT (metabolic aggregation P_agg < P_c=21.51). Verified.
#   - Metabolic aggregation reaches ~λ_met=5 µm, so ADJACENT groups on a linear array sit inside each
#     other's aggregation range -> driving one ignites both (branch-global, L·ETA-4) -> STAGGER
#     could not create group-local structure. Verified fix: at 15 µm separation, driving cluster A
#     ignites A and leaves B DARK (η=[.19,.20,.19,.18 | 0,0,0,0]). Group-local ignition is real.
#   - The 15 µm gap also makes λ_F the DECISIVE variable: cross-cluster w = exp(-15/λ_F) is 0.05 at
#     λ_F=5 (clusters can never bind -> always 2 domains, geometry wins) vs 0.93 at λ_F=214 (clusters
#     CAN bind if co-active). So input can structure the readout only at long λ_F -- the thing to map.
CLUSTER_X = [0.0, 0.5, 1.0, 1.5, 15.0, 15.5, 16.0, 16.5]   # A = first 4, B = last 4
GROUPINGS = {
    "twocluster": ([0, 1, 2, 3], [4, 5, 6, 7], []),
}


def synapse_corr_matrix(tr):
    """W_ij = sum over cross-bridges between synapse i and j (F>Werner) of p_e=(4F-1)/3."""
    W = np.zeros((N_SYN, N_SYN))
    for key, F in tr.cross_synapse_bonds.items():
        F = float(F)
        if F <= WERNER:
            continue
        si, sj = key[0][0], key[1][0]
        if si == sj:
            continue
        p = (4.0 * F - 1.0) / 3.0
        W[si, sj] += p
        W[sj, si] += p
    return W


def modularity(W, labels):
    """Newman weighted modularity for the given community labels (spacer is its own community)."""
    m2 = W.sum()
    if m2 <= 0:
        return 0.0
    k = W.sum(axis=1)
    Q = 0.0
    n = W.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += W[i, j] - k[i] * k[j] / m2
    return Q / m2


def score_Qact(W, A, B, spacer, n_null=200):
    labels = np.empty(N_SYN, dtype=int)
    for i in A: labels[i] = 0
    for i in B: labels[i] = 1
    for i in spacer: labels[i] = 2
    Q_true = modularity(W, labels)
    sizes = [len(A), len(B), len(spacer)]
    nulls = []
    for _ in range(n_null):
        perm = np.random.permutation(N_SYN)   # free draw; no seed
        lab = np.empty(N_SYN, dtype=int)
        lab[perm[:sizes[0]]] = 0
        lab[perm[sizes[0]:sizes[0] + sizes[1]]] = 1
        lab[perm[sizes[0] + sizes[1]:]] = 2
        nulls.append(modularity(W, lab))
    nulls = np.asarray(nulls)
    mu, sd = float(nulls.mean()), float(nulls.std())
    z = (Q_true - mu) / sd if sd > 1e-12 else 0.0
    return dict(Q_true=float(Q_true), null_mu=mu, null_sd=sd,
                null_p05=float(np.percentile(nulls, 5)), null_p95=float(np.percentile(nulls, 95)),
                z=float(z), edge_weight_total=float(W.sum()))


def active_at(t, cond, A, B, spacer, offset_s):
    """Which synapses carry drive at time t. GRADED-OVERLAP design (advisor point 4): cluster A is
    driven [0, WRITE_S]; cluster B is driven [offset_s, offset_s+WRITE_S]. Each cluster gets the SAME
    total drive (WRITE_S) — only the temporal offset varies. Spacer (if any) in [0, WRITE_S].
      offset_s = 0        -> full overlap   (== SYNC)
      offset_s = WRITE_S  -> zero overlap   (== STAGGER)
      0 < offset_s < W    -> partial overlap, fraction = (WRITE_S - offset_s)/WRITE_S
    'sync'/'stagger'/'sync2' kept as named aliases (offset 0 / WRITE_S / 0)."""
    if cond == "stagger":
        offset_s = WRITE_S
    elif cond in ("sync", "sync2"):
        offset_s = 0.0
    act = set()
    if t < WRITE_S:
        act |= set(A) | set(spacer)
    if offset_s <= t < offset_s + WRITE_S:
        act |= set(B)
    return act


def overlap_fraction(cond, offset_s):
    if cond == "stagger": offset_s = WRITE_S
    elif cond in ("sync", "sync2"): offset_s = 0.0
    return max(0.0, (WRITE_S - offset_s) / WRITE_S)


def write_end(cond, offset_s):
    if cond == "stagger": offset_s = WRITE_S
    elif cond in ("sync", "sync2"): offset_s = 0.0
    return offset_s + WRITE_S


def one_run(run_id, lam_F, cond, grouping, offset_s=0.0):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    A, B, spacer = GROUPINGS[grouping]

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, fidelity_length_um=float(lam_F))  # NO seed
    # Override to two separated clusters BEFORE initialize: metabolic weights on λ_met=5 µm,
    # fidelity weights on λ_F. Same recompute the constructor does, on custom positions.
    net.positions = np.array([[x, 0.0, 0.0] for x in CLUSTER_X])
    net.distances = np.sqrt(((net.positions[:, None, :] - net.positions[None, :, :]) ** 2).sum(-1))
    net.coupling_weights = np.exp(-net.distances / 5.0); np.fill_diagonal(net.coupling_weights, 1.0)
    net.fidelity_weights = np.exp(-net.distances / float(lam_F)); np.fill_diagonal(net.fidelity_weights, 1.0)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker

    rels = [PresynapticRelease(None) for _ in range(N_SYN)]   # independent free draws per synapse
    peak_eta = 0.0
    peak_eta_per = np.zeros(N_SYN)     # per-synapse peak η — diagnoses group-local vs branch-global ignition
    Ai0, Bi0 = np.array(A), np.array(B)
    # per-cluster co-IGNITION window: the real x-axis for the graded sweep. A cross-cluster bond needs
    # BOTH clusters ignited (η>0) simultaneously; ignition lags drive by ~13-15 s, so co-drive time
    # overstates the bindable window. Track the interval where BOTH cluster means have η>0.
    coign_steps = 0
    t = 0.0
    nsteps = int(round(write_end(cond, offset_s) / DT))
    for _ in range(nsteps):
        act = active_at(t, cond, A, B, spacer, offset_s)
        per_syn = []
        for i in range(N_SYN):
            if i in act:
                g = rels[i].step(ACT, DT)
                per_syn.append({"voltage": VOLT, "glutamate": g})
            else:
                per_syn.append({"voltage": REST, "glutamate": 0.0})
        net.step(DT, {"per_synapse": per_syn, "reward": False})
        etas = np.array([float(getattr(s, "_backbone_eta", 0.0)) for s in net.synapses])
        peak_eta_per = np.maximum(peak_eta_per, etas)
        peak_eta = max(peak_eta, float(etas.max()) if etas.size else 0.0)
        if etas[Ai0].mean() > 0.0 and etas[Bi0].mean() > 0.0:
            coign_steps += 1
        t += DT
    coignition_s = coign_steps * DT

    snaps, prev = [], 0.0
    for d in DELAYS:
        if d > prev:
            analytical_gap(net, d - prev, dt_sub=1.0)
            net.step(DT, {"voltage": REST, "reward": False})   # tail refresh (needs coupling_weights)
            prev = d
        W = synapse_corr_matrix(tr)
        sc = score_Qact(W, A, B, spacer)   # kept, but geometry-confounded here (see prereg amendment 1)
        # AMENDED PRIMARY STATISTIC (prereg amendment 1): cross-cluster BLOCK weight. Q_act is
        # trivially high for two spatial clusters against a grouping that equals them (the failing-
        # first control showed SYNC not null). What "the clusters merge into one domain" actually
        # means is that the A-B block of the correlation matrix carries weight. That forms only when
        # A and B are CO-ACTIVE (SYNC, not STAGGER) AND λ_F is long enough (w=exp(-15/λ_F) clears the
        # Werner floor). So cross_w is the input×λ signal; within_w is the geometry baseline.
        Ai, Bi = np.array(A), np.array(B)
        within_w = float(W[np.ix_(Ai, Ai)].sum() + W[np.ix_(Bi, Bi)].sum())   # symmetric (double-counts)
        cross_w = float(W[np.ix_(Ai, Bi)].sum() + W[np.ix_(Bi, Ai)].sum())    # A-B block, same convention
        sc["within_w"] = within_w
        sc["cross_w"] = cross_w
        sc["cross_frac"] = cross_w / (within_w + cross_w) if (within_w + cross_w) > 0 else 0.0
        # clock reconciliation (advisor point 3): record P_S so cross_w's fall can be checked against
        # the Werner-floor crossing F_cross = P_S^2 * exp(-15/λ_F) < 1/2. NOTE analytical_gap decays
        # at T_eff = T_base/(spread_factor*template_factor) ~ 140 s, NOT T_singlet=216 s.
        ps = [float(dd["P_S"]) for dd in tr.all_dimers]
        crossFs = [float(v) for k, v in tr.cross_synapse_bonds.items()
                   if float(v) > WERNER and k[0][0] != k[1][0]]
        sc["mean_PS"] = float(np.mean(ps)) if ps else 0.0
        sc["crossF_med"] = float(np.median(crossFs)) if crossFs else 0.0
        sc["delay"] = d
        sc["n_cross_syn_edges"] = int((W > 0).sum() // 2)
        snaps.append(sc)
    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                lam_F=float(lam_F), cond=cond, grouping=grouping,
                offset_s=float(offset_s), overlap_frac=float(overlap_fraction(cond, offset_s)),
                coignition_s=float(coignition_s),
                peak_eta=float(peak_eta), ignited=bool(peak_eta > 0.0),
                peak_eta_per=[round(float(x), 4) for x in peak_eta_per],
                write_end_s=write_end(cond, offset_s), snapshots=snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--lamF", type=float, required=True)
    ap.add_argument("--cond", choices=["sync", "stagger", "sync2", "graded"], required=True)
    ap.add_argument("--offset", type=float, default=0.0,
                    help="temporal offset of cluster B in seconds (0=full overlap .. WRITE_S=none); "
                         "used only when --cond graded")
    ap.add_argument("--grouping", choices=list(GROUPINGS), default="twocluster")
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po9_unitB_{a.tag}_w{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.tag}_w{a.worker}r{j}"
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] draw {j+1}/{a.n} "
                         f"lamF={a.lamF} cond={a.cond} offset={a.offset} grp={a.grouping}\n"); sys.stderr.flush()
        rec = one_run(rid, a.lamF, a.cond, a.grouping, a.offset)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        cw = {s["delay"]: round(s["cross_w"], 0) for s in rec["snapshots"]}
        sys.stderr.write(f"[unitB {a.tag} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"overlap={rec['overlap_frac']:.2f} cross_w_by_delay={cw}\n"); sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
