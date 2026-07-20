#!/usr/bin/env python3
"""
PO-8 UNIT A — acceptance measurement: does the DERIVED release rate extend bond lifetime toward
the Werner-floor crossing? NO SEEDING. Standard rig, standard continuous drive (no invented protocol).

WHAT IS BEING TESTED
  `physical_release_rate` (opt-in, default OFF, off-path bit-identical, network-gate verified at
  digest 515772101786800) replaces the unmoored cross dissolution rate
      k_diss = K_DISENTANGLE_BASE * (1 - eta_factor * P_product)   ~ 0.056 /s  (tau ~ 18 s)
  with the DERIVED driven-NESS rate (PO7_TECHNICAL_BRIEF §5.1.2)
      k_release = 1/T2 + 1/tau_dimer = 9.63e-3 /s                  (tau = 103.8 s)

WHY IT MATTERS (the eligibility trace)
  The advisor's Q4: the trace falls out of T2 — a bond should live until P_S decay drops
  F = P_S^a P_S^b W_ij through the Werner floor (~74 s cross at lambda=5um, ~107 s intra).
  The unmoored rate destroys bridges at ~18 s, LONG before that crossing, so the trace cannot
  express itself. This measures whether removing that obstruction lets bond lifetime approach the
  coherence-limited value — and, critically, WHAT ELSE CAPS IT.

MEASURED (per arm, free-running draws, drive via net.step ONLY)
  - cross-bond lifetime distribution (first-seen -> disappeared), median / p90 / max
  - the same for intra bonds
  - DIMER lifetime distribution (birth_time -> culled), which is the hard cap on any bond:
    a bond dies with its dimer regardless of the release rate. The brief ASSUMES tau_dimer = 200 s;
    this reports what the model actually does.
  - censoring is reported honestly: bonds/dimers still alive at end-of-run are right-censored and
    counted separately, never silently dropped into the median.

ARMS: off (today's physics) vs on (derived rate). Same rig, same drive, free draws, no seeding.
"""
import sys, os, json, argparse
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL6_DIR = os.path.dirname(SWEEP_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODEL6_DIR)))
sys.path.insert(0, MODEL6_DIR)
sys.path.insert(0, SWEEP_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sweep"))   # presynaptic_release lives here
import logging
logging.disable(logging.INFO)

N_SYN, SPACING = 7, 1.0
DT = 5e-3
VOLT = -40e-3
ACT = 0.95
T_SIM = 100.0            # must exceed the ~74 s cross Werner-crossing to be informative
SAMPLE_EVERY = 10        # tracker updates every 10th net.step; sample in lockstep
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
WERNER = 0.5


def one_run(run_id, physical_rate):
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork
    from presynaptic_release import PresynapticRelease

    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    p.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=N_SYN, pattern="linear", spacing_um=SPACING)  # NO seed=
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
        s.dimer_particles.provenance_bonding = True
        s.dimer_particles.spin_resolved = True
    net.disable_auto_commitment = True
    tr = net.entanglement_tracker
    tr.physical_release_rate = bool(physical_rate)      # PO-8 Unit A opt-in
    rel = PresynapticRelease(None)

    first_seen_cross, first_seen_intra = {}, {}
    life_cross, life_intra = [], []
    dimer_birth = {}
    life_dimer = []
    peak_eta = 0.0
    nsteps = int(round(T_SIM / DT))
    for i in range(1, nsteps + 1):
        g = rel.step(ACT, DT)
        net.step(DT, {"voltage": VOLT, "reward": False, "glutamate": g})
        peak_eta = max(peak_eta, max((float(getattr(s, "_backbone_eta", 0.0))
                                      for s in net.synapses), default=0.0))
        if i % SAMPLE_EVERY:
            continue
        t = i * DT
        # --- cross bonds above the Werner bound, spanning two spines ---
        cur_cross = {k for k, v in tr.cross_synapse_bonds.items()
                     if float(v) > WERNER and k[0][0] != k[1][0]}
        for k in cur_cross - first_seen_cross.keys():
            first_seen_cross[k] = t
        for k in list(first_seen_cross):
            if k not in cur_cross:
                life_cross.append(t - first_seen_cross.pop(k))
        # --- intra bonds ---
        cur_intra = set(tr.intra_synapse_bonds_cache.keys())
        for k in cur_intra - first_seen_intra.keys():
            first_seen_intra[k] = t
        for k in list(first_seen_intra):
            if k not in cur_intra:
                life_intra.append(t - first_seen_intra.pop(k))
        # --- dimers (the hard cap: a bond dies with its dimer) ---
        cur_dim = {}
        for si, syn in enumerate(net.synapses):
            for d in syn.dimer_particles.dimers:
                cur_dim[(si, int(d.id))] = float(d.birth_time)
        for gid, bt in cur_dim.items():
            dimer_birth.setdefault(gid, bt)
        for gid in list(dimer_birth):
            if gid not in cur_dim:
                life_dimer.append(t - dimer_birth.pop(gid))

    def summ(a, censored):
        a = np.asarray(a, float)
        if a.size == 0:
            return {"n": 0, "median": None, "p90": None, "max": None, "n_censored": censored}
        return {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "max": float(a.max()), "mean": float(a.mean()), "n_censored": censored}

    return dict(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat(),
                physical_release_rate=bool(physical_rate), t_sim=T_SIM,
                peak_eta=float(peak_eta), ignited=bool(peak_eta > 0.0),
                cross_lifetime=summ(life_cross, len(first_seen_cross)),
                intra_lifetime=summ(life_intra, len(first_seen_intra)),
                dimer_lifetime=summ(life_dimer, len(dimer_birth)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--arm", choices=["off", "on"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl = os.path.join(OUT_DIR, f"po8_unitA_{a.arm}_worker{a.worker}.jsonl")
    open(jsonl, "w").close()
    for j in range(a.n):
        rid = f"{a.arm}_w{a.worker}r{j}"
        sys.stderr.write(f"[unitA {a.arm} w{a.worker}] draw {j+1}/{a.n} ({rid})\n"); sys.stderr.flush()
        rec = one_run(rid, a.arm == "on")
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        sys.stderr.write(f"[unitA {a.arm} w{a.worker}] done {rid} ignited={rec['ignited']} "
                         f"cross_med={rec['cross_lifetime']['median']} "
                         f"dimer_med={rec['dimer_lifetime']['median']}\n")
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
