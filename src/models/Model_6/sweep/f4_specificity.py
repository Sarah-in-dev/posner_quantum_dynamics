#!/usr/bin/env python3
"""
F4 — SYNAPTIC SPECIFICITY: does the RIGHT synapse get the credit?

THE QUESTION (the program's payoff claim, and a CONTESTED keystone — never demonstrated).
Everything validated so far is a SINGLE synapse. The three-factor story says credit is assigned by
  GLOBAL reward  ×  LOCAL eligibility
Dopamine is volume transmission: at reward time it reaches EVERY synapse on the segment equally. The only
thing that distinguishes synapses is whether each one carries a coherent tag from its own recent activity.
So the test is: drive a SUBSET, wait, deliver ONE GLOBAL reward, and ask whether the synapses that commit
are the ones that were driven.

WHY THIS OBSERVABLE (and not the previous one). F2-e ran a multi-synapse consolidation study and returned a
NULL, but it scored the *partition* (po11 partial-correlation vs a permuted-label null). Its own conclusion
was that the open question — "is the RIGHT (locally-driven) synapse the one that commits?" — "needs a
DIFFERENT observable (committed-synapse identity vs the driven one), not more of this partition-scored
design." This harness uses exactly that observable: per-synapse COMMITTED IDENTITY against DRIVEN IDENTITY.

PRE-REGISTERED PREDICTION (written before running):
  - REWARD arm: commit rate among DRIVEN synapses > among UNDRIVEN, i.e. credit is specific.
  - NO-REWARD arm: little or nothing commits anywhere (reward is necessary — the F3 result must survive
    at network scale, not just at one synapse).
  A null (driven ≈ undriven) is a REAL RESULT: it would mean the reward-signed readout is not synapse-
  specific, i.e. global reward smears credit over the whole segment. Report it; do not tune.

SCORING: commit PROBABILITY of driven vs undriven synapses, pooled across seeds, against a two-sided
PERMUTATION null that shuffles the driven/undriven labels WITHIN each run (so it respects each run's
overall commit count and only tests whether commitment tracks the driven identity). Also reports
precision/recall of "committed" as a predictor of "driven". Commitment is stochastic by construction
(DDSC; Jain 2024) so the statistic is a probability contrast, never an all-or-none threshold.

OPERATIONS: self-daemonizes (double-fork + setsid), thread-caps, and appends one JSON line per run so
partial results survive and the run is resumable. `--smoke` runs a tiny fast configuration to verify the
machinery end-to-end BEFORE paying for the real batch (F2-e's lesson: network draws are expensive).
"""
import argparse
import json
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "f4_specificity")

DT = 5e-3
DA_TONIC, DA_BURST = 20e-9, 10e-6
DRIVE_V, REST_V = -40e-3, -70e-3


def daemonize(log_path):
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, sys.stdout.fileno()); os.dup2(fd, sys.stderr.fileno())
    os.dup2(os.open(os.devnull, os.O_RDONLY), sys.stdin.fileno())


def build_network(n_syn, seed):
    import numpy as np
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse
    from multi_synapse_network import MultiSynapseNetwork

    np.random.seed(seed)
    p = Model6Parameters()
    p.em_coupling_enabled = True
    p.multi_synapse_enabled = True
    net = MultiSynapseNetwork(n_synapses=n_syn, pattern="linear", spacing_um=0.5,
                              coupling_length_um=5.0, use_correlated_sampling=True)
    net.initialize(Model6QuantumSynapse, p)
    for s in net.synapses:
        s.set_microtubule_invasion(True)          # the structural gate must be open for cross-bonding
        s._reward_gated_consolidation = True      # the F3 reward-gated readout, per synapse
        s._reward_gating_mode = "quantum"
    return net


def run_one(n_syn, driven, rewarded, seed, elig_s, delay_s, reward_s, settle_s, burst_s, drive_v=DRIVE_V):
    """One network draw. `driven` = indices driven during the eligibility window. Returns per-synapse state."""
    import numpy as np
    from presynaptic_release import PresynapticRelease
    net = build_network(n_syn, seed)
    # === GLUTAMATE (added 2026-08-29, F4-c) ===
    # Every F4 run to date drove VOLTAGE ONLY, which is the ERR-2 defect quantum-system-canonical §2.3 flags
    # as "the live-model glutamate-wiring gap" and that sweep/po7_unit8_eta2_partition.py names by line
    # ("the ERR-2 defect that made L·ETA-1 measure NMDARs structurally silent"; L·ETA-2's ignition came from
    # glutamate reaching the drivers).
    #
    # WHAT IT ACTUALLY BUYS — MEASURED, and NOT what a first (broken) probe suggested. An earlier probe read a
    # NONEXISTENT attribute `_nmdar_open_fraction` via getattr(...,0.0), so it silently reported ca_open = 0
    # everywhere and produced a confident, WRONG story ("NMDARs never open without glutamate"). The correct
    # call is the one po7's working rig uses: `s.calcium.channels.get_open_fraction()`. Re-measured with it,
    # single synapse, -20 mV, 4 s:
    #     voltage only        ca_open peak 0.44   r: 0.039 -> 0.072
    #     voltage + glutamate ca_open peak 0.74   r: 0.039 -> 0.080
    # Channels DO open without glutamate. What glutamate does is raise the peak open fraction ~1.7x, and
    # because P_met = P_BASAL + E_invasion * ca_open * P_active_max (model6_parameters.py:57) is a PRODUCT,
    # that directly relaxes how far the SLOW factor has to climb:
    #     r > 1 requires E_invasion * ca_open > (21.51-0.84)/60 = 0.3445
    #       at ca_open 0.44 (voltage only)  -> E_invasion > 0.783
    #       at ca_open 0.74 (+ glutamate)   -> E_invasion > 0.466
    # From the actin probe (120 s sustained): Ca 0.8 uM -> E_inv 0.531, 1.0 uM -> 0.721. So WITH glutamate the
    # requirement sits inside the reachable range; without it, it needs E_invasion ~0.78, which is far harder.
    # NB the binding constraint is therefore E_invasion, which grows on the SLOW structural-plasticity
    # timescale -- so the eligibility drive must be LONG (tens of seconds), not merely present.
    #
    # Glutamate goes ONLY to the driven synapses, so the eligibility substrate stays perfectly specific by
    # construction (undriven synapses still form no tag) -- the property that makes the credit question well
    # posed in the first place.
    rel = PresynapticRelease(seed=seed)

    def phase(duration, drive_idx, da, glutamate=True):
        n = int(round(duration / DT))
        for _ in range(n):
            g = rel.step(0.95, DT)
            per_syn = [({"voltage": drive_v, "glutamate": g} if (glutamate and i in drive_idx)
                        else {"voltage": (drive_v if i in drive_idx else REST_V)})
                       for i in range(n_syn)]
            for s in net.synapses:
                s._da_signal = da
            net.step(DT, {"per_synapse": per_syn, "reward": False})

    # 1. LOCAL eligibility: only the driven synapses build a coherent tag
    phase(elig_s, set(driven), DA_TONIC)
    tags = [len(s.dimer_particles.dimers) for s in net.synapses]
    ps_tag = [float(getattr(s, "_mean_singlet_prob", float("nan"))) for s in net.synapses]
    # DID THE FULL SYSTEM ENGAGE? eta>0 means the tryptophan/MT backbone condensed (Frohlich) and
    # cross-synapse bonds are possible at all; F4-a ran with eta==0 (backbone inert, local tags only).
    eta = [float(getattr(s, "_backbone_eta", getattr(s, "_condensation_eta", 0.0))) for s in net.synapses]
    tr = net.entanglement_tracker
    n_cross = sum(1 for F in tr.cross_synapse_bonds.values() if F > 0.5)
    n_cross_any = len(tr.cross_synapse_bonds)
    e_inv = [float(getattr(s.spine_plasticity, "E_invasion", 0.0)) for s in net.synapses]

    # 2. delay — everything at rest, tonic dopamine; tags decohere passively
    phase(delay_s, set(), DA_TONIC)
    pre_commit = [bool(getattr(s, "_camkii_committed", False)) for s in net.synapses]

    # 3. GLOBAL reward — identical dopamine at every synapse (volume transmission)
    nb = int(round(burst_s / DT))
    n_rw = int(round(reward_s / DT))
    for k in range(n_rw):
        da = (DA_BURST if (rewarded and k < nb) else DA_TONIC)
        for s in net.synapses:
            s._da_signal = da
        net.step(DT, {"per_synapse": [{"voltage": REST_V}] * n_syn, "reward": False})   # no drive, no glutamate

    # 4. settle
    phase(settle_s, set(), DA_TONIC)

    return dict(
        seed=seed, rewarded=bool(rewarded), driven=sorted(list(driven)),
        n_syn=n_syn, tags=tags, ps_tag=ps_tag, pre_commit=pre_commit,
        eta=eta, e_invasion=e_inv, n_cross_bonds_werner=n_cross, n_cross_bonds_any=n_cross_any,
        committed=[bool(getattr(s, "_camkii_committed", False)) for s in net.synapses],
        mem=[float(s.camkii.molecular_memory) for s in net.synapses],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of seeds this process runs")
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index. Shard across cores by giving each process a distinct "
                         "--seed-start/--n and its own --tag, so parallel workers never redo the same cell.")
    ap.add_argument("--n-syn", type=int, default=8)
    ap.add_argument("--n-driven", type=int, default=3)
    ap.add_argument("--elig-s", type=float, default=None,
                    help="eligibility drive seconds; >=~15 s is needed for E_invasion to cross its "
                         "threshold so the tryptophan backbone can condense (eta>0 => cross-synapse bonds)")
    ap.add_argument("--drive-mv", type=float, default=-40.0, help="eligibility drive voltage in mV")
    ap.add_argument("--tag", type=str, default="runs", help="results file tag")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config to verify the machinery")
    ap.add_argument("--fg", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        n_syn, n_driven = 4, 2
        elig_s, delay_s, reward_s, settle_s, burst_s = 0.2, 1.0, 2.0, 1.0, 0.5
        seeds, arms = [0], [True]
    else:
        n_syn, n_driven = a.n_syn, a.n_driven
        # elig 2.0 s: MEASURED (pre-run) to give the driven synapses a reliable tag while UNDRIVEN synapses
        # stay at exactly zero dimers / P_S = 0.25 (thermal floor) — i.e. the eligibility substrate is
        # perfectly specific by construction, which is what makes the credit question well posed.
        # burst = reward window (SUSTAINED) for this FIRST specificity test: establish whether the effect
        # exists in the favourable regime (commit prob 0.75) before re-testing it with the harder,
        # physiological brief burst (0.375), where the per-run sample would be thin.
        elig_s, delay_s, reward_s, settle_s, burst_s = 2.0, 10.0, 20.0, 10.0, 20.0
        if a.elig_s is not None:
            elig_s = a.elig_s
        seeds, arms = list(range(a.seed_start, a.seed_start + a.n)), [True, False]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "smoke" if a.smoke else a.tag
    jsonl = os.path.join(RESULTS_DIR, f"{tag}.jsonl")
    if not (a.fg or a.smoke):
        daemonize(os.path.join(RESULTS_DIR, "sweep.log"))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _M6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../src/models/Model_6
    sys.path.insert(0, _M6)
    # presynaptic_release lives in the PROJECT-ROOT sweep/, not Model_6's sweep/ (checked 2026-08-29)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_M6))), "sweep"))
    import logging; logging.disable(logging.INFO)
    import numpy as np

    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line); done.add((r["seed"], r["rewarded"]))
            except Exception:
                pass

    print(f"[{time.strftime('%H:%M:%S')}] f4_specificity start: n_syn={n_syn} n_driven={n_driven} "
          f"seeds={len(seeds)} arms={len(arms)} smoke={a.smoke} pid={os.getpid()}", flush=True)

    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        driven = sorted(rng.choice(n_syn, size=n_driven, replace=False).tolist())
        for rewarded in arms:
            if (seed, bool(rewarded)) in done:
                continue
            t0 = time.time()
            rec = run_one(n_syn, driven, rewarded, seed,
                          elig_s, delay_s, reward_s, settle_s, burst_s, a.drive_mv * 1e-3)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            dset = set(rec["driven"])
            cd = sum(1 for i in range(n_syn) if i in dset and rec["committed"][i])
            cu = sum(1 for i in range(n_syn) if i not in dset and rec["committed"][i])
            print(f"[{time.strftime('%H:%M:%S')}] seed={seed} reward={rewarded} driven={rec['driven']} "
                  f"eta_max={max(rec['eta']):.3f} E_inv_max={max(rec['e_invasion']):.3f} "
                  f"xbonds(F>0.5)={rec['n_cross_bonds_werner']}/{rec['n_cross_bonds_any']} "
                  f"tags={rec['tags']} committed_driven={cd}/{len(dset)} "
                  f"committed_undriven={cu}/{n_syn-len(dset)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] F4 COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
