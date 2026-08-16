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


def run_one(n_syn, driven, rewarded, seed, elig_s, delay_s, reward_s, settle_s, burst_s):
    """One network draw. `driven` = indices driven during the eligibility window. Returns per-synapse state."""
    import numpy as np
    net = build_network(n_syn, seed)

    def phase(duration, drive_idx, da):
        n = int(round(duration / DT))
        for _ in range(n):
            per_syn = [{"voltage": (DRIVE_V if i in drive_idx else REST_V)} for i in range(n_syn)]
            for s in net.synapses:
                s._da_signal = da
            net.step(DT, {"per_synapse": per_syn, "reward": False})

    # 1. LOCAL eligibility: only the driven synapses build a coherent tag
    phase(elig_s, set(driven), DA_TONIC)
    tags = [len(s.dimer_particles.dimers) for s in net.synapses]
    ps_tag = [float(getattr(s, "_mean_singlet_prob", float("nan"))) for s in net.synapses]

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
        net.step(DT, {"per_synapse": [{"voltage": REST_V}] * n_syn, "reward": False})

    # 4. settle
    phase(settle_s, set(), DA_TONIC)

    return dict(
        seed=seed, rewarded=bool(rewarded), driven=sorted(list(driven)),
        n_syn=n_syn, tags=tags, ps_tag=ps_tag, pre_commit=pre_commit,
        committed=[bool(getattr(s, "_camkii_committed", False)) for s in net.synapses],
        mem=[float(s.camkii.molecular_memory) for s in net.synapses],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="seeds per arm")
    ap.add_argument("--n-syn", type=int, default=8)
    ap.add_argument("--n-driven", type=int, default=3)
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
        seeds, arms = list(range(a.n)), [True, False]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "smoke" if a.smoke else "runs"
    jsonl = os.path.join(RESULTS_DIR, f"{tag}.jsonl")
    if not (a.fg or a.smoke):
        daemonize(os.path.join(RESULTS_DIR, "sweep.log"))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                          elig_s, delay_s, reward_s, settle_s, burst_s)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            dset = set(rec["driven"])
            cd = sum(1 for i in range(n_syn) if i in dset and rec["committed"][i])
            cu = sum(1 for i in range(n_syn) if i not in dset and rec["committed"][i])
            print(f"[{time.strftime('%H:%M:%S')}] seed={seed} reward={rewarded} driven={rec['driven']} "
                  f"tags={rec['tags']} committed_driven={cd}/{len(dset)} "
                  f"committed_undriven={cu}/{n_syn-len(dset)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] F4 COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
