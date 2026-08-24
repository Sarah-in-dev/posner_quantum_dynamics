#!/usr/bin/env python3
"""
F3 DELAY SWEEP — the temporal-credit CURVE under the CORRECTED mechanism (grounded GluN2B structural-latch
memory + coherence-gated readout). This SUPERSEDES the F3-b/F3-c numbers, which were obtained on the
now-removed credit-bypass code path.

THE QUESTION: as the gap between the eligibility event and the reward widens, does the coherent P_S tag keep
assigning credit where the classical trace cannot — and does destroying the coherence (⁷Li) remove it?

ARMS (identical protocol; only the eligibility carrier / readout rule differs)
  quantum-undoped : the coherent ³¹P tag (T2≈216 s ⇒ readable until P_S crosses the Werner floor ~107 s)
  classical-base  : biology's measured fixed 0.3–2 s eligibility window (Yagishita/Shindou) — the baseline
                    the quantum tag must BEAT at long delays
  quantum-Li7     : the ISOTOPE LEVER — ⁷Li collapses T2 216→~14 s, so the tag decoheres within seconds.
                    Same chemistry, one nuclear-spin degree of freedom changed (the program's one real
                    attribution handle: F1→F2→F3).

DELAYS: 1, 2, 5, 10, 30, 60 s. Prediction (pre-registered here, before the run): quantum-undoped credits
across ALL of these (all inside the ~107 s coherence window); classical dies past ~2 s; ⁷Li dies within a few
seconds. If quantum-undoped ALSO dies at long delay, that is a real negative and gets reported as one.

SCORING: commit PROBABILITY per (delay, arm) — commitment is genuinely stochastic (DDSC; Jain 2024) — with a
two-sided permutation null on the per-seed binary outcomes at each delay. Nothing is tuned; a null is a result.

OPERATIONS: self-daemonizes (double-fork + setsid) so it survives terminal/agent teardown, thread-caps itself,
and appends one JSON line per run so partial results survive and the run is resumable (re-running skips cells
already present in the JSONL).
"""
import argparse
import json
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", os.environ.get("F3_SWEEP_DIR", "f3_delay_sweep"))
DELAYS = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
ARMS = [("quantum-undoped", "quantum", None),
        ("classical-base", "classical", None),
        ("quantum-Li7", "quantum", "Li7"),
        # SUBSTRATE-NECESSITY CONTROL (2026-08-24): identical to quantum-undoped except the eligibility is a
        # DETERMINISTIC exponential with the same time constant instead of the emergent dimer-population P_S.
        # If this arm reproduces quantum-undoped, the computational primitive needs only a float and a decay
        # constant -- i.e. it is implementable on ordinary silicon and the substrate is required for the
        # BIOLOGY, not for the computation. A separation would be the first evidence the substrate does
        # irreducible computational work. This is the cheap discriminator for the (A)-vs-(B) question.
        ("classical-slow", "classical_slow", None)]

DT = 5e-3
BUILD_S, REWARD_S, SETTLE_S = 0.5, 20.0, 10.0
DA_TONIC, DA_BURST = 20e-9, 10e-6


def daemonize(log_path):
    """Double-fork + setsid so the batch survives terminal/agent teardown (macOS has no `setsid` binary)."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())


def one_run(mode, dopant, delay, seed):
    import numpy as np
    from model6_parameters import Model6Parameters
    from model6_core import Model6QuantumSynapse

    np.random.seed(seed)
    p = Model6Parameters(); p.em_coupling_enabled = True
    if dopant is not None:
        p.environment.dopant = dopant
    s = Model6QuantumSynapse(p)
    s._network_controlled = True
    s._reward_gated_consolidation = True
    s._reward_gating_mode = mode

    for _ in range(int(BUILD_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -40e-3})
    s._measurement_gate_opened = True
    s._measurement_time = s.time
    ps_tag = float(getattr(s, "_mean_singlet_prob", float("nan")))
    n_dim = len(s.dimer_particles.dimers)

    for _ in range(int(delay / DT)):                       # the GAP: tag decoheres passively, CaMKII resets
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    ps_reward = float(getattr(s, "_mean_singlet_prob", float("nan")))
    pre_commit = bool(s._camkii_committed)

    for _ in range(int(REWARD_S / DT)):                    # the delayed reward
        s._da_signal = DA_BURST
        s.step(DT, {"voltage": -70e-3})
    for _ in range(int(SETTLE_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})

    return dict(mode=mode, dopant=(dopant or "none"), delay=delay, seed=seed,
                n_dimers=n_dim, ps_tag=ps_tag, ps_reward=ps_reward,
                pre_commit=pre_commit, committed=bool(s._camkii_committed),
                readout_ca_uM=float(getattr(s, "_readout_ca_uM", 0.0)),
                mem=float(s.camkii.molecular_memory))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="seeds per cell")
    ap.add_argument("--fg", action="store_true", help="run in foreground (no daemonize)")
    a = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    jsonl = os.path.join(RESULTS_DIR, "runs.jsonl")
    log = os.path.join(RESULTS_DIR, "sweep.log")
    if not a.fg:
        daemonize(log)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import logging; logging.disable(logging.INFO)

    done = set()
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["mode"], r["dopant"], r["delay"], r["seed"]))
                except Exception:
                    pass

    total = len(DELAYS) * len(ARMS) * a.n
    print(f"[{time.strftime('%H:%M:%S')}] f3_delay_sweep start: {total} runs "
          f"({len(DELAYS)} delays x {len(ARMS)} arms x {a.n} seeds); {len(done)} already done; pid={os.getpid()}",
          flush=True)

    for delay in DELAYS:
        for name, mode, dopant in ARMS:
            for seed in range(a.n):
                key = (mode, dopant or "none", delay, seed)
                if key in done:
                    continue
                t0 = time.time()
                rec = one_run(mode, dopant, delay, seed)
                rec["arm"] = name
                with open(jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{time.strftime('%H:%M:%S')}] {name:>16} delay={delay:>5.1f} seed={seed} "
                      f"commit={rec['committed']} P_S@rew={rec['ps_reward']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
