#!/usr/bin/env python3
"""
F3 COHERENCE CONTROLS — is the reward-signed readout genuinely COHERENCE-GATED?

Every arm receives an IDENTICAL delayed dopamine reward at 10 s. The only thing that differs is whether the
tag is still coherent when the reward arrives, and which readout rule is in force:

  A. undoped, quantum   -> tag still coherent at 10 s (P_S > Werner 0.707)  => should credit
  B. Li6,     quantum   -> dopant effect negligible (F1)                     => should credit (~A)
  C. Li7,     quantum   -> T2 216 -> ~14 s, tag DECOHERED by 10 s            => should NOT credit
  D. undoped, classical -> biology's fixed 0.3-2 s window, closed by 10 s    => should NOT credit

C is the isotope lever (F1->F2->F3, the program's one real attribution handle); D is the temporal-gap
contrast (the classical trace is dead where the coherent tag still credits).

SCORING — a PROBABILITY contrast, not a deterministic threshold. Commitment here is genuinely stochastic:
DDSC is "dendritic, delayed and STOCHASTIC CaMKII activation" (Jain 2024, Nature), so demanding that every
seed commit is a category error (an earlier version of this script did exactly that and mis-reported a
working result as FAIL). We score the COMMIT PROBABILITY of the coherent arms (A+B) against the decohered /
classical arms (C+D) with a two-sided PERMUTATION null over per-seed binary outcomes — the same
decode-vs-null discipline the other keystones in this repo use. A null is a result; nothing is tuned.

DA_BURST_S selects the reward protocol (env var):
  20.0 (default) = reward signal sustained through the DDSC calcium event -> the regime that produces
  0.5            = a brief phasic burst -> a STRUCTURAL negative (the readout Ca drives PP2B, which strips
                   Thr34 and re-engages DAPK1 before the complex can form). Reported, not tuned away.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.INFO)
from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse

DT = 5e-3
BUILD_S, DELAY_S, REWARD_S, SETTLE_S = 0.5, 10.0, 20.0, 10.0
DA_BURST_S = float(os.environ.get("DA_BURST_S", "20.0"))
DA_TONIC, DA_BURST = 20e-9, 10e-6

ARMS = [("undoped", "quantum", None, "coherent"), ("Li6", "quantum", "Li6", "coherent"),
        ("Li7", "quantum", "Li7", "decohered"), ("classical-base", "classical", None, "decohered")]


def run(mode, dopant, seed):
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
    ps_tag = float(getattr(s, "_mean_singlet_prob", np.nan))

    for _ in range(int(DELAY_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    ps_reward = float(getattr(s, "_mean_singlet_prob", np.nan))

    nburst = int(DA_BURST_S / DT)
    for k in range(int(REWARD_S / DT)):
        s._da_signal = DA_BURST if k < nburst else DA_TONIC
        s.step(DT, {"voltage": -70e-3})
    for _ in range(int(SETTLE_S / DT)):
        s._da_signal = DA_TONIC
        s.step(DT, {"voltage": -70e-3})

    return dict(ps_tag=ps_tag, ps_reward=ps_reward, committed=bool(s._camkii_committed),
                mem=float(s.camkii.molecular_memory),
                readout_ca=float(getattr(s, "_readout_ca_uM", 0.0)))


def perm_p(a, b, n_perm=20000, seed=1):
    """Two-sided permutation null on the difference of commit rates."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


if __name__ == "__main__":
    NS = int(os.environ.get("NS", "8"))
    WERNER = 1.0 / np.sqrt(2)
    print("=" * 98)
    print("F3 COHERENCE CONTROLS — identical delayed reward in every arm; only tag coherence / readout rule differ")
    print(f"Werner floor = {WERNER:.3f} | DA_BURST_S = {DA_BURST_S} s | n = {NS} seeds/arm")
    print("=" * 98)
    print(f"  {'arm':>15} | {'P_S@tag':>8}{'P_S@rew':>9} | {'readout_Ca':>11} | {'commit rate':>11} | {'final_mem':>9}")
    print("-" * 98)
    outcomes = {"coherent": [], "decohered": []}
    for name, mode, dop, group in ARMS:
        rs = [run(mode, dop, sd) for sd in range(NS)]
        c = [1.0 if r["committed"] else 0.0 for r in rs]
        outcomes[group].extend(c)
        print(f"  {name:>15} | {np.mean([r['ps_tag'] for r in rs]):>8.3f}"
              f"{np.mean([r['ps_reward'] for r in rs]):>9.3f} | "
              f"{np.mean([r['readout_ca'] for r in rs]):>11.2f} | {np.mean(c):>11.2f} | "
              f"{np.mean([r['mem'] for r in rs]):>9.3f}")
    print("-" * 98)

    coh, dec = outcomes["coherent"], outcomes["decohered"]
    p = perm_p(coh, dec)
    print(f"\n  COMMIT PROBABILITY   coherent  (undoped+Li6,   n={len(coh)}) = {np.mean(coh):.3f}")
    print(f"                       decohered (Li7+classical, n={len(dec)}) = {np.mean(dec):.3f}")
    print(f"  contrast = {np.mean(coh) - np.mean(dec):+.3f}   two-sided permutation p = {p:.4f}")
    credible = (np.mean(coh) > np.mean(dec)) and p < 0.05
    print(f"\n  => {'COHERENCE-GATED readout SUPPORTED (probability contrast, p<0.05)' if credible else 'NOT supported at this protocol — report it; do not tune'}")
    if np.mean(coh) == 0.0:
        print("     NB: nothing committed in ANY arm — this protocol does not commit at all "
              "(see the brief-burst structural finding); the contrast is vacuous, not a refutation.")
