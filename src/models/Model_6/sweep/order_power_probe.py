#!/usr/bin/env python3
"""
IS THE T1' ORDER TEST POWERED? — measure it before spending 4.2 h/seed.

THE WORRY (from the validated d*_eff replay, 2026-07-17)
--------------------------------------------------------
Each synapse's effective radius d*_eff is set by ITS OWN luckiest dimer. T_eff is FIXED
per dimer at creation (dimer_particles.py:47-50, j_couplings_intra drawn once and never
reassigned), so each synapse's tail is a FROZEN random variable — it does not average out
over time within a run. Measured at t=30 s on the ladder: per-pair d*_eff = 3.13 / 3.02 /
3.15 / 3.31, a spread of 0.29 um across pairs whose designed gaps span only 0.30 um.

If the between-pair scatter is comparable to the designed gap separation, then which edge
breaks first is substantially LUCK, and a single-seed order result means nothing — the
same vacuity that made the 2026-07-16 verdict worthless, arriving by a different door.

WHAT THIS MEASURES
------------------
Across N independent seeds (fresh dimers => fresh frozen T_eff per synapse), replay each
population and record the per-pair break order. Report how often the pre-registered order
(widest gap first) comes out. That fraction IS the test's power.

  ~1.0  => the order test is deterministic; one seed suffices.
  ~0.5  => partial power; needs replication, and the seed count is computable from this.
  ~1/24 => (chance for 4 items) the test is a coin flip and the geometry must change.

Uses the same UPPER-BOUND d*_eff as the replay (max over A x max over B, fixed
population). The absolute times are bounds, but the BETWEEN-PAIR SCATTER is a property of
the validated per-dimer physics, so the power estimate is informative.

Also sweeps a WIDER geometry: if 0.1 um rungs are unresolvable, how wide must they be?
"""
import sys, os, time, types, math
from itertools import permutations

# Run from the sweep/ directory (repo convention). sweep/ is auto-added as the
# script dir; dirname(dirname(__file__)) adds Model_6 for its modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, d_star

DT = 0.001
P_THERMAL = 0.25
HORIZON_S = 200.0
LOG_EVERY = 0.5
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))

# Candidate ladders: (name, live gaps widest-first). Pair-isolated by 4.5um dark controls.
CANDIDATES = [
    ("tight  (0.10um rungs)", [3.35, 3.25, 3.15, 3.05]),
    ("medium (0.25um rungs)", [3.35, 3.10, 2.85, 2.60]),
    ("wide   (0.45um rungs)", [3.35, 2.90, 2.45, 2.00]),
]


def gaps_for(live):
    g = []
    for i, x in enumerate(live):
        g.append(x)
        if i < len(live) - 1:
            g.append(4.5)
    return g


def extract(seed, gaps):
    np.random.seed(seed)
    net = build(ladder_positions(gaps))
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    for k in range(int(0.08 / DT)):
        t = k * DT
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(DT, {"voltage": v, "reward": False})
    P0, T_eff, syn = [], [], []
    for si, s in enumerate(net.synapses):
        for d in s.dimer_particles.dimers:
            j_spread = np.std(d.j_couplings_intra)
            j_mean = np.abs(np.mean(d.j_couplings_intra))
            spread_factor = 1.0 + 2.0 * j_spread / (j_mean + 0.1)
            template_factor = 0.7 if d.template_bound else 1.0
            P0.append(d.singlet_probability)
            T_eff.append(max(216.0 / (spread_factor * template_factor), 0.1))
            syn.append(si)
    return np.array(P0), np.array(T_eff), np.array(syn)


def replay_pairmax(P0, T_eff, syn, seed, horizon_s):
    np.random.seed(seed + 10000)
    n = len(P0)
    n_syn = int(syn.max()) + 1
    decay = np.exp(-DT / T_eff)
    sig = 0.01 * np.sqrt(DT)
    P_S = P0.copy()
    P_excess = P_S - P_THERMAL
    every = int(LOG_EVERY / DT)
    masks = [syn == i for i in range(n_syn)]
    ts, per_syn = [], []
    for k in range(int(horizon_s / DT)):
        if k % every == 0:
            ts.append(k * DT)
            per_syn.append([P_S[m].max() for m in masks])
        noise = 1.0 + sig * np.random.randn(n)
        P_S = np.clip(P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0)
        P_excess = P_S - P_THERMAL
    return np.array(ts), np.array(per_syn)


def break_times(ts, per_syn, live):
    """Break time per live pair. Pair i uses synapses (2i, 2i+1)."""
    out = []
    for i, g in enumerate(live):
        a, b = 2 * i, 2 * i + 1
        de = np.array([d_star(per_syn[k, a] * per_syn[k, b]) for k in range(len(ts))])
        below = np.where(de < g)[0]
        out.append(float(ts[below[0]]) if len(below) else None)
    return out


def main():
    t0 = time.time()
    print(f"=== T1' ORDER-TEST POWER — {N_SEEDS} seeds per geometry ===")
    print("Pre-registered order = widest gap breaks first. Power = fraction of seeds "
          "that produce it.")
    print(f"Chance level for 4 rungs = 1/24 = {1/24:.3f}\n")

    for name, live in CANDIDATES:
        gaps = gaps_for(live)
        rows, oks, worst = [], 0, 0.0
        for s in range(N_SEEDS):
            P0, T_eff, syn = extract(s, gaps)
            ts, per_syn = replay_pairmax(P0, T_eff, syn, s, HORIZON_S)
            tb = break_times(ts, per_syn, live)
            if any(x is None for x in tb):
                rows.append((s, tb, None))
                continue
            # order correct iff break times increase as gaps narrow (widest first)
            ok = all(tb[i] < tb[i + 1] for i in range(len(tb) - 1))
            oks += ok
            worst = max(worst, max(tb))
            rows.append((s, tb, ok))
        n_scored = sum(1 for _, _, ok in rows if ok is not None)
        power = oks / n_scored if n_scored else float("nan")

        print(f"--- {name}   live gaps {live} ---")
        print(f"{'seed':>5}  " + "  ".join(f"g{g:<5.2f}" for g in live) + "   order")
        for s, tb, ok in rows:
            cells = "  ".join((f"{x:6.1f}" if x is not None else "  >200") for x in tb)
            print(f"{s:5d}  {cells}   {'OK' if ok else ('--' if ok is None else 'WRONG')}")
        print(f"POWER = {oks}/{n_scored} = {power:.2f}   "
              f"(chance {1/24:.3f})   window needed <= {worst:.1f}s "
              f"=> ~{worst*321/3600:.1f} h/seed")
        # Seeds needed so that observing ~power*k successes beats the 1/24 null at p<0.05.
        # Binomial tail: P(X >= ceil(power*k) | p=1/24) < 0.05.
        if 0 < power < 1:
            k = 1
            while k < 60:
                need = int(np.ceil(power * k))
                tail = sum(math.comb(k, j) * (1 / 24) ** j * (1 - 1 / 24) ** (k - j)
                           for j in range(need, k + 1))
                if tail < 0.05:
                    break
                k += 1
            print(f"  ~{k} seeds to beat the 1/24 null at p<0.05 at this power "
                  f"=> ~{k*worst*321/3600:.0f} h total")
        print()

    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
