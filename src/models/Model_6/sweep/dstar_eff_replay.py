#!/usr/bin/env python3
"""
d*_eff(t) REPLAY — measure the radius that actually governs edge survival, cheaply.

WHY THIS EXISTS
---------------
The 8 s sanity run (2026-07-17) measured the MEDIAN radius falling below all four live
gaps while all four edges stayed alive. Edge survival is a MAX over bonded pairs, so the
governing radius is d*_eff = 5*ln(max_pair(P_S_i*P_S_j)/0.5), and it decays >=3.6x slower
than the median radius. Every geometry choice is currently a guess about d*_eff(t).

WHY IT NEEDS NO NETWORK (verified in the code, not assumed)
-----------------------------------------------------------
  dimer_particles.py:47-50   j_couplings_intra drawn ONCE in __post_init__, never reassigned
  dimer_particles.py:205-211 template_bound set ONCE at creation, never reassigned
  dimer_particles.py:296-297 the external J field is STORED (dimer.local_j_coupling =
                             j_external) and never used in the P_S update
=> T_singlet_eff is FIXED per dimer for its whole life, and P_S is an independent
   geometric random walk per dimer:
       P_S = clip(0.25 + P_excess*exp(-dt/T_i)*(1 + 0.01*sqrt(dt)*randn()), 0.25, 1.0)
   Nothing in the network touches it. So net.step (321 s per sim-second, ~all of it the
   cross-synapse bond loop) is not needed to get the P_S distribution. Replay it in numpy.

THIS IS A BOUND, NOT A PREDICTION — and the direction is the whole point
------------------------------------------------------------------------
Three facts, each pushing the SAME way:
  * The replay holds the population FIXED; the real run dissolves dimers. Removing
    dimers can only LOWER a max  =>  replay max >= true max.
  * An edge needs a BONDED pair; not all cross pairs are bonded. Max over all pairs
    >= max over bonded pairs  =>  same direction.
  * d*_eff for the pair (A,B) uses max over A's dimers x max over B's dimers, NOT the
    global max over all 2223. Per-synapse maxes are computed below for this reason.
    (v1 of this script used the global max — an error; it inflates d*_eff further.)

  =>  replay d*_eff >= true d*_eff  =>  HIGHER radius = edges survive LONGER
  =>  REPLAY BREAK TIMES ARE **UPPER BOUNDS** ON REAL BREAK TIMES.

That is the useful direction for SIZING A WINDOW: a window covering the replay's last
break is GUARANTEED to cover the real cascade. It is the wrong direction for claiming a
break WILL happen by time t — the bound cannot promise that. Size with it; never score
with it.

THE ONE VALIDATION AVAILABLE — and the one that is NOT
-----------------------------------------------------
  1. VALID: the MEDIAN P_S trajectory from today's 8 s run on the real rig, every 0.5 s.
     This tests the replayed physics (the per-dimer walk) against the running code.
  2. NOT AVAILABLE: there is NO valid cascade-level datum to validate d*_eff against.
     v1 of this script used "gap 3.0 broke at t=34.0 s" (2026-07-16) as ground truth.
     THAT IS THE FLICKER — the fabricated data point the 2026-07-16 session was retracted
     over (handoff §1: "That one 'break' was a FLICKER, not a break ... fabricated data").
     Using it as a target was the same reach-for-a-number error this program keeps paying
     for. It is recorded here so nobody re-adds it.
     Consequence: the P_S physics is validated; the P_S -> d*_eff mapping is NOT. That is
     exactly why this script emits a BOUND and not a prediction.
"""
import sys, os, time, types

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
SEED = 0

# VALIDATION 1 — measured on the REAL 8-synapse rig, 2026-07-17 sanity run, seed 0.
MEASURED_MEDIAN = {
    0.0: 0.9986, 0.5: 0.9944, 1.0: 0.9912, 1.5: 0.9882, 2.0: 0.9856, 2.5: 0.9831,
    3.0: 0.9801, 3.5: 0.9776, 4.0: 0.9750, 4.5: 0.9725, 5.0: 0.9703, 5.5: 0.9682,
    6.0: 0.9657, 6.5: 0.9630, 7.0: 0.9607, 7.5: 0.9587,
}
GAPS_DRIVE = [3.35, 4.5, 3.25, 4.5, 3.15, 4.5, 3.05]
LIVE_PAIRS = [((0, 1), 3.35), ((2, 3), 3.25), ((4, 5), 3.15), ((6, 7), 3.05)]


def extract_population():
    """Drive the real rig 0.08 s, then read each dimer's FIXED coherence parameters."""
    np.random.seed(SEED)
    net = build(ladder_positions(GAPS_DRIVE))
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    for k in range(int(0.08 / DT)):
        t = k * DT
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(DT, {"voltage": v, "reward": False})

    P0, T_eff, tmpl, syn = [], [], [], []
    for si, s in enumerate(net.synapses):
        for d in s.dimer_particles.dimers:
            # Mirrors dimer_particles.py:304-315 exactly.
            j_spread = np.std(d.j_couplings_intra)
            j_mean = np.abs(np.mean(d.j_couplings_intra))
            spread_factor = 1.0 + 2.0 * j_spread / (j_mean + 0.1)
            template_factor = 0.7 if d.template_bound else 1.0
            T = max(216.0 / (spread_factor * template_factor), 0.1)
            P0.append(d.singlet_probability)
            T_eff.append(T)
            tmpl.append(bool(d.template_bound))
            syn.append(si)
    return (np.array(P0), np.array(T_eff), np.array(tmpl), np.array(syn))


def replay(P0, T_eff, syn, horizon_s):
    """Vectorised replay of step_coherence.

    Returns (times, median P_S, global max P_S, per-synapse max P_S [n_log, n_syn]).
    Per-synapse because d*_eff for pair (A,B) is set by max over A x max over B — NOT
    by the global max over every dimer in the network.
    """
    np.random.seed(SEED + 1)
    n = len(P0)
    n_syn = int(syn.max()) + 1
    decay = np.exp(-DT / T_eff)               # per-dimer, constant for life
    sig = 0.01 * np.sqrt(DT)
    P_S = P0.copy()
    P_excess = P_S - P_THERMAL
    n_steps = int(horizon_s / DT)
    every = int(LOG_EVERY / DT)
    masks = [syn == i for i in range(n_syn)]
    ts, meds, maxs, per_syn = [], [], [], []
    for k in range(n_steps):
        if k % every == 0:
            ts.append(k * DT); meds.append(np.median(P_S)); maxs.append(P_S.max())
            per_syn.append([P_S[m].max() for m in masks])
        noise = 1.0 + sig * np.random.randn(n)
        # clip feeds back into the next P_excess — exactly as dimer_particles.py:326-329
        P_S = np.clip(P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0)
        P_excess = P_S - P_THERMAL
    return np.array(ts), np.array(meds), np.array(maxs), np.array(per_syn)


def first_crossing(ts, ds_eff, gap):
    """First time d*_eff falls below gap (i.e. the replay's break time for that gap)."""
    below = np.where(ds_eff < gap)[0]
    return float(ts[below[0]]) if len(below) else None


def main():
    t0 = time.time()
    print("=== d*_eff(t) REPLAY — emits an UPPER BOUND on break times, not a prediction ===")
    P0, T_eff, tmpl, syn = extract_population()
    print(f"population extracted from the real rig: n={len(P0)} dimers over "
          f"{syn.max()+1} synapses  [{time.time()-t0:.0f}s]")
    print(f"  P_S(0):  median {np.median(P0):.4f}  max {P0.max():.4f}  min {P0.min():.4f}")
    print(f"  T_eff:   median {np.median(T_eff):.1f}s  max {T_eff.max():.1f}s  "
          f"min {T_eff.min():.1f}s   (analytic ceiling 216/0.7 = 308.6s)")
    print(f"  template_bound: {100*tmpl.mean():.1f}%")

    t1 = time.time()
    ts, meds, maxs, per_syn = replay(P0, T_eff, syn, HORIZON_S)
    print(f"replayed {HORIZON_S}s in {time.time()-t1:.1f}s of compute "
          f"({int(HORIZON_S/DT)} steps x {len(P0)} dimers)")

    ds_med = np.array([d_star(m * m) for m in meds])

    # ---------------- VALIDATION 1 (the only valid one): median trajectory
    print("\n--- VALIDATION 1: median P_S vs the REAL rig (8 s sanity run, seed 0) ---")
    print("Tests the replayed per-dimer physics against the running code.")
    print(f"{'t(s)':>6} {'measured':>10} {'replay':>10} {'diff':>9}")
    errs = []
    for t_m, ps_m in sorted(MEASURED_MEDIAN.items()):
        i = int(round(t_m / LOG_EVERY))
        if i < len(meds):
            d = meds[i] - ps_m
            errs.append(abs(d))
            print(f"{t_m:6.1f} {ps_m:10.4f} {meds[i]:10.4f} {d:+9.4f}")
    max_err = max(errs) if errs else float("nan")
    v1 = max_err < 0.005
    print(f"max |diff| = {max_err:.4f}   VALIDATION 1: {'PASS' if v1 else 'FAIL'} "
          f"(threshold 0.005)")

    print("\n--- VALIDATION 2: NOT AVAILABLE ---")
    print("There is no valid cascade-level datum. The only candidate — 'gap 3.0 broke at")
    print("t=34.0s' (2026-07-16) — is the FLICKER that session was retracted over")
    print("(handoff §1: 'fabricated data'). So the P_S physics is validated; the")
    print("P_S -> d*_eff mapping is NOT. Hence: a bound, never a prediction.")

    # ---------------- the curve, per live pair
    print(f"\n--- d*_eff(t) PER LIVE PAIR (max over A x max over B), vs the median radius ---")
    hdr = "  ".join(f"pair{a}{b}(g{g})" for (a, b), g in LIVE_PAIRS)
    print(f"{'t(s)':>6} {'d*_med':>7}   {hdr}")
    for t_s in [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 200]:
        i = int(round(t_s / LOG_EVERY))
        if i >= len(ts):
            continue
        cells = []
        for (a, b), g in LIVE_PAIRS:
            de = d_star(per_syn[i, a] * per_syn[i, b])
            cells.append(f"{de:12.2f}")
        print(f"{ts[i]:6.1f} {ds_med[i]:7.2f}   " + "  ".join(cells))

    # ---------------- the answer to "how long must the window be"
    print(f"\n--- BREAK-TIME UPPER BOUNDS for this ladder (per-pair d*_eff) ---")
    print("Real breaks land AT OR BEFORE these times. A window covering the last one is")
    print("GUARANTEED to cover the whole cascade. These are NOT predictions and are NOT")
    print("scored — the pre-registration is the ORDER only.")
    print(f"{'pair':>8} {'gap':>6} {'t_break <=':>11}")
    worst = 0.0
    for (a, b), g in LIVE_PAIRS:
        de = np.array([d_star(per_syn[i, a] * per_syn[i, b]) for i in range(len(ts))])
        tb = first_crossing(ts, de, g)
        if tb is not None:
            worst = max(worst, tb)
        print(f"{str((a,b)):>8} {g:6.2f} "
              f"{(f'{tb:.1f}s' if tb is not None else f'>{HORIZON_S:.0f}s'):>11}")
    if worst:
        print(f"\n=> WINDOW SUFFICIENT FOR ALL FOUR: {worst:.1f}s")
        print(f"   at the MEASURED 321 s wall per sim-second => "
              f"~{worst*321/3600:.1f} h per seed")

    print(f"\nVERDICT: per-dimer physics {'VALIDATED' if v1 else 'NOT VALIDATED'}; "
          f"d*_eff mapping UNVALIDATED by construction => use for SIZING ONLY.")
    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
