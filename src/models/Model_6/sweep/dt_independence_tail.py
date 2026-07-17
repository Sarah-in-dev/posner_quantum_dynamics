#!/usr/bin/env python3
"""
IS THE TAIL dt-INDEPENDENT? — the question that decides an 8 h run vs a 48 min run.

WHY THIS IS NOT ALREADY ANSWERED
--------------------------------
The handoff records a dt-independence control (that is how the dissolution-dt bug was
caught): "same 4 s of sim => dimers = 56/619/1501 at dt=1e-3/1e-2/5e-2, while P_S and the
Werner edge set were dt-INDEPENDENT." But that was measured on the MEDIAN, at t=4 s,
before the dissolution fix. Edge survival is governed by the MAX (the tail), and the max
sits pinned against the P_S clip at 1.0 (dimer_particles.py:329). A multiplicative random
walk against a BOUNDARY is not generally dt-safe: larger dt = larger individual kicks =
different clipping statistics. So the tail could be dt-sensitive where the median is not.

Measure it. Do not infer it.

WHAT PASSES
-----------
If max P_S (and hence d*_eff) agrees across dt within the between-seed scatter we already
measured (~0.29 um in d*_eff), then dt=0.01 is safe for the coherence half and the real
run drops ~10x. If the tail drifts with dt, dt=1e-3 stands and the run costs 8 h.

SCOPE: this tests step_coherence only — which is the half that governs d*_eff. Bond
formation/dissolution use the EXACT exponential form (p = 1-exp(-k*dt),
multi_synapse_network.py:341-344), so they compose dt-exactly for constant rates. A full
dt control on the real rig would still be required before trusting a dt=0.01 production
run; this is the cheap screen that says whether that is even worth doing.
"""
import sys, os, time, types

# Run from the sweep/ directory (repo convention). sweep/ is auto-added as the
# script dir; dirname(dirname(__file__)) adds Model_6 for its modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, d_star

P_THERMAL = 0.25
HORIZON_S = 90.0
LOG_EVERY = 0.5
DTS = [0.001, 0.005, 0.01, 0.05]
GAPS = [3.35, 4.5, 2.90, 4.5, 2.45, 4.5, 2.00]   # the WIDE ladder
LIVE = [((0, 1), 3.35), ((2, 3), 2.90), ((4, 5), 2.45), ((6, 7), 2.00)]
N_SEEDS = 3


def extract(seed):
    np.random.seed(seed)
    net = build(ladder_positions(GAPS))
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    for k in range(int(0.08 / 0.001)):
        t = k * 0.001
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(0.001, {"voltage": v, "reward": False})
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


def replay(P0, T_eff, syn, dt, seed):
    """Exactly dimer_particles.step_coherence, at the given dt."""
    np.random.seed(seed + 777)
    n_syn = int(syn.max()) + 1
    decay = np.exp(-dt / T_eff)
    sig = 0.01 * np.sqrt(dt)
    P_S = P0.copy()
    P_excess = P_S - P_THERMAL
    masks = [syn == i for i in range(n_syn)]
    every = max(1, int(round(LOG_EVERY / dt)))
    ts, meds, maxs, per_syn = [], [], [], []
    for k in range(int(HORIZON_S / dt)):
        if k % every == 0:
            ts.append(k * dt); meds.append(np.median(P_S)); maxs.append(P_S.max())
            per_syn.append([P_S[m].max() for m in masks])
        noise = 1.0 + sig * np.random.randn(len(P0))
        P_S = np.clip(P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0)
        P_excess = P_S - P_THERMAL
    return np.array(ts), np.array(meds), np.array(maxs), np.array(per_syn)


def breaks(ts, per_syn):
    out = []
    for (a, b), g in LIVE:
        de = np.array([d_star(per_syn[k, a] * per_syn[k, b]) for k in range(len(ts))])
        below = np.where(de < g)[0]
        out.append(float(ts[below[0]]) if len(below) else None)
    return out


def main():
    t0 = time.time()
    print("=== dt-INDEPENDENCE OF THE TAIL (step_coherence) — WIDE ladder, 90 s ===")
    print("The median is known dt-independent. The MAX governs edge survival and sits")
    print("against the clip at 1.0. That is what is tested here.\n")

    for seed in range(N_SEEDS):
        P0, T_eff, syn = extract(seed)
        print(f"--- seed {seed}  (n={len(P0)} dimers) ---")
        print(f"{'dt':>7} {'t=10 med':>9} {'t=10 max':>9} {'t=45 med':>9} {'t=45 max':>9} "
              f"{'t=89 med':>9} {'t=89 max':>9}   break times (s)")
        rows = []
        for dt in DTS:
            ts, meds, maxs, per_syn = replay(P0, T_eff, syn, dt, seed)
            def at(t_s):
                i = int(round(t_s / LOG_EVERY))
                i = min(i, len(ts) - 1)
                return meds[i], maxs[i]
            m10, x10 = at(10); m45, x45 = at(45); m89, x89 = at(89)
            bt = breaks(ts, per_syn)
            rows.append((dt, x10, x45, x89, bt))
            btxt = " ".join((f"{b:5.1f}" if b is not None else " >90") for b in bt)
            print(f"{dt:7.3f} {m10:9.4f} {x10:9.4f} {m45:9.4f} {x45:9.4f} "
                  f"{m89:9.4f} {x89:9.4f}   {btxt}")

        # Compare every dt against the dt=1e-3 reference.
        ref = rows[0]
        print(f"{'dt':>7} {'d(max@10)':>10} {'d(max@45)':>10} {'d(max@89)':>10} "
              f"{'d(d*_eff@45)um':>14}  order same?")
        for dt, x10, x45, x89, bt in rows:
            dd = d_star(x45 * x45) - d_star(ref[3 - 1] * ref[3 - 1])
            same = all((b is None) == (r is None) for b, r in zip(bt, ref[4]))
            if same:
                same = all(b is None or abs(b - r) <= 10.0
                           for b, r in zip(bt, ref[4]) if r is not None)
            print(f"{dt:7.3f} {x10-ref[1]:+10.4f} {x45-ref[2]:+10.4f} "
                  f"{x89-ref[3]:+10.4f} {dd:+14.3f}  {'yes' if same else 'NO'}")
        print()

    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
