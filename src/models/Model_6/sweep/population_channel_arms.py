#!/usr/bin/env python3
"""
T1'-6 — FOUR-ARM CHANNEL SEPARATION: does POPULATION LOSS, on its own, produce
far-pairs-first ordering?

THE CHARGE THIS ANSWERS (adversarial review of RESULTS_T1prime_far_pairs_first.md §6)
--------------------------------------------------------------------------------------
§6 defends the ordering with: "Dissolution is spatially uniform -- it lowers every pair's
effective radius equally... It cannot generate a consistent spacing-ordered cascade."

The alleged refutation, built from §4's OWN physics:
  1. An edge survives while ANY bonded pair clears F>0.5, so the governing radius is
     d*_eff = lambda*ln(max_pair(P_S^2)/0.5) -- an EXTREME-VALUE statistic.
  2. Bonded pairs scale ~N^2; population falls 2200 -> 98 (~500x fewer pairs).
  3. "The max of fewer draws is smaller" => d*_eff contracts from population loss ALONE.
  4. Any contracting radius crosses gaps in order of width (RESULTS §2).
  => dissolution is uniform in RATE but NOT order-neutral in EFFECT, and replication
     across seeds does not separate the two mechanisms.

Step 3 is the load-bearing step and it assumes RANDOM removal. This model does not remove
randomly (dimer_particles.py:230-241):

    elif difference < 0:
        n_to_remove = abs(difference)
        sorted_dimers = sorted(self.dimers, key=lambda d: d.coherence)
        for i in range(min(n_to_remove, len(sorted_dimers))):
            ... self.dimers.remove(dimer)

and `coherence` is a strictly increasing affine map of P_S (dimer_particles.py:57-62,
`(singlet_probability - 0.25)/0.75`). So removal is RANK-SELECTIVE: the LOWEST-P_S dimers
die first, re-sorted every step, and the argmax is the LAST thing removed. Under that rule
max_pair(P_S^2) is preserved, not eroded, by attrition.

That is an argument from code. This script MEASURES it, and measures the counterfactual --
what the criticism's world (random attrition) would have produced -- so the refutation is
quantified rather than asserted.

THE ARMS (pre-registered in RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md before this was run)
--------------------------------------------------------------------------------------
    arm      P_S decays?   population decays?   attrition rule     purpose
    A        yes           yes                  model (top-n)      reproduce the published result
    B        yes           NO                   --                 coherence-only channel
    C        NO (frozen)   yes                  model (top-n)      population-only channel
    D        NO (frozen)   NO                   --                 null; ordering here = broken rig
    A_rand   yes           yes                  RANDOM             the criticism's world, full
    C_rand   NO (frozen)   yes                  RANDOM             the criticism's world, isolated

A_rand/C_rand are not part of the pre-registered 4; they exist so the answer to "is the
N-dependence material?" is a NUMBER under the criticism's own assumption, not a dismissal.

WHAT IS REPLAYED, AND WHY IT IS LEGITIMATE (verified in code, not assumed)
--------------------------------------------------------------------------
Per dstar_eff_replay.py:12-22 -- T_singlet_eff is FIXED per dimer for life and P_S is an
independent per-dimer geometric random walk that nothing in the network touches. So the
P_S half of this is exact replay, and it is VALIDATED (median P_S matches the real 8 s rig
to <0.0005, dstar_eff_replay VALIDATION 1).

THE HONEST LIMITATION -- READ THIS BEFORE TRUSTING ANY ARM
-----------------------------------------------------------
The population trajectory N(t) is NOT replayed from physics. The raw run logs were
session-scoped scratchpad and are GONE (RESULTS §8 / review item 5.4). N(t) here is
INTERPOLATED THROUGH FIVE TRANSCRIBED SEED-0 ANCHORS recovered from research-log entry
L.T1'-4: n=2200 at t=0, and 1843 / 1043 / 259 / 98 at the four seed-0 break times
(14.5 / 32.5 / 61.5 / 78.0 s). Consequences, stated plainly:
  * the SAME trajectory is applied to all four seeds -- seeds 1-3 had their own, unrecorded;
  * it is log-linear between anchors, which is a choice, not a measurement;
  * therefore arms C/C_rand test the SHAPE of the population channel, not seed-specific
    timing. That is sufficient for the question asked (does the channel order AT ALL?)
    and insufficient for any claim about break TIMES. Times are not scored here either.

GUARDS (same class as the probe's, and they can FALSIFY)
--------------------------------------------------------
  CONSECUTIVE_ABSENT = 3   an edge must sit below its gap for 3 consecutive samples to
                           count as broken; transient dips are logged as FLICKERS.
  MIN_BREAKS = 3           fewer than 3 clean breaks => INCONCLUSIVE. A monotonicity test
                           over <3 points is vacuous -- this is the 683b82f failure.
  verdict returns CONFIRMED / FALSIFIED / INCONCLUSIVE. Arm D existing at all is the
  rig check: ordering in D means the harness is broken and everything else is void.

Run:  ./venv/bin/python src/models/Model_6/sweep/population_channel_arms.py
Emits per-sample traces (n_dimers, max_pair(P_S^2), d*_eff per pair) to
  src/models/Model_6/results/T1prime6_arms/
"""
import sys, os, time, types, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, d_star

DT = 0.001
P_THERMAL = 0.25
LOG_EVERY = 0.5
HORIZON_S = 90.0
SEEDS = [0, 1, 2, 3]

CONSECUTIVE_ABSENT = 3
MIN_BREAKS = 3

# The WIDE ladder actually run for T1'-4/-5 (RESULTS §4).
GAPS_DRIVE = [3.35, 4.5, 2.90, 4.5, 2.45, 4.5, 2.00]
LIVE_PAIRS = [((0, 1), 3.35), ((2, 3), 2.90), ((4, 5), 2.45), ((6, 7), 2.00)]
PRED_ORDER = [3.35, 2.90, 2.45, 2.00]   # the pre-registration: widest breaks FIRST

# Transcribed from research-log L.T1'-4 (seed 0). The ONLY surviving population data.
POP_ANCHORS = [(0.0, 2200.0), (14.5, 1843.0), (32.5, 1043.0), (61.5, 259.0), (78.0, 98.0)]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "T1prime6_arms")


# ---------------------------------------------------------------------------
# population trajectory
# ---------------------------------------------------------------------------
def pop_fraction(t):
    """Surviving fraction at time t: log-linear interpolation through the anchors."""
    ts = np.array([a[0] for a in POP_ANCHORS])
    ns = np.array([a[1] for a in POP_ANCHORS])
    logn = np.interp(t, ts, np.log(ns))
    if t > ts[-1]:
        # extrapolate the last log-slope rather than flat-lining
        slope = (np.log(ns[-1]) - np.log(ns[-2])) / (ts[-1] - ts[-2])
        logn = np.log(ns[-1]) + slope * (t - ts[-1])
    return float(np.exp(logn) / ns[0])


# ---------------------------------------------------------------------------
# population extraction (mirrors dstar_eff_replay.extract_population, per seed)
# ---------------------------------------------------------------------------
def extract_population(seed):
    np.random.seed(seed)
    net = build(ladder_positions(GAPS_DRIVE))
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
            T = max(216.0 / (spread_factor * template_factor), 0.1)
            P0.append(d.singlet_probability)
            T_eff.append(T)
            syn.append(si)
    return np.array(P0), np.array(T_eff), np.array(syn)


# ---------------------------------------------------------------------------
# the arm engine
# ---------------------------------------------------------------------------
def run_arm(arm, P0, T_eff, syn, seed, horizon_s=None, noise_offset=1):
    """Replay one arm. Returns list of per-sample dicts.

    noise_offset selects the P_S noise stream. It is a FREE CHOICE, not a property of the
    run: the real rig interleaves its draws with network stepping, so no replay reproduces
    a specific published realisation. Varying it measures how stable the ORDER is across
    noise draws — which is the honest question, since the published times are not
    reproducible by construction.
    """
    horizon_s = HORIZON_S if horizon_s is None else horizon_s
    decays_PS = arm in ("A", "B", "A_rand")
    attrition = arm in ("A", "C", "A_rand", "C_rand")
    random_rule = arm.endswith("_rand")

    np.random.seed(seed + noise_offset)
    n = len(P0)
    n_syn = int(syn.max()) + 1
    masks = [syn == i for i in range(n_syn)]
    n0 = np.array([int(m.sum()) for m in masks])

    # A fixed random death order per synapse = uniform random attrition (the criticism's
    # implicit rule). Drawn once so a dimer's fate does not resample every sample.
    rand_rank = np.zeros(n, dtype=int)
    for i, m in enumerate(masks):
        idx = np.nonzero(m)[0]
        perm = np.random.permutation(len(idx))
        rand_rank[idx] = perm

    decay = np.exp(-DT / T_eff)
    sig = 0.01 * np.sqrt(DT)
    P_S = P0.copy()
    P_excess = P_S - P_THERMAL

    n_steps = int(horizon_s / DT)
    every = int(LOG_EVERY / DT)
    rows = []

    for k in range(n_steps):
        if k % every == 0:
            t = k * DT
            frac = pop_fraction(t) if attrition else 1.0
            per_syn_max = np.zeros(n_syn)
            n_alive_tot = 0
            for i, m in enumerate(masks):
                keep = max(1, int(round(frac * n0[i])))
                vals = P_S[m]
                if keep >= len(vals):
                    surv = vals
                else:
                    if random_rule:
                        # keep the `keep` dimers earliest in the fixed random order
                        surv = vals[rand_rank[m] < keep]
                    else:
                        # THE MODEL'S RULE: keep the top-`keep` by current P_S
                        surv = np.partition(vals, len(vals) - keep)[len(vals) - keep:]
                per_syn_max[i] = surv.max()
                n_alive_tot += len(surv)

            row = {"t": t, "n_dimers": n_alive_tot}
            for (a, b), g in LIVE_PAIRS:
                pp = per_syn_max[a] * per_syn_max[b]
                row[f"maxpair_PS2_{a}{b}"] = pp
                row[f"dstar_eff_{a}{b}"] = d_star(pp)
            rows.append(row)

        if decays_PS:
            noise = 1.0 + sig * np.random.randn(n)
            P_S = np.clip(P_THERMAL + P_excess * decay * noise, P_THERMAL, 1.0)
            P_excess = P_S - P_THERMAL

    return rows


# ---------------------------------------------------------------------------
# break detection + verdict (same guards as the probe)
# ---------------------------------------------------------------------------
def detect_breaks(rows):
    """Return (breaks, flickers). A break needs CONSECUTIVE_ABSENT samples below gap."""
    absent_run = {g: 0 for _, g in LIVE_PAIRS}
    broken = {}
    flickers = []
    for row in rows:
        for (a, b), g in LIVE_PAIRS:
            if g in broken:
                continue
            below = row[f"dstar_eff_{a}{b}"] < g
            if below:
                absent_run[g] += 1
                if absent_run[g] >= CONSECUTIVE_ABSENT:
                    # break time = first sample of the confirmed absent run
                    broken[g] = row["t"] - (CONSECUTIVE_ABSENT - 1) * LOG_EVERY
            else:
                if absent_run[g] > 0:
                    flickers.append((g, row["t"], absent_run[g]))
                absent_run[g] = 0
    return broken, flickers


def verdict(broken):
    """CONFIRMED / FALSIFIED / INCONCLUSIVE. This function CAN return FALSIFIED."""
    ordered = sorted(broken.items(), key=lambda kv: kv[1])
    seq = [g for g, _ in ordered]
    if len(seq) < MIN_BREAKS:
        return "INCONCLUSIVE", seq
    expected = [g for g in PRED_ORDER if g in broken]
    return ("CONFIRMED" if seq == expected else "FALSIFIED"), seq


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
ARMS = ["A", "B", "C", "D", "A_rand", "C_rand"]
ARM_DESC = {
    "A": "P_S decays + population decays (model rule)  -- reproduce published result",
    "B": "P_S decays, population HELD              -- coherence-only channel",
    "C": "P_S FROZEN, population decays (model rule)-- population-only channel",
    "D": "P_S FROZEN, population HELD              -- NULL (ordering here = broken rig)",
    "A_rand": "P_S decays + RANDOM attrition        -- criticism's world, full",
    "C_rand": "P_S FROZEN + RANDOM attrition        -- criticism's world, isolated",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 100)
    print("T1'-6 FOUR-ARM CHANNEL SEPARATION — does population loss alone order the cascade?")
    print("=" * 100)
    print(f"ladder {GAPS_DRIVE}   pre-registered order {PRED_ORDER}")
    print(f"guards: CONSECUTIVE_ABSENT={CONSECUTIVE_ABSENT}  MIN_BREAKS={MIN_BREAKS}")
    print(f"population trajectory: log-linear through transcribed seed-0 anchors "
          f"{[(t, int(n)) for t, n in POP_ANCHORS]}")
    print(f"  -> applied to ALL seeds (seeds 1-3 trajectories were never persisted)")
    print()

    summary = []
    for seed in SEEDS:
        te = time.time()
        P0, T_eff, syn = extract_population(seed)
        print(f"--- seed {seed}: n={len(P0)} dimers, P_S(0) median {np.median(P0):.4f} "
              f"max {P0.max():.4f}   [{time.time()-te:.0f}s extract]")
        for arm in ARMS:
            rows = run_arm(arm, P0, T_eff, syn, seed)
            broken, flickers = detect_breaks(rows)
            status, seq = verdict(broken)
            summary.append({"seed": seed, "arm": arm, "status": status,
                            "order": seq, "breaks": dict(broken),
                            "n_end": rows[-1]["n_dimers"]})
            path = os.path.join(OUT_DIR, f"trace_seed{seed}_arm{arm}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            btxt = "  ".join(f"{g}:{broken[g]:.1f}s" if g in broken else f"{g}:--"
                             for g in PRED_ORDER)
            print(f"    {arm:7s} {status:13s} {btxt}   flickers={len(flickers)}")
        print()

    # ---------------- per-arm summary table
    print("=" * 100)
    print("PER-ARM SUMMARY (per seed)")
    print("=" * 100)
    print(f"{'arm':8s} {'seed':>4s}  {'verdict':13s}  "
          + "  ".join(f"{g:>6}" for g in PRED_ORDER))
    for arm in ARMS:
        for s in summary:
            if s["arm"] != arm:
                continue
            cells = "  ".join(
                f"{s['breaks'][g]:6.1f}" if g in s["breaks"] else "    --"
                for g in PRED_ORDER)
            print(f"{arm:8s} {s['seed']:>4d}  {s['status']:13s}  {cells}")
        print()

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "arm", "verdict", "order", "n_dimers_end"])
        for s in summary:
            w.writerow([s["seed"], s["arm"], s["status"], "|".join(map(str, s["order"])),
                        s["n_end"]])

    # ---------------- the quantified answer to the charge
    print("=" * 100)
    print("THE CHARGE, QUANTIFIED — d*_eff erosion from ATTRITION ALONE (P_S frozen)")
    print("=" * 100)
    print("Arm C vs C_rand at the observed population levels. Delta is measured against")
    print(f"the {PRED_ORDER[0] - PRED_ORDER[-1]:.2f} um span the cascade must traverse "
          f"({PRED_ORDER[0]} -> {PRED_ORDER[-1]}).")
    print()
    for seed in SEEDS[:1]:
        P0, T_eff, syn = extract_population(seed)
        rc = run_arm("C", P0, T_eff, syn, seed)
        rr = run_arm("C_rand", P0, T_eff, syn, seed)
        (a, b), g = LIVE_PAIRS[0]
        print(f"{'t(s)':>6} {'n_dimers':>9} {'d*_eff C':>10} {'d*_eff C_rand':>14} "
              f"{'delta_C':>9} {'delta_Crand':>12}")
        d0c = rc[0][f"dstar_eff_{a}{b}"]
        d0r = rr[0][f"dstar_eff_{a}{b}"]
        for t_s in [0, 15, 30, 45, 60, 75, 89]:
            i = int(round(t_s / LOG_EVERY))
            if i >= len(rc):
                continue
            dc = rc[i][f"dstar_eff_{a}{b}"]
            dr = rr[i][f"dstar_eff_{a}{b}"]
            print(f"{rc[i]['t']:6.1f} {rc[i]['n_dimers']:9d} {dc:10.3f} {dr:14.3f} "
                  f"{dc-d0c:+9.3f} {dr-d0r:+12.3f}")
    # ---------------- horizon + noise-draw stability control
    print()
    print("=" * 100)
    print("ORDER STABILITY CONTROL — 90 s vs 200 s horizon, and across noise draws")
    print("=" * 100)
    print("WHY: the 90 s arms above truncate (order_power_probe.py:47 uses HORIZON_S=200)")
    print("and use a different noise stream than either the real run or the power probe.")
    print("Neither is reproducible by construction, so the honest question is how stable")
    print("the ORDER is across draws. Arm B (coherence-only, no attrition) is the cleanest")
    print("channel for this. Compare against the published 10/10 power for this ladder.")
    print()
    n_ok = n_fals = n_inc = 0
    for seed in SEEDS:
        P0, T_eff, syn = extract_population(seed)
        cells = []
        for off in range(1, 11):
            rows = run_arm("B", P0, T_eff, syn, seed, horizon_s=200.0, noise_offset=off)
            broken, _ = detect_breaks(rows)
            status, _ = verdict(broken)
            cells.append({"CONFIRMED": "C", "FALSIFIED": "F",
                          "INCONCLUSIVE": "i"}[status])
            n_ok += status == "CONFIRMED"
            n_fals += status == "FALSIFIED"
            n_inc += status == "INCONCLUSIVE"
        print(f"  seed {seed}: " + " ".join(cells)
              + f"    ({cells.count('C')}/10 order-correct)")
    tot = n_ok + n_fals + n_inc
    print(f"\n  TOTAL across {tot} draws: {n_ok} CONFIRMED / {n_fals} FALSIFIED / "
          f"{n_inc} INCONCLUSIVE   => order recovered {100*n_ok/tot:.0f}%")
    print("  (C=order correct, F=order violated, i=<3 clean breaks in 200 s)")

    print()
    print(f"traces + summary written to {OUT_DIR}")
    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
