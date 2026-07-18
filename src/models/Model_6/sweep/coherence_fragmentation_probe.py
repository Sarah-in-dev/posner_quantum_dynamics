#!/usr/bin/env python3
"""
COHERENCE-FRAGMENTATION probe (T1', dynamic half) — does the partition fragment
FAR-PAIRS-FIRST as coherence decays?

REDESIGNED 2026-07-17. The 2026-07-16 run FAILED and printed a FALSE POSITIVE.
Read docs/handoffs/SESSION_HANDOFF_JUL16_TOPOLOGY_DT_FIX.md §1/§3 before touching
this file. What changed here, and why:

  1. THE GEOMETRY (the root error). The old ladder's 2.0-3.0um gaps forced the
     cascade out to t~34-110 s, where BOTH confounds bite: the dimer population is
     collapsing AND the step_coherence noise walk dominates. The gaps are now placed
     JUST UNDER the MEASURED d*(0) so all four cross threshold in the first seconds,
     while the population is still >90% alive.
  2. THE TIMES ARE GONE. The old PRED dict pre-registered break times. They were
     mis-derived THREE times (median 9.5 s -> p95 ~13 s -> "analytic ceiling" 19.3 s)
     against an OBSERVED 34.0 s. They are an extreme-value statistic over a noisy
     random walk across hundreds of pairs and are NOT analytically predictable here.
     SCORE THE ORDER ONLY. Do not re-add a time prediction.
  3. THE VERDICT IS GUARDED. The old one passed vacuously off a single flicker.

THE CLAIM (from coherence_radius_probe.py, static half CONFIRMED 7/7)
---------------------------------------------------------------------
Bond iff d < d* = 5*ln(P_product/0.5). As P_S decays, d* SHRINKS, so edges must
break in GAP ORDER — largest gap first. This is the DISCRIMINATING claim: a
classical scalar eligibility trace decays uniformly and carries no spatial
structure, so it CANNOT produce a spacing-ordered cascade. Order is also robust to
which quantile of P_S governs, which is exactly why it survives when times do not.

Coherence decay is Agarwal-2023 physics already in the code
(dimer_particles.step_coherence): P_S(t) = 0.25 + (P_S(0)-0.25)*exp(-t/T_eff),
T_eff = 216s / (spread_factor * template_factor), plus a MULTIPLICATIVE noise term
(:322) that compounds to ~+/-5.8% on P_excess over 34k steps. P_S decays toward
0.25 — below the 1/sqrt(2)=0.7071 floor — so the partition is guaranteed to die.
The question is only whether it dies in the predicted ORDER.

THE GEOMETRY — WIDE ladder, chosen for STATISTICAL POWER (2026-07-17)
--------------------------------------------------------------------
d*(0) measured in this rig (t=0.08s, n=2300): P_S median 0.9987 (min 0.9922, max 1.0),
so d*(0) = 3.45um (min 3.39, max 3.47). A first redesign put the four live gaps JUST
UNDER d*(0) (3.35/3.25/3.15/3.05) so the cascade would "land early". An 8 s sanity run
and a validated d*_eff replay KILLED that plan, for a reason worth stating so it is not
re-attempted:

  - An edge survives while ANY bonded pair clears F>0.5, so the governing radius is
    d*_eff = 5*ln(max_pair(P_S^2)/0.5) — the extreme TAIL, not the median. The tail
    decays ~3.6x SLOWER than the median, so tight gaps do NOT break early: at t=7.5s the
    median radius was below all four gaps while all four edges were still alive.
  - Worse, each synapse's tail is set by ITS OWN luckiest dimer, whose T_eff is frozen
    at creation. That between-synapse scatter is ~0.29um in d*_eff. Rungs spaced 0.10um
    are FINER than the noise => the break order is decided by luck. Measured power (10
    seeds): tight 0.10um rungs 6/10, medium 0.25um 5/10 — barely better than chance.

So the geometry is chosen to make the ORDER RESOLVABLE, i.e. rungs WIDER than the
between-synapse scatter — accepting a later cascade as the price:

    gap 3.35um  P_S_crit 0.9885   (widest live  -> must break FIRST)
    gap 2.90um  P_S_crit 0.9450
    gap 2.45um  P_S_crit 0.9034
    gap 2.00um  P_S_crit 0.8637   (tightest live -> must break LAST)
    (P_S_crit = sqrt(0.5*exp(gap/5)) — the coherence at which that gap loses its bond.
     CORRECTED 2026-07-17: the first-committed values for 2.90/2.45/2.00 were wrong
     (0.9327/0.8801/0.8305). Annotation only — the code never reads them; it computes
     d* at runtime from measured P_S. The 4/4 result is unaffected.)
    gap 4.50um  P_S_crit 1.1090   DARK CONTROL x3: >1, can NEVER bond at any P_S.
                                  Retrodicts the static probe (both 4.5 gaps dark).

Measured power of THIS geometry (10 seeds, d*_eff replay): 10/10 correct order. The
0.45um rungs clear the ~0.29um scatter decisively. Cost: the last break lands by ~90s
(an upper bound; dissolution pulls it earlier), so --seconds 90 for the full cascade.

Live gaps are pair-ISOLATED by the dark controls, so each live edge is its own
component and each break is independent and unambiguous: betti0 4 -> 5 -> 6 -> 7 -> 8.
In 1D connectivity is decided entirely by CONSECUTIVE gaps; every non-consecutive
distance here is >= 7.35um, far outside d*, so no non-consecutive pair can rescue a
broken link.

ACCEPT  = >= MIN_BREAKS clean breaks, in monotone gap order (widest first).
FALSIFY = the order is not monotone in gap, or fragmentation is UNIFORM (all edges
          break together). Report the negative; do NOT adjust d*, T_eff, or the
          Werner bound. 0.5 is a separability THEOREM, not a knob.
INCONCLUSIVE = fewer than MIN_BREAKS clean breaks. This is a real outcome, not a
          failure to try harder. If it fires, re-tune the GEOMETRY, never the physics.

THE CONFOUND, and the control
------------------------------
The partition could fragment for a SECOND reason: dimers DISSOLVE, and
_update_entanglement prunes bonds whose dimers are gone. Fewer dimers => fewer
cross-pair chances => quotient edges vanish. That is NOT the coherence claim.
So n_dimers is logged as a control, and the two mechanisms are separable by SHAPE:
  - coherence  => SHARP, synchronised loss of every edge at a given gap at once,
                  at a specific P_S (all pairs at that gap cross F=0.5 together).
  - dimer loss => SMOOTH erosion tracking the exponential dimer decay.
If n_dimers collapses, the run is confounded and the ORDER is the only signal.
NOTE on the wide geometry: the cascade now lands LATER (~to 90s), so it overlaps the
population decay more than a tight ladder would. That is an accepted trade for a
resolvable order (see the geometry section). Uniform dissolution lowers every pair's
radius equally, so it PRESERVES the gap order; only total population collapse destroys
it. n_dimers is the control that tells the two apart — watch it, do not assume.

THE FLOOR PREDICTION IS DEMOTED — reported if reached, NEVER scored. Dissolution is
coherence-PROTECTED (k_diss = k_classical*(1-singlet_excess)*template_enhancement,
singlet_excess(t) = exp(-t/T) exactly), so it ACCELERATES as coherence decays; only
~1.9% of the population survives to t=75 s. "Partition gone" there cannot be
separated from "dimers ran out".

eta is CLAMPED at 0.26 throughout — including during silence. That is the
CONTROL, not a cheat: it holds the pump fixed so P_S is the ONLY variable that
moves. If eta were allowed to fall with the drive, fragmentation would be
confounded by the eta gate. (Bonds keep FORMING under clamped eta, but formation
cannot rescue an edge: the Werner cut is applied at READ time in
_find_all_clusters, and F is refreshed every step from the current P_S.)

PROTOCOL: drive 0.08 s (theta bursts, forms the partition) -> silence --seconds.
Silence matters: under sustained drive new dimers are born at P_S=1 and the
population never decoheres.

USAGE:
    python coherence_fragmentation_probe.py --seconds 8            # sanity window
    python coherence_fragmentation_probe.py --seconds 40 --seeds 0 1 2
One run of a stochastic cascade is an anecdote. The order is scored PER SEED.
"""
import sys, os, time, types, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, d_star
from entanglement_topology import compute_synapse_quotient_betti

# THE LADDER — WIDE, chosen for statistical power (2026-07-17; see the docstring).
# 0.45um rungs clear the ~0.29um between-synapse d*_eff scatter => measured 10/10 order
# power (tight 0.10um rungs were only 6/10). Each live gap is pair-ISOLATED by a 4.5um
# dark control so its break is independent. NOTE: deliberately NOT imported from
# coherence_radius_probe — that module's GAPS are the CONFIRMED 7/7 static geometry and
# must not move.
DARK = 4.5
LIVE_GAPS = [3.35, 2.90, 2.45, 2.00]           # widest first == predicted break order
GAPS = [3.35, DARK, 2.90, DARK, 2.45, DARK, 2.00]

# PRE-REGISTERED, and this is the WHOLE prediction: the live edges break in
# descending gap order — 3.35 first, then 3.25, 3.15, 3.05. NO TIMES. The times are an
# extreme-value statistic over a noisy random walk across hundreds of pairs; they were
# mis-derived three times and are not analytically predictable here (handoff §3).
# Do NOT re-introduce a time prediction. Do NOT score anything but the order.
PRED_ORDER = sorted(LIVE_GAPS, reverse=True)

# Expected at t=0 from the geometry alone: 4 live edges, each its own component.
PRED_EDGES_T0 = [(0, 1), (2, 3), (4, 5), (6, 7)]
PRED_B0_T0 = 4

DRIVE_S, DT = 0.08, 0.001
LOG_EVERY = 0.5

# DT is overridable via --dt for the dt-independence CONTROL. Both the coherence decay
# (dimer_particles.py:319 exp(-dt/T)) and the bond form/dissolve draws
# (multi_synapse_network.py:341-344, p = 1-exp(-k*dt)) use the EXACT exponential form, so
# they compose dt-exactly for a constant rate. That is an argument, not a measurement —
# and the dissolution-dt bug (fd83460) is exactly what happens when the argument is
# trusted. Run the control before trusting any dt but 0.001.

# An edge must be absent this many consecutive samples to count as broken (flicker
# guard), and we need at least this many clean breaks before an order claim means
# anything. Both learned the hard way on 2026-07-16 — see the handoff.
CONSECUTIVE_ABSENT = 3
MIN_BREAKS = 3


def observed_edges(tr):
    gid_syn = {d["global_id"]: d["synapse_idx"] for d in tr.all_dimers}
    obs = set()
    for (i, j), f in tr.cross_synapse_bonds.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                obs.add((min(a, b), max(a, b)))
    return obs


def run_one(seed, seconds, dt=DT):
    """One cascade at one seed. Returns the per-seed record; scores the ORDER only."""
    t0 = time.time()
    np.random.seed(seed)
    pos = ladder_positions(GAPS)
    net = build(pos)
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    tr = net.entanglement_tracker

    print(f"\n=== SEED {seed} — silence window {seconds}s ===")

    # --- Phase 1: drive
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    for k in range(int(DRIVE_S / dt)):
        t = k * dt
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(dt, {"voltage": v, "reward": False})

    tr.collect_dimers(net.synapses, net.positions)
    e0 = observed_edges(tr)
    print(f"after drive: edges={sorted(e0)}  dimers={len(tr.all_dimers)}")

    gap_of = {(i, i + 1): GAPS[i] for i in range(len(GAPS))}

    # Did the geometry actually deliver the edges it was designed to? An edge that
    # never FORMS is not a break — it is a rung missing from the ladder, and the old
    # probe could not see it (alive was seeded from e0, so a no-show vanished
    # silently). Report it: it means the gap sat above the population's d*(0).
    missing = [g for e, g in gap_of.items() if g in LIVE_GAPS and e not in e0]
    spurious = [g for e, g in gap_of.items() if g == DARK and e in e0]
    if missing:
        print(f"  !! live gaps that FAILED TO FORM at t=0: {sorted(missing, reverse=True)} "
              f"— geometry too wide vs measured d*(0); these can never be scored as breaks")
    if spurious:
        print(f"  !! DARK CONTROL BONDED: {spurious} — impossible under d*<=3.47um. "
              f"The distance rule itself is in question; stop and investigate.")
    if not missing and not spurious:
        print(f"  t=0 geometry OK: all {len(LIVE_GAPS)} live edges formed, "
              f"all {GAPS.count(DARK)} dark controls dark")

    # --- Phase 2: silence
    alive = {e: True for e in e0}
    broke_at = {}
    absent_run, first_absent, flickers = {}, {}, {}
    all_gone_at = None

    print(f"{'t(s)':>7} {'P_S_med':>8} {'d*_med':>7} {'P_S_max':>8} {'d*_EFF':>7} "
          f"{'dimers':>7} {'b0':>3} {'b1':>3} {'edges':>6}  lost")
    next_log = 0.0
    n_steps = int(seconds / dt)
    for k in range(n_steps):
        t = k * dt
        net.step(dt, {"voltage": -70e-3, "reward": False})
        if t < next_log:
            continue
        next_log += LOG_EVERY

        tr.collect_dimers(net.synapses, net.positions)
        if not tr.all_dimers:
            print(f"{t:7.1f}  (all dimers gone)")
            break
        # Log the MEDIAN and the MAX. The median is decorative; the MAX is the one that
        # governs. An edge survives while ANY bonded pair clears F>0.5, so the effective
        # radius is d*(max P_S ^2), not d*(median^2). The 8 s sanity run (2026-07-17)
        # measured the median radius falling BELOW all four live gaps while all four
        # edges stayed alive — the median radius is not the instrument. Log both so the
        # divergence is visible instead of inferred.
        psv = np.array([d["P_S"] for d in tr.all_dimers])
        ps = float(np.median(psv))
        ps_max = float(psv.max())
        ds = d_star(ps * ps)
        ds_eff = d_star(ps_max * ps_max)
        obs = observed_edges(tr)
        q = compute_synapse_quotient_betti(
            tr.all_dimers, tr.cross_synapse_bonds,
            werner_bound=tr.WERNER_ENTANGLEMENT_BOUND)

        newly = []
        # FLICKER GUARD (2026-07-17). An edge is only "broken" after it has been
        # absent for CONSECUTIVE_ABSENT samples. The first version marked the FIRST
        # absence as a permanent break and never un-marked it — and the 2026-07-16 run
        # proved that wrong: gap 3.0 vanished at t=34.0 s and was BACK at t=34.5 s
        # (edges 4 -> 5). Bonds keep forming under clamped eta and the step_coherence
        # noise walk lets P_S wander back over threshold, so edges flicker near d*.
        # A flicker recorded as a break is fabricated data.
        for e in list(alive):
            if not alive[e]:
                continue
            if e not in obs:
                absent_run[e] = absent_run.get(e, 0) + 1
                if absent_run[e] == 1:
                    first_absent[e] = t
                if absent_run[e] >= CONSECUTIVE_ABSENT:
                    alive[e] = False
                    broke_at[e] = first_absent[e]
                    newly.append(f"gap{gap_of.get(e,'?')}@{first_absent[e]:.1f}s")
            elif absent_run.get(e, 0):
                flickers[e] = flickers.get(e, 0) + 1
                newly.append(f"(flicker gap{gap_of.get(e,'?')}@{t:.1f}s)")
                absent_run[e] = 0
        if all_gone_at is None and len(obs) == 0:
            all_gone_at = t

        print(f"{t:7.1f} {ps:8.4f} {ds:7.2f} {ps_max:8.4f} {ds_eff:7.2f} "
              f"{len(tr.all_dimers):7d} {q.betti0:3d} {q.betti1:3d} {len(obs):6d}  "
              f"{' '.join(newly)}")

        # Stop only once every edge that formed is CONFIRMED broken (survived the
        # flicker guard). NOT when the edge count first hits zero — a zero sample can
        # be a flicker, and stopping there would discard the last break.
        if alive and not any(alive.values()):
            print(f"\ncascade complete at t={t:.1f}s — all {len(broke_at)} live edges "
                  f"confirmed broken")
            break

    # --- Verdict: ORDER ONLY. No times are scored. broke_at is printed for the record.
    print(f"\n{'gap':>6} {'broke at':>10}   (times RECORDED, never scored — handoff §3)")
    rows = []
    for e, br in sorted(broke_at.items(), key=lambda kv: kv[1]):
        g = gap_of.get(e)
        rows.append((g, br))
        print(f"{g:6.2f} {br:9.1f}s")
    if flickers:
        print("\nFLICKERS (edge vanished then returned — NOT breaks): " +
              ", ".join(f"gap{gap_of.get(e)}x{n}" for e, n in flickers.items()))
    if all_gone_at is not None:
        print(f"(edge count first touched zero at t={all_gone_at:.1f}s)")

    # A monotonicity check over <MIN_BREAKS points is VACUOUS: with one break
    # `all(...)` is trivially True. The 2026-07-16 run printed "CONFIRMED" off a
    # single flicker exactly that way. Refuse to pass on too little data.
    order_ok = all(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1))
    print(f"\nbreak ORDER monotone in gap (widest first)? {order_ok}  "
          f"({len(rows)} break(s); need >= {MIN_BREAKS} for this to mean anything)")
    if len(rows) < MIN_BREAKS:
        status = "INCONCLUSIVE"
        verdict = (f"VERDICT[seed {seed}]: INCONCLUSIVE — only {len(rows)} clean "
                   f"break(s). A monotonicity test needs >= {MIN_BREAKS}. NOT a "
                   f"confirmation; do NOT report the order as tested. Widen the "
                   f"window, or re-tune the GEOMETRY (never the physics).")
    elif order_ok:
        status = "CONFIRMED"
        verdict = (f"VERDICT[seed {seed}]: far-pairs-first order CONFIRMED over "
                   f"{len(rows)} breaks")
    else:
        status = "FALSIFIED"
        verdict = (f"VERDICT[seed {seed}]: order FALSIFIED — read the table, "
                   f"report the negative")
    print(f"[{time.time()-t0:.0f}s] {verdict}")
    return {"seed": seed, "status": status, "order_ok": order_ok, "rows": rows,
            "n_breaks": len(rows), "missing": missing, "spurious": spurious}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="silence window. Start SMALL (sanity), size the real run from it.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one cascade per seed. One run of a stochastic cascade is an "
                         "anecdote; the order is scored PER SEED.")
    ap.add_argument("--dt", type=float, default=DT,
                    help="integration step. 0.001 is the validated default. Larger dt is "
                         "~linearly faster; run the dt-independence CONTROL before "
                         "trusting one (see the note by DT).")
    args = ap.parse_args()

    t0 = time.time()
    print("=== COHERENCE-FRAGMENTATION PROBE (T1', dynamic half) — REDESIGNED 2026-07-17 ===")
    print(f"ladder gaps (um): {GAPS}   (live {LIVE_GAPS}, dark {DARK} x{GAPS.count(DARK)})")
    print(f"ladder x   (um): {np.round(ladder_positions(GAPS)[:,0], 2).tolist()}")
    print(f"PRE-REGISTERED ORDER (the whole prediction): {PRED_ORDER}  — widest first")
    print("PRE-REGISTERED TIMES: none. They are not analytically predictable here "
          "(handoff §3). Times are recorded, never scored.")

    if args.dt != DT:
        print(f"!! dt={args.dt} (default {DT}). CONTROL REQUIRED: compare P_S, n_dimers "
              f"and the edge set against a dt={DT} run before trusting any result.")

    results = [run_one(s, args.seconds, dt=args.dt) for s in args.seeds]

    print(f"\n{'='*70}\nSUMMARY over {len(results)} seed(s), window {args.seconds}s, "
          f"dt {args.dt}")
    print(f"{'seed':>6} {'status':>13} {'breaks':>7}  order")
    for r in results:
        order = " > ".join(f"{g:.2f}" for g, _ in r["rows"]) or "-"
        print(f"{r['seed']:6d} {r['status']:>13} {r['n_breaks']:7d}  {order}")
    n_conf = sum(r["status"] == "CONFIRMED" for r in results)
    n_fals = sum(r["status"] == "FALSIFIED" for r in results)
    n_inc = sum(r["status"] == "INCONCLUSIVE" for r in results)
    print(f"\nCONFIRMED {n_conf} / FALSIFIED {n_fals} / INCONCLUSIVE {n_inc}")
    if n_inc == len(results):
        print("ALL SEEDS INCONCLUSIVE — the window is too short or the geometry is "
              "wrong. This is a real outcome. Re-tune the GEOMETRY; never the physics.")
    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
