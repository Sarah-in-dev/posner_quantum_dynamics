#!/usr/bin/env python3
"""
COHERENCE-FRAGMENTATION probe (T1', dynamic half) — does the partition fragment
FAR-PAIRS-FIRST as coherence decays, and die at the P_S = 1/sqrt(2) floor?

THE CLAIM (from coherence_radius_probe.py, static half CONFIRMED 7/7)
---------------------------------------------------------------------
Bond iff d < d* = 5*ln(P_product/0.5). As P_S decays, d* SHRINKS, so edges must
break in GAP ORDER — largest gap first — and the whole cross-synapse partition
must vanish at the hard floor P_S = 1/sqrt(2) = 0.7071 (since w<=1 => F<=P_S^2).

Coherence decay is Agarwal-2023 physics already in the code
(dimer_particles.step_coherence): P_S(t) = 0.25 + (P_S(0)-0.25)*exp(-t/T_eff),
T_eff = 216s / (spread_factor * template_factor). MEASURED in this rig:
T_eff median = 151.6 s. P_S decays toward 0.25 — well BELOW the 0.7071 floor —
so the partition is guaranteed to die; the only question is whether it dies in
the predicted ORDER at the predicted TIMES.

PRE-REGISTERED (computed before the run, T=151.6s, P_S(0)~0.9987)
-----------------------------------------------------------------
An edge at gap g breaks when P_S falls below sqrt(0.5*e^(g/5)):

    gap 3.0um  P_S_crit 0.9545  ->  t =  9.5 s   (breaks FIRST — the widest)
    gap 2.8um  P_S_crit 0.9356  ->  t = 13.6 s
    gap 2.5um  P_S_crit 0.9079  ->  t = 19.9 s
    gap 2.0um  P_S_crit 0.8637  ->  t = 30.4 s   (breaks LAST — the tightest)
    floor      P_S_crit 0.7071  ->  t = 75.1 s   partition GONE entirely
    gap 4.5um  P_S_crit 1.1090  ->  never bonds even at P_S=1 (retrodicts the
                                    static probe, where both 4.5 gaps were dark)

ACCEPT  = edges break in that ORDER, at roughly those times, and betti0_cross
          reaches 0 near t~75s.
FALSIFY = fragmentation is UNIFORM (all edges break together), or the order is
          not monotone in gap, or the partition survives past the 0.7071 floor.
          Report the negative; do NOT adjust d*, T_eff, or the Werner bound.

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

eta is CLAMPED at 0.26 throughout — including during silence. That is the
CONTROL, not a cheat: it holds the pump fixed so P_S is the ONLY variable that
moves. If eta were allowed to fall with the drive, fragmentation would be
confounded by the eta gate. (Bonds keep FORMING under clamped eta, but formation
cannot rescue an edge: the Werner cut is applied at READ time in
_find_all_clusters, and F is refreshed every step from the current P_S.)

PROTOCOL: drive 0.08 s (theta bursts, forms the partition) -> silence 80 s.
Silence matters: under sustained drive new dimers are born at P_S=1 and the
population never decoheres.
"""
import sys, os, time, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, GAPS, d_star
from entanglement_topology import compute_synapse_quotient_betti

PRED = {3.0: 9.5, 2.8: 13.6, 2.5: 19.9, 2.0: 30.4}
PRED_FLOOR = 75.1
DRIVE_S, SILENCE_S, DT = 0.08, 80.0, 0.001
LOG_EVERY = 0.5


def observed_edges(tr):
    gid_syn = {d["global_id"]: d["synapse_idx"] for d in tr.all_dimers}
    obs = set()
    for (i, j), f in tr.cross_synapse_bonds.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                obs.add((min(a, b), max(a, b)))
    return obs


def main():
    t0 = time.time()
    pos = ladder_positions(GAPS)
    net = build(pos)
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    tr = net.entanglement_tracker

    print("=== COHERENCE-FRAGMENTATION PROBE (T1', dynamic half) ===")
    print(f"ladder gaps (um): {GAPS}")
    print("PRE-REGISTERED break times: " +
          "  ".join(f"gap{g}->{t}s" for g, t in sorted(PRED.items(), reverse=True)))
    print(f"PRE-REGISTERED floor (partition gone): t={PRED_FLOOR}s\n")

    # --- Phase 1: drive
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    for k in range(int(DRIVE_S / DT)):
        t = k * DT
        ph = t % theta_period
        v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
        net.step(DT, {"voltage": v, "reward": False})

    tr.collect_dimers(net.synapses, net.positions)
    e0 = observed_edges(tr)
    print(f"after drive: edges={sorted(e0)}  dimers={len(tr.all_dimers)}\n")

    # --- Phase 2: silence
    gap_of = {(i, i + 1): GAPS[i] for i in range(len(GAPS))}
    alive = {e: True for e in e0}
    broke_at = {}
    floor_at = None

    print(f"{'t(s)':>7} {'P_S':>8} {'d*(um)':>7} {'dimers':>7} {'b0':>3} {'b1':>3} "
          f"{'edges':>6}  lost")
    next_log = 0.0
    n_steps = int(SILENCE_S / DT)
    for k in range(n_steps):
        t = k * DT
        net.step(DT, {"voltage": -70e-3, "reward": False})
        if t < next_log:
            continue
        next_log += LOG_EVERY

        tr.collect_dimers(net.synapses, net.positions)
        if not tr.all_dimers:
            print(f"{t:7.1f}  (all dimers gone)")
            break
        ps = float(np.median([d["P_S"] for d in tr.all_dimers]))
        ds = d_star(ps * ps)
        obs = observed_edges(tr)
        q = compute_synapse_quotient_betti(
            tr.all_dimers, tr.cross_synapse_bonds,
            werner_bound=tr.WERNER_ENTANGLEMENT_BOUND)

        newly = []
        for e in list(alive):
            if alive[e] and e not in obs:
                alive[e] = False
                broke_at[e] = t
                newly.append(f"gap{gap_of.get(e,'?')}@{t:.1f}s")
        if floor_at is None and len(obs) == 0:
            floor_at = t

        print(f"{t:7.1f} {ps:8.4f} {ds:7.2f} {len(tr.all_dimers):7d} "
              f"{q.betti0:3d} {q.betti1:3d} {len(obs):6d}  {' '.join(newly)}")
        if floor_at is not None:
            print(f"\npartition GONE at t={floor_at:.1f}s "
                  f"(pre-registered {PRED_FLOOR}s)")
            break

    # --- Verdict
    print(f"\n{'gap':>6} {'predicted':>10} {'observed':>10}")
    rows = []
    for e, br in sorted(broke_at.items(), key=lambda kv: kv[1]):
        g = gap_of.get(e)
        p = PRED.get(g)
        rows.append((g, p, br))
        print(f"{g:6.1f} {(f'{p}s' if p else '-'):>10} {br:9.1f}s")
    order_ok = all(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1))
    print(f"\nbreak ORDER monotone in gap (widest first)? {order_ok}")
    print(f"[{time.time()-t0:.0f}s] "
          + ("VERDICT: far-pairs-first fragmentation CONFIRMED"
             if order_ok and rows else
             "VERDICT: order NOT confirmed — read the table, report honestly"))


if __name__ == "__main__":
    main()
