#!/usr/bin/env python3
"""
IS dt=1e-3 CONVERGED? — drive-only dt sweep. ~10 min.

WHY THIS SUDDENLY MATTERS
-------------------------
The dt=0.01 control (2026-07-17) FAILED, and not in a small way:

    dt=1e-3  ->  2223 dimers after the 0.08 s drive
    dt=1e-2  -> 10072 dimers after the SAME 0.08 s drive      (4.5x MORE)
    plus: RuntimeWarning: overflow encountered in exp
          ca_triphosphate_complex.py:296
          dimer_fraction = 1.0 / (1.0 + np.exp(10 * (ca_p_ratio - 0.5)))

So the dimer CHEMISTRY is dt-sensitive. The exact-exponential argument (p = 1-exp(-k*dt))
covers step_coherence and the bond form/dissolve draws — it does NOT cover
ca_triphosphate_complex, which integrates a stiff nonlinear system explicitly.

That kills the 10x speedup. But it raises a bigger question that nobody has asked:
**is dt=1e-3 itself converged?** The handoff's pre-fix control showed dimers =
56/619/1501 at dt=1e-3/1e-2/5e-2 — monotonically increasing with dt. The dissolution-dt
fix (fd83460) was supposed to remove that. Post-fix we still see 2223 -> 10072 from
1e-3 -> 1e-2. The dt-sensitivity SURVIVED the fix.

If the count is still moving between 1e-4 and 1e-3, then the operating point everything
downstream rests on (the ~155 uM post-fix figure, and D8/D14/D16/D17) is not converged —
it is an artefact of the step size. That is a physics-conclusions question and it is
Sarah's call, but the MEASUREMENT is cheap and nobody has made it.

WHAT THIS RUNS
--------------
Drive only (0.08 s, the same theta burst as every probe), across dt. Reports the dimer
count and the P_S/d* the ladder geometry depends on. Drive-only is cheap because the
population is still growing from zero.

PASS  = dimer count flat from 1e-4 -> 1e-3 (converged; dt=1e-3 is a safe operating point)
FAIL  = still climbing (dt=1e-3 is NOT converged; the operating point is step-size
        dependent and every count-based conclusion downstream inherits that)
"""
import sys, os, time, types, warnings

# Run from the sweep/ directory (repo convention). sweep/ is auto-added as the
# script dir; dirname(dirname(__file__)) adds Model_6 for its modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, GAPS, d_star

DRIVE_S = 0.08
DTS = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
SEED = 0


def drive_at(dt):
    np.random.seed(SEED)
    net = build(ladder_positions(GAPS))
    net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)
    sp, dd, spikes = 0.010, 0.002, 4
    burst_active, theta_period = spikes * sp, 0.125
    n_overflow = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for k in range(int(round(DRIVE_S / dt))):
            t = k * dt
            ph = t % theta_period
            v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
            net.step(dt, {"voltage": v, "reward": False})
        n_overflow = sum(1 for x in w if "overflow" in str(x.message))
    tr = net.entanglement_tracker
    tr.collect_dimers(net.synapses, net.positions)
    ps = np.array([d["P_S"] for d in tr.all_dimers]) if tr.all_dimers else np.array([0.0])
    gid_syn = {d["global_id"]: d["synapse_idx"] for d in tr.all_dimers}
    edges = set()
    for (i, j), f in tr.cross_synapse_bonds.items():
        if f > tr.WERNER_ENTANGLEMENT_BOUND:
            a, b = gid_syn.get(i), gid_syn.get(j)
            if a is not None and b is not None and a != b:
                edges.add((min(a, b), max(a, b)))
    return len(tr.all_dimers), float(np.median(ps)), float(ps.max()), len(edges), n_overflow


def main():
    t0 = time.time()
    print("=== dt CONVERGENCE OF THE DRIVE (0.08 s theta burst, seed 0) ===")
    print("The dt=0.01 control failed: 2223 -> 10072 dimers. Question now: is dt=1e-3")
    print("itself converged, or is the operating point a step-size artefact?\n")
    print(f"{'dt':>8} {'steps':>7} {'dimers':>8} {'vs 1e-4':>9} {'P_S_med':>9} "
          f"{'d*_med':>7} {'edges':>6} {'ovf':>4} {'wall':>7}")
    ref = None
    for dt in DTS:
        t1 = time.time()
        n, med, mx, ne, ovf = drive_at(dt)
        if ref is None:
            ref = n
        ratio = n / ref if ref else float("nan")
        print(f"{dt:8.4f} {int(round(DRIVE_S/dt)):7d} {n:8d} {ratio:8.2f}x "
              f"{med:9.4f} {d_star(med*med):7.2f} {ne:6d} {ovf:4d} "
              f"{time.time()-t1:6.0f}s")
    print(f"\nCONVERGED iff the dimer count is flat across dt. Any climb means the")
    print(f"operating point depends on the step size, and every count-based conclusion")
    print(f"downstream (the ~155 uM figure; D8/D14/D16/D17) inherits it.")
    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
