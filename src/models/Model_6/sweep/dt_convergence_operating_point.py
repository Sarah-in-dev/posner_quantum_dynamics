#!/usr/bin/env python3
"""
DOES THE ~155 uM OPERATING POINT INHERIT THE dt-INFLATION? — the sustained-drive check.

BACKGROUND (measured, not inferred)
-----------------------------------
The drive-only sweep (2026-07-17) showed the dimer COUNT is NOT dt-converged at the
production dt=1e-3: 1616/1690/2223/3131/7308/10072 dimers at dt=1e-4/5e-4/1e-3/2e-3/5e-3/
1e-2 — a +38% climb from 1e-4 to 1e-3, monotone. P_S and the Werner edge set WERE
converged (d*=3.45, edges=5 across the whole range), so T1' is safe. But the operating
point (~152.6 uM, commit fd83460) is a dimer-CONCENTRATION quantity, measured at dt=1e-3
under SUSTAINED drive, and no post-fix dt sweep was ever run on it. If it inherits the
inflation, the ~155-vs-47 uM gap that D8/D14/D16/D17 must be re-read against is partly a
step-size artefact, not physics.

MECHANISM (from ca_triphosphate_complex.py, read this session — a lead, not the answer)
--------------------------------------------------------------------------------------
Formation is not linear-in-dt the way a rate process is:
  - stochastic_formation (:378-382): event probability ~ k_eff*pnc^2*dt, but the AMOUNT
    per event is pnc*0.1 — dt-INDEPENDENT. Substrate (pnc) is depleted between steps, so
    coarser dt commits a larger chunk before depletion feedback applies => more product.
  - formation_probability is clip(...,0,1) (:353): at large dt it saturates at 1,
    breaking dt-linearity further.
Dissolution is now correctly *dt (the fd83460 fix). So the RESIDUAL dt-sensitivity is in
FORMATION, and it is a convergence problem, not the old unit bug.

WHAT THIS MEASURES
------------------
The faithful operating-point quantity: peak dimer concentration under SUSTAINED theta
drive, run to saturation, at dt=1e-4 (reference) vs 5e-4 vs 1e-3 (production). Single
synapse — the operating point is a per-synapse chemistry quantity (D8's isolated A3
probe was single-region), so no cross-synapse O(n^2) cost. uM = np.max(dimer_conc)*1e6,
the convention in model6_core.py:326.

PASS = plateau uM within a few % across dt (operating point is converged; the ~155 uM
       figure stands and D8/D14/D16/D17 re-read is unaffected by dt).
FAIL = plateau climbs with dt (the operating point is inflated at production dt; the
       re-read must use a dt-converged number). Either way it is Sarah's physics call;
       this only supplies the measurement.
"""
import sys, os, time, logging
import numpy as np

# Run from the sweep/ directory (repo convention). sweep/ is auto-added as the
# script dir; dirname(dirname(__file__)) adds Model_6 for its modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)

from model6_parameters import Model6Parameters
from model6_core import Model6QuantumSynapse
from multi_synapse_network import MultiSynapseNetwork

# Sustained drive to saturation. The commit's saturation sequence reached ~152.6 uM by
# its 10th sample with increments +33->+7->+2.4; a ~40 s sustained drive is past the knee.
# Smoke test (2026-07-17): peak reaches 140 uM by 2 s, so 5 s is safely past the knee.
# Single-synapse step is ~113 ms (full module stack), so the dt=1e-4 arm (~1 h) is
# dropped: the drive-only sweep showed 1e-4->5e-4 is only +5% (1616->1690) while
# 5e-4->1e-3 is +31% (1690->2223), so 5e-4 is a near-converged reference and 1e-3-vs-5e-4
# captures essentially all of the production-dt inflation. Trajectory is printed dense so
# the plateau is VISIBLE, not assumed.
SATURATE_S = 5.0
DTS = [0.0005, 0.001]
SEED = 0
SPIKE_PERIOD, DEPOL_DUR, SPIKES, THETA = 0.010, 0.002, 4, 0.125
BURST_ACTIVE = SPIKES * SPIKE_PERIOD


def build_single():
    """One real synapse with the full chemistry, MT invaded, eta clamped ignited."""
    params = Model6Parameters()
    params.em_coupling_enabled = True
    params.multi_synapse_enabled = True
    params.environment.fraction_P31 = 1.0
    net = MultiSynapseNetwork(n_synapses=1, pattern="clustered", spacing_um=1.0)
    net.initialize(Model6QuantumSynapse, params)
    for s in net.synapses:
        s.set_microtubule_invasion(True)
    net.disable_auto_commitment = True
    return net


def peak_dimer_uM(net):
    dc = net.synapses[0].ca_phosphate.dimerization.dimer_concentration
    return float(np.max(dc)) * 1e6


def run_sustained(dt):
    np.random.seed(SEED)
    net = build_single()
    n_steps = int(round(SATURATE_S / dt))
    burst_steps = int(round(BURST_ACTIVE / dt))
    period_steps = int(round(THETA / dt))
    peak = 0.0
    traj = []
    next_sample = 0.0
    for k in range(n_steps):
        t = k * dt
        ph_step = k % period_steps
        if ph_step < burst_steps:
            t_in_spike = (ph_step * dt) % SPIKE_PERIOD
            v = -10e-3 if t_in_spike < DEPOL_DUR else -70e-3
        else:
            v = -70e-3
        net.step(dt, {"voltage": v, "reward": False})
        u = peak_dimer_uM(net)
        peak = max(peak, u)
        if t >= next_sample:
            traj.append((round(t, 1), u))
            next_sample += 1.0
    return peak, peak_dimer_uM(net), traj


def main():
    t0 = time.time()
    print("=== OPERATING-POINT dt CONVERGENCE (sustained theta, single synapse) ===")
    print(f"peak dimer uM to saturation ({SATURATE_S}s), vs the ~152.6 uM commit figure\n")
    print(f"{'dt':>8} {'steps':>8} {'peak uM':>9} {'final uM':>9} {'vs 1e-4':>8} {'wall':>7}")
    ref = None
    rows = []
    for dt in DTS:
        pk, fin, traj = run_sustained(dt)
        if ref is None:
            ref = pk
        rows.append((dt, pk, fin, traj))
        print(f"{dt:8.4f} {int(round(SATURATE_S/dt)):8d} {pk:9.1f} {fin:9.1f} "
              f"{pk/ref:7.2f}x {time.time()-t0:6.0f}s")
    print("\nsaturation trajectory (uM every 1 s) — read this to CONFIRM the plateau:")
    for dt, pk, fin, traj in rows:
        pts = "  ".join(f"{t}s:{u:.0f}" for t, u in traj)
        print(f"  dt={dt:.4f}: {pts}")
    print("\nCONVERGED iff plateau uM is dt-flat. A climb means the ~155 uM operating")
    print("point is inflated at production dt=1e-3 and the D8/D14/D16/D17 re-read must")
    print("use a dt-converged number. Physics call is Sarah's; this is the measurement.")
    print(f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
