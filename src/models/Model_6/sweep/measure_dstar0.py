#!/usr/bin/env python3
"""
MEASURE d*(0) — job #1. Do not assume P_S; measure it in the rig.

Two prior wrong claims came from assuming P_S (handoff §6.3: the "knife-edge" was
computed off an assumed P_S=0.90 when it is ~0.998). So: build the SAME rig the
confirmed static probe uses, drive it the same 0.08 s, and read the P_S
DISTRIBUTION out of it.

Why the distribution and not the median: coherence_radius_probe.py:120 computes
d_star(ps_med*ps_med) from the MEDIAN, but a synapse-quotient edge survives while
ANY bonded pair clears F>0.5 — a MAX over hundreds of pairs (handoff §3). At t~0
the spread should be tiny, so median vs tail should barely move d*. "Should" is
the word this program keeps getting burned by. Measure it.

Reads only. Touches no repo file.
"""
import sys, os

# Run from the sweep/ directory (repo convention). sweep/ is auto-added as the
# script dir; dirname(dirname(__file__)) adds Model_6 for its modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import types

from soc_topology_geometry_discriminator import build, clamp_eta
from coherence_radius_probe import ladder_positions, GAPS, d_star

pos = ladder_positions(GAPS)
net = build(pos)
net._update_backbone_field = types.MethodType(clamp_eta(0.26), net)

sp, dd, spikes = 0.010, 0.002, 4
burst_active, theta_period = spikes * sp, 0.125
dt, seconds = 0.001, 0.08
for k in range(int(seconds / dt)):
    t = k * dt
    ph = t % theta_period
    v = (-10e-3 if (ph < burst_active and (ph % sp) < dd) else -70e-3)
    net.step(dt, {"voltage": v, "reward": False})

tr = net.entanglement_tracker
tr.collect_dimers(net.synapses, net.positions)
ps = np.array([d["P_S"] for d in tr.all_dimers])
tb = np.array([bool(d.get("template_bound", False)) for d in tr.all_dimers]) \
    if tr.all_dimers and "template_bound" in tr.all_dimers[0] else None

print("=== d*(0) MEASUREMENT (rig = confirmed static-probe rig, eta=0.26) ===")
print(f"n_dimers = {len(ps)}   (t = {seconds} s)")
print()
print(f"{'stat':>8} {'P_S':>10} {'P_S^2':>10} {'d*(um)':>9}")
for name, val in [
    ("min",    float(ps.min())),
    ("p05",    float(np.percentile(ps, 5))),
    ("median", float(np.median(ps))),
    ("mean",   float(ps.mean())),
    ("p95",    float(np.percentile(ps, 95))),
    ("p99",    float(np.percentile(ps, 99))),
    ("max",    float(ps.max())),
]:
    print(f"{name:>8} {val:10.6f} {val*val:10.6f} {d_star(val*val):9.4f}")

print()
print(f"P_S std = {ps.std():.6f}   spread(max-min) = {ps.max()-ps.min():.6f}")
print(f"fraction below the 1/sqrt(2)=0.7071 floor: {100*(ps < 0.7071).mean():.2f}%")
if tb is not None:
    print(f"template_bound fraction: {100*tb.mean():.1f}%")

# The geometry decision, derived from the MEASURED number (median-based d*, with
# the tail shown above so the margin is visible rather than assumed).
d0_med = d_star(float(np.median(ps)) ** 2)
d0_max = d_star(float(ps.max()) ** 2)
print()
print(f"d*(0) [median] = {d0_med:.4f} um")
print(f"d*(0) [max]    = {d0_max:.4f} um   <- effective radius (edge survives on ANY pair)")
print()
print("PROPOSED LIVE GAPS = d*(0)[median] - {0.10, 0.20, 0.30, 0.40}:")
for off in (0.10, 0.20, 0.30, 0.40):
    g = d0_med - off
    ps_crit = np.sqrt(0.5 * np.exp(g / 5.0))
    print(f"   gap {g:5.3f} um   P_S_crit = {ps_crit:.6f}   margin vs d*(0)med = {off:.2f} um")
g_dark = 4.5
print(f"   dark control {g_dark} um   P_S_crit = {np.sqrt(0.5*np.exp(g_dark/5.0)):.4f} "
      f"(>1 => can NEVER bond)")
