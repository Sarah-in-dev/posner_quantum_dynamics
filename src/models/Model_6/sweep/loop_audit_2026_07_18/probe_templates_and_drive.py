#!/usr/bin/env python3
"""PROBE (read-only): (1) template feedback channel, (2) does structural_drive
actually matter for spine volume, (3) realistic activation duty cycle.
"""
import sys, os
import numpy as np

ROOT = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6"
sys.path.insert(0, os.path.join(ROOT, "src", "models", "Model_6"))
sys.path.insert(0, os.path.join(ROOT, "sweep"))

from model6_parameters import Model6Parameters
from ca_triphosphate_complex import CaHPO4DimerSystem
from spine_plasticity_module import SpinePlasticityModule, SpinePlasticityParameters

p = Model6Parameters()
grid_shape = (p.spatial.grid_size, p.spatial.grid_size)
dx = (2 * p.spatial.active_zone_radius / p.spatial.grid_size)

# reproduce model6_core.py:228-242 channel/template placement
center = grid_shape[0] // 2
import model6_core  # noqa  (only to confirm import path health)

print("=" * 78)
print("1. TEMPLATE FEEDBACK: n_templates 3 (baseline) vs 4 vs 5 vs 6")
print("=" * 78)
print(f"   grid_shape={grid_shape}  dx={dx:.3e} m   (PHASE 12 fires 3->5 at V>1.25, 3->6 at V>1.5)")
print()

# core seeds 3 template positions from the first 3 channel positions
tpl0 = [(center, center), (center + 1, center), (center, center + 1)]

rows = []
for n in [3, 4, 5, 6, 7, 8]:
    sysn = CaHPO4DimerSystem(grid_shape, dx, p, tpl0)
    sysn.set_n_templates(n)
    tf = sysn.templates.template_field
    te = sysn.template_enhancement
    n_sites = int(np.sum(tf))
    # template_bound is assigned where template_field > 0.5 (dimer_particles.py:205)
    frac_bound_cells = float(np.mean(tf > 0.5))
    k_eff_mean = float(np.mean(te))
    k_eff_max = float(np.max(te))
    n_enhanced = int(np.sum(te > 1.0))
    rows.append((n, n_sites, frac_bound_cells, k_eff_mean, k_eff_max, n_enhanced))

print(f"{'requested':>10} {'actual_sites':>13} {'frac_cells_bound':>17} "
      f"{'mean_enh':>10} {'max_enh':>9} {'cells_enh>1':>12}")
for r in rows:
    print(f"{r[0]:>10} {r[1]:>13} {r[2]:>17.5f} {r[3]:>10.4f} {r[4]:>9.2f} {r[5]:>12}")

base = rows[0]
print()
print("   Relative to n=3 baseline (formation rate = k_base * enhancement * [PNC]^2,")
print("   ca_triphosphate_complex.py:629-630 -> rate scales linearly with enhancement):")
for r in rows:
    print(f"     n={r[0]}: mean_enh x{r[3]/base[3]:.4f}   template_bound cell fraction "
          f"x{(r[2]/base[2] if base[2] else float('nan')):.4f}")

print()
print("   Dissolution also carries template_enhancement "
      "(ca_triphosphate_complex.py:418: k_diss = k_classical*(1-singlet_excess)*template_enhancement)")
print("   -> MORE templates accelerate BOTH formation and dissolution in the same cells.")

print()
print("=" * 78)
print("2. DOES structural_drive DRIVE VOLUME? committed vs uncommitted, long run")
print("=" * 78)
DT = 0.005


def fresh():
    sp = SpinePlasticityParameters()
    sp.volume.stochastic = False
    return SpinePlasticityModule(sp)


def run(m, secs, drive, ca):
    for _ in range(int(round(secs / DT))):
        m.step(DT, drive, ca)
    return m


print("   Protocol: 60 s at Ca=5 uM with drive=0 (NEVER committed) vs drive=1 (committed),")
print("   then silence at Ca=0.1, drive=0.")
res = {}
for label, drive in [("uncommitted (drive=0)", 0.0), ("committed (drive=1)", 1.0)]:
    m = fresh()
    run(m, 60.0, drive, 5.0)
    traj = [(0.0, m.spine_volume)]
    for t in [30, 60, 120, 300, 600, 1200, 3000]:
        run(m, t - traj[-1][0], 0.0, 0.1)
        traj.append((t, m.spine_volume))
    res[label] = traj
    print(f"   {label}:")
    print("      " + "  ".join(f"t={t:<5.0f}V={v:.4f}" for t, v in traj))

u = dict(res["uncommitted (drive=0)"])
c = dict(res["committed (drive=1)"])
print()
print(f"   {'t_silent':>9} {'V_uncommitted':>15} {'V_committed':>13} {'committed - uncommitted':>25}")
for t in [0, 30, 60, 120, 300, 600, 1200, 3000]:
    print(f"   {t:>9.0f} {u[t]:>15.4f} {c[t]:>13.4f} {c[t]-u[t]:>25.4f}")

print()
print("=" * 78)
print("3. REALISTIC ACTIVATION DUTY CYCLE (drives the AMPAR clock)")
print("=" * 78)
try:
    from spatial_environment import SpatialEnvironment, Agent
    env = SpatialEnvironment(n_features=20, seed=42)
    agent = Agent(seed=42)
    agent_dt, budget = 0.5, 60.0
    n_steps = int(budget / agent_dt)
    per_syn_active = np.zeros(env.n_features)
    for _ in range(n_steps):
        acts = env.get_activations(agent.position)
        per_syn_active += (acts > 0.05)
        agent.step(agent_dt, env, np.zeros(env.n_features))  # no learned pull
    duty = per_syn_active / n_steps
    print(f"   {n_steps} agent steps over a {budget:.0f} s trial, {env.n_features} features")
    print(f"   duty cycle (fraction of agent steps a synapse is active > 0.05):")
    print(f"     max={duty.max():.4f}  mean={duty.mean():.4f}  "
          f"median={np.median(duty):.4f}  n_ever_active={int((duty>0).sum())}")
    best = duty.max()
    spine_time_per_trial = best * budget
    print(f"   BEST synapse accumulates ~{spine_time_per_trial:.2f} s of spine time per trial")
    if spine_time_per_trial > 0:
        print(f"   -> trials to reach ampar_onset_delay=1800 s: "
              f"{1800.0/spine_time_per_trial:.0f}")
    print(f"   MEAN synapse: {duty.mean()*budget:.2f} s/trial -> "
          f"{1800.0/max(1e-9, duty.mean()*budget):.0f} trials")
except Exception as e:
    print(f"   [could not run environment probe: {type(e).__name__}: {e}]")
