#!/usr/bin/env python3
"""PROBE (read-only): spine volume dynamics + the AMPAR clock.

Drives SpinePlasticityModule directly (no full physics) to measure:
  A. trajectory under sustained commit
  B. commit-then-silence (does volume decay? how fast?)
  C. the ceiling
  D. what the drivers actually are (structural_drive vs calcium vs quantum_field_kT)
  E. persistence across an analytical_gap (which advances spine time by 0.001 s)
"""
import sys, os
import numpy as np

ROOT = "/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6"
sys.path.insert(0, os.path.join(ROOT, "src", "models", "Model_6"))

from spine_plasticity_module import SpinePlasticityModule, SpinePlasticityParameters

DT = 0.005  # physics_dt used by run_spatial_discovery


def fresh(stochastic=False):
    p = SpinePlasticityParameters()
    p.volume.stochastic = stochastic          # kill thermal noise for determinism
    return SpinePlasticityModule(p)


def run(mod, seconds, drive, calcium, kT=0.0, dt=DT):
    n = int(round(seconds / dt))
    for _ in range(n):
        mod.step(dt, drive, calcium, quantum_field_kT=kT)
    return mod


print("=" * 78)
print("A. SUSTAINED COMMIT: drive=1.0, calcium=5.0 uM (deterministic, noise off)")
print("=" * 78)
m = fresh()
print(f"{'t(s)':>7} {'volume':>8} {'actin_tot':>10} {'enlarge':>8} {'stable':>8} "
      f"{'conf':>6} {'monomer':>8} {'AMPAR':>7} {'phase':>12}")
marks = [0, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 1805, 2400, 3600]
prev = 0.0
for t in marks:
    if t > prev:
        run(m, t - prev, 1.0, 5.0)
        prev = t
    print(f"{m.time:7.1f} {m.spine_volume:8.4f} {m.actin_total:10.4f} "
          f"{m.actin_enlargement:8.4f} {m.actin_stable:8.4f} {m.confinement:6.3f} "
          f"{m.actin_monomer:8.4f} {m.AMPAR_count:7.2f} {m.phase:>12}")

print()
print("=" * 78)
print("B. COMMIT-THEN-SILENCE: 60 s of drive=1/Ca=5, then drive=0/Ca=0.1")
print("=" * 78)
m2 = fresh()
run(m2, 60.0, 1.0, 5.0)
v_at_commit_end = m2.spine_volume
print(f"  end of 60 s commit: V={v_at_commit_end:.4f} enlarge={m2.actin_enlargement:.4f} "
      f"conf={m2.confinement:.4f} stable={m2.actin_stable:.4f}")
print(f"{'t_silent(s)':>12} {'volume':>8} {'enlarge':>8} {'stable':>8} {'conf':>6} {'frac_of_peak':>13}")
prev = 0.0
for t in [0, 10, 30, 60, 120, 300, 600, 1200, 3000]:
    if t > prev:
        run(m2, t - prev, 0.0, 0.1)
        prev = t
    frac = (m2.spine_volume - 1.0) / max(1e-9, v_at_commit_end - 1.0)
    print(f"{t:12.0f} {m2.spine_volume:8.4f} {m2.actin_enlargement:8.4f} "
          f"{m2.actin_stable:8.4f} {m2.confinement:6.3f} {frac:13.4f}")

print()
print("=" * 78)
print("C. CEILING: drive=1, Ca=100 uM (saturating), long run")
print("=" * 78)
m3 = fresh()
run(m3, 3000.0, 1.0, 100.0)
print(f"  V after 3000 s at Ca=100uM: {m3.spine_volume:.4f}  "
      f"(params.max_enlargement_ratio={m3.params.volume.max_enlargement_ratio})")
print(f"  actin_total={m3.actin_total:.4f}  F_max="
      f"{m3.params.volume.max_enlargement_ratio ** (1/m3.params.volume.actin_volume_scaling):.4f}")

print()
print("=" * 78)
print("D. DRIVER SENSITIVITY: V after 60 s, sweeping drive / calcium / quantum_field_kT")
print("=" * 78)
print("  --- calcium sweep (drive=1.0, kT=0) ---")
for ca in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0]:
    mm = fresh(); run(mm, 60.0, 1.0, ca)
    print(f"    Ca={ca:6.2f} uM -> V={mm.spine_volume:.5f}  enlarge={mm.actin_enlargement:.5f}")
print("  --- drive sweep (Ca=5.0, kT=0) ---")
for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mm = fresh(); run(mm, 60.0, d, 5.0)
    print(f"    drive={d:4.2f}   -> V={mm.spine_volume:.5f}  enlarge={mm.actin_enlargement:.5f} "
          f"conf={mm.confinement:.5f}")
print("  --- quantum_field_kT sweep (drive=1.0, Ca=5.0) ---")
for kT in [0.0, 1.0, 5.0, 20.0, 100.0]:
    mm = fresh(); run(mm, 60.0, 1.0, 5.0, kT=kT)
    print(f"    kT={kT:6.1f}   -> V={mm.spine_volume:.8f}")

print()
print("=" * 78)
print("E. PERSISTENCE ACROSS analytical_gap()")
print("=" * 78)
print("  analytical_gap() advances synapse spine state ONLY via the single")
print("  network.step(0.001, ...) at run_spatial_discovery.py:299.")
m4 = fresh()
run(m4, 60.0, 1.0, 5.0)
v_before, t_before = m4.spine_volume, m4.time
m4.step(0.001, 0.0, 0.1)   # the entire 30 s gap, as the code actually applies it
print(f"  V before gap = {v_before:.6f} (spine.time={t_before:.4f})")
print(f"  V after  gap = {m4.spine_volume:.6f} (spine.time={m4.time:.4f})")
print(f"  delta V      = {m4.spine_volume - v_before:+.3e}")
print(f"  spine.time advanced by {m4.time - t_before:.4f} s for a 30.0 s wall gap")
print()
print("  Counterfactual: if the gap were integrated at dt=0.005 for a real 30 s:")
m5 = fresh(); run(m5, 60.0, 1.0, 5.0)
v5 = m5.spine_volume
run(m5, 30.0, 0.0, 0.1)
print(f"  V before = {v5:.6f} -> V after real 30 s silence = {m5.spine_volume:.6f} "
      f"(delta {m5.spine_volume - v5:+.4f})")

print()
print("=" * 78)
print("F. AMPAR CLOCK ARITHMETIC")
print("=" * 78)
delay = 1800.0
print(f"  ampar_onset_delay = {delay} s of spine_plasticity.time (module line 420-425)")
print(f"  spine.time advances by physics_dt={DT} s per syn.step() call.")
print("  A synapse is stepped ONLY when activation > 0.05 (run_spatial_discovery.py:438).")
for budget, gap, ntrials in [(60.0, 30.0, 5), (90.0, 30.0, 25)]:
    best_per_trial = budget          # synapse active on EVERY step
    gap_contrib = 0.001              # the single network.step in analytical_gap
    per_trial = best_per_trial + gap_contrib
    need = delay / per_trial
    print(f"  trial_budget={budget}s gap={gap}s: BEST-CASE spine time/trial = {per_trial:.3f} s")
    print(f"     -> trials needed for AMPAR onset = {need:.1f}")
    print(f"     -> configured n_trials={ntrials} gives max spine time "
          f"{ntrials * per_trial:.1f} s = {100*ntrials*per_trial/delay:.1f}% of the delay")
print("  NOTE: best case assumes activation>0.05 on 100% of steps, which no")
print("  navigating agent achieves; realistic duty cycle is a small fraction.")
