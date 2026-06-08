#!/usr/bin/env python3
"""
Behavior check for the three-pool actin model in SpinePlasticityModule.

Runs six phases (REST → ACTIVITY → DECAY → COMMITMENT → SUSTAINED-UNCOMMITTED
→ SUSTAINED-COMMITTED) and checks:
  1. REST subcritical:       enlargement < 0.01, E_invasion == 0
  2. ACTIVITY phase-correct: enlargement > 0.5, E_invasion > 0 with drive=0
  3. DECAY emergent tau:     enlargement at +180 s ~ 0.37× phase-2-end value
  4. COMMITMENT retention:   actin_stable rises, actin_total stays elevated
  5. BASELINE preserved:     actin_total ~ 1.0 at end of Phase 1
  6. CEILING:                max(actin_total) < F_max in sustained phases
  7. LATCH:                  confinement low (uncommitted), high (committed)
  8. CLEARING:               extrusion clears enlargement; commitment retains stable
  9. NO-CLIP:                stable exceeds retired E_cap=1.0 (regression guard)
"""

import sys, os, math

# Allow running from repo root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.Model_6.spine_plasticity_module import (
    SpinePlasticityModule,
    SpinePlasticityParameters,
    SpineVolumeParameters,
)

# ---------------------------------------------------------------------------
# Setup: deterministic (disable stochastic volume noise)
# ---------------------------------------------------------------------------
params = SpinePlasticityParameters()
params.volume = SpineVolumeParameters(stochastic=False)
mod = SpinePlasticityModule(params)

dt = 0.1  # seconds

def snap(label, m=None):
    """Print the key state variables."""
    if m is None:
        m = mod
    print(f"  [{label:>20s}]  "
          f"dyn={m.actin_dynamic:.4f}  "
          f"enl={m.actin_enlargement:.4f}  "
          f"stb={m.actin_stable:.4f}  "
          f"tot={m.actin_total:.4f}  "
          f"conf={m.confinement:.4f}  "
          f"mono={m.actin_monomer:.4f}  "
          f"Einv={m.E_invasion:.4f}")

def run(seconds, calcium, drive, label=None, sample_times=None, m=None):
    """Step the module; optionally sample at listed offsets (relative to run start).
    Returns (samples_dict, max_actin_total)."""
    if m is None:
        m = mod
    samples = {}
    elapsed = 0.0
    steps = int(round(seconds / dt))
    max_actin_total = m.actin_total
    for i in range(steps):
        m.step(dt, structural_drive=drive, calcium=calcium)
        elapsed += dt
        if m.actin_total > max_actin_total:
            max_actin_total = m.actin_total
        if sample_times:
            for st in sample_times:
                if abs(elapsed - st) < dt / 2 and st not in samples:
                    samples[st] = {
                        "actin_enlargement": m.actin_enlargement,
                        "actin_stable": m.actin_stable,
                        "actin_total": m.actin_total,
                        "E_invasion": m.E_invasion,
                        "confinement": m.confinement,
                        "actin_monomer": m.actin_monomer,
                    }
    if label:
        snap(label, m)
    return samples, max_actin_total

# ---------------------------------------------------------------------------
print("=" * 78)
print("THREE-POOL ACTIN MODEL — BEHAVIOR CHECK")
print("=" * 78)

# F_max: structural ceiling from volume parameters (same formula as source)
vol = params.volume
F_max = vol.max_enlargement_ratio ** (1.0 / vol.actin_volume_scaling)
print(f"  F_max = {vol.max_enlargement_ratio} ** (1/{vol.actin_volume_scaling}) = {F_max:.4f}")

# ---- Phase 1: REST (60 s, calcium 0.1 uM, drive 0) ----------------------
print("\n--- Phase 1: REST (60 s, Ca=0.1 uM, drive=0) ---")
run(60.0, calcium=0.1, drive=0.0, label="end REST")

rest_total = mod.actin_total
rest_enlarge = mod.actin_enlargement
rest_Einv = mod.E_invasion

# ---- Phase 2: ACTIVITY (60 s, calcium 2.0 uM, drive 0) ------------------
print("\n--- Phase 2: ACTIVITY (60 s, Ca=2.0 uM, drive=0) ---")
run(60.0, calcium=2.0, drive=0.0, label="end ACTIVITY")

act_enlarge = mod.actin_enlargement
act_Einv = mod.E_invasion
act_total = mod.actin_total

# ---- Phase 3: DECAY (600 s, calcium 0.1 uM, drive 0) --------------------
print("\n--- Phase 3: DECAY (600 s, Ca=0.1 uM, drive=0) ---")
decay_sample_offsets = [0.0, 90.0, 180.0, 360.0]
# The +0 sample is really the phase-2 end value we already have, but re-sample
# at the very start of the decay run for consistency.
decay_samples, _ = run(600.0, calcium=0.1, drive=0.0, label="end DECAY",
                       sample_times=decay_sample_offsets)

print("  Decay-phase samples (offset from phase-2 end):")
for t_off in decay_sample_offsets:
    if t_off in decay_samples:
        s = decay_samples[t_off]
        print(f"    +{t_off:5.0f}s  enl={s['actin_enlargement']:.4f}  "
              f"stb={s['actin_stable']:.4f}  tot={s['actin_total']:.4f}  "
              f"conf={s['confinement']:.4f}  mono={s['actin_monomer']:.4f}  "
              f"Einv={s['E_invasion']:.4f}")

decay_enlarge = mod.actin_enlargement
decay_stable = mod.actin_stable

# ---- Phase 4: COMMITMENT (120 s, calcium 2.0 uM, drive 1.0) -------------
print("\n--- Phase 4: COMMITMENT (120 s, Ca=2.0 uM, drive=1.0) ---")
pre_commit_stable = mod.actin_stable
pre_commit_total = mod.actin_total
run(120.0, calcium=2.0, drive=1.0, label="end COMMITMENT")

commit_stable = mod.actin_stable
commit_total = mod.actin_total
_st4 = mod.get_state()
print(f"  [endpoint chain]  total={_st4['actin_total']:.4f}  "
      f"spine_volume={_st4['spine_volume']:.4f}  "
      f"AMPAR_count={_st4['AMPAR_count']:.1f}  "
      f"AMPAR_surface={_st4['AMPAR_surface']:.1f}  "
      f"AMPAR_reserve={_st4['AMPAR_reserve']:.1f}  "
      f"strength={_st4['synaptic_strength']:.4f}")

# ---- Phase 5: SUSTAINED UNCOMMITTED (3000 s, Ca=2.0 uM, drive=0, fresh instance) --
print("\n--- Phase 5: SUSTAINED UNCOMMITTED (3000 s, Ca=2.0 uM, drive=0, fresh) ---")
mod5 = SpinePlasticityModule(params)   # same config as phases 1-4, fresh state
phase5_sample_offsets = [60.0, 120.0, 180.0, 300.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0]
sust_uncom_samples, sust_uncom_max_total = run(
    3000.0, calcium=2.0, drive=0.0, label="end SUST-UNCOMMITTED",
    sample_times=phase5_sample_offsets, m=mod5)

print("  Sustained-uncommitted samples:")
for t_off in phase5_sample_offsets:
    if t_off in sust_uncom_samples:
        s = sust_uncom_samples[t_off]
        print(f"    +{t_off:5.0f}s  enl={s['actin_enlargement']:.4f}  "
              f"tot={s['actin_total']:.4f}  conf={s['confinement']:.4f}")

enl_ceiling_uncommitted = mod5.actin_enlargement
Fmax_obs_uncommitted = sust_uncom_max_total
conf_end_uncommitted = mod5.confinement

# ---- Phase 6: SUSTAINED COMMITTED (600 s, Ca=2.0 uM, drive=1.0, fresh instance) --
print("\n--- Phase 6: SUSTAINED COMMITTED (600 s, Ca=2.0 uM, drive=1.0, fresh) ---")
mod6 = SpinePlasticityModule(params)   # same config as phases 1-4, fresh state
phase6_sample_offsets = [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0]
sust_com_samples, sust_com_max_total = run(
    600.0, calcium=2.0, drive=1.0, label="end SUST-COMMITTED",
    sample_times=phase6_sample_offsets, m=mod6)

print("  Sustained-committed samples:")
for t_off in phase6_sample_offsets:
    if t_off in sust_com_samples:
        s = sust_com_samples[t_off]
        print(f"    +{t_off:5.0f}s  enl={s['actin_enlargement']:.4f}  "
              f"tot={s['actin_total']:.4f}  conf={s['confinement']:.4f}")

Fmax_obs_committed = sust_com_max_total
conf_end_committed = mod6.confinement
stable_end_committed = mod6.actin_stable
_st6 = mod6.get_state()
print(f"  [endpoint chain]  total={_st6['actin_total']:.4f}  "
      f"spine_volume={_st6['spine_volume']:.4f}  "
      f"AMPAR_count={_st6['AMPAR_count']:.1f}  "
      f"AMPAR_surface={_st6['AMPAR_surface']:.1f}  "
      f"AMPAR_reserve={_st6['AMPAR_reserve']:.1f}  "
      f"strength={_st6['synaptic_strength']:.4f}")

# ===========================================================================
# CHECKS
# ===========================================================================
print("\n" + "=" * 78)
print("CHECKS")
print("=" * 78)

results = {}

# 1. REST subcritical
rest_ok = rest_enlarge < 0.01 and rest_Einv == 0.0
results["REST subcritical"] = rest_ok
print(f"\n[{'PASS' if rest_ok else 'FAIL'}] REST subcritical: "
      f"enlargement={rest_enlarge:.6f} (<0.01), E_invasion={rest_Einv:.4f} (==0)")

# 2. ACTIVITY phase-correct (enlargement rises, E_invasion > 0 with drive=0)
act_ok = act_enlarge > 0.5 and act_Einv > 0.0
results["ACTIVITY phase-correct"] = act_ok
print(f"[{'PASS' if act_ok else 'FAIL'}] ACTIVITY phase-correct: "
      f"enlargement={act_enlarge:.4f} (>0.5), E_invasion={act_Einv:.4f} (>0)")

# 3. DECAY emergent tau — enlargement at +180 s ~ 0.37× phase-2-end value
if 180.0 in decay_samples:
    enl_at_180 = decay_samples[180.0]["actin_enlargement"]
    ratio_180 = enl_at_180 / act_enlarge if act_enlarge > 0 else float("nan")
    # implied tau: enl(t) = enl0 * exp(-t/tau) => tau = -t / ln(ratio)
    if 0 < ratio_180 < 1:
        implied_tau = -180.0 / math.log(ratio_180)
    else:
        implied_tau = float("nan")
    tau_min = 2.0 * 60   # 2 min  = 120 s (Honkura lower)
    tau_max = 15.0 * 60  # 15 min = 900 s (Honkura upper)
    tau_in_band = tau_min <= implied_tau <= tau_max
    results["DECAY emergent tau"] = tau_in_band
    print(f"[{'PASS' if tau_in_band else 'FAIL'}] DECAY emergent tau: "
          f"ratio@180s={ratio_180:.4f} (~0.37 ideal), "
          f"implied tau={implied_tau:.1f}s = {implied_tau/60:.2f} min")
    print(f"       Honkura 2-15 min band: [{tau_min/60:.0f}, {tau_max/60:.0f}] min  "
          f"{'IN BAND' if tau_in_band else 'OUT OF BAND'}")
    print(f"       Hu invasion residence ~2.5 min (150 s) — "
          f"{'near' if abs(implied_tau - 150) < 60 else 'not near'} "
          f"(check, not target)")
else:
    results["DECAY emergent tau"] = False
    print("[FAIL] DECAY emergent tau: no sample at +180s")

# 4. COMMITMENT retention: actin_stable rises, actin_total stays elevated
stable_rose = commit_stable > pre_commit_stable + 0.01
total_elevated = commit_total > 1.05
commit_ok = stable_rose and total_elevated
results["COMMITMENT retention"] = commit_ok
print(f"[{'PASS' if commit_ok else 'FAIL'}] COMMITMENT retention: "
      f"stable {pre_commit_stable:.4f}->{commit_stable:.4f} "
      f"({'rose' if stable_rose else 'DID NOT rise'}), "
      f"total={commit_total:.4f} "
      f"({'elevated' if total_elevated else 'NOT elevated'})")

# 5. BASELINE preserved: actin_total ~ 1.0 at end of Phase 1
baseline_ok = 0.95 <= rest_total <= 1.05
results["BASELINE preserved"] = baseline_ok
print(f"[{'PASS' if baseline_ok else 'FAIL'}] BASELINE preserved: "
      f"actin_total={rest_total:.4f} (expect ~1.0)")

# 6. CEILING: max(actin_total) over both sustained phases < F_max
ceiling_max = max(Fmax_obs_uncommitted, Fmax_obs_committed)
ceiling_ok = ceiling_max < F_max + 1e-3
results["CEILING"] = ceiling_ok
print(f"\n[{'PASS' if ceiling_ok else 'FAIL'}] CEILING: "
      f"max(actin_total)={ceiling_max:.4f} (<{F_max:.4f}+1e-3)")
print(f"       Asymptotic uncommitted: tot={mod5.actin_total:.4f}  enl={mod5.actin_enlargement:.4f}")
print(f"       Asymptotic committed:   tot={mod6.actin_total:.4f}  enl={mod6.actin_enlargement:.4f}")

# 7. LATCH: confinement low when uncommitted, high when committed
latch_low = conf_end_uncommitted < 0.05
latch_high = conf_end_committed > 0.7
latch_ok = latch_low and latch_high
results["LATCH"] = latch_ok
print(f"[{'PASS' if latch_ok else 'FAIL'}] LATCH: "
      f"conf_uncommitted={conf_end_uncommitted:.4f} (<0.05), "
      f"conf_committed={conf_end_committed:.4f} (>0.7)")

# 8. CLEARING: extrusion clears enlargement during decay; commitment retains stable
clearing_enl = decay_enlarge < 0.15 * act_enlarge
clearing_stb = commit_stable > decay_stable
clearing_ok = clearing_enl and clearing_stb
results["CLEARING"] = clearing_ok
print(f"[{'PASS' if clearing_ok else 'FAIL'}] CLEARING: "
      f"enl_decay={decay_enlarge:.4f} (<0.15*{act_enlarge:.4f}={0.15*act_enlarge:.4f}), "
      f"stable_commit={commit_stable:.4f} > stable_decay={decay_stable:.4f}")

# 9. NO-CLIP: stable exceeds retired E_cap=1.0 (regression guard)
no_clip_ok = stable_end_committed > 1.0
results["NO-CLIP"] = no_clip_ok
print(f"[{'PASS' if no_clip_ok else 'FAIL'}] NO-CLIP: "
      f"stable_committed={stable_end_committed:.4f} (>1.0, exceeds retired E_cap)")

# ---------------------------------------------------------------------------
# E_REF DATA (informational, no assertion)
# ---------------------------------------------------------------------------
print(f"\n  E_ref candidate: enl_ceiling_uncommitted = {enl_ceiling_uncommitted:.4f}")
print(f"    ^ Asymptotic enlargement under sustained Ca without commitment.")
print(f"      Candidate physical anchor for E_ref (decision pending).")

# ===========================================================================
# SUMMARY
# ===========================================================================
n_pass = sum(results.values())
n_total = len(results)
all_pass = n_pass == n_total

print("\n" + "=" * 78)
print(f"SUMMARY: {n_pass}/{n_total} checks passed — "
      f"{'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=" * 78)

sys.exit(0 if all_pass else 1)
