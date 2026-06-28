#!/usr/bin/env python3
"""
Dimer Chemistry Probe — post-thermodynamic-reformulation
=========================================================
Exercises CalciumPhosphateDimerization against the ACTUAL interface
after the 5-change reformulation:
  1. Formation gate removed
  2. Michaelis-Menten overlay removed
  3. k_base grounded from Smoluchowski × productive fraction
  4. k_classical grounded on Turhan cluster lifetime
  5. Detailed balance on dissolution (template_enhancement symmetric)

Checks 0-6 verify smoke, rest behaviour, transient response, IO curve,
boundedness, dimer/trimer split, and coherence-gated persistence.

Usage:  python sweep/dimer_chemistry_probe.py
"""

import sys, os
import numpy as np

# ── path setup ──────────────────────────────────────────────────────
SWEEP_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SWEEP_DIR)
MODEL6_DIR   = os.path.join(PROJECT_ROOT, 'src', 'models', 'Model_6')
sys.path.insert(0, MODEL6_DIR)

import logging
logging.disable(logging.INFO)

from model6_parameters import Model6Parameters
from ca_triphosphate_complex import (
    CalciumPhosphateIonPair,
    CalciumPhosphateDimerization,
    TemplateEffects,
)

# ══════════════════════════ constants ══════════════════════════════
GRID      = (8, 8)
N_VOXELS  = GRID[0] * GRID[1]
DX        = 4e-9         # m  (4 nm; matches TemplateEffects grid-to-nm factor)
DT        = 0.005         # s
PO4       = 1e-3          # M  (1 mM HPO₄²⁻; model6_parameters.phosphate.phosphate_total)
SEED      = 42
MAX_STEPS = 20_000
CONV_REL  = 5e-4          # relative convergence threshold per window
CONV_WIN  = 200           # steps between convergence checks

# ══════════════════════════ helpers ════════════════════════════════

# Ion-pair calculator — uses the real CalciumPhosphateIonPair class
_ip_calc = CalciumPhosphateIonPair(Model6Parameters(), temperature=310.15)


def make_system():
    """Build CalciumPhosphateDimerization + template enhancement on small grid."""
    dim = CalciumPhosphateDimerization(GRID, DX)

    # Place 2 template sites near centre so template path is exercised.
    tmpl = TemplateEffects(GRID, template_positions=[(3, 3), (4, 4)])
    template_enh = tmpl.calculate_surface_enhancement()

    return dim, template_enh


def ion_pair(ca_arr, po4_val=PO4):
    """Compute CaHPO₄ equilibrium via the real CalciumPhosphateIonPair class."""
    po4_arr = np.full_like(ca_arr, po4_val)
    return _ip_calc.calculate_ion_pair_concentration(ca_arr, po4_arr)


def total_cluster(dim):
    """Sum of dimer + trimer across all grid points (M·voxels)."""
    return float(np.sum(dim.dimer_concentration) + np.sum(dim.trimer_concentration))


def mean_cluster(dim):
    """Mean (dimer + trimer) per voxel (M)."""
    return total_cluster(dim) / N_VOXELS


def run_to_steady(dim, template_enh, ca_val, po4_val=PO4,
                  max_steps=MAX_STEPS, singlet_prob=0.25):
    """Step until convergence or max_steps.  Returns (converged, steps, mean_M)."""
    ca  = np.full(GRID, ca_val)
    ip  = ion_pair(ca, po4_val)
    po4 = np.full(GRID, po4_val)
    dim.set_mean_singlet_probability(singlet_prob)

    prev = mean_cluster(dim)
    for step in range(1, max_steps + 1):
        dim.update_dimerization(DT, ip, template_enh, ca, po4)
        if step % CONV_WIN == 0:
            cur = mean_cluster(dim)
            rel = abs(cur - prev) / (abs(prev) + 1e-30)
            if rel < CONV_REL:
                return True, step, cur
            prev = cur
    return False, max_steps, mean_cluster(dim)


# ══════════════════════════ checks ════════════════════════════════
results = {}
np.random.seed(SEED)

# ------------------------------------------------------------------
# CHECK 0 — smoke
# ------------------------------------------------------------------
print("=" * 72)
print("CHECK 0 — smoke: instantiate + one step at Ca = 10 µM")
print("=" * 72)
try:
    dim, template_enh = make_system()
    ca  = np.full(GRID, 1e-5)
    po4 = np.full(GRID, PO4)
    ip  = ion_pair(ca)
    dim.update_dimerization(DT, ip, template_enh, ca, po4)
    mc = mean_cluster(dim)
    print(f"  mean cluster after 1 step : {mc:.3e} M")
    print("  PASS — no error")
    results[0] = "PASS"
except Exception as e:
    print(f"  CONCERN — exception: {e}")
    results[0] = f"CONCERN: {e}"

# ------------------------------------------------------------------
# CHECK 1 — off at rest
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 1 — off at rest: Ca = 100 nM → steady state")
print("=" * 72)
np.random.seed(SEED)
dim, template_enh = make_system()
conv, steps, mc = run_to_steady(dim, template_enh, 1e-7)
mc_nM = mc * 1e9
print(f"  converged : {conv}  ({steps} steps = {steps*DT:.0f} s)")
print(f"  mean cluster : {mc_nM:.4f} nM")
if mc_nM < 100:
    print("  PASS — cluster << pool at rest")
    results[1] = "PASS"
else:
    print(f"  CONCERN — cluster = {mc_nM:.1f} nM, expected << µM")
    results[1] = f"CONCERN: {mc_nM:.1f} nM"

# ------------------------------------------------------------------
# CHECK 2 — on transient
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 2 — on transient: Ca = 10 µM → steady state")
print("=" * 72)
np.random.seed(SEED)
dim, template_enh = make_system()
conv, steps, mc = run_to_steady(dim, template_enh, 1e-5)
frac = mc / PO4
print(f"  converged : {conv}  ({steps} steps = {steps*DT:.0f} s)")
print(f"  mean cluster : {mc*1e6:.4f} µM")
print(f"  cluster / phosphate : {frac*100:.4f} %")
if frac < 0.10:
    print("  PASS — minor fraction of phosphate pool")
    results[2] = "PASS"
else:
    print(f"  CONCERN — cluster is {frac*100:.1f}% of phosphate (expect < 10%)")
    results[2] = f"CONCERN: {frac*100:.1f}%"

# ------------------------------------------------------------------
# CHECK 3 — IO curve
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 3 — IO curve: Ca sweep [0.1, 0.3, 1, 3, 10, 30] µM")
print("=" * 72)
ca_sweep_uM = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
ss_M = []
for ca_uM in ca_sweep_uM:
    np.random.seed(SEED)
    dim, template_enh = make_system()
    _, steps, mc = run_to_steady(dim, template_enh, ca_uM * 1e-6)
    ss_M.append(mc)
    print(f"  Ca = {ca_uM:5.1f} µM  →  cluster = {mc*1e9:12.3f} nM  (steps={steps})")

monotonic = all(ss_M[i] <= ss_M[i + 1] for i in range(len(ss_M) - 1))
ratio_mm  = ss_M[-1] / (ss_M[0] + 1e-30)

if ss_M[0] > 0 and ss_M[1] > 0:
    log_slope = (np.log(ss_M[1]) - np.log(ss_M[0])) / \
                (np.log(ca_sweep_uM[1]) - np.log(ca_sweep_uM[0]))
else:
    log_slope = float('nan')

print(f"\n  monotonic        : {monotonic}")
print(f"  max/min ratio    : {ratio_mm:.1f}×")
print(f"  log-log slope    : {log_slope:.2f}  (expect ~2, i.e. ∝ Ca²)")

ok3 = monotonic and ratio_mm > 10 and 1.0 < log_slope < 4.0
if ok3:
    print("  PASS — graded, monotonic, slope in expected range")
    results[3] = "PASS"
else:
    issues = []
    if not monotonic:                issues.append("non-monotonic")
    if ratio_mm <= 10:               issues.append(f"flat (ratio={ratio_mm:.1f})")
    if not (1.0 < log_slope < 4.0):  issues.append(f"slope={log_slope:.2f}")
    msg = "; ".join(issues)
    print(f"  CONCERN — {msg}")
    results[3] = f"CONCERN: {msg}"

# ------------------------------------------------------------------
# CHECK 4 — bounded
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 4 — bounded: max cluster / phosphate across sweep")
print("=" * 72)
max_frac = max(c / PO4 for c in ss_M)
print(f"  max cluster / phosphate : {max_frac*100:.4f} %")
if max_frac < 0.10:
    print("  PASS — well below phosphate pool")
    results[4] = "PASS"
else:
    print(f"  CONCERN — {max_frac*100:.1f}% of phosphate consumed")
    results[4] = f"CONCERN: {max_frac*100:.1f}%"

# ------------------------------------------------------------------
# CHECK 5 — dimer / trimer split
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 5 — dimer / trimer split")
print("=" * 72)

# 5a: across the IO sweep (PO₄ = 1 mM → all Ca/P < 0.5 → expect dimer-dominant)
print("  (a) across IO sweep (PO₄ = 1 mM, all Ca/P ≪ 0.5):")
for ca_uM in ca_sweep_uM:
    np.random.seed(SEED)
    dim, template_enh = make_system()
    run_to_steady(dim, template_enh, ca_uM * 1e-6)
    d = float(np.sum(dim.dimer_concentration))
    t = float(np.sum(dim.trimer_concentration))
    tot = d + t + 1e-30
    ca_p = (ca_uM * 1e-6) / PO4
    tag = "trimer-dom" if t > d else "dimer-dom"
    print(f"      Ca={ca_uM:5.1f} µM  Ca/P={ca_p:.4f}  "
          f"dimer {d/tot*100:5.1f}%  trimer {t/tot*100:5.1f}%  [{tag}]")

# 5b: dedicated Ca/P > 0.5 test (Ca = 1 mM, PO₄ = 1 mM → Ca/P = 1.0)
print("  (b) dedicated Ca/P > 0.5 test  (Ca = 1 mM, PO₄ = 1 mM → Ca/P = 1.0):")
np.random.seed(SEED)
dim, template_enh = make_system()
run_to_steady(dim, template_enh, 1e-3, po4_val=PO4)
d = float(np.sum(dim.dimer_concentration))
t = float(np.sum(dim.trimer_concentration))
tot = d + t + 1e-30
tag = "trimer-dom" if t > d else "dimer-dom"
print(f"      dimer {d/tot*100:5.1f}%  trimer {t/tot*100:5.1f}%  [{tag}]")
if t > d:
    print("  PASS — Ca/P > 0.5 → trimer-dominant")
    results[5] = "PASS"
else:
    print("  CONCERN — Ca/P = 1.0 did not yield trimer dominance")
    results[5] = "CONCERN: Ca/P=1.0 still dimer-dominant"

# ------------------------------------------------------------------
# CHECK 6 — coherence-gated persistence
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("CHECK 6 — coherence-gated persistence")
print("=" * 72)

DECAY_STEPS  = 60_000    # 300 s total
SAMPLE_EVERY = 200       # sample every 1 s

decay_taus = {}
for label, sp in [("unprotected (p_s=0.25)", 0.25),
                  ("protected   (p_s=0.90)", 0.90)]:
    np.random.seed(SEED)
    dim, template_enh = make_system()

    # Phase 1: build up at Ca = 10 µM
    run_to_steady(dim, template_enh, 1e-5, singlet_prob=sp)
    mc0 = mean_cluster(dim)

    # Phase 2: drop Ca to 100 nM, record decay
    ca_lo = np.full(GRID, 1e-7)
    po4_a = np.full(GRID, PO4)
    ip_lo = ion_pair(ca_lo)
    dim.set_mean_singlet_probability(sp)

    times   = []
    mc_vals = []
    t = 0.0
    for step in range(1, DECAY_STEPS + 1):
        dim.update_dimerization(DT, ip_lo, template_enh, ca_lo, po4_a)
        t += DT
        if step % SAMPLE_EVERY == 0:
            times.append(t)
            mc_vals.append(mean_cluster(dim))

    times   = np.array(times)
    mc_vals = np.array(mc_vals)

    # Estimate τ: time for (mc − floor) to reach 1/e of initial swing
    floor     = mc_vals[-1]
    decaying  = mc_vals - floor
    initial   = decaying[0] if decaying[0] > 0 else mc0
    threshold = initial / np.e
    crossings = np.where(decaying < threshold)[0]
    tau = times[crossings[0]] if len(crossings) > 0 else float('inf')

    decay_taus[label] = tau
    print(f"  {label}:")
    print(f"    initial  : {mc0*1e9:.3f} nM")
    print(f"    final    : {mc_vals[-1]*1e9:.3f} nM")
    print(f"    tau (1/e): {tau:.1f} s")

tau_un = decay_taus["unprotected (p_s=0.25)"]
tau_pr = decay_taus["protected   (p_s=0.90)"]

if tau_un > 0 and np.isfinite(tau_un):
    ratio = tau_pr / tau_un if np.isfinite(tau_pr) else float('inf')
    print(f"\n  tau_protected / tau_unprotected = {ratio:.1f}x")
    if ratio > 3.0:
        print("  PASS — coherence markedly extends cluster lifetime")
        results[6] = "PASS"
    else:
        print(f"  CONCERN — ratio {ratio:.1f}x (expected ~7x)")
        results[6] = f"CONCERN: ratio={ratio:.1f}x"
elif not np.isfinite(tau_un) and not np.isfinite(tau_pr):
    print("\n  CONCERN — neither case decayed to 1/e within 300 s")
    results[6] = "CONCERN: no decay in 300s"
else:
    print(f"\n  CONCERN — unexpected tau values: un={tau_un}, pr={tau_pr}")
    results[6] = f"CONCERN: tau_un={tau_un:.1f}, tau_pr={tau_pr:.1f}"

# ══════════════════════════ SUMMARY ═══════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
labels = {
    0: "smoke (instantiate + 1 step)",
    1: "off at rest (Ca=100nM << pool)",
    2: "on transient (Ca=10uM, minor fraction)",
    3: "IO curve (graded, monotonic, slope~2)",
    4: "bounded (no runaway / exhaustion)",
    5: "dimer/trimer split (Ca/P>0.5 -> trimer)",
    6: "coherence-gated persistence (~7x longer)",
}
print(f"  {'Check':<48} Result")
print(f"  {'-'*48} {'-'*24}")
for i in range(7):
    print(f"  {labels[i]:<48} {results.get(i, 'SKIPPED')}")
print("=" * 72)
