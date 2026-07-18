# Lead: po1-b2 (PO-1 · B2 — retire the per-synapse pump site) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** the per-synapse site calls
`bose_einstein_occupation` (`model6_parameters.py:46`) on the `n_ex = n̄_s` form; no
hand-rolled `hbar` anywhere in `vibrational_cascade_module.py`; `kT_ref` and `r_at_E_ref`
grep-provably gone from the live path; **a measurement shows the per-synapse and backbone
pumps agree on the same mode**; T1′ static probe still 7/7; a superseding entry retires
`DISC-1` in the `RESEARCH_LOG_CALCIUM_DIMER.md` DECISION RECORD.

**Status:** **ACCEPTANCE MET 5/5 — awaiting MO verification.** Two items escalated (below).
**Current unit:** — (B2 landed; nothing in flight)
**Last heartbeat:** 2026-07-18, post-landing.
**Blocked on:** nothing. Two MO decisions requested, neither blocking further B2 work.

**Owns:** `vibrational_cascade_module.py`, backbone params `model6_parameters.py:759-805`.
**Must not touch:** PO-2's `atp_system.py` / phosphate path · PO-3's `spine_plasticity_module.py`
· PO-4's `analytical_gap`, `run_theta_burst_45s.py` · PO-5's `multi_synapse_network.py` and the
T1′ probe family · PO-6's orphan modules, `quantum_dimensions.py`, `sweep_runner.py`.

**Shared-file hazard:** `model6_core.py` — PO-1/PO-2/PO-4. One uncommitted holder at a time.
PO-2 is sequenced to start at this PO's commit boundary. **PO-1 no longer holds any
uncommitted edit to it — `model6_core.py` is FREE for PO-2 as of `c280e85`.**

---

## Acceptance — 5/5, with evidence

| # | Bar | State | Evidence |
|---|---|---|---|
| 1 | calls `bose_einstein_occupation` on `n_ex = n̄_s`; no hand-rolled `hbar` | **MET** | `critical_power()`; both `hbar=1.0546e-34` copies gone |
| 2 | `kT_ref` / `r_at_E_ref` gone from the live path, grep-provable | **MET** | 0 executable refs, proven with comments + docstrings + **string literals** stripped (`ast`/`tokenize`) |
| 3 | measurement: the two pumps agree on the same mode | **MET** | `sweep/pump_mode_agreement_probe.py` — committed FAILING at `fa12009`, PASS at `c280e85` |
| 4 | T1′ static probe still 7/7 | **MET** | `coherence_radius_probe.py`: 7/7, betti0=3, `crosscheck_ok=True`, CONFIRMED, 7 s. Ran unmodified — PO-5's surface |
| 5 | superseding entry retiring DISC-1 | **MET** | log rows B2-1/B2-2/B2-3; DISC-1 marked superseded, row left intact per append-only |

**Commits:** `fa12009` (probe, committed failing) → `c280e85` (the physics) → log + coordination.

## The measurement, and what it does NOT show

A1 mode 5000→1.000000. A2 convention 6.301→1.000000. **C1/C2 positive controls both still
FIRE**, and the probe returns **INVALID, not PASS**, if either goes silent — the L·ETA-4
failure mode is excluded by construction rather than by hope. A2's reference is recomputed
from CODATA, never by calling `bose_einstein_occupation`, so it cannot compare the shipped
function to itself once B2 wires that function in.

Corroborations computed rather than copied: `P_c = 21.514 fW` (pin 21.5), rest `r = 0.039`
(pin 0.04), `n̄ = 8.0742e5` (pin 8.07e5), per-synapse `P_c` ≡ backbone `P_c` to machine
precision.

**LIMIT, stated:** this proves the two sites AGREE. It does **not** prove 8 MHz is the right
mode — that remains the May-30 bet (Q≳10, Pokorný slip-layer vs Foster/Baish). If the bet is
wrong, both sites are now wrong *together*, and this probe would still report PASS.

## Escalated — MO decision, not mine

1. **The drive changed, and it touched `model6_core.py` (2 lines + 1 import).** The pump now
   takes per-spine `p_met_W` instead of `collective_field_kT`. Grounds: the may30 pin
   specifies "per-synapse P_met, NO aggregation" for B2; Step B already rejected
   `collective_field_kT` as a drive at the backbone; and a kT→power conversion would require
   precisely the retired calibration constants. I read this as pinned-not-fresh, the same
   shape as the MO's Q1 ruling — but it is a drive-physics change, so it is flagged for veto.
   **One commit reverts it.** Slice kept minimal and committed at once; explicit-path `git add`.
2. **`k_agg` delta — reported, not damped** (per MO ruling). Max enhancement 1.78→4.40.
   Bigger finding: the OLD η was ~0.216 across the *entire* drive range (0.2160→0.2211) — a
   constant dressed as a variable. **Does that move a standing result? MO's call, not mine.**

## Findings routed, not acted on

- **`Model6Parameters` has no `cascade` attribute** — the *entire* per-synapse pump parameter
  set has never been reachable by `sweep_runner`, not just `kT_ref`. Widens DISC-1 one level;
  still true after B2. Wiring `params.cascade` is unowned — likely PO-6.
- **Skill drift (B2-3):** `model6-architecture`'s "a single synapse is subcritical by design"
  is **contingent, not structural**. The June-7 "~17 fW, r=0.803" reproduces exactly at
  `E_inv=0.495, ca_open=0.55` — the drive achieved by 30 s, not a ceiling. At sustained full
  invasion: 51.24 fW, r=2.38, above threshold with no aggregation at all. Aggregation buys
  crossing at *lower per-spine drive*, not crossing at all. Skills are not my surface.
- **`test_learning_pathway.py` is not a <1 min check on this branch, and that PREDATES B2.**
  Measured on an isolated pre-change copy (`/tmp/b2base`, my two files reverted from git):
  220 s+ without clearing Phase 1, then stopped to free CPU. The `model6-codebase-operations`
  "<1 min" figure is stale. I did **not** use it as my regression floor — the T1′ static probe
  is the floor, and it passes in 7 s. Not my surface to re-time.

**Compute:** stayed inside the light budget — probe 1 s, T1′ probe 7 s, one 98 s bounded
single-synapse smoke, one backgrounded baseline killed once it had answered its question.
PO-3's heavy slot untouched.
