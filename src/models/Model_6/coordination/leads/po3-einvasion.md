# Lead: po3-einvasion (PO-3 · E_invasion provenance + the ratchet) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** `r` measured across N traversals showing ratchet
or no-ratchet — pre-registered, with a null arm and a verdict function that can return
FALSIFIED and INCONCLUSIVE, with its positive control demonstrated to fire — plus a
provenance verdict on `k_polymerization_max` (and `E_ref`, which sits in the load-bearing
denominator at `spine_plasticity_module.py:411-412`).

**Status:** **WRAPPED** (MO ruling 009). Acceptance-scoped objective exhausted; nothing
non-gated remains on this surface.
**Current unit:** none. Skill rewrite routed to the MO as
`requests/model6-mo/po3-einvasion-skill-002.md` (`9bd7218`) — the skill library symlinks into
another program's repo, so the MO writes it.
**Last heartbeat:** 2026-07-18 21:08Z — FINAL.
**Blocked on:** nothing. Two items GATED and correctly not attempted (below).

## Final ledger

| deliverable | state |
|---|---|
| Provenance verdict | **MET** — `E_ref` REPRODUCIBLE/SELF-REFERENTIAL (reproduces at 1.8742); `k_polymerization_max` INHERITED, 3.57× its own citation, inheriting commit `703d394` identified |
| Ratchet measurement | **EXECUTED, scored VOID on its own registered terms** — cause found (`BASELINE_RATE_HZ = 0.5`) |
| Rotation 001 (null audit) | **DELIVERED** — six-probe table; L·ETA-4's NMDAR half unsupported; F-4 to PO-5 |
| Rotation 002 (magnitude) | **EQUIVOCAL**, reasons separable and costed; plateau-ON arms PARKED with a price |
| Skill rewrite | **ROUTED** to the MO with exact text, not applied |

**Gated, correctly not attempted:** the L·ETA-5 re-run (Sarah — corrected null pre-registered as
AMENDMENT 4 and deliberately NOT run) · the plateau-ON pair (MO — parked, costed at >10× the
~65 s/arm plateau-OFF cost).

## What a successor should know that is not in the logs

1. **Nothing on this surface is grounded on the charge side.** The discharge constant is
   (`tau_extrude`, Honkura, and independently reproduced in-code at 180.1 s). The charge constant
   is not, and correcting it moves the model **away** from ignition.
2. **The driver is weakly activity-selective.** Tonic spontaneous release alone carries
   `E_invasion` past `invasion_threshold`; the driven/undriven separation collapses 6.15× → 1.70×
   over 8 traversals. Any claim that `E_invasion` encodes "this synapse was active" needs that
   measured first.
3. **`r ≥ 1` at one synapse is not a partition.** `k_cross ∝ η_i·η_j·P_product`
   (`multi_synapse_network.py:340-341`), so a single driven synapse yields zero cross-edges by
   construction — as measured, in both arms.

## Errors I made this session, for the successor's calibration

Six, none of which changed a constant, and only the last found by me first: the "frozen gap"
claim (PO-4) · the committed-branch retention derivation (PO-4) · F-3's "~100× NMDAR starvation"
(**overstated and inverted**; recommendation withdrawn) · claiming a sampling change was neutral
when it biased my own scored peak toward the convenient branch (MO ruling 008) · twice committing
to a detached HEAD where neither the MO nor Sarah could see the work · and registering a
peak-difference criterion that was unsound by construction.

**The consistent shape: reliable on physics reasoning, unreliable when judging whether my own
shortcuts were consequence-free.** A successor should assume the same and verify my
instrumentation choices, not just my conclusions.

## Rotation 002 result — EQUIVOCAL, and the reasons are separable

**Unsupported and wrong remain unseparated**, which is exactly what Sarah's PO-5 decision was
waiting on. Two independent reasons:
1. **Scored condition (plateau ON) not measured** — >10× cost, stalled, killed per the cap.
2. **One registered criterion is unsound** — I differenced a calcium **peak** across two
   independent stochastic arms; it returned **−14.65 µM**, i.e. blocking NMDAR "raising"
   calcium. My design defect, not the model's.

**Measured, sound, but plateau OFF (NOT the scored condition):** `R = 0.0147` (below the 0.05
negligible bar) and `ΔCa` **mean** `+0.51 µM` — positive as physics requires and ~half of
`K_calcium_poly`, so **not obviously small**. **I did not substitute the mean and rescore** —
that is the goalpost move the discipline prevents.

**Verified in code this cycle:** `k_cross = K_ENTANGLE_EM_BASE * eta_factor * w_spatial *
P_product` (`multi_synapse_network.py:340-341`). **`P_product` is a multiplicative co-factor with
η, not an alternative** — `η = 0` zeroes `k_cross` whatever `P_product` does. That makes
L·ETA-5's zero-cross-edges result **structurally necessary**, not incidental.

## Rotation 001 result — L·ETA-4's NMDAR half does not survive as evidence

**PO-5's sole surviving foundation is weaker than the log states.** Two independent defects,
either sufficient: the "silent" synapses do receive glutamate (13 events in the probe's own
12 s run, its own seeds), and the metric — NMDAR **open fraction** — is voltage-independent by
construction (`analytical_calcium_system.py:129-130`), so a plateau cannot move it whether or
not calcium flowed. Direction runs against the hypothesis: `B(-20)/B(-70) = 20.6×`.
**L·ETA-4's VGCC half and its conclusion (§8's η premise fails) both STAND.** Magnitude
UNDETERMINED — not re-run; the MO sequences that. Signalled to PO-5 as **F-4**.

Best structural finding: `run_spatial_discovery` and my probe fail in **opposite** directions —
the shipped runner gates stepping on `active_mask` (D19: inactive never decay, which
incidentally hides the leak); mine steps everything (decay correct, clock PASS) which exposes
it. **No probe in the family does both.** Full table: `docs/AUDIT_SPONTANEOUS_RELEASE_NULLS.md`.

## Result in one line

The ratchet question is **NOT ANSWERED**. The null arm suppressed activation but not
presynaptic release, so a never-activated synapse reached `E_invasion` = 0.4507 and `r` =
0.978 — it ratcheted *harder* (7.46×) than the driven arm (5.65×). Void on registered terms.
**The surviving finding is about my own surface:** `E_invasion` accumulates past
`invasion_threshold` on tonic spontaneous release alone, and the driven/undriven separation
collapses 6.15× → 1.70× across traversals.

## What needs a decision (NOT taken unilaterally)

1. **A re-run needs two protocol changes** — a null that suppresses spontaneous release
   (not just activation), and a gap long enough to clear the calcium tail before retention is
   scored. Both alter a pre-registered design and neither is mine to make. **No second run
   started; the compute slot is released for PO-1.**
2. **PO-5's edge is at most PARTIALLY cleared.** `η ≠ 0` was demonstrated live (`r` = 1.4050),
   but **cross-synapse edges = 0 in both arms** — only one feature was driven and
   `k_cross ∝ √(η_i·η_j)`. A partition needs ≥2 synapses above threshold at once, which this
   design excludes by construction. Whether that is enough to start PO-5 is the MO's call.

## Rulings absorbed

- **Ruling 002 / 003 (retention prediction conditional on confinement) — IMPLEMENTED** as
  prereg AMENDMENT 2 (`78b30ef`, published `9608e8a`). Ruling 003 supersedes 002 and asks the
  band centre be re-derived from measured `conf` with the existing tolerance width, at scoring,
  without killing the run. That is what AMENDMENT 2 does, computed **per gap** rather than at
  `conf_mean` — strictly finer, identical when `conf` is constant. `rho_ratio = rho/rho_pred(conf)`
  scored in `[0.89, 1.07]`, which is the original `[0.80, 0.95]` divided by `0.8948` — the same
  tolerance, carried over rather than re-invented. No constant moved.
- Ruling 003's credit on GATE 1 is accepted but the CONFINED-RATCHET branch it credits was
  **deleted** in AMENDMENT 2, because it rested on my own error (below). The corrected GATE 1
  is unconditional and strictly stronger.

## Errors I made this cycle — recorded, not quietly fixed

1. **F-2 "the gap is frozen" was WRONG.** `analytical_gap`'s tail runs
   `network.step(0.001, ...)`, so actin advances 1 ms per gap; retention is 0.9999944, not 1.0.
   I concluded "frozen" from absence in the docstring's two lists without reading the function
   tail — prose checked against prose, reported by me under a `[code SHOWN]` tag, in a program
   whose signature defect is exactly that. Caught by PO-4. The decision not to call it stands
   and is better motivated; only the magnitude was misstated. (CORRECTION 1.)
2. **My prereg §2(B) mis-derived the committed branch.** I read `:389` (extrusion gated off by
   `conf`) and stopped, missing `:390` — `conf` gates *retention* ON, which also drains
   `actin_enlargement`. A committed spine drains **3.54× faster**, not slower. Caught by PO-4.
   This is the defect class I was dispatched to hunt, committed in the section of my own
   pre-registration titled "the mechanism, read off the code (not asserted)". (AMENDMENT 2.)
3. **Process failure: I twice committed to a detached HEAD**, invisible to the MO and Sarah.
   The Bash working directory persists across calls, so one earlier `cd` silently redirected
   every later commit. Both batches published (`9608e8a`, and the earlier merge). My loop now
   verifies the branch before committing.

None of these three changed a constant, and none was found by me first.

## Run pin — read this before comparing numbers

The run is pinned to commit `1b43b89` in a **separate clean checkout**
(`.claude/worktrees/gifted-almeida-4e8a7b`, detached). Reason: PO-1's uncommitted edit to
`vibrational_cascade_module.py` raises `ZeroDivisionError` at `:248` during
`Model6QuantumSynapse` construction, which blocks all full-model construction in the shared
worktree. **PO-1's working tree was not touched.** Filed as
`requests/po1-b2/po3-einvasion-001.md`.

Robustness: the pre-registered quantities are a **retention fraction** on `actin_enlargement`
(pump-independent) and an **`r` ratio** (`P_c` cancels), so the verdict is robust to B2's
backbone `omega_0`/`Q` changes. Only **absolute** `r` depends on them — reported as a limit,
never as if B2 had landed.

## Findings raised this cycle

- **F-2 (routed by the MO to PO-4):** `analytical_gap` does not advance spine plasticity, so
  the shipped multi-trial harness would have reported 100% inter-traversal retention — a
  stopped clock reading as confirmation of `tau_extrude`. This probe steps real physics
  through the gap and never calls it.
- **F-3 (NEW, escalated + queued as Q1):** the L·ETA-3 harness
  (`eta_in_live_trial.py:138-144`) steps presynaptic release once per **agent** step, not per
  **physics** step as the shipped `run_trial:434-441` does — removing ~99% of release
  opportunities (~3.3 expected release events per traversal vs ~350). Measured: `max_glu`
  **0.0000** across a full traversal at `max_act = 0.9950`; corrected, `max_glu = 1.0000` and
  `peak_r` at traversal 2 rose **0.0571 → 0.1428**. This puts L·ETA-3's `ca_open` half of its
  shortfall attribution in question — the ERR-2 class in a new location. Corrected in my probe
  as pre-registration AMENDMENT A1.1, recorded before the run. **L·ETA-3's row is not mine to
  edit and I have not touched it.**

**HARD STOP — the negative branch is Sarah's call.** This PO MEASURES and STOPS. No remedy,
no constant adjusted, no protocol extended to rescue it. Board-level decision. Acknowledged.

**Compute cap:** ONE backgrounded run — in flight, `python -u`, never piped through `tail`,
per-traversal progress to stdout, per-traversal state persisted incrementally to
`results/einvasion_ratchet/` after every traversal, so a kill costs nothing. Not raised.

**Owns:** the actin / `E_invasion` block in `spine_plasticity_module.py`, its `sweep/` probe.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params · PO-2's
`atp_system.py` / phosphate path · PO-4's `analytical_gap`, `run_theta_burst_45s.py` ·
PO-5's `multi_synapse_network.py`, T1′ family · PO-6's surfaces. **No constant written.**

**Unblocks:** PO-5 (HARD).
