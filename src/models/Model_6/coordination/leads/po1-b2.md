# Lead: po1-b2 → **PO-6a** (sweep-harness validity) — OWNED BY THIS PO

## ROTATION — 2026-07-18 19:49:34Z (MO rotation-001, `bdb2d64`)

**B2 CLOSED.** Rotated to **PO-6a — the sweep harness is lying.**
**Now owns:** `sweep_runner.py`, `quantum_dimensions.py`, the orphan modules.
**No longer holds:** `vibrational_cascade_module.py` / backbone params (drop a `requests/` file
like any other PO if they need changing).
**Unit 1 (live):** dimension-consumer audit — every dimension in `quantum_dimensions.py` either
demonstrated to reach a live consumer *by measurement*, or marked INERT with `file:line`.
**Unit 2:** orphan modules + dead fields, `ast`-level proof before any deletion.
**NOT in scope:** the Q × drive sweep (HARD-blocked on PO-2) · Q6 large-D (escalated to Sarah).
**Collision watch:** `model6_core.py:84` holds my `em_coupling_module` import but PO-2 is live in
that file — coordinate via `requests/po2-phosphate/` before touching it at all.

**Heartbeat:** 2026-07-18 19:49:34Z — rotation read, grounding on the two new files done
(`quantum_dimensions.py` 21 dimensions across 4 groups; `sweep_runner.py` apply path read in
full). Building the Unit-1 audit harness now.

---

# (Closed) B2 — retire the per-synapse pump site

**Objective (the done-bar, a MEASUREMENT):** the per-synapse site calls
`bose_einstein_occupation` (`model6_parameters.py:46`) on the `n_ex = n̄_s` form; no
hand-rolled `hbar` anywhere in `vibrational_cascade_module.py`; `kT_ref` and `r_at_E_ref`
grep-provably gone from the live path; **a measurement shows the per-synapse and backbone
pumps agree on the same mode**; T1′ static probe still 7/7; a superseding entry retires
`DISC-1` in the `RESEARCH_LOG_CALCIUM_DIMER.md` DECISION RECORD.

**Status:** **ACCEPTANCE VERIFIED by the MO (`9ebda0e`) + ruling 005 §2 CLOSED (`1f75582`).**
B2 is complete against both the acceptance bar and the pin's second obligation.
**Current unit:** — nothing in flight. Available for the next unit.
**Last heartbeat:** 2026-07-18, after closing ruling 005 §2.
**Blocked on:** nothing. Two decisions sit with Sarah/MO (drive veto; flat-η re-read); neither
blocks me, and I have no further unblocked work on my own surface.

### Ruling 005 §2 — CLOSED (`1f75582`)
`D_modes`, `phi_dissipation`, `chi_redistribution` do **NO physics at either pump site**
post-B2 — verified on executable code with comments/docstrings stripped. Per-synapse: each is
a declaration plus prints (φ also feeds the χ derivation). Backbone: all three are
declarations with **zero reads** in `multi_synapse_network.py`. Live chain consumes ω₀, Q, T
only. Retained but explicitly marked inert; changing them moves nothing.
**Stated limit, not resolved:** `η = (r−1)/(r+1)` is the large-D limit; pin wants D ≳ 200,
sites run 20 and 50. D doesn't enter the formula, so this doesn't change the number — it
bears on whether the large-D *form* applies. Finite-D corrections not derived ⇒ **UNVERIFIED,
now qualifying every η these sites report.** D was NOT raised to fit (MO_MODEL6 §7).
**Two false statements corrected, both mine from `c280e85`** — χ "needed for the nonlinear
term" (the quadratic is deleted) and D/χ "survive as slope parameters" (nothing consumes
them). The program's characteristic defect, caught twice in my own diff.
**Hazard routed to PO-6:** `sweep_runner.py` writes `dendritic_backbone.D_modes` from the
`q1_d_modes` dimension; **nothing reads it.** A sweep over it varies a parameter with no
consumer and returns a flat response readable as a physical null.

### Rulings 002 / 004 — resolved, no action outstanding
Both concerned a broken tree I was holding uncommitted. Resolved at `c280e85`: I did **not**
patch `chi_redistribution` defensively (ruling 002's trap — the ZeroDivisionError dissolved
with the ordered `_critical_threshold` deletion, as ruled), and `model6_core.py` was committed
and released. MO has since confirmed `model CONSTRUCTS OK`. I checked ruling 002's second
point directly: `__post_init__` **does** fire and the derivation **does** land on the instance
the pump uses — φ = 0.8 MHz, χ = 4 kHz on the live object.

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

---

## PO-6a Unit 2 — orphan-module audit (AST-level). **The board's list is wrong.**

**Heartbeat:** 2026-07-18 (see commit timestamps) — Unit 1 delivered; Unit 2 audited, deletions
deliberately HELD (reasons below).

Proved at AST level across **134 parsed files** (import/ImportFrom nodes, not text grep), with
the import *context* resolved so a guarded or conditional import is not miscounted as live:

| module | verdict | evidence |
|---|---|---|
| `eligibility_trace` | **TRUE ORPHAN** — no importer anywhere | deletable; isotope constraint cleared (below) |
| `singlet_dynamics` | **TRUE ORPHAN** — no importer anywhere | deletable, but **held** — see Q7 |
| `calcium_system` | **NOT AN ORPHAN** | imported at `analytical_calcium_system.py:535`, inside `create_calcium_system()` → `else` of `if use_analytical` — a **live conditional fallback** |
| `implicit_diffusion` | **NOT AN ORPHAN** | imported by `calcium_system.py:28` — dependency of the above |
| `em_coupling_module` | imported, never instantiated | `model6_core.py:84` — **PO-2 holds that file**; request filed at `requests/po2-phosphate/po1-6a-001.md` |

**This is why the MO required AST-level proof.** The board lists six orphan modules; deleting
`calcium_system` on that say-so would have broken `analytical_calcium_system.py` — a **live**
module — via its `use_analytical=False` fallback. A text grep would have shown the same import
without revealing it sits on a reachable branch. The second `calcium_system` import
(`:588`) *is* dead-safe — `if __name__ == '__main__'` inside a `try` — which is exactly the
distinction that decides whether deletion is safe, and exactly what grep cannot see.

**The MO's named constraint is CLEARED.** `eligibility_trace.py` carries a P31/P32
parameterisation (`PhosphorusIsotope` enum, `create_P31_module`/`create_P32_module`), but the
isotope kill-switch does **not** depend on it: the live control is the continuous
`environment.fraction_P31` (`model6_parameters.py:896`), consumed at `dimer_particles.py:292,301`
and `model6_core.py:297`, with a preset at `model6_parameters.py:950`. That path is strictly more
general — it supports mixtures, not just two discrete isotopes — and is what the live probes use.

**Deletions HELD this cycle, deliberately, not from caution-as-stalling:**
1. `singlet_dynamics` is the **only reader of `T_singlet_dimer`**, which is the subject of open
   **Q7**. Deleting it now would destroy the evidence for a question the MO/Sarah has not yet
   answered. It goes when Q7 resolves — and if `q2_t2_p31` is re-pointed, the deletion becomes
   trivially safe.
2. `eligibility_trace` is clear to delete on the MO's word. Holding one cycle so the deletion
   lands as its own reviewable commit alongside `singlet_dynamics`, per the "commit deletions
   separately so a revert is surgical" instruction.
3. `calcium_system` / `implicit_diffusion` — **KEEP, with the reason recorded above.** Retiring
   them requires first establishing that `create_calcium_system(use_analytical=False)` is never
   called, which is a separate determination, not a deletion.

**Worth preserving before `eligibility_trace` goes:** it carries isotope T2 figures (P31 ~68 s,
P32 ~0.3 s) that differ from the live `T_singlet_P31/P32`. Provenance, not code — I will log the
numbers in the research log as part of the deletion commit rather than lose them.

---

## PO-6a Unit 3 — heartbeat 2026-07-18 20:36:43Z → complete

**Delivered:** every dimension resolved — reached-and-demonstrated, or INERT-and-labelled.
`427b47c` (labelling), `5dcd224` (log + queue), `de8e0df` (effect-test stage).

**22 dimensions: 13 live, 9 INERT, 2 of the inert declared `critical`.**

**The critical two — investigated first, and the answers differ. I did not invent a consumer.**
- `q2_t2_p31` — **consumer EXISTS but is hardcoded, and the values disagree.** Live dimer
  singlet lifetime is `T_singlet_P31 = 216.0` s at `dimer_particles.py:288`, duplicated at
  `quantum_coherence.py:107`. The swept parameter `quantum.T_singlet_dimer = 500.0` is read
  only by `singlet_dynamics.py:122`, an orphan. **The declared 500 s is not the 216 s the
  model runs.** 216 is inside Agarwal's 100–200 s band; 500 is not. ROUTED to PO-5 (their
  live file, next to the §8 keystone) + queue Q8.
- `q2_j_coupling_hz` — **no consumer, and scale-mismatched to its own name.**
  `J_intrinsic_dimer = 15.0 Hz`: one write, zero reads. J-coupling is live by another route
  (ATP field + per-dimer `N(0.15,0.15)` at `dimer_particles.py:49`, ~100× below 15 Hz).
  Re-targeting = choosing which J is meant. Physics call. ROUTED.
- `q2_k_agg_baseline` — **NOT mechanically fixable.** `k_base = 18918.67` vs declared values
  `[0.001…0.05]`, which match `k_classical = 0.005` exactly: the values were written for a
  dissolution rate. "Fixing the guard" would inject values ~10⁶ off and produce a curve that
  looks like physics. Queue Q9.

**`hasattr` defect CLASS (asked for the class, not the instance):** 46 guarded assignment
blocks, but most are legitimate — optional-subsystem reads and the lazy-init idiom. The
defect is the subset guarding **application of an external input**, where False silently
discards it: `sweep_runner.py:92`, `exp_sensitivity_analysis.py:176-179`. Those must fail
loudly. Same mechanism as the missing `cascade` attribute (B2-1).

**Landed on my own surface:** `INERT_DIMENSIONS` — a *machine-readable* registry (a comment
can be skimmed past; a registry can be asserted against) + `assert_no_inert()`. `sweep_runner`
warns **before** the run and stamps `inert`/`inert_reason` into the **results JSON**, so a
saved file outlives the console warning. It **warns rather than dropping** — silently removing
inert dimensions would hide the defect.

**A miss I caught in my own work, recorded because it is the same class I am auditing:** the
effect test's first version used one global fingerprint and reported "NO EFFECT" for
`q1_n_tryptophan` and `q1_f_coherent_base`. Both are live — they move `collective_field_kT`
(18.6→23.0, 14.0→22.1). The fingerprint was blind to the Q1 channel. **A null from an
instrument that cannot see the channel is a blind spot, not a null** — and it would have
condemned two working dimensions on the same page where I accused the harness of
manufacturing false nulls. Observables are now declared per-dimension, and a non-moving
observable reports **UNDEMONSTRATED**, never INERT.

Also verified while there: B2's drive change did **not** orphan `collective_field_kT` — it
retains live consumers including the gate at `model6_core.py:734`.

**Open with Sarah/MO:** Q7 (nine inert), Q8 (216 vs 500 s — the one that matters beyond the
harness), Q9 (`k_agg` mis-specified). **Deletions still held** pending the isotope question,
per rotation-002.
