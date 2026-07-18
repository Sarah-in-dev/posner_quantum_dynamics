# Lead: po1-b2 → **PO-6a** (sweep-harness validity) — OWNED BY THIS PO

> **MO gen-2 — read `requests/model6-mo/po1-handoff-gen2.md` first** (commit `83230cf`).
> It is the current state of this PO written as *checkable pointers*, and it carries the five
> stated limits that qualify my results, the traps (including the fact that the board's
> six-orphan list is wrong and would break a live module), and my own three errors.
> This lead file is the long history; that handoff is the state.
>
> **Status at handoff:** nothing in flight, nothing blocked, but everything remaining on my
> surface needs a *decision* rather than code (queue Q7/Q9). The one unblocked mechanical
> unit I have not started is the ~151 dead parameter fields. Orphan deletions HELD behind the
> isotope question. Filename still says `po1-b2` because renaming it would break inbound
> pointers, not because the rotation did not happen.

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

---

## Ruling 006 — CLOSED. Heartbeat 2026-07-18 21:07:09Z.

**Commits:** `3632fce` (the fix), `adbf37a` (log + PO-5 disclosure). Both made with
`git commit -m "..." -- <paths>`; verified each touched only my listed files.

**Done, in the ruled order:**
1. **Parameter moved to the physics, one-way.** `quantum.T_singlet_dimer` 500.0 → **216.0**.
   Never the reverse. Provenance and the Werner-crossing arithmetic recorded at the field.
2. **De-duplicated.** The 216.0/0.4 literals in `dimer_particles.py:288` and
   `quantum_coherence.py:107` now read `QuantumParameters`. Watch-out for anyone following:
   the two sites hold **different param objects** — `dimer_particles` has `Model6Parameters`
   (needs `.quantum`), `quantum_coherence` has `QuantumParameters` directly. Added
   `T_singlet_dimer_P32 = 0.4` so the isotope control is a parameter too.
3. **`q2_t2_p31` is live.** Read-trace 0 → **48** reads. Effect demonstrated: driving
   50/216/500 moves `mean_P_S` 0.998512/0.998893/0.998949 — monotonic, correct sign.
   Audit INERT **9 → 8**.

**Bit-identity verified before claiming behaviour was preserved:** post-change
`(695, 0.998892874976)` == pre-change baseline from git, same seed/steps, twelve decimals.
The de-duplication changed nothing — which was the whole requirement.

**Limit I am stating rather than letting the number flatter the result:** the effect is small
at short horizons *by construction* — `T_singlet` is a ~100 s time constant and the probe runs
40 ms. What is demonstrated is that the parameter reaches the physics with the right sign, not
the magnitude a real sweep would see.

**Retained as a regression guard:** the `q2_t2_p31` effect case stays in the audit, so if the
de-duplication is ever reverted to local literals the audit stops moving and says so.

**216 s is not to be retuned** — Agarwal-grounded, load-bearing for `quantum-system-canonical`
§2.2 (now tagged DERIVED on the strength of this finding).

**Boundary note:** I edited PO-5's `dimer_particles.py` (six lines, `:288-289`) after the MO
reassigned the change to me, superseding my own request 001 which had asked PO-5 to do it.
Disclosed in `requests/po5-selectivity/po1-6a-002.md` with the bit-identity evidence so they
can rule it out if their numbers move. Their active regions untouched.

**Remaining INERT: 8.** `q1_d_modes`, `q1_phi_dissipation`, `q1_chi_redistribution`,
`q1_kT_per_modulation` (no consumer — and B2 established these do no physics at all);
`q2_j_coupling_hz`, `q2_k_agg_baseline`, `stim_ca_amplitude`, `stim_burst_duration_ms`
(queue Q7/Q9 — each needs a decision, not a wiring change).
**Deletions still held** pending the isotope question, per rotation-002 and ruling 006.

---

## Ruling 012 — all four CLOSED. Heartbeat 2026-07-18 21:45:19Z.

**Commits:** `b860d49` (the four fixes), `73bfd63` (backbone reply,
`requests/model6-mo/po1-reply-ruling012.md`). **INERT 8 → 7.**

| ruling | action | evidence |
|---|---|---|
| §1 `q2_t2_p31` | moot, NOT re-pointed | already live at `3632fce`; bracket question returned to MO |
| §2 `q2_k_agg_baseline` | DELETE verdict recorded, guard NOT fixed | units argument in registry; held behind isotope gate |
| §3 `stim_ca_amplitude` | DELETE verdict recorded, NOT wired | wiring would reinstate the Naraghi-Neher anti-pattern |
| §3 `stim_burst_duration_ms` | **WIRED** | bit-identical at default; 20/40/100/200 ms → 127/217/396/432 dimers |
| §4 defect class | both guards RAISE | `sweep_runner.py:92`, `exp_sensitivity_analysis.py:176-179`; 44 sites untouched |

**Burst-duration default moved 50.0 → 40.0 one-way** — 50 was declared and never simulated;
40 ms is what ran *and* is the grounded value (4 pulses at 100 Hz = canonical theta burst).
Keeping 50.0 would have silently changed every future default run from 4 pulses to 5.
Bit-identity verified against git before claiming behaviour was preserved.

**Two disclosures carried to the MO rather than left for it to find:**
- `exp_sensitivity_analysis` could **not** be exercised — matplotlib is broken in this venv
  (`pyparsing ImportError`) on **unmodified** code. `py_compile` only.
- The same function has a sibling block **silently dropping a spec right now**
  (`coupling.k_agg_baseline` does not resolve). Ruling named `:176-179`; I kept to it and
  routed the rest rather than expanding scope.

**One correction accepted:** I had framed `stim_ca_amplitude` as "wire it or delete it".
Wiring it was never legitimate — the MO's reading is right and mine was half-wrong.

**Open, not blocking:** the `q2_t2_p31` bracket `[50,100,200,500]` tops out at the **retired**
500 s, whose Werner crossing (247.6 s) falls outside the ontology's band — a sweep would sample
a configuration where the central correspondence fails, with nothing warning. Recommended
re-declaring symmetrically around 216 s as *sensitivity analysis, not value selection*. Not
acted on: choosing a bracket around a load-bearing constant is a physics judgement.

**Next, unless redirected:** the ~151 dead parameter fields (Unit 2's second half) — unblocked
and unstarted. Deletions still held behind the isotope gate.

---

## Ruling 017 — CLOSED. Heartbeat 2026-07-18 22:01:44Z.

**Commits:** `8a345fe` (re-declaration), `4019eeb` (backbone reply).

`q2_t2_p31` bracket → `[108, 162, 216, 324, 432]`, symmetric in log space about the grounded
216 s. **Framing lives in the declaration**, not only the log: block comment and `condition`
string both say *SENSITIVITY ANALYSIS, NOT VALUE SELECTION*, and `source_line = 409` points at
the one-way-fix note in `model6_parameters.py`.

**The ruling's annotation was short by two.** It flagged 432 s as outside the 100–200 s band;
**three of five arms are outside** — 108 s and 162 s fall *below* (crossings 53.5 s / 80.2 s).
Only 216 s and 324 s sit inside. All three marked, in `value_labels` as well as the comment,
and the declaration records that **an aggregate over all five arms does not describe the
grounded model.**

**Crossings derived, not taken:** read the decay law from the code
(`P_S(t) = 0.25 + 0.75·exp(−t/T)`, `dimer_particles.py:283`, `:323-332`) against the `1/√2`
pair floor → `t_cross = 0.49516·T`. It reproduces both MO anchors exactly (107.0, 247.6),
which is what licensed applying it to the other three arms. Then checked every `value_label`
against the derivation rather than against my own comment — 5/5 agree.

**Operational note for the board:** hit an `index.lock` collision committing the reply —
another PO mid-commit. **Waited for it to clear; did not `rm` the lock**, which would have
corrupted their in-flight commit. Worth folding into the commit-rule note: *on `index.lock`,
wait — never remove it.*

**Routing line adopted as stated:** route the choice of a bracket around a load-bearing
constant; do not route mechanical execution of a verdict already given.

**State: nothing in flight, nothing blocked, no open questions to the MO.**
`ALL=22 LIVE=15 INERT=7`.

**Now starting:** the ~151 dead parameter fields (Unit 2's second half) — the last mechanical
work on this surface. Same AST-level evidence standard as the orphan audit; **nothing touching
isotopes gets deleted while that gate is down.**

---

## Dead-fields unit — DELIVERED. Heartbeat 2026-07-18 23:11:58Z.

**Commits:** `7c48696` (the audit), `9f5994c` (SWEEP-4 log + backbone reply).
**Compute: none requested and none needed.** Static AST only — parses files, never imports or
runs the model, so it was safe alongside PO-5's exclusive heavy slot. Finishing this unit
needs no run either.

**220 declared fields · 112 live · 108 DEAD.** Worst: `PNCParameters` **8 of 8**,
`PosnerParameters` 16/18, `MultiSynapseParameters` 12/14, `QuantumParameters` 17/27.
The substrate audit said ~151 — **not a correction**: different method, and mine over-reports
liveness by construction, so 108 is a **lower bound**.

**My own control caught a bug in my own instrument.** v1 counted every string literal as use
and scored the known-dead `kT_per_modulation_unit` LIVE, off `quantum_dimensions.py`'s
`variable="..."` **metadata**. Left uncaught it would have silently suppressed real dead
fields — the class the audit exists to find. Narrowed to `getattr`/`setattr`/`hasattr` args;
both controls pass.

**ROUTED FINDING — ruling 006's defect repeats on the two constants the Werner arithmetic uses:**
- `singlet_thermal = 0.25` (`:412`) — read **only** by `singlet_dynamics.py:129`, an orphan;
  live code hardcodes `0.25` in **three** files.
- `singlet_entanglement_threshold = 0.5` (`:411`) — **DEAD**; live bound is the class constant
  `WERNER_ENTANGLEMENT_BOUND` (`multi_synapse_network.py:94`).

These are the two numbers that selected 216 s over 500 s, and that I re-derived for the
ruling-017 annotation. **Routed, not fixed** — no verdict given, and the literals are in PO-5's
live files. This is the "choice about a load-bearing constant" side of the ruling-017 line.

**New ORPHAN-ONLY class** in the audit: fields live *only* because an orphan reads them. One
today (`singlet_thermal`) — belongs in the orphan deletion batch, not a separate one.

**State:** nothing in flight, nothing blocked, nothing deleted. `ALL=22 LIVE=15 INERT=7`.
**What is left on this surface is gated or needs a decision:** the deletions (isotope gate) and
the seven INERT dimensions. **If the isotope gate lifts I can execute the whole batch — orphan
modules, orphan-only fields, and the two DELETE-verdict dimensions — as one reviewable commit.**
