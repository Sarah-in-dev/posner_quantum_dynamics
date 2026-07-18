# Model 6 coordination board — MO-owned (read by all POs; written only by the MO)

**Updated:** 2026-07-18. **Worktree:** `/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/nervous-hertz-7ccff6`
(branch `claude/nervous-hertz-7ccff6`) — NOT the master tree, which is ~20 commits behind.
All POs work in that worktree. Python: `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`.

**The program board is `src/models/Model_6/docs/MO_MODEL6.md`** — six POs with coordinates,
acceptance bars, dependency edges, surface ownership, the kickoff convention, the LOCKED
list. This file is the live registry; that file is the durable plan. Read both.

## Active POs
| PO | Objective | Status | Owner files | Last update |
|---|---|---|---|---|
| `model6-mo` | the MO itself (meta/coordinator) | GROUNDED — brief committed `093c675` | `board.md`, `MO_MODEL6.md` | 2026-07-18 17:30Z |
| `po1-b2` | B2 — retire the per-synapse pump site | **BUILDING** — acceptance probe committed FAILING first `fa12009` (mode ratio 5000, FAIL) | `vibrational_cascade_module.py`, backbone params (`model6_parameters.py:759-805`) | 2026-07-18 17:35Z |
| `po3-einvasion` | E_invasion provenance + the ratchet test | **BUILDING** — pre-registration committed `2084960` (retention fraction, predicted 0.8948) | `spine_plasticity_module.py` actin/E_invasion block, its `sweep/` probe | 2026-07-18 17:50Z |
| `po4-gap` | the analytical gap, biologically grounded | **DISPATCHED — chip pending** (`task_daa62deb`) | `analytical_gap` in BOTH drivers, `run_theta_burst_45s.py` | 2026-07-18 17:52Z |

_(The MO adds a row here + the PO's `leads/`/`queue/` files when it spawns one.)_

**Status vocabulary — read it literally.** `DISPATCHED — chip pending` means the kickoff is
written and the chip is showing, **not** that a PO is running. A PO becomes `LIVE` only when
its session starts and its `leads/<name>.md` carries a heartbeat. The MO does not mark a PO
live off its own spawn call — that would be the producer-green failure applied to dispatch.

**Both kickoffs pin the worktree explicitly** (`cwd` = `.claude/worktrees/nervous-hertz-7ccff6`,
branch `claude/nervous-hertz-7ccff6`) because master is ~20 commits behind.

### MO ERROR AND CORRECTION — 2026-07-18 17:45Z (worktree topology)

The chips created **fresh** worktrees (`mystifying-lichterman-83f6f5` = PO-1,
`gifted-almeida-4e8a7b` = PO-3), both branched off `093c675` — i.e. *without* their own
`leads/`/`queue/` files or these board rows, and without the untracked `.claude/skills`
symlink. Both POs correctly obeyed the pinned worktree in their kickoff and worked in
`nervous-hertz-7ccff6` instead, which left **two POs sharing one working tree** — the exact
collision the surface-ownership map exists to prevent — and that tree at a **detached HEAD**.

**This was the MO's error**, not either PO's: the kickoffs pinned a worktree while the spawn
mechanism was creating its own, and the note above anticipated only the opposite failure.

**Corrected:** `nervous-hertz-7ccff6` reattached to `claude/nervous-hertz-7ccff6` at `895d55f`
with a verified-clean tree (`git status --porcelain` → 0 lines; no work lost). `.claude/skills`
symlinked into all three worktrees.

**Standing arrangement, decided:** both POs continue in `nervous-hertz-7ccff6` on the shared
branch. Their file sets are disjoint (PO-1: `vibrational_cascade_module.py`,
`model6_parameters.py`; PO-3: a NEW file under `sweep/`, with zero edits to
`spine_plasticity_module.py`), and relocating two already-grounded POs mid-flight costs more
than it buys. **Isolation is by explicit-path commit only — never `git add -A`/`-a`.** The
`mystifying-*` / `gifted-*` worktrees are vestigial; leave them.

## Surface-ownership map (collision spine — never edit another owner's files; drop a `requests/` file)
| Surface | Owner |
|---|---|
| `vibrational_cascade_module.py`, backbone params (`model6_parameters.py:759-805`) | **PO-1** |
| `atp_system.py`, the phosphate path in `model6_core.py` | **PO-2** |
| `spine_plasticity_module.py` actin / E_invasion block | **PO-3** |
| `analytical_gap` in the drivers, `run_theta_burst_45s.py` | **PO-4** |
| `multi_synapse_network.py` partition path, the T1' probe family | **PO-5** |
| the orphan modules, `quantum_dimensions.py`, `sweep_runner.py` | **PO-6** |
| the research logs | ALL — each PO writes its OWN entries; nobody rewrites another's |

**Shared-file hazard:** `model6_core.py` is touched by PO-1, PO-2 and PO-4. Only ONE may hold
uncommitted edits at a time. Sequence, or require commit-at-boundary.

## Dependency edges (sequence, route through the MO)
- **PO-2 → PO-6 (HARD).** The sweep tests self-organization; if the phosphate loop is not
  mass-conserving the SOC engine does not exist and the sweep measures nothing.
- **PO-3 → PO-5 (HARD).** At eta=0 there is no partition to be selective (L·ETA-3: zero
  cross-synapse edges all trial).
- **PO-1 ∥ everything** — file-disjoint.
- **PO-4 ∥ PO-1, PO-2** — but PO-4's `K_CLASSICAL` decision touches chemistry rates PO-2
  reasons about. **`K_CLASSICAL` routes through the MO, never through a PO** (50× spread
  across three sites: 0.05 / 0.005 / 0.001). Parked this round.

## Decided, do not re-open
- Dispatch **PO-1 and PO-3 first, in parallel**. PO-2 follows at PO-1's commit boundary.
- PO-3 compute cap: one backgrounded run, per-traversal progress output, `r`-vs-traversal
  persisted incrementally. Raise once with a stated reason; beyond that, escalate.
- **PO-3's negative branch is Sarah's call.** The PO measures and STOPS.

## Human-gated (escalate; do not attempt)
Any physics call · anything on the `MO_MODEL6.md` §7 LOCKED list · anything that would
re-validate a closed result (the T1' probe family is deliberately NMDAR-shut) · a PO moving
a constant to reach an outcome (bounce first, then report) · any acceptance the MO cannot
verify directly.

---

## MO RULINGS — 2026-07-18 17:45Z (both briefs ACCEPTED)

Both grounding briefs pass the gate: line-located verbatim quotes, every named skill covered,
each with a real self-understanding delta. Neither paraphrased. Both are resumed.

**PO-1, open Q1 — delete `_critical_threshold` (`vibrational_cascade_module.py:211-214`)?
RESOLVED BY THE MO: delete it, and delete the init/`__main__` prints that read it.** This is
not a fresh physics call — the may30 pin already retired the Zhang Eq. 4 `r_c` as an artificial
reference scale that →0 in large-D, and that retirement is the stated basis for "there is no
derivation to do". Keeping a live computation of a retired artificial scale is the
scaffolding-with-nothing-underneath the log names. *Sarah can veto in one line; flagged to her.*

**PO-1, open Q2 — the ω₀ 40 GHz → 8 MHz change moves downstream chemistry through
`k_agg_enhanced`. CONFIRMED as proposed: measure and report the delta, do not damp it.**
Damping it would be tuning a constant to protect a downstream result. If it moves a standing
result, that is an escalation, not a regression.

**PO-3, open Q1 — the silence model. APPROVED as recommended:** step real physics through the
gap inside PO-3's own probe; do **not** call `analytical_gap` and do **not** edit it (PO-4's
surface). State the deviation as a limit of the measurement. Protocol choice inside PO-3's own
scope, not a physics call.

**PO-3, open Q2 — CONFIRMED:** fixed inter-traversal gap pre-registered as primary; the
agent's emergent revisit interval reported as descriptive only.

## FINDINGS ROUTED (both escalated to Sarah in chat, 17:45Z)

**F-1 (PO-1) — DISC-1 understates its own scope by one level.** `Model6Parameters` carries no
`cascade` attribute (one hit in `model6_parameters.py`, a comment at `:784`), so
`VibrationalCascadeModule.__init__` always constructs `TubulinCascadeParameters()` defaults.
**The entire per-synapse pump parameter set has never been reachable by `sweep_runner`** — not
just `kT_ref`. No sweep ever run in this program could have varied it. PO-1 carries this into
the DISC-1 superseding entry.

**F-2 (PO-3) — `analytical_gap` does not advance spine plasticity at all**
(`run_spatial_discovery.py:55-78`): actin appears in neither its computed list nor its
"NOT computed" list, so `E_invasion` is silently **frozen** across every inter-trial gap.
**Consequence: the shipped multi-trial harness would have shown a ratchet at 100% retention
instead of the grounded 89% — an artifact of a stopped clock, not evidence of `tau_extrude`.**
This is the program's characteristic defect class sitting directly in the path of the
measurement PO-3 was dispatched to make. PO-3 sidesteps it (above) rather than fixing it.
**The defect itself belongs to PO-4** — see `requests/po4-analytical-gap/mo-f2-001.md`. PO-4 is
not yet spawned; this is now a required input to its kickoff, and it sharpens PO-4's own
acceptance bar (`MO_MODEL6.md` §3 PO-4 already names the 1 ms / 30 s clock freeze — F-2 says
the actin clock is not merely slow there, it does not advance at all).

---

## STANDING DIRECTIVE — THE CONTINUOUS POLL LOOP (MO defect, corrected 2026-07-18 17:50Z)

**This was missing from both kickoffs and that is an MO defect, named in the method:**
`consumer-acceptance-gate:34` — *"Every PO kickoff mandates the continuous poll loop ... A
kickoff without the poll mandate is bounced before spawn. (Scar: 2026-07-10 — the mandate
existed in memory, was omitted from a kickoff, and the PO sat idle with a GO signal in its
inbox.)"* Both POs returned their brief, correctly stopped on an open question, and then sat
idle — because nothing told them to keep polling. The MO then had to spend human clicks on
`send_message` to restart them. **The messaging cost was self-inflicted.** It ends here.

### Binding on every PO, from now, without re-dispatch

1. **Poll this board and your own `requests/` directory at the top of every cycle**, and again
   before you would otherwise end a turn. `board.md` is the MO's channel to you. A ruling
   addressed to you lands here, not in your inbox.
2. **Do not end your turn on an open question.** Write the question to `queue/<you>.md` with
   your recommendation, then **pick up the next non-blocked unit** and keep working. Idling
   with an unanswered question is the failure this directive exists to prevent.
3. **Only stop when everything remaining is genuinely gated** — a Sarah decision, a hard
   dependency edge, or your acceptance is met. Then say so explicitly in `leads/<you>.md` and
   name what would unblock you.
4. **Heartbeat every cycle** into `leads/<you>.md` with a UTC timestamp from `date -u`. The MO
   polls those files; a stale heartbeat is how it detects a stalled PO.
5. **`send_message` to the MO is a last resort, not the channel.** It costs Sarah a click.
   Write to the backbone; the MO is polling continuously.

### Binding on the MO (this seat)

Poll the backbone and the branch continuously — every PO commit, every `leads/`/`queue/`
write, every new `requests/` file. Do not wait to be told. **An idle PO means the MO failed to
give it work.** Verify every claim against the code before accepting it; relaying a PO's
self-report as a finding is the producer-green failure one level up.

## MO VERIFICATIONS — 2026-07-18 17:50Z (both PO findings re-checked against the code)

The MO does not relay. Both findings were re-verified independently:

- **F-1 CONFIRMED.** `grep -n cascade model6_parameters.py` → a single hit, a comment at `:784`.
  `vibrational_cascade_module.py:589-592` reads `if hasattr(params, 'cascade')` … `else:
  self.cascade_params = TubulinCascadeParameters()`. The attribute does not exist, so the
  `else` branch always fires. PO-1's read is exact.
- **F-2 CONFIRMED AND WIDENED — see the addendum in `requests/po4-analytical-gap/mo-f2-001.md`.**
  The docstring's two lists omit actin entirely, as PO-3 said. **But `analytical_gap` is
  DUPLICATED** — `sweep/run_spatial_discovery.py:55` and
  `src/models/Model_6/sweep/run_theta_burst_45s.py:44` — so a fix to one leaves the other live,
  which is *exactly* the partial-fix shape audit item 16 already recorded on this same pair of
  files. Neither PO caught the duplicate.

**Navigation hazard, MO-found, binding on all POs:** there are **two `sweep/` trees** and they
are not copies — `./sweep/` (19 entries) and `./src/models/Model_6/sweep/` (26 entries), with 18
files existing only in the former (including `run_spatial_discovery.py`,
`test_learning_pathway.py`, `verify_metabolic_wiring.py`). **A bare `sweep/...` reference is
ambiguous in this repo.** Every path in a PO return must be repo-root-relative. The MO hit this
resolving PO-3's own citation, which did not resolve as written.

---

## MO CYCLE — 2026-07-18 17:52Z (backbone poll; no messages sent)

**Routed, both via `requests/` rather than `send_message`:**
`requests/po1-b2/mo-ruling-002.md` · `requests/po3-einvasion/mo-ruling-001.md`.

**PO-3 → PO-1 request handled (`requests/po1-b2/po3-einvasion-001.md`).** PO-1's uncommitted
edit makes every `Model6QuantumSynapse` construction raise `ZeroDivisionError` —
`chi_redistribution` is `0.0` when `_critical_threshold` divides by it. **The MO traced it: all
three call sites (`:253`, `:303`, `:672`) plus the definition (`:264`) are inside the code B2
deletes.** Ruling 001 already ordered that deletion, so the crash dissolves with it.
**PO-1 is instructed NOT to patch `chi_redistribution` defensively** — that would be fixing a
bug in code scheduled for deletion, the exact error the B2 framing exists to prevent, and it
would strand a defensive clause in retired physics. *Neither PO saw that the crash was already
inside the deletion set; PO-3 reported the symptom and correctly left the fix to its owner.*

**NEW DEPENDENCY EDGE — PO-1 → PO-3, NUMERICAL, not file-based.** The board records PO-1 as
`∥ everything` because it is file-disjoint. That is still true and it is not sufficient. B2
moves backbone `omega_0`/`Q`, hence `P_c`, hence the **absolute** `r` scale. PO-3's verdict is
ratio-based (`peak_r[8]/peak_r[1]`, in which `P_c` cancels within a run) and retention-based
(pump-independent), so **the ratchet verdict is B2-robust and final**. But PO-3's dispatch
question — *is the 13× shortfall real?* — is an absolute claim against an absolute threshold
`r ≥ 1`. **Any threshold-crossing claim from PO-3 is PROVISIONAL PENDING B2** and must be
labelled so in both the probe output and the log entry. Does not block; does not change the
pre-registration.

**PO-3's separate pinned checkout: APPROVED.** It runs L·ETA-5 against commit `2084960` rather
than the shared tree, because that tree currently carries PO-1's crashing edit. A measurement
taken against a tree with a known crash is not a measurement. Disclosed by PO-3 unprompted.

**Directive violation, flagged to PO-3 (correction, not a strike — the directive postdates its
dispatch):** `leads/po3-einvasion.md` still reads `Last heartbeat: —` while two commits have
landed. A stale heartbeat on a working PO is a false stall signal to the MO's poll.

**Shared-tree state:** PO-1 holds uncommitted edits to `vibrational_cascade_module.py` and the
tree does not construct. No collision (its own file), but PO-4 is arriving and will need to
construct the model. PO-1 instructed to commit at its next boundary rather than hold a broken
tree.

**MO poll mechanism is now armed and running** — a persistent watch on branch commits and on
every `leads/` / `queue/` / `requests/` write. The MO is notified on change; it no longer waits
to be asked. This cycle was triggered by that watch, not by Sarah.

---

## MO CYCLE — 2026-07-18 17:55Z · **the MO was wrong and a PO corrected it**

**PO-4's brief ACCEPTED — and it is the strongest return so far, because it corrected the seat
that dispatched it.**

### The correction, verified against the code

`analytical_gap` does **not** freeze plasticity. Its tail runs
`network.step(0.001, {...})` after jumping `network.time` by the full gap
(`src/models/Model_6/sweep/run_theta_burst_45s.py:284-288`), so actin / `E_invasion` / CaMKII /
DDSC each advance **1 ms per 30 s** — the figure already sitting in `MO_MODEL6.md:128`.
Retention is **0.9999944, not 1.0**. PO-4's framing is right and sharper than the MO's: this is
*worse*, because 99.9994% reads as an even cleaner ratchet than a frozen clock would.

**The MO's error, named:** it verified the **docstring's two lists** and never read the function
tail — **prose checked against prose, in a program whose signature defect is prose that
contradicts code**, and then reported it tagged `[code SHOWN]`. The correction is written into
`requests/po4-analytical-gap/mo-f2-001.md` as a superseding section; the original is left in
place per the log convention.

### Two additions from PO-4, MO-verified

- **A third consumer:** `src/models/Model_6/sweep/run_place_field_learning.py:58` imports the
  `run_theta_burst_45s` definition, calls it at `:343`, and carries `# (analytical_gap doesn't
  advance plasticity dynamics)` at `:347`. **Two definitions, three consumers** — one of which
  documented the defect while continuing to depend on it.
- **The two copies are byte-identical bar one Unicode arrow** (difflib over both 252-line
  bodies). No divergence to preserve.
- **`K_CLASSICAL = 0.05` is live in BOTH copies** — the gap runs the **retired** dissolution
  rate, not the `0.005` the chemistry skill moved to. **MO-owned, still parked, priority raised,
  escalated to Sarah.**

### RULINGS to PO-4

1. **Consolidation APPROVED** — one definition in `run_theta_burst_45s.py`; `run_spatial_discovery.py`
   imports it and its 252-line copy is deleted. This structurally cannot recur as audit item
   16's partial fix, which is a better outcome than dual-patching. **Scope limit:** you touch
   the `analytical_gap` definition, its import, and its call sites — **nothing else** in
   `run_place_field_learning.py`, and delete its stale `:347` comment as part of the fix.
2. **Q1 (dt for the plasticity advance) APPROVED as recommended** — pre-register a
   dt-convergence check against the existing 5 s full-physics validator
   (`run_theta_burst_45s.py:405-415`) rather than asserting `dt_sub=1.0` is fine. There is
   precedent: DECISION RECORD `dt-1` established convergence for `P_S`/edges but explicitly
   **not** for transient-phase counts. Do not assume it transfers.
3. **Q2 `K_CLASSICAL`** — correct to report and not touch. MO holds it.
4. **Q3 (DDSC window changes commitment counts) CONFIRMED** — measure and report the delta, do
   not damp it. Matches the PO-1 Q2 ruling. If it moves a standing result, that is an
   escalation, not a regression.
5. **Do NOT set commitment state analytically.** `model6-commitment-pathway` is LOCKED against
   it and PO-4 surfaced the lock itself. An honest gap advances CaMKII through its own dynamics.

---

## MO CYCLE — 2026-07-18 18:03Z · a cross-PO catch that would have inverted a verdict

**Routed via `requests/` only. No messages.** `requests/po3-einvasion/mo-ruling-002.md` ·
`requests/po4-analytical-gap/mo-ruling-003.md`.

### PO-4 → PO-3 (`po4-conf-001`): the 89% retention prediction is CONFINEMENT-CONDITIONAL

PO-4, establishing the timescales its gap fix must advance, found that L·ETA-5's grounded
prediction `exp(-gap/tau_extrude) ⇒ 0.8948` **is the uncommitted branch only.**
`actin_enlargement` drains by TWO paths (`spine_plasticity_module.py:388-390`) and commitment
switches which one runs; `E_invasion` reads `actin_enlargement` alone (`:412`).

**MO re-derived it independently — every number reproduces:**

| state | conf | τ_eff | retention @20 s |
|---|---|---|---|
| never committed | 0 | 180.0 s | **0.8948** |
| committed (steady) | 0.97561 | 50.9 s | **0.6751** |

3.54× faster drain when committed, and `k_unconf = 0.0005 s⁻¹` means confinement **persists**.

**RULED (not a physics call — choosing between two formulas already in the code, no rate
moves):** PO-3 re-derives its retention prediction conditional on confinement state, logging
`self.confinement` per traversal, **before scoring L·ETA-5.**

**Why this was urgent.** `master` HEAD carries `64346a0 ... flag the mis-derived T1'
pre-registration` immediately followed by `683b82f probe(model6): T1' FAILED with a false
positive`. **A mis-derived pre-registration preceded a false positive in this program's own
recent history.** Same failure, caught before the run this time. Concretely: had PO-3 scored a
committed-spine retention of 0.6751 against a predicted 0.8948, it would have read as **ratchet
FALSIFIED** and its hard stop would have routed a *negative result about the network story* to
Sarah off a wrong number.

*Neither the MO nor PO-3 caught this. PO-4 found it on someone else's surface, handed it over
with the evidence and an explicit "your call, not mine", and touched nothing.*

### Q4-1 RULED — B2's acceptance does NOT span a gap

PO-4 flagged that after B2 the per-synapse pump reads `E_invasion`, so the stopped clock would
freeze **pump drive** across silence too, not just plasticity. Correct. **MO checked the
specific risk:** `sweep/pump_mode_agreement_probe.py` has zero `analytical_gap` / gap /
traversal / `step(` references — a static two-site `n̄` comparison. **B2's acceptance is
uncontaminated.** No re-sequencing. **Standing constraint:** any *pump* measurement spanning a
multi-trial gap before PO-4 lands carries the 1 ms-per-30 s clock and must say so.

### Q4-2 CONFIRMED — the acceptance bar's own numbers are prose-only, and that is the MO's defect

`MO_MODEL6.md:140`'s "1.291 vs 2.389 at +300 s" has **no artifact** — two hits, both
coordination prose. **The MO put unsourced numbers into a PO's definition of done.** PO-4's
reproduction (3.7031 ± 0.0649 committed vs 3.0432 ± 0.0572 uncommitted) disagrees in magnitude
**and ordering**. Ruled: pre-register against the reproduction; the disagreement is the finding.
**`MO_MODEL6.md` §3 PO-4 owes a correction once PO-4 reports.**

### ESCALATED TO SARAH — physics, not the MO's to rule

PO-4 measured `E_invasion` **26× higher in the uncommitted arm** (0.8222 vs 0.0313 at 300 s)
while spine *volume* was higher in the committed arm — the two move in **opposite** directions
under commitment, because commitment redirects enlargement into `actin_stable` and `E_invasion`
reads the transient pool only. Since `r ∝ E_invasion × ca_open`, this says **the condensation
pump is driven by the UNCOMMITTED transient actin pool — a synapse that commits loses pump
drive.** That is architectural, bears on §8 and on PO-5's `P_product` selectivity hypothesis,
and is **Sarah's call.** PO-4 is instructed not to pursue it; it is outside its acceptance.

---

## MO CYCLE — 2026-07-18 18:06Z · a false-FALSIFIED band, and an MO error in a kickoff

**Backbone only.** `requests/po3-einvasion/mo-ruling-003.md` ·
`requests/po5-selectivity/mo-f3-001.md` (pre-positioned for a PO not yet spawned).

### RULING 003 to PO-3 — supersedes 002, apply at scoring, run NOT killed

**Credit first:** PO-3's pre-registration already anticipated confinement — GATE 1 logs `conf`
per traversal and reports **CONFINED-RATCHET** as a distinct outcome rather than folding it into
CONFIRMED. Ruling 002 under-credited that design.

**The hole ruling 002 was aimed at survives anyway, one gate lower.** GATE 1 only fires at
`rho_mean ≥ 0.99`; GATE 2's band is the FIXED `[0.80, 0.95]` from the uncommitted `rho_pred =
0.8948`. Between them sits the partially-confined range. MO computed from PO-3's own constants:

```
conf    rho@20s   GATE1(>=0.99)   GATE2 [0.80,0.95]
0.200    0.8446        -            PASS
0.400    0.7972        -            FALSIFIED   <-- hole opens
0.500    0.7745        -            FALSIFIED
0.976    0.6751        -            FALSIFIED   <-- committed steady state
```

**For every `conf` from ~0.4 to 0.976, a spine retaining EXACTLY what the physics predicts prints
FALSIFIED** — and PO-3's hard stop then routes a false negative-result-about-the-network-story to
Sarah. The inversion arrives through GATE 2, which is why GATE 1 does not catch it.

**Ruled:** centre the band on `rho_pred(conf) = exp(-GAP_S·(k_extrude·(1−conf) +
k_stabilization_max·conf))` using the `conf` already logged, tolerance width unchanged. A
derivation correction; no constant moves and no threshold widens. Registered as AMENDMENT A1.2.
`0.8948` remains exactly right for the `conf → 0` arm — the special case, not the rule.

### F-3 ACCEPTED — and part of it is the MO's error

PO-3 found `sweep/eta_in_live_trial.py` under-delivers glutamate **~100×** (release stepped once
per 0.5 s agent step instead of per 0.005 s physics step; ~3.3 expected release events per
traversal vs ~350). Measured `max_glu` **0.0000 → 1.0000** corrected, `peak_r` traversal 2
**0.0571 → 0.1428**.

**The MO's error:** PO-3's kickoff named that file as *"prior art to reuse, not rebuild"*.
**The MO named prior art without verifying it** — the same failure shape as putting the unsourced
1.291/2.389 pair into PO-4's acceptance bar. Two MO defects of the same class in one session; both
caught by POs, neither by the MO.

**PO-3's handling was correct on every count:** corrected in its own probe only, registered as
AMENDMENT A1.1 **before** the scored run, no verdict threshold moved, and **L·ETA-3's log row left
untouched** — another PO's entry and the verdict under test.

### Provenance verdict ACCEPTED (PO-3 acceptance item 2 MET)

`E_ref` **upgraded** UNVERIFIED → **REPRODUCIBLE, SELF-REFERENTIAL**, with the consequence stated:
not a literature measurement, so *"`E_invasion` is grounded in measurement"* would be false at that
constant. `k_polymerization_max` **INHERITED and 3.57× its own citation**, inheriting commit
identified by `git log -S`. A verdict with a mechanism, not a label.

### PO-5 pre-positioned, NOT dispatched

`requests/po5-selectivity/mo-f3-001.md` written now so the defect cannot be inherited later:
PO-5 must not reuse `eta_in_live_trial.py` as-is, because NMDAR opening is glutamate-gated and
**NMDAR is exactly the channel L·ETA-4 found selectivity surviving in**. A selectivity test on that
harness would test the `P_product` hypothesis on the channel the defect suppresses — a guaranteed
false negative. PO-5 remains HARD-BLOCKED on PO-3 regardless.

---

## MO CYCLE — 2026-07-18 18:12Z · **PO-1's ACCEPTANCE IS VERIFIED — the MO ran it, not PO-1**

B2 landed `c280e85`. Per `consumer-acceptance-gate` the MO verifies directly and never relays a
self-report. **Every item below was executed by the MO in this worktree.**

| acceptance item | MO's independent result |
|---|---|
| per-synapse calls `bose_einstein_occupation` on `n_ex = n̄_s` | ✅ `:289 n_bar_s = bose_einstein_occupation(p.omega_0, p.T_body)` |
| no hand-rolled `hbar` | ✅ imports `hbar` from `model6_parameters:90`; the `1.0546e-34` hits are retirement *documentation* only |
| `kT_ref` / `r_at_E_ref` / `pump_exponent` gone from the live path | ✅ all remaining hits are comments/docstrings describing the retirement |
| `_critical_threshold` deleted (MO ruling 001) | ✅ gone; only a comment at `:263` explaining why |
| **the two pumps agree on one mode** | ✅ **A1 ratio = 1 exactly**; both `omega_0 = 8.000000e+06 Hz` |
| convention agreement | ✅ **A2 ratio = 1.000000**; `n̄ = 8.074185e5` vs CODATA `h·f` reference |
| **the verdict can still fail** | ✅ **BOTH positive controls FIRED** — C1 rejects an injected 40 GHz (ratio 5000); C2 detects the `ℏ·f` shortcut at **6.283189** vs 2π = 6.283185 |
| model constructs (the ZeroDivisionError) | ✅ `CONSTRUCTS OK` — **resolved by the ordered deletion, not by a defensive patch** (ruling 002 held) |
| **T1′ static regression floor 7/7** | ✅ observed edge list ≡ pre-registered `[(0,1),(1,2),(3,4),(4,5),(6,7)]`, `betti0_cross=3`, `sizes=[3,3,2]`, `betti1_cross=0`, `crosscheck_ok=True` |
| DISC-1 superseded | ✅ row **B2-1** in `RESEARCH_LOG_CALCIUM_DIMER.md` |

**This is the first measurement-level acceptance on this board, and it is the standard.** The
positive controls firing is what separates it from `683b82f`: a probe whose comparator has gone
silent returns INVALID, not PASS.

**PO-1's own stated limit is the right one and is not to be softened:** this proves the two sites
**agree**; it does not prove 8 MHz is the correct mode. That remains the May-30 bet (Q≳10,
Pokorný slip-layer vs Foster/Baish). **If the bet is wrong, both sites are now wrong together and
the probe would still say PASS.** Independent corroborations it computed rather than copied:
`P_c = 21.514 fW` (pin 21.5), rest `r = 0.039` (pin 0.04, subcritical), `n̄(8 MHz) = 8.0742e5`
(pin 8.07e5).

### NOT WRAPPED — PO-1 is still editing

`git status` shows `vibrational_cascade_module.py` and the calcium log modified again. The
**acceptance measurement** is met and verified; the **PO** is not at rest. Declaring WRAPPED here
would be the producer-green error in its politest form. **PO-2 therefore stays gated** — the
boundary is a *clean* tree, not a single clean instant.

**Directive violation, PO-1:** `leads/po1-b2.md` still reads `Status: DISPATCHED … chip pending`,
`Current unit: —`, `Last heartbeat: —` — through a landed B2, a passing acceptance and a
superseding log entry. Same violation flagged to PO-3 earlier. Update it.

### An uncomfortable finding about the MO's own grounding

**The `1 ms per 30 s` fact was already in the log the MO was told to skim.** Row **D20** of
`RESEARCH_LOG_CALCIUM_DIMER.md`'s DECISION RECORD reads: *"`analytical_gap` advances the
plasticity clock by **1 ms per 30 s gap** (observed `network.time`=46.5 vs
`spine_plasticity.time`=16.5–31.5)"* — measured, with evidence.

So the sequence was: **the answer was already logged; PO-3 reported "frozen"; the MO "confirmed"
it from a docstring; PO-4 rediscovered it from code.** The MO read that table's top rows and
stopped before D20. `MO_MODEL6.md` §1 sends the MO to those tables precisely so this cannot
happen. **Skimming a decision record is not reading it**, and this is the concrete cost: a
correction cycle across three POs for a fact already written down.

---

## MO CYCLE — 2026-07-18 18:22Z · PO-3's Q2 closed, and a new standing rule on skill writes

**PO-3 Q2 — `E_ref` provenance. RULED, executed, not escalated.** PO-3 found that
`model6-actin-invasion-driver:129` described `E_ref = 1.87` as *"read once off a 3000 s uncommitted
run"* with **no pointer** — so the substrate audit recorded it UNVERIFIED, PO-3's kickoff inherited
that as fact, and it was used to argue the 13× shortfall might not be readable as physics.

**MO verified before writing:** `tests/check_actin_three_pool.py:142-148` is the Phase 5 run
(3000 s, Ca=2.0 µM, drive=0) and `:286-288` prints *"Candidate physical anchor for E_ref (decision
pending)"*. Re-run gives `1.8742` vs coded `1.87`. Skill corrected to **REPRODUCIBLE,
SELF-REFERENTIAL** — reproducible from a named in-repo run, **not a literature measurement**.
Committed `4bba978e3`.

**Not escalated to Sarah:** a `file:line` pointer plus a status-label fix is a factual correction,
not a decision. Escalating it would have been escalating plumbing.

### NEW STANDING RULE — the MO makes all skill-library writes

`posner_quantum_dynamics/.claude/skills` is a **symlink into
`murmur-platform/murmur-platform/.claude/skills`** — a **different program's repo**, currently
carrying **325 uncommitted files across at least four other live seats**. An edit left uncommitted
there can be swept into an unrelated seat's commit.

**Therefore:** a PO needing a `model6-*` skill changed writes a `requests/` file to the MO with the
exact proposed text; **the MO makes the write**, because only the MO is positioned to assess the
cross-repo state. PO-3's instinct not to edit unilaterally was correct — for a reason it could not
have seen from inside this repo.

**The pattern the MO used, for the record:** confirm `.claude/skills/` is clean and the target file
untouched → single-file edit → **immediate explicit-path commit** to minimise the sweep window →
verify `git show --stat` reports `1 file changed`. Confirmed: 1 file, nothing swept.

---

## MO CYCLE — 2026-07-18 18:55Z · defect #8, and an audit of every relay the MO has made

### F-3 WITHDRAWN by PO-3, and the MO had already propagated it

PO-3 measured what it had computed and withdrew F-3's central claim: the glutamate event ratio is
**19×, not ~100×** (19.0 vs 1.0 events/traversal, 20 seeds), and the mechanism is **probably
backwards** — the L·ETA-3 pattern *holds* each release for 100 physics steps, so on exposure
duration it delivers **more** glutamate, not less. *"Starves the NMDARs"* is not established. Its
`max_glu = 0.0000` observation is consistent with a Poisson mean of ~1 event (P(0) ≈ 37%). PO-3's
own words: *"I read a single sample as confirmation of a mechanism I had not measured."*

**MO DEFECT #8 — the worst of the eight.** The MO verified F-1 and F-2 against code and **did not
verify F-3's arithmetic**, then wrote it into `requests/po5-selectivity/mo-f3-001.md` as a required
input to a future kickoff and escalated it to Sarah. **A PO that does not yet exist was one dispatch
away from acting on it.** The MO had *already recorded* defect #3 as "named prior art without
verifying it" — and then propagated an unverified finding built on that same prior art.
**Relaying a PO's number without checking it is producer-green, committed by the seat that enforces
the gate.** Corrected in place, original preserved per the log convention.

### The audit this triggered — every MO relay, checked

Defect #8 is a pattern, not a slip, so the MO audited everything it has relayed upward or written
into a durable artifact:

| relayed claim | status |
|---|---|
| F-1 — `Model6Parameters` has no `cascade` attr | **VERIFIED** by the MO at the time (grep + `hasattr` branch) |
| F-2 — `analytical_gap` "frozen" | **WAS WRONG** — MO checked a docstring, not the tail. Corrected; it ticks 1 ms |
| F-3 — glutamate ~100× | **WAS WRONG, unverified by the MO.** Defect #8, corrected above |
| PO-1's B2 acceptance | **VERIFIED** — MO ran the probe, T1′ floor and construction itself |
| **PO-4's 26× `E_invasion` claim (the physics call with Sarah)** | **WAS UNVERIFIED — now VERIFIED, see below** |

### PO-4's 26× reproduced independently — the physics call stands

The MO re-ran it (5 reps, dt=0.005, 300 s, seeds 1000–1004), rather than continuing to quote it:

```
committed     volume 3.7124 +/- 0.0601   E_invasion 0.0313   conf 0.9735
uncommitted   volume 3.0107 +/- 0.0528   E_invasion 0.8222   conf 0.0000

VOLUME ordering : committed HIGHER  -> True
E_invasion ratio: uncommitted/committed = 26.2x   (PO-4 claimed ~26x)
```

**PO-4's numbers reproduce.** `E_invasion` matches to four decimals in both arms (0.0313, 0.8222);
volumes agree within one standard deviation; the committed arm's `conf = 0.9735` sits on the
0.97561 steady state the MO derived from the constants independently.

**So the escalation to Sarah — commitment depletes pump drive, since `r ∝ E_invasion × ca_open` —
is MO-verified, not relayed.** It is also independently corroborated by D19 (*"commitment buys
durability, not amplitude … drive 0→1 at fixed Ca lowers enlargement 1.447→1.099"*), a different
probe on a different day. **Two independent measurements plus a mechanism derived from the code.**

### Consequence for Sarah's queue

**Decision 3 (the L·ETA-3 correction banner) is WITHDRAWN** — F-3 does not support it. The queue is
four items, and the one at the top is now the best-evidenced claim on this board.

---

## MO CYCLE — 2026-07-18 19:03Z · **L·ETA-5 VERIFIED. The edge to PO-5 is NOT cleared.**

### The MO ran PO-3's scorer itself. Every number reproduces.

```
peak_r by traversal : [0.24877, 0.65751, 1.07214, 0.99114, 1.03435, 1.36434, 1.37288, 1.405]
rho_ratio (SCORED)  : ratio_mean 1.1080   band (0.89, 1.07)
conf at every gap   : 0.0000  -> uncommitted branch correct throughout
gain peak_r[8]/[1]  : 5.6478
positive control    : glutamate 1.0 at target t1, 39 release events   [FIRED]
NULL: max E_invasion 0.450671, peak_r gain 7.4601   (LARGER than the drive arm's 5.6478)
=> INCONCLUSIVE_NULL_RATCHETED
```

**PO-3's report is accurate in every particular.** Cause confirmed independently:
`BASELINE_RATE_HZ = 0.5` (`sweep/presynaptic_release.py:65`) — tonic spontaneous release the
activation floor does not suppress. The MO measured the same floor before PO-3 reported it
(20 events/100 s at `act = 0.0`).

### ACCEPTANCE — PARTIAL, and that is the honest call

- **Item 2 (provenance verdict): MET.** `E_ref` upgraded to REPRODUCIBLE/SELF-REFERENTIAL,
  `k_polymerization_max` INHERITED and 3.57× its own citation, inheriting commit identified.
- **Item 1 (the ratchet measurement): EXECUTED CORRECTLY, QUESTION UNANSWERED.** Pre-registered
  before the run, amended twice before scoring, null arm present, verdict function demonstrably
  capable of six outcomes, positive control fired, scored VOID on its own registered terms.
  **That is a properly conducted measurement that did not answer its question** — a result about
  the instrument, not a failure of the PO.

**PO-3 is NOT WRAPPED and is correctly STOPPED.** The re-run it names — a null suppressing
spontaneous release, and a gap clearing the calcium tail — is a **protocol change**, which is
Sarah's call. PO-3 did not make it unilaterally. That is exactly right.

### ⚠ THE PO-3 → PO-5 EDGE IS **NOT** CLEARED — do not read the threshold crossing as clearing it

`MO_MODEL6.md` §4 blocks PO-5 because *"at η = 0 there is no partition to be selective."*
**Driven `r` did cross threshold** — 1.0721 at t3, 1.4050 at t8, **the first live-regime crossing
since L·ETA-3's 0.0768** — and η reached 0.1684. **But no partition formed, in either arm.**

MO verified the structure rather than taking it on report: `k_cross ∝ √(η_i·η_j)`, and **only one
feature was driven**, so every pair contains an η = 0 term and `√(η_i·η_j) = 0` for **all** pairs.
**Zero cross-synapse edges follows necessarily, not incidentally.**

**Therefore: η ≠ 0 was demonstrated; a PARTITION was not.** PO-5 remains **HARD-BLOCKED**. PO-3's
own phrasing — *"at most PARTIALLY cleared"* — is right, and the MO is recording the stronger form:
**the edge as written in §4 is not satisfied.**

### ESCALATED TO SARAH — the substantive finding, and it is the session's biggest

**`E_invasion` accumulates on tonic spontaneous release alone**, past `invasion_threshold`, growing
even through silent gaps (`rho` up to 2.26), with **null `peak_r` gain 7.46× exceeding the driven
arm's 5.65×**. And the driven/undriven separation **collapses with traversal count: 6.15× → 1.70×.**

**This is a second, independent line of evidence that the `E_invasion → r → η` channel cannot carry
input-selectivity** — and unlike L·ETA-4 it is **plateau-free**, so it is not the same mechanism
arriving twice. L·ETA-4 showed the plateau makes the drive branch-global; L·ETA-5 shows the driver
accumulates without any drive at all. **§8's premise now fails for two independent reasons**, and
PO-5's surviving `P_product` hypothesis is the only route left.

**This is Sarah's to rule on, not the MO's.** It bears on §8, on PO-5's scope, and on whether the
re-run is worth the compute before that ruling.
