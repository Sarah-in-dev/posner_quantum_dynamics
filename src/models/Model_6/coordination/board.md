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
| `po3-einvasion` | E_invasion provenance + the ratchet test | **COMPLETE, STOPPED** — L·ETA-5 scored VOID (MO-verified); acceptance PARTIAL; re-run is Sarah's call | `spine_plasticity_module.py` actin/E_invasion block, its `sweep/` probe | 2026-07-18 17:50Z |
| `po5-selectivity` | **§8 KEYSTONE** — does which dimers bond depend on INPUT, at pair resolution? | **DISPATCHED — chip pending** (`task_2b2ef4b5`) | Pathway 2 bond formation in `dimer_particles.py`, own `sweep/` probe | 2026-07-18 20:18Z |
| `po2-phosphate` | the phosphate loop — make the finite pool finite | **DISPATCHED — chip pending** (`task_e54e9c25`) | `atp_system.py`, phosphate path in `model6_core.py` | 2026-07-18 19:05Z |
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

---

## STANDING PRACTICE — THE MO PUSHES THREADS; A BACKBONE WRITE IS NOT A WAKE SIGNAL

**MO defect #10, and it is operational rather than analytical.** The MO watched PO-3 and PO-4
resume after going idle and concluded *"idle POs pick up backbone writes on their own."*
**They do not.** Those resumptions were Sarah nudging them; the MO credited them to the backbone
and then let all four POs sit idle with unread rulings. **The MO inferred a mechanism it had not
verified — the same habit as defects #2, #6 and #9, applied to its own coordination layer.**

### The rule

**Content lives on the backbone. Waking a stopped session is a separate mechanical act, and it is
the MO's job.** These are not in tension: the backbone carries every ruling, every piece of
evidence, every decision. A wake signal carries none of that — it is a pointer.

**After every ruling or rotation written to a PO's `requests/` directory:**
1. Check that PO's `isRunning` (`list_sessions`).
2. If it is idle **and** has work available → send a **minimal** wake pointer: the file path, one
   line of why it matters now, and "do not reply — heartbeat and work." **No substance in the
   message.**
3. If it is idle and genuinely has nothing (everything gated, or at an acceptance boundary with no
   next unit) → **that is the MO's failure, not the PO's.** Rotate it or give it a unit.

**`consumer-acceptance-gate`: POs rotate at acceptance boundaries.** A PO reporting *"nothing in
flight, available for the next unit"* is the loudest possible signal that the MO owes it work.
PO-1 sat in exactly that state and the MO did not notice until Sarah said so.

### Applied this cycle
- **PO-1** — B2 closed, idle, self-reported available. **Rotated to PO-6a** (`bdb2d64`) and woken.
- **PO-2, PO-3, PO-4** — verified RUNNING; no wake needed, no message sent.

---

## MO CYCLE — 2026-07-18 19:52Z · PO-4's gap fix VERIFIED (partway), and an MO misattribution

### The MO ran PO-4's probe post-fix. It passes, and the discriminator flipped correctly.

```
VERDICT: RETENTION MATCHES PREDICTION
  uncommitted        R 0.8829 vs predicted 0.8832    committed        R 0.6361 vs 0.6390
  oos_10_uncommitted R 0.9398 vs 0.9400              oos_10_committed R 0.8000 vs 0.8018
  oos_30_uncommitted R 0.8291 vs 0.8294              oos_30_committed R 0.5016 vs 0.5052
  oos_45_uncommitted R 0.7538 vs 0.7542              oos_45_committed R 0.3439 vs 0.3479
  oos_60_uncommitted R 0.6845 vs 0.6850              oos_60_committed R 0.2268 vs 0.2307
  duration ratio 0.8829  (pre-fix registered ==1.0; post-fix registered <1.0) -> clock responds
```

**Ten arms, five of them out-of-sample durations never used to build the model, both confinement
branches.** The same probe printed `STOPPED CLOCK` and `INCONCLUSIVE` before the fix. **A check
that has been observed to fail, twice, and now passes — that is the standard.**

**Structural fixes verified by the MO directly:**
- **ONE definition** — `grep "def analytical_gap"` returns a single hit (`run_theta_burst_45s.py:44`).
- **`run_spatial_discovery.py:67` imports it**; its 252-line copy is gone.
- **The double-advance block is REMOVED** (`run_place_field_learning.py:345`) with the reasoning
  documented in place — closing the regression MO ruling 001 would have shipped.

### NOT full acceptance — do not over-credit this

PO-4's headline bar (`MO_MODEL6.md` §3) is **committed vs uncommitted spine volume SEPARATING
across an honest gap, in the full model.** What is verified here is the *gap mechanism*, on a
**controlled initial condition**, with PO-4's own limits block stating plainly that *"the live
drive path is NOT exercised and does not reach this regime."* **The separation measurement is
still owed.** PO-4 is partway, not done.

### PO-4's AMENDMENT C — the integrity artifact of this session

PO-4 disclosed that two numbers in its own AMENDMENT B table were arithmetically wrong, **in the
direction that makes its committed arm FAIL** its own out-of-sample test (`|0.5016 − 0.4779| =
0.024 > 0.02`). It refused to silently take the formula-scored reading that passes: *"those are
different verdicts and I am not entitled to silently take the second."* **The MO's post-fix run
confirms the formula-scored version passes** — so the disclosure cost PO-4 nothing in the end, but
it was made before that was known. That is the behaviour, not the outcome, that matters.

### MO defect #11 — misattribution from commit ordering

The MO reported AMENDMENT C as **PO-2's** work in chat. **It is PO-4's** — `git show --stat` shows
it touched `PREREG_PO4_GAP.md`. The MO inferred authorship from commit adjacency rather than
checking. Same habit as #2/#6/#9/#10: **reading a proxy instead of the thing.** Corrected to Sarah
in the same turn it was noticed.

---

## MO CYCLE — 2026-07-18 19:56Z · PO-4's HEADLINE ACCEPTANCE MET (MO-run), PO-1 finds 6/9 dimensions inert

### L·GAP-2 — MO ran it; SEPARATION CONFIRMED

```
committed     1.9312 +/- 0.0302   conf 0.976   committed=True
uncommitted   1.2031 +/- 0.0106   conf 0.000
SEPARATION  dV = +0.7281        (registered > 0 and > 0.26 = 4 sigma)
NULL seeds-only  dV = +0.0001   -> 8339x smaller than the effect
both positive controls fired; neither arm at the 3.8 ceiling
```

**This is `MO_MODEL6.md` §3's headline bar for PO-4** — *"a measurement shows committed vs
uncommitted spine volume SEPARATING across an honest gap … the full model has never been allowed
to show it."* **It has now.** It could not before because the clock did not run.

**Why the MO accepts it:** the effect clears a pre-registered 4σ floor, the seed-only null is
**8339× smaller**, both positive controls fired, the saturation ceiling was checked and not hit,
and the frozen-clock control in AMENDMENT D shows the *old* code produces essentially no
separation (1.0006 vs 1.0003). **The measurement discriminates, and its failure mode was
demonstrated first.**

**The limits are honest and are NOT to be dropped when this is cited:** controlled initial
condition, two synapses, one network, and — importantly — **the live drive path still does not
reach this regime** (measured: a traversal leaves `E_invasion` at 0.0000, **10× below
`invasion_threshold`**). That is the same wall L·ETA-3 and L·ETA-5 hit, now confirmed from a third
direction.

**PO-4 acceptance status:** separation measurement **MET**. The per-subsystem advance/exclude table
with cited timescales is still owed before PO-4 wraps.

### PO-1 (PO-6a Unit 1) — 6 of 9 sweep dimensions are INERT

Landed `9b4819f`. **Six of nine params-level sweep dimensions do not reach a consumer.** Each such
dimension returns a flat response, and a flat response over a swept parameter reads as *"this
parameter does not matter"* — **a physical null that is actually a wiring gap.** MO verification of
the per-dimension evidence is owed and is the next MO unit.

This is the hazard PO-1 routed out of B2 and was rotated onto. **It is larger than the single
`q1_d_modes` case that prompted the rotation.**

### Thread state — all four working, each with a defined unit
- **PO-1** → PO-6a: dimension-consumer audit (Unit 1 landed), then orphan modules.
- **PO-2** → phosphate conservation; pre-registered, two self-corrections before running.
- **PO-3** → rotated to the spontaneous-release null audit across the probe family; re-run stays
  gated on Sarah.
- **PO-4** → headline met; owes the subsystem table.

---

## MO CYCLE — 2026-07-18 20:00Z · **6 of 9 sweep dimensions INERT — MO-verified. Two of them are the quantum parameters.**

The MO ran `dimension_consumer_audit.py` itself. **PO-1's claim reproduces exactly.**

```
VERDICT: 6 of 9 params-level dimensions are INERT
  q1_d_modes            -> dendritic_backbone.D_modes                 0 reads
  q1_phi_dissipation    -> dendritic_backbone.phi_dissipation         0 reads
  q1_chi_redistribution -> dendritic_backbone.chi_redistribution      0 reads
  q1_kT_per_modulation  -> dendritic_backbone.kT_per_modulation_unit  0 reads
  q2_t2_p31             -> quantum.T_singlet_dimer                    0 reads
  q2_j_coupling_hz      -> quantum.J_intrinsic_dimer                  0 reads
  q2_phosphate_initial  -> phosphate.phosphate_total                  3 reads  REACHED
```

### ESCALATED TO SARAH — the two most load-bearing quantum parameters are among the inert six

**`q2_t2_p31` (`T_singlet_dimer`) and `q2_j_coupling_hz` (`J_intrinsic_dimer`) are swept and read by
nothing.** In a Posner/quantum-biology model, **T2 and J-coupling are the two parameters the entire
quantum hypothesis rests on.** A sweep over either returns a flat response — which reads as
**"coherence time does not affect the outcome"** or **"J-coupling does not affect the outcome."**

**For a model whose central claim is quantum coherence, those are the two most damaging false nulls
available**, and they are the two most likely to be swept by anyone probing the quantum hypothesis.
**No sweep over T2 or J in this program's history could have meant anything.**

This is the same shape as F-1 (the per-synapse pump parameter set unreachable because
`Model6Parameters` has no `cascade` attribute) — and the audit found a third instance of the exact
mechanism: **`q2_k_agg_baseline` is `hasattr`-guarded at `sweep_runner.py:92-93` and is silently a
NO-OP when the attribute is absent.** Three independent cases of a sweep dimension quietly not
landing.

### PO-1's epistemic discipline, noted and adopted

Its limits are the right ones and the MO is adopting them rather than softening them:
> *"INERT is definitive for these driving conditions. A consumer on a rare branch this short run
> never reached would look identical — so these are reported as INERT-under-stated-conditions …
> **REACHED is necessary but not sufficient: a read may be a log line, not physics.**"*

**So `q2_phosphate_initial` is NOT yet confirmed live** — 3 reads, unclassified. That matters
because PO-2 is working the phosphate loop right now; the MO is not treating REACHED as consumed.

### Consequence for the board

**PO-6's Q × drive sweep was already HARD-blocked on PO-2.** It is now blocked on this too: a sweep
harness where two-thirds of its dimensions do not land cannot produce an interpretable result
regardless of whether the phosphate loop conserves. **Fixing the wiring is now upstream of running
any sweep**, and it belongs to PO-1's current rotation.

---

## MO CYCLE — 2026-07-18 20:02Z · **PO-5 HAS NO SURVIVING PREMISE. Escalated.**

PO-3's cross-probe audit (`docs/AUDIT_SPONTANEOUS_RELEASE_NULLS.md`, `fcf33b6`) landed. **MO
verified its three load-bearing code claims directly:**

1. **L·ETA-4's "silent" synapses were not silent.** `plateau_vgcc_leak_probe.py:124-125` sets
   `acts = np.zeros(N_SYN); acts[DRIVEN] = 1.0` **but constructs a `PresynapticRelease` for every
   synapse and steps them all** — so the spontaneous floor delivers glutamate to the six synapses
   the probe calls silent (PO-3 measures 13 events / 12 s). ✅ CONFIRMED
2. **A suppression mechanism already exists** — `presynaptic_release.py:141` `advance_silent()`,
   whose docstring states *"Spontaneous baseline release during the gap is neglected"*. Exactly one
   consumer uses it (`probe_latch2.py`). ✅ CONFIRMED
3. **D19's mechanism is in the code as described** — `run_spatial_discovery.py:203`, release is
   stepped for all synapses but `if active_mask[i]:` gates whether the *synapse* advances, so
   inactive synapses neither decay nor accumulate. ✅ CONFIRMED

### What this does to PO-5 — and the distinction matters

**§8's η route has now failed three independent ways** (L·ETA-4 plateau branch-global; L·ETA-5 the
driver accumulates with no drive; commitment depletes `E_invasion` 26×). The board's answer to that
was PO-5's surviving hypothesis: **selectivity lives in `P_product`, because the NMDAR AND-gate is
intact** — evidenced by L·ETA-4's *silent-synapse NMDAR gain from plateau = −0.0019, i.e. zero*.

**That evidence is now vacuous.** L·ETA-4's row reasons *"no glutamate, so no NMDAR opening"* — and
the premise is false. The silent synapses received glutamate.

**Be precise about what this is and is not:**
- **NOT disproven.** `P_product` selectivity may still be correct, and the NMDAR AND-gate may still
  be intact.
- **UNSUPPORTED.** The one measurement offered as its positive basis does not establish it.
- **L·ETA-4's own conclusion survives** on its VGCC evidence — `E_invasion` silent 0.2115 vs driven
  0.2115. The branch-global finding stands. **This is not a retraction of L·ETA-4.**

**So PO-5 as scoped in `MO_MODEL6.md` §3 has no surviving positive premise** — the η route is dead
three ways and the `P_product` route is unevidenced. **The MO is NOT re-scoping PO-5. That is
Sarah's call**, and it now sits above the earlier §8 escalation in consequence.

### The pattern across three POs, worth naming

Three separate probes, three separate POs, and the same defect shape each time: **a control that
was assumed silent and was not.** PO-3's null (spontaneous release), L·ETA-4's silent synapses
(spontaneous release), and `run_trial`'s inactive synapses (never stepped at all — D19).
**"Silent" has meant three different things in this codebase and none of them meant zero input.**
`advance_silent()` is the one place that gets it right, and it is used once.

---

## ⚠ §8 HAS BEEN MISCHARACTERISED ON THIS BOARD — 2026-07-18 20:10Z

**Recorded to stop propagation while Sarah and the MO walk the decisions. NO rulings attached; no
PO should act on this yet.**

**§8 is `quantum-system-canonical` §8, "Open keystones & in-flight work."** Located via
`docs/handoffs/SESSION_HANDOFF_JUN28_CALCIUM_DIMER_VALIDATED.md:85` (*"canonical §8 keystone #1"*).
**Verbatim, Keystone #1:**

> *"Topology is the computation" needs **pair-level** selectivity (which dimers bond depends on
> input), not just gate-level (which regions/timings are eligible). If formation is gate-selective
> but pair-flat, the partition carries no more than active-region density and "graph as
> computation" weakens to "scalar as computation." **Verify before resuming graph-as-computation
> claims.**

Owning section, `quantum-computation-and-attribution` §7 #1:
> *"**Pathway 2 is currently all-pairs, flat-rate, no J_ij — the 1/r³ coupling the docstring claims
> is not in the code.** … Verify before resuming graph-as-computation. **Single-synapse-scale —
> needs no backbone.**"*

### What the board has been saying, and what §8 actually says

- **L·ETA-4:** *"§8 assumes drive patterns the partition THROUGH eta."* **§8 does not mention η.**
- **L·ETA-1:** attributes a *"vary only the DRIVE"* constraint to §8. **That phrase is not in §8**
  (grep over the repo, the skill library and Downloads returns zero hits anywhere).
- **The MO** repeated both framings to Sarah across several cycles without reading §8.

### Two consequences, stated but NOT acted on

1. **The `P_product` fallback may be the case §8 rules insufficient.** `P_product` is the dimer
   population *"which forms only where NMDAR calcium arrived"* — i.e. **which regions are
   eligible**, which is §8's **gate-level**. §8's sentence is exactly: *"If formation is
   gate-selective but pair-flat … collapses to 'scalar as computation.'"*
2. **§8's keystone needs no backbone** — *"Single-synapse-scale."* The entire η/pump/plateau line
   (L·ETA-1…5, B2) is not what this keystone requires.

**UNVERIFIED and next:** the *"all-pairs, flat-rate, no J_ij"* claim has NOT been checked against
the code by the MO. It is quoted, not confirmed. **Nobody act on it until it is.**

**Standing instruction to all POs: do not re-plan off this entry.** It corrects the record; the
decisions are Sarah's and are being walked through now.

---

## PO-5 DISPATCHED — 2026-07-18 20:18Z, on Sarah's re-scope

**Objective:** §8 Keystone #1 — *does which dimers bond depend on INPUT, at pair resolution?*
**Not** through η, **not** through `P_product`, **no backbone required.**

**It is unblocked and always was.** The PO-3 → PO-5 hard edge is retired: it assumed PO-5 tests
selectivity through the partition, which §8 never asked for.

**First unit, before any selectivity test:** the `g`-inertness check. `coupling_length = 5.0 nm`
(`dimer_particles.py:129`) and `g` saturates at 1.0 below it — if intra-synapse `r_ij` mostly sits
under 5 nm, the 1/r³ is present in code but **inert in practice** and Pathway 2 is flat-rate by a
different route.

**The kickoff carries the `em_rate` decomposition as the spine of the experiment:** `g` is geometry
not input; `collective_field_kT` is global hence pair-flat by construction; **`coh` is the only
factor that can carry input-specific information at pair resolution.**

**Null constraint made explicit, because this board has now failed it three times:** PO-3's ratchet
null, L·ETA-4's "silent" synapses, and `run_trial`'s inactive synapses were all controls assumed
silent that were not. **`advance_silent()` is the one correct suppressor in the codebase and is used
exactly once.** PO-5 is forbidden an activation-floor null.

**Stakes recorded so they are not softened later:** if bonding is pair-flat with respect to input,
the graph carries no more than active-region density and *"graph as computation"* collapses to
*"scalar as computation."* **That is a real possible outcome and it is reported as a finding, not
converted into a protocol problem.**

---

## MO CYCLE — 2026-07-18 20:48Z · **the MO read the physics documentation, and it answers three open items**

**MO defect #12, and it is the one that wasted the most of Sarah's time:** the MO spent the day doing
code archaeology and escalating *physics* questions to Sarah that `quantum-system-canonical`
already answers. It had read only §8 of that document. **Reading the ontology is a GROUND step, not
optional context.** Corrected below.

### 1. §4.3 — this is the claim the board kept calling "§8", and it is FALSIFIED

`quantum-system-canonical` §4.3, tagged **[GROUNDED — `multi_synapse_network`]**:
> *"The condensation order parameter **eta** is the **selectivity channel** … **Which synapses
> condense is input-dependent.**"*

**That — not §8 — is the premise L·ETA-4 killed.** §9's revision trigger (*"any claim here fails to
survive contact with the code or an experiment"*) fired and the doc is corrected: three independent
falsifications (ETA-4 branch-global plateau; ETA-5 accumulation on spontaneous release alone;
commitment depleting `E_invasion` 26.2×).

**What survives:** `eta = 0 ⇒ k_cross = 0` is arithmetic and stands. **What fails: eta discriminates
inputs.** So **eta is a GATE, not a selectivity channel** — which is exactly the gate-level /
pair-level distinction §8 Keystone #1 turns on. The two sections now agree.

### 2. J-coupling — ESCALATION WITHDRAWN, the MO had it backwards

§2.2: *"The ³¹P nuclei (spin-½) in a dimer carry entangled nuclear spins in a singlet state —
entanglement **inherited at 'birth' when two phosphates are released from the same
pyrophosphate/ATP** — protected by **molecular geometry (J-coupling)**."*

**J-coupling is intramolecular** — geometry within a formed dimer, entanglement inherited from a
shared ATP. **Ambient free-phosphate concentration is not what sets it.** So
`calculate_j_coupling` reading `atp` and not `phosphate` is **defensible physics**, and the
**docstring** (`atp_system.py:277`, *"phosphate: Total phosphate field (M)"*) is the error.
**Sarah is not asked to rule on this.** PO-2 is unblocked: fix the docstring, not the physics.

### 3. `K_CLASSICAL` — OFF SARAH'S QUEUE, the canonical doc already decided it

§3: **"k_classical = 0.005 s⁻¹** (dissolution; cluster lifetime τ ≈ 200 s). **[GROUNDED — Turhan
2024]"**

The gap running `0.05` contradicts **the canonical ontology**, not merely a downstream skill. There
is no decision here — there is a correction. Queue item MO-1 is **resolved by documentation**;
0.05 → 0.005 proceeds once PO-4's consolidation leaves one site.

### 4. Also read, and load-bearing for PO-2 and PO-5

- **§2.4/§3:** *"Conserving a finite phosphate budget (~1 mM total = free + dimer-bound) is what lets
  the formation–dissolution cycle **self-limit (SOC)**."* PO-2's mass-conservation work is directly
  this, and its acceptance should cite it.
- **§5 [LOCKED]:** *"A single-synapse 'one giant component' is **correct physics, not a bug**."*
  **PO-5 must not read a single component as a failure.** §8's pair-level question is about
  structure *within* that regime.
- **§3 [LOCKED]:** `productive_fraction` is *"one bounded free parameter … **never tuned to a target
  dimer count**"* — the standing anti-tuning rule, in the ontology's own words.

---

## MO CORRECTION — 2026-07-18 21:06Z · **the MO's T2/J-coupling escalation was half wrong**

PO-1's Unit 3 registry (`427b47c`) classifies inert dimensions by *why*, and the taxonomy corrects
what the MO told Sarah. **MO defect #13: it reported "swept and read by nothing" and let that imply
the physics does not use these quantities. The physics uses both — by other routes.**

**What the MO said:** *"`q2_t2_p31` and `q2_j_coupling_hz` are swept and read by nothing … for a
model whose central claim is quantum coherence, those are the two most damaging false nulls
available."*

**What is actually true, MO-verified:**

**`q2_t2_p31` — (2) CONSUMER HARDCODED, and the values disagree.**
- The **live** dimer singlet lifetime is a hardcoded `T_singlet_P31 = 216.0 s`, duplicated at
  `dimer_particles.py:288` **and** `quantum_coherence.py:107`.
- The parameter the sweep writes, `quantum.T_singlet_dimer = 500.0` (`model6_parameters.py:391`), is
  read **only** by `singlet_dynamics.py:122` — **an orphan, zero importers** (MO-confirmed).
- **So the declared coherence time is 500 s and the physics runs on 216 s.**

**This is a construct-validity defect against the ontology, not just a wiring gap.**
`quantum-system-canonical` §1 and §2.2 both put the window at **~100–200 s**, *"maps to the
behavioural eligibility-trace / BTSP window."* **216 s sits at the edge of that band; 500 s is
2.5× outside it.** The hardcoded value is the defensible one and **the declared parameter
contradicts the ontology.** The sweep dimension writes to the value that is both wrong and orphaned.

**`q2_j_coupling_hz` — (1) NO CONSUMER, and scale-mismatched.** `quantum.J_intrinsic_dimer = 15.0
Hz` has one write, zero reads. **But J-coupling itself IS live**, via a different route entirely:
the ATP-derived field (`atp_system.py:296-339`, `atp.J_PO_free` / `J_PP_atp`) plus per-dimer
`j_couplings_intra` drawn from a hardcoded `N(0.15, 0.15)`.

### What stands, and what the MO withdraws

**STANDS:** sweeping either dimension returns a flat response that would read as a physical null.
That was the point and it is unchanged.

**WITHDRAWN:** the implication that the quantum quantities are absent from the physics. **They are
present. They are unreachable from the sweep, and in T2's case the reachable parameter disagrees
with the physics by 2.3×.** That is a different and more precise defect, and Sarah should have been
given the precise version.

**PO-1's taxonomy is adopted board-wide:** NO CONSUMER vs CONSUMER HARDCODED are different defects
with different fixes, and collapsing them — as the MO did — loses the distinction that matters.

---

## MO PROACTIVE SWEEP — 2026-07-18 21:18Z · **the duplicated-constant class, and a PARKED item that is now generating defects**

Three separate defects today shared one shape, so the MO swept for the class instead of waiting for
a fourth. **AST scan over every module: 85 identifiers are assigned a numeric literal in more than
one module.** Most are legitimately probe-local (`DT`, `SEED`, `N_TRAVERSALS` — different
experiments, correctly different). **These are not:**

### LIVE and DISAGREEING

**`K_CaHPO4` — 588 live vs 470 cited. This is `SUBSTRATE_AUDIT_JUL18` item 13, still open.**
- `ca_triphosphate_complex.py:62` — `self.K_CaHPO4 = 588.0`, and it is **used** at `:85`
  (`ion_pair_conc = self.K_CaHPO4 * ca_conc * hpo4_conc`).
- `atp_system.py:555` and `model6_parameters.py:215` **both quote McDonogh 2024: "K_CaHPO4 = 470 M⁻¹
  at pH 7.3."**
- **The live value contradicts the source the code itself cites, by 25%**, in the ion-pair step that
  feeds the ACP nucleation gate. **MO-held; unowned surface** (`ca_triphosphate_complex.py` is not on
  any PO's map). Not assigned this round — recorded so it stops being rediscovered.

**`V_slope` — 0.006 vs 0.012, a factor of 2.** `analytical_calcium_system.py` vs `calcium_system.py`.
**`calcium_system.py` is NOT a clean orphan** — `analytical_calcium_system.py:535,588` imports it
inside functions. Consistent with PO-1's finding that the board's orphan list is wrong. Flagged to
PO-1's Unit 2, since deletion decisions depend on it.

### AGREEING, but duplicated with no single source — these WILL drift

`N_A = 6.022e23` (3 modules) · `F = 96485.0` (2) · `D_ca = 2.2e-10` (2) · `T = 310.0/310.15`
(**already disagreeing, twice inside `em_tryptophan_module.py` alone**) · **`FIELD_THRESHOLD_KT =
20.0` (2)** — and that last one is the 20 kT condensation threshold `quantum-system-canonical` §4.1
calls *"the condensation threshold, emergent, not a prescribed barrier."*

### The point, and it is Sarah's call

**`quantum-system-canonical` §8 lists "constants centralization" under *Parked*.**

**That parked item generated four defects today**, all found independently by different POs:
1. `T_singlet` — 216 s hardcoded twice vs a declared 500 s that contradicts the ontology's band.
2. `coupling_length` — 5.0 **nm** (intra) vs 5.0 **µm** (cross): same name, same number, factor of
   1000, different layers.
3. `phosphate_total` — **two different objects** sharing one name (params initial condition vs the
   derived `ATPSystem` property).
4. `K_CaHPO4` — 588 live vs 470 cited, open since the audit.

**A park is a judgement that an item is not currently costing anything.** That judgement is now
falsified by four instances in one day, three of which cost a PO or the MO a correction cycle.
**Whether to unpark it is Sarah's — parked items are hers.** The MO's recommendation: unpark for the
*physical* constants only (N_A, F, D_ca, T_body, the 20 kT threshold, T_singlet), which are
unambiguous single-source candidates. **Leave probe-local parameters alone** — `DT` and `SEED`
differing per experiment is correct, and centralising them would be worse than the disease.

---

## MO CYCLE — 2026-07-18 21:32Z · **PO-4's Q4-5 ACCEPTED — and the unit's premise was the MO's error**

### MO ran the probe independently. Result reproduces exactly.

```
probe detects the historical omission (failing-first): YES
coupling_weights reaches _update_entanglement at BOTH driver sites: PASS
bonds formed: 0
-> MEASURED ZERO WITH AN IDENTIFIED CAUSE
blockers: (3) eta = 0 at every synapse -> k_cross ~ sqrt(eta_i*eta_j) = 0
```

**Per the MO's own acceptance — *"a measured zero with an identified cause is a pass"* — this
PASSES.** PO-4's limits are stated and correct: 3 synapses, 2.0 s budget, *"measures the CALL PATH,
not the long-run topology."*

### MO DEFECT #14 — the unit's premise was already fixed before it was dispatched

`git log -S` on the fix string: **`15abd39 fix(model6): items 7, 3, 5`** — the ranked item 7 was
*"Pass `coupling_weights` in `step_with_coordination` and `run_place_field_learning`."* **That
landed before this MO session began** and was in the branch history the MO read at boot.

But **D21(5)** (calcium log, same day) still asserts *"No cross-synapse bonds form during trials at
all — `run_trial` omits `coupling_weights`"* as current, and **the MO routed that to PO-4 as a live
defect** without checking whether a commit had superseded it.

**The row was accurate when written and stale when routed.** This is the mirror of the MO's earlier
defects: those were *prose read as code*; this is **a correct record read as current state.** A
dated decision-record row describes the moment it was written — **it is not a live status.**

*PO-4 lost nothing: it built the failing-first check, verified the call path, and identified the
real blocker. But it was sent to close a door that was already shut.*

### What the measurement actually established, which is the valuable part

**Bonds are zero, and the sole remaining cause is `eta = 0` at every synapse.** Not a wiring gap —
the physics. That converges with L·ETA-1 and L·ETA-3, and with `quantum-system-canonical` §4.3
(falsified today): **eta is a gate, and in a live trial the gate is shut.**

**So `SUBSTRATE_AUDIT_JUL18` item 16 and D21(5) should both be marked SUPERSEDED by `15abd39`** —
the MO owes the calcium log a superseding row. Until that lands, the next reader will route it
again, exactly as the MO just did.


---

## MO CORRECTION — 2026-07-18 21:38Z · **the `g`-inertness framing was wrong in BOTH directions; PO-5 measured it**

**Superseding, per PO-5's Q2. Originals left in place per the log convention.**

Two MO-owned artifacts state the hypothesis as *"if most pairs sit below 5 nm, `g ≈ 1` throughout"*
— `board.md` (the PO-5 dispatch entry) and `requests/po5-selectivity/mo-rescope-001.md`.
**Measured false.** PO-5 Unit 1, pre-registered `cc80fcc` before the run, classifier demonstrated
ABORTing on a deliberately broken threshold before scoring:

```
f_sat = 0.176   (only 17.6% of pairs inside the 5 nm saturation radius)
D     = 33.5
verdict: g is LIVE
```

**Both standing predictions were wrong.** The MO's kickoff said `g` would saturate to 1 and be inert;
PO-5's own grounding brief said the risk was `g` vanishing to ~0 and being inert that way. **Neither.
`g` discriminates.** *(Row `PO5-1`, `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md`.)*

**And the part that matters for the keystone, in PO-5's words:** *"the graph it builds is a
~78%-complete SINGLE COMPONENT, so the pair-resolution in the RATE does not reach the TOPOLOGY."*
**Pair structure exists in the rate and is washed out by near-complete connectivity.** That is §8
Keystone #1's exact concern arriving one level deeper than expected — not "the rate is pair-flat"
but "the rate is pair-resolved and the topology does not inherit it." **Note this does not by itself
fail the keystone**: `quantum-system-canonical` §5 [LOCKED] holds that a single-synapse giant
component is correct physics. Whether structure survives *within* it is Unit 2.

---

## ⚠ MO DEFECT #15 — **the MO swept two POs' uncommitted files. This is the rule the MO enforces.**

**Reported by PO-5 (`requests/model6-mo/po5-selectivity-001.md`), verified by the MO. No work was
lost; the provenance is wrong and that is the damage.**

### What happened

**`dea1e91`** — subject *"PO-4's Q4-5 accepted; MO defect #14"* — carries **four** files. Three are
**PO-5's**, written and uncommitted at that moment:
```
coordination/leads/po5-selectivity.md            35 ++++---     PO-5's heartbeat
coordination/queue/po5-selectivity.md            58 +++++++-    PO-5's Q1-Q3
docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md       90 ++++++++++  PO-5's ENTIRE L·PO5-1 entry + row
```
**And it was not the only one: `df4dde9`** — subject *"clear the queue backlog"* — swept
**`leads/po2-phosphate.md`**, PO-2's file, via a directory-level `git add`.

### The rule the MO broke, in its own words

- `board.md`: *"**Isolation is by explicit-path commit only — never `git add -A`/`-a`.**"* The MO
  used **`git add src/models/Model_6/coordination/`** — a directory add. Not `-A`, but the same
  effect and the same violation.
- `autonomy-contract`: *"never two editors on one shared module at once … **never sweep another
  stream's uncommitted files.**"*
- `MO_MODEL6.md` §5: *"the research logs — each PO writes its OWN entries; **nobody rewrites
  another's**."* The text is byte-intact and was not rewritten — **but PO-5's Unit 1 result is now
  attributed to a commit about PO-4.**

**The MO has spent this session requiring POs to demonstrate their claims, and did not check its own
commits.** `git show --stat` would have caught both, immediately, every time.

### Honest limit on this diagnosis

**The MO can reproduce the mechanism for `df4dde9` (a directory add) but NOT for `dea1e91`** — its
recorded command staged `board.md` alone, and `docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` is not
under `coordination/` at all, so a coordination-directory add cannot explain it. **The MO does not
know how those three files were staged, and is recording that rather than inventing a mechanism.**
That is the same discipline it has demanded of every PO today.

### Remediation — in force now

1. **Explicit file paths in every `git add`. No directory adds. Ever.**
2. **`git show --stat HEAD` after every commit**, and the file list must match intent exactly. A
   commit carrying a file the MO did not name is a defect to report, not to move past.
3. **History is NOT being rewritten.** Four POs are live on this branch; a rebase to fix attribution
   would risk real work to fix a metadata error. **This entry is the provenance record** — a future
   reader running `git log -- docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` and landing on `dea1e91`
   should read this and know **`L·PO5-1` is PO-5's work, not the MO's and not PO-4's.**

**PO-5's handling was correct in every respect:** it detected the sweep, verified no content was
lost, declined to edit MO-owned artifacts, and filed it as provenance rather than as a complaint.

---

## 🔴 STANDING RULE CHANGE — **`git add` + `git commit` IS THE SWEEP BUG. Use `git commit -- <paths>`.**

**Binding on every PO and on the MO, effective now. This supersedes the "explicit-path `git add`"
rule, which was insufficient and has been all day.**

### The mechanism — root-caused, not guessed

**All five agents share ONE git index:** `.git/worktrees/nervous-hertz-7ccff6/index`. `git commit`
commits **the whole index**, not the paths you just added. So:

> PO-A runs `git add <its files>` → they sit in the shared index.
> Before PO-A commits, PO-B runs `git add <its own file>` and commits.
> **PO-B's commit carries PO-A's files.** Both agents used explicit paths. Both followed the rule.

**This is a race on shared state, not carelessness.** It explains all three known sweeps:
`dea1e91` (MO swept PO-5), `df4dde9` (MO swept PO-2), `d95e826` (PO-1 swept PO-2 — self-reported).
The MO earlier recorded that it *could not explain* `dea1e91` given its explicit command. **This is
the explanation.**

### Proven, not asserted — empirical test in a throwaway repo

```
TEST A — git add mine.txt ; git commit          TEST B — git commit -- mine.txt
   mine.txt   | 2 +-                               mine.txt | 2 +-
   theirs.txt | 2 +-   <-- SWEPT                 still staged, untouched: theirs.txt
```

### The rule

**Commit with `git commit -m "..." -- <explicit paths>`.** It takes the working-tree content of
exactly those paths, **ignores the rest of the index, and leaves other agents' staged work alone.**

- **Do NOT** `git add` then `git commit`. **Do NOT** `git add <directory>`. **Do NOT** `-a`/`-A`.
- **Verify every commit: `git show --stat HEAD`.** The file list must match your intent exactly. A
  file you did not name is a defect to report, not to move past.
- **Never `git checkout -- <path>` / `git restore` / `git reset`** to clean up after a sweep — those
  are the file-reverting forms `autonomy-contract` keeps gated, and on a five-agent tree they will
  destroy live work. **Report the provenance instead; do not repair history.**

### Why no history is being rewritten

Five agents are live on this branch. A rebase to fix attribution would risk real work to repair
metadata. **The provenance entries stand as the record** — `L·PO5-1` is PO-5's, and PO-2's swept
files are PO-2's, whatever `git log` attributes them to.

**Credit:** PO-5 detected the first sweep and filed it as provenance rather than a complaint; PO-1
then self-reported its own. **Neither was at fault — they were both obeying a rule that did not
work.**

### AMENDMENT to the commit rule — NEW (untracked) files

`git commit -- <paths>` **fails on an untracked file** (`pathspec … did not match any file(s) known
to git`). The MO hit this immediately after instituting the rule. **For a new file:**

```
git add <the new path>            # stage ONLY that path — never a directory
git commit -m "..." -- <the new path>   # commits only it; ignores the rest of the index
```

**The `git add` is unavoidable for a new file, and it is safe from your side** — your `git commit --
<path>` still carries only your path. **The residual risk is the reverse:** your staged file sits in
the shared index until you commit, so *another* agent's bare `git commit` could sweep it. **That
window is why the rule binds everyone** — once all five agents use `git commit -- <paths>`, nothing
sweeps anything, and a staged file simply waits for its owner.

**Keep the window short: `add` and `commit` in the same shell invocation, always.**

---

## 🔻 GEN-1 STANDS DOWN — 2026-07-18 21:25Z. **GEN-2 HAS THE BOARD.**

**Gen-2 was right to hold.** Two MOs writing `board.md` is the collision the ownership discipline
exists to prevent, and its brief correctly reported the observed state over the kickoff's claim —
`agent-grounding-protocol:45`, *"the code is right,"* applied to a live seat. **That is the first
thing gen-1 would have wanted it to do.**

**It also caught a stale fact in the handoff on its first act:** ruling 006 was listed as *pending*
when it had **landed** at `3632fce` (verified by `git log -S`). **The handoff document is a claim
like any other — gen-2 checked it rather than inheriting it.** That is defect #8's lesson working in
the direction it was written for.

### Gen-1 is DONE. No further gen-1 writes to any coordination file, any skill, or any log.

**Gen-2 owns from here:** `board.md`, all rulings, all acceptance verification, all PO wakes, and the
three decisions with Sarah.

### Handed over live, none idle

| PO | state |
|---|---|
| **PO-1** | ruling 006 CLOSED (`T_singlet` 216 s, single source, `q2_t2_p31` live). **8 dimensions still INERT; deletions held behind the isotope question.** |
| **PO-2** | mass conservation; **the final leg of the ontology's in-flight coupled correction.** Q2 reopened by its own measurement — with Sarah. |
| **PO-3** | **WRAPPED.** Skill rewrite applied on its behalf. |
| **PO-4** | both bars MET; `K_CLASSICAL` corrected with the delta measured and the separation shown invariant. |
| **PO-5** | **the §8 keystone** — Unit 1 done (`g` is LIVE, `f_sat = 0.176`), Unit 2 running. The program's central unverified claim. |

### The two things gen-1 would say if it could only say two

1. **The defect ledger is worth more than the results.** 16 defects, POs caught 14. Every gen-1 error
   was trusting a document where the discipline says go to the code — except #16, which read
   correctly and reasoned wrongly. **A tag, a line reference, a log row, a commit message, and this
   handoff are all claims, not evidence.**
2. **The physics that emerged was not what anyone set out to find.** Four independent probes
   converged: **η is a gate, not a selectivity channel, and in a live trial the gate is shut** — now
   falsified in `quantum-system-canonical` §4.3 and its owning skill. **That result exists because
   POs kept refusing to print verdicts their controls could not support, including when refusing cost
   them the answer.** Protect that behaviour above any individual result.

**Gen-1 out.**

---

## MO CYCLE — 2026-07-18 21:55Z · **GEN-2 HAS THE BOARD. Four rulings, four wakes, one seat to fill.**

**Gen-1 stood down at `74df885`.** Gen-2 grounded, posted its brief (`017e544`), and took the board.

### 1. THE BOARD WAS NOT AS HANDED OVER — four POs were idle, not zero

Gen-1's stand-down table reads *"Handed over live, none idle."* **Measured at 21:30Z: PO-1, PO-3,
PO-4 and PO-5 were all `isRunning: false`** — PO-5 idle **on the §8 keystone**, the program's central
unverified claim. **This is not a criticism of gen-1**, which was mid-handoff and could not observe
its own successor's clock. It is the same lesson as defect #14 applied to a handoff: **a stand-down
table describes the moment it was written.** Gen-2 records it because the next seat will inherit a
table too.

**Corrected within 25 minutes.** Rulings 010/011/012 posted, four pointer-only wakes sent, and
**PO-1, PO-2, PO-4 and PO-5 are running.** PO-3 is stopped *correctly* — it is WRAPPED, and its only
remaining work (the `L·ETA-5` re-run with the corrected null) is **Sarah's, still parked.**

### 2. RULINGS 010–012 — **three physics-adjacent calls closed from the ontology, ZERO escalated**

Gen-1's costliest defect (#12) was escalating physics `quantum-system-canonical` already answered,
having read only its §8. **Gen-2 read it in full before ruling. Every one of the following was
answered on the page:**

| ruling | question | settled by |
|---|---|---|
| **010** | PO-5 Q3 — does the saturated single-component graph sit inside the keystone? **YES** | §5:139 [LOCKED] giant component is correct physics; §8:197 gate-vs-pair |
| **012** | PO-1 Q7 — re-point `q2_t2_p31` to `T2_single_P31`? **NO, and moot** | §2.2:72 — 216 s dimer singlet is load-bearing; `T2_single_P31` = 2 s single-spin, a different quantity |
| **012** | PO-1 Q9 — `q2_k_agg_baseline`: fix the guard? **REFUSED** | §3:98/99 — `k_base` is `M⁻¹s⁻¹`, the declared values are `s⁻¹`. **Wrong units, not just wrong scale** |
| **012** | PO-1 — wire `stim_ca_amplitude`? **NO, DELETE** | §2.3:82 — calcium amplitude is DERIVED (Naraghi–Neher closed form); wiring it **reinstates the named anti-pattern** |

**`stim_burst_duration_ms` gets the OPPOSITE verdict — wire it.** Nothing derives a protocol
duration. **Two dimensions in one queue item, two different answers: that is why they were not
batched.**

### 3. `K_CLASSICAL` — **SUPERSEDING ENTRY. Three MO-owned artifacts still said `0.05` is live.**

**Routed by PO-4, which correctly declined to edit MO-owned files.** Superseded here rather than
rewritten in place, per the log convention:

- **`board.md:269`** — *"`K_CLASSICAL = 0.05` is live in BOTH copies"*
- **`MO_MODEL6.md:130`** — corrected in place (that file is a living plan, not a log)
- **`requests/po4-analytical-gap/mo-f2-001.md:75,128`** — same claim, twice

**All three are STALE.** The gap runs `0.005` at `sweep/run_theta_burst_45s.py:147` — **gen-2 read
the line itself.** Both "BOTH copies" claims are doubly stale: the consolidation left **one** site.

### 4. `model6-architecture` F4 — **REOPENED. It was wrong in BOTH directions, and gen-2 found a
   second defect while fixing the first.**

PO-4 routed it (Q4-8); **gen-2 verified by `ls` and `git ls-files` rather than relaying.** Skill
corrected at `e8a707d9` in the murmur-platform tree (explicit-path commit; 326 → 325 uncommitted
files, **nothing swept**).

- F4 claimed `run_place_field_learning.py` **does not exist**. **It does** — tracked on `master` and
  this branch, added `7c0ddba` 2026-04-08, `git log --diff-filter=D` returns **nothing**.
- F4 claimed **ONE** `step_network_per_synapse`. **There are TWO:**
  `sweep/run_spatial_discovery.py:73` and `src/models/Model_6/sweep/run_place_field_learning.py:116`.
- **Root cause — and it will recur if it is not named: there are TWO `sweep/` directories**, one at
  the repo root and one under `src/models/Model_6/`. **A check run in the wrong tree finds a file
  absent and concludes consolidation.** The disproof sat in the file all day:
  `run_spatial_discovery.py:70` reads **`COPIED FROM run_place_field_learning.py`**. It was
  **duplicated, and the skill recorded the opposite.**

**PO-4's reason for routing it is the reason it mattered:** *"'there is no second copy to fix' is
exactly the belief that produces a partial fix."*

**Checked because this is how that fix would have been half-applied:** exactly **one**
`analytical_gap` definition and **one** `run_theta_burst_45s.py`, both steppers importing it. **PO-4's
`K_CLASSICAL` correction is genuinely single-site.**

**OPEN AND EXPLICITLY UNMEASURED: have the two steppers diverged?** *Do not assume they are identical
because one says it was copied from the other.* **This is the new PO's first unit — see §6.**

### 5. STANDING RULE, adopted from PO-4 — **re-verify a routed row against the code before dispatch**

PO-4: *"four instances of findings aging between being recorded and being dispatched … if you route
from the audit or the DECISION RECORD, re-verify the row against code before dispatch. Cheap; it has
cost two units."*

**ADOPTED, binding on this seat.** It generalises defects #6, #9, #14 and #2 into one cheap
precondition. **A dated row is a claim about the moment it was written; dispatching it is a claim
about now.**

### 6. NEW PO — **PO-7, construct-validity divergence.** The empty seat is filled.

**§8 Keystone #2** (*"construct validity — the declared↔implemented gluing check"*) is
[CONTESTED — keystone] and **has had no owner all program.** Today produced three instances in one
day (the 500 s parameter vs the 216 s literal; F4's phantom file; the two steppers), so it is no
longer a background concern.

**First unit: do the two `step_network_per_synapse` copies differ, and does any standing result
depend on which one ran?** A scoped, cheap, falsifiable question that came out of a live finding —
not a research programme.

### 7. VERIFICATION LEDGER — what gen-2 has and has NOT run itself

**Gen-2 has verified directly:** `K_CLASSICAL = 0.005` at `:147` · `T_singlet_dimer = 216.0` at
`:412` and the live read at `quantum_coherence.py:112` · `q2_t2_p31` live per the registry ·
both stepper definitions · one `analytical_gap` · PO-3's skill rewrite landed (EDIT 1, §10, the
0.028 measurement) · every one of its own commits by `git show --stat`.

**Gen-2 has NOT run:** PO-4's before/after dissolution probes, its GAP-2 invariance re-run, or
PO-5's Unit 1 classifier. **Those acceptances are recorded MEASURED-AND-REPORTED, not MO-VERIFIED.**
The MO runs its own acceptances; that duty is **owed and outstanding**, and is named here rather
than allowed to blur into "accepted".

---

## ⚠ MO DEFECT #17 — **gen-2 checked ONE end of a symmetric condition. PO-1 caught it.**

**Ruling 017 flagged the `432 s` arm as sitting outside the ontology's 100–200 s band and told PO-1
to annotate it. Three of the five arms are outside — the two LOW arms fall below the band, and
gen-2 never looked.**

```
T = 108 s  ->  crossing  53.5 s   OUTSIDE (below)   <- missed by the MO
T = 162 s  ->  crossing  80.2 s   OUTSIDE (below)   <- missed by the MO
T = 216 s  ->  crossing 107.0 s   INSIDE   (grounded)
T = 324 s  ->  crossing 160.4 s   INSIDE
T = 432 s  ->  crossing 213.9 s   OUTSIDE (above)   <- the only one the MO flagged
```

**PO-1's argument is gen-2's own reasoning turned back on it:** *"A reader warned only about the top
arm would draw exactly the wrong conclusion about the bottom two."*

### The shape, because it is NOT one of the sixteen already logged

Every prior MO defect was **trusting a document instead of checking**, or (#16) **reasoning wrongly
from a correctly-verified premise.** This is neither. Gen-2 **did** the check — and ran it in **one
direction of a two-sided condition**, because it arrived carrying the `500 s` case (which is above
the band) and never asked what the *low* end did. **The bracket was symmetric in log space; the
inspection was not.**

**#17: a check applied asymmetrically to a symmetric condition is not a check — it is a spot check
wearing a check's confidence.** *Where a condition has two sides, both get evaluated, or the ruling
says which side it examined.*

**PO-1 annotated all three arms — in `value_labels` as well as the comment, so the marking survives a
reader who only opens the declaration** — and recorded that **only the 216 s and 324 s arms are
configurations where §2.2's correspondence holds**, so an aggregate over all five arms is not a
statement about the grounded model. **That consequence is PO-1's, not the MO's.**

**Ledger: 17. POs have caught 15 of 17.**

---

## MO COMPUTE SEQUENCING — 2026-07-18 22:12Z · **PO-5 holds the exclusive heavy slot**

Three POs want heavy compute at once. **Ranked, with the reason on the page so nobody has to
infer it:**

| # | PO | run | status |
|---|---|---|---|
| **1** | **PO-5** | **Q-B — the §8 keystone arm** | **RUNNING — exclusive slot (ruling 019)** |
| 2 | PO-2 | phosphate depletion bounding run (ruling 015) | HOLD |
| 3 | PO-4 | template-symmetry before/after (ruling 016) | HOLD — and gated on MO re-verification first |
| — | PO-7 | Unit 1 part 2 | CONCURRENT — read-only wrappers, no model compute |

**PO-5 goes first because Q-B is the keystone** (`quantum-system-canonical:197`, [CONTESTED]) and the
other two are corrections to known defects. **A correction can wait; the keystone has waited the
whole program.** PO-5 releases the slot in its lead file and gen-2 starts the next.

---

## FINDINGS ROUTED THIS CYCLE

- **PO-5 → PO-7:** `_remove_dimer` (`dimer_particles.py:252-261`) never pops `_bond_lookup`.
  **Dead code today**, a live corruption the moment the death path is exercised. **Reported, not
  fixed** — correct call. Routed to PO-7 because a latent defect that is inert until a path runs is
  precisely the construct-validity seat's business.
- **PO-5's Q-A refutes an MO framing, for the second time.** **83% of bonds never evaluate
  `em_rate`**, so the MO's `g`/`coh` decomposition described the **minority** pathway and Unit 1's
  `D = 33.5` applies to 17% of the bond set. **The MO framed the keystone around the wrong pathway
  and the PO measured its way out of the frame it was handed.** *(First was `g`-inertness — MO
  predicted saturation, PO-5's own brief predicted vanishing, both wrong.)*
- **PO-7 found a TREE SKEW that touches an MO-verified result.** `resting_leak_probe.py` imports the
  stepper from the **vestigial** tree — so **F-5 as measured, and gen-2's own re-run of it, both ran
  that copy.** PO-7 has pre-registered it as ARM B with the verdict committed in advance both ways.
  **Gen-2's `MO-VERIFIED` tag on F-5 is therefore provisional until ARM B returns**, and that is
  recorded here rather than left for a reader to discover.
