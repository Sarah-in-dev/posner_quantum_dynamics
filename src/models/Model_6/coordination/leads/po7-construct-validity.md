# Lead: po7-construct-validity (PO-7 · §8 Keystone #2 — declared vs implemented) — OWNED BY THIS PO

**Seated 2026-07-18 21:58Z by MODEL6-MASTER gen-2.**

**Objective (the done-bar, a MEASUREMENT):** for a bounded set of load-bearing quantities, show
**by execution** whether the *declared* model (parameters, docstrings, skills) and the *implemented*
model (the code that runs) agree — and where they do not, show which standing result depends on the
gap. **Never "I read both and they match."**

**Why this seat exists.** `quantum-system-canonical:198` — *"**Keystone #2 — construct validity.**
The declared↔implemented gluing check (§6). [CONTESTED — keystone]"* — has had **no owner** for the
whole program. On 2026-07-18 it produced **three** instances in one day:

1. `T_singlet_dimer` **declared 500 s**, live path running a **216 s literal** — the parameter that
   makes §2.2's central correspondence work. (Closed by PO-1, ruling 006.)
2. `model6-architecture` **F4** asserting a file does not exist **that does**, and **one** stepper
   where there are **two**. (Closed by MO gen-2, `e8a707d9`.)
3. **Nine of nineteen sweep dimensions inert** — values written, never read. (PO-1, PO-6a.)

**That is not background drift. It is the program's characteristic defect class**, and every instance
was found by accident while someone was looking for something else.

---

## FIRST UNIT — **have the two `step_network_per_synapse` copies diverged?**

**The finding that opened it (MO gen-2, verified by direct read — do not re-derive, but DO re-verify
before building on it, per the standing rule):**

- `sweep/run_spatial_discovery.py:73` — defines `step_network_per_synapse`; its own comment at `:70`
  reads **`COPIED FROM run_place_field_learning.py`**.
- `src/models/Model_6/sweep/run_place_field_learning.py:116` — defines it too.
- **Two different `sweep/` directories exist** (repo root, and under `src/models/Model_6/`). That
  conflation is what produced F4's false "consolidated" claim.

**The question, in three parts:**
1. **Do the two implementations differ?** Diff them. A textual diff is the start, not the answer.
2. **If they differ, does the difference change physics** — step order, what advances, what is
   skipped — or is it cosmetic (naming, logging, dead branches)?
3. **Does any standing result depend on which copy ran?** Each copy has its own callers. **This is
   the part that matters**; 1 and 2 are how you get here.

**Verdict must be able to come back "no material divergence."** That is a **passing** result and it
retires a live hazard. **A check that cannot return the boring answer is not a check.**

### What would make this unit fail its own bar
- Diffing the files and reporting the diff. **That is 1 of 3.**
- Concluding "identical, so no result is affected" **without establishing which callers run which
  copy.** Identical *today* does not tell you what ran when a standing result was produced.
- Asserting divergence-in-behaviour from reading. **Gen-1's defect #16: a correctly-verified premise
  does not license an unmeasured conclusion.** If you cannot measure it, say NEEDS MEASUREMENT.

---

## GROUNDING — required before any code (return the brief FIRST)

**Your first returned message is a `### GROUNDING BRIEF`** — line-located verbatim quotes tagged
`[skill X]` / `[recent conversation]` / `[code SHOWN]` / `[data]`, per `agent-grounding-protocol`.
A bare tag is not enough; a fabricated line number mismatches a drifted file, which is the point.
**"I read it; nothing in it bears on this task" is a required and PASSING answer.**

**Read in full:** `session-discipline` · `agent-grounding-protocol` · **`quantum-system-canonical`
(IN FULL — §6 construct validity and §8 are yours, but the seat's costliest recorded defect was
reading only §8)** · `quantum-computation-and-attribution` §6.5 · `model6-architecture` (**note its
F4 entry was corrected today — read the corrected text**) · `model6-codebase-operations` ·
`self-comprehension-discipline` (the proven > observed > inferred spine).

**Prior art to compose from, not rebuild:**
- `sweep/dimension_consumer_audit.py` (PO-1) — **the model to follow.** Read-tracing via
  `__getattribute__`, driving the real model, `reads == 0` as a definitive verdict, **with three
  controls including a calibration against known-LIVE and known-INERT ground truth.** It is the
  program's best example of an instrument that demonstrably discriminates.
- `sweep/gap_phase_coverage_check.py` (PO-4) — a mechanical check that **fails** when a docstring
  and the code disagree. The enforcement pattern: *a rule that only holds when someone re-reads it
  by hand is not enforced.*

---

## THE BARS THAT DO NOT MOVE

- **Acceptance is a MEASUREMENT.** Not "it ran", not "committed", not "errors=0", not a printed
  CONFIRMED. **Two live scars:** `683b82f` printed CONFIRMED off one flickered edge, and a probe
  printed "selectivity holds" while its own positive control never fired. **Demonstrate your check
  CAN FAIL before you are allowed to report that it passed.**
- **Pre-register** the discriminating quantity, the null, the positive control, and a verdict
  function that can return FALSIFIED — **before** the run, committed.
- **Never tune a constant to reach an outcome** (`MO_MODEL6.md` §7 LOCKED). Propose it and you will
  be bounced and reported.
- **Never `datetime.utcnow()`** — `datetime.now(timezone.utc)`.
- **Surgical edits.** You are auditing, not refactoring. **Finding a duplication does not authorise
  de-duplicating it** — report it and let the MO route it.

## OWNS / BOUNDARIES (the collision spine — five agents share one tree)

**Owns:** your own new probe under `sweep/`; your entries in the research logs; this lead + queue.
**Must NOT touch:** PO-1's `vibrational_cascade_module.py`, `sweep_runner.py`, `quantum_dimensions.py`
· PO-2's `atp_system.py` + phosphate path in `model6_core.py` (**live — the final leg of the
ontology's coupled correction**) · PO-4's `analytical_gap` + `run_theta_burst_45s.py` · PO-5's
`dimer_particles.py` Pathway 1/2 + `multi_synapse_network.py` (**live on the §8 keystone**) ·
**both stepper files — READ them, never edit them** · **any skill file** (skill writes are the MO's;
`.claude/skills` is a symlink into another repo carrying other seats' uncommitted work).
**Collision → drop a `requests/<owner>/po7-NNN.md`. Never edit another owner's file.**

## 🔴 THE COMMIT RULE — all five agents share ONE git index

```
git commit -m "..." -- <explicit paths>     # correct
git add <path> ; git commit                 # SWEEPS other agents' staged work
```
New (untracked) file: `git add <that exact path>` then `git commit -- <that path>`, **same shell
invocation**. **Verify every commit with `git show --stat HEAD`** — a file you did not name is a
defect to **report**, not to move past. **Never `git checkout -- <path>` / `restore` / `reset`.**
*Three sweeps happened before this was root-caused; nobody was careless — they obeyed a rule that
did not work.*

## COMPUTE
**PO-2 and PO-5 hold the live heavy work. Never two heavy runs at once.** Your first unit should
need no heavy slot. If you conclude it does, **say so and wait.** Any run: background it with
progress instrumentation, **never pipe through `tail`.** Probes have run 63 and 130+ minutes today.

## ESCALATE, DO NOT DECIDE
Any physics call **the documentation does not answer** (check `quantum-system-canonical` FIRST —
escalating what it already answers is the seat's costliest recorded defect) · anything on §7 LOCKED ·
anything re-validating a closed result (**T1′ is CLOSED, 4/4, p≈3×10⁻⁶**) · any deletion or
de-duplication your audit motivates.

## WORKING PROTOCOL
**Backbone-first: substance goes in `leads/po7-construct-validity.md` and
`queue/po7-construct-validity.md`, never in a chat message.** Heartbeat your lead file as you go.
**Poll your queue** — the MO pushes rulings there and does not expect a chat reply.
**Do not idle silently:** if you are blocked, write the block into your queue and say what would
unblock it.

**Status:** GROUNDED. Unit 1 parts 1 and 3 have SHOWN answers; part 2 is pre-registered, not yet
measured. **Last heartbeat:** 2026-07-18 — grounding brief posted.

---

# GROUNDING BRIEF — PO-7, 2026-07-18

## (a) Verified current state

**`[skill quantum-system-canonical]` — read in full (215 lines), not only §8.** §6 is this seat's
charter, `:163`:
> *"**Construct validity [LOCKED].** The *declared* model (docstrings, skills, this document) and
> the *implemented* model (code) must be reconciled; until they are, a behavior cannot be attributed
> to the claimed physics — it could be an artifact of the gap."*

`:198` — *"**Keystone #2 — construct validity.** The declared↔implemented gluing check (§6).
[CONTESTED — keystone]"*

Two sections bear directly on whether a stepper difference is physics. `:140` (§5):
> *"Cross-bond fidelity carries coupling: F = P_S_i·P_S_j·w_spatial. An edge counts toward
> connectivity/collapse **only if F > 0.5**"*

and `:118` (§4.3), the identity that survived the 2026-07-18 falsification:
> *"the *identity* `eta = 0 ⇒ k_cross = 0` is arithmetic and stands."*

**That second quote is why the divergence below is not cosmetic** — it converts "the backbone
update is missing" from a code-shape observation into a physics prediction I can score.

`[skill quantum-computation-and-attribution]` `:74` (§6.5), the seat's own mandate:
> *"the **declared** model (docstrings, skills) and the **implemented** model (code) have diverged
> … **Build the declared↔implemented gluing check before any behavioral result is load-bearing.**"*

`[skill self-comprehension-discipline]` `:90` — the epistemic spine I am scoring against:
> *"prove what can be proven; observe what can't be proven; infer only what can't be observed"*

and its named characteristic defect, `:95-97`: *"checking something at a **weaker** level than is
available."* **Bearing on this unit:** which copy a caller runs is *provable* from the import
statement — it must NOT be inferred, and must not be answered by running something.

`[skill model6-architecture]` — F4 as corrected today, `:333`:
> *"**Open:** whether the two steppers have diverged in behaviour. **Not measured. Do not assume
> they are identical because one says it was copied from the other.**"*

`[skill session-discipline]` and `[skill agent-grounding-protocol]` — read in full. The
load-bearing line for this unit is `agent-grounding-protocol:45`:
> *"**Code + DB = what IS.** … If a skill says X and the code does Y, the code is right"*

`[skill model6-codebase-operations]` — read; nothing in it bears on the stepper-divergence
question beyond the runtime/backgrounding discipline already carried in the kickoff. That is a
passing answer, recorded as one.

`[recent conversation — board.md]` MO gen-2 cycle 21:55Z §4, the finding that opened this unit:
> *"**OPEN AND EXPLICITLY UNMEASURED: have the two steppers diverged?** *Do not assume they are
> identical because one says it was copied from the other.***"*

`[recent conversation — MO_MODEL6.md §2.3]`, the acceptance bar this unit is graded on:
> *"**A verdict that cannot distinguish its outcomes is not a result.**"*

## (b) Decisions + locked items found

- **§7 LOCKED, `MO_MODEL6.md:271`** — *"No constant tuned to a downstream target."* Nothing in this
  unit proposes a constant change.
- **`MO_MODEL6.md:279`** — *"**T1′ is CLOSED** — 4/4, p≈3×10⁻⁶. Do not re-run."* Relevant: one
  caller of the RSD copy is `sweep/test_learning_pathway.py`, adjacent to that probe family. **I
  will read it, not re-run it.**
- **Prior failure in this exact area:** F4 asserted the place-field file did not exist and that
  there was one stepper. Both false. Root cause named at `board.md` §4 — **two `sweep/`
  directories**, and *"a check run in the wrong tree finds a file absent and concludes
  consolidation."* **My unit is run in the wrong tree by construction if I am careless**, so every
  path below is repo-root-relative and both trees are enumerated explicitly.
- **The commit rule** (`board.md`, standing rule change): `git commit -m "..." -- <paths>`; never
  `git add` then bare `git commit`; verify with `git show --stat HEAD`.

## (c) Prior art to compose from

- `src/models/Model_6/sweep/dimension_consumer_audit.py` (PO-1) — the model for a verdict with a
  calibration against known-LIVE and known-INERT ground truth.
- `src/models/Model_6/sweep/gap_phase_coverage_check.py` (PO-4) — the mechanical docstring-vs-code
  check that FAILS rather than reports.
- `src/models/Model_6/sweep/coupling_weights_reach_probe.py:127-144` — **the closest prior art, and
  it is directly on my target.** It already instruments SITE 1 = `step_network_per_synapse` for
  `coupling_weights` arrival, with an `all_arrived` boolean. Its harness shape is what I extend;
  I do not rebuild it.

## (d) Verdict so far — parts 1 and 3 are ANSWERED FROM CODE; part 2 is NOT

**Part 1 — do they differ? YES, and it is not cosmetic.** Two structural differences, SHOWN:

1. **The backbone update is absent from the place-field copy.**
   `sweep/run_spatial_discovery.py:88-89` runs `network._update_backbone_field()` between the
   per-synapse step and the entanglement tracker. `src/models/Model_6/sweep/run_place_field_learning.py:116-156`
   **has no such call at any line.**
2. **The coordinated gate is called at a different rate.**
   RSD `:101-106` calls `network._evaluate_coordinated_gate(...)` **every step**, with the reason in
   the comment: *"it must also see the falling edge of reward to re-arm its one-shot measurement
   latch, otherwise the measurement fires once per experiment rather than once per reward episode
   (D19)."* RPFL `:148-150` calls it **only when `any_reward` is true** — i.e. **the D19 defect is
   still live in the place-field copy.**

**Part 3 — does any standing result depend on which copy ran? ANSWERED, and the answer is the
boring one.** Every external consumer resolves to the **RSD** copy, provably from the import
statement, not inferred:

| caller | import | resolves to |
|---|---|---|
| `sweep/test_learning_pathway.py:33` | `from run_spatial_discovery import …` | RSD |
| `src/models/Model_6/sweep/gap_clock_assert.py:20` | same | RSD |
| `src/models/Model_6/sweep/resting_leak_probe.py:8` | same | RSD |
| `src/models/Model_6/sweep/eta_in_live_trial.py:58` | same | RSD |
| `src/models/Model_6/sweep/einvasion_ratchet_probe.py:66` | same | RSD |
| `src/models/Model_6/sweep/gap_dissolution_probe.py:53` | `import run_spatial_discovery as RSD` | RSD |
| `src/models/Model_6/sweep/coupling_weights_reach_probe.py:51` | same | RSD |

**Zero files import `run_place_field_learning`.** Its copy has exactly **one** consumer — its own
`:268`, in-file. And exactly one `run_spatial_discovery.py` exists in the repo, so the bare-name
import is unambiguous.

**Consequence, and it is the load-bearing part:** the two copies are **not** a hazard to any
standing result, because **no standing result flows through the place-field copy.** The divergence
is real but quarantined. **The RSD copy is the one every probe on this board has been running.**

**Still to establish before I will call the unit done:** the sys.path claim above is read from
source; I owe a **resolution check that would FAIL if a caller resolved to the other copy** — I will
demonstrate the check failing (by pointing it at a deliberately shadowed path) before reporting that
it passes. `MO_MODEL6.md:53` — *"A verdict that cannot distinguish its outcomes is not a result."*

**Part 2 — does the difference change physics? NEEDS MEASUREMENT, and I am not asserting it.**
The two differences predict, from `quantum-system-canonical:118` (`eta = 0 ⇒ k_cross = 0`), that a
place-field run condenses no backbone and therefore forms **zero cross-synapse bonds** — the same
end-state the file's own `:134-141` audit note already records for the `coupling_weights` omission,
**arriving by a second, independent route.** *That is a prediction from a verified premise, not a
result.* Per the kickoff's own bar — *"a correctly-verified premise does not license an unmeasured
conclusion"* — it is logged as **NEEDS MEASUREMENT**.

**Minimum surgical change proposed:** none to model code. One new read-only probe under
`src/models/Model_6/sweep/`, extending `coupling_weights_reach_probe.py`'s harness. **I am not
proposing to de-duplicate the steppers** — the kickoff reserves that to the MO.

## (e) Open questions (routed to my queue, not decided here)

1. The place-field copy carries a **known-defective** gate call (D19) and no backbone update, and
   has **no consumer but itself**. Whether that is delete / fix / freeze-with-a-banner is a routing
   call, not mine.
2. Does `run_place_field_learning.py` have standing results in the research logs at all? If it does,
   those results were produced by a driver with no backbone condensation **and** the documented
   zero-cross-bond defect — which would make them re-examine-not-re-run, per its own `:140`.

## (f) Self-understanding delta — one sentence

The program believed the open question was *whether* the two steppers diverged; the code shows they
diverge in two physics-bearing ways **and** that it does not matter to any standing result, because
every consumer on the board provably runs the same one — so the live hazard was never divergence,
it was a defective orphan driver nobody imports.

---

# PRE-REGISTRATION — Unit 1 part 2. **Committed BEFORE the probe exists or any run.**

Ruling 018 §3 cleared part 2 with three constraints: verdict must be able to return *no material
divergence*; pre-register before the run; positive control demonstrated to fire. Ruling 018 §4 added
a fourth question — whether F-5 and `mo-ruling-014` get stronger or need re-reading.

**Grounding this against the seat's own bar** (`MO_MODEL6.md:53`): *"A verdict that cannot
distinguish its outcomes is not a result."*

## AN UNPLANNED FINDING FOUND WHILE SCOPING ARM B — recorded here because it changes the unit

Ruling 018 §4 says `resting_leak_probe.py` runs the RSD copy. **True — but not the RSD copy in this
worktree.** `src/models/Model_6/sweep/resting_leak_probe.py:6-7`, SHOWN:

```python
GA='/Users/sarahdavidson/posner_quantum_dynamics/.claude/worktrees/gifted-almeida-4e8a7b'
sys.path.insert(0, GA+'/src/models/Model_6'); sys.path.insert(0, GA+'/sweep')
```

**It hardcodes an absolute path into a THIRD worktree** — `gifted-almeida-4e8a7b`, at detached HEAD
`78b30ef`, the tree `board.md` calls *vestigial*. So F-5 was produced by code in a tree nobody is
working in, imported by a probe living in the tree everyone is working in.

**This is F4's root cause recurring in a new form.** `board.md` §4: *"a check run in the wrong tree
finds a file absent and concludes consolidation."* Here it is not a file-absence error — it is a
**silent version skew**, which is worse, because nothing is absent and nothing errors.

**Measured, not assumed** — `sweep/run_spatial_discovery.py` across the two trees:
- the **`step_network_per_synapse` body is IDENTICAL** across both trees (diff confines to the
  `analytical_gap` region, lines 52–305 vs 54–67);
- the vestigial tree still carries the **inlined `analytical_gap` with `K_CLASSICAL = 0.05`** — the
  value PO-4 corrected to `0.005`. **Irrelevant to F-5, which never calls `analytical_gap`** — but
  it means that tree is a live copy of a retired defect.
- **Dependencies DO differ:** `model6_parameters.py` (T_singlet 500 vs 216), `model6_core.py` (the
  B2 pump-drive rewire), `multi_synapse_network.py` (a warning only).
- **`spine_plasticity_module.py` — the sole writer of `E_invasion` and `actin_enlargement`, the two
  quantities F-5 reports — is BYTE-IDENTICAL across the trees.** The only changed line in any diff
  that mentions `E_invasion` **reads** it as an input to `compute_metabolic_power`; none writes it.

**That is a static argument, and static is not measurement.** It is the premise, not the verdict.
Arm B measures it.

## ARM A — do the two stepper copies differ in BEHAVIOUR? (part 2 proper)

**Setup:** one tree (pinned), one seed, 2 synapses (cross-bonds need ≥2), identical stimulus
sequence including at least one reward-on→reward-off transition (required to exercise D19). Short:
seconds of sim time, no heavy slot.

**Discriminating quantities**, all three recorded per arm:
1. `eta` on the backbone after N steps.
2. cross-synapse bond count.
3. count of `_evaluate_coordinated_gate` invocations (instrumented by wrapper, not by editing model code).

**Pre-registered prediction** (from `quantum-system-canonical:118`, `eta = 0 ⇒ k_cross = 0`):
RPFL never calls `_update_backbone_field`, so **RPFL `eta` stays exactly 0.0 and its cross-bond
count stays 0**, while RSD may be nonzero; and **RPFL's gate-call count is strictly less than
RSD's** across a reward falling edge.

**NULL:** RSD vs RSD — two fresh networks, same seed, same stimuli, both stepped by the *same*
function. **Must return zero divergence on all three quantities.** If the null diverges, the harness
is measuring seed/stochastic noise rather than stepper divergence → **INCONCLUSIVE, report nothing
else.**

**POSITIVE CONTROL, demonstrated to FIRE BEFORE any verdict is reported:** take RSD, wrap it to skip
`_update_backbone_field`, and run it against unmodified RSD. The verdict function **must return
DIVERGENT**. If it does not, the instrument cannot see the very difference it exists to detect →
**ABORT**. *(Scar this guards: a probe printed "selectivity holds" while its own positive control
never fired.)*

**Verdict function returns exactly one of:** `DIVERGENT` · `NO MATERIAL DIVERGENCE` · `INCONCLUSIVE`
· `ABORT`. **`NO MATERIAL DIVERGENCE` is a PASS**, per ruling 018 §3 — a confirmed prediction must
not be written up as a defect discovered.

## ARM B — does the tree skew change what F-5 measured?

**Setup:** replicate `resting_leak_probe.py`'s exact config — 1 synapse, `seed=7`, `dt=0.005`,
V = −70 mV, `glutamate=0.0`, no reward — and run it **twice**: once importing from the vestigial
tree (what F-5 actually ran), once from the pinned tree. **I do not edit `resting_leak_probe.py`**;
it is PO-3's file. I replicate its config in my own probe.

**Discriminating quantity:** `actin_enlargement` and `E_invasion` sampled on a fixed grid, plus the
**threshold-crossing time** — F-5's actual claim.

**Pre-registered prediction:** **identical to floating-point tolerance**, because the sole writer is
byte-identical and no diff hunk writes either quantity.

**NULL:** same tree vs same tree at the same seed → must be bit-identical. If not, the model has
nondeterminism at fixed seed and **the comparison is INCONCLUSIVE** — which would itself be a
finding worth more than the unit.

**POSITIVE CONTROL:** perturb `invasion_threshold` in one arm and confirm the crossing-time
comparator reports a difference. If it does not fire → **ABORT**.

**VERDICT MAPS DIRECTLY ONTO RULING 018 §4, and I commit to stating it either way:**
- crossings agree ⇒ **F-5 and `mo-ruling-014` get STRONGER** — the finding is robust to a
  dependency skew that could have invalidated it, and the tree skew is a hygiene defect with no
  results consequence.
- crossings disagree ⇒ **F-5 NEEDS RE-READING, and so does `mo-ruling-014` built on it** — and I
  say so plainly regardless of how load-bearing the ruling is.

**Neither arm touches model code.** Both are read-only wrappers. **No de-duplication is proposed** —
that stays routed to the MO per ruling 018 §5.

---

## HEARTBEAT — ARM A ABORTED ON ITS OWN POSITIVE CONTROL. **Recorded before it is fixed.**

First run of `po7_stepper_divergence_probe.py` (committed `f2cbadc` *before* the run):

```
NULL   (RSD vs RSD)             -> NO MATERIAL DIVERGENCE  []
POSCTL (RSD vs RSD-no-backbone) -> NO MATERIAL DIVERGENCE  []
ARM A = ABORT: the positive control did NOT fire.
```

**The instrument refused to report a verdict. That is the design working**, and it is the exact
scar the kickoff names — *"a probe printed 'selectivity holds' while its own positive control never
fired."* Mine did not fire and mine printed nothing.

**Root cause — my design error, not a model defect.** The backbone diagnostic from the same run:

```
[backbone diag] P_met=0.84fW  P_agg=0.84fW  P_c=21.51fW  r=0.039  eta=0.0000  invaded=True
```

At 0.5 s of sim time `E_invasion` has barely started, so **`r = 0.039`, `eta = 0.0000` in BOTH
arms.** Removing `_update_backbone_field` from a regime where it already outputs zero changes
nothing. **Two of my three discriminators were degenerate at the duration I chose** — I registered a
control that could not fire in the regime I ran it in.

This is `model6-architecture:115` operating exactly as documented: *"the continuous term honestly
reads `E_invasion`, which **builds over seconds**."* I knew that from grounding and still chose
0.5 s. **The prereg's control was sound in form and untested in placement.**

### Correction — one positive control PER discriminator, plus a regime precondition

The three discriminators do not share a regime, and pairing all of them to one control was the
error:

1. **`gate_calls`** — differs *structurally*, independent of regime (RSD calls the gate every step;
   RPFL only on reward steps). Its control must be a **gate-frequency perturbation**, not a backbone
   skip. Valid at 0.5 s.
2. **`eta_max` / `cross_bonds`** — only distinguishable where the backbone update produces `eta > 0`,
   i.e. `r ≥ 1`. This needs a **regime precondition asserted before the comparison**: if the RSD arm
   itself shows `eta_max == 0`, the discriminator is degenerate and returns **NEEDS MEASUREMENT** —
   **not** a pass. A comparison of two zeros is not a null result, it is no result.

**Arithmetic for the regime, from the model's own constants** (not tuned — read off the diag):
`P_met = P_BASAL + E_inv·ca_open·p_active_max` with `P_c = 21.51 fW`, `p_active_max = 60 fW`,
`P_BASAL ≈ 0.84 fW` ⇒ crossing needs `E_inv·ca_open ≳ 0.345`. At `ca_open ≈ 0.55` that is
`E_inv ≳ 0.63`, and `model6-architecture:119` records `E_invasion` reaching **0.495 by 30 s** and
tracking toward **~0.74 by ~45 s**. **So ~45 s is the shortest arm that can distinguish the
outcomes** — chosen from the constants, before the run, not raised until something happened.

**I am NOT relaxing the verdict to make it fire.** The threshold, the discriminators and the verdict
vocabulary are unchanged from the prereg; only the control placement and the run duration change,
and both are stated here before the corrected run.

---

# UNIT 1 PART 2 — VERDICTS. All runs done; every control shown firing before any pass.

## ARM B — the F-5 tree skew. **VERDICT: NO MATERIAL DIVERGENCE. F-5 GETS STRONGER.**

```
pinned    : crossed_at=56.78  enl=0.175726  E_inv=0.042783
NULL   (pinned vs pinned)  -> NO MATERIAL DIVERGENCE   []
POSCTL (threshold x0.5)    -> DIVERGENT  ['crossing 56.78 vs 28.595']
vestigial : crossed_at=56.78  enl=0.175726  E_inv=0.042783
```

**The positive control fired** (crossing moved 56.78 → 28.595), and **the null was bit-identical**,
so the instrument both discriminates and is stable. The measurement is then **identical to all
printed digits across the two trees.**

**Answering ruling 018 §4 explicitly, as committed:** **F-5 and `mo-ruling-014` GET STRONGER.**
F-5's finding survives a dependency skew that could have invalidated it — the vestigial tree differs
in `model6_parameters.py` (T_singlet 500 vs 216), `model6_core.py` (the B2 pump rewire) and
`multi_synapse_network.py`, and **none of it reaches the measured quantities**, because
`spine_plasticity_module.py` — the sole writer of `actin_enlargement` and `E_invasion` — is
byte-identical across the trees. **My run also independently reproduces the F-5 crossing at 56.78 s**,
consistent with its "well under 100 s".

**The hardcoded cross-worktree path at `resting_leak_probe.py:6-7` remains a real hygiene defect
with no results consequence** — it is PO-3's file and I have not touched it.

## ARM A — the two stepper copies

### Gate discriminator (RNG-independent). **VERDICT: DIVERGENT.**

```
NULL   RSD vs RSD       : 100 vs 100
POSCTL reward-only gate : 100 vs 20  -> FIRED
RSD gate_calls = 100    RPFL gate_calls = 20
```

**Exact arithmetic, not an estimate:** 0.5 s at `dt=0.005` is 100 steps; RSD calls the coordinated
gate every step (100), RPFL only inside the reward window 0.20–0.30 s (20). **The positive control
independently landed on 20** — it replicated RPFL's behaviour exactly, which is a consistency check
I did not design for and got for free.

**So part 2 is ANSWERED for this discriminator: the divergence is real, 5×, and it is the D19
defect** — RSD's own comment says the every-step call exists so the gate *"sees the falling edge of
reward to re-arm its one-shot measurement latch, otherwise the measurement fires once per experiment
rather than once per reward episode."* **RPFL never sees the falling edge.**

### Backbone discriminators (eta, cross_bonds). **VERDICT: INCONCLUSIVE — and why is the finding.**

The corrected 45 s arm **failed its own null**: RSD vs RSD, same seed, same stimuli →
`eta_max 0 vs 0.0708632`, `cross_bonds 1059 vs 585`. The probe returned INCONCLUSIVE and printed no
verdict, per the prereg.

**I did not report that as a model defect on a single in-process observation.** I tested whether it
was my harness — **two separate processes, fresh interpreter, identical seed and config:**

```
PROC 1: eta_max 0.09396788  cross_bonds 1848  dimers 796
PROC 2: eta_max 0.10690230  cross_bonds 1179  dimers 822
```

**It is not my harness.** See the escalation below.

---

# ESCALATION — **the model is NOT reproducible at a fixed seed under drive.** Board-level.

**MEASURED (separate processes, fixed seed):** `cross_bonds` **1179 vs 1848** — a **1.57×** spread on
a topology count. Across all four driven runs today `eta_max` took **0.0, 0.0709, 0.0940, 0.1069** —
i.e. **whether the backbone condenses at all was not reproducible at the same seed.**

**MEASURED (the scope limit, and it matters):** Arm B's resting 1-synapse null was **bit-identical**.
**The nondeterminism is regime-dependent: the resting/E_invasion path is reproducible; the driven
multi-synapse path is not.** This is why F-5 stands and why I am not generalising the alarm.

**SHOWN (code) — three unseeded generators, none reached by `np.random.seed()`:**
- `camkii_module.py:199` — `self.rng = np.random.default_rng()`, drawn at `:300,301,318,388,389,438,439`
- `spine_plasticity_module.py:274` — same constructor; drawn **once**, at `:441-442`
- `multi_synapse_network.py:1188` — same, in `sample_correlated_eligibilities`

`np.random.default_rng()` **with no argument seeds from OS entropy**, so a caller's
`np.random.seed(7)` has no effect on it. **That is a declared-vs-implemented gap of exactly this
seat's class:** drivers accept a `seed=` argument and thread it carefully through
`SeedSequence(...).spawn(...)`, which advertises reproducibility the model does not deliver.

**SHOWN — and it explains the regime split precisely:** `spine_plasticity_module.py:441-442` uses its
rng **only** for `thermal_noise` added to `spine_volume`, gated by `p.stochastic`. It never touches
`actin_enlargement` or `E_invasion` — **which is exactly why Arm B was bit-identical while Arm A was
not.** The prediction and the measurement agree.

**NOT ESTABLISHED, and I am not claiming it:** the causal chain from the CaMKII rng to `eta`. It is
the leading candidate — CaMKII → DDSC commitment → cross-bond dynamics → eta, and DDSC is stochastic
by design (Jain 2024) — but I have **not** measured it. **NEEDS MEASUREMENT.**

## What this bears on — stated as questions for the MO, not as rulings

1. **Any standing result that reads `cross_bonds`, `eta`, or the partition from a SINGLE driven run
   carries an unquantified run-to-run spread.** On today's numbers that spread is up to **1.57× on
   cross-bond count**, and it straddles the condensation threshold. **PO-5's §8 keystone work reads
   exactly these quantities** (`f_sat = 0.176`, the ~78%-complete component).
2. **I am NOT asserting any specific result is wrong.** Single-run *is* legitimate if the quantity's
   spread is small relative to the effect — that is precisely what nobody has measured yet.
3. **The fix is one-line-per-site and I have not made it.** All three files are other POs' surfaces
   (`camkii_module.py`, `spine_plasticity_module.py` = PO-3's block, `multi_synapse_network.py` =
   PO-5's, **live on the keystone**). Per my boundaries I **report and let the MO route it.**
   A fix also silently changes every future run's trajectory, which is a call above this seat.

## Unit 1 closing summary

| part | verdict |
|---|---|
| 1 — do the copies differ? | **YES**, two physics-bearing differences (SHOWN) |
| 2 — does it change physics? | **DIVERGENT** on the gate (5×, exact); **INCONCLUSIVE** on eta/cross_bonds, blocked by fixed-seed nondeterminism |
| 3 — does any standing result depend on which ran? | **NO** — all seven external consumers provably run RSD (proved from import statements) |
| ruling 018 §4 — F-5 / ruling 014 | **STRONGER**, measured across both trees with a control that fired |

**The hazard this seat was opened on is retired. A larger one was found on the way, and it is
escalated rather than fixed.**

---

# HEARTBEAT 2026-07-18 — ruling 023 received. **HOLDING. No unit started, no generator seeded.**

**Compliance first, so it is not buried:** no new unit opened · the three generators are **untouched**
· `test_learning_pathway.py` still never re-run · nothing de-duplicated · PO-4's four modified files
in this tree left alone.

## 1. THE BOUNDARY CORRECTION IS ACCEPTED — and it is sharper than either of us wrote

**My "driven vs resting" boundary is FALSIFIED by the MO's measurement,** and the falsification is
clean: a 2-synapse, 30-drive-step run reproduced to the last particle (1915, te 34.59/32.81, twice at
`3928f2d`, both outcomes pre-registered). **Driven is not sufficient for nondeterminism. My boundary
was drawn from two regimes and generalised past both.**

**But the MO's replacement — "which MODULES a run reaches" — does not survive the code either, and
this matters because it would send the next unit the wrong way.** Established by read, no run:

- `model6_core.py:679` steps CaMKII **every step**, under a comment that says so explicitly:
  *"runs every step regardless of gate"*.
- The draws inside are gated on `p.stochastic`, and **`p.stochastic` defaults to `True`** —
  `camkii_module.py:70,129,153` and `spine_plasticity_module.py:175`.

**So PO-4's probe DOES reach two of the three generators, and DOES draw from them every step — and
it is still bit-identical.** Reach is therefore not the discriminator.

**The correct boundary is per-OUTPUT dependence, not per-run reach.** Three distinct cases, and only
the third is nondeterministic:

| | case | example, measured |
|---|---|---|
| (a) | the generator is never drawn | not yet observed in any run |
| (b) | **drawn, but the measured quantity has no dependence path from the draw** | Arm B: `spine_plasticity` draws every step, yet `E_invasion`/`actin_enlargement` are bit-identical — the draw lands only on `spine_volume` (`:441-442`). **PO-4's probe is also (b).** |
| (c) | drawn, and the measured quantity depends on it | my 45 s arm: `eta`, `cross_bonds` |

**This is the same shape as my Arm B prediction, which is why I believe it:** I predicted from
`:441-442` that `E_invasion` would be immune *while the generator was still being drawn*, and then
measured it immune. **PO-4's probe is that prediction holding a second time on a different output.**

**Consequence for the next unit, stated so it is ready and NOT started:** a module-reach trace would
report PO-4's probe as at-risk and be **wrong**. The question is which *outputs* carry a dependence
path — answerable either statically (taint from each `self.rng` draw to each reported quantity) or
empirically (per-output repeat runs). **I am not choosing between them tonight and not running
either.**

## 2. ONE CORRECTION OWED TO RULING 023 ITSELF — my own number, superseded before you quoted it

Ruling 023 §1 and §3 cite **1.57×** on `cross_bonds`, and §3 builds the new standing rule on it:
*"a quoted delta smaller than the seed-to-seed spread is indistinguishable from noise — and PO-7 has
measured that spread at up to 1.57×."*

**That figure is mine and I superseded it at `306840b`, probably after 023 was drafted.** The
comparable four-run set gives **2.19×** (`cross_bonds` 1179 → 2578); `eta_max` spread is also 2.19×
(0.0487 → 0.1069) and the range **includes 0.0** counting the Arm A null.

**The standing rule is strengthened, not weakened — but the number inside a standing rule should be
the right one**, and a rule quoting a superseded figure is the drift this seat exists to catch.
**Routed as a correction to the MO's artifact, not edited by me.**

## 3. WHAT I AM NOT DOING, EXPLICITLY

- **Not seeding the three generators.** Agreed on every ground given — and the one I would add is
  that `p.stochastic = True` is a **declared modelling choice**, so seeding is a decision about which
  stochasticity is physics (DDSC, Jain 2024, `quantum-system-canonical:131`) and which is accidental.
  That is Sarah's call, not a hygiene fix.
- **Not starting the boundary unit** despite having its correct shape above.
- **Not touching PO-5's or PO-3's surfaces**, and not re-running anything of PO-4's.

**Status:** Unit 1 COMPLETE and ACCEPTED. **HOLDING for Sarah's ruling on the seeding question.**

---

# CLOSING HEARTBEAT — PO-7 construct-validity. 2026-07-18. **Seat wrapped.**

Written on the MO's closing request. **Status: Unit 1 COMPLETE and ACCEPTED (ruling 023). Held from
23:45Z for a seeding decision that did not come; hold released into this wrap. No unit started, no
generator seeded, nothing de-duplicated.**

## 1. UNIT 1 — final verdict in one place

**Question:** do the two `step_network_per_synapse` copies differ, does the difference change
physics, and does any standing result depend on which one ran?

**Part 1 — they differ, in two physics-bearing ways (SHOWN):**
1. `sweep/run_spatial_discovery.py:88-89` calls `network._update_backbone_field()`;
   `src/models/Model_6/sweep/run_place_field_learning.py:116-156` **has no such call at any line.**
   Physics via `quantum-system-canonical:118` — *"`eta = 0 ⇒ k_cross = 0` is arithmetic and stands."*
2. RSD `:101-106` calls the coordinated gate **every step** (its comment cites the D19 falling-edge
   latch re-arm); RPFL `:148-150` calls it **only on reward steps**. **D19 is live in the RPFL copy.**

**Part 2 — split, and the split is the finding:**
- **Gate discriminator: DIVERGENT, exact.** Null 100 vs 100; positive control FIRED (100 → 20);
  measurement **RSD 100 vs RPFL 20** over the same 100 steps. Closed-form: 0.5 s at `dt=0.005` is 100
  steps, reward window 0.20–0.30 s is 20. The positive control independently landed on 20.
- **Backbone discriminators (`eta`, `cross_bonds`): INCONCLUSIVE — cause stated and measured.** The
  45 s arm **failed its own null**. Not reported as a model defect on that alone: two separate
  processes at a fixed seed diverged, so it is not the harness. **The instrument underneath is
  nondeterministic; see §2.**

**Part 3 — NO standing result depends on which copy ran.** All seven external consumers resolve to
the RSD copy, **proved from import statements, not inferred and not run**:
`sweep/test_learning_pathway.py:33` · `gap_clock_assert.py:20` · `resting_leak_probe.py:8` ·
`eta_in_live_trial.py:58` · `einvasion_ratchet_probe.py:66` · `gap_dissolution_probe.py:53` ·
`coupling_weights_reach_probe.py:51`. **Zero files import `run_place_field_learning`;** its copy's
only consumer is its own `:268`. **The hazard this seat was opened on is retired.**

**Arm B (F-5 tree skew): NO MATERIAL DIVERGENCE — F-5 and ruling 014 STRONGER.** Null bit-identical,
positive control FIRED (crossing 56.78 → 28.595), measurement identical to all printed digits across
both worktrees. F-5's `MO-VERIFIED` tag restored on this. `resting_leak_probe.py:6-7` still hardcodes
an absolute path into the vestigial `gifted-almeida-4e8a7b` tree — **hygiene defect, no results
consequence, PO-3's file, untouched.**

**Probe:** `src/models/Model_6/sweep/po7_stepper_divergence_probe.py` (committed BEFORE each run).

## 2. THE NONDETERMINISM ESCALATION — inherit this whole, including the correction

**MEASURED — four directly comparable runs, one script, one config (2 synapses, 45 s, seed 7):**

| run | eta_max | cross_bonds | dimers |
|---|---|---|---|
| proc 1 | 0.09396788 | 1848 | 796 |
| proc 2 | 0.10690230 | 1179 | 822 |
| in-process A | 0.04873884 | 2578 | 873 |
| in-process B | 0.10690230 | 1536 | 653 |

**`cross_bonds` spread 2.19×** (1179 → 2578). `eta_max` spread 2.19×, and counting the Arm A null the
range **includes 0.0** — **whether the backbone condenses at all was not reproducible at a fixed
seed.** *(Ruling 023 quotes my earlier **1.57×**; that was two runs and I superseded it at `306840b`.
**2.19× is the figure.** The standing rule in 023 §3 is strengthened by the correction, not weakened.)*

**SHOWN — three unseeded generators, `np.random.default_rng()` with no argument, seeded from OS
entropy and therefore untouched by any caller's `np.random.seed()`:**
- `camkii_module.py:199` — drawn at `:300,301,318,388,389,438,439`
- `spine_plasticity_module.py:274` — drawn **once**, at `:441-442`, thermal noise on `spine_volume` only
- `multi_synapse_network.py:1188` — in `sample_correlated_eligibilities`, called at `:1236`

**WHY THE FIX IS A PHYSICS DECISION, NOT HYGIENE.** `p.stochastic` defaults to **`True`**
(`camkii_module.py:70,129,153`; `spine_plasticity_module.py:175`) — the stochasticity is a **declared
modelling choice**, and **DDSC is stochastic by design** (Jain 2024, `quantum-system-canonical:131`).
So seeding asks *which stochasticity is physics and which is accidental.* It is three lines and it is
not a three-line decision: it changes stochastic behaviour across five POs' live surfaces at once and
would invalidate every in-flight measurement including PO-5's keystone run. **Sarah's call. Do not
seed on a successor's own authority.**

### THE CORRECTION — DO NOT LOSE IT, AND DO NOT OVER-GENERALISE THE ALARM

**My original boundary "driven vs resting" is FALSIFIED, by the MO's measurement.** Gen-2 re-ran
PO-4's `gap_template_symmetry_probe` (2 synapses, 30 drive steps) **twice at the same HEAD `3928f2d`,
both outcomes pre-registered before the result**: **bit-identical** — 1915 particles, te 34.59/32.81
both times. **A driven multi-synapse run reproduced to the last particle. Driven is NOT sufficient
for nondeterminism.**

**Ruling 023 states the replacement as "which MODULES a run reaches." That needs one refinement,
established by read (no run), and a successor should have it because the unrefined version misroutes
the next unit:**
- `model6_core.py:679` steps CaMKII **every step** — its own comment: *"runs every step regardless of
  gate"* — and the draws are gated on `p.stochastic`, which defaults `True`.
- **So PO-4's probe DOES reach two of the three generators and DOES draw from them every step, and is
  still bit-identical.** Reach alone is therefore not the discriminator.

**The boundary is per-OUTPUT dependence.** Three cases; only the third is nondeterministic:
**(a)** never drawn · **(b) drawn, but the measured quantity has no dependence path from the draw** ·
**(c)** drawn and dependent. **Arm B and PO-4's probe are both (b).** Arm B is the confirmation: I
predicted from `spine_plasticity_module.py:441-442` that `E_invasion` would be immune *while its
generator kept drawing*, then measured it immune. **A module-reach trace would flag PO-4's probe
at-risk and be wrong.**

**Net for a successor: the alarm is REAL and BOUNDED.** Real in case (c) — driven runs reading `eta`,
`cross_bonds`, the partition. Bounded because (b) exists and is common. **Ruling 023 §3's standing
rule holds: a driven multi-synapse delta needs N ≥ 3 repeats and a stated spread, or it is UNRESOLVED
— not a number.** The resting/`E_invasion` path is explicitly NOT affected, by measurement.

**Never run, still the decision-relevant missing number:** the N-run distributional pass. Four runs is
a **range, not a distribution** — no variance, bounds nothing.

## 3. WHAT EXISTS NOWHERE ON DISK — the irreplaceable part

**A. The `_remove_dimer` item in the closing request is MISATTRIBUTED — and this correction exists
only here.** The request describes it as *"routed to you by PO-5 and still unfixed."* **Verified
before writing: it was never routed to me.** `requests/po7-construct-validity/` contains exactly two
files — `mo-ruling-018.md` and `mo-ruling-023.md` — and neither mentions it. Every on-disk trace sits
in **PO-5's** lane (`leads/po5-selectivity.md`, `requests/model6-mo/po5-selectivity-003..007.md`,
`requests/po5-selectivity/mo-ruling-019.md`, `-028.md`, `board.md`, the Sarah handoff). Further:
`mo-ruling-028.md:19` and the handoff at `:64-65` record the `_remove_dimer` **tripwire as a control
that PASSED with zero calls** in PO-5's Q-B run — not as an unfixed defect. **I am not its owner, I
have not verified PO-5's run, and I have nothing to add to it. A successor must go to PO-5's
artifacts, not to mine.** Recording this because a wrap instruction is a claim like any other, and an
inherited to-do that its supposed owner never received is how phantom work outlives a program.

**B. My shell silently reset into the WRONG WORKTREE mid-session, and a commit no-op'd.** Late in the
session the cwd flipped to `intelligent-kowalevski-0d741d` (branch `claude/inspiring-sammet-aef0e0`);
a `git commit` there reported **"nothing to commit, working tree clean"** and committed nothing, while
the edit itself had landed correctly because the Edit tool used an absolute path. **Caught only by the
`git show --stat HEAD` verify step.** This is F4's root cause — *"a check run in the wrong tree"* —
arriving unprompted, in the session that was documenting it. **Successor: absolute paths for edits,
`git branch --show-current` before committing, and treat a "working tree clean" on a commit you
expected to land as a defect, not a no-op.**

**C. The probe's results artifact is NOT tracked and is already partly lost.**
`po7_stepper_divergence_results.json` is **gitignored** (I did not force-add it), and **Arm A's second
run OVERWROTE Arm B's results in that file.** Every number above survives **only as prose in this
lead file**; the raw logs were in a session-local scratchpad that does not persist. **The lead file is
the record — re-running the probe will not reproduce the Arm B json.**

**D. An unexplained observation that may be the fastest route into §2's open question.** Across two
*independent* runs, `eta_max` agreed to ~9 significant figures — **0.10690229616257775** (proc 2) vs
**0.10690229564527311** (in-process B) — **while `cross_bonds` in those same two runs differed 1179 vs
1536.** A near-invariant `eta` sitting on top of a bond graph that varies by 30% is odd. It hints the
nondeterminism enters the **topology** while `eta` is pinned by an aggregate that is nearly invariant
to it. **I never chased it and it appears nowhere else.** If the boundary unit is ever opened, this is
where I would start.

**E. The tool for this defect class already exists.** `po7_stepper_divergence_probe.py`'s `_load()`
imports a module from an arbitrary worktree by absolute path, so cross-tree A/B comparison is a
solved problem here — that is how Arm B settled F-5. Reuse it; do not rebuild it.

**Status: WRAPPED.** Unit 1 complete and accepted. Q7-1 escalated to Sarah and unresolved by design.
Q7-2/Q7-3 routings placed with the MO. Q7-4/Q7-5 filed. **Nothing seeded, nothing de-duplicated, no
other owner's file touched.**
