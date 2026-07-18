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
