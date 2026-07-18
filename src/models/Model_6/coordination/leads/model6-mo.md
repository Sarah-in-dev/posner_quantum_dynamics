# Lead: model6-mo (Model 6 · Master Orchestrator) — OWNED BY THIS LEAD

**Objective (the done-bar):** run the Model 6 PO board to acceptance. Every PO claim is verified as
a MEASUREMENT at the data level before acceptance — never "it ran", "committed", "errors=0", or a
printed CONFIRMED.

**Worktree:** `.claude/worktrees/nervous-hertz-7ccff6`, branch `claude/nervous-hertz-7ccff6`.
**NOT master** (~20 commits behind, HEAD describes a superseded state).
Python: `/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python`.

**Status:** GEN-1 HANDING OFF 2026-07-18 ~21:10Z on context pressure. Board is live and healthy.
**Last heartbeat:** 2026-07-18 21:10 UTC.

---

## 🔴 READ FIRST — THE COMMIT RULE (root-caused today; three sweeps before it was found)

**All five agents share ONE git index** (`.git/worktrees/nervous-hertz-7ccff6/index`). `git commit`
commits **the whole index**, so a bare commit carries whatever another agent staged seconds earlier.

```
git commit -m "..." -- <explicit paths>     # correct
git add <path> ; git commit                 # SWEEPS other agents' staged work
```
**New (untracked) file:** `git add <that exact path>` then `git commit -- <that path>`, same shell
invocation. **Verify every commit: `git show --stat HEAD`** — the file list must match intent.
**Never** `git checkout -- <path>` / `restore` / `reset` to clean up; report provenance instead.

*Proven empirically, not assumed. Three sweeps happened before this was found — two the MO's
(`dea1e91` swept PO-5's entire `L·PO5-1` log entry; `df4dde9` swept PO-2's lead), one PO-1's
(`d95e826`). **No one was careless — they obeyed a rule that did not work.** No history was
rewritten; provenance entries in `board.md` are the record.*

---

## LIVE POs — session ids are needed to wake an idle one

| PO | session | state | owes |
|---|---|---|---|
| **PO-1** | `local_e4593171-4394-42ba-8067-c70dc97cc48e` | on PO-6a (rotated from B2, which is ACCEPTED) | ruling 006: fix `T_singlet_dimer` → 216 s, de-dup the two literals |
| **PO-2** | `local_19dce6f2-542d-4e05-a2b5-d6964082ab56` | running; **final leg of the ontology's coupled correction** | mass-conservation measurement citing §2.4/§3 SOC linkage |
| **PO-3** | `local_b7aeedcf-4425-4416-8389-2a57e288f7b5` | **WRAPPING** (ruling 009) | rewrite `model6-actin-invasion-driver`; route the skill edit to the MO |
| **PO-4** | `local_0b183e0f-3949-4a59-83e0-51c4c9aec7f7` | acceptance MET both bars; on `K_CLASSICAL` | 0.05→0.005 at `run_theta_burst_45s.py:145` (ONE site) + before/after dimer delta |
| **PO-5** | `local_d85b8dab-6753-4049-821d-90dddb7630c6` | running Unit 2 — **the §8 keystone** | pair-level selectivity: does which dimers bond depend on INPUT |
| PO-6 | — | folded into PO-1's PO-6a rotation | — |

**Waking an idle PO:** content goes on the backbone; the message is a **pointer only** (path + one
line of why-now + "do not reply"). **An idle PO with work available is the MO's failure.** Check
`isRunning` after every ruling.

---

## WITH SARAH — THREE, all genuinely hers

1. **PO-1 Q1 — the `P_met` drive change.** MO verified the may30 pin states *"per-synapse P_met, NO
   aggregation"* verbatim, so it is pinned not fresh. **MO endorses CONFIRM.** Revert cost: one
   `model6_core.py` hunk; mode and 2π fixes independent.
2. **The flat-η re-read scope.** Old per-synapse η moved **1% across the whole drive range**
   (`above_threshold` flipped at 22.1 because 22.1 *was* `kT_ref`). **Physics bounds it hard:** its
   only consumer is `condensation_boost → k_agg_enhanced → dimer formation`. It never reaches the
   partition. The T1′ corroboration the MO initially worried about is **backbone** η — untouched.

3. **Which pool should ATP recovery debit — metabolic-first or proportional?** *(PO-2 Q2;
   gen-1 wrongly closed this as "dissolved" and PO-2 measured it back open — see defect #16.)*
   The debit lands on the **chemically active** structural pool: proportional takes ~99.8% of a
   debit from it, metabolic-first spares it. **Measured, same seed: structural differs 0.116%
   between modes — deterministic, not statistical.** PO-2 correctly refuses to claim the ~2.3%
   downstream dimer difference (sign-test p = 0.25). **Gen-1's read, as input not a ruling:**
   metabolic-first looks more physical — recovery should take back what hydrolysis just released
   before drawing on the ~1 mM buffered structural reserve (§2.4). PO-2 is running
   metabolic-first pre-registered meanwhile; item 1 is unaffected either way.

**Standing suggestion, not a decision:** unpark *"constants centralization"* (`quantum-system-canonical`
§8) for **physical** constants only. It generated four defects today. Leave probe-local params alone.

---

## PARKED, with price tags
- **Plateau-ON arms for L·ETA-6** — >10× the ~65 s/arm plateau-OFF cost (O(n²) entanglement growth).
  Parked because the decision it informed (PO-5's `P_product` premise) was retired by Sarah's re-scope.
  **Open for the record only.**
- **L·ETA-5 re-run** with the corrected null — **Sarah's call**; PO-3 pre-registered it and did NOT run it.
- **`K_CLASSICAL`** — no longer MO-held; released to PO-4, settled by documentation.

---

## PHYSICS THAT LANDED TODAY (all MO-verified; cite these, do not re-derive)

- **`quantum-system-canonical` §4.3 FALSIFIED** — *"eta is the selectivity channel … which synapses
  condense is input-dependent."* **eta is a GATE, not a selectivity channel.** Three independent
  falsifications (ETA-4 branch-global plateau; ETA-5 accumulation on spontaneous release alone;
  commitment depleting `E_invasion` 26.2×). Mirrored in `model6-entanglement-partition-werner` §2.
  **This — not §8 — is the claim the board spent a day misattributing.**
- **§2.2 UPGRADED to DERIVED** — with `P_S` floor 0.25, Werner floor 1/√2, and live `T_singlet =
  216 s`, `P_S` crosses the Werner bound at **107.0 s**, inside the ontology's ~100–200 s band. The
  orphan's 500 s gives 247.6 s, outside it. **216 s is the value that makes the correspondence work.**
- **`P_product` is a CO-FACTOR with eta**, not an alternative — `multi_synapse_network.py:320-321`,
  and both enter dissolution protection at `:341`. The board treated them as rivals all day.
- **`K_CLASSICAL` settled by documentation** — §3 gives `0.005 [GROUNDED — Turhan 2024]`; the gap ran
  the retired **uncited** `0.05`.
- **J-coupling is intramolecular** (§2.2, inherited at birth from a shared ATP). `calculate_j_coupling`
  reading `atp` and not `phosphate` is defensible; **the docstring is the error.**
- **`SUPER-1`** (calcium log) — D21(5)/audit item 16 were **already fixed** by `15abd39`.
- **Two `coupling_length = 5.0`** — nm (intra, `dimer_particles.py:129`) vs µm (cross,
  `multi_synapse_network.py:96`). Factor of 1000, different forms, different layers.

---

## MO DEFECT LEDGER — 16. **The most useful thing in this file. Do not repeat these.**

**The through-line: the MO trusted a document where the discipline says go to the code.** POs caught
14 of 16; the MO caught 2 while verifying something else.

1. Verified **prose against prose** — read `analytical_gap`'s docstring lists, concluded "frozen"; the
   function tail steps 1 ms. Tagged it `[code SHOWN]`. *(PO-4)*
2. **Unsourced numbers in an acceptance bar** — `1.291/2.389` had no artifact. *(PO-4)*
3. **Named prior art without verifying it** — `eta_in_live_trial.py` as "prior art to reuse". *(PO-3)*
4. **Omitted the poll mandate** from the first two kickoffs — the exact `consumer-acceptance-gate:34` scar.
5. **Credited a design element that was itself the error** — praised GATE 1's `CONFINED-RATCHET`; PO-3 deleted it.
6. **Skimmed a decision record and called it read** — D20 already held the 1 ms fact.
7. **Wrote an acceptance bar short of its own authority** — omitted the pin's D/φ obligation.
8. **Propagated a PO's number unverified** (F-3 "~100×") into a durable PO-5 artifact AND to Sarah.
9. **Characterised code from a line reference** — called a live 20 s workaround a "stale comment";
   the ruling would have shipped a double-advance. *(PO-4)*
10. **Assumed idle POs self-resume** — they do not; the MO must push. *(Sarah)*
11. **Misattributed authorship from commit adjacency.** *(MO, while verifying)*
12. **Escalated physics questions the ontology answers** — had read only §8 of `quantum-system-canonical`. *(Sarah)*
13. **Collapsed NO CONSUMER vs CONSUMER HARDCODED** — overstated T2/J-coupling to Sarah. *(PO-1)*
14. **Read a dated log row as live status** — routed an already-fixed defect to PO-4. *(MO, while verifying)*
15. **Swept two POs' uncommitted files** — later root-caused as the shared-index race, not carelessness. *(PO-5)*
16. **Verified a premise correctly, then drew a conclusion that did not follow.** Ruling 002 §1 said
    the ATP-recovery debit *"cannot affect the chemistry, only the ledger."* The premise (ATP-derived
    Pi lands in a pool speciation ignores) was right; the conclusion was wrong, because the *debit*
    comes out of the chemically active pool. **PO-2 measured it: 0.116% structural difference.**
    **A NEW SHAPE — every defect above was reading instead of checking; this one read correctly and
    reasoned wrongly.** *"Go to the code" would not have caught it; only a measurement did.*
    **Lesson: verifying a premise is not verifying a conclusion. A ruling that says "therefore this
    cannot matter" is itself a claim that needs a measurement.** *(PO-2)*

---

## VERIFICATION DUTY — the MO runs it, never relays
Every acceptance was executed by the MO itself: PO-1's pump probe + T1′ 7/7, PO-4's coverage checker
+ separation probe + reach probe, PO-1's dimension audit, PO-3's scorer, PO-4's retention probe.
**A PO's self-report is never the evidence.** The demonstration that a check *can fail* is what gets
verified — not the passing run.

---

## ### GROUNDING BRIEF — MODEL6-MASTER **gen-2**, 2026-07-18 21:26Z

**`date -u` run first: `Sat Jul 18 21:15:01 UTC 2026`** (elapsed time not reasoned about without it).

### (a) Verified current state — every fact tagged, quotes located

- `[skill session-discipline:47]` — *"Did you read every skill in `<available_skills>` that could
  plausibly apply? Not the titles. The full contents."* Read in full, not skimmed.
- `[skill agent-grounding-protocol:45]` — *"If a skill says X and the code does Y, the code is right
  and the skill has drifted."* This is the rule gen-2 applied to its own kickoff, below.
- `[skill quantum-system-canonical:99]` — *"**k_classical = 0.005 s⁻¹** (dissolution; cluster
  lifetime τ ≈ 200 s). [GROUNDED — Turhan 2024]"*. Read **IN FULL** (all 216 lines), per the
  handoff's instruction that defect #12 came from having read only §8.
- `[skill quantum-system-canonical:124]` — *"eta remains a **gate**, and is not a **selectivity
  channel**"* (§4.3, FALSIFIED 2026-07-18). Located §4.3, **not** §8 — the misattribution is not
  repeated here.
- `[skill quantum-system-canonical:139]` — *"A single-synapse 'one giant component' is **correct
  physics, not a bug.**"* [LOCKED] Directly bears on PO-5 Q3; see (e).
- `[code SHOWN]` `model6_parameters.py:412` — `T_singlet_dimer: float = 216.0`. **Ruling 006 has
  LANDED and is committed** (`3632fce`, verified by `git log -S`), *not* pending as the handoff table
  states. `git diff` on that file returns empty.
- `[code SHOWN]` `sweep/run_theta_burst_45s.py:147` — `K_CLASSICAL = 0.005  # s^-1 … Turhan 2024,
  tau ~ 200 s`. **PO-4's correction is landed**, matching §3's GROUNDED value.
- `[code SHOWN]` `sweep/run_place_field_learning.py` **exists** (18455 bytes) — PO-4's Q4-8 drift
  report against `model6-architecture` F4 is **confirmed by direct `ls`**, not relayed.
- `[data — git]` **38 commits in the 35 minutes before this brief.** The board is hot.

### (b) Decisions + locked items found

- **[LOCKED]** `quantum-system-canonical:140` — *"The 0.5 bound is physics, not a fitted knob — do
  NOT lower it to rescue a result."*
- **[LOCKED]** `:101` — `productive_fraction` *"**never tuned to a target dimer count.**"*
- **[LOCKED]** `:144` — *"**The current model is (A).**"* Never described in (B)'s language.
- **T1′ is CLOSED** (4/4, p≈3×10⁻⁶) — re-validation is human-gated.
- **The commit rule** (`git commit -- <paths>`, verify `git show --stat HEAD`, never
  `checkout`/`restore`/`reset`) is in force and was proven empirically, not assumed.

### (c) THE FINDING THAT GATES THIS SEAT — **gen-1 is still LIVE**

**Gen-2's kickoff states gen-1 "is handing off." The observed state says it is still running and
still ruling.** Per `agent-grounding-protocol:45`, the evidence wins over the prose:

- `[session state]` `local_0ce7902b` *"Seat MODEL6-MASTER"* — **`isRunning: true`**, last activity
  **21:21:38Z**, i.e. *after* gen-2 was seated.
- `[data — git]` `ec2d028` (21:21:45Z) — a **new ruling**, withdrawing ruling 002 §1 and opening
  **MO defect #16**. `2fcf12e` (21:22:11Z) — *"correct the handoff **before gen-2 reads it**"*.

**Gen-1 is not competing; it is finishing, and it is correcting this very file for gen-2's benefit.**
Its last act caught a false fact it was about to hand over (PO-2's Q2 was **not** dissolved) — the
inheritance failure it had itself logged as defect #8.

**So gen-2 has NOT taken the board.** Two MOs writing `board.md` is the exact two-writer collision
the surface-ownership map exists to prevent, and **which seat holds the board is Sarah's call, not
gen-2's** — the same reason a PO never resolves its own collision. Escalated to Sarah in chat.

### (d) Consequence — gen-2 has issued NO rulings and woken NO PO

`[session state]` **PO-3, PO-4 and PO-5 are all idle** (`isRunning: false`, 21:10–21:15Z), and PO-5
is idle on the **§8 keystone**. Under the standing directive that is the MO's failure to fix — **but
waking them now would give five POs two masters**, and PO-4's and PO-5's next units both depend on
rulings gen-1 may be mid-way through writing. **Deliberately parked for minutes, not dropped**, with
the reason recorded here rather than left as an unexplained gap.

### (e) Rulings PREPARED, held pending the seat question (grounded, ready to post)

1. **PO-5 Q3 — does the trivial partition sit inside PO-5's acceptance? → YES, keep it.**
   Grounded, not a fresh physics call: §5 [LOCKED] already holds *"a single-synapse 'one giant
   component' is correct physics, not a bug"* (`:139`), and §8 Keystone #1 turns on the gate-level
   vs pair-level distinction (`:197`). PO-5's *"pair-selective in the rate and saturated in the
   graph"* is that same question one layer down, so splitting it would let each half report "not
   mine." **PO-5's note is correct and endorsed: this is not licence to touch the saturation** —
   that would be `MO_MODEL6.md` §7's tuning-to-an-outcome.
2. **PO-4 Q4-8 — `model6-architecture` F4 drift. CONFIRMED by gen-2's own `ls`**, not relayed:
   `run_place_field_learning.py` exists. **The skill write is the MO's** (`.claude/skills` is a
   symlink into another program's repo). Owed, unwritten.
3. **The handoff's own PO table is stale on PO-1** — it lists ruling 006 as owed; `3632fce` landed
   it. Recorded so the next reader does not route a closed item, which is **defect #14's exact
   shape** committed against the handoff instead of against a log row.

### (f) Self-understanding delta — one sentence

**The program's handoff document was wrong about the program's own live state in two independent
places within twelve minutes of being written** — gen-1's PO table still owes ruling 006 that
`3632fce` had already landed, and gen-1 itself is still running while the document says it is
handing off — which is the sharpest available demonstration that **a handoff is a claim about a
moving system, and ages exactly like the dated log row that produced defect #14.**

**Gen-2's standing correction to itself, taken from gen-1's defect #16:** verifying a premise is not
verifying a conclusion; a ruling that says *"therefore this cannot matter"* is itself a claim that
needs a measurement.
