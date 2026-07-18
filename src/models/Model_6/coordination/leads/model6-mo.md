# Lead: model6-mo (Model 6 · Master Orchestrator) — OWNED BY THIS LEAD

**Objective (the done-bar):** run the Model 6 PO board to acceptance. Every PO's claim is
verified as a MEASUREMENT at the data level before it is accepted — never "it ran",
"committed", "errors=0", or a printed CONFIRMED.

**Worktree:** `.claude/worktrees/nervous-hertz-7ccff6` (branch `claude/nervous-hertz-7ccff6`).
**Mode:** supervised autonomy — dispatches without per-PO approval; escalates physics calls.

**Status:** GROUNDED (2026-07-18 17:30 UTC, `date -u` run). Brief below. Next: dispatch PO-1 + PO-3.

**Current unit:** posting the grounding brief, then spawning PO-1 (B2) and PO-3 (E_invasion).
**Last heartbeat:** 2026-07-18 17:30 UTC
**Blocked on:** nothing.

**Live POs (agentId / task-id → PO):** (none yet — record every handle here; you need it to
resume a worker that pinged with NEED_GUIDANCE. A worker that pinged and got no reply is
stalled, not done.)

**Landed this run:** (none yet)

**Notes / decomposition (living — rewrite each cycle):**
- Board came from the 2026-07-18 substrate audit (`docs/SUBSTRATE_AUDIT_JUL18.md`) plus
  Sarah's re-rank of its open items. Her reorder is load-bearing and already encoded:
  items 7/3/5 are DONE and committed; items 1+2+4 collapse into PO-1 (B2) because fixing a
  bug in code scheduled for deletion is the wrong order of operations; item 6 splits, and
  its ATP↔Pi half is what breaks Step E.
- The recurring defect class in this program is **prose asserting mechanisms the code does
  not implement** (a Hill function on the wrong gate, a 30% collapse that never runs, "no
  fitted parameters" beside two, a quantum barrier measured inert, a 2% ATP replenish that
  does not exist). Expect it; check for it in every PO return.

---

## GROUNDING BRIEF — 2026-07-18 (MO boot)

### (a) Verified current state

**Time.** `date -u` → `Sat Jul 18 17:30:36 UTC 2026`. `[shell]` Not inferred.

**The tree.** `git log --oneline -5 claude/nervous-hertz-7ccff6` → `e6c7bc5 chore(model6):
coordination backbone, tracked` / `83c3b11` / `7d97fcb` / `15abd39` / `32f2840`. `master` is at
`683b82f`. An existing worktree for the branch is at `.claude/worktrees/nervous-hertz-7ccff6`.
`[code SHOWN]`

**The 2π site is real and reads as described.** `vibrational_cascade_module.py:312-315`
`[code SHOWN]`:
> `hbar = 1.0546e-34  # J·s`
> `x = hbar * p.omega_0 / (k_B * p.T_body)`

and `:85` declares `omega_0: float = 40.0e9           # Hz — lowest dominant mode` — i.e. the
symbol is named for an *angular* frequency but declared in *Hz*, so `ℏ·f` is used where `h·f`
is required. Duplicated at `:753-755`. The correct form sits 200 lines away in
`model6_parameters.py:41-44` `[code SHOWN]`:
> `"""Bose-Einstein occupation n̄ = 1/(exp(h·f/kT) − 1) for frequency f in Hz.`
> `Uses h·f (≡ ℏ·2πf = ℏω), not the ℏ·f shortcut that drops the 2π.`

**The calibration fiction is a function-body literal, as claimed.**
`vibrational_cascade_module.py:243-248` `[code SHOWN]`:
> `# Reference: 22.1 kT (MT+ full invasion) should give r well above threshold`
> `kT_ref = 22.1  # Reference field energy for MT+ condition`
> `r = p.r_at_E_ref * (collective_field_kT / kT_ref) ** p.pump_exponent`

`kT_ref` is bound inside the pump-rate function, not on the dataclass — so it is invisible to
`TubulinCascadeParameters` and to `sweep_runner`. **Structurally unsweepable, confirmed.**
`r_at_E_ref: float = 100.0e9` is at `:115`. (The audit and `RESEARCH_LOG_CALCIUM_DIMER` DISC-1
both cite `:246` for `kT_ref`; that is exact.)

**The false backbone docstring is live.** `model6_parameters.py:768-769` `[code SHOWN]`:
> `backbone toward Fröhlich condensation ... The critical pump rate r_c is computed from the`
> `backbone lattice parameters (D, phi, chi) — not hand-tuned.`

This is the claim PO-1 must correct; it is the recurring defect class in one sentence.

**PO-3's constants read as the provenance audit describes.** `spine_plasticity_module.py`
`[code SHOWN]`: `:90 k_polymerization_max: float = 0.1    # s⁻¹, rate constant for new actin`
(no source on the line), `:109 tau_extrude: float = 180.0       # s; ... Honkura band`
(grounded), `:115 E_ref: float = 1.87              # asymptotic enlargement under sustained
maximal Ca (3000 s uncommitted run).` — a model asymptote frozen as a constant. `:411-412`
shows `E_invasion` is exactly `clip((actin_enlargement − invasion_threshold) / (E_ref −
invasion_threshold), 0, 1)`, i.e. **`E_ref` is the denominator setting the whole scale of
`r ∝ E_invasion × ca_open`.** That is why the ratchet test is the cheap decisive one.

**The interim disclosure is DISCHARGED, verified on disk — not self-reported.**
`MO_MODEL6.md:237-240` lists it as a standing MO action. It is already written:
`RESEARCH_LOG_CALCIUM_DIMER.md` DECISION RECORD row `DISC-1`, dated 2026-07-18, committed as
`7d97fcb`. `[code SHOWN]` I re-read the file; it is there. No MO action outstanding.

### (b) Decisions + locked items found

`MO_MODEL6.md:216-229` §7 LOCKED `[MO_MODEL6.md]` — quoted in part:
> `**Emergent physics only.** No constant tuned to a downstream target.`
> `**Score the ORDER, never the times** (the T1′ scar).`
> `**The −40 mV synaptic cap stays.** Raising it to make η ignite destroys the`
> `plateau/synaptic separation the BTSP grounding rests on (L·ETA-3).`
> `**T1′ is CLOSED** — 4/4, p≈3×10⁻⁶. Do not re-run, re-tune, or "improve" the geometry.`

`consumer-acceptance-gate:22` `[skill consumer-acceptance-gate]`:
> `**DONE = demonstrated at the consumer.** ... "Deployed" is a status report. "Merged" is a`
> `status report. Neither is done.`

Translated by `MO_MODEL6.md:47-53` §2.3 `[MO_MODEL6.md]`:
> `**A verdict that cannot distinguish its outcomes is not a result.**`

`orchestrator-session:19-20` `[skill orchestrator-session]`:
> `It does NOT do the work and does NOT ingest workstream substance ... Pulling substance into`
> `the orchestrator is exactly what bloated and killed the May 21/22/23 orchestrator`
> `conversations.` — the reason this file stays thin and I read the logs' top tables only.

`talon-orchestrator:43` `[skill talon-orchestrator]`:
> `A kickoff is a **detailed, thoughtful prompt** written fresh for the workstream ... It is`
> `**not** a generic preamble pasted verbatim, and it is **not** a template with placeholder`
> `blocks to fill mechanically.`

`autonomy-contract:31-33` `[skill autonomy-contract]`:
> `Threads coordinate by reading and writing a shared coordination file ... NOT by messaging`
> `each other.` — matches `coordination/README.md:5-7`.

`agent-grounding-protocol:45` `[skill agent-grounding-protocol]` — the reconciliation rule and
the 2026-06-24 scar:
> `The conversation claimed; the file disproved. File wins.`

This is why DISC-1 above was re-read on disk rather than taken from the commit message.

`session-discipline:122` `[skill session-discipline]`:
> `Never use ask_user_input pop-up tool.` — consistent with the standing memory; decisions go
> to Sarah as plain chat text.

**Prior failures in this area, from the logs' DECISION RECORD tables (top-of-file only):**
`ETA-4` — the probe's first auto-verdict printed "eta stays SELECTIVE" because η==0 at silent
synapses, but η==0 at the *driven* synapse too, so the test was vacuous. `ERR-2` — a
measurement dispatched without reading `model6-input-engine`, which owned the input path and
predicted the symptom verbatim. Both are dispatch-discipline failures, not physics failures,
and both are things a kickoff can prevent. Both are written into the two kickoffs.

### (c) Prior art to reuse, not rebuild

- Correct pump physics for PO-1: `model6_parameters.py:46` `bose_einstein_occupation` — the
  audit states it is verified correct across eight call sites. PO-1 **calls** it; it does not
  re-derive a Planck factor.
- Probe scaffolding for PO-3: `sweep/eta_in_live_trial.py` (the L·ETA-3 live-trial harness,
  already wired to the input engine and already emitting `r` / `E_invasion` / `ca_open`) and
  `sweep/plateau_vgcc_leak_probe.py` (the L·ETA-4 multi-synapse rig, and the source of the
  corrected verdict logic). The ratchet test is a traversal-count extension of the former, not
  a new harness.
- Kickoff/acceptance method: `MO_MODEL6.md` §6 (kickoff convention) + §2.3/§2.4.

### (d) The MINIMUM surgical change (this MO cycle)

Post this brief; spawn exactly two POs — PO-1 (B2) and PO-3 (E_invasion ratchet) — as their
own sessions with fresh, self-contained kickoffs; register both on `board.md` with their
`leads/`/`queue/` files. Nothing else. PO-2 waits for PO-1's commit boundary
(`model6_core.py` shared-file hazard, `board.md:31-32`). `K_CLASSICAL` stays parked with me.

### (e) Open questions (Sarah's, escalated — not decided here)

1. **The skill library was not reachable from this worktree.** `.claude/skills` is a symlink at
   the repo root (`→ murmur-platform/murmur-platform/.claude/skills`) but absent from
   `.claude/worktrees/nervous-hertz-7ccff6/.claude/`. `CLAUDE.md` asserts the library is
   symlinked here; for a worktree that was false, so every named method skill was unlisted and
   had to be read by absolute path. POs are therefore spawned at the repo root, where the
   symlink resolves.
2. **I cannot set my own session title.** `set_session_title` rejects the current session by
   design. `[MASTER] Model 6` needs Sarah, or the app UI.
3. PO-3's negative branch remains Sarah's call, per `board.md:48`. The PO measures and STOPS.

### (f) Self-understanding delta

The program's own board carried one line the code contradicts and one the code has already
discharged: `MO_MODEL6.md:82` calls the `model6_parameters.py:759` docstring "false at the
backbone today" — verified true, the docstring at `:768-769` still claims `r_c` is computed
from D/φ/χ — while `MO_MODEL6.md:237` still lists the `kT_ref` interim disclosure as a standing
MO action that `RESEARCH_LOG_CALCIUM_DIMER` DISC-1 shows was already written and committed at
`7d97fcb`. The board is stale in both directions at once, which is the same defect class the
audit named — prose asserting a state the artifacts do not hold — appearing in the
orchestration layer rather than in the physics.
