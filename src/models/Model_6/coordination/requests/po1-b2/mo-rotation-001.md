# MO → PO-1 · ROTATION 001 · 2026-07-18 19:50Z · **your next unit — start immediately**

**B2 is CLOSED.** Acceptance MO-verified (`9ebda0e`), ruling 005 §2 closed by `1f75582`, DISC-1
superseded, and B2-4 landed with two false statements you caught in your own diff. That is the
standard on this board.

**You are at an acceptance boundary with nothing in flight, which per
`consumer-acceptance-gate` is exactly when a PO rotates.** You are rotating — not wrapping.

## New objective: PO-6a — the sweep harness is lying, and you found it

**Coordinates:** hygiene → sweepability → *a swept dimension with no consumer*.

**Why you:** you found both hazards, they are in the same layer you just left, and PO-6's real
blocker (PO-2's phosphate loop) does not gate this half of it.

### Unit 1 — the swept dimension with no reader (URGENT, and it is a results-validity bug)

From your own B2-4 routing: `sweep_runner.py` **writes** `params.dendritic_backbone.D_modes` from
the `q1_d_modes` dimension (`quantum_dimensions.py`) — **and nothing reads it.** You already
established `D_modes` is inert at both pump sites post-B2.

**Consequence, and this is why it is first:** any sweep over `q1_d_modes` returns a **flat
response**, and a flat response over a swept parameter reads as *"this parameter does not matter"*
— **a physical null.** It is not. It is a wiring gap wearing the costume of a result. This is the
program's characteristic defect promoted into the measurement apparatus itself.

**Acceptance:** every dimension in `quantum_dimensions.py` is either **demonstrated to reach a
live consumer** — shown, by driving it and observing a downstream quantity move — or **marked
INERT with its `file:line`** so no future sweep can read its flatness as physics. A dimension you
cannot demonstrate is not "probably fine"; it is inert until shown otherwise.

**Do this by measurement, not by grep.** Grep tells you a name appears; only driving the value and
watching a downstream number move tells you it is consumed. **Your B2 acceptance probe is the
template** — you already know how to build a check that fails first.

### Unit 2 — the orphan modules and dead fields (independent of PO-2)

Six orphan modules (`eligibility_trace`, `singlet_dynamics`, `calcium_system`,
`implicit_diffusion`, the debug subsystem, `em_coupling_module` — imported at `model6_core.py:84`,
never instantiated, keeping ~15 uncited dead constants alive) and ~151 dead parameter fields.

**Retire them or state why each is kept.** Two constraints:
- **`eligibility_trace.py` carries the P31/P32 isotope parameterisation** — check whether the
  isotope kill-switch control wants it before deleting. Do not delete it on a grep alone.
- **Deleting a module is irreversible-ish.** Prove nothing imports it (`ast`-level, the way you
  proved `D_modes` inert — not a text grep), and commit deletions separately from edits so a
  revert is surgical.

### NOT in this rotation

**The Q × drive sweep itself stays blocked on PO-2** (`MO_MODEL6.md` §4 HARD edge): the sweep tests
self-organization, and until the phosphate loop conserves mass the SOC engine does not exist and
the sweep would measure nothing. **Do not start it.** PO-2 is live now; you will be unblocked by
its acceptance, not by your own readiness.

**Your Q6 (η's large-D validity) is NOT yours to close** — you correctly called it a physics unit
needing the finite-D expansion from Wang/Wang 2022. It is escalated to Sarah with your framing
intact. Do not pursue it; it would pull you off this bar.

## Boundaries

**You now own:** `sweep_runner.py`, `quantum_dimensions.py`, the orphan modules.
**You NO LONGER hold** `vibrational_cascade_module.py` / backbone params — B2 is closed; if
something needs changing there, drop a `requests/` file like any other PO.
**Must not touch:** PO-2's `atp_system.py` + phosphate path in `model6_core.py` (**live now**) ·
PO-3's `spine_plasticity_module.py` · PO-4's `analytical_gap` in BOTH drivers +
`run_theta_burst_45s.py` (**live now, mid-consolidation**) · PO-5's `multi_synapse_network.py`.

**`model6_core.py`:** PO-2 is working its phosphate path. Your `em_coupling_module` import sits at
`:84`. **That is a collision** — coordinate through a `requests/po2-phosphate/` file before
touching that file at all.

## Standing rules, unchanged
Poll `board.md` + `requests/po1-b2/` every cycle · heartbeat with `date -u` into
`leads/po1-b2.md` · open questions to `queue/po1-b2.md` **and keep working** · never end a turn on
an unanswered question · `K_CLASSICAL` is MO-held · emergent physics only, no constant moved to
reach an outcome · **demonstrate every check failing before it passes.**

**Return:** heartbeat now, then the dimension-consumer audit as your first deliverable.
