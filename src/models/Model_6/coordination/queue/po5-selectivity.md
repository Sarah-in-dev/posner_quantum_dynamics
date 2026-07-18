# Queue: po5-selectivity — actions awaiting the MO / Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready.

---

## Q1 — CLOSED BY MEASUREMENT, no decision needed · 2026-07-18

**Was:** the brief flagged that the board's `g`-saturation framing and PO-5's own `g`-vanishing
framing could not both be right, and asked that neither be carried in an artifact until measured.
**Resolved:** both were wrong. `f_sat = 0.176`, `D = 33.5`, verdict `LIVE`. See DECISION RECORD row
**PO5-1**. **No MO action.** Recorded so the question is not re-opened from the brief.

---

## Q2 — OPEN · three artifacts carry a `g`-inertness framing that is now measured false

**The ask:** the MO corrects the artifacts it owns, or tells PO-5 it prefers to leave them and let
the log row carry the correction.

**Why:** `board.md:919-922` and `requests/po5-selectivity/mo-rescope-001.md:55-59` both state the
hypothesis as *"if most pairs sit below 5 nm, `g ≈ 1` throughout"*. Measured: **17.6%** of pairs sit
below 5 nm. Those are MO-owned coordination artifacts, and a PO reading them cold would inherit a
refuted premise — the inheritance failure the MO recorded as defect #8.
`quantum-computation-and-attribution` §7 #1 (`:78`) also still carries *"whether `g` is inert in
practice, since `coupling_length = 5.0 nm` and `g` saturates at 1.0 below that"* as open; it is now
answered.

**PO-5's recommendation:** a one-line superseding note on each, pointing at DECISION RECORD row
**PO5-1**, originals left in place per the log convention. **PO-5 does not make these writes** —
`board.md` and `mo-rescope-001.md` are MO-owned, and the standing rule (`board.md:495-505`) is that
**the MO makes all skill-library writes**, because `.claude/skills` is a symlink into another
program's repo carrying 325 uncommitted files.

**Not blocking.** PO-5 proceeds to Unit 2 regardless.

---

## Q3 — OPEN · does the trivial partition fall inside PO-5's acceptance?

**The ask:** is the near-complete, single-component intra graph part of PO-5's verdict, or a
separate finding to route elsewhere?

**Why it is a real question.** PO-5's acceptance is *"whether the realised bond set depends on input
at pair resolution."* Unit 1 measured, incidentally, that the realised bond set is **0.75–0.83
saturated and forms exactly ONE connected component** (`comps = 1`, `largest_frac = 1.000`, t = 5 s
and 10 s). If the intra graph is a near-complete blob, a pair-level input dependence in the *rate*
can be real and still leave the *partition* — which `model6-entanglement-partition-werner` LOCKS as
the computation — carrying no information.

**So §8's keystone could fail for a reason §8 does not name:** not *"gate-selective but pair-flat"*,
but *"pair-selective in the rate and saturated in the graph."* Same destination — *"scalar as
computation"* — different mechanism.

**PO-5's recommendation:** keep it inside PO-5's acceptance and report it as part of the keystone
verdict, because it is the same question one layer down and splitting it would let each half report
"not mine". **Proceeding on that assumption; one line redirects it.**

**Note for whoever rules:** this is **not** a proposal to touch the saturation. Changing a formation
rate to un-saturate the graph would be tuning a constant to reach an outcome (`MO_MODEL6.md` §7
LOCKED, *"Emergent physics only"*). PO-5 measures and reports.
