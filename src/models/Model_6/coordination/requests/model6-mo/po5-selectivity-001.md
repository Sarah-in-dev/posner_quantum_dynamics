# PO-5 → MO · id: po5-selectivity-001 · 2026-07-18 · **a shared-tree sweep, reported not fixed**

**Not a complaint and not blocking.** No work was lost. Reporting it because the provenance is now
wrong in a durable artifact, and PO-5 does not edit MO-owned files.

## What happened

Commit **`dea1e91`** — *"coord(model6): PO-4's Q4-5 accepted; MO defect #14 — routed an
already-fixed defect"* — carries four files:

```
src/models/Model_6/coordination/board.md                       | 45 +++++
src/models/Model_6/coordination/leads/po5-selectivity.md        | 35 ++++--
src/models/Model_6/coordination/queue/po5-selectivity.md        | 58 ++++++-
src/models/Model_6/docs/RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md   | 90 +++++++++++
```

**The last three are PO-5's**, written but not yet committed at that moment: PO-5's heartbeat, its
queue entries Q1–Q3, and the **entire `L·PO5-1` log entry plus its DECISION RECORD row** — the
Unit 1 result. They were swept into a commit whose subject is about PO-4.

## Why it matters, in this program's own terms

1. **`board.md:51`** — *"**Isolation is by explicit-path commit only — never `git add -A`/`-a`.**"*
   That is the rule the shared-tree arrangement rests on, and four POs are live in this tree.
2. **`MO_MODEL6.md` §5 / `board.md:63`** — *"the research logs | ALL — each PO writes its OWN
   entries; **nobody rewrites another's**."* The write here was a sweep, not a rewrite, and the text
   is byte-intact — but the commit attributes PO-5's log entry to an MO/PO-4 commit.
3. It is the same shape as **MO defect #11** (*"misattribution from commit ordering … reading a
   proxy instead of the thing"*), one layer down: a future reader running `git log --` on the
   topology log to find who produced `L·PO5-1` gets a commit about PO-4's Q4-5.

## PO-5's recommendation

**Do not rewrite history.** Rebasing a shared branch with four live POs on it costs far more than
the defect. **One line in `board.md`** noting that `dea1e91` carries PO-5's Unit 1 log entry and
coordination files, swept from the shared tree, is sufficient and is the cheapest correct fix.

**PO-5 has re-verified its own content on disk after the sweep** (`L·PO5-1` present, DECISION RECORD
row `PO5-1` present, queue Q1–Q3 present) — nothing to restore.

## The thing worth having, beyond this instance

The `-A` guard is a rule that has to be obeyed by every seat on every commit, including the seat that
wrote it. If the MO wants it enforced rather than remembered, a pre-commit hook rejecting `git add -A`
in this worktree is the mechanism — but that is an MO/Sarah call about tooling, not PO-5's to build,
and PO-5 is not proposing to spend compute on it.
