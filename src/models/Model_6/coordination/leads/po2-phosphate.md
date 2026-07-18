# Lead: po2-phosphate (PO-2 · the phosphate loop) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** total phosphate conserved around a full
hydrolysis → consumption → dissolution → recovery cycle to a **stated** tolerance, with the
check **shown failing on current code first**; and J-coupling demonstrably tracking dimer
consumption.

**Status:** LIVE — grounding brief returned 2026-07-18 19:46Z.
**Current unit:** pre-registration (§2.4), then the conservation probe committed FAILING first.
**Last heartbeat:** 2026-07-18 19:46Z
**Blocked on:** — (two open questions in `queue/po2-phosphate.md`, neither blocking; working on)

## HEARTBEAT LOG

### 2026-07-18 19:46Z — GROUNDED. Three corrections to the dispatch, all code-verified.

Both defects confirmed at the code level as the kickoff described them. Three things the
kickoff did not have:

1. **TWO consumption sites, not one.** `model6_core.py:450-452` (EM path) **and
   `model6_core.py:756-757`** (`=== NON-EM PATH ===`) carry byte-identical decrement blocks;
   neither updates `phosphate_total`. Same two-definitions shape the MO recorded for
   `analytical_gap`. A fix to one leaves the other live. Fix must be structural, not
   dual-patched.

2. **Dissolution ALREADY returns Pi — do not "fix" it.** `ca_triphosphate_complex.py:430-438`:
   `d_dimer_dt = dimer_formation - dimer_dissociation`, then
   `_po4_consumed = 4.0*d_dimer_dt + 6.0*d_trimer_dt`, commented *"Net formation: negative
   values return Ca to pool on dissolution"*. The quantity is **signed**. Step E's "return Pi
   from particle dissolution" is satisfied at this site. The live defect there is instead the
   `np.maximum(..., 0.0)` clamp at `:451`/`:757`, which creates phosphate at pool exhaustion —
   and the pin says "not a cap".

3. **DECISION RECORD D14 contradicts this dispatch, and both are right about different
   halves.** D14: *"SOC loop already closed in live code (no B3 edit needed) … phosphate
   feedback **mimicking model6_core**"*; D8: *"exact conservation (2e-17 M)"*. Verified against
   code, not prose: `grep -n "ATP|hydrolys|recovery" sweep/phosphate_conservation_probe.py`
   returns **ZERO hits**. The A3 probe has **no ATP arm**. D8/D14 measured the
   formation↔dissolution half — which does conserve — and that has been read as the whole
   loop. **The loop D14 declared closed was never wired to the leak.**

**`K_CLASSICAL` report (MO-held, untouched):** `ca_triphosphate_complex.py:160` = `0.005` ✅ ·
`sweep/phosphate_conservation_probe.py:70` = `0.005` + Turhan citation ✅ · `analytical_gap`
both copies = `0.05` (retired; PO-4's surface) · `dimer_particles.py:127` = `0.001`. Three-way
spread stands. Reported, not touched.

**Tree state on arrival:** NOT clean — PO-4 holds uncommitted edits to `PREREG_PO4_GAP.md` and
`src/models/Model_6/sweep/gap_retention_probe.py`. Those are PO-4's own files; `atp_system.py`
and `model6_core.py` are untouched, so my slice is free. No collision.

**Why now:** PO-1's B2 landed and is MO-verified; the tree is clean, so the `model6_core.py`
shared-file boundary the board gated PO-2 on is met.

**Two defects, kept separate:**
- `phosphate_total` stale (`atp_system.py:428` recomputes only inside `add_phosphate_from_atp`;
  `model6_core.py:450-452` decrements `phosphate_structural` without it) ⇒ J-coupling
  (`atp_system.py:485`) reads a field ignoring dimer consumption. **Contained bug.**
- **ATP↔Pi not mass-conserving** — hydrolysis credits Pi (`:130`), recovery regenerates ATP
  (`:163`, `:169-170`) without debiting any pool. **This breaks Step E: the finite pool is not
  finite around the loop, the reset feedback does not close, and the SOC engine does not exist.**

**Owns:** `atp_system.py`, the phosphate path in `model6_core.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params (PO-1 owes a D/φ
item) · PO-3's `spine_plasticity_module.py` · PO-4's `analytical_gap` in BOTH drivers and
`run_theta_burst_45s.py` (actively consolidating) · PO-5's `multi_synapse_network.py` + T1′
family · PO-6's surfaces.

**Shared-file hazard:** `model6_core.py` — PO-1/PO-4/PO-2. One uncommitted holder at a time.
Commit at the boundary; a broken shared tree blocked two POs earlier today.

**NOT this PO's to decide:** `K_CLASSICAL` — MO-held, decision-ready with Sarah. `0.005` is live
and correct in `ca_triphosphate_complex.py:160`; `analytical_gap` still runs the retired `0.05`.
Report, do not touch.

**Gates:** PO-6 (HARD). The drive×damping sweep measures nothing against a non-conserving loop.

**Prior art:** `sweep/phosphate_conservation_probe.py` (repo root) already exists and already
uses the correct `K_CLASSICAL = 0.005` with its citation. Extend it; do not rebuild.
