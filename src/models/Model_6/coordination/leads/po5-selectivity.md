# Lead: po5-selectivity (PO-5 · §8 keystone — pair-level selectivity) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** does which dimers bond depend on INPUT at pair
resolution? Pre-registered, null that cannot show the effect, positive control demonstrated to
fire, verdict able to return FALSIFIED.

**Status:** LIVE. Re-scoped by Sarah 2026-07-18 20:14Z — see `requests/po5-selectivity/mo-rescope-001.md`.
**Current unit:** UNIT 1 COMPLETE (`g`-inertness). Opening UNIT 2 — pathway attribution of the
realised bond set (Pathway 1 birth vs Pathway 2 EM), which is the prerequisite for the input test.
**Last heartbeat:** 2026-07-18 20:58Z

**UNIT 1 RESULT — `g` is LIVE, not inert.** Pre-registered `docs/PREREG_PO5_UNIT1_G_INERTNESS.md`
(`cc80fcc`, before the probe existed); probe `src/models/Model_6/sweep/po5_unit1_g_inertness.py`
(`1dbef17`); classifier demonstrated ABORTing before it was allowed to score. **`f_sat = 0.176`**
(registered saturation bar was ≥0.90), `r_p50 = 9.75 nm`, **`D = g_p90/g_p10 = 33.5`**, stable
across four samples. Log row **PO5-1**.

**Both priors were refuted, including this PO's own.** The board predicted inert-by-saturation
(`board.md:919-922`); PO-5's brief predicted inert-by-vanishing off the 400 nm birth domain. Dimers
cluster at templates (`dimer_particles.py:189-196`), so the brief was wrong by ~15× in `r`.
`model6-entanglement-partition-werner:60`'s *"~7 nm"* is the prose that was right — **no correction
owed to that skill**, and the tension the brief flagged resolves in its favour.

**The finding that matters, and it relocates the keystone's failure mode:** the graph `g` builds is
**0.75–0.83 saturated, one connected component, `largest_frac = 1.000`**. A rate varying 33× across
pairs yields a near-complete graph with a trivial partition. Pair-resolution in the RATE that does
not survive into the TOPOLOGY buys §8 nothing. **UNVERIFIED and not claimed:** which pathway causes
the saturation — Unit 2.

**Open, non-blocking:** `queue/po5-selectivity.md` Q2 (three MO-owned artifacts carry the refuted
inertness framing) and Q3 (does the trivial partition sit inside PO-5's acceptance? — proceeding on
"yes").
**Blocked on:** nothing. **The old η/partition gate is RETIRED — PO-5 needs no backbone.**

**Scope change:** `MO_MODEL6.md` §3 scoped PO-5 to selectivity through the partition, gated on η
reaching threshold. §8 never asked for that (it mentions η nowhere) and its owning section says
the keystone is **single-synapse-scale, needs no backbone.** The `P_product` fallback is also
retired — it is the gate-level case §8 rules insufficient.

**First unit:** the `g`-inertness check. `coupling_length = 5.0 nm` and `g` saturates at 1.0 below
that, so measure intra-synapse `r_ij` — if most pairs are under 5 nm the 1/r³ term is inert in
practice and Pathway 2 is flat-rate by a different route.

**Live, MO-verified:** the 1/r³ IS implemented (`dimer_particles.py:451-455`), so
`quantum-computation-and-attribution` §7 #1's "no J_ij" claim is STALE — MO owes that skill a fix.

**Carried:** `mo-f3-001.md` (read the MO CORRECTION, not the superseded top) · F-4 — L·ETA-4's
NMDAR half is vacuous, do not build on it.
