# Lead: po5-selectivity (PO-5 · §8 keystone — pair-level selectivity) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** does which dimers bond depend on INPUT at pair
resolution? Pre-registered, null that cannot show the effect, positive control demonstrated to
fire, verdict able to return FALSIFIED.

**Status:** LIVE. Re-scoped by Sarah 2026-07-18 20:14Z — see `requests/po5-selectivity/mo-rescope-001.md`.
**Current unit:** UNIT 1 COMPLETE and self-corrected. Opening UNIT 2 — pathway attribution of the
realised bond set (Pathway 1 birth vs Pathway 2 EM), **now core rather than prerequisite** per MO
ruling-001 §3 ("both are in scope; the kickoff named only Pathway 2 and that was too narrow").
**Last heartbeat:** 2026-07-18 21:37Z

**RULINGS ABSORBED:** `mo-ruling-001` (Pathway 1 in scope; the two 5.0 `coupling_length`s — checked,
my probe reads the nm one off the live object; `P_product` framing corrected) · `mo-ruling-010`
(Q3 = YES, the trivial-partition finding stays in my acceptance; Q2 closed) · `po1-6a-002`
(PO-1 edited `dimer_particles.py:288-289`, behaviour-identical, my regions untouched — accepted).

**SELF-CORRECTION, made this cycle:** `L·PO5-1` CORRECTION 1. My claim that the single connected
component meant the pair-resolution *"does not reach the topology"* was an INFERENCE, not a
measurement, and it read the intra layer against a network-layer standard.
`quantum-system-canonical:139` [LOCKED] makes single-synapse one-giant-component **correct physics.**
**All measured numbers survive; only the inference is withdrawn.** Caught by the MO, not by me — and
notably I had quoted §5's neighbouring lines in my own brief and reasoned past them.

**RAISED, not resolved:** `requests/model6-mo/po5-selectivity-002.md` §2 — §8 wants pair-level
selectivity, §7 #1 says single-synapse-scale, but §5 LOCKS one-component-per-synapse as correct and
puts the meaningful partition cross-synapse. **Those cannot all be operative as written.** Three
readings offered; recommending (a) the unbonded margin + (c) Pathway 1 birth structure, which keeps
Sarah's re-scope intact. **Not blocking — Unit 2 proceeds under (a)+(c).**

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
