# Lead: po5-selectivity (PO-5 · §8 keystone — pair-level selectivity) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** does which dimers bond depend on INPUT at pair
resolution? Pre-registered, null that cannot show the effect, positive control demonstrated to
fire, verdict able to return FALSIFIED.

**Status:** LIVE. Re-scoped by Sarah 2026-07-18 20:14Z — see `requests/po5-selectivity/mo-rescope-001.md`.
**Current unit:** GROUNDING BRIEF returned; opening UNIT 1 (the `g` measurement).
**Last heartbeat:** 2026-07-18 20:23Z

**Brief-time finding, ahead of the measurement (`[code SHOWN]`):** `Dimer.position` is in **nm**
(`dimer_particles.py:30`, `:199-203`), drawn over a grid of `grid_shape=(100,100)` at
`dx_nm = 4.0` (`:108-112`) — i.e. a **400 × 400 × 20 nm** domain. `coupling_length = 5.0` nm
(`:129`). So the a-priori risk is **not** `g` saturating at 1.0 as the kickoff anticipated; it is
`g` being **vanishingly small** (`(5/150)³ ≈ 3.7e-5`) across most of the pair set. Unit 1 measures
the actual `r_ij` distribution and reports whichever way it lands. It also puts
`model6-entanglement-partition-werner:60` (*"intra edges at ~7 nm"*) in tension with this geometry.
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
