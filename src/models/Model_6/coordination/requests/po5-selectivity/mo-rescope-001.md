# MO → PO-5 · RE-SCOPE 001 · 2026-07-18 20:14Z · **Sarah's decision. This supersedes PO-5's scope in `MO_MODEL6.md` §3.**

**Decided by Sarah, 2026-07-18:** PO-5 is re-scoped to **§8's keystone as actually written**. The
partition/η framing is retired as PO-5's objective.

## Why — the board was chasing something §8 never asked for

**§8 is `quantum-system-canonical` §8, Keystone #1**, verbatim:

> *"Topology is the computation" needs **pair-level** selectivity (which dimers bond depends on
> input), not just gate-level (which regions/timings are eligible). If formation is gate-selective
> but pair-flat, the partition carries no more than active-region density and "graph as
> computation" weakens to "scalar as computation." **Verify before resuming graph-as-computation
> claims.**

**§8 mentions η nowhere.** L·ETA-4's *"§8 assumes drive patterns the partition THROUGH eta"* and
L·ETA-1's *"vary only the DRIVE"* are not in §8 — the latter phrase returns zero hits anywhere on
disk. The owning section (`quantum-computation-and-attribution` §7 #1) adds the part that matters
most: **"Single-synapse-scale — needs no backbone."**

So the η/pump/plateau line (L·ETA-1…5, B2) was never what this keystone required, and **PO-5 was
never actually blocked** by η failing to reach threshold.

**Also retired:** the `P_product` fallback as PO-5's hypothesis. `P_product` is the dimer population
*"which forms only where NMDAR calcium arrived"* — **which regions are eligible**, i.e. §8's
**gate-level**. That is the case §8 says *"collapses to 'scalar as computation.'"* Chasing it would
answer the wrong question.

## The MO checked the one thing that decided whether this is a real experiment

`quantum-computation-and-attribution` §7 #1 states *"Pathway 2 is currently all-pairs, flat-rate, no
J_ij — the 1/r³ coupling the docstring claims is not in the code."* **That is STALE.**
`dimer_particles.py:451-455`:

```python
diff = pos[iu] - pos[ju]
r_ij = np.sqrt(np.einsum("ij,ij->i", diff, diff))
g = (self.coupling_length / np.maximum(r_ij, self.coupling_length)) ** 3
em_rate = k_entangle_em_base * (collective_field_kT / reference_kT) * coh * g
```

**The 1/r³ falloff is implemented and pair-resolved.** The keystone is therefore a live experiment,
not a foregone conclusion.

## Your objective

**Does which dimers bond depend on INPUT — at pair resolution?**

Decompose the `em_rate` factors, because they are not equal:
- **`g` (distance)** — pair-resolved, but **geometry, not input**. Does not satisfy §8 on its own.
- **`collective_field_kT`** — **global, identical for every pair.** Pair-flat by construction.
- **`coh`** — per-dimer coherence. **The only channel that can carry input-specific information at
  pair resolution.** This is where the keystone lives.

**FIRST UNIT, before any selectivity test — the `g` inertness check.** `coupling_length = 5.0 nm`
(`dimer_particles.py:129`) and `g` saturates at 1.0 for every pair closer than that. **Measure the
distribution of intra-synapse dimer separations `r_ij`.** If most pairs sit below 5 nm, `g ≈ 1`
throughout and the 1/r³ term is **present in code but inert in practice** — flat-rate again by a
different route. That is a measurement, and it changes what the rest of the work means.

## Acceptance

A **pre-registered** measurement of whether the realised bond set depends on input at pair
resolution, with:
- a **null that cannot show the effect** — and note the standing scar: three probes on this board
  used a control that was assumed silent and was not. **`advance_silent()`
  (`presynaptic_release.py:141`) is the one correct suppressor in this codebase; it is used once.**
  Do not build an activation-floor null.
- a **positive control demonstrated to fire**, and a verdict that can return FALSIFIED.
- the `g`-inertness result stated whichever way it lands.

**Either answer is a result.** If bonding is pair-flat with respect to input, §8's keystone fails
and *"graph as computation"* weakens to *"scalar as computation"* — that is a real finding about the
program's central claim, and it is reported as one, not worked around.

## Boundaries
Single-synapse scale; **you need no backbone and must not wait on one.** `dimer_particles.py` bond
formation is yours for this unit. Not yours: `spine_plasticity_module.py` (PO-3) · `atp_system.py`
and the phosphate path (PO-2) · `analytical_gap`/`run_theta_burst_45s.py` (PO-4) · `sweep_runner.py`,
`quantum_dimensions.py`, orphan modules (PO-1/PO-6a) · **`K_CLASSICAL` (MO-held)** · another PO's
log rows.

## Carried forward, still binding
`requests/po5-selectivity/mo-f3-001.md` — read its **MO CORRECTION** section, not the superseded
top. And **F-4 (PO-3):** L·ETA-4's silent-synapse NMDAR result is vacuous; PO-3 is measuring its
magnitude now. **Do not build on L·ETA-4's NMDAR half.**

## Standing
Poll `board.md` + `requests/po5-selectivity/` every cycle · heartbeat with `date -u` · open
questions to `queue/po5-selectivity.md` **and keep working** · pre-register before scoring ·
demonstrate every check failing before it passes · emergent physics only.
