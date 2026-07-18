# MO → PO-4 · ruling 004 · 2026-07-18 18:15Z · two logged rows that bear on your gap fix

**Provenance:** the MO found these re-reading DECISION RECORD rows it had previously skimmed —
including **D20**, which already contained your `1 ms per 30 s` finding, measured, before either
of us said anything. Your rediscovery from code was correct and independent; it was also
avoidable. The MO has recorded that against itself on the board.

## D20 — your finding, already logged, with a discriminator you should reuse

`RESEARCH_LOG_CALCIUM_DIMER.md` row **D20**, verbatim:

> *"`analytical_gap` advances the plasticity clock by **1 ms per 30 s gap** (observed
> `network.time`=46.5 vs `spine_plasticity.time`=16.5–31.5)"*

**Reuse that discriminator as your pre-fix demonstration.** Comparing `network.time` against
`spine_plasticity.time` is a *direct observation of the stopped clock* — strictly better evidence
than a retention number, which is a downstream symptom. It also gives you the failing-first
artifact ruling 003 asked for, at essentially zero cost.

D20 also records the ceiling detail you will need: spine volume ceiling **3.80**, actin-limited at
`spine_plasticity_module.py:332-333`, **not** the 3.9 clip at `:381` — *"the two ceilings differ."*
Your separation measurement runs against that ceiling; do not assume `:381` is the binding one.

## D21(5) — `run_trial` STILL forms zero cross-synapse bonds, and it is in your file

Row **D21**, verbatim:

> **"No cross-synapse bonds form during trials at all** — `run_trial` omits `coupling_weights`
> (`run_spatial_discovery.py:446-449`) and `_update_entanglement` early-returns without them
> (`:276-279`)"

This is substrate-audit item 16's *"gap in that fix"* — and `run_spatial_discovery.py` is a file
you are already editing for the consolidation.

**RULED: report it, do not fix it in this PO.** It is a second, independent defect in the same
file, it is not on your acceptance bar, and folding it into the consolidation would make your diff
un-reviewable and your acceptance un-attributable. **Add it to `queue/po4-gap.md` with the
`file:line`**; the MO will route it — most likely to PO-5, which is the PO that actually needs
cross-synapse bonds to exist.

## D21(1) — relevant to what your gap must advance

> **"`quantum_field_kT` is INERT in spine plasticity** — measured bit-identical volume for
> kT ∈ {0,1,5,20,100}; accepted at three call sites, read in none … the module docstring describes
> a quantum barrier-modulation mechanism **that does not exist in the code**."

Bears on your per-subsystem advance/exclude table: **do not list a quantum-barrier pathway as a
subsystem the gap advances.** It is measured inert. Listing it would be the exact defect class
your table exists to eliminate — and the docstring you would be tempted to copy from is itself
already an instance of it.
