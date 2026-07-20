# PRE-REGISTRATION (SKELETON) — PO-8 Unit B: the readout-time input-selectivity keystone

**Status: SKELETON, NOT YET SCORABLE.** Registered 2026-07-20 to fix the design invariants I have
already grounded, BEFORE any scored run. Three slots are gated on Sarah's rulings (queue Q1/Q2/Q3)
and are marked `⟨GATED⟩`; the run is INVALID if scored before they are filled and this file is
promoted to a full pre-registration. Append-only once promoted.

Prior art / scars this design inherits:
- `PREREG_PO7_UNIT2_MULTISYNAPSE_KEYSTONE.md` (SUPERSEDED) — its **dimer-level-scoring trap** and its
  **contiguous-vs-interleaved** discriminator are carried forward here (see Scoring + Design).
- `L·PO5-13` (spatial-half false positive), `L·PO5-11` (topology = density alone), `L·PO7-2`
  (cross-synapse provenance withdrawn), `L·PO7-5` (the reframe: the unit is the CORRELATED DOMAIN,
  no F-threshold).

---

## The question (on the right object, at the right time)

Does **input identity determine which synapses share a correlated domain AT READOUT** — beyond what
active-region **density** alone explains, and beyond **geometry** (spatial locality)?

- Object: the **correlated-domain partition** over the correlation metric d(u,v)=Σ(−ln p_e),
  p_e=(4F−1)/3, effective correlation e^{−d} (`L·PO7-5`; NO F-threshold). Reuse
  `po7_unit18_correlation_domains.build_weighted_graph` / `bounded_dijkstra` verbatim.
- Time: **readout**, a dopamine event at a realistic delay. Dopamine is the CLOCK, not a decoherence
  mechanism (it is state-inert in this model — grounding-brief correction 2); the readout object is
  the partition of the graph as it stands at the delay. ⟨GATED: Sarah confirms this framing, queue
  Q2 tail⟩

---

## SCORING — at the SYNAPSE level, never the dimer level (inherited hard lesson)

The superseded Unit-2 prereg caught, in its own words, that a dimer-level statistic against
activation identity *"would sit near 1.0 by construction, for any input whatsoever"* — because
`intra_synapse_bonds_cache` edges all lie inside one synapse, hence inside one activation label. That
is a statistic that can only pass. **The keystone graph is the 7-node SYNAPSE graph:** nodes are
synapses; inter-synapse weight = the effective correlation between the two synapses' dimer sets
(aggregate of e^{−d} across cross-bridges above the Werner bound). Intra cliques are excluded from
the score — they carry no cross-synapse information and only saturate it.

Scored statistic (fixed now): **Q_act = Newman modularity of the synapse-level correlation graph
against the input-group partition.** Two conditions differ in input IDENTITY at MATCHED density.

### The falsifier / guard (state BEFORE scoring — the decomposition null)
Input does **NOT** structure the readout if: the synapse-level correlated-domain partition is
**independent of input identity at matched density** — operationally, Q_act for the true
input-group labelling is within the null band of Q_act computed against **random equal-size
synapse relabellings** (the partition splits per-synapse / by geometry, not by which input drove
which synapse). I will **demonstrate the verdict function FAILING first**: run it on a
density-matched control whose two conditions are the SAME input (identity removed) and show it
returns NULL, before trusting a positive on the real contrast. A clean decomposition null here is a
real, reportable negative (`L·PO7-5` says either answer is a result).

---

## DESIGN (density matched BY CONSTRUCTION — the on/off confound is banned)

- 7 synapses, `pattern="linear"`. Every condition activates the **same number** of synapses at the
  **same drive strength** — conditions differ only in **which/when**, never in how many. This avoids
  the on/off density confound that sank PO-7 Unit 2 (`notes.md` Q6).
- **Primary contrast — synchronous vs staggered neighbour-group activation** (the kickoff's cleanest
  design): same synapses, same total drive, activation either simultaneous or time-staggered, so the
  provenance/coincidence structure differs while density is identical. ⟨GATED: requires per-synapse
  stimuli on `net.step`, queue Q2⟩
- **Geometry control (carried from Unit-2 ARM 2):** also run an **interleaved vs contiguous** group
  assignment so a positive cannot be re-explained as spatial locality. Under `pattern="linear"`,
  {0,1,2} is the spatial half — identity must be scored ORTHOGONAL to space.
- **Readout delay is an independent variable**, swept across the pre-ignition→write→collapse range
  informed by Unit B0 (below). The realistic-delay value is chosen from B0 BEFORE scoring, by a rule
  fixed there (NOT the delay that maximises Q_act).
- **≥5 free-running draws per cell. NO SEEDING.** Drive via `net.step` only.

---

## PRECONDITIONS (must hold or the run is INVALID and unscored)
1. Network-path regression digest `515772101786800` (`PO7_U11_MODE=offpath`) reproduces with all
   Unit-A physics edits in place and the flag OFF — re-run immediately before scoring.
2. Ignition confirmed on each scored draw (peak η > 0) — a dead condensate has no cross channel and
   the keystone is void (the η≡0 trap, `L·PO7-4` §7).
3. `coupling_weights` passed to `tracker.step` (else the η-graph is silently empty).
4. ⟨GATED: λ decided (queue Q3) — it sets whether the readout is one- or two-timescale, which changes
   what "the partition at delay t" contains.⟩

---

## Unit B0 (characterization, runs NOW — feeds this prereg, does not need the rulings)
`sweep/po8_unit_b0_readout_time_sweep.py` — free-running readout-time domain sweep, no input contrast,
no scored verdict. Establishes: (a) the domain-size time course (flat-then-cliff vs gradual), (b) the
collapse time (~107 s predicted), (c) the realistic-delay choice for Unit B, (d) the λ consequence
(run at λ=5 and λ=214). Corrects an earlier PO-8 overstatement: domain size is CROSS-bridge dominated
(pre-ignition intra-only domains are ~10 dimers), hence λ-DEPENDENT — so B0 is run at both λ values.
