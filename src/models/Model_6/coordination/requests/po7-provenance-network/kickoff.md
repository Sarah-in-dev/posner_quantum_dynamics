# PO-7 KICKOFF — network-shared provenance events + the multi-synapse keystone test
**Dispatched by PO-5, 2026-07-19, on Sarah's instruction. Advisor round-3 design.**

## Your one objective
Lift provenance hydrolysis events from PER-SYNAPSE to a NETWORK-SHARED pool so cross-synapse
entanglement edges can form WITHOUT η, then run the multi-synapse §8 keystone test. §8 is
**structurally void at single-synapse scale** (one nanodomain = one component = zero bits, LOCKED
`quantum-system-canonical:139`); the keystone lives at the CROSS-synapse scale where input has a real
degree of freedom — **which synapses are active.**

## GROUND FIRST (return a `### GROUNDING BRIEF` before any code, line-quoted, tagged by source)
Read in full: `.claude/skills/agent-grounding-protocol`, `session-discipline`,
`quantum-system-canonical` §5 (LOCKED: partition is cross-synapse), `model6-entanglement-partition-werner`,
`experiment-design-patterns`. Then the research log `RESEARCH_LOG_ENTANGLEMENT_TOPOLOGY.md` entries
**L·PO5-10 … L·PO5-13** (the LOCC finding, the RIG de-risk Q=0.65, the provenance build, the
false-positive override) and `docs/PREREG_PO5_UNIT16_PROVENANCE_BUILD.md`.

## What exists (PO-5 built it; DO NOT rebuild)
`dimer_particles.py`: `provenance_bonding` flag (opt-in, off = bit-identical `1034/369740/0.991922159684`).
Events at Ca-elevated cells (2 slots), dimer claims ≤2 nearest, bond iff shared, EM pathway skipped,
coherence-death kept. **This is PER-SYNAPSE** — each `synapse.dimer_particles` makes its own events from
its own local calcium, so cross-synapse edges are currently ZERO. That is the gap you close.

## The build (surface: `multi_synapse_network.py` partition path — PO-5's, cleared for you)
A **network-level shared event pool**: hydrolysis events carry absolute (x,y) network coordinates; a
dimer in ANY synapse can claim an event within its reach. Two dimers in DIFFERENT synapses that claim
the two daughters of one boundary event → a cross-synapse edge, Fisher-inherited, η-free.
- Opt-in flag; **all-off bit-identical** (verify `1034/369740/0.991922159684`), same discipline as U16.
- Event locations must reflect per-synapse calcium (input-correlated) placed in network coordinates.

## The test (advisor R3 design — pre-register BEFORE running)
6 synapses, **partial** event-pool overlap (place them so pools overlap — VERIFY overlap fraction
before scoring; zero overlap = trivial, full overlap = trivial; partial is the §8 regime). Two
conditions, **same number active, different identity**: {1,2,3} vs {4,5,6}. Density matched by
construction.
- **Metric: Newman modularity of the partition against the ACTIVATION-IDENTITY label** (NOT spatial
  half — that was PO-5's false positive, L·PO5-13). Effect size on component structure across
  conditions, ≥5 seeds.
- **Pre-register the decomposition null:** if partitions split cleanly into per-synapse blocks with NO
  cross-synapse component, the mechanism is input-LOCATED but not input-COMPUTING — a real negative,
  reported as such.
- **THE PRE-REGISTRATION GUARD (advisor R3, now standing):** before running, STATE what value of the
  statistic would make you conclude the OPPOSITE. If no achievable value flips the verdict, the
  statistic is wrong. (This catches the mis-registration that nulled Units 8/9/13/16b.)

## Boundaries / discipline
Explicit-path `git commit -- <paths>` only, never `git add -A`. `git show --stat HEAD` after each.
Opt-in + bit-identical-off for every physics change. Do NOT touch: `spine_plasticity_module.py`,
`atp_system.py` phosphate path, `analytical_gap`, `sweep_runner.py`. Pre-register before scoring;
demonstrate the verdict function failing before it passes; ≥5 seeds (the 3-seed scars). Persist results
+ logs beside the probe under `src/models/Model_6/sweep/` (NOT gitignored `results/`), force-add JSON.

## Return
The grounding brief; the network-shared-event build committed (bit-identical-off proven); the
pre-registered multi-synapse test with its guard-statement; the verdict WITH the decomposition-null
result; a `RESEARCH_LOG` entry + DECISION RECORD row. **Either answer is a result** — a clean negative
(partition decomposes per-synapse) is as valuable as a pass.

---

## ⚠ HANDOFF ADDENDUM — 2026-07-19 · you INHERIT in-progress work, do not start from scratch

A prior worker on this task built ~156 lines of the network-shared-event layer before being
stopped. **PO-5 erroneously reverted it, then recovered it from transcript and committed it** —
commit message begins `wip(model6): RECOVERED PO-7 work`. It is on the branch now.

**What is already built (verify, don't rebuild):**
- `multi_synapse_network.py`: opt-in flag `self.provenance_network = False` (+ `_net_event_rate`,
  `_net_reach_nm = 500`, `_net_age_s`, `_net_slots = 2`, `_net_k = 2`, `_prov_seen`), and a
  `_step_network_provenance(dt, synapses, positions)` method called from `step()` under the flag.
- `sweep/po7_bitident_check.py`: the bit-identity harness.
- **Off-path bit-identity is VERIFIED with this edit in place** (`1034/369740/0.991922159684`).

**What is NOT done (your work starts here):**
1. **Validate the network-provenance layer at the DATA level** — the prior worker's own note:
   *"each synapse is internally sparse; test whether cross-synapse edges actually form, sweeping
   spacing/reach/rate."* That validation was never completed. **Confirm cross-synapse edges form,
   and measure the event-pool OVERLAP fraction — the §8 regime is PARTIAL overlap (0 = trivial,
   full = trivial).** Read `_step_network_provenance` in full first; do not trust it until you have
   SHOWN it produces cross-synapse edges.
2. Everything else in this kickoff above: the pre-registered multi-synapse test (activation-identity
   modularity, decomposition null, the pre-registration guard), ≥5 seeds, verdict with limits.

**Still begin with the `### GROUNDING BRIEF`** — including a line-quoted read of the recovered
`_step_network_provenance` so we know YOU verified what you inherited, not just trusted it.
