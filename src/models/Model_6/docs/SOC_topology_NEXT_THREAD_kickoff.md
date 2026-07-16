# KICKOFF — SOC × entanglement-topology (continue from validated instrument)

**You are a fresh posner-rooted thread.** Per this repo's CLAUDE.md this is a
human-bridged thread: run the GROUND sequence and return a `### GROUNDING BRIEF`
as your first message, then STOP for Sarah's confirmation before building.

## Mission (one line)
The cross-synapse entanglement-topology instrument is built and VALIDATED; run the
SOC drive×damping sweep to test for a power-law (self-organized criticality), and
make the natural-emergence run tractable.

## Ground here (read in full before the brief)
- Skills: `entanglement-topology-measurement`, `model6-entanglement-partition-werner`,
  `model6-network-layer-feasibility-may30`, `model6-architecture`,
  `model6-codebase-operations`, `experiment-design-patterns`,
  `origin-of-life-thermodynamic-thesis`. Cross-domain origin: `self-comprehension-discipline`
  (TALON's Betti/sheaf lessons — the whole approach was imported from there).
- The full arc + results: `src/models/Model_6/docs/SOC_topology_experiment_handoff.md`.
- Code SHOWN (verify, don't trust this doc):
  - `src/models/Model_6/entanglement_topology.py` — `compute_betti` (Betti0/1 via
    coboundary, numpy cross-check) + `compute_synapse_quotient_betti` (the honest
    cross-synapse lens). Run `python3 entanglement_topology.py` — self-test passes.
  - `src/models/Model_6/multi_synapse_network.py` — `NetworkEntanglementTracker.compute_entanglement_topology()`;
    Betti in `step()` + `get_experimental_metrics()` (`betti0_raw/betti1_raw`,
    `betti0_cross/betti1_cross`). Pump: `_update_backbone_field` (line ~1050),
    cross-bond formation gated on `eta_factor=(η_i·η_j)^0.5` (line ~284).
  - `src/models/Model_6/sweep/soc_pump_threshold_stage1.py` ·
    `src/models/Model_6/sweep/soc_topology_stage2.py` ·
    `src/models/Model_6/sweep/soc_topology_forced_eta.py` ·
    `src/models/Model_6/sweep/soc_topology_geometry_discriminator.py`
    (each reproduces in <~2 min; run them from that directory, and use the repo
    venv — `./venv/bin/python` — the system scipy lacks `constants.Boltzmann`).
    NOTE: there are TWO `sweep/` trees. The SOC scripts live under
    `src/models/Model_6/sweep/`; the repo-root `sweep/` holds the older
    `observe_network_partition.py` / `observe_pathway2_selectivity.py` probes.
    Paths here were repo-root-relative and wrong until 2026-07-16.

## What IS (validated, with controls) — do not relitigate, verify
1. Raw whole-graph Betti1 is dominated by within-spine clique-fill (WRONG lens);
   the honest signal is the synapse-quotient `betti1_cross`.
2. Pump threshold `r=p_met_agg/P_c` is analytic in drive `d=E_inv·ca_open`,
   `p_active_max`, coupling `W`. Two-cluster geometry: in-cluster critical
   `d*≈0.087` vs isolated `0.345` (collective coupling lowers ignition 4×),
   honest ceiling `d~0.74` ⇒ pump reachable, falsifier does NOT fire.
3. Ignited pump (η=0.26) on two clusters ⇒ `betti0_cross=2` (Werner partition
   {0,1,2,3}{4,5,6,7}), `betti1_cross=6`. η=0 control ⇒ `betti1_cross=0`
   (same drive/geometry) ⇒ the pump CAUSES the cross-synapse loops.
4. Instrument validated: CHAIN of 6 → `betti1_cross=0` (path/tree); RING of 6 →
   `betti1_cross=1` (they differ by one closing edge). It tracks real topology,
   not node count.

## Tasks (with acceptance)
**T1 — SOC sweep (primary).** Sweep drive d and damping Q; at each cell read the
`betti0_cross` component-size distribution. ACCEPT = evidence for/against a
POWER-LAW that is DRIVE-INDEPENDENT (a real SOC attractor), vs a single tuned
fixed point. Report the negative honestly if that's what the data shows.
Start from `soc_topology_forced_eta.py` (clamped-η is a valid probe — Stage 1
proved η reachable); parameterize a geometry with a NON-complete quotient (avoid
the small-clique K4 regime) so the distribution has range.

**T2 — natural emergence (secondary).** Let η climb on its own (no clamp) to
confirm the ignited-regime topology arises naturally. BLOCKER = the O(n²) compute
wall (~6 min / 0.375 s; a ~30 s ignition run is hours). Fix first: extend
`run_theta_burst_45s.analytical_gap` acceleration to the DRIVE phase, and/or cap
per-spine dimers/bonds (intra-blob clique-fill is topologically trivial and
`betti1_cross` ignores it), and/or run on EC2 per `model6-codebase-operations`.

## Discipline (pre-registered — hold the line)
If honest values won't ignite the pump, won't form cross-synapse loops, or the
critical edge appears only at one tuned drive → ACCEPT the negative; do NOT crank
Q/D/drive to rescue it. `p_active_max` sweep range is [10,65] fW; ω₀/2π=8 MHz and
Q≳10 are pinned (see `model6-network-layer-feasibility-may30`). Werner bound 0.5
is a separability THEOREM — never a fitted knob. Validate at the data/result
level; surgical edits only.

Companion memory (Murmur side): `talon-informs-model6-topology-sequencing`.
