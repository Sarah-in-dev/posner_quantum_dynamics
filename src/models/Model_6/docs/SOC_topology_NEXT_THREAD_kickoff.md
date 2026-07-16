# KICKOFF — SOC × entanglement-topology (continue from validated instrument)

**You are a fresh posner-rooted thread.** Per this repo's CLAUDE.md this is a
human-bridged thread: run the GROUND sequence and return a `### GROUNDING BRIEF`
as your first message, then STOP for Sarah's confirmation before building.

## Mission (one line)
The cross-synapse entanglement-topology instrument is built and VALIDATED, and the
partition is now known to be a **coherence-set distance graph** (`d*=3.45µm`);
run the DYNAMIC half — does the partition fragment far-pairs-first as coherence
decays, dying at the `P_S=1/√2` floor?

**Superseded 2026-07-16:** this doc previously set the mission as "run the SOC
drive×damping sweep to test for a power-law." **That task (T1) is RETIRED** — the
1D geometry forbids a power law analytically and D18's quantal/bistable nucleation
rules it out mechanistically. See Tasks below. The compute wall it was blocked on
is also gone (vectorised, 862481d).

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
5. **(2026-07-16) The Werner partition IS a coherence-set distance graph.**
   `F = P_S_i·P_S_j·w`, `w = exp(-d/5µm)`, cut at `F>0.5` ⇒ two synapses bond iff
   `d < d* = 5·ln(P_product/0.5)`. Measured P_S≈0.998 ⇒ **d*=3.45µm**. Confirmed
   against a pre-registered 8-synapse ladder (`sweep/coherence_radius_probe.py`,
   ~43 s): 7/7 gaps called correctly, exact edge list, `betti0_cross=3`,
   `component_sizes=[3,3,2]`, `betti1_cross=0`. It also **retrodicts finding 4
   with no free parameters** (chain: 2.5<3.45 bonds, 5.0 doesn't → 5 path edges;
   ring hexagon: side 2.5 bonds, chord 4.33 and opposite 5.0 don't → 6 edges).
6. **`_update_entanglement` is vectorised** (862481d) — physics unchanged, 23–43×,
   growing with dimer count; cost linear in cross-pairs. Compute is no longer the
   binding constraint.

## Tasks (with acceptance)

**T1 (SOC power-law sweep) is RETIRED — do not run it.** Killed on structural
grounds for ~0 compute, 2026-07-16, not on a negative result. Two independent
reasons, both from work already in this repo:
- **1D geometry forbids it.** The honest dendrite is 1D (`_generate_positions`
  `'linear'` = "synapses along a straight dendrite"; coupling runs along the MT
  backbone). In 1D connectivity is decided entirely by CONSECUTIVE gaps (if i and
  i+2 are within d*, i+1 must be), so clumps are runs of small gaps and clump size
  is *exactly* geometric: `P(k)=p^(k-1)(1-p)` — an exponential tail. Verified:
  geometric fit R²=0.97–0.996 at every density, beating the power-law fit every
  time. Regular spacing (what every current rig uses) is worse still — binary:
  ≤d* → one giant clump, >d* → all singletons. No distribution to fit at all.
  A power law IS reachable in 2D near percolation, but that is PURE GEOMETRY —
  no drive, no η, no quantum anything.
- **D18 rules it out mechanistically.** `RESEARCH_LOG_CALCIUM_DIMER.md` D18:
  dimer nucleation is an all-or-none bistable switch with a FORBIDDEN GAP (0 of
  480 replicates between ~8 and ~125 dimers) and a QUANTAL, drive-independent ON
  amplitude (~135). That is a characteristic SCALE. SOC means scale-FREE. Drive
  tunes only `P(catch)`, a sharp sigmoid.

**Two different "SOC" claims were conflated.** *SOC-chemistry* — phosphate
depletion self-limits formation, S self-organises to 1 — is **established** (D8,
D14: "SOC loop already closed in live code"); it is a self-organised fixed point,
single-synapse. *SOC-topology* — power-law clump sizes — is what T1 chased, and
the geometry forbids it. T1 inherited the name from the chemistry result.
(NOTE: `quantum-computation-and-attribution:81` still says "the phosphate-runaway
means the reset feedback does not yet exist" — **stale**; D8/D14 closed it on
2026-06-28. That skill owes an update.)

**T1′ — the coherence-radius program (replaces T1).** The static half is DONE
(finding 5). The open half is DYNAMIC and is the real experiment:

> As coherence decays over the ~100 s window, `d*` shrinks and the partition must
> fragment **geometrically — the most distant pairs decoupling FIRST**, dying
> entirely at the hard floor `P_S = 1/√2 ≈ 0.7071` (since `w ≤ 1` ⇒ `F ≤ P_S²`;
> the Werner separability bound IS a coherence threshold — a theorem, not a knob).

ACCEPT = the observed edge-loss ORDER matches the pre-registered one. On the
`coherence_radius_probe.py` ladder (gaps 2.0/2.5/4.5/3.0/2.0/4.5/2.8), edges must
break as P_S falls through `P_S = sqrt(0.5·e^(gap/5))`:

| gap (µm) | edge breaks when P_S falls below |
|---|---|
| 3.0 | 0.9545 |
| 2.8 | 0.9356 |
| 2.5 | 0.9080 |
| 2.0 | 0.8637 |
| any | 0.7071 → partition gone entirely |

FALSIFIED IF the partition fragments uniformly rather than far-pairs-first, or if
fragmentation does not track P_S through d*. **This is the discriminating test:**
a classical scalar eligibility trace decays uniformly and carries no spatial
structure — it cannot produce a spacing-ordered fragmentation. This is
`coherence-gated-learning`'s "the eligibility trace IS the persisting topology"
made measurable.

NEEDS: a run long enough for P_S to actually decay (at 0.08 s it has not moved —
measured P_S=0.9987). Now ~1–3.5 h post-vectorisation, i.e. one background job.

**T2 — natural emergence (unchanged, secondary).** Let η climb on its own (no
clamp) to confirm the ignited-regime topology arises naturally. ~12 h → ~1–3.5 h
post-vectorisation; a background job, no longer a blocker. If it needs to be
faster the target is the bond DICT (O(live bonds) Python writes per step), not
the pair arithmetic — per-synapse-pair arrays instead of a global dict. Do NOT
cap per-spine dimers as a shortcut: the cap is NOT physics-neutral (a quotient
edge needs any of d² cross-pairs to clear Werner, so capping d changes the
quotient topology) and would need its own convergence control. Vectorising
removed the need for it.

## Discipline (pre-registered — hold the line)
If honest values won't ignite the pump, won't form cross-synapse loops, or the
critical edge appears only at one tuned drive → ACCEPT the negative; do NOT crank
Q/D/drive to rescue it. `p_active_max` sweep range is [10,65] fW; ω₀/2π=8 MHz and
Q≳10 are pinned (see `model6-network-layer-feasibility-may30`). Werner bound 0.5
is a separability THEOREM — never a fitted knob. Validate at the data/result
level; surgical edits only.

Companion memory (Murmur side): `talon-informs-model6-topology-sequencing`.
