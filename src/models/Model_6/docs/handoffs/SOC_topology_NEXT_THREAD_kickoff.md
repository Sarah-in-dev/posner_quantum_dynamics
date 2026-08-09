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

**T1′ WAS BLOCKED by a UNIT BUG in dissolution — found, fixed, verified (2026-07-16).**
Attempted; the run did not survive its own control. What the control found:

**The bug: dimer dissolution was missing its `dt`.** In
`ca_triphosphate_complex.update_dimerization` (which RECEIVES `dt`, L303), every
FORMATION term correctly carries it — `formation_probability = k_eff·pnc²·dt` (L353),
`nucleation_events = rand < (p·dt)` (L358), `deterministic_formation = k_eff·pnc²·dt·0.5`
(L374) — but the dissolution terms did NOT:
`dimer_dissociation = k_diss·dimer·noise` (L420) and the trimer twin (L421), with the
result added straight in as `self.dimer_concentration += d_dimer_dt` (L427). `k_diss`
is s⁻¹ (`k_classical = 0.005`, declared at L160 as *"s⁻¹; cluster lifetime τ≈200s"*),
so the per-step decrement must be `k_diss·[X]·dt`. **Without it, dissolution ran once
per STEP instead of per SECOND — 1/dt too fast, i.e. 1000× at dt=1e-3.**

**The dt-independence control is what exposed it:** at an identical 4 s of sim,
dimers = **56 (dt=1e-3) / 619 (dt=1e-2) / 1501 (dt=5e-2)** — a 27× spread, which no
rate process can produce. P_S agreed across dt (0.9886/0.9879/0.9802 vs analytic
0.9805) and the edge set was identical — so the coherence and Werner paths were sound
and only the population path was dt-dependent. Consequence pre-fix: in silence dimers
collapsed **2314 → 56 in 4 s** while P_S was still 0.9886, so there was nothing left
to decohere and the far-pairs-first cascade could not be reached.

**FIXED:** `* dt` on L420-421. **Verified:** under sustained drive the system remains
**BOUNDED** — peak dimer saturates (µM by t: 40.5 / 61.6 / 94.5 / 105.6 / 113.0 /
123.3 / 132.3 / 143.8 / 150.1 / **152.6**, increments collapsing +33 → +7 → +2.4).
**So D14's "SOC loop already closed / self-limiting" and D17's "BOUNDED, no runaway"
SURVIVE the fix — the phosphate self-limitation is REAL, not an artifact.**

**BUT the operating point moves ~3×:** post-fix the driven plateau is ~**155 µM**
against A3/D8's **47 µM** and D16's live **49 µM**. Dissolution had been 1000× too
strong and was holding the level down, so D16's *"integration loop now forms 49 µM
(matches A3's 47 µM)"* agreement was partly the bug cancelling out. **Bounded: yes.
Same operating point: no.** D8/D14/D16/D17 need re-reading against the fix — a
physics-conclusions call, Sarah's, not a code call.

**WHAT I GOT WRONG (recorded so the next thread doesn't inherit it).** An earlier
revision of this doc claimed *"the dimer population path is BROKEN — dead
`k_dissolution`"* and called it a §6.5 construct-validity gap. **That was wrong on
mechanism.** Dissolution IS implemented, in the OTHER module
(`ca_triphosphate_complex.py:418`, `k_diss = k_classical·(1-singlet_excess)·template_enhancement`).
The particle-system slaving in `dimer_particles.step_population` is DECLARED DESIGN
(L163-165: *"Chemistry determines HOW MANY dimers exist... Particle system tracks
WHICH ONES and their quantum state"*) and tracks the chemistry faithfully.
`k_dissolution` (L127) is a vestigial duplicate in the wrong module, not a missing
mechanism. The template term multiplying dissolution is correct Option-B detailed
balance; `(1-singlet_excess)` is real coherence-protected persistence. The failure was
grepping `RESEARCH_LOG_CALCIUM_DIMER.md` instead of reading D1-D18.

**T1′ IS NOW UNBLOCKED ON PHYSICS — and re-blocked on COMPUTE.** With dimers persisting,
`step_entanglement`'s O(n²) intra loop explodes: measured **~4.4 s/step** at ~900
dimers/synapse (vs 644 ms pre-fix at ~56). T1′'s 80 s run is then ~98 h. **Gate: vectorise
or gate the INTRA loop** (79% of `net.step`, and topologically irrelevant to
`betti*_cross`). Pre-registration in `sweep/coherence_fragmentation_probe.py` is intact.

**T2 — natural emergence (secondary).** Let η climb on its own (no clamp).
**COST CORRECTED 2026-07-16:** an earlier revision of this doc said "~12 h → ~1–3.5 h
post-vectorisation." **That was wrong** — it extrapolated from a microbenchmark of
`_update_entanglement` alone. Profiled end-to-end, a full `net.step` in silence is
**644 ms**, of which **79% is `dimer_particles.step_entanglement`** (the INTRA-synapse
bond loop: ~241k `np.linalg.norm` calls per step). 80 s at dt=1e-3 = **14.3 hours**.
The cross-synapse loop that was vectorised was never the dominant term at this scale;
Amdahl applies. **The real wall is the intra loop** — and it is topologically IRRELEVANT
to the cross-synapse measurement (`compute_synapse_quotient_betti` reads only
`cross_bonds`; neither `_update_entanglement` nor `step_coherence` reads intra bonds),
so vectorising or gating it is the highest-leverage remaining perf work. Do NOT cap
per-spine DIMERS as a shortcut: a quotient edge needs any of d² cross-pairs to clear
Werner, so capping d moves the quotient topology and is not physics-neutral. (Gating
intra BONDS is a different, neutral matter — it changes no dimer count, no P_S, and no
cross bond — but wants its own control.)

## Discipline (pre-registered — hold the line)
If honest values won't ignite the pump, won't form cross-synapse loops, or the
critical edge appears only at one tuned drive → ACCEPT the negative; do NOT crank
Q/D/drive to rescue it. `p_active_max` sweep range is [10,65] fW; ω₀/2π=8 MHz and
Q≳10 are pinned (see `model6-network-layer-feasibility-may30`). Werner bound 0.5
is a separability THEOREM — never a fitted knob. Validate at the data/result
level; surgical edits only.

Companion memory (Murmur side): `talon-informs-model6-topology-sequencing`.
