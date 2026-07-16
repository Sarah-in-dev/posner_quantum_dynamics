# SOC × entanglement-topology experiment — handoff (2026-07-15)

Origin: cross-pollination from the TALON self-comprehension program (the murmur
platform). TALON built and mostly *killed* the expensive topology methods
(persistent-homology early-warning = vacuous; connection-Laplacian/Procrustes
sheaf = degenerate on hollow nodes) and found the one thing that earns is the
cheap invariant: connected-component gluing + a linear-algebra Betti count over
the coboundary. That lesson is imported here: **measure the cheap honest
invariant first; make it earn before investing in vineyards.**

## What was built (all in `src/models/Model_6/`)

- **`entanglement_topology.py`** — `compute_betti(...)` (Betti0=components,
  Betti1=independent loops = `E − rank(δ)`, with numpy coboundary cross-check),
  and `compute_synapse_quotient_betti(...)`. Self-tests pass.
- **`multi_synapse_network.py`** — `NetworkEntanglementTracker.compute_entanglement_topology()`;
  Betti fields threaded into `step()` and `get_experimental_metrics()` as
  `betti0_raw / betti1_raw / raw_component_sizes` and **`betti0_cross / betti1_cross`**.
All sweep scripts below live under **`src/models/Model_6/sweep/`** — NOT the
repo-root `sweep/` tree (which holds the older `observe_network_partition.py`
probes). Run them from that directory with the repo venv (`./venv/bin/python`);
the system scipy lacks `constants.Boltzmann`. (Paths in this doc were
repo-root-relative and wrong until 2026-07-16.)

- **`sweep/read_live_betti.py`** — drive a short run, read live Betti.
- **`sweep/soc_pump_threshold_stage1.py`** — cheap analytic pump-threshold sweep.
- **`sweep/soc_topology_stage2.py`** — full two-cluster ignition run (gated).

## Findings so far

1. **Raw whole-graph Betti1 is the WRONG lens.** First live run (5 synapses,
   3 theta cycles): Betti0=5, raw Betti1=219,594 — but `cross_synapse_bonds=0`.
   Every "loop" is intra-spine clique-fill (one dense blob per spine). The
   honest signal is **`betti1_cross`** (synapse-quotient graph: synapses=nodes,
   Werner-cleared cross-bonds=edges). It was 0 (no cross-bonds formed).

2. **No cross-bonds formed because the Fröhlich pump was sub-threshold.**
   `P_met=0.84fW ≪ P_c=21.51fW ⇒ η=0`. Cause: at 0.375 s of sim, `E_invasion`
   hadn't reached its 0.1 onset (it only hits ~0.5 by 30 s). Geometry was fine;
   drive-time was the issue.

3. **Stage 1 (analytic) — the pump IS reachable with honest drive.** Threshold
   `r=p_met_agg/P_c` depends only on drive `d=E_inv·ca_open`, `p_active_max`, and
   the coupling matrix `W`. Two-cluster geometry (4+4, 15 µm apart, λ=5 µm):
   clustered-synapse row-sum = 3.96, so critical drive **d\* = 0.087**
   (isolated synapse d\* = 0.345 — collective coupling lowers ignition **4.0×**).
   Honest ceiling `d ~ 0.74`. **Falsifier does NOT fire.** At d=0.15 → η≈0.26.
   → This is the concrete network effect: a lone spine barely ignites (d\*≈0.35,
   near ceiling); a co-active cluster ignites easily (d\*≈0.09).

4. **Stage 2 (forced-η probe) — betti1_cross OBSERVED; the open question is
   ANSWERED.** `sweep/soc_topology_forced_eta.py` clamps η into the ignited
   regime (valid because Stage 1 proved η reachable) on a short window (~46 s
   compute), dodging the wall. At η=0.26 on the two-cluster 8-synapse net:
   **betti0_cross = 2** (the Werner partition {0,1,2,3}{4,5,6,7} confirmed at the
   synapse-quotient level) and **betti1_cross = 6** — closed cross-synapse loops
   (each cluster = complete K4 of synapse-nodes ⇒ 3 loops ×2; quotient 8 nodes /
   12 edges; coboundary cross-check ok). Topology plateaued by t≈0.04 s.
   **Negative control η=0: betti1_cross = 0** (same drive/geometry/~2300 dimers,
   zero cross-bonds) ⇒ the ignited pump CAUSES the cross-synapse loops.
   Caveats: η clamped (not naturally climbed); K4-per-cluster is *complete* only
   because 4 nodes ~1 µm apart all couple — a larger/sparser geometry is needed
   to show betti1_cross tracks genuine structure vs small-cluster completeness.

5. **Stage 3 validation — the instrument tracks REAL topology, not node count.**
   `sweep/soc_topology_geometry_discriminator.py` (eta=0.26): a CHAIN of 6
   synapses @2.5 µm realizes a quotient PATH (edges 0-1..4-5) → **betti1_cross=0**
   (tree); a RING of 6 @2.5 µm realizes the same path PLUS the closing edge (0,5)
   → **betti1_cross=1** (6-cycle). Identical node count and near-neighbour
   coupling; they differ by exactly one closing edge, and the instrument reads
   0→1 accordingly (coboundary cross-check ok in both). The physics realized the
   predicted graph exactly (nearest-neighbour cross-bonds form; next-nearest fall
   below the Werner bound). ⇒ Stage-2 betti1_cross=6 is a trustworthy topological
   measurement; the "K4 completeness" caveat is retired.

6. **(2026-07-16) The partition IS a coherence-set distance graph — and the SOC
   sweep is RETIRED.** `F = P_S_i·P_S_j·w` with `w = exp(-d/5µm)`, cut at `F>0.5`,
   is algebraically a distance rule: bond iff `d < d* = 5·ln(P_product/0.5)`.
   Measured P_S≈0.998 ⇒ **d* = 3.45µm**. Two theorems fall out of the Werner bound,
   neither tuned: (a) `w≤1 ⇒ F≤P_S²`, so `F>0.5` requires **P_S > 1/√2 ≈ 0.7071** —
   a hard coherence floor below which the cross-synapse partition cannot exist at
   any distance; (b) the radius SHRINKS as coherence decays (3.47µm at P_S=1 → 0 at
   0.7071), so the partition must fragment **far-pairs-first**.
   Confirmed against a pre-registered 8-synapse ladder (`sweep/coherence_radius_probe.py`,
   ~43 s): 7/7 gaps called correctly, exact predicted edge list, `betti0_cross=3`,
   `component_sizes=[3,3,2]`, `betti1_cross=0`, crosscheck ok. It **retrodicts
   finding 5 with no free parameters** (chain: 2.5<3.45 bonds, next-nearest 5.0
   doesn't → the 5 observed path edges; ring: hexagon side 2.5 bonds, chord 4.33 and
   opposite 5.0 don't → the 6 observed edges).

7. **(2026-07-16) `_update_entanglement` vectorised** (862481d). Physics unchanged;
   23–43× and growing with dimer count; cost linear in cross-pairs. The compute wall
   is no longer the binding constraint *for the cross loop*. Do NOT cap per-spine
   dimers as a shortcut — a quotient edge needs any of d² cross-pairs to clear
   Werner, so the cap moves the quotient topology and is not physics-neutral.
   **COST CLAIM CORRECTED (same day):** an earlier revision of this line said
   "T2: ~12 h → ~1–3.5 h." **Wrong** — extrapolated from a microbenchmark of
   `_update_entanglement` alone. Profiled end-to-end, a full `net.step` in silence is
   **644 ms**, of which **79% is `dimer_particles.step_entanglement`** (the INTRA bond
   loop, ~241k `np.linalg.norm` per step). 80 s at dt=1e-3 = **14.3 hours**. The
   vectorised cross loop was never the dominant term at this scale — Amdahl. The real
   wall is the INTRA loop, which is topologically IRRELEVANT to the cross-synapse
   measurement (`compute_synapse_quotient_betti` reads only `cross_bonds`; neither
   `_update_entanglement` nor `step_coherence` reads intra bonds) — so vectorising or
   gating it is the highest-leverage perf work left.

8. **(2026-07-16) UNIT BUG: dimer dissolution was missing its `dt`. Found, fixed,
   verified.** `ca_triphosphate_complex.update_dimerization` RECEIVES `dt` (L303) and
   every FORMATION term carries it — `k_eff·pnc²·dt` (L353), `rand < p·dt` (L358),
   `k_eff·pnc²·dt·0.5` (L374) — but DISSOLUTION did not:
   `dimer_dissociation = k_diss·dimer·noise` (L420) plus the trimer twin (L421), added
   straight in via `self.dimer_concentration += d_dimer_dt` (L427). `k_diss` is s⁻¹
   (`k_classical = 0.005`, declared L160 *"s⁻¹; cluster lifetime τ≈200s"*), so the
   per-step decrement must be `k_diss·[X]·dt`. **Without it dissolution ran per STEP
   instead of per SECOND — 1/dt too fast, 1000× at dt=1e-3.**
   **The dt control exposed it:** same 4 s of sim → dimers = 56/619/1501 at
   dt=1e-3/1e-2/5e-2 (27× spread — impossible for a rate process), while P_S
   (0.9886/0.9879/0.9802 vs analytic 0.9805) and the edge set were dt-INDEPENDENT.
   Coherence and Werner paths sound; only the population path was wrong. Pre-fix
   consequence: in silence dimers collapsed 2314 → 56 in 4 s while P_S was still
   0.9886 — nothing left to decohere.
   **FIXED** (`* dt` on L420-421). **VERIFIED:** sustained drive stays **BOUNDED** —
   peak dimer saturates (µM: 40.5/61.6/94.5/105.6/113.0/123.3/132.3/143.8/150.1/**152.6**,
   increments collapsing +33 → +7 → +2.4). **⇒ D14's "SOC loop already closed /
   self-limiting" and D17's "BOUNDED, no runaway" SURVIVE — the phosphate
   self-limitation is REAL, not an artifact of over-dissolution.**
   **BUT the operating point moves ~3×:** post-fix driven plateau ~**155 µM** vs
   A3/D8's **47 µM** and D16's live **49 µM**. Dissolution had been 1000× too strong
   and was holding the level down, so D16's *"forms 49 µM (matches A3's 47 µM)"* was
   partly the bug cancelling out. **Bounded: yes. Same operating point: no.**
   D8/D14/D16/D17 want re-reading against the fix — a physics-conclusions call.

   **CORRECTION — what an earlier revision of this doc got wrong.** It claimed *"the
   dimer population path is BROKEN — dead `k_dissolution`"* and called it a §6.5
   construct-validity gap. **Wrong on mechanism.** Dissolution IS implemented, in the
   OTHER module (`ca_triphosphate_complex.py:418`,
   `k_diss = k_classical·(1-singlet_excess)·template_enhancement`). The particle-system
   slaving in `dimer_particles.step_population` is DECLARED DESIGN (L163-165: *"Chemistry
   determines HOW MANY dimers exist... Particle system tracks WHICH ONES and their
   quantum state"*) and tracks the chemistry faithfully. `k_dissolution` (L127) is a
   vestigial duplicate in the wrong module, not a missing mechanism. The template term
   multiplying dissolution is correct Option-B detailed balance; `(1-singlet_excess)` is
   real coherence-protected persistence (the D15 hypothesis, implemented). The failure
   was grepping `RESEARCH_LOG_CALCIUM_DIMER.md` instead of reading D1-D18.

## The remaining tasks (dedicated thread)

- **~~SOC drive×damping sweep~~ — RETIRED 2026-07-16, on structural grounds, for
  ~0 compute.** A power law in `betti0_cross` component sizes cannot arise here.
  (a) *1D forbids it:* the honest dendrite is 1D, connectivity is decided entirely
  by consecutive gaps, so clump size is exactly geometric `P(k)=p^(k-1)(1-p)` —
  exponential tail (verified: geometric fit R²=0.97–0.996 at every density, beating
  the power-law fit every time). Regular spacing is worse — binary, no distribution
  at all. 2D near percolation DOES give a power law, but it is pure geometry: no
  drive, no η, nothing quantum. (b) *D18 forbids it:* nucleation is an all-or-none
  bistable switch with a forbidden gap and a QUANTAL drive-independent ON amplitude
  — a characteristic scale, where SOC requires scale-freedom.
  **Two "SOC" claims were conflated:** *SOC-chemistry* (phosphate depletion → S
  self-organises to 1) is ESTABLISHED (D8/D14) and stands; *SOC-topology* (power-law
  clump sizes) is what this task chased. It inherited the name.

- **Coherence-radius, dynamic half — UNBLOCKED ON PHYSICS by the finding-8 fix; now
  gated on COMPUTE.** The question stands: does the partition fragment far-pairs-first
  as P_S decays, and die at P_S=1/√2? Pre-registered from measured T_eff=151.6 s
  (`sweep/coherence_fragmentation_probe.py`, written before the run and still intact):
  gap 3.0 breaks at **t=9.5 s**, 2.8 at **13.6 s**, 2.5 at **19.9 s**, 2.0 at
  **30.4 s**, partition gone at **75.1 s**; the 4.5 gaps have P_S_crit=1.109>1 so they
  never bond at all — which independently retrodicts the static probe. **Discriminating:**
  a classical scalar eligibility trace decays uniformly and carries no spatial structure,
  so it cannot produce a spacing-ordered fragmentation.
  Pre-fix this was unreachable (population died ~40× faster than coherence). **With the
  `* dt` fix, dimers persist — which is exactly what the experiment needs.** The blocker
  moved from physics to compute: persistent dimers make `step_entanglement`'s O(n²) intra
  loop explode to **~4.4 s/step** (~900 dimers/synapse vs ~56 pre-fix), so an 80 s run is
  ~98 h. **GATE: vectorise or gate the INTRA loop** — 79% of `net.step`, and topologically
  irrelevant to `betti*_cross`. Same shape as the cross-loop fix already landed in 862481d.

- **Natural emergence.** Let η climb on its own (no clamp). See the corrected cost in
  finding 7: 14.3 h at dt=1e-3, and the real wall is the INTRA `step_entanglement` loop
  (79% of `net.step`), not the vectorised cross loop. Gating/vectorising the intra loop
  is the highest-leverage perf work left, and it is topologically neutral for
  `betti*_cross`. Also blocked by finding 8 if the run needs P_S to decay.

## The original next task (Stage 2, pre-result — kept for context)

Run `soc_topology_stage2.py` driving both clusters to `d ≳ 0.15` and HOLD until
`E_invasion` climbs and η>0, then read **`betti1_cross`** in the ignited regime.
The open scientific question: once the pump is on and cross-synapse bonds form,
does the cross-synapse graph carry **loops** (`betti1_cross > 0` — closed
entanglement paths across spines) or stay a tree/forest? That is the entangle-
ment-topology-IS-the-computation claim, tested honestly.

### Compute wall (must solve first)
O(n²) bond recalc: ~0.375 s sim ≈ 6 min at 5 synapses/2400 dimers; a faithful
~30–40 s ignition run is many hours. Options: (1) extend `analytical_gap`
acceleration to the drive phase or throttle tracker-recalc cadence; (2) cap
per-spine dimers/bonds (intra-blob clique-fill is topologically trivial and
`betti1_cross` ignores it anyway); (3) run as a long background / Fargate job.

### Discipline (pre-registered)
If honest values won't ignite the pump, or won't form cross-synapse loops, or the
critical edge appears only at one tuned drive — **accept the negative**; do NOT
crank Q/D/drive to rescue it. The SOC signature to look for is a **power-law in
`betti0_cross` component sizes** across a drive×damping sweep (drive-independent
attractor), not a single tuned fixed point.

Companion (murmur-side memory): `talon-informs-model6-topology-sequencing`.
