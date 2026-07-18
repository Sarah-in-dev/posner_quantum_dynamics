# SUBSTRATE AUDIT — Model 6, 2026-07-18

*Adversarial code audit. Commissioned in response to an external reviewer's question set.
Every claim carries `file:line`. Claims that could not be verified from CODE are marked
**UNVERIFIED** — skills, docstrings, and research-log prose were explicitly NOT accepted as
evidence, because several were found false in the course of this audit.*

**Method.** Four parallel read-only agents against worktree `nervous-hertz-7ccff6` at
`268a64d`. Rules given to each: cite `file:line` for every claim; write UNVERIFIED rather
than infer; a docstring stating a value is not the same as the code using it. Sections F/G
were done directly.

**Scope note.** Line numbers are against `268a64d` plus the uncommitted plateau-voltage
change. `multi_synapse_network.py` and `model6_core.py` were edited the same day; several
previously-cited line numbers had moved (e.g. `collapse_factor` 1317 → 1424) and were
re-located by name.

---

## 0. THE HEADLINE — ranked by severity

1. **A factor-of-2π error on the per-synapse pump.** `vibrational_cascade_module.py:315`
   hand-rolls Bose-Einstein with a locally shadowed `hbar` against a **linear** frequency:
   `x = hbar * p.omega_0 / (k_B * T)` where `p.omega_0 = 40.0e9 Hz` (`:85`). Correct
   `x = 6.19e-3`; live `x = 9.85e-4`. **n̄ inflated 6.28×.** Propagates to `N_total` (`:325`),
   `b` (`:334`), `source` (`:339`), `gamma_0` (`:363`) ⇒ **η** (`:357`),
   `lifetime_enhancement` (`:367`), regime classification (`:370-379`). Does NOT affect
   `r_c` (`:329`), so the above/below-threshold boolean survives; the condensation
   MAGNITUDE is wrong. Second instance at `:755-756`.
   **The backbone pump is CORRECT** — `model6_parameters.py:46` uses `h * f_hz`, verified
   across eight call sites, none passing an angular frequency.

2. **The calibration fiction survives on the live path, and is structurally unsweepable.**
   `kT_ref = 22.1` is a **function-body literal** at `vibrational_cascade_module.py:246` —
   not a dataclass field, therefore invisible to `TubulinCascadeParameters` and to
   `sweep/sweep_runner.py`. With `r_at_E_ref = 100.0e9` (`:115`, comment: *"Calibrated so
   that full MT invasion (22 kT field) produces r > r_c"*) and
   `r_c = (φ/(D+1))(1+φ/χ) ≈ 9.57e10` (`:212-214`):
   **r/r_c ≈ 1.045 at MT+ is an ARITHMETIC IDENTITY of two tuned numbers, not a result.**

3. **Docstrings assert mechanisms that do not exist in the code.** Three confirmed:
   - `multi_synapse_network.py:1332-1334` claims plasticity drive is
     *"Hill(committed_count) with n=4, K_half=20"*. **CORRECTED 2026-07-18 — the audit's
     original wording ("no Hill function exists") was WRONG and is retracted.**
     `_committed_count_to_drive` DOES exist (`:1319`) and IS live. Its **only** caller is
     `_evaluate_independent_gate` (`:1561`) — the CONTROL condition. The coordinated gate
     does not use it: the May-12 DDSC rewire replaced direct commitment with a token
     consumed by CaMKII, and the docstring was never updated. So the real finding is
     sharper: **the control and the experimental condition convert `committed_count` to
     drive by DIFFERENT mechanisms** (control = Hill; coordinated = CaMKII integrator),
     and `:1501` states this backwards, claiming the control uses "the same Hill function
     as the coordinated gate".
   - `:1238-1242` claims *"EMERGENT from physics… No fitted parameters!"* while
     `field_threshold_kT = 20.0` (`:717`) and `mean_eligibility > 0.3` (`:1245`) sit on that
     exact path.
   - `:1423-1425` claims bonds are *"reduced to 30% strength"* via `collapse_factor = 0.3`
     (`:1424`) — **never read**. The actual effect is `np.random.random() < 0.8` removal of
     discordant bonds only (`:1466`).

4. **Cited sources contradict the values attached to them.** `phi_dissipation = 10.0e9`
   (`vibrational_cascade_module.py:99`) cites Zhang 2019, **which gives 6 GHz**.
   `chi_redistribution = 0.05e9` (`:104`) cites the same paper, **which gives 0.07 GHz**.

5. **The two pump sites run different threshold physics.** Backbone: `n_ex = n̄_s`
   (`multi_synapse_network.py:1120-1123`). Per-synapse: Zhang Eq. 4
   (`vibrational_cascade_module.py:213-214`, `:329`). The May-30 migration was applied to
   one and not the other.

---

## A. PINNED-SET CONFORMANCE

### Backbone (`multi_synapse_network.py:1118-1141`)

| pinned | live | file:line | status |
|---|---|---|---|
| ω₀ = 8 MHz | `8.0e6` | `model6_parameters.py:800` | DONE |
| Q ≳ 10 | `10.0` | `model6_parameters.py:801` | DONE |
| D ≳ 200 | `50` | `model6_parameters.py:775` | **DRIFTED** — and dead (no live reader) |
| φ ≤ ω₀ | `8.0e9` vs `8.0e6` | `:781` / `:800` | **DRIFTED, 1000× over** — masked by being dead |
| χ | `0.06e9` | `:786` | DRIFTED (dead) |
| `n_ex = n̄_s` | `P_c = n̄·ℏ·(2πf)²/Q` | `multi_synapse_network.py:1123` | DONE |

### Per-synapse (`vibrational_cascade_module.py`)

| pinned | live | file:line | status |
|---|---|---|---|
| ω₀ | `40.0e9` | `:85` | different lattice scale (cited: Pandey & Cifra 2024, 40-160 GHz) |
| Q | **absent from the file entirely** | — | NOT STARTED |
| D | `20` | `:87` | DRIFTED |
| φ ≤ ω₀ | `10e9` ≤ `40e9` | `:99` | DONE |
| threshold form | Zhang Eq. 4 | `:213-214`, `:329` | NOT STARTED |

---

## B. MAY-30 OPEN ITEMS

| item | status | evidence |
|---|---|---|
| B2 per-synapse pump on `n_ex = n̄_s` | NOT STARTED | `vibrational_cascade_module.py:213-214`, `:329` |
| f_coherent 0.10 → 0.08 | **NOT STARTED — still degenerate** | base `0.10` (`model6_parameters.py:496`), max `0.10` (`em_tryptophan_module.py:695`); `:696` computes `0.10 + (0.10−0.10)·η` ≡ 0.10. **The backbone→spine coupling channel is a provable no-op regardless of η.** |
| phosphate pool finite | DONE | `atp_system.py:352`; consumed `model6_core.py:450-452` |
| dissolution returns Pi | DONE | `ca_triphosphate_complex.py:438` — `_po4_consumed` is net, goes negative on dissolution |
| 2% ATP replenish grounded | **mechanism does not exist** | `atp_system.py:154-155` is exponential toward baseline (τ≈5 s, Rangaraju 2014 cited for the FORM). No 2% step anywhere. |
| D, φ at per-synapse site | DRIFTED (D=20) / DONE (φ) | `:87`, `:99` |
| `collapse_factor` unused | still dead | `multi_synapse_network.py:1424` |

**Note on f_coherent:** this is a KNOWN PARKED item, not a new discovery —
`model6-entanglement-partition-werner` §2 states *"Keep f_coherent parked… the
eta/condensation part the partition relies on works."* η reaches the partition via
`k_cross ∝ sqrt(η_i·η_j)` (`multi_synapse_network.py:309,321`), not via `f_coherent`.

---

## C. CALIBRATION-FICTION SWEEP

### C1 — named constants

| token | file:line | value | live? |
|---|---|---|---|
| `r_at_E_ref` | `vibrational_cascade_module.py:115`, read `:248` | `100.0e9` | **LIVE, primary path** |
| `pump_exponent` | `:118`, read `:248` | `2.0` | LIVE |
| `kT_ref` / `22.1` | `:246` | `22.1` | **LIVE — function-body literal, unsweepable** |
| `40e9` | `:85` | `40.0e9` | LIVE (cited) |
| `10e9` | `:99` | `10.0e9` | LIVE |
| `100e9` | — | — | absent (written `100.0e9`) |

The `:197` fallback branch is reached only when `collective_field_kT` is None/≤0, and
`model6_core.py:565` always passes it — effectively dead in the wired system.

### C2 — citation status

**~90 constants carry NO citation text at all.** Highest-impact live ones:
`nmda_fraction=0.5` (`model6_parameters.py:140`), `omega_0=8.0e6` (`:800`), `Q=10.0` (`:801`,
"(slip-layer damping)" — no source), `p_active_max_W=60e-15` (`:804`, "sweep range" — a
bound, not a derivation), `productive_fraction=0.01` (`ca_triphosphate_complex.py:155`,
**self-declared free parameter**), `tau_reference/chi_reference/alpha = 60.0/0.5/6.0`
(`:261-263`, three bare tuning constants), `P_S_threshold=0.5` (`quantum_coherence.py:102`),
all `DDSCParameters` (`ddsc_module.py:16-24`), `K_half=20.0` (`ddsc_module.py:116`,
comment: *"Tune this based on expected integration"*).

**~27 more are CITATION-SHAPED BUT UNRESOLVABLE.** Three failure modes:

- **(a) Paper cited for the phenomenon, number tuned.** `phi_dissipation` and
  `chi_redistribution` (above); `E_ref_pump = 1.4e9` (`vibrational_cascade_module.py:113`)
  cites Azizi/Kurian 2023 while the value is *"(from current em_tryptophan_module typical
  output)"* — self-referential; `modulation_per_sqrt_dimer = 0.15`
  (`local_dimer_tubulin_coupling.py:97`) cites Reimers 2009 for the regime then states
  *"Prefactor 0.15: puts backbone at r/r_c ≈ 2-5"* — calibration wearing a citation.
- **(b) Circular self-validation.** `local_dimer_tubulin_coupling.py:211-212` claims Model
  6's ~50 dimers *"independently validates Fisher's 50-dimer prediction"*, but
  `n_dimer_threshold = 50` (`model6_parameters.py:657`) is itself declared "Fisher's
  prediction".
- **(c) A locator that appears fabricated.** `model6_parameters.py:230`: *"Wang et al. 2024
  Nature Commun 15:1234"* — page `1234` is placeholder-shaped. **Do not cite externally
  before verifying.**

**Arithmetic that contradicts its own comment:** `photon_flux_baseline = 20.0`
(`model6_parameters.py:567`) — the derivation in the two lines above computes ≈1.5.

**Two live values for one constant:** `K_CaHPO4` = 588 M⁻¹ used by the chemistry
(`ca_triphosphate_complex.py:85`, Moreno & Brown 1966) vs 470 M⁻¹ declared
(`model6_parameters.py:216`, McDonogh 2024; same 470 at `atp_system.py:436`).

---

## D. EMERGENT vs PRESCRIBED

| item | verdict | evidence |
|---|---|---|
| MT invasion — `_mt_invaded` | **PRESCRIBED** | `model6_core.py:1006-1012`; step change in tryptophan count |
| MT invasion — cross-synapse gate | **PRESCRIBED** | `multi_synapse_network.py:303`, hard boolean AND |
| MT invasion — `E_invasion` | **EMERGENT** | `spine_plasticity_module.py:411-412`, continuous actin ODE |
| …do the two connect? | **NO** | Nothing reads `E_invasion` to set `_mt_invaded` or vice versa. **There is no continuous path to bypass** — the mechanisms run in parallel and never meet. |
| Ion pair → PNC | **PRESCRIBED** | `ca_triphosphate_complex.py:216-223`, fixed 50% binding fraction, no rate |
| Dimer formation | **EMERGENT form, PRESCRIBED coefficients** | second-order mass action `:346,353`; bare `*0.5` `:356`, `pnc*0.1` `:363`, flat `1e-10` `:369` |
| Supersaturation gate | **PRESCRIBED** | `:398-401`, hard on/off, Ksp `1e-26` (comment: "validate vs the Ksp BAND") |
| Dissolution / detailed balance | **PARTIAL** | `template_enhancement` multiplies BOTH directions (`:346`, `:418`) — real detailed balance, restored. But `singlet_excess` (`:413`) multiplies **only the reverse rate**, so **coherence shifts the equilibrium by construction.** |
| Commitment (network) | **EMERGENT integrator + PRESCRIBED threshold; genuinely DDSC-delayed** | `multi_synapse_network.py:1399-1404` → `model6_core.py:671-685`; `molecular_memory = pT286·GluN2B` (`camkii_module.py:266`); threshold 0.5 chosen |
| Commitment (standalone) | **PRESCRIBED, immediate** | `model6_core.py:620-654`, guarded by `not _network_controlled` |
| Three-factor gate | **PRESCRIBED** | `:1385-1392`, `calcium_uM > 0.5` AND `count > 0` |
| Cross-synapse criterion | **THEOREM** | `WERNER_ENTANGLEMENT_BOUND = 0.5` (`:94`, `:527`) |
| …the fidelity it tests | **PRESCRIBED scaling** | `F = P_S_i·P_S_j·w`, `w = exp(−d/5.0µm)`, `coupling_length_um = 5.0` (`:97`) |
| Intra-synapse edges | **NO Werner test** | `:521-525` unioned by bare existence. The partition is asymmetric. |
| Condensate η | **EMERGENT** | `:1124-1140`, exactly `(r−1)/(r+1)`; input scales `P_BASAL_W`, `p_active_max_W` chosen |

**Widest-blast-radius CHOSEN constants** (full inventory ~55 entries; these scale whole
mechanisms rather than bounding them): `productive_fraction=0.01`
(`ca_triphosphate_complex.py:155`) scales ALL dimer formation; `pnc_binding_fraction=0.5`
(`:125`) sets the entire substrate pool; Ksp `1e-26` (`:399`); `E_ref=1.87`
(`spine_plasticity_module.py:119`) — a measured run's asymptote fed back as a constant, and
**UNVERIFIED**: no artifact in the tree ties it to a result; `p_active_max_W=60e-15` — no
derivation in code.

**THEOREM-grounded (the clean list):** Werner 0.5; P_S thermal floor 0.25
(`dimer_particles.py:283,329`); `η = (r−1)/(r+1)`; Bose-Einstein `P_c`; Jahr-Stevens Mg
block (`analytical_calcium_system.py:215`); Naraghi-Neher (`:222-224`).

---

## E. DEBT INVENTORY DELTA

| orphan (may29 list) | status | evidence |
|---|---|---|
| `eligibility_trace.py` | DEAD-BUT-PRESENT | zero importers tree-wide |
| `singlet_dynamics.py` | DEAD-BUT-PRESENT | zero importers |
| `calcium_system.py` | dead on core path, **live in probes** | core import commented out `model6_core.py:61`; probes re-attached (`sweep/calcium_pde_reference_probe.py:44,53`) |
| `implicit_diffusion.py` | imported, transitively dead | sole importer `calcium_system.py:28` |
| debug subsystem | DEAD-BUT-PRESENT | `debug_entanglement.py`, `debug_photons.py`, zero importers |
| **`em_coupling_module.py` (NEW, 6th)** | superseded, still imported | `vibrational_cascade_module.py:45` declares itself a "drop-in replacement"; the old class is imported at `model6_core.py:84` and **never instantiated**, keeping ~15 uncited dead constants alive incl. a duplicate `FIELD_THRESHOLD_KT=20.0` (`em_coupling_module.py:453`) |

**Dead-parameter count: ~151** declared-but-never-read dataclass fields (vs ~120 at may29 —
**grown, not shrunk**). Largest clusters in `model6_parameters.py`: `QuantumParameters` 17,
`PosnerParameters` 15, `MultiSynapseParameters` 12, `PNCParameters` 8, `MetabolicUVParameters` 8.
Method: field-level parse + tree-wide reader search including string-literal access
(`getattr`). Stated limits: under-counts on cross-class name collisions; under-counts fields
read only inside `__main__` self-tests.

---

## F. EXPERIMENT RIG STATE

| item | status | evidence |
|---|---|---|
| `run_spatial_discovery.py` end-to-end | PARTIAL | components ran 40 s today (`sweep/eta_in_live_trial.py`); shipped `run_experiment(n_trials=25, n_features=40)` **UNVERIFIED** |
| input engine gap — learning drivers | **CLOSED 2026-07-18** | `sweep/run_spatial_discovery.py:442,478`; `src/models/Model_6/sweep/run_place_field_learning.py:247` |
| input engine gap — T1′ probe family | **OPEN, deliberately** | `coherence_fragmentation_probe.py`, `dstar_eff_replay.py`, `measure_dstar0.py`, `order_power_probe.py`, `coherence_radius_probe.py` — 0 glutamate refs each. Wiring them changes `d*(0)=3.4521` and every break time ⇒ re-validates a closed result. |
| drive × damping sweep harness | PARTIAL | `sweep_runner.py` + `quantum_dimensions.py`: `q1_d_modes` (D, `:48`), `q1_phi_dissipation` (φ, `:60`), `q1_chi_redistribution` (`:72`), `q1_f_coherent_base` (`:36`), `q1_n_tryptophan` (`:24`); stimulus `stim_ca_amplitude`, `stim_theta_cycles`, `stim_n_traversals` (`:208+`) |
| **Q as a sweep dimension** | **ABSENT** | zero hits for Q/quality_factor in `quantum_dimensions.py` |

**To complete a drive × damping sweep:** add a `Q` dimension to `NETWORK_DIMENSIONS` and an
apply-branch in `sweep_runner.py` beside the `q1_phi_dissipation` handler (`~:62`); add a
drive-amplitude dimension keyed to the plateau (`stim_ca_amplitude` is the calcium knob, not
the drive); give `sweep_runner` an η/`r` readout — it currently has no condensation
observable.

---

## G. CHANGES NOT COVERED ABOVE

41 commits since 2026-06-02; **14 on 2026-07-18 alone**. Load-bearing:
`ed43838` input engine finished (η 0 → 0.2376, DDSC fires first time) · uncommitted
plateau→voltage term in `model6_core.py` (driving-force-scaled, not summation) ·
`92c623f` measurement latch re-armed, control gate freed, `coupling_weights` passed during
trials · `57c2068` D17 cross-trial reading RETRACTED · `acc0d32` T1′ population confound
measured inert · `9580f82` T1′ power revised 10/10 → 37/40 · `268a64d` η does not clear in a
live trial (`r` = 0.077 vs 1.0).

---

## THE DRIFT LIST (consolidated)

1. **2π convention** — `vibrational_cascade_module.py:315` uses `ℏ·f` on a linear frequency
   vs the repo's own stated convention at `model6_parameters.py:43`. Duplicated `:755-756`.
2. **Hill function** — claimed `multi_synapse_network.py:1332-1334`, absent `:1381-1392`.
3. **"No fitted parameters!"** — claimed `:1238-1242`, contradicted by `:717` and `:1245`.
4. **`collapse_factor` 30%** — claimed `:1423-1425`, never read; actual effect `:1466`.
5. **"linear in eta is correct"** — comment `:243-247` vs geometric mean at `:314`.
6. **"one-time effect"** — `:1416`; now runs once per reward episode after the 2026-07-18
   latch change.
7. **Backbone D/φ/χ "not hand-tuned"** — `model6_parameters.py:759` vs `_update_backbone_field`
   reading only ω₀ and Q.
8. **φ, χ vs Zhang 2019** — code 10 GHz / 0.05 GHz vs cited 6 GHz / 0.07 GHz.
9. **f_coherent** — comment `~0.08` (`em_tryptophan_module.py:692`) vs live `0.10`.
10. **PNC stoichiometry** — `n_ion_pairs_per_pnc = 3  # Ca(HPO₄)₃⁴⁻` (one Ca) vs arithmetic
    asserting "3 Ca per PNC" (`ca_triphosphate_complex.py:127` vs `:211,216-217`).
11. **`phosphate_total` goes stale** — recomputed only in `add_phosphate_from_atp`
    (`atp_system.py:428`); dimer consumption decrements `phosphate_structural` without
    updating it (`model6_core.py:450-452`). **J-coupling (`atp_system.py:485`) reads a
    phosphate field that ignores dimer consumption.**
12. **ATP↔Pi not mass-conserving** — hydrolysis credits Pi (`atp_system.py:130`); recovery
    regenerates ATP (`:163,169-170`) without debiting any phosphate pool.
13. **`K_CaHPO4`** — 588 live vs 470 declared.
14. **`photon_flux_baseline`** — 20.0 declared vs ≈1.5 derived in its own comment.
15. **Editing artifacts** — `# (you modified this earlier)`, `# ← YOUR NEW CODE STARTS HERE`
    (`ca_triphosphate_complex.py:434,439`).
16. **`step_with_coordination` forms no cross-synapse bonds** — `:1274` omits
    `coupling_weights` ⇒ `:279-280` early-return. Same at
    `src/models/Model_6/sweep/run_place_field_learning.py:134`. (The 2026-07-18 fix covered
    `sweep/run_spatial_discovery.py` only — **a gap in that fix**.)

---

## WHAT SURVIVES, AND WHAT DOES NOT

**Survives.** The entanglement/partition layer: Werner 0.5 is a theorem, not a tuned cutoff;
η is exactly the threshold form with no fitted curve; commitment is a genuine CaMKII
integrator with a real DDSC delay; the backbone pump uses the correct `h·f` convention; the
P_S thermal floor, Jahr-Stevens, and Naraghi-Neher are all literature forms. The
2026-07-18 measurements (T1′-6 channel separation, L·ETA-2 ignition, L·ETA-3 live-trial
null) are grounded, pre-registered where scoreable, and logged with their limits.

**Does not survive.** The EM / vibrational-cascade layer: a factor-of-2π error, an
unsweepable calibration anchor that makes its headline threshold result an arithmetic
identity, and two constants whose cited source gives different values. The chemistry layer
is mass-action in form but scaled by self-declared free parameters. Parameter hygiene has
regressed (~151 dead fields, six orphan modules, none removed).

**The class of defect that recurs.** Prose asserting mechanisms the code does not implement
— a Hill function, a 30% collapse, "no fitted parameters", a quantum barrier modulation
(`spine_plasticity_module`, measured inert 2026-07-18), a 2% ATP replenish. This is the same
failure the citation audit found earlier the same day, and the same failure the T1′ §6
rewrite corrected. It is the program's characteristic error and it is not yet under control.

---

## OPEN ITEMS, RANKED

1. Fix the 2π error (`vibrational_cascade_module.py:315`, `:755-756`) — a real physics bug
   with a known correct form 200 lines away in `model6_parameters.py:46`.
2. Decide the fate of `kT_ref`/`r_at_E_ref`. Either derive them or state plainly that the
   per-synapse threshold result is calibrated and not evidential.
3. Delete or correct the three false docstrings (Hill, 30% collapse, "no fitted
   parameters"). Cheapest fix, highest credibility return.
4. Resolve φ/χ against Zhang 2019 — use the cited values or stop citing the paper.
5. Verify or remove the `Wang et al. 2024 Nature Commun 15:1234` locator before any external
   citation.
6. Fix `phosphate_total` staleness — J-coupling currently reads a field that ignores dimer
   consumption.
7. Pass `coupling_weights` in `step_with_coordination` and `run_place_field_learning`.
8. Retire the six orphan modules, or state why they are kept.
