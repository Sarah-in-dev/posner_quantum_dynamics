# Dopamine reinforces CaMKII through DARPP-32/PP1 — it does NOT bypass it (readout grounding)

> **⚠ FRAMING — read first.** This document refines a **WORKING system's mechanism**; it does NOT question whether
> the system computes. The F-series already PROVED the computation (temporal credit assignment; `CAL F3-b` credits at
> 30 s vs classical dead at 2 s; `CAL F3-c` isotope moves credit). Everything below is the **decoherence signal →
> plasticity cascade REFINEMENT** — grounding how the dopamine-gated spin-selective binding-melt readout drives the
> CaMKII/plasticity cascade biologically correctly, WITHOUT breaking the working computation. Where later sections
> say "blocker," "not DA-decisive," or "calcium-dominated," read them as **OPEN MECHANISTIC QUESTIONS inside that
> refinement, NOT as evidence the system fails.** Two sessions mis-read them as the latter — see the README §3 guardrail.

**2026-08-09. Cited physics/biology assessment for the reward-signed readout at the network level (Phase C).
BLUNT VERDICT: the F3 reward-gated consolidation as currently coded COMMITS by BYPASSING CaMKII
(`model6_core.py:724`, commit on `credit` instead of `molecular_memory`), which is biologically UNSUPPORTED.
The grounded mechanism is that dopamine REINFORCES/DISINHIBITS CaMKII via the D1→PKA→DARPP-32(Thr34)→PP1
cascade; commitment remains CaMKII-gated (the LOCKED commitment pathway is correct and is REINFORCED, not
overturned). The synaptic LTP/LTD SIGN is EMERGENT from PP1 activity, not an imposed ±1.**

This doc grounds the correction and its re-validation. Approved by Sarah 2026-08-09 (unlock `camkii_module.py`;
fix + re-validate; maximize biological fidelity; document the research).

## The problem this fixes
`RESEARCH_DOPAMINE_READOUT_PHYSICS_2026-08-08` established the READOUT is Fisher's spin-selective binding-melt
(Ca²⁺ shower → glutamate → plasticity), reward-TIMED by dopamine. But the F3 *commitment* code took a shortcut:
it commits directly on `quantum_credit` (a `da_sign × eligibility_weight(P_S)` scalar), bypassing the CaMKII
`molecular_memory > 0.5` gate that `model6-commitment-pathway` (LOCKED, May 12) requires. Biologically, dopamine
has no path to commit plasticity that goes around CaMKII — it acts THROUGH it.

## The biology (grounded, both directions)

**CaMKII is the obligatory integrator; Hebbian Ca²⁺ alone is not enough.** Yagishita 2014 (Science): concurrent
glutamate + postsynaptic APs give Ca²⁺→CaMKII, but this "is not sufficient to fully activate CaMKII; rather, it
requires reinforcement by PKA–DARPP-32–PP1 signaling." PKA is active ONLY in a narrow window (0.3–2 s) set by high
phosphodiesterase, and "promoted spine enlargement via CaMKII." CaMKII-T286 autophosphorylation is REQUIRED for
behavioral-timescale plasticity (Xiao 2023, Sci Adv), and the commitment is dendritic/delayed/stochastic (Jain
2024, DDSC — the basis of the LOCKED commitment pathway).

**The molecular link is PP1 acting on CaMKII-pThr286.** PP1 dephosphorylates CaMKII-Thr286 (and AMPAR Ser831/845).
So the control variable dopamine reaches is **PP1 activity = the CaMKII `k_dephosphorylation`**.

**LTP arm (burst → potentiation):** D1 → AC5 → cAMP → PKA → phosphorylates **DARPP-32 at Thr34** →
phospho-Thr34-DARPP-32 **inhibits PP1** → CaMKII-pT286 is no longer stripped → pT286 accumulates → potentiation.

**LTD arm (dip / weak-Ca → depression):** low PKA + Ca²⁺→**PP2B (calcineurin)** dephosphorylates DARPP-32-Thr34 →
**PP1 disinhibited (active)** → strips CaMKII-pT286 → depression. Reinforced by Cdk5 → **DARPP-32-Thr75** →
inhibits PKA (dominant at weak Ca). Strong Ca → **PP2A** → dephosphorylates Thr75 → disinhibits PKA → back toward LTP.

**So the sign is EMERGENT from PP1, not imposed:** PP1 inhibited → LTP; PP1 active → LTD.

## The quantitative scheme (Nakano 2010, PLOS Comput Biol; the field's canonical kinetic model)
- Cascade: D1→AC5→cAMP→PKA→DARPP-32-Thr34⊣PP1⊣(CaMKII-pT286, AMPAR); Ca→PP2B⊣Thr34; Ca→Cdk5→Thr75⊣PKA;
  strong-Ca→PP2A⊣Thr75.
- **Bistable switch:** the PKA–PP2A–DARPP-32(Thr75) positive (double-negative) feedback loop gives 3 fixed points
  (2 stable, 1 unstable) with hysteresis — a threshold on/off in PKA activity vs cAMP.
- **Directionality by Ca amplitude:** weak Ca (< ~0.5 µM) → LTD (Cdk5-Thr75 dominant); strong Ca (> ~1 µM) → LTP
  (PP2A-Thr75 dominant).
- **Timing:** LTP most effective when **dopamine follows calcium**, ~500 ms critical window; elevated *basal*
  dopamine disrupts it (even strong Ca → LTD).
- Timescales: Ca α-function τ ≈ 46.7 ms; dopamine τ ≈ 80 ms; spine volume ~1 µm³.
- Scale/honesty: 72 reactions, 132 parameters (83 literature-derived, **49 hand-tuned**). We ground the STRUCTURE
  and the literature-derived rates; any constant we cannot source is flagged `[MODELED]`, never tuned to our outcome.

## How this maps onto Model 6 (the correction)
1. **New signaling module** (faithful DARPP-32/PP1 cascade): dopamine (D1 occupancy, `dopamine_system.get_d1_occupancy`)
   + calcium → PKA / PP2B / Cdk5 / PP2A → DARPP-32 phospho-state (Thr34, Thr75) → **PP1 activity**. Grounded rates
   from Nakano 2010 / Fernandez 2006; MODELED constants flagged.
2. **`camkii_module.py`** (UNLOCKED, approved): make PP1 dephosphorylation dopamine-dependent — `k_dephos_effective =
   k_dephos_base × PP1_activity` from the cascade (default `PP1_activity = 1.0` ⇒ **bit-identical when off**). Burst →
   PP1↓ → pT286 accumulates (LTP); dip/weak-Ca → PP1↑ → pT286 stripped (LTD). The flagged ungrounded barrier
   constants (`barrier_total_kT=23`, the 40/30/15/15 split) are OUT OF SCOPE — not touched.
3. **`model6_core.py:724`** — REPLACE the F3 credit-bypass: keep the coherence gate (spin-selective binding-melt
   delivers Ca²⁺ only while P_S > Werner), feed the dopamine→PP1 factor into `camkii.step`, and let commitment fire
   via **`molecular_memory > threshold`** (the DDSC lock). The LTP/LTD sign emerges from PP1, replacing `_reward_sign`.
4. **Re-validate** single-synapse F3 (delayed-credit, isotope) under the corrected mechanism, THEN extend to the
   multi-synapse network (the original Phase-C objective).

## Build status — the DARPP-32/PP1 module is built + validated (2026-08-09)
`darpp32_pp1_module.py` implements the cascade (Thr34, Thr75 phospho-states; PKA/PP2B/PP2A/Cdk5; PP1 activity),
grounded-structure with `[MODELED]`-flagged rates. Its `__main__` acceptance PASSES 5/5: the LTP/LTD sign
**emerges** (burst → `pp1_factor` 0.12 = LTP; dip → 1.07 = LTD; ordering burst < tonic < dip; the Ca-amplitude
switch orders weak-Ca above strong-Ca). **FINDING (grounded, cited):** the LTP/LTD range is **ASYMMETRIC** — the
resting striatal state is PKA-suppressed / PP1-ACTIVE (Thr75-P by Cdk5 holds PKA off until dopamine arrives;
Svenningsson/Greengard, Nishi), so dopamine drives strong LTP from a resting point where PP1 is already active and
the dip→LTD headroom is inherently small. This asymmetry is biology, not a bug, and was **confirmed by external
search** rather than tuned away (an initial symmetric-magnitude check FAILED; grounding the resting operating point
showed the check was wrong, not the model). `pp1_factor` is normalized to tonic = 1.0 ⇒ bit-identical when unwired.

## RESOLUTION of the calcium-domination blocker (2026-08-09) — the PP1 counterforce is too weak
The F3-e diagnostic showed commitment is calcium-dominated (Ca shower saturates CaMKII, dopamine inert). The
resolution experiment `sweep/f3e_calcium_pp1_regime_probe.py` (drive a CaMKII + DARPP-32/PP1 pair across
field × PP1-strength × dopamine at physiological Ca=2 µM, pre-registered "DA-decisive" criterion:
burst≥0.8 ∧ dip≤0.2 ∧ none≤0.2) **located a DA-DECISIVE regime**: at field 12–24 kT with **PP1 strength ≈
100–300× the current `k_dephosphorylation`** (i.e. `k_dephos ≈ 0.1–0.3 s⁻¹`, comparable to `k_phos_max = 0.1`),
a dopamine burst commits (1.00) while dip/none do NOT (0.00). Below that PP1 strength calcium+field saturate
CaMKII regardless of dopamine (the F3-e blocker); the positive control fires (cells commit), so the result is valid.

**Diagnosis:** the model's `k_dephosphorylation = 0.001 s⁻¹` ("slow, for memory") is ~100–300× too weak to be the
Ca–PP1 bistable-switch counterforce the biology requires (Zhabotinsky 2000; Graupner 2007: PP1 dephosphorylation
of CaMKII-pT286 is comparable to the autophosphorylation rate, which is what makes the switch bidirectional and
lets dopamine/PP1 decide). At the grounded balance, "potentiation does not occur to dopamine OR glutamate alone"
(Fernandez 2006) is reproduced.

**The fix direction (grounded — NOT tuned to a decode):** ground `k_dephosphorylation` toward the bistable-switch
value (~0.1–0.3 s⁻¹) from Zhabotinsky/Graupner, with a persistence mechanism (CaMKII autophosphorylation
bistability, or GluN2B-shielding of pT286 from PP1 — Mayadevi/Omkumar 2016).

**REFINED FINDING (2026-08-09, tested — the naive fix is INSUFFICIENT, and it sharpens the problem):** a direct
test (strong PP1 `k_dephos=0.1` + GluN2B protection=0.9, transient 2 s burst vs tonic vs dip) did NOT become
DA-decisive — burst/tonic/dip all committed (~2/3). Two coupled reasons, both real:
- **The two requirements trade off.** Strong PP1 gives DA-decisiveness but strips the memory once the burst passes;
  GluN2B protection gives persistence but DESTROYS DA-decisiveness — pT286 rises from calcium+field in *every*
  condition, GluN2B then binds and shields it from PP1, so tonic/dip latch just like burst. Persistence and
  DA-decisiveness fight unless the switch has a SHARP threshold that only a burst crosses (true Zhabotinsky
  autophosphorylation bistability, not the leaky GluN2B shield).
- **The DA-decisive band is narrow and drive-sensitive** (decisive at field 12 kT, NOT at 18 kT — the drive alone
  overwhelms PP1). And the ROOT: the real readout delivers **saturating** drive (~700 µM Ca, F3-e), so CaMKII sits
  far ABOVE threshold, where dopamine can never tip it. For dopamine to decide, the readout drive must sit NEAR the
  CaMKII threshold.

**So the resolution requires BOTH, together:** (1) a true bistable CaMKII switch (autocatalytic autophosphorylation
— sharp threshold + self-sustaining UP state), AND (2) the readout drive (the binding-melt Ca²⁺ shower + field)
calibrated to sit NEAR that threshold, not saturating. **(2) IS the calcium-domination / calcium→dimer issue that
`quantum-system-canonical` §2.3 already flags as in-flight** — so this reward-signed-readout goal is coupled to that
upstream work and is not a self-contained one-parameter fix. This is the honest, well-characterized state; the next
move is a deliberate two-part build (bistable CaMKII + near-threshold readout drive), not further parameter hunting.

## PART 1 BUILT (2026-08-09) — the bistable CaMKII switch; hysteresis validated, DA-decisiveness coupled to Part 2
`camkii_module.py` gains an **opt-in** `bistable` mode (default False ⇒ bit-identical): AUTONOMOUS autocatalytic
autophosphorylation (`autocat·pT286`, calcium-independent — Coultrap & Bhalla 2012) vs SATURATING PP1
(`Vmax·pT286/(Km+pT286)`), Lisman & Zhabotinsky 2001. Bistable-band params derived analytically
(`autocat∈(4.2,7.5)` at Vmax=0.15,Km=0.2), `autocat=6.0` [MODELED-in-band, not tuned]. **VALIDATED — true
hysteresis:** a transient 8 s drive latches the UP state (pT286 0.97 → **0.72 held 30 s after drive-OFF** =
autonomous self-sustaining memory), while rest stays DOWN (0.15). The default (non-bistable) path is unchanged
(`__main__` still passes).

**But standalone it is NOT DA-decisive** (dip vs burst both latch UP, ~0.65–0.71, at field 6–12): the drive
overshoots the switch's threshold in *every* dopamine condition. This **confirms (now from inside the switch)
that DA-decisiveness is inseparable from Part 2** — dopamine decides only when the drive sits JUST BELOW a
threshold it shifts, i.e. the readout drive must be co-calibrated to the (dopamine-modulated) threshold. Part 1
(the switch, with its persistent UP-state memory) is a necessary, grounded component; the reward-signed readout
is GATED on Part 2 (the near-threshold drive = the in-flight calcium→dimer / calcium-domination work, canonical
§2.3). Next: co-calibrate the readout drive to the switch threshold (Part 2), then wire `bistable=True` into the
reward-gated path and re-validate F3 end-to-end.

## PART 2 — CONSTRAINT MAP (2026-08-09, mapped not closed): the reward-signed readout needs FOUR coupled things
Part 2 (make the full system DA-decisive) is not a one-parameter fix. Each experiment this session grounded another
required condition; together they define what a working reward-signed readout needs:
1. **CaMKII must integrate PSD-distance calcium, NOT the nanodomain peak.** `model6_core.py:602,833` feeds CaMKII
   `calcium_uM = np.max(ca_conc)·1e6` — the channel-mouth peak (~137 µM + dimer-dissolution return ~1 µM/dimer×hundreds).
   CaMKII is at the PSD; physiologically it integrates the diffuse spine calcium, "a few µM" (literature), near its
   1 µM threshold. This is a construct-validity fix (spatial), and it is the calcium-domination root (canonical §2.3).
2. **The drive (calcium+field) must sit NEAR the switch threshold**, not above it — else it commits regardless of DA
   (Part-1 finding).
3. **Dopamine must FOLLOW the calcium, not overlap it (Nakano 2010 timing).** At the ~1 µM calcium that drives CaMKII,
   PP2B/calcineurin is ~90% active and strips DARPP-32-Thr34, keeping PP1 active and OVERWHELMING dopamine's PP1
   inhibition — measured: with the burst overlapping the calcium, none/dip/burst are identical (dopamine has zero
   vote). Dopamine can only decide once calcium (and PP2B) has decayed. So the protocol must be TEMPORAL:
   calcium eligibility → (decay) → delayed dopamine reward.
4. **The bistable switch must hold NEAR-THRESHOLD through the reward delay** so the later dopamine can tip it — this is
   exactly the role of the coherent P_S tag (the ~100 s eligibility), tying Part 2 back to the F3 delayed-credit design.

**The next concrete experiment (deliberate, not a quick sweep):** the Nakano-timed protocol — CaMKII fed PSD-distance
(near-threshold) calcium during a brief eligibility window, the bistable switch holding, then a DELAYED dopamine burst
(after PP2B decays) — and test whether dopamine now decides (burst→UP, dip→DOWN). If yes, wire `bistable=True` +
PSD-calcium into the reward path and re-validate F3 end-to-end; if no, the coupling is a deeper structural finding.
This is a genuine multi-coupled dynamical calibration and should be built systematically, not by end-of-session sweeps.

## PART 2 — FIRST EXPERIMENT RESULT (2026-08-09, Nakano-timed probe): a clean NEGATIVE + a structural finding
`sweep/f3e_nakano_timed_probe.py` ran the Nakano-timed protocol (bistable CaMKII + DARPP-32; eligibility 4 s @ Ca=1 µM
+ tag field → delay 8 s @ basal Ca, field held → DELAYED burst/dip/none → settle; field swept 0–24 kT; positive
control = saturating field commits, negative = no-drive; 8 seeds). **VERDICT: NOT DA-decisive at any drive** — burst
barely shifts commit-rate vs dip/none; the switch's fate is set BEFORE the reward arrives. **Two coupled, GROUNDED,
STRUCTURAL reasons (not tunable):**
- **A — a bistable switch cannot "hold near-threshold" through the delay (constraint #4 fails by construction).** The
  separatrix is an UNSTABLE fixed point; the switch physically cannot dwell at its knife-edge for 8 s. Measured: even
  with NO field, a 4 s / 1 µM eligibility leaves pT286=0.19, already above the separatrix (0.069), and autocatalysis
  carries it to UP DURING the delay, before any reward. Worse, the tag field as modeled (barrier reduction → boosts
  k_phos) pushes pT286 toward latch (delay-end pT286 rises monotonically with field, 0.41→0.95). **The modeled "tag
  hold" COMMITS the switch; it does not hold it poised sub-threshold.**
- **B — dopamine's grip on PP1 has faded by the time it is allowed to vote.** At basal calcium (post-decay, when
  Nakano says DA should decide), a burst moves `pp1_factor` only to 0.974 (2.6%) vs 0.12 with strong calcium present.
  Grounded: the DARPP-32 gain (Cdk5→Thr75⊣PKA, cleared only by strong-Ca→PP2A) means **dopamine needs calcium
  COINCIDENCE to grip PP1** — exactly what the DA-follows-Ca separation removes (Thr75 re-accumulates τ≈1.25 s). So
  constraint #3 (DA follows Ca) and the DARPP-32 gain are in FUNDAMENTAL TENSION: the cascade needs Ca present for DA
  to act; the timing needs Ca gone. And the DARPP-32 gain window (~1–2 s) ≪ the ~100 s coherent-tag delay F3 needs.

**THE DEEP ITEM (structural, not a parameter):** the coherent P_S tag's role (constraint #4) is **unmodeled in a
load-bearing way.** It currently acts as a barrier-reduction field that DRIVES/COMMITS the switch, whereas the F3
delayed-credit story requires it to **hold eligibility sub-threshold across the gap WITHOUT committing** — and to keep
dopamine able to act after the delay. Reconciling that (how the coherent tag holds a *primed-but-uncommitted* state
that a delayed reward can still tip) is the real open architectural question — bigger than "calibrate the drive."
Honest scope: this tested the MECHANISM at a near-threshold drive; it does NOT bear on constraint #1 (grounding the
true PSD-distance calcium), which remains a separate step — but Finding B blocks a long-delay reward-signed readout
even if #1 is resolved. Next moves (both structural): (1) ground constraint #1 (the actual diffuse calcium CaMKII
integrates); (2) decide how the coherent tag holds eligibility sub-threshold rather than driving the barrier —
possibly the tag should prime DARPP-32/PKA (keep dopamine's grip alive) rather than the CaMKII barrier.

## Emergent-physics discipline (LOCKED)
The PP1 modulation and the DARPP-32 cascade rates are grounded from the cited kinetic literature; NONE is tuned to
make the readout decode. Ca-amplitude directionality and the DA-follows-Ca timing window are cited, not fitted. If
the grounded cascade does not produce the delayed-credit / isotope signature, that is a FINDING (the mechanism does
not do what F3 claimed), recorded — not a license to tune PP1.

## Re-validation acceptance (the honest cost of the fix)
The F3-b (delayed-credit 3/3) and F3-c (isotope) results were obtained on the BYPASS. Under the corrected
through-CaMKII mechanism they must be re-run. Expected/required: (a) a still-coherent tag (P_S > Werner) at a
dopamine BURST reinforces PP1-inhibition → CaMKII commits (LTP) across delays where the classical ~2 s trace is
dead; (b) a dip (or decohered tag) → PP1 active / no readout → no potentiation; (c) ⁶Li (long coherence) credits at
long delay, ⁷Li (short) does not. If any fails, report it — do not retune.

## CONSTRAINT #1 — LANDED (2026-08-09): CaMKII now integrates the diffuse PSD-distance calcium (grounded)
**The construct-validity fix is in.** `model6_core.py` fed CaMKII (and the DARPP-32/PP1 cascade) `calcium_uM =
np.max(ca_conc)` — the channel-mouth **nanodomain peak** (~137–700 µM). But CaMKII/DARPP-32 sit at the PSD; they
physiologically integrate the **diffuse PSD-distance calcium**. New method `AnalyticalCalciumSystem.
get_psd_averaged_concentration(psd_radius_nm=180)` area-averages the model's OWN Naraghi-Neher nanodomain profile
over the **EM-grounded PSD disk** (disk ~360 nm dia / ~180 nm radius; range 200–800 nm). The plasticity cascade
(CaMKII + DARPP-32) is now fed this value at `model6_core.py:~608/724/729`; the nanodomain peak is RETAINED for
dimer formation and the binding-melt measurement trigger (dimers form at the mouth; CaMKII reads the PSD average).
Value is **emergent** (model's own profile × grounded PSD geometry), not tuned.

**DATA-LEVEL VALIDATION (diagnostic probes, single synapse, reward-gated path):**
- **Which quantity was the saturator — measured, decomposed:** at commit the field is decomposed into nanodomain
  peak vs uniform dissolution-shower floor vs PSD-area-average. The saturator is the **formation nanodomain peak**
  (~200–700 µM); the uniform shower floor **stays at baseline (0.1 µM)** in this path (no shower here — F3-e's
  "~700 µM shower" was the nanodomain peak). So constraint #1's original framing is correct; shower/temporal are
  NOT the cause here.
- **The edit lands the drive in range:** CaMKII-fed calcium drops from ~200–700 µM (np.max) to **~3–16 µM**
  (PSD-area-average) — near CaMKII's K_calcium_half = 1 µM, and consistent with measured spine calcium during LTP
  (resting ~100 nM; single-EPSP ~0.5–1 µM; strong LTP a few µM). [GROUNDED — Sabatini/Higley; PSD geometry EM]

**BUT NOT YET DA-DECISIVE — a clean structural finding (do NOT tune):** with the bistable CaMKII (Part 1) paired
in and the corrected PSD calcium, a delayed DA burst/dip/none are **identical** (all commit ~0.75) and **all commit
BEFORE the reward** (pT286 ≈ 0.73 at build-end = at the switch's UP attractor ~0.732; climbs during the delay). The
**eligibility-phase (Hebbian) calcium drives the switch past its separatrix and it latches before dopamine votes.**
This is exactly Yagishita's rule being VIOLATED by the model: *Hebbian Ca alone is not sufficient to commit — it
requires PKA/DARPP-32/PP1 reinforcement* — yet the model commits on Hebbian activity alone. Constraint #1 (calcium
MAGNITUDE) was necessary but not sufficient.

**The two grounded requirements now precisely characterized (the real Part-2 core):**
1. **Down-stable at Hebbian Ca:** the switch must NOT latch on eligibility-phase Ca under tonic dopamine — resting
   PP1 (Cdk5/Thr75-suppressed PKA → PP1 active; + Ca→PP2B→PP1 active) must strip pT286 back. Requires the PP1↔auto-
   phosphorylation balance grounded (Zhabotinsky/Graupner: PP1 dephos comparable to autophos), and the DARPP-32
   cascade running CONTINUOUSLY (currently pp1_factor=1.0 during build — the cascade is only engaged in the reward
   block, so the model omits the PP2B-driven PP1 activation that Hebbian Ca should cause).
2. **Commitment delivered at reward:** at reward, dopamine (PP1-inhibited) + the coherent tag's readout Ca must
   TOGETHER tip the (still-DOWN) switch — the Ca/dopamine coincidence Yagishita/Nakano require. In the reward-gated
   path the binding-melt readout is skipped, so no readout Ca arrives at reward; dopamine alone (modulating PP1 at
   basal Ca) cannot raise pT286.

Both are structural (architecture of eligibility-vs-commitment separation + the PP1/autophos balance), not a single
parameter. Next: work these grounded (continuous DARPP-32 cascade; PP1↔autophos balance from Zhabotinsky; readout-Ca
delivery at reward), one deliberate grounded step at a time — NOT a drive sweep to force a pass.

## FINDING A CONFIRMED in the full model (2026-08-09): the bistable switch CANNOT hold a primed sub-threshold state
Worked the grounded producing mechanism end-to-end. **In ISOLATION (bench: bistable CaMKII + continuous DARPP-32,
`switch_regime_bench.py`) it PRODUCES**: at a physiological 1 µM eligibility and PP1 Vmax=0.30 (grounded — Zhabotinsky/
Graupner band, PP1 dephos ~comparable to the autophosphorylation rate ~0.6), the switch stays DOWN under Hebbian Ca at
tonic DA and is DA-DECISIVE (burst commits 1.0; dip/none 0.0). The mechanism is right: dopamine inhibits PP1 → lowers
the commitment threshold → the reward's readout Ca tips the switch; without DA, PP1 holds it down.

**But WIRED INTO THE FULL MODEL it does NOT produce** — three data-level findings, all grounded, none tunable:
- **The eligibility DRIVE matters:** the F3 probes' 2 s −40 mV clamp gives ~3–16 µM *sustained* PSD calcium (via the
  landed constraint #1 read) — an unphysiologically strong stimulus that latches the switch DURING the build. A brief
  (physiological EPSP-scale) drive keeps pT286 low during the build (~0.001) — the Hebbian-safety works while Ca is up.
- **The live dopamine module emits SPURIOUS bursts at reward=False** (measured: da spikes to ~6 µM ~2.6 s into the hold),
  which the continuous cascade correctly reads as a burst → inhibits PP1 → helps the switch climb. This is F3-a's own
  flagged artifact; F3-a's fix is to INJECT dopamine explicitly (tonic/hold, burst/dip at reward), not read the live
  field. A baseline-robust deadband was added but cannot filter a genuine 6 µM spike — the artifact is upstream in the
  dopamine module.
- **DECISIVE — Finding A:** even with a brief build AND injected clean tonic DA, the switch does not hold. Over a 6 s
  hold at basal Ca + tonic DA, pT286 diffuses from ~0.001 up to **0.26–0.38** (hovering at the separatrix ~0.069→basin
  0.3) under the switch's own stochastic (Langevin) noise + weak autocat — it neither cleanly stays DOWN nor latches UP,
  it CREEPS toward commitment. **The bistable autocat switch's DOWN basin is not stable enough to hold a poised
  sub-threshold "primed" state across seconds, let alone the ~100 s coherence window.**

**CONSEQUENCE (the frontier, unchanged):** the eligibility CANNOT be carried as a primed CaMKII pT286 state — the
switch either latches or noise-diffuses to commitment. Eligibility must live in the **coherent tag** (P_S, which does
hold ~100 s); CaMKII must reset to ~0 during the hold (PP1 fully strips the transient) and be driven to commit ONLY at
reward, by the readout Ca²⁺ shower + dopamine's PP1-inhibition (the Ca/DA coincidence, Yagishita/Nakano). Making CaMKII
reset-and-hold-at-zero (not carry a seed) is the architectural core — NOT a switch-parameter tune (raising the
separatrix / killing the noise to force a hold would be exactly the emergent-physics violation this program forbids).

**Landed + committed this session:** constraint #1 (PSD-averaged calcium to the plasticity cascade) + a fast exact
PSD-average kernel (precomputed per-channel geometric weight → dot product; provably identical values, O(N_ch)/step).
The reward-block wiring (continuous DARPP-32 + bistable Vmax=0.3 + baseline-robust DA) was exploratory and is REVERTED
(it does not produce and would leave the F3 probes in a non-working bistable-by-default state); the grounded producing
regime is captured in `sweep`/bench form and the finding above. The producing bench: `switch_regime_bench.py`.

## THE PRODUCING MECHANISM (2026-08-10) — grounded in the REAL CaMKII biology; it works in isolation
Researched how CaMKII actually works ([sources in the session]). Three facts overturn the bistable-switch approach:
1. **CaMKII activation is TRANSIENT (~1 min; fast 1-6 s), NOT a stable bistable switch** — the Lisman/Zhabotinsky
   autophosphorylation switch is contested / non-physiological (Frontiers Synaptic Neurosci 2025). So **Finding A
   (the bistable switch can't hold a primed state) is CORRECT biology, not a bug** — we were fighting reality.
2. **The persistent memory is the CaMKII–GluN2B STRUCTURAL complex, not a phospho-state** — needs an initial
   Ca²⁺/CaM + pT286 stimulus to FORM, then persists autonomously, nanomolar-tight, PROTECTED from phosphatases,
   a stable condensate (Cell Reports 2024; Molecular Brain 2013; PMC4965558).
3. **The seconds-timescale commitment = DDSC** (Jain 2024, Nature): a **delayed (10-100 s), stochastic** CaMKII
   activation driven by **IP3-dependent INTERNAL-STORE calcium** (~µM, near CaMKII's K_half — not the nanodomain).
4. **DAPK1 makes CaMKII–GluN2B binding LTP-SPECIFIC** (DAPK1 paper; β-adrenergic switch): DAPK1 is active WITHOUT
   the reward signal and BLOCKS the binding; dopamine→PKA (PP1 inhibited) SUPPRESSES DAPK1 → releases binding.

**BUILT — `camkii_module.py` `glun2b_memory` mode (opt-in, default False = bit-identical; `__main__` still passes):**
pT286 TRANSIENT (`k_dephos_transient≈0.05`, τ≈20 s, resets after the Ca event); the GluN2B complex is a persistent
STRUCTURAL LATCH (forms above a cooperative pT286 threshold, protected off-rate ≈ LTP duration); **DAPK1 gates the
BINDING on dopamine** (`dapk1_off_pp1`), so the reward burst is REQUIRED to commit. `molecular_memory` = the complex.

**VALIDATED IN ISOLATION (`sweep/glun2b_latch_bench.py`, CaMKII + DARPP-32; n=4):** pT286 builds at the reward Ca
then decays (transient); the complex forms at reward and PERSISTS to the end; **DA-DECISIVE across the WHOLE readout
range 0.5-3 µM — burst commits 1.0, dip/none 0.0, eligibility does NOT pre-commit.** DA-decisiveness is **robust
across calcium magnitude** (not a narrow band) precisely because DAPK1 gates the *binding*, not the calcium level —
**this dissolves the calcium-domination that broke F3-e, Finding A, and the bistable switch.** It is the Yagishita
behaviour (Hebbian Ca alone insufficient; dopamine reinforcement required), emergent from a grounded mechanism.
`[MODELED]`-flagged constants (formation threshold, k_off, `dapk1_off_pp1`, k_dephos_transient) are set from the
biology's timescales/thresholds, NOT to a decode — and the robustness across Ca is the evidence they are not fitted.

**NEXT (the build continues):** wire `glun2b_memory` into the reward-gated path in `model6_core` (readout Ca as the
DDSC-analog commitment event; the coherent tag as eligibility; DAPK1/dopamine gating), and re-validate F3 end-to-end
in the full model. The isolation result is the green light.

## F3 END-TO-END IN THE FULL MODEL (2026-08-15) — the reward-signed readout PRODUCES, and it is COHERENCE-GATED
`glun2b_memory` wired into the reward-gated path of `model6_core` (EM path): the coherent P_S tag is the eligibility
carrier; a phasic dopamine transient GATES/TIMES the spin-selective binding-melt of a STILL-COHERENT tag (F3-d),
delivering the **DDSC-analog commitment calcium** (`CA_READOUT_UM=3.0 × eligibility_weight(P_S)` — low-µM scale
[GROUNDED — DDSC internal-store Ca, Jain 2024]; exact peak [MODELED]; magnitude rides on remaining coherence so the
quantum lever is preserved); the DARPP-32/PP1 cascade runs CONTINUOUSLY so the DAPK1 gate sees PP1 every step.

**RESULT (`sweep/f3_glun2b_fullmodel.py`, n=3; build 0.5 s → delay 10 s → reward → settle):**
| cond | commit | pre-commit (delay end) | final_mem |
|---|---|---|---|
| none | **0.00** | 0.00 | 0.044 |
| dip | **0.00** | 0.00 | 0.039 |
| burst | **1.00** | 0.00 | 0.354 |
⇒ **(1) HEBBIAN-SAFE** (nothing commits before the reward) and **(2) DA-DECISIVE** (only the burst commits). P_S at
reward = 0.934 (above the Werner floor) — the coherent tag carried the eligibility across the 10 s gap.

**CONTROLS — is it genuinely coherence-gated? (`sweep/f3_coherence_controls.py`, n=3; IDENTICAL delayed burst in
every arm, only the tag's coherence / readout mode differ; physiological brief 0.5 s burst per Yagishita):**
| arm | P_S@reward | readout Ca (µM) | commit | final_mem |
|---|---|---|---|---|
| undoped, quantum | 0.934 | 2.72 | **1.00** | 0.876 |
| ⁶Li, quantum | 0.934 | 2.72 | **1.00** | 0.916 |
| ⁷Li, quantum | **0.446** (decohered) | **0.00** | **0.00** | 0.763 |
| undoped, CLASSICAL baseline | 0.934 | 0.00 | **0.00** | 0.577 |
⇒ **ISOTOPE LEVER holds** (⁷Li's tag decoheres below the Werner floor → no readout → no credit; ⁶Li ≈ undoped) and
⇒ **TEMPORAL GAP holds** (the classical 0.3–2 s trace is dead at 10 s where the coherent tag still credits).
This is F1→F2→F3 closed end-to-end **under the corrected through-CaMKII mechanism** (the earlier F3-b/F3-c results
were obtained on the now-removed bypass; this supersedes them with the grounded architecture).

**CORRECTION MADE MID-BUILD (grounded, not tuned):** the first wiring let DAPK1 both block formation AND strip the
already-formed complex, so the memory was destroyed the moment dopamine returned to tonic (final_mem 0.035). That
contradicts the cited biology — **once formed the complex is PROTECTED from phosphatases and persists** (Cell Reports
2024; PMC4965558); DAPK1 suppresses *binding* **during LTD**. Fixed: DAPK1 gates FORMATION; disruption requires an
actual LTD signal (PP1 disinhibited above tonic, i.e. a dip). Memory then persists (final_mem 0.88–0.92).

**HONEST CAVEAT + a REAL pre-existing defect found (NOT fixed — surfaced for a decision):** in the non-credited arms
`molecular_memory` still drifts up (⁷Li 0.763, classical 0.577) even though `committed` is correctly 0.00 (the
coherence/window gate closes the commitment token). Root cause, measured: **CaMKII is spontaneously active at RESTING
calcium because of noise rectification.** `_update_CaCaM` adds `activation_noise = 0.02·√dt·randn` to `d_active` and
then clips `CaMKII_active` to [0,1]; at rest the target is ~0, so clipping rectifies the noise upward. Measured
(CaMKII alone, 60 s at 0.1 µM, tonic PP1): **stochastic → CaMKII_active 0.053, pT286 0.333; deterministic → 0.002,
0.011.** So resting CaMKII activity (and hence pT286 and slow complex formation) is a numerical artifact, not physics.
This is PRE-EXISTING (not introduced by this work) and prior F-series results carry it. Fixing it changes core model
behaviour broadly (reflecting-boundary or multiplicative noise instead of additive+clip), so it is **surfaced, not
unilaterally changed** — Sarah's call. Fixing it should only SHARPEN the discrimination above (the commit flag is
already correct; it would clean the memory variable in the non-credited arms too).

## NOISE-RECTIFICATION DEFECT FIXED + DAPK1 GATE CORRECTED (2026-08-15) — approved; with honest mixed results
### (a) The noise defect — FIXED, grounded, validated
Two coupled errors in `camkii_module`, both numerical rather than physical:
1. The conformational-activation noise was a **constant-amplitude** term (`0.02·√dt·randn`) that did not vanish as
   the transition fluxes vanish. Replaced with the **Chemical Langevin** form (σ ∝ √flux) already used elsewhere in
   the module, written via explicit activate/deactivate fluxes.
2. The CLE noise was applied to **FRACTIONAL** variables without the **1/√N** molecule-number scaling — over-
   amplifying it by ~√N (~50×). Added `n_holoenzymes = 2590` [GROUNDED — ~16 µM cytoplasmic CaMKII in a spine,
   Feng & Kennedy 2011; the PSD sub-pool is ~80–240, so this is the low-noise end].
**MEASURED (CaMKII alone, 60 s @ 0.1 µM resting, tonic PP1):** before `CaMKII_active 0.053 / pT286 0.333`;
**after `0.0035 / 0.0000`** (deterministic reference `0.0020 / 0.0109`) — the spurious resting activity is gone.
**Genuine activation IMPROVED** (30 s @ 5 µM: `CaMKII_active 0.337 → 0.933`) — the over-amplified noise had been
corrupting real activation too. **Stochasticity retained** (pT286 across 8 seeds @1 µM: mean 0.562, sd 0.116) —
important, because DDSC commitment is *biologically* stochastic. `__main__` still passes.

### (b) The DAPK1 gate — CORRECTED to the right upstream signal
The gate read `pp1_factor`, which **conflates** dopamine (PKA) with calcium (PP2B/PP2A). Measured: at tonic DA
**with calcium present**, PP2A slightly disinhibits PKA → `pp1` dips to ~0.97 → the gate **leaked ~6% open**, letting
Hebbian calcium alone slowly form the complex. "LTP-specific" means *requires the reward signal*, so the suppressor
is now **phospho-DARPP-32-Thr34** (the canonical PKA/reward node), threaded through `camkii.step(reward_thr34=…)`.
Thr34 separates the conditions ~50× (tonic+Ca ≈0.002–0.007 vs burst ≈0.25–0.4).
**ROBUSTNESS MEASURED (and my first written claim CORRECTED):** the result is decisive for `dapk1_half_thr34 ≥ 0.05`
and still decisive at 0.25 (≥5× plateau), but **FAILS at 0.035 and 0.02**. So it is a **one-sided bound, not a
two-sided fit** — an earlier code comment claiming "verified across 0.02–0.15" was wrong and was corrected. Value set
to 0.1 (an order of magnitude above tonic Thr34, well below burst) — inside the plateau, not at its edge.

### (c) Results after the fixes — what PASSES and what does NOT
- **F3 end-to-end, SUSTAINED reward (`sweep/f3_glun2b_fullmodel.py`, n=3): PASSES, and sharper than before.**
  burst commit **1.00** (final_mem **0.984**) vs none/dip **0.00** (mem 0.136/0.114); nothing pre-commits.
- **Coherence controls, SUSTAINED reward (n=8):** undoped **0.75**, ⁶Li **0.75**, ⁷Li **0.00**, classical **0.00**.
  The **coherence gate and isotope lever HOLD**; but commitment is **STOCHASTIC (6/8)**, so the script's
  deterministic `≥0.99` criterion reports FAIL. **The criterion is wrong, not the result** — DDSC is *dendritic,
  delayed and STOCHASTIC* (Jain 2024); the correct acceptance is a **commit-PROBABILITY contrast** (0.75 vs 0.00),
  which should be scored against a permutation null like the other keystones. Rewriting that criterion is owed.
- **BRIEF (0.5 s) phasic burst: NOTHING commits (n=8, all arms 0.00) — a STRUCTURAL finding, not a tuning target.**
  Mechanism, measured: the readout Ca²⁺ itself activates **PP2B, which strips Thr34 and re-engages DAPK1** before the
  complex can form. So **the mechanism predicts the dopamine/PKA signal must OVERLAP the DDSC calcium event for
  several seconds**, not merely trigger it. This is Nakano's Ca-vs-DA tension resurfacing at the binding step, and it
  is a falsifiable prediction of the architecture. (A first wiring also tied the readout-Ca *duration* to the burst
  duration; corrected — dopamine gates the ONSET, the melt then runs on the DDSC timescale, `READOUT_DURATION_S=20`
  [GROUNDED-in-band 10–100 s]. That correction did not rescue the brief-burst case.)
- **REMAINING WART (honest):** in the non-credited arms `molecular_memory` still rises (≈0.97) even though `commit`
  is correctly 0.00 — the persistent **dimer field** keeps phosphorylating T286 at basal calcium (reverse coupling
  runs every step), so a *prolonged* DAPK1 suppression can form the complex without a valid readout. Only the commit
  flag discriminates. This edges against Fernandez 2006 ("potentiation does not occur to dopamine OR glutamate
  alone") and is the next thing to resolve — the tag's field should not substitute for the readout.

## BOTH OWED ITEMS CLOSED (2026-08-15): CaMKII trace grounded, acceptance made probabilistic
### (d) The classical residual trace was rivalling the quantum tag — FIXED by grounding it
`k_dephos_transient` was 0.05 s⁻¹ (τ≈20 s), so ~half of pT286 survived a 10 s delay: the model carried a
**CLASSICAL ~20 s eligibility trace** that competed with the coherent tag as the carrier across the gap — and let a
prolonged reward commit with no readout at all. **[GROUNDED]** CaMKII stays active only **1–6 s** after a stimulus
(Jain 2024) and T286 prolongs the deactivation constant only to **~5–9 s** (Chang 2017). Set to **0.2 s⁻¹ (τ≈5 s)**.
Consequence: the coherent tag is the ONLY carrier across the gap — which is the model's actual claim — and the
memory separation sharpens. **F3 end-to-end (n=3): burst commit 1.00, final_mem 0.991; none/dip 0.00, mem
0.031/0.044** (was 0.14/0.11) — the "dopamine alone / no readout" path is now essentially closed in that protocol.

### (e) The acceptance criterion was a category error — REWRITTEN as a probability contrast
DDSC commitment is *dendritic, delayed and* **STOCHASTIC** (Jain 2024), so the controls' deterministic `≥0.99`
criterion was wrong and had mis-reported a working result as FAIL. `sweep/f3_coherence_controls.py` now scores the
**commit PROBABILITY** of the coherent arms against the decohered/classical arms with a **two-sided permutation
null** (the repo's standard decode-vs-null discipline), and takes the reward protocol from `DA_BURST_S`.
**RESULT (n=8/arm, sustained reward):** coherent (undoped+⁶Li) **0.750** vs decohered (⁷Li+classical) **0.000**,
contrast **+0.750**, **permutation p = 0.0000** ⇒ **COHERENCE-GATED READOUT SUPPORTED.** The isotope lever and the
temporal-gap contrast are both carried by that statistic.

### Residual, still open (narrower than before, stated plainly)
In the CONTROLS every arm receives a 20 s burst, and there the decohered arms' `molecular_memory` still rises
(≈0.84–0.89) while `commit` correctly stays 0.00. So the **commit flag** (coherence-gated) is the discriminating
variable, not the memory level.

**CAUSE DIAGNOSED — and an earlier attribution here was WRONG.** This was first written up as a *field*-driven pT286
floor (the reverse-coupling barrier reduction). Measured, that is **not** it: the driver is **CALCIUM returned by the
tag's own dissolving dimers.** Full-resolution trace of the calcium fed to CaMKII during a 10 s "rest" phase: median
**0.100 µM** (basal) but **3.5% of steps exceed 1 µM, max 3.2 µM**. Each dissolving dimer returns 6 Ca²⁺ into the
0.01 µm³ active-zone volume ≈ **1 µM**, added as **FREE** calcium with no buffer partitioning (`model6_core` →
`calcium.apply_return`). Because CaM activation is Hill **n=4** with K_half=1 µM, a single 3 µM spike is essentially
FULL activation, and with CaCaM decay τ≈2 s these spikes hold `CaCaM_bound ≈ 0.44` and pT286 ≈ 0.42 at "rest" — which
a prolonged DAPK1 suppression can then convert into complex formation with no valid readout. **So the tag substitutes
for its own readout via CALCIUM, not via the field.**

> **⚠ THIS DIAGNOSIS WAS ALSO WRONG — see "THE TWO DEBTS RESOLVED" below (2026-08-15).** The dissolution-return
> calcium is NOT the cause of the resting CaMKII activity either. Measured: the resting calcium spikes correlate
> with channel openings at **r = 1.000** and with dissolution events at **r = −0.010**. Kept here for the trail.

**Proposed grounded fix — NOT applied, wants Sarah's nod:** the returned calcium should enter as **free** calcium,
i.e. divided by the model's own buffer capacity `(1 + κ_s)` with κ_s = 60 (`params.calcium.buffer_capacity`, already
used by `get_buffer_capacity`; standard Neher buffer-partition formalism, and the same buffering the nanodomain λ
already encodes). That is ~61× smaller spikes (~0.016 µM) ⇒ CaMKII stays off at rest. It is a **physics correction to
a magnitude**, and the DDSC dissolution→Ca-return→CaMKII *mechanism* (LOCKED, May 12) is preserved unchanged — but it
does alter a locked pathway's quantitative behaviour and every F-series result carries the current magnitude, so it is
surfaced with evidence rather than changed unilaterally. It must NOT be papered over by tuning the formation threshold.

## Sources (verified this session)
- Yagishita et al. 2014, Science 345:1616 — dopamine window, PKA→CaMKII reinforcement.
- Nakano et al. 2010, PLoS Comput Biol 6:e1000670 — the kinetic DA/Ca striatal plasticity model (scheme + thresholds).
- Fernandez et al. 2006, PLoS Comput Biol 2:e176 — DARPP-32 as a robust DA/glutamate integrator (Thr34/Thr75).
- Svenningsson/Greengard — the DARPP-32/PP-1 cascade (Thr34⊣PP1; Thr75⊣PKA; PP2B/PP2A phosphatase roles).
- Xiao et al. 2023, Sci Adv — CaMKII autophosphorylation required for BTSP.
- Jain et al. 2024 (DDSC) — dendritic/delayed/stochastic CaMKII underlies BTSP (the LOCKED commitment pathway).

### Sources added 2026-08-10 → 2026-08-15 (the real-CaMKII-biology regrounding), by what each one grounds
**CaMKII is TRANSIENT, not a bistable switch** — retiring the Lisman/Zhabotinsky switch framing:
- Chang et al. 2017, Neuron — T286 prolongs the CaMKII deactivation constant only to ~5–9 s (from ~2 s).
- PLOS One 2015 (Camui FRET) — single-spine CaMKII activation is transient, ~1 min.
- Jain et al. 2024, Nature (also above) — CaMKII stays active only **1–6 s** after a stimulus; DDSC is **delayed
  10–100 s** and driven by **IP3-dependent INTERNAL-STORE** Ca²⁺, and is explicitly **stochastic**.
- Frontiers Synaptic Neurosci 2025 — the CaMKII/PP1 bistable switch is **not realizable at physiological spine
  concentrations**; autophosphorylation "does not constitute a bistable switch under biologically realistic
  conditions." (This is why our bistable-switch "Finding A" was correct biology, not a blocker.)

**The persistent memory is the CaMKII–GluN2B STRUCTURAL complex** (what actually latches):
- Cell Reports 2024 — LTP expression via **autonomous activity of GluN2B-bound CaMKII**; structural rather than
  enzymatic requirement.
- Molecular Brain 2013; PMC4965558 — the CaMKII/NMDAR complex as a molecular memory; binding needs an initial
  Ca²⁺/CaM stimulus then **persists after CaM dissociates**, is nanomolar-tight and **protected from phosphatases**.

**DAPK1 makes the binding LTP-SPECIFIC** — the gate that dissolved calcium-domination:
- DAPK1 / CaMKII–GluN2B (escholarship qt0zc5v40w) — DAPK1 suppresses CaMKII synaptic accumulation and GluN2B
  binding during LTD, making these mechanisms LTP-specific; the β-adrenergic "depression→potentiation" switch.

**Ser130/CK1 protection of Thr34** — the missing node that fixed the brief-burst arm (Debt 1):
- **Desdouits et al. 1995** — *"Phosphorylation of Ser-137 by casein kinase I inhibits dephosphorylation of Thr-34
  by calcineurin"* (the title result), in vitro **and** in vivo.
- Frontiers Behav Neurosci 2011 (DARPP-32 review, verified at source) — CK1 phosphorylates Ser-130 (Ser-137 rat)
  under **basal** conditions; **PP2C** removes it and that **facilitates** Thr-34 dephosphorylation; **PP2B
  dephosphorylates CK1**, so Ca²⁺ raises Ser-137 — the **incoherent feedforward** (Ca both drives and brakes Thr34
  removal). States the CK1 mechanism "remains incompletely understood" ⇒ our rates are `[MODELED]`.
- DARPP-32 **T34A** phospho-mutant work — higher PP1 activity left PKA-response **amplitude unchanged but
  "strongly reduced the DURATION"** ⇒ phospho-Thr34 *extends* the reward signal (a second extension mechanism,
  **still unmodelled**); PKA activity also persists after cAMP decays.

**Quantitative anchors used in fixes:**
- Feng & Kennedy 2011 (PMC3221876) — CaMKII pools in spines: ~16 µM cytoplasmic ≈ **2590 holoenzymes/spine**
  (PSD sub-pool ~80–240) ⇒ the `n_holoenzymes` used for the 1/√N Chemical-Langevin noise correction.
- EM postsynaptic-density morphometry — PSD is a disk ~**360 nm diameter** (range 200–800) ⇒ the ~180 nm radius
  used for the PSD-averaged calcium (constraint #1).
- Sabatini/Higley — spine calcium: resting ~100 nM; low-µM transients ⇒ the band the PSD-average must land in.
- Neher buffer-capacity formalism (κ_s ≈ 60) ⇒ the free/bound partition on the dissolution Ca-return.

## THE TWO DEBTS RESOLVED (2026-08-15) — one is a model gap, one had a mis-diagnosed cause (twice)

### DEBT 1 — "a brief phasic burst does not commit": researched, and it is substantially a MODEL GAP, not a clean prediction
The failure mode is real: the readout Ca²⁺ activates PP2B/calcineurin, which strips DARPP-32-Thr34 and re-engages
DAPK1 before the CaMKII–GluN2B complex can form, so a sub-second dopamine burst commits nothing. I had framed that as
a falsifiable *prediction* ("the DA/PKA signal must overlap the DDSC calcium event for seconds"). **The literature says
biology has at least two mechanisms for exactly this problem, and our cascade models NEITHER:**
1. **Ser130/CK1 PROTECTION of Thr34 (the direct answer).** Casein kinase 1 phosphorylates DARPP-32 at **Ser-130**
   (Ser-137 in rat), and "**this phosphorylation decreases the dephosphorylation of Thr-34 by calcineurin, *in vitro*
   and *in vivo***" (Frontiers Behav Neurosci 2011 review, verified at source; Desdouits 1995). So the very
   Ca²⁺→PP2B→Thr34-stripping that kills our brief-burst case is **physiologically braked**, and `darpp32_pp1_module.py`
   has no Ser130 state at all — it models only Thr34 and Thr75.
2. **Thr34 PROLONGS its own PKA signal.** In DARPP-32 **T34A** mice, higher PP1 activity "**strongly reduced the
   duration of the PKA response**" with no change in amplitude — i.e. phospho-Thr34 (by inhibiting PP1) *extends* how
   long the reward signal stays effective. And PKA activity is reported to **persist after cAMP has decayed**. Our
   cascade has no such signal-extension.
**VERDICT: the brief-burst negative is substantially an artifact of an INCOMPLETE cascade, not a property of the
mechanism.** It should NOT be reported as a prediction of the architecture. The grounded next step is to add the
**Ser130/CK1 protection** node to the DARPP-32 module (structure is cited; the rate is not published quantitatively —
the review says the CK1 mechanism "remains incompletely understood" — so it would carry a flagged `[MODELED]` rate),
then re-run the brief-burst arm. Until then the working protocol requires a sustained reward signal, and that
requirement is a **known model limitation, stated as such**.

### DEBT 2 — the resting-CaMKII activity: cause found on the THIRD attempt; my first two diagnoses were WRONG
Both earlier attributions in this document are **retracted**:
- ✗ "field-driven pT286 floor" (the reverse-coupling barrier reduction) — wrong.
- ✗ "the tag's own dissolving dimers return ~1 µM of free calcium" — **also wrong**, and the buffer-partition fix that
  was applied on that premise is a near-no-op (dissolution occurs on ~0.3% of steps and contributes ~0.016 µM).
**✓ THE ACTUAL CAUSE, measured decisively: SPONTANEOUS CALCIUM-CHANNEL OPENINGS, inflated by the timestep.**
In a 10 s rest phase: calcium exceeds 1 µM on 3.2% of steps; the indicator variable correlates with "≥1 channel open"
at **r = 1.000** and with "a dissolution occurred" at **r = −0.010**. With **zero** channels open the PSD calcium is a
clean **0.1001 µM** (max 0.1163); with one open it is **3.23 µM**. The arithmetic closes exactly:
| quantity | value |
|---|---|
| VGCC Boltzmann `P_open(−70 mV)` | 2.403×10⁻⁴ |
| opening rate per channel (`α·P`) | 0.240 s⁻¹ |
| 25 VGCCs ⇒ openings/s | 6.0 |
| **predicted** fraction of 5 ms steps containing an opening | **0.030** |
| **measured** in the full model | **0.032** |
| mean OPEN duration (`1/β_eff`) | **0.5 ms** |
| simulation `dt` | **5 ms** |
⇒ **a channel that opens is held open for a full timestep — 10× its true open time.** So the resting calcium
transients are ~10× too much charge per event, and with Hill n=4 CaM activation they keep CaMKII partly on at rest.
This is a **dt-resolution construct-validity artifact in the channel layer**, not a plasticity-cascade problem.
**Grounded fix (NOT applied — blast radius is the whole model, so it is Sarah's call):** scale a channel's delivered
calcium by the expected fraction of the timestep it is actually open (sub-timestep occupancy, ≈0.5/5 = 0.1 at rest),
or reduce `dt` toward the channel kinetics (10× the compute everywhere). Either changes EVERY result that runs through
the calcium layer — the whole F-series and the topology work — which is precisely why it is surfaced, not changed.
**What was applied:** only the Neher buffer partition on the dissolution return (`/(1+κ_s)`, κ_s=60) — correct physics,
documented in-code as a near-no-op that does NOT explain the symptom. Headline results are unaffected by it.

## DEBT 1 CLOSED (2026-08-15) — the Ser130/CK1 protection node, and the brief-burst arm now works
Built the missing node identified by the Debt-1 research, in `darpp32_pp1_module.py` (opt-out via
`ser130_protection=False`, which reproduces the previous behaviour exactly):
- **New state `ser130`** — the CK1 site (Ser-130 mouse / Ser-137 rat). [GROUNDED — Desdouits 1995, whose title
  result is "Phosphorylation of Ser-137 by casein kinase I inhibits dephosphorylation of Thr-34 by calcineurin,"
  confirmed in vitro AND in vivo; DARPP-32 carries this phosphorylation under BASAL conditions; PP2C removes it, and
  loss of Ser-137 FACILITATES Thr-34 dephosphorylation.]
- **The incoherent feedforward** [GROUNDED]: PP2B dephosphorylates (activates) CK1, so **calcium RAISES Ser130** —
  Ca²⁺ therefore simultaneously *drives* Thr34 removal (via PP2B) and *brakes* it (via CK1→Ser130→protection). That
  brake is exactly what the model lacked.
- Rates are `[MODELED]` (the CK1 mechanism is "incompletely understood" in the literature), set from timescales and
  the two qualitative constraints: basal Ser130 substantially phosphorylated, and calcium increasing it.

**RESULT — the physiological brief burst now commits.** With an identical 0.5 s phasic burst (n=8/arm):
| | before Ser130 | after Ser130 |
|---|---|---|
| coherent (undoped+⁶Li) commit | **0.000** | **0.375** |
| decohered (⁷Li+classical) | 0.000 | **0.000** |
| permutation p | — (vacuous) | **0.0168** |
So a sub-second, physiologically realistic reward burst now produces **coherence-gated** credit, where before the
model committed nothing at all. **Nothing else moved:** the sustained-reward controls are unchanged (coherent 0.750
vs decohered 0.000, p=0.0000) and the F3 end-to-end suite still passes (burst 1.00 / mem 0.991; dip & none 0.00).
`darpp32_pp1_module.__main__` still passes 5/5.

**IS THE RESCUE FITTED TO THE `[MODELED]` CONSTANT? — swept, and no.** Commit rate vs `prot_frac_max` (0.5 s burst,
undoped, n=6): **0.00 at 0.0** (which *is* the pre-Ser130 model, and it correctly reproduces the failure), 0.00 at
0.3, 0.17 at 0.5 and 0.65, then a **plateau at 0.33 across 0.8 / 0.9 / 0.95**. A monotone rise to a broad plateau —
i.e. the MECHANISM does the work, not a tuned value. As with `dapk1_half_thr34`, it is a **one-sided threshold with a
wide plateau above it**, and that is how it is reported.

**HONEST RESIDUAL:** the brief-burst commit probability (0.375) is about **half** the sustained-reward value (0.750),
so the model still favours longer reward signals — the limitation is *reduced*, not eliminated. And the research
surfaced a SECOND grounded extension mechanism that is still unmodelled: **phospho-Thr34 prolongs its own PKA signal**
(in DARPP-32 T34A mice the PKA response has unchanged amplitude but "strongly reduced duration"), plus PKA persisting
after cAMP decays. Adding that positive feedback is the next available grounded improvement to this arm.
