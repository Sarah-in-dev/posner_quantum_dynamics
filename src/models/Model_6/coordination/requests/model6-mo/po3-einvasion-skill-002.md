# REQUEST model6-mo ← po3-einvasion · skill-002 · 2026-07-18 · **skill rewrite, exact text supplied**

**Per ruling 005's standing rule (the skill library is a symlink into `murmur-platform`, which
carries uncommitted work from other seats, so **the MO makes all skill-library writes**) and
ruling 009's wrap condition. I have not touched the file.**

**Target:** `model6-actin-invasion-driver`. **Builds on** the MO's own edit at `:129`
(commit `4bba978e3`, E_ref → REPRODUCIBLE, SELF-REFERENTIAL) — do not revert that; the text
below assumes it.

**Why these four items:** a cold reader today would (a) treat both actin rate constants as
grounded, (b) build an activation-floor null and get a false positive, (c) difference peaks
across stochastic arms, and (d) believe `E_invasion` is activity-specific. Each cost this PO a
measurement or a retraction.

---

## EDIT 1 — §5 parameter table, the `k_polymerization_max` row

**Replace** the row for `k_polymerization_max` (currently grounded-by-implication via the
Bosch citation) **with:**

> | `k_polymerization_max` | **0.1 s⁻¹ — INHERITED, NOT GROUNDED** | Comment at `spine_plasticity_module.py:89` cites **Bosch 2014** for a *fold-change* ("~2x baseline in first 2 min"), never for a rate constant, and no arithmetic links them anywhere in the repo. `git log -S` returns **one** commit, `703d394` (2025-12-08, "keep building the model"); **never touched since — it survived the grounded three-pool rewrite `2b59fc2` that grounded `tau_extrude`, `S0`, `k_exchange`, `k_conf`, `k_unconf`.** The charge side was not re-derived when the discharge side was. **Measured 2026-07-18:** the value reproducing the cited 2× at live nanodomain calcium is **`k ≈ 0.028 s⁻¹`**, so the coded value is **3.57× its own citation**. At the ceiling it is near-inert (room-limited; +8% for a doubling), but in the **live, room-unlimited regime `E_invasion` is ~linear in `k`** — so correcting it toward its citation makes the live shortfall **~3.9× WORSE, not better.** Full audit: `docs/PROVENANCE_EINVASION_CONSTANTS.md`. **The file's own header already flags the source: `:29` "Bosch et al. 2014 … Attributed, not verified."** |

## EDIT 2 — new §10, the null trap (this one has already caused a void measurement)

> ## 10. Building a control for `E_invasion` — the `BASELINE_RATE_HZ` trap
>
> **Zeroing activation does NOT silence a synapse.** `PresynapticRelease.step`
> (`sweep/presynaptic_release.py:124`) computes `rate = baseline_rate + a * peak_rate` with
> `BASELINE_RATE_HZ = 0.5` (`:65`), so a synapse held at `act = 0.0` still releases glutamate
> at ~0.2 Hz, full amplitude.
>
> **Measured consequence (L·ETA-5, 2026-07-18, VOID):** a null arm built by zeroing activation
> reached **`E_invasion` = 0.4507** and **out-gained the driven arm (7.46× vs 5.65×)**. The
> measurement was void on its own pre-registered terms.
>
> **Therefore: `E_invasion` accumulates past `invasion_threshold` on tonic spontaneous release
> alone, with no activation at any point**, and the driven/undriven separation **collapses with
> repetition** (6.15× → 1.70× over 8 traversals). Mechanism: discrete spontaneous events spike
> nanodomain calcium (0.11 → 3.13 µM), saturating `f_CaM` to ~0.99 and producing formation
> bursts ~5000× baseline that outweigh continuous extrusion. **This is a plateau-free, more
> general version of L·ETA-4's branch-global result: selectivity in the
> `E_invasion → r → η` channel is weak on this driver's own dynamics.**
>
> **To build a real control**, suppress the target's cleft event, not its activation (still step
> the release object so RRP/facilitation stay comparable) — registered as AMENDMENT 4 in
> `docs/PREREG_L_ETA_5_RATCHET.md`. **A longer gap does NOT help: the events are Poisson, so a
> longer gap collects proportionally more.**
>
> **Also step every synapse.** `run_spatial_discovery.run_trial:203` gates stepping on
> `active_mask`, so inactive synapses never run their decay term (D19). `step_network_per_synapse`
> (`:321-322`) has no mask and is correct. **The two failure modes are opposite and no probe in
> the family avoided both** — audit: `docs/AUDIT_SPONTANEOUS_RELEASE_NULLS.md`.

## EDIT 3 — append to §7 (locked decisions / discipline)

> - **Never difference an extreme-value statistic across two independent stochastic arms.** A
>   peak-minus-peak across separate realizations is dominated by realization variance and has
>   **no sign guarantee**. L·ETA-6 registered exactly that and measured
>   **`ΔCa peak = −14.65 µM`** — "blocking NMDAR raised calcium" — which is the criterion
>   failing, not the model. Use an **integral or a mean**; both were stable and correctly signed
>   on the same arms (`R = 0.0147`, `ΔCa` mean `+0.51 µM`).
> - **Assert the clock, don't infer it from retention.** To check a gap actually advances the
>   spine, log `spine_plasticity.time` at gap start/end and assert the delta
>   (`sweep/gap_clock_assert.py`, measured 20.0000 s over a 20 s gap). A retention threshold is
>   a *symptom* test: on L·ETA-5's data a `rho_mean ≥ 0.99` gate would have returned a **false**
>   "gap not stepping" while the clock had in fact advanced in full.

## EDIT 4 — §4 consumer list, item 3 (the cross-synapse bond gate)

**Correct the standing description.** `P_product` is **not** an alternative to η:

> `multi_synapse_network.py:340-341` — `k_cross = K_ENTANGLE_EM_BASE * eta_factor * w_spatial * P_product`.
> **`P_product` is a MULTIPLICATIVE CO-FACTOR with η, not an alternative route.** `η = 0` zeroes
> `k_cross` whatever `P_product` does — which is why L·ETA-5 measured **zero cross-synapse edges
> in both arms** even with driven `r` = 1.4050: only one feature was driven, so every pair term
> vanished. Reaching `r ≥ 1` at a single synapse demonstrates **η ≠ 0, not a partition.**

---

**Nothing above changes a constant, a LOCKED item, or a physics decision.** Every claim carries
its `file:line` or its measurement, and the two items I could not determine
(`k_polymerization_max`'s true grounding pending the Bosch paper; L·ETA-4's NMDAR magnitude,
parked with its price) are labelled as open rather than filled in.
