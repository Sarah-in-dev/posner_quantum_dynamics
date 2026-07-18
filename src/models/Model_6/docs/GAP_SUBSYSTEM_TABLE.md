# The analytical gap — per-subsystem ADVANCE / EXCLUDE table

**PO-4 · 2026-07-18 · the deliverable the MO named as still owed at `5e12712`.**
Function: `analytical_gap`, `src/models/Model_6/sweep/run_theta_burst_45s.py` — **the single
definition**, imported by all three consumers since the consolidation (`7b05153`).

## The rule this table exists to enforce

**Nothing may be in neither column.** The defect this PO was dispatched to fix was not a wrong
timescale — it was four subsystems (actin, `E_invasion`, CaMKII, DDSC) appearing in **neither**
the docstring's "computed" list nor its "NOT computed" list, and therefore advancing 1 ms per
30 s while the prose read as complete. That is this program's characteristic defect class: prose
asserting mechanisms the code does not implement.

Every entry below is either **ADVANCED with a cited timescale**, or **EXCLUDED with a stated
reason**, or **DERIVED — carries no integrated state, so there is no clock to advance** (which
is itself a stated reason, not a third way to be silent).

---

## A. ADVANCED during the gap — integrated at `dt_sub`

| # | subsystem | timescale that makes integration necessary | source |
|---|---|---|---|
| 1 | P_S decoherence | per-dimer `T_eff`, `T_base = 216 s` at `fraction_P31 = 1.0` | Agarwal 2023 (dimer coherence, hundreds of s) |
| 2 | Dissolution | `k_diss = K_CLASSICAL·(1 − singlet_excess)`, `K_CLASSICAL = 0.05 s⁻¹` | **⚠ the RETIRED rate — see §D** |
| 3 | Particle removal | tracks (2); no independent timescale | — |
| 4 | Bond cleanup (`P_S < 0.5`) | tracks (1); no independent timescale | — |
| 5 | Stochastic disentanglement | `k_decohere = 0.01·(1 − P_Si·P_Sj)` s⁻¹ | in-code |
| 6 | **Actin enlargement / `E_invasion`** | `tau_extrude = 180 s` unconfined; **≈ 50.9 s confined**, where commitment redirects the pool to stabilization | Honkura 2008 (2–15 min band) |
| 7 | **Spine volume** | `tau_volume_follow_actin = 5 s`, follows `actin_total` | Matsuzaki 2004 |
| 8 | **CaMKII** | DDSC window **30–40 s post-induction** — *inside* the interval that used to be skipped | Jain 2024 (Nature) |
| 9 | **DDSC** | `tau_rise = 15 s`, `tau_decay = 50 s`, peak ≈ 35 s | `ddsc_module.py:16-17,73` |

**6–9 are the fix.** Before `7b05153` they were in neither column.

**Consequence for the program, not just for this function:** Jain 2024's 30–40 s DDSC window fell
*entirely* inside the skipped interval, so **delayed commitment could not resolve in any
gap-based experiment as previously written.** It now can.

## B. EXCLUDED — "settles fast, then clamp", with the timescale that justifies it

These are **honest exclusions and a defensible modelling choice.** They are listed separately
from §A so they are not tarred with the silent freeze that §A rows 6–9 used to represent.

| subsystem | timescale | reason |
|---|---|---|
| Calcium dynamics | at baseline within **~2 s** | settles far inside any gap; clamped at baseline |
| Dopamine | clears in **~2 s** | no reward is delivered during a gap, by construction |
| ATP level | **τ ≈ 5 s** | re-equilibrates far inside any gap |
| ATP hydrolysis / phosphate production | — | no drive at rest |
| EM field / tryptophan superradiance | — | no drive at rest (V = −70 mV, no reward) |
| New dimer formation | — | requires the calcium transient; none at baseline |
| New bond formation | — | requires the EM field; none at rest |
| Quantum measurement gate | — | requires dopamine; none in a gap |
| Network entanglement tracker O(n²) recalc | — | **cost, not physics** — deferred to the single refresh in the tail, stated plainly rather than dressed as a physical argument |

## C. EXCLUDED — no integrated state, therefore no clock

**Corrected 2026-07-18 on MO ruling 007.** This section previously sat as a third category,
"DERIVED", and the **docstring did not carry phases 9 and 12 at all** — so measured against the
docstring's own two columns, two phases were in **neither**. That is the exact defect this table
exists to eliminate, reintroduced at the edge of the fix that removed it. **"It has no
integrated state" is a stated reason, so these belong in the EXCLUDED column** and are now listed
there in both artifacts.

| phase | quantity | reason for exclusion |
|---|---|---|
| 4 | local dimer→tubulin modulation | no state; derived from `n_dimers`, `mean_coherence` (both advanced), recomputed at tail |
| 5 | network modulation integration | no state; derived from per-synapse modulations |
| 7 | `k_agg` forward coupling | no state; derived from `E_invasion` (advanced) + channel open fraction |
| **9** | **eligibility from the particle system** | **see below** |
| **12** | **template feedback** | **see below — its reachability CHANGED with this fix** |

### Phase 12 — template feedback. The substantive one, and this fix created the question.

Gated on `spine_volume > 1.5` (`model6_core.py:715-720`) — **which this gap now advances.** Before
the fix, volume was frozen and this could not fire in a gap. **After it, it genuinely is reached:**
L·GAP-2 measured the committed arm at **1.9312 inside a 300 s gap**, and D20 records the pathway
*does* fire (~8 s onset, roughly doubling the template-bound fraction, raising `T_eff`).

**Excluded anyway, and the reason survives that fact — verified, not assumed:**

1. `set_n_templates` mutates `templates.template_field` (`ca_triphosphate_complex.py:643-670`).
2. That field is read **only at dimer creation**, to set `dimer.template_bound`
   (`dimer_particles.py:205`). **Existing dimers are never re-flagged.**
3. **Formation is excluded in a gap**, so the field has **no consumer** there.
4. Gap dissolution does **not** use the template term at all — `k_diss` is computed inline from
   `K_CLASSICAL` and `singlet_excess`, never via `update_dimerization`.
5. The pathway is **memoryless** — phase 12 recomputes `n_templates` from the *current* volume
   every step, with no latch — so the tail step's single evaluation from the post-gap volume
   lands on **exactly** the value stepping would have produced.

**Reworking template feedback is not PO-4's surface.** If the memoryless recompute is itself
wrong (e.g. it should latch on a mid-gap excursion), that is a separate finding for its owner.

### Phase 9 — eligibility from the particle system

A derived readout, `(mean_P_S − 0.25)/0.75` over existing dimers, whose `P_S` §A advances. Its
**only** consumer is `ddsc.check_trigger` (`model6_core.py:735`), which sits inside `if plateau:`
— and a plateau is stimulus-driven, so **it cannot occur during silence.** Recomputed at the tail.

**Note on phase 7:** before B2 this row was inert with respect to the gap. **After B2 the
per-synapse pump drive reads `E_invasion`** — so the stopped clock would have frozen the *pump
drive* across every silence too. Raised as `queue/po4-gap.md` Q4-1 and ruled on: B2's own
acceptance probe does not span a gap, so it is uncontaminated.

## D. NOT A PATHWAY — recorded so it is never listed as one

**`quantum_field_kT`.** Accepted by `spine_plasticity.step` at three call sites and **read at
none** — measured **bit-identical volume for kT ∈ {0, 1, 5, 20, 100}** (DECISION RECORD D21(1));
five `ActinParameters` barrier fields are declared and never referenced, and the module docstring
describes a quantum barrier-modulation mechanism **that does not exist in the code.** The gap
passes `0.0` (the honest value — no drive at rest) and **claims nothing.** Listing it as an
advanced pathway would have been the exact defect class this table eliminates, and the docstring
one would be tempted to copy from is itself an instance of it.

**`K_CLASSICAL = 0.05`** (§A row 2) is the rate `model6-dimer-formation-chemistry:64` **RETIRED**
(to `0.005`; cluster lifetime τ ≈ 200 s, Turhan 2024). It is **MO-held and deliberately untouched
by PO-4.** Every dissolution number this function produces inherits it. After consolidation it
sits at **one** site rather than two, so the MO's decision is a one-line change.

## E. What the tail step is, and is not

The function ends with `network.time += gap_duration_s` then
`network.step(0.001, {...})`. **That 1 ms step is a state-sync, not an advance** — it refreshes
§C's derived quantities and runs the real commitment threshold test.

**Before the fix it was the *only* plasticity advance in the entire gap**, which is why actin,
`E_invasion`, CaMKII and DDSC each moved exactly 1 ms regardless of gap length. Measured:
`spine_plasticity.time` +0.0010 s against `network.time` +20.0010 s.

**Commitment is still evaluated only there,** never in the gap loop. CaMKII *integrates* across
the gap (that is what needs the time); the commitment threshold test is instantaneous and stays
in `model6_core.py:671-685`. This keeps `model6-commitment-pathway` (**LOCKED**) satisfied — no
commitment state is written analytically — and avoids duplicating that condition where it would
drift.

## F. Evidence

| claim | artifact |
|---|---|
| clock advances with the gap | `spine_plasticity.time` +20.0010 s vs `network.time` +20.0010 s (was 0.0010) |
| retention matches the closed form | **8/8 out-of-sample points**, 10–60 s, both arms, max err **0.0039** vs 0.02 tol |
| residual is discretisation, not modelling | first-order Euler convergence, ratio **2.04–2.08** (`0497aa1`) |
| the fix changes the outcome | separation **ΔV = +0.7764** vs **+0.000299** on pre-fix code — **2595×** |

Pre-registration and its four amendments (including **AMENDMENT C, disclosing that PO-4's own
registered numbers were wrong**): `docs/PREREG_PO4_GAP.md`.

## G. Standing limits — carry these whenever the above is cited

- **Controlled initial condition.** The spine state is set directly, not driven through the live
  glutamate→calcium→actin path. **That path does not reach this regime:** a 12-cycle theta
  traversal leaves `actin_enlargement = 0.0106` and `E_invasion = 0.0000` — an order of magnitude
  below `invasion_threshold = 0.1` — at **~127× slower than realtime**. The same wall L·ETA-3 and
  L·ETA-5 hit, now from a third direction.
- **Two synapses, one network.** No scaling claim.
- **This validates the gap, not the drive.** Nothing here says anything about the physics *during*
  a traversal.
- **`K_CLASSICAL` is the retired rate** and is live in every dissolution number above.
