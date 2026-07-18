# Signals: po3-einvasion — findings other POs need before this PO's work lands

---

## F-3 · 2026-07-18 · **SUPERSEDED BY F-3-CORRECTED BELOW — the ~100× figure is WRONG**

> **DO NOT ACT ON THE NUMBERS IN THIS SECTION.** The original text is left in place per the
> log convention (supersede, never rewrite). The measured correction is at the bottom of this
> file. Original heading was: "The L·ETA-3 harness under-delivers glutamate ~100× — affects
> PO-5 directly".

**Who needs this:** the **MO** (L·ETA-3's `ca_open` attribution is affected, and that verdict
is what PO-3 exists to test) and **PO-5** (blocked on PO-3; if PO-5 reuses
`eta_in_live_trial.py` as prior art it will inherit this defect into a selectivity test).

**The defect.** `sweep/eta_in_live_trial.py:138-144` steps presynaptic release once per
**agent** step (`AGENT_DT = 0.5 s`) and then runs 100 physics steps against that one stale
stimulus. The shipped reference implementation `sweep/run_spatial_discovery.py:434-441`
(`run_trial`) steps it **inside** the physics loop, once per `physics_dt = 0.005 s`.

`PresynapticRelease.step` (`sweep/presynaptic_release.py:110-139`) is a per-timestep
Bernoulli draw, `p_spike = 1 - exp(-rate*dt)`. Calling it at 0.5 s intervals removes ~99% of
release opportunities, and the single returned value is then held constant across the whole
0.5 s window.

| | release opportunities / 14 s traversal | expected release events |
|---|---|---|
| shipped `run_trial` | 2800 | ~350 |
| L·ETA-3 harness | 28 | **~3.3** |

**Measured, not inferred.** Same seed, same geometry, same feature:

| | `max_glu` at target | `peak_r` traversal 2 |
|---|---|---|
| inheriting the L·ETA-3 pattern | **0.0000** (at `max_act = 0.9950`) | 0.0571 |
| release stepped per physics step | **1.0000** | **0.1428** |

**Consequence.** `ca_open` is `get_open_fraction()`, which includes NMDAR opening, and NMDAR
opening is glutamate-gated. L·ETA-3's `ca_open = 0.140` (vs the rig's 0.38) was therefore
measured with the glutamate contingency substantially unsatisfied — the **ERR-2 failure
class** in a new location. The 13× shortfall, the zero cross-synapse edges and the
`E_invasion` trace mechanism all survive; the *split* of the shortfall between
`E_invasion` and `ca_open` does not.

**What PO-3 did and did not do.** Corrected in PO-3's own probe only, as pre-registration
**AMENDMENT A1.1** (`docs/PREREG_L_ETA_5_RATCHET.md`), recorded **before** the scored run,
with no verdict threshold changed. **L·ETA-3's log row was NOT touched** — another PO's
entry, and the verdict under test. Escalated to the MO and queued for Sarah as Q1 with a
recommendation (ERR-2-style narrowing banner, not a retraction, and no re-run of L·ETA-3).

**Action for PO-5:** if you reuse `eta_in_live_trial.py`, step release inside the physics
loop. A selectivity test run on the uncorrected harness would be measuring a partition
under starved NMDARs — and NMDAR is precisely the channel L·ETA-4 found selectivity
surviving in (`P_product`).


---

## F-3-CORRECTED · 2026-07-18 · measured, and it inverts the original claim

**PO-5: the action item below is downgraded, not withdrawn. Read this, not F-3 above.**

I escalated F-3 with figures I derived from `rate × time` and never measured. Measured
directly (20 seeds, `act = 0.995`, 14 s traversal):

| | release EVENTS / traversal | physics-steps WITH glutamate present |
|---|---|---|
| shipped `run_trial` (per physics step) | **19.0** | **19** |
| L·ETA-3 `eta_in_live_trial` (per agent step) | **1.0** | **100** (each held 100 steps) |

- The event ratio is **19×, not ~100×**. My ~350 and ~3.3 were both wrong — they ignore the
  10 ms refractory and vesicle depletion, which dominate at 25 Hz.
- **The claim inverts on exposure duration:** the L·ETA-3 pattern holds each release for 100
  physics steps, so glutamate is *present* ~5× longer per traversal than in the shipped
  pattern. **"Starves the NMDARs" is not established and is probably backwards.**

**What PO-5 should actually do:** still match the shipped `run_trial` call pattern in any new
probe — a probe diverging from its reference implementation is a defect regardless of sign —
but do **not** carry forward "the old harness starves NMDARs" as a fact, and do not treat
L·ETA-3's `ca_open` as compromised. **My recommendation that L·ETA-3 carry a correction
banner is WITHDRAWN** (see `queue/po3-einvasion.md` Q1-CORRECTION).

**What remains genuinely open and is worth PO-5 knowing:** the two patterns differ in the
*temporal structure* of glutamate (19 brief events vs 1 sustained event), and **which of those
the NMDAR gate responds to is UNMEASURED.** `peak_r` moved 0.0571 → 0.1428 between them, so
the structure matters — but the direction and mechanism are not established, and I am not
claiming them.

---

## F-4 · 2026-07-18 19:59Z · **PO-5: your foundation is weaker than the log states — L·ETA-4's NMDAR half is vacuous**

**Directed at PO-5 (and the MO). Audit only — L·ETA-4 was not re-run and its row was not
edited.** Full evidence: `docs/AUDIT_SPONTANEOUS_RELEASE_NULLS.md` §1.

L·ETA-4's row states: *"Selectivity survives in NMDAR exactly as Jain 2024 requires
(silent-synapse NMDAR gain from plateau **-0.0019**, i.e. zero — no glutamate, no current,
however depolarized)."* That is **the sole surviving basis for the `P_product` selectivity
hypothesis** after §8's η premise failed twice. **It does not hold as evidence**, for two
independent reasons, either sufficient:

**1. The premise is false.** The probe's six "silent" synapses sit at `act = 0.0`
(`plateau_vgcc_leak_probe.py:125`) and therefore still release: replaying its own seeds
(`3000+i`) for its own `T_S = 12 s` gives **13 release events**, ≈2.16 synapse-seconds of
NMDAR occupancy. There *is* glutamate at the silent synapses.

**2. The metric cannot see the effect.** `split_open` returns `ch.state` — open **fraction**.
NMDAR gating (`analytical_calcium_system.py:129-130`) is `alpha*g_bind` / `beta`, with **no
voltage term**; `B(V)` scales **current** only. So a plateau cannot change silent-synapse NMDAR
open fraction **by construction**, whether or not calcium flowed. `-0.0019` is RNG residual.
**This is the same vacuity class already fixed once in this probe at the verdict layer — it is
still present one layer down, in the metric.**

**The direction runs against you.** `B(-20)/B(-70) = 20.6×`, so during that occupancy
silent-synapse NMDAR calcium current is ~21× its resting value. **The plateau plausibly DOES
drive NMDAR calcium at silent synapses** — the opposite of what the row concludes.

**What still stands, so you do not over-correct:** the **VGCC half is sound** (open fraction
*is* the right metric there — VGCC gating is voltage-dependent), and **L·ETA-4's conclusion
stands on it**: η cannot carry input-selectivity under a plateau, §8's premise still fails.
Nothing here rehabilitates §8.

**What PO-5 should do:** do **not** treat "the NMDAR channel is clean under plateau" as
established. Before building on `P_product`, the discriminating measurement is silent-synapse
NMDAR **current or calcium** (not open fraction) with the plateau on and off, at a control that
suppresses spontaneous release. **The magnitude is UNDETERMINED** — I did not re-run L·ETA-4;
that is the MO's to sequence.

**Reusable pattern, from L·ETA-5's own failure:** `AMENDMENT 4` in
`docs/PREREG_L_ETA_5_RATCHET.md` registers a control that suppresses the target's cleft event
entirely while still stepping the release object (so RRP/facilitation stay comparable). Any
probe needing a genuinely silent synapse can copy it.


---

## F-5 · 2026-07-18 21:53Z · **`E_invasion` HAS NO ZERO — it crosses threshold on resting leak alone in ~80 s**

**For the MO and PO-5 (live now). Measured, not extrapolated:** `sweep/resting_leak_probe.py`,
1 synapse, 252 s, **glutamate never supplied, held at −70 mV**:

```
  t= 60s  enl=0.09599  E_inv=0.00000
  t= 80s  enl=0.12137  E_inv=0.01208   <-- crosses invasion_threshold = 0.1
  t=252s  enl=0.24164  E_inv=0.08002
```

**Cause:** the resting VGCC leak. `P_open ≈ 2.4e-4` at −70 mV (Boltzmann, `V_half = −0.020`,
`V_slope = 0.006`) — tiny, but non-zero across 50,400 steps, and `tau_extrude` clears too slowly
to hold it down. Consistent with the **missing VGCC inactivation term** already documented as
open in `model6-input-engine`.

**This is stronger than F-4 and than L·ETA-5's tonic-release finding.** Those needed spontaneous
glutamate. **This needs no input at all.**

**What it means for PO-5:** any control you build — silent synapse, unpaired arm, no-input
baseline — **will show non-zero `E_invasion` if your protocol runs longer than ~80 s.** Do not
write a criterion requiring a control to sit at `E_invasion = 0`; it is unsatisfiable. **Score a
separation at matched elapsed time instead.** I have escalated the same correction for my own
pre-registration (queue Q5) rather than quietly fixing it, and I have NOT self-approved it,
because the change flatters my own result.

**Related, already signalled:** the `BASELINE_RATE_HZ` null trap (F-3-CORRECTED) and the
extreme-value-differencing rule. **This one is the floor beneath both.**
