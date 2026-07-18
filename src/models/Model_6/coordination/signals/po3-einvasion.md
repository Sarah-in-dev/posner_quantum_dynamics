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
