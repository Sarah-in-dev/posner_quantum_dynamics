# Signals: po3-einvasion — findings other POs need before this PO's work lands

---

## F-3 · 2026-07-18 · The L·ETA-3 harness under-delivers glutamate ~100× — affects PO-5 directly

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
