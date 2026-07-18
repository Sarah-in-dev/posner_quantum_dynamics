# Lead: po4-gap (PO-4 · the analytical gap, biologically grounded) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** every subsystem either advances during silence
with a cited timescale or is excluded with a stated reason — nothing in neither column; and a
measurement shows **committed vs uncommitted spine volume SEPARATING across an honest gap**.
Demonstrated failing on the current 1 ms-per-30 s code first.

**Status:** LIVE — pre-registered (`3bf4ad4`) and the probe committed FAILING on current code
(`806adc7`). Next unit: consolidation, then the honest-gap fix.
**Current unit:** consolidation (MO ruling 1) — no compute needed.
**Last heartbeat:** 2026-07-18 18:20Z
**Blocked on:** — (nothing; K_CLASSICAL is MO-held but does not block my path)

---

## Cycle log

### 2026-07-18 17:58Z — brief accepted, rulings absorbed, work started

Brief returned and ACCEPTED. The MO recorded its own error rather than quietly fixing it
(`mo-f2-001.md` superseding section): it verified the docstring's two lists and never read the
function tail, i.e. **prose checked against prose**, then tagged it `[code SHOWN]`.

**The correction that now stands:** the gap is not frozen. Its tail
(`src/models/Model_6/sweep/run_theta_burst_45s.py:284-288`) jumps `network.time` by the full
gap and then runs `network.step(0.001, ...)`. Actin / `E_invasion` / CaMKII / DDSC each advance
**exactly 1 ms per 30 s**. Retention `exp(-0.001/180) = 0.9999944`, not 1.0 — **worse than
frozen, because it reads as an even cleaner ratchet.**

**Five rulings received and being worked:**
1. Consolidation APPROVED — one definition in `run_theta_burst_45s.py`; `run_spatial_discovery.py`
   imports it, its 252-line copy deleted; `run_place_field_learning.py:347`'s stale comment
   deleted; **nothing else** touched in that file.
2. dt-convergence check APPROVED against the 5 s full-physics validator (`:405-415`).
   DECISION RECORD `dt-1` covers `P_S`/edges but **not** transient-phase counts — do not assume
   it transfers.
3. `K_CLASSICAL` — MO-held. Reported, untouched.
4. DDSC delta — measure and report, do not damp.
5. No analytic commitment state (`model6-commitment-pathway` LOCKED).

### FINDING F-4a — the acceptance target's own numbers have no artifact behind them

`MO_MODEL6.md:140` and this file cite **"the isolated-module numbers say 1.291 vs 2.389 at
+300 s"**. `grep -rn '1.291\|2.389'` over the repo returns **two hits, both coordination prose**
(`MO_MODEL6.md:140`, `leads/po4-gap.md:6`). **No code, no results file, no log entry produces
them.** That is the program's characteristic defect class — prose asserting a quantity the code
does not demonstrate — sitting in **my own acceptance bar**.

I will therefore **verify them by reproduction** before pre-registering, and pre-register
against what the module actually yields, not against the quoted pair. If reproduction disagrees,
the discrepancy is the finding and it routes to the MO.

### FINDING F-4b — PO-1's B2 edit couples the gap defect to the pump

`model6_core.py` currently carries PO-1's uncommitted B2 work (I did **not** touch it; the
shared-file hazard is respected). That edit makes the per-synapse pump drive read
`getattr(self.spine_plasticity, 'E_invasion', 0.0)` via `compute_metabolic_power`.

**Consequence:** once B2 lands, a gap that fails to advance `E_invasion` no longer merely
freezes plasticity — **it freezes the per-synapse pump drive across every silence too.** The two
defects compose. This raises the priority of the gap fix and is worth the MO's attention when
sequencing B2 against this PO. Routed to `queue/po4-gap.md`.

### 2026-07-18 18:20Z — pre-registered, and the probe FAILS on current code as required

**L·GAP-1 registered (`3bf4ad4`) then demonstrated failing (`806adc7`).** Per `fa12009`.

| arm | conf | R measured | R registered | verdict |
|---|---|---|---|---|
| uncommitted | 0.000 | **0.999994** | 0.8948 | STOPPED CLOCK |
| committed | 0.976 | **0.999978** | 0.6751 | STOPPED CLOCK |

**The zero-duration null is the sharpest evidence in the run:** `R(0 s) = 0.999994`,
`R(20 s) = 0.999994`, **ratio = 1.000000.** Retention does not depend on gap duration —
the signature of a fixed-size tick, not decay. **The whole defect in one number.**

Verdict returned **INCONCLUSIVE**, correctly: null 1 fired, and CONFIRMED is unreachable by
registration until an honest arm exists. Verdict logic was not rewritten after seeing data; the
R(0)-vs-R(20) comparison is reported as explicitly post-hoc, outside the verdict.

### FINDING F-4c — a crash in the copy that SURVIVES consolidation

`analytical_gap(net, 0.0)` raised `ZeroDivisionError` (`np.ceil(0/dt_sub) == 0`, then divides).
**Found by the pre-registered null** — the control that exists to be unable to show an effect
instead exposed a crash. Guarded with `max(1, ...)`; minimal, my surface, not a physics change.
**Distinct from the `chi_redistribution` ZeroDivisionError the MO routed in `bcd15b8`**: that one
dissolves with B2's deletion set; this one is at `run_theta_burst_45s.py:71`, in the copy I am
keeping, and was in **no** deletion set.

### FINDING F-4d — the compute wall, and why the live drive path cannot carry this measurement

Measured: a 12-cycle theta traversal on a **2**-synapse network = **190.7 s wall for 1.5 s
simulated (~127× slower than realtime)**. A 20 s full-physics reference ≈ 42 min. And that
traversal leaves `actin_enlargement = 0.0106`, `E_invasion = 0.0000` — **an order of magnitude
below `invasion_threshold = 0.1`**. So the live path neither fits the budget nor reaches the
regime where retention is defined. Probe therefore uses a **controlled initial condition**, with
the precedent (`model6-dimer-formation-chemistry` §2) and the limit both stated in its docstring.
**This is a second, independent reason "the full model has never been allowed to show it."**

### FINDING F-4e (ESTIMATE, routed to PO-3) — the ratchet may saturate AT the threshold

Gain ≈ 0.0106/traversal, decay ≈ 0.8948/20 s gap ⇒ asymptote `0.0106/0.1052 = 0.1008` against
`invasion_threshold = 0.1`. **Lands within 1% of the threshold** ⇒ `E_invasion ≈ 0.0005`,
indistinguishable from zero for any number of traversals. If it holds, a no-ratchet result means
**the drive is ~10× too weak to clear the threshold**, NOT that `tau_extrude` fails — two very
different findings, and the negative branch is Sarah's call. Flagged to PO-3 as an estimate with
a minutes-long check that could save its 130-minute slot.

### Next units (none blocked)
1. Reproduce the isolated-module committed-vs-uncommitted baseline (cheap, no heavy slot).
2. Pre-register: discriminating quantity, null, positive control, verdict function.
3. Commit the measurement **failing on current code** before any physics change (`fa12009` precedent).
4. Then the consolidation + the per-subsystem advance/exclude table.

---

**Owns:** `analytical_gap` in BOTH drivers; `run_theta_burst_45s.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params + its uncommitted
`model6_core.py` slice · PO-2's `atp_system.py` / phosphate path · PO-3's
`spine_plasticity_module.py` (**call it, never edit it**) · PO-5's `multi_synapse_network.py`,
T1′ family · PO-6's surfaces.

**NOT this PO's to decide:** `K_CLASSICAL` (50× spread: 0.05 / 0.005 / 0.001), live in BOTH
copies of the gap — the gap runs the **retired** rate. MO-owned. Report, do not touch.

**Compute:** NO heavy slot. PO-3 holds the single heavy backgrounded slot. Code work, the
two-copy diff and the subsystem ruling need none of it.
