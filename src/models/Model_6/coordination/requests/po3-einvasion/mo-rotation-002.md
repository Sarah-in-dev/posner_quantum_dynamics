# MO → PO-3 · ROTATION 002 · 2026-07-18 20:02Z · **next unit — the magnitude you left open**

Rotation 001 is complete and **accepted**: the MO verified your three load-bearing code claims
directly (the L·ETA-4 probe stepping release for all seven synapses; `advance_silent()` as the one
correct suppressor; `active_mask` as D19's mechanism). The cross-probe table is exactly the shape
asked for. **F-4 is routed and PO-5's pre-positioned file now carries it.**

## The unit: how much NMDAR calcium did L·ETA-4's "silent" synapses actually get?

**You named this gap yourself:** *"The magnitude of NMDAR calcium at L·ETA-4's silent synapses is
not established here."* **That is the next question and it is decisive.**

Right now the MO has escalated to Sarah that PO-5's `P_product` premise is **unsupported** — the
evidence is vacuous because the premise "no glutamate" is false. **But unsupported and wrong are
very different, and which one it is depends entirely on magnitude:**

- If the spontaneous floor delivered a **negligible** NMDAR calcium contribution at the silent
  synapses, then L·ETA-4's −0.0019 may still be very nearly right, and the `P_product` hypothesis
  is **weakly supported rather than unevidenced**. PO-5 survives roughly as scoped.
- If it delivered a **material** contribution, then the silent synapses had a live NMDAR AND-gate,
  and the selectivity claim is not merely unevidenced — **it may be contradicted.** PO-5 needs
  re-scoping.

**Sarah is holding a decision that turns on this, and the MO would rather she rule on a number than
on an absence.**

## What to measure

At L·ETA-4's silent synapses, under its own conditions: the NMDAR-attributable calcium from the
spontaneous floor, against the driven synapse's, and against whatever threshold makes it matter
for `P_product` formation. **State it as a ratio and as an absolute.**

**Constraints:**
- **You are auditing another PO's probe, not re-running its verdict.** Measure the input magnitude;
  do not re-derive L·ETA-4's conclusion, do not edit its log row, do not touch
  `plateau_vgcc_leak_probe.py` beyond what a read-only instrumented copy requires.
- **Pre-register** the discriminating quantity and what counts as negligible **before** measuring —
  "negligible" decided after seeing the number is not a finding.
- **Compute:** this should be cheap (seconds-to-minutes, seven synapses, short window). If your
  design needs more than a few minutes, request the slot in your queue first. PO-2 is running a
  conservation probe and PO-1 is mid-audit.

## Acceptance

A measured NMDAR-calcium magnitude at the silent synapses with its pre-registered
negligible/material threshold, and an explicit verdict: **does L·ETA-4's `−0.0019` survive as
approximately correct, or is it contradicted?** Either answer is a result. **"Cannot determine
without re-running L·ETA-4" is also a valid answer** — say so, with what it would cost, and stop.

## Still gated, still not yours
The L·ETA-5 re-run (Sarah) · `K_CLASSICAL` (MO) · any re-scoping of PO-5 (Sarah) · editing another
PO's log row.

## Standing
Poll `board.md` + `requests/po3-einvasion/` every cycle · heartbeat with `date -u` · open questions
to your queue **and keep working** · never end a turn on an unanswered question · **do not park
while non-gated work exists.**
