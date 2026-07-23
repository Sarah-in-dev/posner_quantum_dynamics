# PO-10 Advisor Update — Unit C: the first WEIGHT-LEVEL result

**2026-07-22 · for external review.** Follows `PO9_ADVISOR_PACKET_R3` (localization discharged) and the
Unit C pre-registration + Amendment 1. Honest headline: **a modest but robust weight-level keystone with a
clean control ladder.** The write-up is deliberately front-loaded with what did NOT work, because two of
the three informative moves here were negative.

---

## 1. What Unit C tested (and why it's new)

Everything in the program to date was measured at the **substrate/graph** level (components, cross_w,
domains, fidelity). Unit C is the **first measurement at the weight level**: an input goes in, a synaptic
weight-change vector Δw comes out, and we ask whether Δw carries input structure **a classical readout of
the identical substrate cannot recover**. That is the "computation for a purpose" bar — not "the graph has
structure" (known) but "the structure becomes a usable weight pattern."

## 2. The design (built to make binding NECESSARY, not incidental)

Four synapse clusters A,B,C,D in **one dendritic compartment**, all mutually within λ_F. Two trial types
("pairings"), decoder label = pairing:
- **pairing 1:** A,B co-active; C,D co-active → domains {AB},{CD}
- **pairing 2:** A,C co-active; B,D co-active → domains {AC},{BD}

**Marginals are identical by construction** — every cluster is active the same duration/intensity in both
types; total drive, per-synapse calcium, everything first-order matched. **Only which pairs coincide
differs.** So any first-order/scalar mechanism is at chance by construction; only a *second-order* (pairing)
channel can separate the types. In this model the only cross-synapse second-order channel is the
entanglement partition.

**Readout:** dopamine at a fixed delay → joint collapse → per-synapse Δw. Each correlated domain collapses
to a shared random ±1, so Δw's sign is arbitrary; we decode the **sign-invariant pairwise agreement
structure** (which clusters co-commit with the same sign) with a linear classifier, LOO-CV vs a
2000-shuffle null. Four arms: **full** (binding on), **bindoff** (independent collapse — the classical
control), **scramble** (same domain sizes, shuffled membership), **lamshort** (λ_F=5, binding below the
Werner floor). Built entirely as experiment scripts — **zero model .py changes**, so it is off-path by
construction.

## 3. The two gating pilots (both passed before any scored run)

- **Pilot A (geometry):** the substrate partition tracks the pairing cleanly — pairing1 → {AB},{CD},
  pairing2 → {AC},{BD}, with **hard zeros on the non-co-active pairs across every draw**, and no
  branch-global contamination at 10 µm spacing in one compartment.
- **Pilot B (classical blindness):** the compartment-aggregate calcium scalar is matched between types
  (integral 1.2%), and code inspection confirms the only cross-synapse channel to the readout is the
  entanglement partition. So the classical arm is blind to pairing by construction.

## 4. The honest arc (three moves; two were negative — recorded, not hidden)

1. **Batch 1 (forward-order only, n≈3/cell) — NULL.** `full` decoded at 0.667 = exactly chance. Diagnosed
   (from the per-draw agreements, NOT by tuning): only the LATE pair carried reliable signal, because the
   EARLY pair's cross-bond had decayed below the Werner floor by the 20 s readout (early pair ~50–65 s old,
   past the ~57 s Werner crossing). The registered decoder was **not** swapped for a late-pair-only variant
   to force a pass.
2. **Amendment 1 — counterbalance the ordering.** Run each pairing in both orderings so each pair is
   late-position (hence reliably co-signed) in half its trials. This fixes the decay asymmetry AND the
   positional confound (which pair fired late is now balanced across classes → the decode must ride the
   pairing, not timing). Registered before re-scoring.
3. **Strengthening at delay 5 — NULL (dead end).** We tried a shorter readout to save the early pair.
   `full` at delay 5 = 0.667, chance — it did NOT help; if anything the pairing is *less* cleanly readable
   early. Consistent with the partition needing decay-driven **self-cleaning** (weak cross-pairing bonds die
   below the Werner floor, sharpening the two pairs) — the L·PO7-5 idea — though n is too small to claim a
   real delay-dependence. Recorded as a negative.

## 5. The result (counterbalanced, delay 20)

| arm | n/cell | decode | null p95 | verdict | rules out |
|---|---|---|---|---|---|
| **full** | 6 | 0.750 | 0.708 | DECODES (marginal) | — |
| **full** | **12** | **0.767** | **0.674** | **DECODES (robust)** | — |
| bindoff | 6 | 0.458 | 0.708 | chance | "it's not the quantum joint collapse" |
| scramble | 6 | 0.000 | 0.583 | chance | "any grouping would do" |
| lamshort | 6 | 0.500 | 0.708 | chance | "it's not really the Werner-gated binding" |

**Robustness:** doubling `full` to n=12/cell held the decode (0.767) while the null band tightened
(p95 0.708 → 0.674), so it clears by a **wider** margin — the effect is not a marginal fluke.
(`bindoff` is being extended to n=12 for symmetry; already chance at n=6, expected to remain so.)

**Interpretation:** the input's pairing structure **reaches the synaptic weights** and is recoverable there,
while a classical readout of the identical substrate cannot recover it. Every alternative explanation is
closed by the control ladder.

## 6. Honest caveats
- **Modest, not ceiling** (~0.77, not ~0.95). The pre-reg predicted near-ceiling one-shot; we got a real
  but modest effect. It does **not** strengthen at shorter delay (§4.3).
- **Sign unresolved.** Δw carries the pairing as an *agreement pattern* (sign-invariant), so it is
  **informative but not yet consistently useful** — a learning rule needs a direction, and nothing in the
  architecture currently resolves the per-domain sign. This is the open architectural question directly
  behind Unit C.
- **Still the (A) reading** (canonical §5.1): Δw is a common-cause correlation; no non-classicality is
  claimed. Consistent with the deep-research verdict that a classical correlation reservoir computes the
  same class — the entanglement is doing the *constraining*, the classical readout the *computing*.

## 7. Where we'd value your judgement
1. **Is the modest effect the honest ceiling of this mechanism, or a power/design limit?** The self-cleaning
   delay-dependence (worse at delay 5) suggests the effect is intrinsic to the decay dynamics, not just
   underpowered. Do you read 0.77 as "the computation is real but low-capacity here," or is there a design
   move we're missing to raise it?
2. **The sign problem.** An agreement pattern is information; a weight update needs a direction. Where should
   the sign come from — is there a physically-motivated symmetry-breaker (a reference domain, a
   neuromodulatory bias) that resolves it, or is sign-invariant information the honest output of this
   architecture?
3. **Is the matched-marginal pairing task the right discriminating object**, or would you want a different
   input structure to press the "computation vs. correlation" distinction harder?

Data: `results/po10_unitC/` (force-added). Harness/decoder: `sweep/po10_unitC_*.py`. Pre-reg + Amendment 1:
`docs/PREREG_PO10_UNIT_C_WEIGHT_LEVEL_KEYSTONE.md`. Research log: `L·PO10-2`.
