# Queue: po4-gap — actions awaiting the MO / Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready.

---

## Q4-1 · 2026-07-18 17:58Z · **MO sequencing:** B2 composes with the gap defect

**The ask:** the MO should know, when sequencing PO-1's B2 against this PO, that the two
defects compose — and decide whether B2's landing order matters.

**Evidence** `[code SHOWN]` — PO-1's *uncommitted* slice in `src/models/Model_6/model6_core.py`
(I did not touch it; shared-file hazard respected) replaces the pump drive with:
```python
p_met_W = compute_metabolic_power(
    getattr(self.spine_plasticity, 'E_invasion', 0.0),
    self.calcium.channels.get_open_fraction(),
    self.params.dendritic_backbone.p_active_max_W,
)
```

**Why it matters:** today a gap that advances `E_invasion` by 1 ms per 30 s freezes *plasticity*
across silence. After B2, `E_invasion` also drives the per-synapse pump — so the same stopped
clock **freezes the per-synapse pump drive across every silence too**. The gap defect stops
being contained to plasticity and starts contaminating the pump result B2 exists to clean up.

**PO-4's recommendation:** no re-sequencing needed — the two are file-disjoint and both fixes
are additive. But **any pump measurement taken across a multi-trial gap before this PO lands
should be read as carrying the stopped clock**, and B2's acceptance probe should be checked for
whether it spans a gap. Flagging rather than deciding: this is the MO's edge to rule on.

---

## Q4-2 · 2026-07-18 17:58Z · **FYI, not an ask:** the acceptance bar's own numbers are unsourced

`MO_MODEL6.md:140` cites "the isolated-module numbers say 1.291 vs 2.389 at +300 s".
`grep -rn '1.291\|2.389'` over the repo → **two hits, both coordination prose.** No code, no
results artifact, no log entry produces them.

**PO-4 is not blocked by this** — it is verifying by reproduction and will pre-register against
what the module actually yields. Raised so the MO knows the number in its own board is
currently prose-only, and so that if reproduction disagrees, the disagreement is expected
rather than a surprise.

---

## Q4-3 · 2026-07-18 17:58Z · **Reported, not touched:** `K_CLASSICAL` is the retired rate

Per ruling 3 the MO holds this. Recording the PO's finding for the record: `K_CLASSICAL = 0.05`
is live in **both** gap copies (`sweep/run_spatial_discovery.py:80`,
`src/models/Model_6/sweep/run_theta_burst_45s.py:69`), i.e. the gap dissolves dimers at the
rate `model6-dimer-formation-chemistry:64` **retired** (`0.05 → 0.005`, cluster lifetime
τ≈200 s, Turhan 2024). Untouched by this PO. After consolidation there will be **one** site
carrying it rather than two, which makes the MO's eventual decision a one-line change.
