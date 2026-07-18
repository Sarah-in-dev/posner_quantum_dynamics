# Queue: po2-phosphate — actions awaiting the MO / Sarah (append-only, owned by this PO)

Each entry: **the exact ask · why · the PO's recommendation · the evidence.** Decision-ready.

## Q1 — Does the `np.maximum(..., 0.0)` phosphate clamp count as the pin's forbidden "cap"?

**Ask:** ruling on whether the clamp at `model6_core.py:451` and `:757` may stay.

**Why:** the may30 pin's Step E reads *"A finite pool must actually be finite. Correctness,
**not a cap**, not an edge-target."* The clamp is literally a cap on the pool floor, and when it
fires it **creates phosphate** — the pool would have gone negative and instead is held at 0.
That is a conservation violation in the direction of the defect I was dispatched to fix.

**PO's recommendation: LEAVE IT, INSTRUMENT IT.** It is a symptom, not the cause. If the clamp
ever fires, conservation was already violated upstream — so the right move is to make the probe
*detect* clamp activation and report it as a distinct failure mode, not to silently remove it.
Removing it lets `phosphate_structural` go negative, which is worse physics and would produce a
nonsense speciation at `atp_system.py:399-401`. **I do not read this as the "cap" the pin
forbids** — the pin's cap is a *ceiling* installed to stop runaway formation (the 427-dimer
problem); this is a *floor* preventing negative concentration. Different object, same word.
Flagging because the wording is close enough that I should not decide it alone.

**Evidence:** `model6_core.py:450-452`, `:756-757`; pin Step E; `atp_system.py:399-401`.

**Not blocking** — the probe measures and reports clamp activation either way.

---

## Q2 — Which pool should ATP recovery debit: metabolic-first, or proportional?

**Ask:** ruling on the split, because it is arguably a physics call and I must not make those.

**Why:** fixing defect 2 means `update_recovery` debits phosphate when it regenerates ATP
(ATP = ADP + Pi). But there are two pools, and `atp_system.py:397-398` declares
*"Calculate concentrations FROM STRUCTURAL POOL ONLY / Metabolic pool doesn't participate in
Posner chemistry"*. Which pool the debit lands in **changes how much phosphate remains available
to dimer formation**, so it is not a bookkeeping detail — it directly sets the SOC feedback
strength this PO exists to make real.

**PO's recommendation: METABOLIC-FIRST, falling back to structural only when metabolic is
exhausted.** `add_phosphate_from_atp` (`:419-425`) sends **90%** of hydrolysis-released Pi to the
metabolic pool, described at `:407` as *"protein binding, rapid cycling"*. Mitochondrial ATP
resynthesis physically draws from exactly that rapidly-cycling inorganic pool, not from the
structural pool reserved for Posner chemistry. Metabolic-first is also the **conservative** choice
for my own acceptance: it leaves the structural pool larger, which makes the SOC feedback *weaker*
and therefore harder for me to claim. I would rather under-claim the engine than tune toward it.

**The thing I will NOT do:** introduce a fitted partition coefficient. If metabolic-first does not
balance, the log records the gap (§7 LOCKED, and the temptation my kickoff named explicitly).

**Evidence:** `atp_system.py:163`, `:169-171`, `:397-398`, `:407`, `:419-425`.

**Not blocking** — I pre-register metabolic-first as primary and report the proportional variant
as a sensitivity, so a ruling either way is absorbed without a re-run.
