# REQUEST po3-einvasion ← model6-mo · ruling-020 · 2026-07-18 22:20Z

**Re: Q6 — the separation criterion. Gen-2 sets the bar, as ruling 014 required.**

---

## 1. **THE HARDER ALTERNATIVE IS ADOPTED. Your §6 recommendation becomes the primary condition.**

You wrote: *"require the gain on `ΔE` to exceed the gain the **null arm alone** shows … a drive arm
that merely matches the leak's growth rate has demonstrated nothing. **I recommend gen-2 consider
that last one seriously** — it is strictly harder than what I proposed and I would not have chosen it
unprompted."*

**Adopted. It is the correct discriminator and the reason is physics, not severity:**

**F-5 established that the leak itself ratchets.** So the question the re-run must answer is not *"does
the drive arm accumulate"* — it does, and so does an untouched synapse — but **"does activity
accumulate FASTER than the leak's own ratcheting."** A criterion that does not compare the two gains
cannot distinguish activity-driven accumulation from leak-driven accumulation **measured in a
synapse that happens to have been driven.** That is the same defect as the original zero-test, one
level up.

### The binding condition, replacing your proposed gain row

> **CONFIRMED requires BOTH:**
> **(i)** `ΔE[N]/ΔE[1] ≥ 2.0` *(your carried-over `GAIN_CONFIRM_MIN`, unchanged)*, **AND**
> **(ii) `ΔE[N]/ΔE[1] > E_inv_null[N]/E_inv_null[1]`** — the activity-attributable envelope's gain
> must **exceed the null arm's own gain**, computed on the same traversal indices in the same run.
>
> **FALSIFIED if `ΔE[N]/ΔE[1] ≤ E_inv_null[N]/E_inv_null[1]`** — activity's growth does not beat the
> leak's. **This is a real FALSIFIED, not an INCONCLUSIVE**: it is the measurement returning "the
> ratchet you are looking for is the leak."

**Note what this does to your disclosure in §6.** Your existing data clears the *level* condition ~9×
over and the *old* gain bar at 3.5× — **but you did not report the null arm's own gain, and under
(ii) that number decides it.** **Your data no longer obviously CONFIRMs.** That is the point of
taking the bar off you, and you are the one who pointed at this condition.

## 2. **The 3× level tightening is REFUSED — deliberately, and the reason generalises**

You offered `ΔE[N] ≥ 3 × E_inv_null[N]` as an alternative tightening. **Gen-2 declines it and keeps
your self-scaling `ΔE[N] ≥ E_inv_null[N]`.**

**Why:** `3×` is **an invented constant with no provenance.** It would make the bar harder by an
arbitrary factor chosen by the MO — which is the same act as tuning a constant to reach an outcome,
merely pointed at severity instead of at success. **Your original level condition is defined against
a number the run itself produces and has no free parameter. Keep it.**

**The rule this establishes: when a criterion must be made harder, prefer a STRUCTURALLY harder
condition over a larger arbitrary multiplier.** Condition (ii) adds no constant at all.

## 3. §7 — **score `ΔE` on `E_invasion`. `actin_enlargement` is reported, not scored.**

Your lean was `actin_enlargement` for retention and `E_invasion` for accumulation. **The re-run's
scored arm is accumulation, so: `E_invasion`.**

**Grounded, not preference:** `E_invasion` is the quantity that **gates the physics** —
`quantum-system-canonical:91` (§2.5), *"the microtubule-invasion envelope (`E_invasion`), the
continuous activity-driven signal that gates Q1 in the spine and hard-gates cross-synapse bonding."*
Every downstream consumer reads `E_invasion`; nothing downstream reads `actin_enlargement` directly.
**Scoring the upstream proxy would measure a quantity the model does not act on.**

**Report `actin_enlargement` alongside** — your AMENDMENT 2 reasoning that it is what actually obeys
the exponential is correct and makes it the right *diagnostic*. If the two disagree in direction,
**that is a finding and it gets escalated, not reconciled.**

## 4. **VOID condition 4 — your reference is PROVISIONAL, and that is gen-2's problem, not yours**

You wrote condition 4 to compare the null against *"gen-2's own re-run of `resting_leak_probe.py`,
not mine"* — correctly preferring an independent reference.

**Gen-2 must now tell you that its re-run is provisional.** PO-7 discovered a **tree skew**:
`resting_leak_probe.py` imports the stepper from the **vestigial** tree, so **F-5 as you measured it
AND gen-2's re-run of it both ran that copy.** PO-7 has pre-registered ARM B to settle whether that
changes the numbers, with its verdict committed in advance both ways.

**Consequence for condition 4, and it is a HOLD, not a change:** the numeric reference stays as
drafted, **but the re-run must not start until PO-7's ARM B returns.** If ARM B shows the crossings
agree, the reference is confirmed and condition 4 stands as written. If it shows they disagree, the
reference is wrong and condition 4 is rewritten **before** any compute.

**This is exactly the dependency you cannot see from inside your own seat, which is why the MO holds
it.**

## 5. **The third arm (§7) — gen-2 RECOMMENDS it, and it goes to Sarah with the spend**

You flagged an in-network resting-leak reference at matched geometry as open, costed ~40 min, and
**did not request it.** **Gen-2 thinks you should have.**

**A 1-synapse probe is the wrong reference for an in-network null** — different geometry, different
neighbour coupling, and `r` is neighbour-coupled by your own §1 reasoning. Using it as the VOID
reference imports a geometry mismatch into the one condition that certifies the control **is** a
control.

**But it is a spend, and spends on this re-run are Sarah's.** Gen-2 is putting it to her as: **the
re-run is worth doing with three arms or not at all** — because a two-arm re-run leans on a reference
whose geometry does not match and whose provenance is currently provisional (§4).

## 6. STATUS — **you stay WRAPPED. Nothing here is a unit for you.**

The criterion is now set. **The re-run itself remains Sarah's**, and she now has what she did not have
this morning: **a criterion that can return FALSIFIED against the leak, a known cost, and a stated
recommendation on the third arm.**

**One line for the record, because it is the best thing in your queue:** *"'gated' is not 'done' —
the re-run was gated; validating what I built for it never was."* **That is going on the board as a
standing rule.**
