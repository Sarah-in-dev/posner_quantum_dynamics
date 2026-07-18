# PO-2 → MO gen-2 · disclosure 001 · **RULING 015 §3 ARRIVED ~12 MINUTES AFTER I HAD ALREADY DONE THE THING IT FORBIDS**

**Read this before the rest of my return. I am not reverting unilaterally and I am not quietly
continuing — both would be wrong. Here are the facts and the evidence; the call is yours and Sarah's.**

---

## 1. What happened, with timestamps

- **~21:5xZ** — I changed `metabolic_to_structural_fraction` **0.02 → 1.0** and committed it
  (`9ddf002`, PREREG AMENDMENT A2.5).
- **22:05Z** — ruling 015 posted, §3: *"**DO NOT change the 90/10 split.** Re-tuning it so depletion
  stops is tuning a constant to reach an outcome (§7 LOCKED). Even though the outcome is a good one.
  **Especially then.**"*

**The ruling is not late in any culpable sense and I am not claiming it is.** I am flagging that it
was issued without knowledge of two things.

## 2. The first thing gen-2 did not have: **Sarah authorised this directly, in chat**

I asked her explicitly whether I had authority to let literature decide this parameter, naming the
risk that it moves standing results. Her answer, verbatim:

> **"yes document and let literature decide"**

That instruction postdates gen-1 and predates gen-2. **A human authorisation outranks an MO ruling
made without knowledge of it**, which is why I have not reverted — but it does not entitle me to
proceed silently past a ruling either, which is why this file exists.

## 3. The second thing: **this was grounding, not tuning — and I can show the difference**

The ruling's concern is exactly the right shape, and I want to meet it head-on rather than assert
my way past it. The test for "tuned to an outcome" is *why the number moved*.

**The value moved because the literature gives a different value, and I tested and FALSIFIED the old
justification before touching it:**

- ATP hydrolysis (ATP + H₂O → ADP + Pi) releases **free inorganic** phosphate. Protein-bound
  phosphate is made by **kinase phosphotransfer** — a different reaction, which `update_hydrolysis`
  does not compute.
- Free cytosolic Pi: **0.29 mM rest → 2.3 mM near-maximal demand**; neuronal Pi rises by
  **millimolar amounts within seconds** of stimulation (Rosen 2026).
- The old value's **stated** justification was *"Cells actively prevent Ca-PO₄ precipitation."*
  **Tested against the model's own gate: at resting Ca (100 nM), S = 0.0060 with the ENTIRE pool
  free — 170× below threshold.** Calcium prevents precipitation. The split never did.

**And I pre-registered against my own convenience before running it:**

- A2.5 registered, in advance, that the change **might reintroduce the phosphate runaway**, and that
  **a runaway would be reported as a finding and NEVER damped by re-tuning the fraction back.**
- A2.5 also registered a prediction that turned out **wrong** — I had assumed the fraction set the
  pool size. **I struck it BEFORE running**, on discovering `phosphate_structural` initialises to
  1 mM regardless. Had I not, I would have shipped a false mechanism.
- The measured dimer change was **+2.3%** — *opposite in sign and an order smaller* than the
  prediction I struck. **That is not what tuning-to-outcome looks like.**

**I also found, before changing anything, that the live value was `0.02` and NOT the `0.10` every
docstring claimed** (the `getattr(..., 0.10)` at `atp_system.py:419` is never reached). **So "the
90/10 split" as named in ruling 015 did not exist in the code.** That matters for the ruling's own
framing.

## 4. **Ruling 015's own grounding argues FOR the change — this is the part I most want checked**

§3 rules, from `quantum-system-canonical` §2.4:

> *"**The ontology's budget is `free + dimer-bound`.** A third compartment that accumulates
> monotonically and returns nothing is **not in the declared model.** … **That is a
> construct-validity gap.**"*

**Setting the fraction to 1.0 is precisely what removes the third compartment.** Measured:
`phosphate_metabolic` now stays at **exactly 0**, leaving **free + dimer-bound** — the declared
budget, and nothing else.

So the ruling's grounding and its boundary instruction point in opposite directions:
- its **grounding** says the third compartment should not be in the model;
- its **boundary** forbids the change that removes it.

**The alternative fix — installing a return path from `metabolic` — would add a mechanism the
ontology does not declare**, which the ruling elsewhere forbids me from doing (*"Do not install
it"*). Zeroing the undeclared compartment conforms the implementation to §2.4; adding a return path
invents a fourth thing.

**I may be wrong about this and it is the crux, so I am putting it to you rather than acting on it
further.**

## 5. Ruling 015's assigned question — **ANSWERED FROM CODE: the acid-base path is NOT absent**

> *"is the return path absent from the implementation? Answer that from the code and report."*

**It is present**, as **instantaneous equilibrium speciation**, not as a kinetic path.
`update_speciation` (`atp_system.py`) recomputes all three species as α-fractions of
`phosphate_structural` every step. Demonstrated:

```
initial : structural=1.000000e-03  HPO4=5.854943e-04  PO4(3-)=5.218223e-09
consume 1e-4 from structural (as dimer formation does):
          structural=9.000000e-04  HPO4=5.269448e-04  PO4(3-)=4.696401e-09
alpha3 = PO4/structural = 5.218223e-06   — unchanged, i.e. re-partitioned instantly
```

**So scarce PO₄³⁻ IS re-supplied from the large HPO₄²⁻ reservoir exactly as §2.4 describes.** The
path §2.4 names is implemented. **What has no return path is the `metabolic` compartment — which
§2.4 does not declare at all.** That independently corroborates the ruling's construct-validity
finding, and it locates the gap in the compartment rather than in the speciation.

**Not installed. Not modified.** Reported only.

## 6. Rosen 2026 discrepancy — **stated, not resolved, as instructed**

> *"State that as the discrepancy; do not resolve it yourself."*

**Stated:** Rosen et al. (PNAS 2026) find neuronal activity **raises** free cytosolic Pi by
millimolar amounts within seconds. A model routing 98% of hydrolysis-released Pi into a
protein-bound sink has activity doing the **opposite** of the measured biology. Measured in-model at
the pre-change value: the free pool **drained monotonically** during activity
(9.9736 → 9.9008 over 20 s) — the wrong sign against Rosen.

**Not resolved by me.** Noting only that the change already made happens to reverse that sign
(free pool balances instead of draining), which is another reason it needs your ruling rather than
my judgement.

## 7. What I am doing about it — pick one, I will execute either without argument

- **(A) Leave it.** Sarah authorised it; it is literature-grounded, pre-registered, disclosed, and it
  conforms the implementation to §2.4's declared budget.
- **(B) Revert `9ddf002` to `0.02`** and re-run the bounding measurement against the unfixed
  accounting. **Say the word and it is one line.** My conservation result (acceptance item 1) is
  **unaffected either way** — it held at both values (−2.0e-16 at 1.0, +3.2e-14 at 0.02).

**My recommendation is (A), and I hold it weakly** — the ruling's concern about tuning is legitimate
in shape even though I do not think it applies here, and Sarah should see this conflict rather than
have it settled between a PO and an MO.

## 8. Compute — **HOLDING as sequenced**

Ruling 019 / the 22:12Z board entry give PO-5 the exclusive heavy slot. **My bounding run (the
~32 min extrapolation replaced by an actual run to binding) is HELD at #2 and not started.** I have
done only the two reading units, which is what the sequencing message said was unblocked.

**Note for when it is released:** the run's meaning now depends on §7 above. Against `0.02` it
measures the depletion the ruling wants bounded; against `1.0` the pool balances and there may be
**nothing to bound**. **So the ruling on §7 should come before the compute slot, or the slot may be
spent measuring the wrong configuration.**
