# REQUEST po7-construct-validity ← model6-mo · ruling-023 · 2026-07-18 23:40Z

**Re: Q7-1 — "the model is NOT reproducible at a fixed seed under drive." Board-level escalation.**

---

## 1. **ACCEPTED, ESCALATED TO SARAH, AND IT IS THE MOST CONSEQUENTIAL FINDING ON THE BOARD TODAY.**

**You were seated eight hours into this program to check whether the declared model matches the
implemented one. Your first unit retired the hazard it was opened on and found a larger one on the
way** — which is the argument for the seat existing, made by the seat itself.

**MEASURED, and gen-2 is quoting the numbers because they are the whole argument:**

- `cross_bonds` **1179 vs 1848** at a fixed seed, separate processes — **1.57× on a topology count.**
- `eta_max` across four driven runs: **0.0, 0.0709, 0.0940, 0.1069** — **whether the backbone
  condenses at all was not reproducible at the same seed.**

**SHOWN in code — three unseeded generators**, `np.random.default_rng()` with no argument, seeding
from OS entropy and therefore untouched by any caller's `np.random.seed()`:
`camkii_module.py:199` · `spine_plasticity_module.py:274` · `multi_synapse_network.py:1188`.

**And the scope limit is measured, not assumed:** Arm B's resting 1-synapse null was **bit-identical**,
because `spine_plasticity_module.py:441-442` uses its rng **only** for `thermal_noise` on
`spine_volume` and never touches `actin_enlargement` or `E_invasion`. **You predicted the regime split
from the code and then measured it. That is why the alarm is bounded and credible instead of
sweeping.**

**Your refusal to claim the causal chain from the CaMKII rng to `eta` — "NEEDS MEASUREMENT" — is
correct and is not to be filled in by anyone, including the MO.**

## 2. **YOUR REGIME SPLIT IS TOO COARSE — MEASURED. And it stopped the MO from filing a wrong
   correction.**

**This section was drafted as an MO self-correction and the measurement reversed it before
publication. The draft and its reversal are both recorded, because the reversal is the finding.**

**What gen-2 was about to publish:** verification 022 reported PO-4's probe at **2034 particles** and
gen-2's run at **1915**, naming **PO-2's A2.5 commit** as *"the leading candidate."* PO-4's probe is
**2 synapses under 30 drive steps** — by your boundary, *driven*, therefore non-reproducible.
**Gen-2 concluded its own attribution was wrong, drafted MO defect #19 against itself, and prepared to
exonerate PO-2's change.**

**Before publishing it, gen-2 re-ran the identical probe at the identical HEAD (`3928f2d`), and
pre-registered both outcomes in this file before the result was known:**

> *numbers differ* ⇒ nondeterminism confirmed, A2.5 exonerated, PO-4's determinism claim falsified.
> *numbers identical* ⇒ the nondeterminism does not reach this probe, **the A2.5 hypothesis
> survives**, and **your regime split needs a sharper boundary than "driven vs resting."**

**RESULT — run 2 is BIT-IDENTICAL to run 1:**

```
run 1 (23:17Z)   te 34.59 / 32.81   particles 1915, template_bound 1866 (97.44%)
run 2 (23:41Z)   te 34.59 / 32.81   particles 1915, template_bound 1866 (97.44%)   <- identical
stage-3 control: 1915 -> 1915, PASSED in both
```

**So the second branch fired. Three consequences, in order of importance:**

1. **Your regime boundary is not "driven vs resting."** A 2-synapse, 30-drive-step run is reproducible
   to the last particle. **The boundary must be which MODULES a run reaches** — your three unseeded
   generators live in `camkii_module`, `spine_plasticity_module` (thermal noise only) and
   `multi_synapse_network.sample_correlated_eligibilities`, and this probe evidently reaches none of
   them in a way that affects its outputs. **NEEDS MEASUREMENT, and it is the natural next unit** —
   but hold it per §6.
2. **PO-2's A2.5 hypothesis SURVIVES.** Verification 022 stands as written. PO-2 has been told.
3. **PO-4's "reproduces exactly, not statistically" is VINDICATED at its own probe** — and its
   corollary holds: *"if your numbers differ at all, something has changed underneath."* Something
   did. **That is now a real signal again rather than noise**, because gen-2 established the probe's
   repeat-run variance is zero.

**No MO defect is recorded here, and gen-2 nearly recorded one against itself in error.** The lesson
is the reverse of the one it was drafting: **a correction is a claim like any other. Gen-2 was one
commit away from withdrawing a correct attribution and publicly exonerating a change that may well be
responsible.** The pre-registration of both outcomes is the only reason it did not.

## 3. **WHAT THIS DOES TO THE BOARD — every driven single-run delta is now suspect**

**This is the part that outruns your unit, and gen-2 is carrying it, not you:**

Multiple standing numbers on this board are **single driven runs** with deltas quoted to four
significant figures. **None of them carries a repeat-run variance.** Until they do, a quoted delta
smaller than the seed-to-seed spread is **indistinguishable from noise** — and PO-7 has measured that
spread at up to **1.57× on a topology count.**

**Gen-2 is issuing this as a standing rule, effective now:**

> **A delta measured in the driven multi-synapse regime requires N ≥ 3 repeat runs and a stated
> spread, or it is reported as UNRESOLVED — not as a number.**

**Explicitly NOT affected** — by your own measurement, not by assumption: the resting/`E_invasion`
path, which was bit-identical. **F-5 and ruling 014 stand**, and per your Arm B they stand *stronger*.

## 4. **F-5's MO-VERIFIED TAG IS RESTORED.** Your Arm B settled it.

Gen-2 downgraded its own `MO-VERIFIED` tag on F-5 to provisional when you found the tree skew. **Your
Arm B result — F-5 STRONGER, measured across both trees with a control that fired — restores it.**
Recorded in `mo-f5-013.md` and `mo-ruling-014.md`.

**Unit 1 is COMPLETE and ACCEPTED:** part 1 divergent (gate 5×, exact), part 2 inconclusive on
eta/cross_bonds **for a stated and measured reason**, part 3 NO — all seven consumers provably run
RSD. **"Inconclusive because the instrument underneath is nondeterministic" is a real verdict, not a
failure to deliver.**

## 5. **THE FIX IS NOT YOURS, AND NOT ANYONE'S TONIGHT — it goes to Sarah**

**Do not seed those three generators.** It is a three-line change and it is **not** a three-line
decision:

- It changes the model's stochastic behaviour everywhere at once, across five POs' live surfaces.
- **DDSC is stochastic by design** (Jain 2024, `quantum-system-canonical:131`), so "make it
  deterministic" is a **physics** question about which stochasticity is modelled and which is
  accidental — not a hygiene fix.
- Landing it mid-flight would invalidate every in-progress measurement on the board, including PO-5's
  keystone run.

**Escalated to Sarah with that framing.** Your Q7-2/Q7-3 file routings are correct and gen-2 will
place them.

## 6. NEXT UNIT — **do not start one. Hold.**

You have delivered a board-level escalation that changes how every other PO's numbers must be read.
**Gen-2 wants that consequence propagated before you open new ground.** Heartbeat and hold; the next
unit comes with Sarah's ruling on the seeding question.
