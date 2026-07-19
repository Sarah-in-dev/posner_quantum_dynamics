# PO-2 → Sarah (crux #2) and the MO/PO-6 (redesign spec) · 004 · 2026-07-19

**Filed after my closing heartbeat, because that heartbeat reported a completed checklist and never
answered the question the seat existed for.** Sarah named the gap directly. This file closes it.

**No new compute. Everything here is arithmetic on measurements already in the research log
(PO2-1, PO2-3, PO2-4, PO2-10) plus the may30 pin's own text.**

---

## PART A — THE FINDING (research log row PO2-11)

**The finite phosphate pool cannot be the SOC reset feedback.**

| quantity | measured |
|---|---|
| depletion rate, grounded value (PO2-10, t = −5.13) | 3.3707e-05 pool-units/s |
| time to full depletion | **82.4 h simulated** |
| program trial timescale | 60 s |
| **ratio** | **4,945× too slow** |
| pool consumed per trial | 0.0202% |

**On the quantity that actually gates formation** — `S = (Ca³·[PO₄³⁻]²/Ksp)^0.2`, so `S ∝ P^0.4`
and `S ∝ Ca^0.6`:

- phosphate moves `S` by **0.008% per trial**
- calcium (rest 100 nM → 616 µM nanodomain) moves `S` by **188× within one burst**
- **calcium out-drives phosphate on the gating quantity by ~2.3 × 10⁶**

**Formation is switched off by the calcium transient ending, not by phosphate running out.**

**Two things that make this robust rather than a configuration artifact:**

1. **Not caused by my A2.5 change.** At the retired `frac = 0.02` the pool depleted in 34.4 min —
   **still 34× slower than a trial.** The leg was never fast enough at either value.
2. **Cannot be rescued by shrinking the pool.** Depletion time = pool/rate, and the rate is set by
   ATP turnover *independent of pool size*, so the required pool scales linearly with the needed
   speed-up. Trial-timescale self-limiting needs **0.202 µM** against a literature free-cytosolic-Pi
   range of **0.29–2.3 mM** — **1,434× below the literature floor.** Doing it anyway would be tuning
   a constant outside its literature range to reach an outcome (§7 LOCKED).

**STRICT SCOPE — what this does NOT say.** The pin's SOC section names **three** reset mechanisms:
*"finite phosphate pool self-limits dimer formation; dissolution returns Ca/PO₄; commitment collapse
decoheres the condensate."* **I have ruled out the first, at the drive tested. The other two are
untouched by this PO.** This is **not** "SOC is dead." It is "the phosphate leg is, and the other two
are now the only candidates."

---

## PART B — FOR SARAH: this lands on crux #2 and triggers the pin's own strike clause

**Not a decision I am making. Routing it to the person whose call it is, with the pin's words.**

The pin reduces the whole network-layer feasibility to **two** cruxes. This is one of them:

> *"2. **The SOC feedback** — does the depletion loop (finite phosphate, dissolution, collapse)
> actually self-organize the bias to the edge (#2)?"*

And feasibility calc **#2** is the one that depends on it:

> *"The **write coupling g** … is NOT derived. So the absolute trigger magnitude is genuinely
> under-determined — unlike #1. **Resolution:** self-organized criticality removes the need to pin g…
> This is feasible **only if** a negative depletion feedback closes the loop (firing consumes the
> drive). **If no feedback → it reverts to a fine-tuning problem (a real strike).**"*

**One of the three legs of that feedback is now measured out of range by ~5,000×, and out of
physiological reach by ~1,400×.** That does not fire the strike clause on its own — two legs remain.
**But it is the first hard evidence on crux #2, and it removes the leg the pin lists first and the
one the chemistry was said to make "already latent."**

**What I recommend you decide:** whether the remaining two legs get tested before the program
continues to treat #2 as "resolved by SOC." If they also fail, the pin's own language says `g`
reverts to a fine-tuning problem and that is a real strike against the backbone-network story —
which the pin pre-committed to accepting *"without retuning."*

**I am not qualified to make that call and I am not making it.**

---

## PART C — FOR PO-6: the sweep needs redesigning, not just unblocking

**I gate PO-6, and I treated that as a binary for the whole seat. It isn't. Here is what PO-6
actually needs from me.**

**The problem with the sweep as specified.** The emergence test is *"Sweep drive rate × damping.
Real SOC parks at the edge ACROSS the range (drive-independent attractor)."* But **the phosphate
axis is inert on that sweep's own timescale** — 0.0202% per trial, moving `S` by 0.008%. **So a
drive×damping sweep over trial-length runs would measure calcium dynamics and report them as
self-organization.** It would produce a clean-looking result about the wrong mechanism.

**Four concrete requirements:**

1. **Run length must be matched to the depletion constant, not the trial.** The relevant timescale
   is **82.4 h simulated**, not 60 s. At the measured 15.2 steps/s that is ~7.5 h wall per point —
   so a full drive×damping grid at depletion timescale is **not affordable**, and that fact should
   shape the design rather than be discovered mid-sweep.
2. **Sweep the pool size explicitly.** `sweep_runner.py:77` already sweeps `q2_phosphate_initial`,
   so the axis exists. **But note Part A: the physiologically admissible range cannot reach
   trial-timescale self-limiting.** Sweeping it is still worth doing — to *demonstrate* the
   insensitivity rather than assert it — but it must be reported as a null, not tuned until
   something happens.
3. **Report the phosphate axis's own sensitivity alongside any SOC claim.** If `η`/`r` move while
   the pool moves 0.02%, the cause is not phosphate. **A sweep that does not carry this check can
   attribute a calcium effect to the SOC engine.**
4. **The other two legs are now the priority.** Dissolution return and commitment-collapse
   decoherence are the only remaining candidates for the reset feedback. **If PO-6 is testing SOC,
   those are where the signal must come from.**

**On the edge itself:** conservation is fixed, so the sweep would no longer measure a loop that
creates mass — **the hard block's stated reason is discharged.** But the sweep would test a loop
that *can* self-limit and, on the phosphate axis, demonstrably *does not* on any timescale it will
run at. **Unblocking is the MO's call; I am supplying the reason it should be unblocked with
conditions rather than cleanly.**

---

## PART D — why this was late, recorded so the pattern is visible

The acceptance bar was two measurable items, and rulings arrived steadily with well-formed next
actions. **Executing them felt like progress and was measurable, so I kept doing it.** Every piece
of Part A was already in my own rows — PO2-3 had the signal-to-noise, PO2-4 had the
calcium-vs-phosphate exponents, PO2-10 had the rate. **I filed each as an incidental "limit" on a
checklist item instead of assembling them into the answer to the question the seat existed for, and
I closed the seat without noticing.** A correct measurement filed against the wrong question is
still the wrong deliverable.
