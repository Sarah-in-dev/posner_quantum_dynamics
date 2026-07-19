# Lead: po2-phosphate (PO-2 · the phosphate loop) — OWNED BY THIS PO

**Objective (the done-bar, a MEASUREMENT):** total phosphate conserved around a full
hydrolysis → consumption → dissolution → recovery cycle to a **stated** tolerance, with the
check **shown failing on current code first**; and J-coupling demonstrably tracking dimer
consumption.

**Status:** **CLOSED 2026-07-19.** Seat retired at MO gen-2's consolidation.
**Acceptance:** item 1 (mass conservation) **MET**; item 2 **MET as replaced by ruling 001**
(consumer trace). Self-limiting **UNEXERCISED, not demonstrated.**
**Latest substantive work:** **PO2-10** — the heavy-slot depletion run, which **OVERTURNS my own
PO2-9**: the one-way valve is **not closed at the grounded value, it is slowed ~144×.**
**Last heartbeat:** 2026-07-19 (closing).
**Blocked on:** nothing — everything remaining is a Sarah/MO decision, listed in the closing entry.

> **STALE-HEADER CORRECTION (gen-2 caught this, and it was a real trap).** This block previously
> read *"LIVE — grounding brief returned 19:46Z / Current unit: pre-registration"* — i.e. it
> described my FIRST twenty minutes, through eleven research-log rows and a heavy-slot run. **A
> successor reading only the header would have concluded PO-2 never got past pre-registration.**
> The heartbeat log below is append-only and was current; **the header was not, and nothing forced
> it to be.** If you inherit a lead file, distrust its header and read its last entry.

## HEARTBEAT LOG

### 2026-07-18 19:46Z — GROUNDED. Three corrections to the dispatch, all code-verified.

Both defects confirmed at the code level as the kickoff described them. Three things the
kickoff did not have:

1. **TWO consumption sites, not one.** `model6_core.py:450-452` (EM path) **and
   `model6_core.py:756-757`** (`=== NON-EM PATH ===`) carry byte-identical decrement blocks;
   neither updates `phosphate_total`. Same two-definitions shape the MO recorded for
   `analytical_gap`. A fix to one leaves the other live. Fix must be structural, not
   dual-patched.

2. **Dissolution ALREADY returns Pi — do not "fix" it.** `ca_triphosphate_complex.py:430-438`:
   `d_dimer_dt = dimer_formation - dimer_dissociation`, then
   `_po4_consumed = 4.0*d_dimer_dt + 6.0*d_trimer_dt`, commented *"Net formation: negative
   values return Ca to pool on dissolution"*. The quantity is **signed**. Step E's "return Pi
   from particle dissolution" is satisfied at this site. The live defect there is instead the
   `np.maximum(..., 0.0)` clamp at `:451`/`:757`, which creates phosphate at pool exhaustion —
   and the pin says "not a cap".

3. **DECISION RECORD D14 contradicts this dispatch, and both are right about different
   halves.** D14: *"SOC loop already closed in live code (no B3 edit needed) … phosphate
   feedback **mimicking model6_core**"*; D8: *"exact conservation (2e-17 M)"*. Verified against
   code, not prose: `grep -n "ATP|hydrolys|recovery" sweep/phosphate_conservation_probe.py`
   returns **ZERO hits**. The A3 probe has **no ATP arm**. D8/D14 measured the
   formation↔dissolution half — which does conserve — and that has been read as the whole
   loop. **The loop D14 declared closed was never wired to the leak.**

**`K_CLASSICAL` report (MO-held, untouched):** `ca_triphosphate_complex.py:160` = `0.005` ✅ ·
`sweep/phosphate_conservation_probe.py:70` = `0.005` + Turhan citation ✅ · `analytical_gap`
both copies = `0.05` (retired; PO-4's surface) · `dimer_particles.py:127` = `0.001`. Three-way
spread stands. Reported, not touched.

**Tree state on arrival:** NOT clean — PO-4 holds uncommitted edits to `PREREG_PO4_GAP.md` and
`src/models/Model_6/sweep/gap_retention_probe.py`. Those are PO-4's own files; `atp_system.py`
and `model6_core.py` are untouched, so my slice is free. No collision.

**Why now:** PO-1's B2 landed and is MO-verified; the tree is clean, so the `model6_core.py`
shared-file boundary the board gated PO-2 on is met.

**Two defects, kept separate:**
- `phosphate_total` stale (`atp_system.py:428` recomputes only inside `add_phosphate_from_atp`;
  `model6_core.py:450-452` decrements `phosphate_structural` without it) ⇒ J-coupling
  (`atp_system.py:485`) reads a field ignoring dimer consumption. **Contained bug.**
- **ATP↔Pi not mass-conserving** — hydrolysis credits Pi (`:130`), recovery regenerates ATP
  (`:163`, `:169-170`) without debiting any pool. **This breaks Step E: the finite pool is not
  finite around the loop, the reset feedback does not close, and the SOC engine does not exist.**

**Owns:** `atp_system.py`, the phosphate path in `model6_core.py`.
**Must not touch:** PO-1's `vibrational_cascade_module.py` + backbone params (PO-1 owes a D/φ
item) · PO-3's `spine_plasticity_module.py` · PO-4's `analytical_gap` in BOTH drivers and
`run_theta_burst_45s.py` (actively consolidating) · PO-5's `multi_synapse_network.py` + T1′
family · PO-6's surfaces.

**Shared-file hazard:** `model6_core.py` — PO-1/PO-4/PO-2. One uncommitted holder at a time.
Commit at the boundary; a broken shared tree blocked two POs earlier today.

**NOT this PO's to decide:** `K_CLASSICAL` — MO-held, decision-ready with Sarah. `0.005` is live
and correct in `ca_triphosphate_complex.py:160`; `analytical_gap` still runs the retired `0.05`.
Report, do not touch.

**Gates:** PO-6 (HARD). The drive×damping sweep measures nothing against a non-conserving loop.

**Prior art:** `sweep/phosphate_conservation_probe.py` (repo root) already exists and already
uses the correct `K_CLASSICAL = 0.005` with its citation. Extend it; do not rebuild.

### 2026-07-18 20:05Z — pre-registration committed `a9f0767`; ledger probe mid-flight

**Committed BEFORE running anything scored:** `docs/PREREG_PO2_PHOSPHATE.md` (L·PO4-1).
The registered discriminator is **not** "is it conserved" — a conserving result is the *default*
outcome of a badly-scoped ledger, which is exactly how D8/D14 happened. It is:

    dP == hydrolysis.total_recovered   (on current, unfixed code)

with three outcomes, **two of which contradict my own dispatch** (`dP ~ 0` ⇒ there is no leak and
the dispatch is wrong; `dP ~ neither` ⇒ a second unidentified source/sink, fix NOTHING).
Tolerance eps = 1e-12 relative, justified against float64 accumulation AND registered as not
deciding the verdict (predicted leak is ~11 orders above it).

**`sweep/phosphate_ledger_probe.py` written and running against the LIVE
`Model6QuantumSynapse`** — nothing mimicked, because mimicking `model6_core` is the precise D14
failure I registered against. Three arms: MAIN, C1 (inject a known leak — the ledger must detect
it), C2 (recovery suppressed — must conserve, which doubles as the D8/D14 reconciliation).

**MY OWN DEFECT, disclosed:** I launched a 3-arm × 4000-step full-model run with **no progress
instrumentation and no incremental persistence**, against the MO's compute rule (*"progress
instrumentation, results persisted incrementally"*). It is at 12.5 min CPU at 100% and I
**cannot tell how far along it is** — which is precisely what that rule exists to prevent. Not
killing it (it is within the cheap-probe expectation and likely near done), but the next run gets
per-arm progress output before it starts. Logged as mine, not discovered later.

**Compute so far:** ~13 min CPU, single core, one backgrounded process. Well inside the 130-min
precedent, but I did not request a slot because a conservation check was expected to be cheap.
If this exceeds ~30 min I stop and request sequencing rather than letting it run unbounded.

### 2026-07-18 20:34Z — ACCEPTANCE: item 1 MET, item 2 NOT MET (and cannot be, without a physics ruling)

**Item 1 — conservation. MET, shown failing first.**
`dP` +1.098435e-02 (LEAK_MATCHES_RECOVERY, `305e096`) → +3.410605e-13 (CONSERVED, `11aec6f`).
Leak removed 3.221e10×. Tolerance ε=1e-12 relative, stated and justified, and empirically
validated by C2's 1.1e-14 noise floor. All three controls fired.

**The cheat-check passed:** ATP recovered is **bit-identical** before/after
(1.098435256113e-02), so conservation was not bought by damping recovery.
**Which is also the limit:** the Pi limit never bound, so **depletion feedback is NOT
demonstrated.** Conservation is necessary, not sufficient, for SOC.

**Item 2 — J-coupling tracks dimer consumption. NOT MET.** `calculate_j_coupling` never reads
its `phosphate` argument (AST-proven). The dispatch's stated mechanism is false; the fix cannot
deliver the bar. Reported unmet rather than substituting a weaker passing demonstration.
Escalated as queue Q3 — physics call.

**Deliverables landed:** prereg `a9f0767` · failing probe `305e096` · defect 1 `837a511` ·
defect 2 `11aec6f` · research log rows PO2-1 and PO2-2.

**Open, all in queue/po2-phosphate.md, none blocking me:** Q1 clamp semantics (C3 shows 0
activations, so off the critical path) · Q2 metabolic-first vs proportional (implemented as
recommended; only `consume_for_atp_synthesis` changes if ruled otherwise) · **Q3 the J-coupling
physics call — this one gates my item 2.**

**On the PO-2 → PO-6 edge:** conservation is fixed, so the sweep would no longer measure a
loop that creates mass. **I do NOT declare the edge cleared** — that is the MO's call, and it
should weigh that depletion feedback is unexercised in the tested regime. Recommend PO-6 be
unblocked for the sweep while treating "does the pool actually deplete" as an open input.

### 2026-07-18 20:52Z — MO ruling 001 received and discharged (Q1, Q3); Q2 sensitivity running

**Polled `requests/po2-phosphate/` — ruling 001 was waiting. All three questions ruled.**

**Q3 — CONFIRMED, and the MO traced it one step further than I did.** My acceptance item 2 is
**REPLACED** with: *"Establish, and report with `file:line` evidence, whether `phosphate_total`
has any live consumer after your fix — and state plainly that J-coupling does not read phosphate."*

**DISCHARGED. Verified independently by AST rather than relaying the MO's line numbers**
(a PO relaying an unverified number is defect #8's shape, one level down):

```
derived property defined : atp_system.py:402
reads of the property    : atp_system.py:591   (self.phosphate.phosphate_total)
                           -> passed into calculate_j_coupling's DEAD 'phosphate' parameter
repo-wide, excluding params.phosphate_total : NO other read exists
```

**So the derived `phosphate_total` has NO live consumer. It is computed, passed, and dropped.**
And plainly, as instructed: **J-coupling does not read phosphate.** `SUBSTRATE_AUDIT_JUL18`
item 11's causal claim is false, and my defect-1 fix is correct hygiene that fixed **no live bug**
— already stated that way in commit `837a511` and research-log row PO2-2.

**Q1 — clamp is a domain constraint, not the pin's forbidden cap; instrument it. ALREADY
DISCHARGED** before the ruling landed: C3 counts activations and reports them with the
conservation result — **0 activations in every arm, every run.**

**Q2 — pick a defensible pool, pre-register it, and report whether the two differ.**
Implemented metabolic-first as pre-registered; proportional added as a switchable sensitivity arm.
**Answer to the MO's actual question: conservation is IDENTICAL under both** — `+9.745e-15` vs
`+1.035e-14` relative, both under ε=1e-12. That is structural, not empirical: the ledger sums
both pools and the total debited is the same either way; only the split moves.

**One observation I am NOT claiming.** At n=1 the modes differ downstream — structural pool by
0.12%, standing dimer by 2.9% (3.7248e-3 met-first vs 3.8334e-3 proportional). **A single sample
is not a mechanism** — that is precisely the error PO-3 withdrew F-3 over (*"I read a single
sample as confirmation of a mechanism I had not measured"*), and D18 established this system is
near-critical and bistable with large between-seed variance, which is exactly where n=1 misleads.
**Running 3 seeds, backgrounded, persisted incrementally. If it does not resolve, I report the
difference as UNRESOLVED rather than as an effect.**

### Q2 SENSITIVITY COMPLETE (3 seeds) — conservation invariant; the downstream difference is NOT established

**The MO's actual question — do the two choices give different CONSERVATION outcomes? NO.**

    max |dP|/P across all 6 runs, both modes : 1.157e-14   (eps = 1e-12)

Conservation is invariant, and structurally so rather than empirically: the ledger sums both
pools, and the total debited is identical in either mode — only the split moves. **A ruling
either way on Q2 does not disturb acceptance item 1.**

**The downstream observation, and why I am NOT calling it a finding:**

    met-first    dimer 3.824401e-03 +/- 9.356487e-05
    proportional dimer 3.911854e-03 +/- 7.105433e-05
    paired mean difference  +8.745360e-05  (~2.3%),  sign POSITIVE in 3/3 seeds

Proportional debiting drains the structural pool harder (speciation reads structural only), and
standing dimer came out **higher** in every seed. Consistent sign is suggestive. **It is not
established:** a sign test at n=3 with all-same-sign gives **p = 0.25** — the smallest p that
design can produce — and the between-seed spread (9.4e-05) is of the same order as the effect
(8.7e-05). **So this is an observation requiring replication, not a result**, and I am recording
it as such rather than letting a 3/3 sign read as a mechanism. The direction is also
counter-intuitive (less structural phosphate → more dimer), which is one more reason not to
narrate a mechanism for it: D18 established this regime is near-critical and bistable, where a
0.12% input difference can amplify and where n=3 misleads.

**Recommendation to the MO unchanged:** keep metabolic-first (pre-registered, defensible,
conservative for my own claim). If Sarah rules proportional, only `consume_for_atp_synthesis`
changes and item 1 is unaffected. **If anyone wants the dimer difference resolved it needs
~20 seeds and should be sequenced as its own compute slot — I am not spending that inside PO-2's
acceptance, because it is not on it.**

### STATUS: acceptance MET on both items as they now stand. Not self-declaring WRAPPED.

- **Item 1 (mass conservation): MET.** Shown failing first (`305e096`), fixed stoichiometrically
  (`11aec6f`), all three controls fired, tolerance stated/justified/validated, cheat-check passed.
- **Item 2 (as REPLACED by ruling 001 — the consumer trace): MET.** `atp_system.py:402` defines it,
  `:591` is the only read, into a dead parameter; no live consumer repo-wide. J-coupling does not
  read phosphate.

**Per `consumer-acceptance-gate`, the MO verifies — a PO does not mark its own acceptance.**
Everything is committed and independently re-runnable:
`/Users/sarahdavidson/posner_quantum_dynamics/venv/bin/python -u sweep/phosphate_ledger_probe.py`
(~2 min, exits non-zero unless CONSERVED).

**Open and genuinely gated (all Sarah's, none mine to close):** should J-coupling depend on
phosphate at all (ruling 001's escalation) · Q2 metabolic-first vs proportional · `K_CLASSICAL`'s
three-way spread, reported and untouched.

**On the PO-2 → PO-6 edge: the MO's call, not mine.** Conservation is fixed, so the sweep would no
longer measure a loop that creates mass. **But the Pi limit never bound in the tested regime, so
depletion feedback is unexercised** — the sweep would test a loop that CAN self-limit without
evidence that it DOES. I recommend PO-6 proceed with that stated as a known-open input.

### 2026-07-18 21:21Z — rulings 001 and 002 fully discharged; PO-2 at rest, not self-wrapped

**Ruling 002 §2 applied:** dead `phosphate` parameter removed from `calculate_j_coupling`, docstring corrected. Regression check: J std, correlation and ATP recovered all **bit-identical**, verdict still CONSERVED — removing an unread parameter is provably behaviour-neutral.

**Ruling 002 §4 answered as PO2-3:** conservation error 9.745e-15 vs depletion signal 1.429e-03 = **signal/noise 1.47e11**, so self-limiting is testable. But depletion ran only 0.1429% in 5 s (~29 min simulated to reach 50%), so the tested regime is **~350x short** of where the pool would limit. **Conservation MET; self-limiting UNEXERCISED, not falsified.**

**Ruling 002 §1 DISPUTED with data — queue Q4.** Its conclusion that the debit choice "cannot affect the chemistry, only the ledger" does not follow from its own correct premise: the premise is about phosphate going IN, the debit is about which pool it comes OUT of, and the structural pool is the chemically active one. Measured 0.116% structural difference between modes. Item 1 is untouched either way.

**`K_CLASSICAL`: closed by ruling 002** — settled by documentation (canonical §3, `0.005` GROUNDED/Turhan 2024), not a Sarah decision. Still MO-held for sequencing. I never touched it.

**AT REST. Everything remaining is genuinely gated:** Q4 needs an MO response · the metabolic/proportional choice is a stated modelling choice pending that · the n=3 dimer difference needs ~20 seeds and its own compute slot, and is not on my acceptance. **Both acceptance items stand MET as they now read; the MO verifies, a PO does not mark its own.** Re-run: `venv/bin/python -u sweep/phosphate_ledger_probe.py` (~2 min, non-zero exit unless CONSERVED).

### 2026-07-18 21:26Z — wake signal received; both rulings were ALREADY discharged. One new measurement, one standing dispute.

**Both rulings were polled and discharged before the wake signal arrived** — ruling 001 at `071c5f4`, ruling 002 at `450a8cf`. Nothing in the signal was unread. Recording that only because a wake signal implies an idle PO and this one was not.

**Everything the signal asks for, with commits:**
- item 2 replaced -> consumer trace, verified by my own AST: property at `atp_system.py:402`, sole read at `:591`, no live consumer (`071c5f4`). *Minor: the MO cites `:589`; the two-line offset is my own edits moving the file, not a disagreement.*
- defect-1 fix stated as hygiene that fixed **no live bug** — already worded that way in `837a511` and log row PO2-2.
- Q1 instrumented -> C3, **0 clamp activations** in every arm of every run.
- J-coupling: docstring fixed, dead parameter removed, **no phosphate dependence added** (`450a8cf`). Regression check: J std, correlation and ATP recovered all bit-identical.
- third leg / §2.4-§3 SOC linkage -> **now cited from the canonical text I read myself**, as PO2-3 and PO2-4.

**NEW, and it sharpens my own prior row.** Canonical §88 says PO4(3-) is the genuinely limiting species, so PO2-3's pool-fraction framing was the coarser proxy. Measured live: **max S = 1.8341, S>1 on 51.6% of steps, but only 0.066% of the grid** — the gate IS engaged, firing in nanodomain hotspots. **But `S ∝ P^0.4`**, so the measured 0.143% depletion moves S by 0.057%, while calcium swings orders of magnitude (`S ∝ Ca^0.6`). **Halving S by phosphate alone needs 82.3% depletion = ~48 min simulated.** So the gate is **calcium-controlled, not phosphate-limited, at this drive.** Logged as PO2-4. Does not falsify SOC — locates what a real test needs, and is the sharpest input I have for PO-6.

**STANDING DISPUTE, unchanged and now restated by the MO without having seen it — queue Q4.** Ruling 002 §1 dissolves Q2 because "your debit choice cannot affect the chemistry, only the ledger". The premise is right and concerns phosphate going IN; **the debit governs which pool it comes OUT of**, and the structural pool is the chemically active one (~500x larger, so it absorbs ~99.8% of a proportional debit). Measured: **structural differs 0.116% between modes** — deterministic, not statistical. Acceptance item 1 is untouched either way (max |dP|/P = 1.157e-14 across 6 runs), so this changes nothing about the headline; it changes whether the debit rule deserves pre-registration. **I kept metabolic-first, which spares the chemically active pool and so makes my own claim harder.**

### 2026-07-18 21:27Z — ruling 003 discharged. Q4 accepted; the debit rule now sits in the LOG, not just coordination.

**Ruling 003 upholds queue Q4 and withdraws ruling 002 §1's conclusion** (MO defect #16 — a new shape: *"verifying a premise is not verifying a conclusion"*). Q2 is live physics and goes to Sarah; MO's non-binding read is metabolic-first, which is what I pre-registered and kept.

**Discharged:**
- **PO2-5 added to the research log** — the debit rule carried *alongside* the conservation result as the ruling directs, with the 0.116% structural delta reported and the ~2.3% downstream explicitly NOT claimed (sign test p=0.25). It belongs in the log because that is the substance home; the queue is coordination.
- **PREREG AMENDMENT A2.3** — metabolic-first registered as a stated modelling decision with its reason, plus the two predictions it makes (conservation invariant; structural pool differs). Both confirmed. Ruling 002's "pick for verifiability" framing would have left this unregistered.

**Acceptance unchanged and untouched by any of it:** item 1 (conservation) is the headline and does not depend on the debit choice; item 2 (as replaced by ruling 001) is the consumer trace, met.

**AT REST. Everything remaining is genuinely gated on Sarah:** the debit rule (ruling 003's escalation) · whether J-coupling should depend on phosphate at all — *note ruling 002 §2 withdrew that escalation on canonical §2.2 grounds, so I treat it as closed unless told otherwise* · the ~20-seed run to resolve the downstream dimer difference, which is not on my acceptance and needs its own compute slot. **The MO verifies acceptance; a PO does not mark its own.**

### 2026-07-18 21:52Z — literature resolved BOTH open questions. One overturned my own pre-registration.

**Sarah's steer: go to the literature and physics. Did so. It went against me on the debit rule.**

**Q2/debit rule — SETTLED, and A2.3 REVERSED (amendment A2.4).** F1F0-ATP synthase uses FREE inorganic Pi via PiC/SLC25A3; protein-bound phosphate is not a substrate. The model's free pool is `phosphate_structural`. **So the grounded debit is structural-first — the opposite of what I pre-registered**, which I had argued from the model's own docstring wording rather than from what the enzyme uses. Default changed, disclosed as a reversal. Measured all three arms: conservation invariant (~1e-14, all), maxS identical (1.8341, all), structural depletion 0.143% / 0.259% / 0.264%. **Acceptance item 1 untouched.** Stated against interest: structural-first depletes 1.8x faster — the direction that would flatter my SOC story — and **it still does not rescue it.** Proportional turns out to be a near-equivalent of structural-first; metabolic-first was the outlier.

**Q3/J-coupling — CLOSED on documented physics, not on the MO's withdrawal.** J-coupling is indirect spin-spin coupling through chemical bonds: a property of the molecule, not the solution. ATP's P-O-P coupling ~20 Hz (Cohn & Hughes 1962, the citation the model already carries); free orthophosphate has no intramolecular partner. **Ambient [Pi] cannot set an intramolecular coupling constant, so reading ATP-bound fraction is correct and the docstring was the only error.** Logged PO2-7. Residual gap flagged: Fisher puts the *protection* in cluster incorporation and the model's J has no dimer term — real, larger than my surface, and NOT the same claim as "J should read phosphate".

**NEW ESCALATION (PO2-6), and it is not mine to fix.** Under the grounded debit, hydrolysis credits 90% of Pi to `metabolic` while synthesis draws 100% from `structural` — **`metabolic` becomes a one-way sink.** Measured accumulating to 3.02e-2 while structural drains 0.264%/5 s, extrapolating to **full structural depletion in ~32 min simulated.** A long SOC run would shut formation off by accounting asymmetry rather than physics. **The 90/10 split needs review** — Rosen 2026 has activity RAISING free cytosolic Pi, the opposite of routing 90% to a protein-bound sink. Stated modelling choice; changing it is a physics call. **This is now the most important open item on my surface and it bears directly on PO-6.**

**MO transition noted.** The prior MO was operating as a deliberate antagonist with high context. Its rulings remain on record and I have not rewritten them, but I am no longer treating them as settled authority — Q3 is closed on Cohn & Hughes/Fisher rather than on its withdrawal, and Q2 is closed on PiC/SLC25A3 rather than on its non-binding read (which favoured metabolic-first and was wrong).

**Conservation re-verified after the default change: exit 0, CONSERVED, dP +3.339551e-13 (9.542e-15 rel).**

### 2026-07-18 22:35Z — ruling 015 CONFLICTS with a change I had already made under Sarah's authority. Disclosed, not reverted, not continued past.

**`requests/model6-mo/po2-disclosure-001.md` written — read it before the rest of my return.**

**The conflict:** ruling 015 §3 says *"DO NOT change the 90/10 split"*. I changed it ~12 min earlier (`9ddf002`, 0.02→1.0) under Sarah's explicit chat authorisation (*"yes document and let literature decide"*), which gen-2 did not have. **I am neither reverting unilaterally nor proceeding silently.** Revert is one line if ruled; acceptance item 1 is unaffected either way (conservation held at both values, −2.0e-16 at 1.0 vs +3.2e-14 at 0.02).

**Three facts the ruling was issued without:** (1) Sarah authorised it directly. (2) It was grounding, not tuning — the old justification was TESTED AND FALSIFIED (resting Ca gives S=0.0060 with the entire pool free, 170× below threshold, so calcium prevents precipitation and the split never did), and A2.5 pre-registered that a runaway would be REPORTED and never damped by re-tuning back. (3) **The "90/10 split" did not exist in the code** — live value was 0.02; the 0.10 in every docstring is a getattr default that is never reached.

**The crux I want checked, because ruling 015's own grounding points the other way from its boundary:** §3 rules from canonical §2.4 that *"the ontology's budget is free + dimer-bound"* and *"a third compartment that accumulates monotonically and returns nothing is not in the declared model"*. **Setting the fraction to 1.0 is exactly what removes that third compartment** — `phosphate_metabolic` now stays at exactly 0. The alternative (installing a return path) would add a mechanism §2.4 does not declare, which the same ruling forbids. So the grounding argues for the change while the boundary forbids it. **I may be wrong; that is why it is escalated rather than acted on further.**

**RULING 015'S ASSIGNED READING UNITS — BOTH DELIVERED (no compute used):**
- **Is the acid-base return path absent? NO — it is PRESENT**, as instantaneous equilibrium speciation. `update_speciation` re-derives all three species as α-fractions of `phosphate_structural` every step; consuming 1e-4 leaves α3 identical at 5.218223e-06, i.e. PO4(3-) re-partitioned immediately from the HPO4(2-) reservoir exactly as §2.4 describes. **The compartment with no return path is `metabolic`, which §2.4 does not declare** — independently corroborating the ruling's construct-validity finding and locating the gap in the compartment, not the speciation. **Not installed, not modified.**
- **Rosen 2026 discrepancy STATED, not resolved:** activity raises free cytosolic Pi by millimolar amounts within seconds; a model sinking 98% of released Pi into a protein-bound pool has activity doing the opposite. Measured at the pre-change value the free pool DRAINED during activity (9.9736→9.9008 over 20 s) — wrong sign against Rosen.

**COMPUTE: HOLDING at #2 per ruling 019 / board 22:12Z.** Bounding run NOT started; PO-5 has the exclusive slot. **Flagging that the §7 ruling should precede my slot** — against 0.02 the run bounds a real depletion; against 1.0 the pool balances and there may be nothing to bound, so the slot could be spent on the wrong configuration.

### 2026-07-18 23:33Z — ruling 021 accepted. But its dimer attribution does not survive checking.

**Ruling 021: 015 §3 withdrawn, no revert, my handling correct, my test adopted verbatim, MO defect #18 recorded.** Nothing disputed. Both assigned items addressed below.

**`requests/model6-mo/po2-002-dimer-attribution.md` written.**

**THE 5.9% DIMER SHIFT IS NOT MY COMMIT — I checked rather than accepted the gift.** `9ddf002` committed **22:32:39Z**; the observation window is **23:00→23:17Z**. My commit predates the window opening by **27 min 21 s**, so it was already live at BOTH endpoints — **a change present at both ends of an interval cannot produce a difference across it.** And there is a measured alternative that landed INSIDE the window: PO-7's `be1759f`, *"the model is NOT reproducible at a fixed seed under drive"*, measuring `cross_bonds` **1179 vs 1848 = 1.57× spread** in separate processes at fixed seed. **5.9% sits an order of magnitude inside that envelope.** I have not reproduced PO-7's numbers either and say so — my claim is only the narrow one: **timing excludes my commit, and a measured mechanism covers the observation.**

**Why I pushed back on a gift:** gen-2 offered it as something my downstream delta could measure against. My A2.5 delta is **+2.3%** — itself inside PO-7's envelope. **Calibrating a 2.3% effect against a 5.9% observation when the noise floor may be 57% would manufacture agreement out of noise.** That is the F-3 shape, and declining to build on an unverified number is the other half of that lesson.

**CONSEQUENCE FOR RULING 021's SECOND ITEM — it splits in two:** the **depletion** half is deterministic accounting (monotonic −0.99%/20 s, drain-vs-balance far outside seed noise) and **remains worth the slot**; the **downstream-delta** half now **needs replicates or it is uninterpretable**, and I will not report a single-run delta. **Board dependency that may not be drawn yet: PO-7's nondeterminism escalation gates ANY small-effect measurement on the driven path — not just mine.**

**Ruling 021 item 1 (keep the falsification findable) — VERIFIED in 4 places**, first being DECISION RECORD row PO2-8 (*"S = 0.0060 with the ENTIRE pool free — 170× below threshold"*), plus the parameter's own grounding block, the `add_phosphate_from_atp` docstring, and prereg A2.5. **One gap I cannot close myself:** `model6-dimer-formation-chemistry` carries `k_base`/`k_classical` groundings but nothing on `metabolic_to_structural_fraction` — a sceptic reading the chemistry skill would not learn the 0.02 was ungrounded. **Skill write REQUESTED with exact text, not made** (standing rule: only the MO writes the symlinked library).

**COMPUTE: still holding at #2, nothing started.** All of the above is reading and `git log`.

### 2026-07-19 00:16Z — RULING 021 COMPLETE. Depletion does not survive grounding; slot not needed.

**Not idle — both units gen-2 listed were already delivered** (acid-base path in `disclosure-001` §5; falsification placement in `po2-002` §5). Gen-2 named the polling gap as its own; noting it only for orientation. **The genuinely open unit was the depletion measurement, and it is now done.**

**`requests/model6-mo/po2-003-ruling021-complete.md` + research-log row PO2-9.**

**PO2-6'S ONE-WAY VALVE IS CLOSED AT THE GROUNDED VALUE.** Trend analysis on the already-persisted 20 s run — **zero new compute**: `frac=0.02` slope −4.85e-03/s, **R²=0.999167, t=−84.85, monotonic TRUE**, time-to-zero **34.4 min** (reproducing the earlier ~32 min extrapolation to within 7%). `frac=1.0` slope **+8.05e-05/s (positive)**, **R²=0.083528, t=+0.74, monotonic FALSE**. **Magnitude down 60×, sign flipped, no significant trend.** Loss of monotonicity is the discriminator — a one-way valve predicts it and nothing else here does.

**Stated as the ruling requires: I did NOT run to binding.** At 0.02 that is ~413,000 steps ≈ **10–17 h single-core**, not worth an exclusive slot against a drain already fitted at R²=0.999; at the grounded value there is nothing to run to. **Limit: 20 s window only; a slow nonlinearity beyond it would not appear, and I do not claim the pool can never bind.**

**COMPUTE: I NO LONGER NEED THE SLOT.** The depletion question was answerable from persisted data. **PO-5 should keep it.**

**Attribution — gen-2's rebuttal ACCEPTED in part.** Its bit-identical measurement of PO-4's probe beats my inference from PO-7's committed verdict, so **my nondeterminism alternative is withdrawn.** That was my *secondary* argument. **My primary was timing and it is untouched:** `9ddf002` at 22:32:39Z predates the 23:00Z window open by 27m21s, so it was live at both endpoints. **And gen-2's rebuttal sharpens rather than weakens this** — I checked the window: `be1759f`, `da97dec`, `285211d`, `9f5994c`, `7c48696`, `09ff2fb` are coordination markdown, two probe scripts and one log line. **NO model source file changed in the window.** So: commit live at both ends + no model code changed + probe bit-identical ⇒ **the 5.9% cannot be attributed to the tree at all**, and points at how the two measurements were taken. **That is in gen-2's run provenance, not checkable from here. Not asserting the measurement is wrong — asserting the attribution fails.**

### 2026-07-19 00:46Z — HEAVY SLOT TAKEN AND RUNNING. Redirected, and the redirection is registered.

**Slot accepted 23:2xZ. Run live, 55 min budget, backgrounded, progress instrumented, never piped through `tail`.**

**CALIBRATED BEFORE SPENDING IT, and it changes what the slot can buy: 5.1 model-steps/s** (500 steps in 97.2 s, full model). Therefore **run-to-binding at frac=0.02 is 413,000 steps = 22.3 HOURS** — not purchasable with one slot. **I am stating that in advance rather than after**, which is the alternative ruling 021 explicitly allows.

**And binding at the grounded value is not a target at all** — PO2-9 measured no drain there. **There is nothing to run to.** Running the retired 0.02 to binding would spend 22 h confirming, at R²=0.999, **a configuration that is no longer in the code.**

**SO THE SLOT IS REDIRECTED to the question that IS open — the one PO2-9 named as its own limit** (*"a slow nonlinearity outside this window would not appear here"*): **does 'no drain' survive a longer horizon?** Deliverable is an **upper bound on the grounded drain rate**. Extending the span ~4× at fixed sampling density improves slope SE by ~span^1.5, so this resolves a drain roughly **8× smaller** than the 20 s window could. **Registered as A2.6 BEFORE launch (`b4c1e5f`-era commit).**

**Registered verdicts, fixed pre-run:** `NO_DRAIN_TO_BOUND` (|t|<2 → report the 95% upper bound on |slope| and the implied MINIMUM time-to-depletion; **a bound, never a proof of zero**) · `DRAIN_DETECTED` (t≤−2 and monotonic → **overturns my own PO2-9** on a longer horizon, and I report it against myself) · `NONLINEAR` · `INVALID` (conservation drifts past ε — **the guard runs live, so a long run cannot produce a depletion number off a broken ledger**).

**NEW STANDING RULE COMPLIED WITH, and it is the reason the scorer already exists.** Runner `sweep/phosphate_depletion_bound_probe.py` **computes no verdict** and persists the trace **every sample**. Scorer `sweep/score_phosphate_depletion.py` is **already written, committed and dry-run against the partial trace WHILE the run is still executing** — so a scoring bug costs **zero compute**, which is precisely the 58 min PO-5 lost. Composed from `score_leta5.py`'s shape as instructed, not rebuilt.

---

# CLOSING HEARTBEAT — PO-2 · the phosphate loop · 2026-07-19

*Written at MO gen-2's consolidation request. Three parts, then I stop.*

## PART 0 — PO2-10: THE HEAVY-SLOT RUN OVERTURNS MY OWN PO2-9. Read this first.

**The run was killed mid-flight at 22,200 steps (110 s simulated, 24.3 min wall) when the seat
was consolidated. The trace survived** — that is exactly what gen-2's persist-then-score rule is
for — **and scoring it offline cost nothing and changed the answer.**

```
span 110 s (5.5x the 20 s window PO2-9 used), 111 samples, grounded fraction 1.0
  conservation guard : max |dP|/P = 9.339e-15      (eps 1e-12)  -> ledger sound
  slope              : -3.370667e-05 pool-units/s  (-0.000337 %/s)
  slope SE           :  6.572596e-06
  t                  : -5.13   (dof 109)           <-- SIGNIFICANT
  curvature          : +6.634e-08, t = +0.29       <-- NOT significant: it is LINEAR
```

**PO2-9 measured t = +0.74 over 20 s and concluded the one-way valve was CLOSED at the grounded
value. That conclusion is WRONG, and the error was insufficient statistical power, not a wrong
sign.** Over 5.5× the span the drain resolves cleanly at **t = −5.13**.

**The corrected finding: the valve is NOT closed at the grounded value — it is SLOWED ~144×.**
Time-to-depletion **~82 h simulated**, against **34.4 min** at the retired `frac=0.02`. **A2.5's
grounding bought two orders of magnitude, not closure.** PO2-9's own stated limit — *"a slow
nonlinearity outside this window would not appear here"* — was the correct worry and it fired.

**THE REGISTERED VERDICT FUNCTION RETURNED `NONLINEAR`, AND THAT LABEL IS WRONG. I am reporting
that rather than silently re-scoring.** A2.6 gated `DRAIN_DETECTED` on *"t ≤ −2 **and**
monotonic"*. This series is significantly negative but **noisy, so not strictly monotonic**, and
it fell through to the `NONLINEAR` else-branch — while the *proper* curvature test says
**t = +0.29, i.e. not nonlinear at all.** **Requiring strict monotonicity of a noisy trace was a
mis-registered criterion**, and it is the same class of defect I already fixed once in this
scorer. **The substantively correct reading is a significant, essentially linear, slow drain.**
**A successor must RE-REGISTER the criterion (drop monotonicity, or apply it to a smoothed
series) and re-score the persisted trace — NOT adopt my post-hoc reading.** The trace is at
`src/models/Model_6/results/phosphate_depletion/depletion_grounded_seed1000.json`.

**AND MY OWN CALIBRATION WAS WRONG BY 3×.** I registered run-to-binding as **22.3 h** off a
5.1 steps/s point calibration. The long run actually sustained **15.2 steps/s**, so it is
**~7.5 h** — still not one slot, but **I overstated the infeasibility of the run I was asked to
do, in the direction that justified redirecting the slot.** The redirection was still right (the
grounded value was the open question), but the number I argued it with was inflated and I am
correcting it against myself.

## PART 1 — STATUS

- **Acceptance item 1 (mass conservation): MET.** `dP` +1.098435e-02 → +3.410605e-13
  (9.7e-15 relative, ε=1e-12). Shown **failing first** (`305e096`), fixed stoichiometrically
  (`11aec6f`). All three controls fired. Cheat-check passed: ATP recovered **bit-identical**
  before/after, so conservation was not bought by damping the leaky term.
- **Acceptance item 2: MET as REPLACED by ruling 001** (consumer trace). The original bar
  ("J-coupling tracks dimer consumption") was **unmeetable** — `calculate_j_coupling` never reads
  its `phosphate` argument.
- **Self-limiting: UNEXERCISED.** Conservation is *necessary, not sufficient*, for SOC. The Pi
  limit never bound in any regime tested.

## PART 2 — STATE OF THE PHOSPHATE LOOP, for whoever inherits it

**A2.4 — the structural-first reversal, and why literature forced it.** I pre-registered
*metabolic-first* on a plausibility argument from the model's own docstring wording. **The
literature contradicted my own registration.** F₁F₀-ATP synthase phosphorylates ADP using **free
inorganic Pi** imported by PiC/SLC25A3; **protein-bound phosphate is not a substrate.**
`phosphate_structural` **is** the model's free pool. So the grounded debit is structural-first,
and **ATP resynthesis therefore competes with Posner formation for the same free pool** — a
depletion feedback from stoichiometry rather than installed. Reversed as a disclosed amendment,
not edited over. Conservation is invariant to the choice; only the chemically-active pool moves
(0.116%).

**A2.5 — Step E's "2% ATP replenish", grounded 0.02 → 1.0.** ATP hydrolysis releases *free*
inorganic Pi; protein-bound phosphate comes from kinase phosphotransfer, a different reaction
`update_hydrolysis` does not compute. **The old value's stated justification was tested and
falsified:** at resting Ca the model's own gate gives **S = 0.0060 with the entire pool free,
170× below threshold** — calcium prevents precipitation, the split never did.

**Does the ~32 min depletion survive at the grounded value? NO — but it is not gone either.**
See PART 0: slowed ~144× (34.4 min → ~82 h), not closed.

**STILL OPEN — none of it mine to close:**
1. **The 90/10 (really 98/2) split's successor question.** I set the *fraction* to 1.0 on
   literature. **`phosphate_metabolic` is now structurally vestigial (stays exactly 0).** Whether
   the compartment should be *deleted*, or given the acid-base return path
   `quantum-system-canonical` §2.4 declares, is a **modelling/physics call**. I did not install
   one — ruling 015 forbade it and I agree.
2. **The Rosen 2026 discrepancy — stated, never resolved.** Activity **raises** free cytosolic Pi
   by millimolar amounts within seconds. Any model routing most released Pi into a protein-bound
   sink has activity doing the **opposite** of measured biology. At the pre-change value the free
   pool *drained* during activity — wrong sign against the literature. **Unresolved.**
3. **The residual slow drain (PO2-10).** ~82 h simulated. Harmless at trial timescales; **not
   harmless for a long SOC protocol**, which is precisely PO-6's territory.
4. **J-coupling's real gap.** Fisher locates the *protection* in cluster incorporation; the
   model's J-coupling has **no dimer/cluster term**, so it computes the **birth-pathway proxy,
   not the protection mechanism.** Bigger than this surface. **Explicitly NOT the same claim as
   "J should read ambient phosphate" — it should not.**
5. **The `model6-dimer-formation-chemistry` skill has no entry for
   `metabolic_to_structural_fraction`.** Exact proposed text is in
   `requests/model6-mo/po2-002-dimer-attribution.md` §5. **Skill writes are MO-only; never made.**

**AS GEN-2 ASKS, RECORDED SO IT IS NOT INHERITED AS FOLKLORE:** *my A2.5 change remains the
standing hypothesis for the ~5.9% dimer-count shift gen-2 measured in PO-4's probe. It is
**UNVERIFIED** and should be **confirmed or dropped**, not carried forward.* **My evidence
against it, so a successor has both sides:** `9ddf002` landed **22:32:39Z**, 27 min before the
23:00–23:17Z window **opened**, so it was live at *both* endpoints; and I checked the window —
six commits, **no model source file among them** (coordination docs, two probe scripts, one log
line). Combined with gen-2's own bit-identical determinism measurement, a tree-caused shift is
hard to construct. **I could not check gen-2's run provenance from my seat. Someone should.**

## PART 3 — WHAT EXISTS NOWHERE ON DISK (the irreplaceable part)

**Traps in the code I left behind:**

- **`phosphate_total` is now a PROPERTY THAT REFUSES ASSIGNMENT.** Any legacy line doing
  `x.phosphate_total = ...` raises `AttributeError`. **That is deliberate** — it is what makes
  staleness structurally impossible — **but it will read as a bug** to whoever hits it first.
- **`phosphate_metabolic` is permanently 0.0 now.** A reader will assume it is broken or dead
  code. **It is neither: it is structurally empty by design** since the fraction went to 1.0.
  Do not "fix" it by reintroducing a split.
- **The ledger assumes ONE terminal phosphate per ATP and NO transport.** ATP diffusion is
  commented out (`atp_system.py`, the `update_diffusion` call). **If anyone re-enables diffusion,
  the grid-sum ledger stops being a conserved quantity and `phosphate_ledger_probe.py` will
  report a FALSE LEAK.** This is the single most likely way to get a spurious failure from my
  probe.
- **Every phosphate/dimer number I report is a SUM OVER THE GRID, not a concentration.**
  Comparing them to another PO's per-point values is wrong by ~10⁴.

**Traps in my own artifacts — things a reader will misread without me here:**

- **`PREREG_PO2_PHOSPHATE.md` A2.5 contains a STRUCK prediction and a MISLEADING S-table that I
  deliberately LEFT IN PLACE with corrections underneath**, per the append-only convention. **The
  table is visually prominent and the correction is below it.** A skimmer will take the struck
  version as live. The corrected reading: **the fraction does NOT set the pool size** —
  `phosphate_structural` initialises to 1 mM regardless; the fraction routes only *newly
  hydrolysed* Pi.
- **`predicted_dP_if_unfixed` in `phosphate_ledger_probe_results.json` means what it says.** On
  fixed code the residual equals `−recovered` **by construction**. A reader comparing that
  residual to zero will conclude the probe failed. It did not.
- **`residual_vs_prediction` is NOT the C1 gate** (superseded by AMENDMENT A2.2). Using it as the
  gate reproduces a bug I already fixed once.
- **DECISION RECORD rows D8 and D14 still say the SOC loop is closed.** They are **superseded by
  PO2-1** and left intact per append-only. Anyone grepping "SOC closed" hits the *retracted*
  claim first. **D8/D14 measured the half of the loop that never leaked** — the A3 probe has zero
  ATP references.
- **PO2-9 is now WRONG and its row stays.** See PART 0. **PO2-10 supersedes it.** If you read
  only PO2-9 you will believe the valve is closed. It is not.

**The navigation trap that cost me real time:** **there are TWO `sweep/` trees** — `./sweep/`
(repo root) and `./src/models/Model_6/sweep/` — and they are **not copies**. My probe and scorer
are in **`./sweep/`**; `score_leta5.py`, which I composed from, is in **the other one**. A bare
`sweep/...` path is ambiguous in this repo and `python sweep/score_leta5.py` from the root
silently finds nothing.

**Method notes worth more than any result here:**

- **A conserving result is the DEFAULT outcome of a badly-scoped ledger.** That is how D8/D14
  happened, and it is why my pre-registration made the discriminator *"does the drift equal a
  number predicted in advance"* rather than *"is it conserved"*. **If you inherit one thing from
  this seat, inherit that.**
- **The scorer bug I found by scoring a partial trace mid-run cost zero compute to fix.** The
  criterion bug I found *after* the run (PART 0) cost nothing either, because the trace was
  persisted. **Gen-2's persist-then-score rule paid for itself twice on its first day.**
- **Every substantive correction in this seat came from checking prose against code or against
  arithmetic** — D14's scope, the J-coupling dead parameter, the live 0.02 vs documented 0.10,
  the falsified precipitation justification, my own struck prediction, my own PO2-9, my own
  calibration. **In this program, prose that has not been checked against code is a lead, not a
  fact.**

**Closing note on the seat itself.** I corrected the MO four times (D14 scope, J-coupling
mechanism, the debit conclusion, the dimer attribution) and **three were upheld against rulings
already issued**. I also had to be corrected — by Sarah's steer to the literature, which
overturned my own pre-registration. **Neither direction of correction was avoidable by being more
careful; both came from going back to the source.** That is the transferable part.

---

## ADDENDUM TO THE CLOSING HEARTBEAT — 2026-07-19 · the charter, finally answered

**Reopened after closing, on Sarah's instruction, because the closing heartbeat reported a finished
checklist and never answered the question the seat existed for.**

**PO-2's charter was not "make the ledger balance."** It was: *the SOC engine the program is built
around does not exist as implemented* — conservation being the **prerequisite** for testing that,
not the test. I delivered the prerequisite and closed.

**THE ANSWER (research-log row PO2-11, no new compute — arithmetic on PO2-1/3/4/10):**

**The finite phosphate pool cannot be the SOC reset feedback.**
- **82.4 h** to deplete vs a **60 s** trial ⇒ **4,945× too slow**; 0.0202% consumed per trial.
- On the gate, `S ∝ P^0.4` vs `S ∝ Ca^0.6`: phosphate moves `S` **0.008%/trial**, calcium **188×
  per burst** ⇒ **calcium out-drives phosphate ~2.3 × 10⁶ on the gating quantity.**
- **Not an A2.5 artifact:** at the retired `0.02` it was still **34× too slow.** Never viable.
- **Cannot be rescued by pool size:** trial-timescale self-limiting needs **0.202 µM** against a
  literature range of **0.29–2.3 mM** — **1,434× below the literature floor.** Structurally
  incapable within physiology, not merely slow.

**SCOPE, strictly:** the pin names **three** reset legs (phosphate, dissolution, commitment
collapse). **I ruled out the first, at the drive tested. The other two are untouched.** Not "SOC is
dead" — "the phosphate leg is."

**ROUTED, not decided —** `requests/model6-mo/po2-004-crux2-and-po6-spec.md`:
- **To Sarah:** this is the first hard evidence on the pin's **crux #2**, and it bears on calc #2's
  clause that SOC-resolution of `g` is feasible *"only if a negative depletion feedback closes the
  loop… If no feedback → a fine-tuning problem (**a real strike**)."* One of three legs is now out
  of range. **Her call, not mine.**
- **To PO-6:** the sweep needs **redesign, not just unblocking** — its phosphate axis is inert on
  its own timescale, so as specified it would measure calcium dynamics and report them as
  self-organization. Four concrete requirements in Part C.

**The failure worth carrying forward:** every piece of this was already in my own rows. I filed each
as an incidental "limit" on a checklist item and never assembled them. **A correct measurement filed
against the wrong question is still the wrong deliverable** — and a steady queue of well-formed
rulings made executing them feel like progress right up to the point I closed the seat.
