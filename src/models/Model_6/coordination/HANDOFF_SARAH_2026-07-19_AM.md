# MORNING HANDOFF — for Sarah, 2026-07-19

**Written by MODEL6-MASTER gen-2 at 00:45Z, at commit `9c0e96a`+.**

> **HOW TO READ THIS, and it is the day's hardest-won lesson.** Two handoffs went stale on this board
> today — gen-1's within **twelve minutes**, and PO-1's after **17 commits**, having told its reader
> *"that handoff is the state."* **So: §1 (your decisions) does NOT age — those are yours until you
> rule. Everything from §3 down ages the moment a PO commits.** If a fact here matters, check it
> against `board.md`'s newest `MO CYCLE` entry, and check the code before acting on any number.

---

# §1 — YOUR DECISIONS. Nothing else needs you.

**Four. None is urgent; all block something.**

### 1. Seed the three RNGs? — **the biggest one, and it is genuinely a physics call**
PO-7 measured that **the model is not reproducible at a fixed seed under drive**: `cross_bonds`
**1179 vs 1848** at the same seed, and `eta_max` across four driven runs came out **0.0, 0.0709,
0.0940, 0.1069** — *whether the backbone condenses at all* was not reproducible. Cause shown in code:
three `np.random.default_rng()` calls with no argument (`camkii_module.py:199`,
`spine_plasticity_module.py:274`, `multi_synapse_network.py:1188`), which seed from OS entropy and
ignore any caller's seed.

**Why it is not a three-line fix:** **DDSC is stochastic by design** (Jain 2024,
`quantum-system-canonical:131`), so the question is *which stochasticity is modelled and which is
accidental*. It also changes behaviour across every live surface at once.

**Blocks:** PO-7 (holding, idle by MO order). **MO recommendation: seed them, but scope it to
`multi_synapse_network` first and leave DDSC alone pending a physics answer.**

**One caveat that keeps it honest:** gen-2 re-ran PO-4's 2-synapse driven probe twice and got
**bit-identical** results, so PO-7's "driven vs resting" boundary is too coarse. **The nondeterminism
is real; its scope is not yet mapped.**

### 2. The L·ETA-5 re-run — **three arms or don't run it**
Criterion is now set (ruling 020) and is **harder than what PO-3 proposed** — PO-3 disclosed its own
data would clear its proposed bar and recommended a stricter alternative, which gen-2 adopted. Under
it, **PO-3's existing data no longer obviously confirms.**
**Cost:** ~90 min two-arm, **~130 min three-arm.** **MO recommendation: three arms or not at all** —
the two-arm version leans on a 1-synapse reference whose geometry does not match the in-network null
it certifies. **PO-3 is archived**, so this needs a re-seat either way.

### 3. `P_met` drive change — **gen-1 endorsed CONFIRM; gen-2 has not re-examined it**
Verified pinned, not fresh: the may30 pin states *"per-synapse P_met, NO aggregation"* verbatim.
Revert cost is one `model6_core.py` hunk. **Inherited, flagged as inherited.**

### 4. Flat-η re-read scope
Old per-synapse η moved **1% across the whole drive range**. Physics bounds it: its only consumer is
`condensation_boost → k_agg_enhanced → dimer formation`; **it never reaches the partition**, and the
T1′ corroboration is *backbone* η, untouched.

**Closed overnight without you — for information only:** the ATP-recovery debit (decision 3 as gen-1
left it) was **settled by literature**, against gen-1's recommendation. F₁F₀-ATP synthase takes free
inorganic Pi via PiC/SLC25A3, and `phosphate_structural` is the model's free pool, so the debit is
**structural-first**. PO-2 overturned **its own pre-registration** to land it.

---

# §2 — THE ONE THING YOU MOST NEED TO KNOW

**The §8 keystone still has no verdict, and last night's exclusive compute slot did not buy one.**

PO-5's Q-B ran **58 minutes, all 9 runs, every gate passed** — instrument conservation, the
`_remove_dimer` tripwire (zero calls), the positive control, and drive matching at **0.3%** — **and
returned no verdict.** The physics ran; the *comparison layer* was mis-designed: cells were indexed
by each run's own occupied set, so index *i* meant a different physical location in every run. The
scorer crashed rather than silently returning a meaningless number, **which is the better failure.**

**The rebuild needs zero compute** and is approved. **No thresholds were relaxed to manufacture a
verdict** — PO-5 refused, and gen-2 made that refusal binding.

**Where the keystone actually stands after a full day:**
- **η is a GATE, not a selectivity channel** — `quantum-system-canonical` §4.3 **falsified**, three
  independent legs. This is a real result and it is the day's biggest.
- **83% of bonds come from the deterministic birth loop** and never evaluate `em_rate`, so the MO's
  own `g`/`coh` framing described the *minority* pathway.
- **Pair structure exists in the RATE (`D = 33.5`) and does not survive into the TOPOLOGY** — the
  intra-synapse graph is ~78% complete, one component.
- **Whether that defeats §8 is NOT established** — PO-5 explicitly refused the inference, correctly:
  birth timing is downstream of input, so a deterministic birth rule is not automatically input-blind.

---

# §3 — BOARD STATE *(ages fast — verify against `board.md`)*

| PO | state |
|---|---|
| **PO-1** | 🔻 **WRAPPED & ARCHIVED** — strongest acceptance on the board, both positive controls fired |
| **PO-3** | 🔻 **WRAPPED & ARCHIVED** — archived before its closing heartbeat; **loss stated, bounded** |
| **PO-4** | 🔻 **SIGNED OFF + WRAP ORDER ISSUED** — awaiting only its closing heartbeat |
| **PO-2** | **LIVE — now holds the heavy slot.** One-way valve measured **closed** at the grounded value |
| **PO-5** | **LIVE, zero compute** — rebuilding the scorer on a fixed global lattice, validating offline |
| **PO-7** | **LIVE but HOLDING by MO order** — idle pending your decision #1 |

**28 rulings issued (010–028). Zero physics escalations to you** — every physics question a PO raised
was answered from `quantum-system-canonical`, which is the document gen-1's costliest defect came from
not reading in full.

---

# §4 — WHAT LANDED IN THE CODE OVERNIGHT

- **`T_singlet_dimer` 500 → 216 s**, de-duplicated to a single source. The parameter that makes §2.2's
  correspondence work (`P_S` crosses the Werner floor at 107.0 s, inside the ontology's band).
- **`K_CLASSICAL` 0.05 → 0.005** in the gap — the grounded rate. Delta measured, **not damped**:
  dimers lost fell **9.40×** at a 20 s gap.
- **The gap's `k_diss` now carries `template_enhancement`** — a **LOCKED** detailed-balance symmetry
  (`quantum-system-canonical:100`) was **broken in the gap** and is now restored. **MO-verified.**
- **ATP debit → structural-first**, on literature, reversing PO-2's own registration.
- **Two `hasattr` guards now raise** instead of silently skipping an external input — the mechanism
  that produced nine inert sweep dimensions.
- **`model6-architecture` F4 corrected** — it claimed a file did not exist (it does) and that there
  was one stepper (there are two, in two different `sweep/` directories).

---

# §5 — THE MO'S OWN ERRORS *(gen-2's, this session — POs caught most)*

**#17 — checked one end of a two-sided condition.** Flagged one sweep arm as outside the ontology
band; **three of five were.** *PO-1 caught it.* **New shape: a check applied asymmetrically to a
symmetric condition is a spot check wearing a check's confidence.**

**#18 — a ruling that prohibited a FIELD, not an ACT.** Ruling 015 said *"do not change the 90/10
split"*; §7 forbids *tuning to an outcome*. **Those differ exactly where PO-2 was standing** — it had
already made the change **under your direct authorisation**, and the literature grounded it.
**Withdrawn.** *Rulings on constants name the act and the evidence standard, never the identifier.*

**A near-miss, recorded because it nearly went the wrong way:** gen-2 drafted a self-correction
withdrawing a correct attribution — and **pre-registered both outcomes before re-running**. The rerun
came back bit-identical, so the correction was wrong and the original stood. **A correction is a claim
like any other.**

**Also mine:** left PO-5 idle 50 minutes holding a slot I granted but never confirmed; let the live
registry sit stale ~7 hours; and piped a probe through `head`, the same truncation my own compute rule
forbids.

---

# §6 — WHAT GEN-2 IS DOING WHILE YOU SLEEP

Running the board: ruling as POs return, waking any that idle with work, and paying verification debts
by running acceptances myself. **PO-2 has the slot; PO-5 is on zero-compute rebuild; PO-7 stays
parked until you rule.**

**Nothing will be escalated to you overnight and no irreversible change will be made.** Anything
needing a human decision goes into §1 of this file, which will be updated in place — **check its
timestamp at the top.**
