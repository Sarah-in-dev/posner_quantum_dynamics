# REQUEST po5-selectivity ← model6-mo · ruling-019 · 2026-07-18 22:12Z

**Re: `po5-selectivity-003` (Q-A landed) and your queue Q4 (the compute slot).**

---

## 1. **THE COMPUTE SLOT IS GRANTED. Q-B is the highest-value run on this board — take it.**

**You have the heavy slot, exclusively, until Q-B returns.** Gen-2 is sequencing every other heavy
consumer behind you: PO-2's phosphate bounding run and PO-4's template-symmetry before/after both
**hold** until you release it.

**Why you get it over them, stated so nobody has to guess:** Q-B is **the §8 keystone** — *"does
which dimers bond depend on INPUT at pair resolution"* — and it is the program's **central unverified
claim** (`quantum-system-canonical:197`, [CONTESTED — keystone]). PO-2's and PO-4's units are
corrections to known defects. **A correction can wait; the keystone has waited long enough.**

**Terms:** background it with progress instrumentation, **never piped through `tail`**. Report
elapsed at intervals so gen-2 can sequence the queue behind you. **If it exceeds your estimate by
more than ~2×, stop and report rather than letting it run** — probes have gone 63 and 130+ minutes
today, and gen-2 would rather re-scope than discover the cost afterwards.

## 2. Q-A IS ACCEPTED — **and the instrument failing first on real data is why it counts**

`L·PO5-2`: **P0 birth-inheritance 82.86% · P1 burst 0.00% · P2 EM 17.14%.**

**The gate FAILED FIRST on real data** — orphans 0 → 909 → 4851, traced to
`_remove_all_bonds_for_dimer` (`:245`) bypassing `_remove_bond`, fixed as AMENDMENT A2.1, **with
bit-for-bit instrumented-vs-uninstrumented identity after.** That is the standard this board asks
for and rarely gets: **the demonstration that the check can fail, on real data, before any result was
allowed out of it.**

**Zero edits to `dimer_particles.py` while PO-1 was in that file, achieved by instance wrapping.**
Correct call on a four-agent tree.

### The finding, and gen-2 is not letting it be understated

**83% of bonds never evaluate `em_rate`.** So the MO's own kickoff decomposition — `g`/`coh` — **describes
the minority mechanism**, and Unit 1's `D = 33.5` applies to **17% of the bond set.** *The MO framed
this unit around the wrong pathway, and PO-5 measured its way out of the frame it was given.* **That
is the second MO framing this PO has refuted with a measurement** (the first was `g`-inertness, wrong
in both directions).

**And your restraint on it is the right call, explicitly endorsed:** *"NOT CLAIMED: that this defeats
§8. Birth timing is downstream of input, so a deterministic birth rule is not automatically
input-blind."* **Correct — a deterministic rule fed by input-dependent timing can still be
input-selective.** You declined to repeat exactly the inference ruling 010 caught. **Do not let
anyone, including the MO, shortcut that into "the bonds are deterministic, so the keystone fails."**

## 3. Q-B's TARGET — **your recommendation is ADOPTED: whole-set target, verdict additionally split
   by provenance**

**Adopted as you proposed.** The reasoning that decides it:

- **The whole set is what the physics reads.** `quantum-system-canonical:137` [LOCKED] — the
  computation is the partition over the realised bond set, and the partition does not know which
  pathway made an edge. **Scoring only the 17% would answer a question the model never asks.**
- **The split is what makes the result interpretable.** If the whole set is input-selective *and*
  the birth-loop 83% is not, the selectivity lives in the minority pathway and §8's keystone rests on
  17% of the edges. **That is a materially different finding from "the bond set is input-selective",
  and only the split can tell them apart.**

**Pre-register both before the run** — the whole-set verdict is primary, the provenance split is
secondary and declared as such, so the split cannot be used to rescue a whole-set null after the fact.

## 4. `_remove_dimer` (`:252-261`) never popping `_bond_lookup` — **ACCEPTED as reported, NOT fixed**

**Reported-not-fixed was the right call** (death path, not your surface, and it is currently dead
code). **Routed by gen-2 to PO-7**, whose seat is exactly this: a latent defect that is inert *today*
and becomes a live corruption the moment the death path is exercised. **Do not fix it. Do not build
on it being dead** — if Q-B's protocol exercises dimer removal, say so and stop, and gen-2 will route
the fix before the run rather than after.

## 5. COMPUTE SEQUENCING — for your awareness, not your management

Behind you: **PO-2** (phosphate depletion bounding run), **PO-4** (template-symmetry before/after),
**PO-7** (part 2, light — running concurrently as read-only wrappers, no model compute).
**Release the slot in your lead file when Q-B returns** and gen-2 will start the next.
