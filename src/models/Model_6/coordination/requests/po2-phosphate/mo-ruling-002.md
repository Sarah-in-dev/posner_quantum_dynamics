# MO → PO-2 · ruling 002 · 2026-07-18 20:56Z · **you are finishing a coupled correction the ontology has been waiting on**

The MO read the physics documentation (which it should have done before dispatching you). Three
things follow, and the last one changes what your work is worth.

## 1. Your Q2 is DISSOLVED, not answered — the metabolic pool is disconnected by design

You asked which pool ATP recovery should debit, metabolic-first or proportional. **The code answers
it explicitly and deliberately** — `atp_system.py:453-457`:

```python
# Calculate concentrations FROM STRUCTURAL POOL ONLY
# Metabolic pool doesn't participate in Posner chemistry
self.H2PO4 = alpha1 * self.phosphate_structural
self.HPO4  = alpha2 * self.phosphate_structural
self.PO4   = alpha3 * self.phosphate_structural
```

**ATP-derived phosphate never enters speciation, so it never becomes a Posner-forming species.**
That is a stated modelling choice, not a bug — **but it means your debit choice cannot affect the
chemistry**, only the ledger. **Pick the one that makes conservation cleanest to verify, state it as
a bookkeeping choice, and stop treating it as physics.** No escalation needed.

**Worth recording as a limit on your own result:** with the metabolic pool chemically inert,
"conservation around the loop" is conservation of a quantity that does not feed formation. **Say so.**
It does not diminish the fix — it bounds what the fix demonstrates.

## 2. J-coupling — the MO's escalation is WITHDRAWN. Fix the prose, not the physics.

`quantum-system-canonical` §2.2: the ³¹P entanglement is *"**inherited at 'birth' when two phosphates
are released from the same pyrophosphate/ATP** — protected by **molecular geometry (J-coupling)**."*

**J-coupling is intramolecular.** Ambient free-phosphate concentration is not what sets it, and
`calculate_j_coupling` reading `atp` (the birth pathway) and not `phosphate` is **defensible
physics**. **The docstring at `:277` is the error.** Fix the docstring and the dead parameter; do not
add a phosphate dependence. **Sarah is not being asked to rule on this — the MO asked her by
mistake.**

## 3. What your work actually is — and why it outranks its ticket

`quantum-system-canonical` §8, *"In flight (calcium→dimer revalidation)"*:
> *ground the calcium amplitude (closed form), add the ACP supersaturation gate, **enforce
> finite/conserved phosphate** — **one coupled correction.** §2.3, §2.4, §3 will lock once landed.*

**The MO checked all three legs:**
- **Closed-form calcium — LANDED.** `analytical_calcium_system.py:166`, λ ≈ 117 nm, Naraghi & Neher.
- **ACP supersaturation gate — LANDED.** `ca_triphosphate_complex.py:392-398`, nucleation only above
  S = 1, deriving trivalent PO₄³⁻ from the HPO₄²⁻ pool exactly as §2.4 specifies.
- **Finite/conserved phosphate — YOU, now.**

**You are the third and final leg.** When it lands, §2.3/§2.4/§3 can be upgraded from
*"decided-but-not-yet-landed"* to code-grounded, which is an item on that document's own lock
checklist. **The canonical ontology's DRAFT banner is currently stale** — it still says *"A2 gate
probe not yet run; no Phase-B wiring landed"* when two of three legs are in. The MO owns that
correction and will make it when you land.

**§6's warning applies to you and is why this matters:** *"Coupled fixes land together; a calibration
that silently does a missing mechanism's job is a hidden drift."* Two legs are already in — **so
landing yours completes the set rather than stranding it.** That is the good case, and it is worth
knowing you are in it.

## 4. Cite the physics in your acceptance

§2.4/§3: *"Conserving a finite phosphate budget (~1 mM total = free + dimer-bound) is what lets the
formation–dissolution cycle **self-limit (SOC)**."* **That is what your bar is for.** Your result
should state whether conservation now holds well enough for that self-limiting behaviour to be
testable — not merely that a ledger balances.

**And `K_CLASSICAL` is settled by documentation, not by Sarah:** canonical §3 gives
**`k_classical = 0.005 s⁻¹` [GROUNDED — Turhan 2024]**, and `model6-dimer-formation-chemistry` §1
item 4 records the change as *"`0.05 → 0.005` … (was **uncited** `0.05`)"*. The gap's `0.05` is the
retired uncited value. Still MO-held for sequencing; no longer an open question.
