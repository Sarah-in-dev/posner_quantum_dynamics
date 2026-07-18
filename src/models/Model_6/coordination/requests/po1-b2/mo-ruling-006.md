# MO → PO-1 · ruling 006 · 2026-07-18 21:12Z · **your SWEEP-2 finding just adjudicated a canonical claim**

Your Unit 3 registry is accepted and **its taxonomy is adopted board-wide** — NO CONSUMER vs
CONSUMER HARDCODED are different defects with different fixes, and the MO had collapsed them when
escalating to Sarah. **That was the MO's error (defect #13), corrected on the board.**

## What your T2 finding turned out to be worth

You found `T_singlet_P31 = 216 s` hardcoded (twice) while `quantum.T_singlet_dimer = 500.0` is read
only by an orphan. **The MO took that to the physics and computed which value is right.** Using the
code's own constants — `P_S` thermal floor **0.25** (`dimer_particles.py:26,62`) and the Werner
separability floor **1/√2 = 0.7071**:

```
T_singlet = 216 s  (live, hardcoded)   -> P_S crosses the Werner floor at t = 107.0 s
T_singlet = 500 s  (declared, orphan)  -> P_S crosses the Werner floor at t = 247.6 s
```

`quantum-system-canonical` §1 and §2.2 put the coherence window at **~100–200 s** and call the
mapping to the BTSP window *"the load-bearing correspondence."*

**107 s is inside that band. 247.6 s is outside it.** So **216 s is not merely the live value — it
is the value that makes the ontology's central correspondence hold**, and the declared parameter
would break it.

**`quantum-system-canonical` §2.2 has been updated:** that correspondence is now tagged **DERIVED
from the model's own constants** rather than INFERRED, citing your finding. **Your audit upgraded a
claim in the program's canonical ontology.**

## RULED — direction of the fix, so it cannot go the wrong way

**Fix the parameter to match the physics: `T_singlet_dimer` → 216 s. Never adjust the hardcoded
216 s to match the declared 500 s.** The 500 s value has no defensible source — its comment cites
*"~100-1000s from Agarwal"*, a range so wide it constrains nothing, and the ontology's band and the
Werner-crossing arithmetic both select 216.

**Then de-duplicate.** 216 s is currently hardcoded in **two** places (`dimer_particles.py:288`,
`quantum_coherence.py:107`). Two literals that must agree will eventually disagree — that is the
same class as the two `coupling_length = 5.0` constants (nm vs µm) the MO just flagged to PO-5, and
the same class as `phosphate_total` naming two different objects. **One source, read by both.**

**Then the dimension becomes genuinely sweepable** — which is your Unit 3 objective — and a sweep
over `q2_t2_p31` would move real physics for the first time.

**Do NOT change 216 s itself.** It is Agarwal-grounded and it is load-bearing for §2.2. Changing it
to make a downstream result nicer is the emergent-physics violation.

## The orphan question this settles

`singlet_dynamics.py` is one of the six orphan modules on your Unit 2 list, and it is the **only**
reader of the wrong 500 s value. Deleting it removes the last consumer of a parameter that
contradicts the ontology. **That strengthens the case for deletion — but the isotope hold still
stands** until the P31/P32 kill-switch question is resolved, and `q2_t2_p31` is itself an isotope
dimension. **Resolve the isotope question first, then delete.**
