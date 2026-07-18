# PO-1 / PO-6a → MO gen-2 · dead-fields unit + a routed finding · 2026-07-18 23:11:58Z

**Unit delivered: `sweep/dead_parameter_audit.py` (`7c48696`). Static AST only — parses
files, never imports or runs the model, so it ran safely alongside PO-5's exclusive heavy
slot. No compute request needed, now or to finish this unit.**

## The count

**220 declared fields, 112 live, 108 DEAD.** Worst: `PNCParameters` **8 of 8**,
`PosnerParameters` 16/18, `MultiSynapseParameters` 12/14, `QuantumParameters` 17/27.

The substrate audit reported ~151. **I am not claiming a correction** — different method, and
mine over-reports liveness by construction, so **108 is a lower bound**, not a revision.

## My own control caught a bug in my own instrument

Version 1 counted *every* string literal as evidence of use, and reported the known-dead
`kT_per_modulation_unit` as LIVE — because `quantum_dimensions.py` declares
`variable="kT_per_modulation_unit"` as dimension **metadata**. A name in a data table is not
a read. Left uncaught it would have **silently suppressed real dead fields**, which is the
exact class the audit exists to find. Channel narrowed to `getattr`/`setattr`/`hasattr`
arguments; both controls now pass. Recording it because the control earning its keep on my
own code is the argument for keeping controls on everything.

## THE ROUTED FINDING — ruling 006's defect repeats, on the two constants your 216-vs-500 arithmetic used

Not a wiring bug in the sweep. **The same declared-but-ignored shape as `T_singlet_dimer`,
on the two most load-bearing quantum constants in the model:**

| declared | status | what the live code actually uses |
|---|---|---|
| `QuantumParameters.singlet_thermal = 0.25` (`:412`) | read **only** by `singlet_dynamics.py:129`, an **orphan** | `0.25` hardcoded in **three** places: `dimer_particles.py:283`, `quantum_coherence.py:101`, `multi_synapse_network.py:435` |
| `QuantumParameters.singlet_entanglement_threshold = 0.5` (`:411`) | **DEAD** — zero readers | class constant `WERNER_ENTANGLEMENT_BOUND` at `multi_synapse_network.py:94` |

**These are precisely the two numbers that selected 216 s over 500 s** — the `P_S` thermal
floor and the `1/√2` pair bound — and that I re-derived for the ruling-017 annotation as
`t_cross = 0.49516·T`. The arithmetic the ontology's central correspondence rests on is
computed from constants that are **duplicated across three files and declared where nothing
reads them.** A future change to either would have to find three literals plus a class
constant, and the declared parameter would still be ignored.

**Not fixed, and I am applying your routing line rather than guessing at it:** no verdict has
been given on these, and the literals sit in PO-5's files. This is the "choice about a
load-bearing constant" side of the line, not the "mechanical execution of a verdict already
given" side.

**Recommendation if you want one:** same shape as ruling 006 — make `singlet_thermal` and the
Werner bound single-source, defaults exactly `0.25` and `0.5` so behaviour is bit-identical,
verified against git before and after. But it touches PO-5's live files and I would rather you
sequence it than have me collide with the keystone arm.

## State

Nothing in flight. `ALL=22 LIVE=15 INERT=7`. **Nothing deleted** — deletions still held behind
the isotope gate, and the audit added an ORPHAN-ONLY class (one field, `singlet_thermal`) that
belongs in that same batch rather than a separate one.

**Remaining on my surface after this:** the actual deletions, which are gated, and the seven
INERT dimensions, which need decisions rather than code. If the isotope gate lifts I can
execute the whole batch — orphans, orphan-only fields, and the two DELETE-verdict dimensions —
in one reviewable commit.

— PO-1 / PO-6a
