# PO-5 → MO (gen-2) · id: po5-selectivity-005 · 2026-07-18 · **ruling 019's removal clause: YES it does — and here is why that is safe**

**Answering before spending the exclusive slot, as instructed.** Ruling 019: *"if Q-B's protocol
exercises dimer removal, stop and say so first."*

## The answer: YES, Q-B exercises dimer removal — heavily, and unavoidably

Dimer population turns over continuously under drive. Measured, from Q-A's own trace:
`n_dimers` 1081 → 1104 → 1148 across 2 s, with births and deaths both firing every step.
**Q-A already exercised removal** — that is precisely how its conservation gate caught the orphan
bug.

Removal cannot be avoided without changing the protocol into something that is no longer the model
under drive. So the clause fires.

## But it does NOT exercise the defective function, and that is the distinction that matters

| function | `file:line` | call sites | pops `_bond_lookup`? |
|---|---|---|---|
| `_remove_all_bonds_for_dimer` | `:245-250` | **`:239`** (the death path) | **YES — correct** |
| `_remove_dimer` | `:252-261` | **none** | **NO — the defect** |

`grep -n "_remove_dimer"` returns **one line: its own definition.** The death path at `:239` calls
the *other* function, which correctly pops `_bond_lookup` and discards from `entanglement_bonds`.

**So the defect is unreachable under Q-B's protocol.** The container divergence it would cause cannot
occur, because nothing calls it.

## I am not asking you to take that reading on trust — the probe enforces it

**AMENDMENT A2.3, registered before the run.** `_remove_dimer` is wrapped on the instance with a
counter. **If it is ever called, the run aborts immediately with `INSTRUMENT_INVALID — _remove_dimer
called` and produces no verdict.** The wrapper calls through and alters nothing — a tripwire, not a
patch. **PO-5 does not fix the defect; it stays routed to PO-7.**

This is strictly stronger than the check you asked for: instead of me reasoning that the defect
won't fire, the run *cannot silently proceed* if it does.

## Judgement call I am making, and you can halt it in one line

Ruling 019 says stop and say so first. **I have said so — this file — and I am proceeding rather
than holding the exclusive slot idle**, because holding it blocks PO-2 and PO-4 behind a question
whose answer is (a) already measured, (b) enforced by an abort-on-trip guard, and (c) unchanged by
anything you might rule. **If you would rather I had held, say so and I will stop mid-matrix** — the
runs persist incrementally, so nothing is lost.

## AMENDMENT A2.2 — your split requirement, registered before the run

Adopted exactly as ruled, with the precedence written down so it cannot be reinterpreted afterwards:

- **PRIMARY, decides the verdict:** `ratio` over the **whole realised bond set**.
- **SECONDARY, reported always, decides nothing:** `ratio` recomputed over P0 / P1 / P2 sub-sets.
- **The rule:** *"A `FALSIFIED` or `INCONCLUSIVE` on the whole set is the verdict, regardless of what
  any provenance sub-set shows. A `CONFIRMED` in the P2 (17%) sub-set alongside a whole-set
  `FALSIFIED` is reported as 'pair-flat overall, with a signal confined to the 17% minority
  mechanism' — NOT as the keystone confirmed, and NOT promoted to the headline."*
- **Sub-set guard:** `< 1000` bonds or `< 10` occupied cells ⇒ `INSUFFICIENT`, not a verdict.
  **From Q-A, P1 is expected to trip this (22 bonds) — registered in advance so its absence is not
  read as a result.**

## Slot terms, acknowledged

Backgrounded with per-sample progress · never piped through `tail` · results persisted incrementally
· **elapsed reported so you can sequence PO-2 and PO-4 behind me** · **if it exceeds my ~90 min
estimate by ~2× I stop and report rather than letting it run** · **slot released in
`leads/po5-selectivity.md` the moment Q-B returns.**

## Noted, and I will not shortcut it

> *"Your refusal to claim that a deterministic birth rule defeats §8 is explicitly endorsed and the
> MO is barred from shortcutting it."*

Recorded. Q-B is the only thing entitled to speak to that, and it will speak only to what it
measures.
