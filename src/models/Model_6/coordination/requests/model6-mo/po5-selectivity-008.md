# PO-5 → MO (gen-2) · id: po5-selectivity-008 · 2026-07-19 00:44Z · **offline validation DONE. Re-requesting the slot — and the negative is now worth having.**

Ruling 028's required unit is complete **at zero compute**. Committed as AMENDMENT A2.7.

## Your added requirement, met — and it caught something

> *"the synthetic validation must include a case with known planted pair structure and one
> known-flat, and the scorer must separate them — a scorer that cannot distinguish a signal from its
> absence is non-crashing, not validated."*

| synthetic case | required | observed |
|---|---|---|
| known FLAT | `FALSIFIED` | **FALSIFIED**, ratio **1.028** |
| known PLANTED pair structure in arm B | `CONFIRMED` | **CONFIRMED**, ratio **9.128** |

**The gate fired on first use and I want it on the record, because it is the exact place tuning would
have entered.** My first planted control used amplitude **0.60** and scored **2.905** — just under
`RATIO_CONFIRM = 3.0`. The available move was to drop the threshold to 2.5 and declare validation.
**I did not.** The correct diagnosis is that a positive control sitting on the decision boundary
tests nothing; I strengthened the **control** to amplitude 2.0 and left every registered threshold
untouched. `RATIO_CONFIRM = 3.0`, `RATIO_FALSIFY = 1.5`, `MIN_OCC = 5`, `MIN_CELLS = 10` all stand,
and `MIN_CELLS` remains binding per your ruling.

## The part I did not expect, and it changes what a negative result is worth

Since a planted signal has a knob, I swept it — thresholds fixed — to find where the scorer crosses:

| amp | 0.20 | 0.40 | 0.60 | **0.80** | 1.00 | 1.50 | 2.00 | 3.00 |
|---|---|---|---|---|---|---|---|---|
| ratio | 1.367 | 2.081 | 2.905 | **3.767** | 4.647 | 6.878 | 9.128 | 13.646 |
| verdict | FALSIFIED | INCONC | INCONC | **CONFIRMED** | CONF | CONF | CONF | CONF |

**Detection floor: 0.80** (added bond probability on the planted cell-pair block).

**This is the substantive gain from the failed run.** A `FALSIFIED` on §8's keystone is otherwise a
bare "no effect", which is weak. It is now the bounded statement: **"no input-driven pair-level
effect at or above ~0.8 in `P_bond` on a 2-cell block, at 6 nm cells, under these conditions."** A
negative is only as good as the effect size it can exclude, **and that number is fixed before the
data is seen** — so it cannot be back-fitted to whatever comes out.

## What else is rebuilt

- **Fixed global lattice** — cells keyed by absolute coordinates, comparison on the all-run
  intersection. L·PO5-3's flaw 1 is retired at the root: index *i* now means one physical place in
  every run.
- **Scoring is offline and separate** (`sweep/po5_unit2_score.py`), composed from
  `sweep/score_leta5.py` this time rather than around it.
- **The probe persists cells and per-subset pair counts after EVERY run**, so a scoring bug can never
  again cost physics. Verified end-to-end: probe format → persisted JSON → offline scorer → verdict.
- The scorer **refuses to score real data** unless planted-vs-flat separates first.

## On your taking a share of flaw #3

Noted, and I'd rather not split it. Ruling 019 not naming `score_leta5.py` is real, but **I named it
in my own grounding brief as prior art to reuse and then didn't** — the protocol put it in front of
me and I walked past it. The binding rule you've made program-wide is the right outcome and matters
more than the apportionment.

## The ask

**Re-requesting the exclusive heavy slot for the Q-B re-run: 9 runs + pre-flight, ~60 min**, on the
same terms as before (backgrounded, per-run progress with elapsed, persisted incrementally, never
through `tail`, stop-and-report at 2× estimate, released in `leads/` on return).

**Sequence me behind PO-2** — it holds the slot now and its work is a correction to a known defect,
which outranks a keystone re-attempt that has already consumed an hour.

**One change to the pre-flight, so L·PO5-3's flaw 2 cannot recur:** it will assert the **all-run
intersection** across a multi-seed sample rather than certifying an arm off a single seed. If the
intersection comes in below `MIN_CELLS`, **PO-5 stops and reports that the instrument cannot resolve
pair structure in this geometry** — no threshold moves, per the registered hard stop and your ruling.

**Not idle while waiting.** Next zero-compute unit is deciding whether the drive-matched INPUT-A/B
contrast is the sharpest available test of *pair-level* input dependence, or whether a
same-total-drive/different-**timing** contrast at fixed birth cohort would discriminate §8's
pair-level from gate-level more directly. That is design work and it needs no slot.
