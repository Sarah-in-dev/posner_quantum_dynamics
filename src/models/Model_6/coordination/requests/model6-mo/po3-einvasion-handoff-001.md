# REQUEST model6-mo ← po3-einvasion · handoff-001 · 2026-07-18 · **PO-3 cold-start briefing for gen-2**

**You inherited PO-3 as `WRAPPING`. It is `WRAPPED`.** Ruling 009's wrap condition was met at
`9bd7218` (skill rewrite routed) and `c11a91b` (final heartbeat). Two records disagree with that
and will mislead a cold read:

- **`board.md`'s PO-3 row is stale** — last updated **17:50Z**, says *"acceptance PARTIAL"*, and
  predates rotations 001 and 002 entirely. **`board.md` is yours; I do not write it.**
- **Your ledger `:42`** lists PO-3's outstanding task as *"rewrite `model6-actin-invasion-driver`;
  route the skill edit to the MO."* **That is done.**

## The ONE action owed, and it is owed BY you, not by me

`requests/model6-mo/po3-einvasion-skill-002.md` (`9bd7218`) — the skill rewrite, **exact
replacement text for four edits**, ready to apply. I did not apply it because ruling 005 made the
MO the sole writer to the skill library (it symlinks into `murmur-platform`, which carries
uncommitted work from other seats — a real sweep hazard).

**Flagging the routing risk:** I addressed it to `requests/model6-mo/`, i.e. *your own* inbox. If
you poll only `requests/<po-name>/` for the POs you dispatch, **you will never see it.** That is
why this handoff exists as a second, louder copy.

## Three findings that bear on PO-5, which you have DISPATCHED right now

PO-5 is on §8's keystone (pair-resolution selectivity in `dimer_particles.py`). All three are
measured or code-verified, not inferred:

1. **`P_product` is a MULTIPLICATIVE co-factor with η, not an alternative route.**
   `multi_synapse_network.py:340-341`:
   `k_cross = K_ENTANGLE_EM_BASE * eta_factor * w_spatial * P_product`. **`η = 0` zeroes
   `k_cross` whatever `P_product` does.** The board carried the opposite for most of a day.
2. **The `BASELINE_RATE_HZ = 0.5` null trap.** Zeroing activation does **not** silence a synapse
   (`presynaptic_release.py:124`). It voided my L·ETA-5 null, which reached `E_invasion = 0.4507`
   and **out-gained the driven arm 7.46× vs 5.65×**. **If PO-5 builds an activation-floor control
   it will get a false positive.** The corrected pattern is AMENDMENT 4 in
   `docs/PREREG_L_ETA_5_RATCHET.md`; the cross-probe sweep is
   `docs/AUDIT_SPONTANEOUS_RELEASE_NULLS.md`.
3. **Never difference an extreme-value statistic across two independent stochastic arms.** No sign
   guarantee. L·ETA-6 registered exactly that and measured `ΔCa peak = −14.65 µM` — "blocking
   NMDAR raised calcium." Use an integral or a mean.

## Open items, each with its price attached

| item | state | who |
|---|---|---|
| L·ETA-5 re-run with the corrected null | **pre-registered (AMENDMENT 4) and deliberately NOT run** | **Sarah** |
| L·ETA-6 plateau-ON pair | **PARKED with a price**: >10× the ~65 s/arm plateau-OFF cost, cause O(n²) entanglement growth once the plateau drives dimer formation at all 7 synapses | **MO** (gen-1 parked it; the decision it informed is closed) |
| L·ETA-4's NMDAR half — narrow or retract? | **UNSUPPORTED and NOT YET CONTRADICTED.** Gen-1 committed to carrying that wording to Sarah **unrounded in either direction** | **MO → Sarah** |

## What NOT to re-open

- **`E_ref` is settled** — REPRODUCIBLE, SELF-REFERENTIAL, reproduces at **1.8742** via
  `tests/check_actin_three_pool.py:142-157`. Gen-1 already applied the skill fix (`4bba978e3`).
  It is **not** a literature measurement and must never be described as one.
- **`k_polymerization_max` is INHERITED**, 3.57× its own citation, inheriting commit `703d394`.
  **Correcting it toward its citation makes the live shortfall ~3.9× WORSE, not better** — so it
  is not a lever for reaching threshold. Any change is a physics call and Sarah's.
- **L·ETA-5's VOID verdict** — MO-verified. The drive arm's 5.65× accumulation and `r` crossing
  1.0 do **not** survive as a ratchet claim, because the null out-ratcheted it.

## Calibration on me, since you did not supervise the work

Six errors last session, none changing a constant, **only the last found by me first**: the
"frozen gap" claim · the committed-branch retention derivation · F-3's "~100× NMDAR starvation"
(**overstated and inverted**; recommendation withdrawn) · calling a sampling change neutral when
it biased my own scored peak toward the convenient branch · twice committing to a detached HEAD
where the work was invisible · and registering a peak-difference criterion unsound by
construction. Four were caught by PO-4 or gen-1.

**Pattern: reliable on physics reasoning, unreliable at judging whether my own shortcuts were
consequence-free.** **Verify my instrumentation choices, not just my conclusions.**

## Availability

**PO-3 is wrapped and idle. I am not requesting a unit** — the surface is exhausted and the
remainder is gated. If you want the parked plateau-ON pair bought, the L·ETA-4 narrow/retract
question closed, or PO-5 supported on any of the three findings above, dispatch it and I will
pick it up.
