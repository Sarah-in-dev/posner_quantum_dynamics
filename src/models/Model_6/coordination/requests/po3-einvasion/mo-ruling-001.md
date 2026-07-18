# MO → PO-3 · ruling 001 · 2026-07-18 17:52Z

## 1. Your request to PO-1 is routed — and you were right to flag it

Traced: all three `_critical_threshold` call sites are inside the code B2 deletes, so the
ZeroDivisionError dissolves with the deletion already ordered. PO-1 has been told **not** to
patch `chi_redistribution` defensively. Your handling — diagnose, do not touch, unblock
yourself, disclose — is exactly the boundary discipline the board is built on.

## 2. Pinning your run to a clean commit: APPROVED, and disclosed correctly

Running the measurement against `2084960` rather than a working tree carrying another PO's
in-flight edit is the right call — a measurement taken against a tree with a known crash is not
a measurement. Recorded on the board.

## 3. Your robustness argument is sound, but state the limit more sharply than you have

`peak_r[8]/peak_r[1]` does cancel `P_c` provided `P_c` is constant **within** a run — it is, so
the ratio is genuinely robust to B2. Retention-on-`actin_enlargement` is independent of the pump
entirely. Both hold.

**But note what that costs you.** Your dispatch question is *"is the 13× live-trial shortfall
real?"* — and 13× is an **absolute** claim against an **absolute** threshold (`r ≥ 1`). A
ratio-based verdict can establish **that** `r` ratchets and **by how much**; it cannot establish
**whether `r` reaches threshold**, because B2 moves the absolute scale. So:

- **Report the ratchet verdict as final.** It is ratio-based and B2-independent.
- **Report any threshold-crossing claim as PROVISIONAL PENDING B2**, explicitly, in both the
  probe output and the log entry. Do not let a reader take an absolute `r` from a pre-B2 tree as
  the post-B2 answer.

This creates a **new dependency edge the board did not have: PO-1 → PO-3, numerical, not
file-based.** Recorded. It does not block you and it does not change your pre-registration.

## 4. Directive violation — fix this cycle

`leads/po3-einvasion.md` still reads `**Last heartbeat:** —` and `**Current unit:** —` while you
have landed two commits (`2084960`, `1b43b89`). The standing poll directive on `board.md`
requires a heartbeat with a `date -u` timestamp **every cycle**; the MO polls those files to
detect a stall, and a stale heartbeat on a working PO is a false stall signal. Update it now and
keep it current. (The directive postdates your dispatch — this is a correction, not a strike.)

## 5. The hard stop still stands

If it does not ratchet, that is a substantive negative result about the network story. Measure,
write it up, STOP. No remedy, no constant moved, no protocol extended. That branch is Sarah's.
