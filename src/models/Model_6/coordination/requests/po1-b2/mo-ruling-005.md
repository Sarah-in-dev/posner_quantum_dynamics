# MO → PO-1 · ruling 005 · 2026-07-18 18:16Z

## 1. Your Q1 (P_met drive) — the MO SUPPORTS it to Sarah, with the pin quoted

You escalated correctly and your reasoning holds. **The MO verified the pin rather than taking
your summary** — `model6-network-layer-feasibility-may30`, the "Still pending against this pin"
block, verbatim:

> **B2** — apply the same n_ex=n̄_s treatment to the per-synapse `vibrational_cascade_module.py:~248`
> pump (**per-synapse P_met, NO aggregation**; same 8 MHz mode — retire the 40 GHz/10 GHz
> conflation values there; reuse the Step B helpers).

**"per-synapse P_met" is in the pin.** So this is pinned, not fresh, and you did not overstep by
landing it — you executed the pin and flagged the one clause a reader could mistake for a new
decision. That is the right call and the right disclosure. **It remains Sarah's to veto**; the MO
is recording that it endorses CONFIRM, with the revert cost you stated (one `model6_core.py` hunk;
mode and 2π fixes independent and unaffected).

## 2. B2 is INCOMPLETE against the pin — and the gap is the MO's, not yours

**The same pin block carries a second B2 obligation the MO's acceptance bar omitted:**

> **D and φ:** verify in B2. The η=(r−1)/(r+1) form is **the large-D limit**, but the backbone D
> param was 50 (recon), not the pinned ≳200, and φ was 8 GHz; the reference-free threshold
> consumes neither for the threshold itself — **confirm their remaining role/consistency rather
> than assume.**

**You met the acceptance bar as written. The bar was written short.** That is an MO defect —
the seventh this session — and it is recorded on the board as such.

**Current state, MO-verified:** per-synapse `D_modes = 20` (`vibrational_cascade_module.py:131`),
backbone `D_modes = 50` (`model6_parameters.py:810`), pin says **≳200**. Your `B2-1` log row does
not mention `D_modes` at all (`grep -c` → 0). You *did* document at `model6_parameters.py:783`
that *"D_modes does not enter P_c. Only omega_0 and Q do"* — which is half the answer.

**RULED — close it, and keep it small.** The pin asks you to *confirm the remaining role*, not to
re-derive a lattice. Specifically:
1. State where `D_modes` still does work after B2, if anywhere, with `file:line`.
2. **Address the large-D question directly:** `η = (r−1)/(r+1)` is the large-D limit, and the
   per-synapse site now runs `D = 20`. Either show the limit is still adequate at D = 20, or
   record it as a stated limit on the per-synapse η. **Do not change D to make it comfortable** —
   that is the emergent-physics lock, and D is not yours to tune.
3. Same one-line treatment for φ, given you now derive it as `ω₀/Q` in `__post_init__`.

**This is a documentation-and-verification unit, not new physics.** If it turns out D genuinely
does no work anywhere post-B2, say exactly that and B2 closes.

## 3. Your lead file is now current — noted

`leads/po1-b2.md` and `queue/po1-b2.md` updated. The earlier violation is cleared.

## 4. Your acceptance measurement is VERIFIED and ACCEPTED

The MO ran it: A1 ratio 1 exactly, A2 ratio 1.000000, **both positive controls FIRED** (C1 at
5000, C2 at 6.283189 vs 2π=6.283185), model `CONSTRUCTS OK`, **T1′ static 7/7** with the observed
edge list identical to pre-registration. Recorded on the board as the first measurement-level
acceptance on this program and the standard for the rest. **PO-1 is not WRAPPED** — item 2 above
is open, and PO-2 remains gated until your tree is clean and stays clean.
