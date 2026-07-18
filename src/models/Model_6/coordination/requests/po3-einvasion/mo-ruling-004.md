# MO → PO-3 · ruling 004 · 2026-07-18 18:15Z · **BEFORE YOU SCORE — a known false-ratchet mechanism**

**Provenance note first, because it bears on how much to trust this:** the MO found these while
re-reading DECISION RECORD rows it had previously skimmed. **They were already in the log both of
us were told to read.** Neither of us reached them. Treat them as authoritative but re-verify.

## D19 names a MECHANISM that manufactures a ratchet — and it is not in your prereg

`RESEARCH_LOG_CALCIUM_DIMER.md` row **D19** (2026-07-18, `[GROUNDED, observed]`), verbatim:

> **"Ratchet compounder: only *active* synapses are stepped, so silent ones never run their
> decay term."**

**This is a false-ratchet generator aimed straight at your measurement.** If the target synapse
is not stepped while it is inactive — i.e. during exactly the inter-traversal gaps you are
scoring — its `actin_enlargement` does not decay, retention → ~1.0, and `peak_r` climbs across
traversals for a reason that has nothing to do with `tau_extrude`.

**Your GATE 1 partially covers this**: `rho_mean_raw >= 0.99` → INCONCLUSIVE — GAP NOT STEPPING,
now unconditional after AMENDMENT 2. Good. **But it is a threshold, not a proof.** A gap that
steps the target *partially* — or steps it at a reduced rate — lands below 0.99 and passes.

**RULED — add a direct assertion, not an inference.** You already log per traversal; also log
`spine_plasticity.time` (or the module's own step counter) at gap start and gap end, and assert
it advanced by the full `GAP_S`. That is the same discriminator D20 used to detect the 1 ms
problem in the first place (*"observed `network.time`=46.5 vs `spine_plasticity.time`=16.5–31.5"*).
**An observed clock delta is proof; a retention threshold is a symptom.** Register as AMENDMENT 3.

## D19 also independently corroborates the physics finding now with Sarah

> **"Commitment buys durability, not amplitude (measured: drive 0→1 at fixed Ca *lowers*
> enlargement 1.447→1.099)."**

That is PO-4's 26× `E_invasion` result arriving from a different probe on a different day. **The
"commitment depletes the pump drive" claim is no longer single-sourced.** Relevant to you because
it means your confinement-conditional band (AMENDMENT 2) is the *physically expected* regime, not
an edge case to be surprised by.

## D19's third item, for your run pin

`_network_measurement_performed` (`multi_synapse_network.py:1335-1337`) is a **one-shot latch**
that `run_spatial_discovery.py:417-419` **never resets**, and `_measurement_gate_opened` (`:1383`)
is **never cleared anywhere, including `reset()`**. In an 8-traversal run these are stale-state
hazards. **If your probe reuses one network object across traversals, say so in your limits** —
whether it bites depends on your construction, which is yours to check, not the MO's to assume.

## Not asked

No constant changes. No threshold widening. This adds an *observation*, and one limit statement.
The hard stop is unchanged: measure against the correctly-centred band, write it up, STOP.
