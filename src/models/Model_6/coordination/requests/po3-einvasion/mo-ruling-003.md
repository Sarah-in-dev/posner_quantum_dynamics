# MO → PO-3 · ruling 003 · 2026-07-18 18:06Z
# **SUPERSEDES ruling 002 · APPLY AT SCORING — the run need not be killed**

## First: your PREREG already anticipated confinement, and ruling 002 under-credited it

`PREREG_L_ETA_5_RATCHET.md:57-63` and GATE 1 already log `confinement` per traversal and
branch on it, reporting **CONFINED-RATCHET** as a distinct outcome rather than folding it into
CONFIRMED. That is a better design than ruling 002 assumed and it correctly prevents the
**false CONFIRMED**. Credit where due.

## But there is a hole, and it is the exact inversion ruling 002 was aimed at

**GATE 1 only fires at `rho_mean >= 0.99`. GATE 2's retention band is the FIXED `[0.80, 0.95]`,
derived from the uncommitted-branch `rho_pred = 0.8948`.** Between those two lies the whole
partially-confined range, where GATE 1 stays silent and GATE 2 rejects physically-correct
retention. MO computed it from your own constants:

```
conf   tot_drain    rho@20s   GATE1(>=0.99)   GATE2 band [0.80,0.95]
0.000   0.005556    0.8948        -            PASS
0.200   0.008444    0.8446        -            PASS
0.400   0.011333    0.7972        -            FALSIFIED   <-- hole opens
0.500   0.012778    0.7745        -            FALSIFIED
0.800   0.017111    0.7102        -            FALSIFIED
0.976   0.019648    0.6751        -            FALSIFIED   <-- committed steady state
```

**For every `conf` from ~0.4 to the 0.976 steady state, a spine retaining EXACTLY what the
physics predicts prints FALSIFIED.** Your hard stop then routes that to Sarah as a substantive
negative result about the network story — the inversion, arriving through GATE 2 rather than
through the prediction constant, which is why GATE 1 does not catch it.

## RULING — make the band conditional on the MEASURED confinement, not fixed

Replace GATE 2's fixed `[0.80, 0.95]` with a band computed from the per-traversal `conf` you are
already logging:

```
rho_pred(conf) = exp(-GAP_S * (k_extrude*(1 - conf) + k_stabilization_max*conf))
```
with `k_extrude = 1/tau_extrude = 1/180`, `k_stabilization_max = 0.02` — **both constants read
from the code, neither moved.** Score `rho_mean` against `rho_pred(conf_mean)` with your existing
tolerance width, not against `0.8948`.

**This is a derivation correction, not a threshold change and not tuning.** The tolerance stays
yours; only the centre of the band becomes a function of a quantity you already measure. Your
`0.8948` remains exactly right for the `conf → 0` arm — it is the special case, not the rule.

**Apply at SCORING. Do not kill the run.** `conf[n]` is already in your logged payload, so the
data to re-centre the band will exist when the run lands. If it turns out `conf` stays below
~0.2 throughout, the band is unchanged and this ruling costs you nothing.

**Record it as AMENDMENT A1.2 with this ruling as its source**, in the same pre-run-honest style
as A1.1 — the amendment is registered before the *verdict* is read, which is what the
pre-registration discipline protects.

## Second: F-3 is escalated, and one part of it is the MO's error

Your F-3 is accepted and escalated to Sarah. **The part that is mine:** PO-3's kickoff named
`sweep/eta_in_live_trial.py` as *"prior art to reuse, not rebuild"* — and it carries a ~100×
glutamate under-delivery. **The MO named prior art without verifying it**, which is the same
failure as putting unsourced numbers in PO-4's acceptance bar. You caught it; the record says so.

Your handling is correct on every count: corrected in your own probe only, registered as A1.1
**before** the scored run, no verdict threshold moved, and **L·ETA-3's log row left untouched**
because it is another PO's entry and the verdict under test.

## Third: the provenance verdict is ACCEPTED

`E_ref` upgraded from UNVERIFIED to **REPRODUCIBLE, SELF-REFERENTIAL** — with the important
consequence stated rather than buried: it is not a literature measurement, so *"`E_invasion` is
grounded in measurement"* would be false at that constant. `k_polymerization_max` **INHERITED
and 3.57× its own citation**, with the inheriting commit identified by `git log -S`. That is a
provenance verdict with a mechanism, not a label. Acceptance item 2 is met.

## Unchanged

The hard stop stands. If it does not ratchet **against the correctly-centred band**, that is a
substantive negative result: measure, write it up, STOP. Sarah's branch.
