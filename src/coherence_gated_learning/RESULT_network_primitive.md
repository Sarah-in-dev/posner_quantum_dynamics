# THE NETWORK PRIMITIVE — two real findings, then a THIRD clear negative on task

## What was built

`network_cgl.py` adds the three constraints the abstraction was missing, taken from
`multi_synapse_network.py` with values traced to specific lines:
1. spatial coupling `w_ij = exp(-d_ij/5.0um)` (line 718)
2. bounded degree — 4 31P nuclei per dimer, a bond claims one at EACH end or is REFUSED (`_claim_cross_spins`, line 324)
3. global gate `k_cross = 0.5*sqrt(eta_i*eta_j)*w*P_i*P_j` (line 494); F = P_i*P_j*w > 0.5; k_release = 9.63e-3/s

## Finding 1 — the "blob" was an artifact of an unphysical parameter

22 units presented simultaneously, 20 seeds/cell:

| eta | (r) | #comps | max size | mean size | frustrated | giant? |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 22.0 | 1.0 | 1.00 | 0.0 | 0% |
| 0.10 | 1.22 | 20.4 | 2.1 | 1.08 | 0.0 | 0% |
| 0.33 | 1.99 | 16.0 | 4.2 | 1.40 | 0.2 | 0% |
| 0.60 | 4.00 | 12.6 | 5.7 | 1.83 | 2.2 | 0% |
| 1.00 | inf | 10.2 | 8.2 | 2.35 | 10.0 | **15%** |

`eta = (r-1)/(r+1)`, so eta=1.0 requires INFINITE metabolic power. Across the whole physically reachable
range the giant-component failure never occurs. **eta is a continuous GRANULARITY knob** setting the ORDER of
representable conjunctions (~0.1 pairs, 0.33 4-way, 0.6 6-way) — a concrete computational role for Frohlich
condensation, derived from the model's own rate law. `eta=0` reproduces F4-b exactly: 22 singletons, no
distributed component possible.

## Finding 2 — a REGIME-DEPENDENCE that corrected an earlier claim of mine

I first reported "frustration is 0.0 across the physical range; the spin ledger never fires." That was true
only for a SINGLE presentation. Under REPEATED presentation bonds accumulate and the ledger dominates:

| eta | bonds | max degree | frustrated | comps/trial | mean size | reuse |
|---|---|---|---|---|---|---|
| 0.10 | 140 | 4 | 27340 | 14.3 | 4.37 | 11.2x |
| 0.20 | 152 | 4 | 56941 | 12.7 | 5.29 | 9.0x |
| 0.33 | 155 | 4 | 94867 | 15.3 | 4.32 | 11.2x |

Bonds saturate at 140-155 regardless of eta, so **at saturation capacity washes out eta's granularity role**.
Both regimes are real; the first claim generalised from one.

## Finding 3 — ON TASK IT FAILS, badly

Mushroom bandit, components DERIVED by the network instead of hand-picked, everything else identical
(same running-mean table, same optimistic init), so the only difference is where conjunctions come from:

| arm | regret |
|---|---|
| network-derived, eta = 0.10 / 0.20 / 0.33 | **47052 / 47815 / 47520** |
| hand-picked d=3 subsets | 2581 |
| eps-greedy linear (no conjunctions) | 9621 |
| *always abstain* | *51800* |

**Barely better than never eating.** 18x worse than hand-picked subsets, 5x worse than linear.

### Why (measured, not guessed)

| statistic | value |
|---|---|
| keys seen exactly ONCE | **67.1%** |
| mean key reuse | 2.56x |
| component size | mean 3.05, max 13, 86% <= 4 units |
| small components containing an ODOR unit (the 98.5%-predictive feature) | **10.6%** |

Component SIZE is fine. Two things kill it:

1. **Units are feature-VALUES (117), not feature-SLOTS (22).** Binding value-units produces value-conjunctions
   that rarely recur — 67% appear once, and a key that never recurs can never accumulate credit. The
   hand-picked ensemble keys on fixed SLOTS, so each synapse has a small key space that recurs constantly.
2. **The network binds by CO-OCCURRENCE IN THE INPUT, which is unrelated to PREDICTIVENESS OF REWARD.** It
   faithfully discovers correlational structure; on this task that structure is not the reward-relevant one,
   and the single most predictive feature is absent from ~90% of the components formed.

## Honest standing

Three benchmark attempts, three negatives. The network primitive is a genuine advance in FIDELITY to the
biology and produced two real findings about the model (eta as granularity; the capacity/eta regime split),
but it has **not** produced a task win, and on this task it is close to the do-nothing baseline.

The implied next design — persistent units with fixed receptive fields (a synapse is a PERSISTENT structure
with a fixed input source, not a transient value-group), letting the graph decide which SLOTS group rather
than which VALUES — is a real and grounded idea. It is recorded here rather than immediately pursued: this
would be the fourth consecutive fix-and-retry, and that pattern deserves a deliberate decision, not momentum.
