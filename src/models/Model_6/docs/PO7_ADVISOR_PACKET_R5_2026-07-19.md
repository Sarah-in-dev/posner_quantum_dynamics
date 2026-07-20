# Advisor packet — round 5: you were right about the spin, and the blob breaks
**PO-7 · 2026-07-19 · replaces the withdrawn R4 packet**

**R4 is withdrawn in full.** Its premise — phosphate provenance acting as a *cross-synapse*
channel — was unphysical, and Sarah caught it. Provenance is local; cross-synapse entanglement
is mediated by the condensate backbone (`k_cross = 0.5·√(η_i·η_j)·w_spatial·P_product`, with
`w_spatial = exp(−d/5 µm)` the **condensate** coupling length). We had read §1 of the partition
skill and treated it as licensing a mechanism only §2 describes. Everything built on it is
withdrawn and recorded in `L·PO7-2`.

**What replaced it is your R4 diagnosis, built and measured. It works.**

---

## 1. YOUR CALLS, CHECKED

| your claim | outcome |
|---|---|
| **"Your graphs violate monogamy"** | **CONFIRMED, and worse than an order of magnitude.** Mean degree **715.2** (179× the 4-spin bound), max **902**, **1034/1034 dimers over bound**, **99.44% of edges physically inadmissible**, max admissible E = 2068 vs actual 369,740. |
| **"H⁰_engaged is six graph Laplacians in a trench coat"** | **CONFIRMED numerically** — cross-block edges **0 / 369,740**, decomposition identity holds exactly. **One correction: it is THREE blocks, not six.** The channel rule sets `ka = 2·ax + (0/1)` and `kb` from `−d`, so every edge joins channel 2m ↔ 2m+1 of the same axis; channels fuse in pairs → `{0,1},{2,3},{4,5}`. At the operating point each block has exactly 1 component, so **H⁰_engaged = 3 = the number of spatial axes.** More degenerate than stated, same conclusion. |
| **"The stalk is the spin state, not the J-couplings"** | **Adopted.** The six J's are the intra-dimer Hamiltonian acting on the stalk. The old stalk was `np.random.normal(0.15,0.15,6)` — input-blind by construction. |
| **"Adopt the monogamy constraint immediately regardless"** | **DONE — and it is the result below.** |

---

## 2. THE RESULT: THE BLOB BREAKS

Spin-resolved bonding, opt-in, **OFF ⇒ bit-identical** (`1034/369740/0.991922159684`, gated
before and after). Every dimer owns 4 ³¹P slots; a bond must claim a **free** slot at both ends
or it does not form; provenance bonds must claim their **named inherited** slot. **Degree ≤ 4 is
derived, never capped.**

Standing fingerprint rig, 1 synapse, 200 steps, seed 31337:

| | OFF | ON |
|---|---|---|
| edges | 369,740 | **2,031** (0.55%) |
| mean degree | 715.16 | **3.93** |
| max degree | 902 | **4** |
| dimers over bound | 1034 (100%) | **0** |
| components | 1 | **184** |
| **largest_frac** | **1.000** | **0.112** |
| frustrated bonds | — | **491,566** |

**The first physically admissible entanglement graph in this investigation, and it carries
non-trivial partition structure** where the inadmissible graph was one component containing
every dimer.

The 491,566 refusals are **frustration**, exactly as you framed it: pairs individually
satisfiable, jointly not — the H¹ obstruction a direct sum of graph Laplacians cannot express.

### Why it works — the percolation number

Pre-ignition the partition is **7 components, largest_frac 0.22** (one per synapse, as §5
predicts). Then:

```
t=9.05   comps=7   largest_frac=0.218   cross_bonds=0
t=9.55   comps=1   largest_frac=1.000   cross_bonds=98
```

**98 cross-bonds — already past the Werner F>0.5 filter — collapse 100% of dimers into one
component.** Each synapse was a near-complete clique, so a single cross-bond fused two dense
balls wholesale. Shattering the cliques is what removes the percolation, and the spin bound
shatters them without touching the Werner bound.

---

## 3. TWO STALE FIGURES THIS SUPERSEDES

**(a) "The pump is dead, r ≈ 0.077."** Never measured — it propagated from a kickoff. A direct
measurement returns r = 0.0390, which is **L·ETA-1's documented rest floor**, not a ceiling.
Re-running L·ETA-2's rig with glutamate wired: `E_invasion` **0.3508 vs 0.3518 (0.28%)**, peak
r **2.53**, **7/7 synapses condensed**. **The pump ignites.** (The archived `eta_probe.py` still
drives voltage-only — the ERR-2 defect — so anyone re-running it will reproduce the silent-NMDAR
artifact.)

**(b) L·ETA-1's geometry table — "rowsum > 6.89; unreachable at ≥2 µm at any N."** Ignition
observed at **row-sums 4.14–5.05**. That table was computed with NMDARs silent (`ca_open` ≈
0.05); with glutamate wired `ca_open` reaches 0.34–0.70. **The table needs re-deriving before it
is cited again.** Also noted: condensation **strobes rather than latches** — `n_cond` flicks
0 → 3 → 7 within 0.15 s on stochastic calcium.

---

## 4. WHAT WE ARE NOT CLAIMING

- **No §8 keystone claim.** Spin resolution currently governs **intra-synapse bonds only**;
  cross-synapse bonds are created elsewhere (`_update_entanglement`) and are **not spin-accounted**.
  A dimer's four nuclei must be shared between its intra *and* cross bonds — until the ledger
  spans both, cross-bonds consume no spins and can still weld cliques for free. Extending it is
  the next build.
- **184 components is one synapse, one seed.** Structure, not yet a partition over synapses.
- **Nothing revives cross-synapse provenance.** The R4 withdrawal stands.
- **The ℂ¹⁶ stalk is not built.** We implemented the *constraint* the spin state imposes
  (occupancy + monogamy + named-slot inheritance), not the state itself. Your step 4 — the full
  spin-state stalk with partial-trace restrictions — remains open, and on your own sequencing
  should wait until a partition is shown to carry input.

---

## 5. THE QUESTION FOR YOU

**Is enforcing occupancy enough to make the sheaf real, or does H¹ require the actual ℂ¹⁶ state?**

We now have named mediating spins and measurable frustration — which is the *combinatorial*
shadow of monogamy. What we do not have is amplitude: whether spin *a* can still bond depends,
in the full theory, on how it is entangled with the other three in its own dimer, and that is
what makes the restriction a partial trace rather than a slot check.

Concretely: **is our frustration count a faithful H¹, or only a lower bound that ignores partial
entanglement?** If the latter, the ℂ¹⁶ build moves up your ordering rather than waiting — because
the constraint we can already enforce fragments the graph, and the question becomes whether the
*structure* of that fragmentation is physical or an artifact of treating spins as binary slots.

Secondary: with cross-synapse bonds brought into the same spin ledger, do you expect the
cross-synapse partition to survive at all — or is the honest prediction that a dimer spending its
four nuclei locally has none left for the network scale, making the cross-synapse keystone
**structurally** starved rather than merely weak?
