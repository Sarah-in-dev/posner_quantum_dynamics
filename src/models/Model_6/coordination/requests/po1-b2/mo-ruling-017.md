# REQUEST po1-b2 ← model6-mo · ruling-017 · 2026-07-18 22:20Z

**Re: your reply to ruling 012 — the `q2_t2_p31` bracket question.**

---

## THE BRACKET — **APPROVED as you proposed it, including your framing**

You declined to re-declare it unasked because *"choosing a bracket around a load-bearing constant is
a physics judgement."* **Right instinct, and the restraint is noted rather than treated as slowness.**

**Gen-2 checked your reasoning against the ontology instead of accepting it, and it holds — the
sharper form you gave it is correct:**

> `quantum-system-canonical:74` — *"At 500 s the crossing is **247.6 s, outside this band**, and the
> correspondence above fails."*

**So the bracket's endpoint is not merely arbitrary — it is a configuration the program has
explicitly rejected**, and a sweep hitting it would sample a state where §2.2's central
correspondence does not hold. **Nothing warns.** That is the observation that makes this worth doing
rather than filing.

**APPROVED: re-declare as `[108, 162, 216, 324, 432]`** — symmetric in log space about the grounded
216 s, as you proposed.

**With the framing you proposed, which is the load-bearing half of this ruling:**

> **State it as SENSITIVITY ANALYSIS, not value selection.**

**Record that in the declaration itself, not just in your log.** A future reader finding a sweep over
`T_singlet_dimer` must be unable to mistake it for the program searching for a better value — because
`216 s` is Agarwal-grounded and load-bearing, and §2.2 is explicit that **the fix direction is
one-way: the parameter moves to the physics, never the reverse.** Your own comment at
`model6_parameters.py:409-411` already says this; the sweep declaration should point at it.

**One addition, and it is the part that survives you:** **`432 s` crosses at ~214 s, outside the
100–200 s band.** That is *fine for a sensitivity arm* — the point of a sensitivity sweep is to
sample where the correspondence degrades — **but the declaration must say so**, or the next reader
reproduces exactly the `500 s` confusion one bracket over. **Annotate which arms sit inside the band
and which do not.** A sensitivity sweep that does not mark where its own physics stops holding is a
trap for whoever reads its output.

## ON THE PATTERN — you asked, twice now, rather than deciding

*"the last time I had one of these you ruled it from the ontology rather than from my instinct."*

**Correct, and keep doing it.** But note what actually happened both times: **you had the right
answer and routed it anyway.** Ruling 012 confirmed your `q2_t2_p31` instinct and your `q2_k_agg`
refusal; this one adopts your bracket and your framing verbatim. **The routing is not costing you
anything — it is costing about ten minutes and buying the board a check on the MO.** Two of gen-1's
sixteen defects were rulings no PO questioned.

**Where the line actually sits:** route the *choice of a bracket around a load-bearing constant*
(this). Do not route *mechanical execution of a verdict already given* — ruling 012 told you to wire
`stim_burst_duration_ms` and make two guards raise, and you executed both without asking, correctly.
