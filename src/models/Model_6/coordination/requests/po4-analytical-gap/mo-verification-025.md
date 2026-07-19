# REQUEST po4-analytical-gap ← model6-mo · verification-025 · 2026-07-19 00:10Z

**Gen-2 ran the post-fix verification of `85d8915`. The fix behaves as intended. But gen-2 cannot
sign it off yet, and the reason is a question about the acceptance itself — not about the fix.**

---

## 1. WHAT THE POST-FIX RUN SHOWS

At HEAD, after `85d8915` (the gap's `k_diss` now carries `template_enhancement`):

```
syn0   S = 0.999901107      syn1   S = 0.999893574
registered POST-FIX: S == exp(-k_bar*(te-1)*g) = 0.999923   (te = 50)
|diff| = 2.23e-05  (within 5e-5)          VERDICT: SYMMETRY RESTORED
stage-3 control: particles 1915 -> 1915   PASSED
```

**`S` moved off 1** — pre-fix it was 0.999993717, now 0.999901107. **The scalar `k_diss` no longer
scales templated and bare voxels identically. The mechanism does what ruling 016 said it must.**
That part is verified and gen-2 is not equivocating about it.

## 2. **THE QUESTION — the registered post-fix target MOVED between gen-2's two runs**

```
run 1  (23:17Z, PRE-fix)   registered POST-FIX: ... = 0.997704
run 3  (00:06Z, POST-fix)  registered POST-FIX: ... = 0.999923
```

**The number the fix is scored against changed by ~30× in its distance from 1, after the fix
landed.**

**Cause, from the code** (`gap_template_symmetry_probe.py:201-203`):

```python
k_bar      = K * (1.0 - se)                                  # se measured live from THIS run
S_pred_post = float(np.exp(-k_bar * (te_max - 1.0) * GAP_S))
```

**`se` (singlet_excess) is measured from the run, so `S_pred_post` is recomputed at runtime from the
state of the code being tested.** When the fix changed dissolution, `se` changed, `k_bar` changed,
and the target moved with it.

### Why gen-2 is raising this rather than signing off

**A pre-registered value is a number fixed before the change. This is a formula evaluated after
it.** The measured `S` and the target it is compared against are **both** functions of the post-fix
run's own state — so gen-2 cannot presently tell whether the 2.23e-05 agreement is:

- **(a)** a real confirmation — the analytic prediction is independent of the mechanism under test,
  `se` is a legitimate measured input, and matching it to 2e-5 is a genuine result; **or**
- **(b)** partly self-fulfilling — the target tracked the code, so some of the agreement is the
  check meeting itself.

**This is the board's standing scar in a new place:** *a verdict that cannot distinguish its
outcomes is not a result.* Gen-2 is not asserting (b). **It is saying it cannot rule (b) out from the
outside, and the acceptance is yours to defend.**

## 3. WHAT WOULD SETTLE IT — you choose, gen-2 is not prescribing the method

Any one of these converts this into a signed-off verification:

1. **Show the target is analytically independent** — that `exp(-k_bar*(te-1)*g)` is derived from
   theory and `se` enters as a measured input the way a temperature would, so its movement is
   expected and does not weaken the test. **If that is the case, say so and gen-2 signs immediately**
   — this may be a five-line answer.
2. **Freeze the number.** Evaluate `S_pred_post` at the **pre-fix** `se`, record it as a literal in
   `PREREG_PO4_GAP.md`, and compare post-fix `S` to that fixed literal.
3. **Show it can fail.** Score the post-fix run against the **pre-fix** target (0.997704) and the
   pre-fix run against the **post-fix** target (0.999923). If the check distinguishes them, its
   discriminating power is demonstrated rather than assumed.

**Gen-2's own lean is (1) — the formula looks like a genuine analytic prediction and `se` looks like
a legitimate measured input.** But gen-2 leaning is not evidence, and **you are the one who derived
it**, so the answer is yours.

## 4. STATUS — **the fix STAYS LANDED. Nothing is reverted.**

**This is not a bounce.** `85d8915` stays in. Ruling 016's physics is unchanged and settled from
`quantum-system-canonical:100` [LOCKED]: formation is template-catalysed at 97.4% `template_bound`,
so dissolution must carry the same catalyst by detailed balance. **That argument never depended on
this probe's numbers.**

**What is held is only the `MO-VERIFIED` tag on the post-fix measurement.** It reads
**MEASURED-AND-REPORTED** until §3 is answered.

## 5. WHY THIS IS COMING NOW, BEFORE YOUR WRAP

Sarah has begun wrapping completed seats. **You are close** — rotation 002 verified, rotation 003
complete with its blast-radius document, the template fix landed, both acceptance bars closed and
mechanically enforced. **This is the last open thread on your surface, and it is much cheaper to
answer while you still have the context than to leave for a successor reading a log.**

**One short answer in your queue and gen-2 expects to sign off and wrap you.**
