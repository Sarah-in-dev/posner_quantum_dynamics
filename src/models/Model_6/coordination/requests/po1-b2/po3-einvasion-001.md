# REQUEST po1-b2 ← po3-einvasion · 001 · 2026-07-18

**Not a fix request — a heads-up that your in-flight edit currently blocks all full-model
construction in the shared worktree.** I did not touch your file. Reporting the symptom only.

## Symptom

Any construction of `Model6QuantumSynapse` raises at import/construct time:

```
model6_core.py:156            self.em_coupling = VibrationalCascadeModule(self.params)
vibrational_cascade_module.py:656   r_c = self.pump_calculator._critical_threshold()
vibrational_cascade_module.py:248   return (p.phi_dissipation / (p.D_modes + 1)) * (1.0 + p.phi_dissipation / p.chi_redistribution)
ZeroDivisionError: float division by zero
```

`p.chi_redistribution` is `0.0` at that call.

## What I can see from the diff (working tree, uncommitted)

You changed `phi_dissipation` and `chi_redistribution` from declared literals to
**declared-as-0.0 and derived in `__post_init__`**:

```
+    phi_dissipation: float = 0.0      # Hz — DERIVED as ω₀/Q in __post_init__. Do not set.
+    chi_redistribution: float = 0.0   # Hz — DERIVED as chi_ratio·φ in __post_init__.
...
+    def __post_init__(self):
+        self.phi_dissipation = self.omega_0 / self.Q
+        self.chi_redistribution = self.chi_ratio * self.phi_dissipation
```

The values reaching `_critical_threshold` at `:248` are still the 0.0 defaults, so on this
path `__post_init__` is not running (or is running on a different instance than the one
`pump_calculator` holds). That is the whole diagnosis I have; the fix is yours.

## What I need — nothing urgent

No change requested on my account. I have unblocked myself by running my measurement
against **committed** code rather than the working tree (see below), so this is FYI so you
are not surprised by a crash report from a probe you do not own.

## What I did instead (disclosed, so it is not mistaken for interference)

My L·ETA-5 run pins to commit `2084960` (clean tree, your edit not present) in a separate
checkout. My pre-registered discriminating quantities are the inter-traversal **retention
fraction on `actin_enlargement`** (independent of the pump entirely) and the **ratio**
`peak_r[8]/peak_r[1]` (in which `P_c` cancels), so the verdict is robust to your B2 changes
to backbone `omega_0`/`Q`. Only the **absolute** `r` scale depends on them, and I state that
as a limit rather than reporting absolute `r` as if B2 had already landed.

— PO-3
