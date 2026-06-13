# Observability instrumentation plan

## PART A — Throttle the backbone diag flood

**File:** `src/models/Model_6/multi_synapse_network.py`
**Location:** `_update_backbone_field`, around line 1053 (`if i == 0:`)

```diff
         for i, synapse in enumerate(self.synapses):
             r = p_met_agg[i] / P_c
             eta = (r - 1.0) / (r + 1.0) if r >= 1.0 else 0.0
             synapse.set_backbone_condensation_eta(eta)

-            if i == 0:
-                mt_inv = getattr(synapse, '_mt_invaded', False)
-                print(f"[backbone diag] P_met={p_met[i]*1e15:.2f}fW  "
-                      f"P_agg={p_met_agg[i]*1e15:.2f}fW  P_c={P_c*1e15:.2f}fW  "
-                      f"r={r:.3f}  eta={eta:.4f}  invaded={mt_inv}")
+        # Throttled diagnostics (every 200 backbone updates)
+        if not hasattr(self, '_backbone_diag_n'):
+            self._backbone_diag_n = 0
+        self._backbone_diag_n += 1
+        if self._backbone_diag_n % 200 == 0:
+            s = self.synapses[0]
+            mt_inv = getattr(s, '_mt_invaded', False)
+            print(f"[backbone diag] P_met={p_met[0]*1e15:.2f}fW  "
+                  f"P_agg={p_met_agg[0]*1e15:.2f}fW  P_c={P_c*1e15:.2f}fW  "
+                  f"r={p_met_agg[0]/P_c:.3f}  eta={(lambda r: (r-1)/(r+1) if r>=1 else 0)(p_met_agg[0]/P_c):.4f}  invaded={mt_inv}")
```

Note: the `r` and `eta` in the original print used the **last** synapse's values (from the loop), not synapse 0's. The replacement recomputes for synapse 0 explicitly. Also the print is moved **outside** the per-synapse for-loop (after it ends) to avoid indentation within the loop.

---

## PART B — Instrument the actin→envelope chain

### B.1 — Stash debug attributes in `_update_actin`

**File:** `src/models/Model_6/spine_plasticity_module.py`
**Location:** `_update_actin`, after line 320 (after `f_CaM = ...`)

```diff
         # CaM activation by the calcium transient (uM, Hill n=4).
         ca = max(0.0, calcium)
         f_CaM = ca**a.hill_calcium / (a.K_calcium_poly**a.hill_calcium + ca**a.hill_calcium)
+
+        # Observability: stash for external diagnostics
+        self._dbg_calcium_uM = float(ca)
+        self._dbg_f_CaM = float(f_CaM)

         s = float(np.clip(structural_drive, 0.0, 1.0))  # commitment (DDSC)
```

### B.2 — Add chain-detail line to throttled diag

**File:** `src/models/Model_6/multi_synapse_network.py`
**Location:** Inside the throttled diag block added in Part A, immediately after the existing print.

```diff
+            print(f"[backbone diag] P_met={p_met[0]*1e15:.2f}fW  "
+                  f"P_agg={p_met_agg[0]*1e15:.2f}fW  P_c={P_c*1e15:.2f}fW  "
+                  f"r={p_met_agg[0]/P_c:.3f}  eta={(lambda r: (r-1)/(r+1) if r>=1 else 0)(p_met_agg[0]/P_c):.4f}  invaded={mt_inv}")
+            sp = s.spine_plasticity
+            _room = max(0.0, 1.0 - (sp.actin_dynamic + sp.actin_enlargement + sp.actin_stable)
+                        / (self.params.dendritic_backbone.omega_0 and  # just need any truthy gate; compute inline:
+                           sp.params.volume.max_enlargement_ratio ** (1.0 / sp.params.volume.actin_volume_scaling)))
```

Actually, let me simplify the room computation. Here is the combined final form for the entire throttled block:

```diff
+        if not hasattr(self, '_backbone_diag_n'):
+            self._backbone_diag_n = 0
+        self._backbone_diag_n += 1
+        if self._backbone_diag_n % 200 == 0:
+            s = self.synapses[0]
+            mt_inv = getattr(s, '_mt_invaded', False)
+            r0 = p_met_agg[0] / P_c
+            eta0 = (r0 - 1.0) / (r0 + 1.0) if r0 >= 1.0 else 0.0
+            print(f"[backbone diag] P_met={p_met[0]*1e15:.2f}fW  "
+                  f"P_agg={p_met_agg[0]*1e15:.2f}fW  P_c={P_c*1e15:.2f}fW  "
+                  f"r={r0:.3f}  eta={eta0:.4f}  invaded={mt_inv}")
+            sp = s.spine_plasticity
+            F = sp.actin_dynamic + sp.actin_enlargement + sp.actin_stable
+            F_max = sp.params.volume.max_enlargement_ratio ** (1.0 / sp.params.volume.actin_volume_scaling)
+            room = max(0.0, 1.0 - F / F_max)
+            print(f"[actin chain s0] ca_uM={sp._dbg_calcium_uM:.4f}  f_CaM={sp._dbg_f_CaM:.4f}  "
+                  f"enl={sp.actin_enlargement:.4f}  mono={sp.actin_monomer:.4f}  "
+                  f"room={room:.4f}  conf={sp.confinement:.4f}  E_inv={sp.E_invasion:.4f}")
```

---

## PART C — Read-only answers

### C1 — check_actin_three_pool.py, Phase 2 (ACTIVITY)

Lines 98–100:
```python
# ---- Phase 2: ACTIVITY (60 s, calcium 2.0 uM, drive 0) ------------------
print("\n--- Phase 2: ACTIVITY (60 s, Ca=2.0 uM, drive=0) ---")
run(60.0, calcium=2.0, drive=0.0, label="end ACTIVITY")
```
Where `run()` calls `m.step(dt, structural_drive=drive, calcium=calcium)` (line 61) with `dt=0.005`.

**Calcium = 2.0 µM, duration = 60 s (12 000 steps at dt=0.005), drive=0.**

### C2 — Network calcium chain in model6_core.py

Lines 324/526–527:
```python
ca_conc = self.calcium.get_concentration()       # returns array in Molar
calcium_uM = float(np.max(ca_conc)) * 1e6        # converts M → µM
```
Then at lines 620–628:
```python
spine_state = self.spine_plasticity.step(
    dt,
    self._committed_memory_level,   # or 0.0
    calcium_uM,                     # ← µM
    quantum_field_kT=dimer_field_kT
)
```

**The calcium argument arrives at `spine_plasticity.step()` in µM**, converted from the analytical calcium system's Molar output via `* 1e6`.

### C3 — Hill K units match?

`ActinParameters.K_calcium_poly = 1.0` — the comment on `_update_actin` line 318 says "calcium transient (uM, Hill n=4)", and the `step()` docstring (line 283) says "calcium: Current calcium concentration in **µM**".

**Units match: both calcium and K_calcium_poly are in µM. No unit mismatch.**

However, the standalone test feeds **2.0 µM** (2× the K of 1.0 µM → f_CaM ≈ 0.94 at Hill n=4). The network path feeds whatever `np.max(ca_conc) * 1e6` produces under `voltage = -10 mV` depolarization — which could easily be ≪ 1 µM, placing f_CaM deep in its toe region. The magnitude must be measured at runtime (Part B provides this).

---

## One-line verdict

No unit mismatch exists; the calcium simply needs to be **measured at runtime** to determine whether it sits below the Hill knee (~1 µM), which Part B's `_dbg_calcium_uM` will reveal.
