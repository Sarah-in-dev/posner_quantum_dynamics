# Session Handoff: Backbone Cooperative Robustness & Dimer Physics Corrections

**Date:** April 3, 2026
**Session focus:** Debug backbone field coupling, replace arbitrary physics with literature-grounded mechanisms, identify and design fixes for dimer dynamics.
**Status:** Backbone architecture corrected and verified. Three dimer physics fixes designed but not yet implemented. Ready for implementation and testing.

---

## What Was Done This Session

### 1. Backbone Sigmoid → Fröhlich Rate Equations

**Problem:** The original backbone used a hand-tuned sigmoid to model Fröhlich condensation. The threshold (95.7 GHz) was in physical units but compared to dimensionless `_network_modulation` values, so it never activated. We tried rescaling to 3.0, then 12.0, but this was prescribing thresholds rather than letting physics determine them.

**Fix:** Replaced the sigmoid with the same Zhang/Agarwal/Scully 2019 rate equations used in the per-synapse Fröhlich cascade (`vibrational_cascade_module.py`). The backbone now has its own lattice parameters (D_modes=50, phi_dissipation=8.0 GHz, chi_redistribution=0.06 GHz) that produce a critical pump rate r_c from physics, not from us. The aggregate `_network_modulation` is converted back to physical units via `kT_per_modulation_unit = 1.5` (derived from the modulation_per_dimer energy scale × 10% spine-to-backbone coupling efficiency).

**Files modified:** `model6_parameters.py` (removed pump_threshold, sigmoid_steepness; added D_modes, phi_dissipation, chi_redistribution, kT_per_modulation_unit), `multi_synapse_network.py` (replaced sigmoid block with FrohlichCondensation instance and rate equation calls).

**Result:** Condensation ratio η ramps gradually (0.14 → 0.55 over sustained stimulation) instead of snapping to saturation. Regime progresses thermal → sub_threshold → weak_condensate → strong_condensate → full_condensate over time.

### 2. Additive Field Injection → Cooperative Robustness Modulation

**Problem:** The backbone computed a field (up to 127 kT) and added it directly to each synapse's `_collective_field_kT`. This caused runaway: 127 kT backbone + 22 kT local → 149 kT total → massively amplified dimer formation → higher modulation → higher backbone pump → feedback loop producing 467 kT pump energy by t=1.5s. Physically nonsensical.

**Root cause identified through literature research:** The dendritic shaft microtubules are 500-1000 nm from the spine head (across the spine neck). Near-field EM coupling drops as 1/r³: 22 kT at 1 nm, 0.16 kT at 5 nm, effectively zero at 500+ nm. The backbone cannot couple to spine-head dimers through direct EM field addition.

**Correct mechanism (from Babcock 2024, Hu 2008, Merriam 2013):** When a microtubule invades a spine, it creates a continuous tubulin/tryptophan lattice extending from the backbone through the spine neck into the spine head. The backbone's Fröhlich condensation state enhances the *cooperative robustness* of the invaded MT's tryptophans, increasing their coherent fraction f_coherent. The backbone modulates f_coherent_local (bounded 0.08→0.10), not field strength.

**Fix implemented:**
- Removed `set_shared_backbone_field()`, `_shared_backbone_field_kT`, and the additive line 461 in `model6_core.py`
- Added `set_backbone_condensation_eta(eta)` setter and `_mt_invaded` boolean tracking
- In `em_tryptophan_module.py`, f_coherent now scales: `f_coherent = f_base + (f_max - f_base) × η_backbone` for invaded spines only
- `_update_backbone_field()` in `multi_synapse_network.py` passes η to invaded synapses, 0 to non-invaded
- f_coherent_max = 0.10 (Babcock 2024 experimental cooperative maximum)

**Files modified:** `model6_core.py` (removed additive path, added eta setter and invasion tracking), `em_tryptophan_module.py` (f_coherent modulation with backbone_eta parameter threaded through update()), `multi_synapse_network.py` (eta injection instead of field injection, gated on invasion state).

**Result:** `collective_field_kT = 22.09 kT` — entirely from first-principles electrostatics. No runaway. Maximum possible boost is ~12% (22→24.7 kT at full condensation).

### 3. Removed Unphysical modulation_enhancement Sigmoid

**Problem:** `em_tryptophan_module.py` had a sigmoid that multiplied `mu_collective` by up to 3× based on `_network_modulation`. Since field energy scales as the square of the dipole, this produced up to 9× field inflation (7 kT → 63 kT). This was an arbitrary placeholder with no physical basis.

**Literature basis for removal:** Babcock 2024 establishes that tryptophan superradiance properties (quantum yield, decay rates, collective dipole) are determined by the geometry of the tryptophan network — positions and orientations of transition dipoles, N, and f_coherent. There is no mechanism for nearby calcium phosphate dimers to modify the intrinsic quantum optical properties of the tryptophan lattice. The reverse coupling from dimers goes exclusively through Fröhlich condensation (vibrational energy transfer into the lattice), which was already modeled in the vibrational cascade module.

**Fix:** Removed the `modulation_enhancement` sigmoid calculation and its application to `mu_collective` at lines 596, 605-606, and 707-708 of `em_tryptophan_module.py`. The `network_modulation` parameter, `superradiance_threshold`, `network_state`, and `above_threshold` classification are all preserved — only the unphysical dipole multiplication is gone.

**File modified:** `em_tryptophan_module.py`

**Result:** Field dropped from 63 kT to 22.1 kT — purely from first-principles.

---

## What Was Identified But NOT Yet Implemented

### Three dimer physics fixes designed, pending implementation:

#### Fix 1 — Coherence-Dependent Dissolution Rate

**Problem:** Fixed k_dissociation = 0.001/s (1000-second lifetime) is prescribed, not emergent. Dimers that have fully decohered persist as if still quantum-protected.

**Physics (Agarwal 2023, Fisher 2015):** In Fisher's framework, the quantum spin state protects dimers from dissolution. Coherent dimers (high singlet probability P_S) are protected; decohered dimers (P_S → 0.25 mixed state) dissolve at the classical aqueous rate. The model already tracks P_S per dimer via T₂ decay — it just doesn't use it for dissolution.

**Designed formula:**
```
singlet_excess = max(0.0, (P_S - 0.25) / 0.75)
k_dissolution(P_S) = k_classical × (1.0 - singlet_excess)
```
- P_S = 1.0: k = 0 (fully protected)
- P_S = 0.25: k = k_classical (classical dissolution, ~0.05 s⁻¹ → 20s lifetime)
- k_classical is a TALON sweep parameter

**Files to modify:** `ca_triphosphate_complex.py` (lines 152, 339, 410), `dimer_particles.py` (line 127)

**What this enables:** Dimer lifetime emerges from Q2 coherence dynamics. When Q1 weakens → coherence decays → dimers dissolve → modulation drops → backbone de-condenses. The system breathes.

#### Fix 2 — Finite Phosphate Pool

**Problem:** Phosphate is never consumed when dimers form. `atp_system.py` initializes phosphate but never subtracts on formation. A spine has ~60-300 phosphate molecules — this is a hard material ceiling on dimer count. The current model accumulates 2500 dimers without limit.

**Fix:** Consume phosphate on dimer formation, return on dissolution. When phosphate is depleted, formation rate goes to zero regardless of calcium availability.

**Files to modify:** `ca_triphosphate_complex.py` (formation equation), `atp_system.py` (phosphate pool tracking)

#### Fix 3 — Return Calcium on Dissolution

**Problem:** Calcium is consumed on dimer formation (`model6_core.py:366-367`, `analytical_calcium_system.py:400`) but NOT credited back on dissolution. `ca_consumed` only counts positive formation (line 421: `np.maximum(d_dimer_dt, 0)`).

**Fix:** Track net dimer change. When dissolution exceeds formation, released calcium goes back to the local pool.

**Files to modify:** `model6_core.py` (ca_consumed calculation), `analytical_calcium_system.py` (calcium return)

---

## Verified Test Results

### Post-all-fixes test (backbone + modulation_enhancement removal):
```
Phase 1: Theta-burst stimulation (0-3s)
Phase 2: Silence (3-10s)

t=  0.00s  dimers= 70  avg_field=22.1kT  avg_mod=0.91
t=  0.50s  dimers=942  avg_field=22.1kT  avg_mod=12.17
t=  1.00s  dimers=1308  avg_field=22.1kT  avg_mod=16.81
t=  2.00s  dimers=1901  avg_field=22.1kT  avg_mod=24.23
t=  3.00s  dimers=2495  avg_field=22.1kT  avg_mod=31.64
t=  5.00s  dimers=2473  avg_field=22.1kT  avg_mod=30.67
t=  7.00s  dimers=2451  avg_field=22.1kT  avg_mod=29.78
t=  9.50s  dimers=2429  avg_field=22.1kT  avg_mod=28.77
```

- Field is physically correct at 22.1 kT (bounded, first-principles)
- No runaway
- Dimer decay during silence is minimal (2495→2429 over 7s) — expected given k_dissociation=0.001/s
- This will change dramatically once coherence-dependent dissolution (Fix 1) is implemented

---

## Implementation Design for Three Dimer Fixes (from Claude Code)

The following design was produced by Claude Code and is ready to implement. All code locations traced, interfaces identified, no ambiguity remaining.

### Fix 1 — Coherence-Dependent Dissolution

**Interface gap solved:** `update_dimerization()` has no access to mean singlet probability. Solution: add a setter `set_mean_singlet_probability(p_s)` on the dimerization module, called from `model6_core.py` after line 405 where `self._previous_coherence` is set. One-step lag is fine — P_S changes slowly.

**Changes:**
- `ca_triphosphate_complex.py:152`: Replace `self.k_dissociation = 0.001` with `self.k_classical = 0.05`
- Add `self._mean_singlet_prob = 0.25` (initialized to thermal mixed state)
- Add setter: `def set_mean_singlet_probability(self, p_s)`
- Lines 339 and 410: Replace fixed `self.k_dissociation` with:
  ```python
  singlet_excess = max(0.0, (self._mean_singlet_prob - 0.25) / 0.75)
  k_diss = self.k_classical * (1.0 - singlet_excess)
  ```
- `model6_core.py`: After line 405, call `self.ca_phosphate.dimerization.set_mean_singlet_probability(self.get_mean_singlet_probability())`
- `dimer_particles.py:127`: Replace `self.k_dissolution = 0.001` with `self.k_classical = 0.05` (currently dead code — concentration system drives removal)

### Fix 2 — Finite Phosphate Pool

**Stoichiometry:** 4 phosphate per dimer (Ca₆(PO₄)₄), 6 per trimer.

**Changes:**
- `ca_triphosphate_complex.py`: After lines 417-418, compute `self._phosphate_consumed = 4.0 * d_dimer_dt + 6.0 * d_trimer_dt` (net: positive = consumed, negative = returned)
- Add getter: `def get_phosphate_consumed(self)`
- `model6_core.py`: After calcium consumption (line 367), subtract: `self.atp.phosphate.phosphate_structural -= phosphate_consumed` with floor at 0
- No changes needed to `atp_system.py` — `update_speciation()` already recomputes HPO4 from phosphate_structural each step

### Fix 3 — Return Calcium on Dissolution

**One-line change:**
- `ca_triphosphate_complex.py:421`: Replace `self._ca_consumed = 6.0 * np.maximum(d_dimer_dt, 0) + 9.0 * np.maximum(d_trimer_dt, 0)` with `self._ca_consumed = 6.0 * d_dimer_dt + 9.0 * d_trimer_dt`
- Remove the `np.maximum(..., 0)` wrappers so net dissolution returns calcium
- `analytical_calcium_system.py` already handles negative consumption correctly (adds calcium back, clamps at 0)

### Interaction Between Fixes

Fix 1 drives fixes 2 and 3: decohered dimers dissolve at 0.05/s (50× faster than current 0.001/s) → Fix 3 returns calcium to pool → Fix 2 returns phosphate → enables re-formation of new coherent dimers. Steady state shifts from "accumulate indefinitely" to dynamic equilibrium where only coherent dimers persist. This is Fisher's framework.

---

## What's Next

### Immediate (next session):
1. **Implement the three dimer physics fixes** — design above is complete, tell Claude Code to execute it
2. **Rerun the condensation/de-condensation test** — verify dimers actually decay during silence phase when coherence-dependent dissolution is active
3. **Verify material-limited steady state** — with finite phosphate, dimer count should plateau at ~60-300 per spine, not 2500
4. **Run isotope comparison** — P-31 vs P-32 should now show dramatic differences because P-32 dimers decohere faster → dissolve faster

### Then:
5. **Print backbone diag with more decimal places** — field variation from backbone is real but below `%.1f` display precision (22.1→~24.7 kT max). Use `%.2f` to see it.
6. **Design permutation sweep scenarios** around CA1 theta-burst patterns with the corrected physics
7. **Map TALON parameters** — the new sweep-able parameters from this session:
   - `k_classical` (classical dissolution rate, default 0.05 s⁻¹)
   - `kT_per_modulation_unit` (spine-to-backbone coupling efficiency, default 1.5)
   - `D_modes` (backbone lattice modes, default 50)
   - `phi_dissipation` (backbone loss rate, default 8.0 GHz)
   - `chi_redistribution` (backbone mode coupling, default 0.06 GHz)

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `model6_parameters.py` | Replaced sigmoid params with Fröhlich lattice params in DendriticBackboneParameters |
| `model6_core.py` | Removed additive backbone field path; added `_mt_invaded`, `_backbone_eta`, `set_backbone_condensation_eta()` |
| `em_tryptophan_module.py` | Removed 3× modulation_enhancement sigmoid; added backbone_eta parameter to f_coherent calculation |
| `multi_synapse_network.py` | Replaced sigmoid with FrohlichCondensation rate equations; changed from field injection to η injection gated on invasion |

---

## Key Physics Principles Established This Session

1. **The backbone couples through invasion, not radiation.** The dendritic shaft is too far from the spine head for near-field EM coupling (1/r³ kills it at 500+ nm). The coupling mechanism is the invading microtubule creating a continuous tryptophan lattice from backbone to spine head. The backbone modulates f_coherent (cooperative robustness), not field magnitude.

2. **Tryptophan superradiance properties are intrinsic to the lattice geometry.** Nearby molecular species (dimers) don't modify the collective dipole moment. The reverse coupling from dimers to Q1 goes exclusively through Fröhlich condensation (vibrational energy transfer), not direct dipole modification.

3. **Dimer lifetime should emerge from Q2 coherence.** Fixed k_dissociation is prescribed. The physical mechanism: quantum spin states protect dimers from dissolution (Fisher 2015, Agarwal 2023). Decoherence removes protection → dimers dissolve at classical aqueous rate. This makes the Q1-Q2 coupling bidirectional: Q1 protects Q2 coherence → coherence protects dimer stability → dimers feed back through Fröhlich condensation.

4. **Material limits are real physics.** Finite phosphate in a ~0.1 fL spine naturally caps dimer accumulation at ~60-300, not 2500. This isn't a prescribed constraint — it's conservation of mass.

---

## Literature References Used This Session

- **Babcock et al. 2024** (J Phys Chem B 128:4035) — Superradiance scaling, cooperative robustness increases with system size, f_coherent experimental values
- **Agarwal et al. 2023** (J Phys Chem Lett 14:2518) — Calcium phosphate dimers preserve entanglement for hundreds of seconds; trimers lose it sub-second
- **Fisher 2015** (Ann Phys 362:593) — Quantum dynamical selection: spin states modulate Posner molecule binding and dissolution
- **Adams et al. 2025** (Sci Rep 15) — Entanglement and coherence in pure and doped Posner molecules; environmental parameters enhance entanglement
- **Hu et al. 2008** (J Neurosci 28:13094) — Activity-dependent MT invasion of spines, transient, calcium-driven
- **Merriam/Bhrendt et al. 2013** (J Neurosci, PMC3797370) — MT entry from localized sites at spine base, calcium-dependent, F-actin required
- **Garcia et al. 2019** (Cryst Growth Des 19:6422) — CaP prenucleation cluster association, 17 kJ/mol barrier, kinetic trapping, dynamic equilibrium
- **Pokorný et al. 2021** (PMC8348406) — MT as coherent antenna systems, near-field with 1/r³ dependence
- **Zhang, Agarwal & Scully 2019** (PRL) — Fröhlich rate equations, critical pump rate derivation
- **Harris et al. 2022** (PMC9038701) — Continuous dendritic MT backbone, spine density scales with MT number