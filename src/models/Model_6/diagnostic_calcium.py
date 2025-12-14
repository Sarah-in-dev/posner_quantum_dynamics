"""
Diagnostic Script for Calcium Diffusion Stability
==================================================

Tests the CFL stability condition and proposes solutions for the 
multi-synapse framework.

PROBLEM: Explicit finite difference diffusion violates CFL condition
at the timesteps needed for efficient simulation.
"""

import numpy as np
from scipy import ndimage

# ============================================================================
# PARAMETERS FROM MODEL 6
# ============================================================================

# Grid
GRID_SIZE = 100
ACTIVE_ZONE_RADIUS = 200e-9  # m (200 nm)
DX = 2 * ACTIVE_ZONE_RADIUS / GRID_SIZE  # = 4 nm

# Calcium diffusion
D_CA_FREE = 220e-12  # m²/s (free calcium)
BUFFER_CAPACITY = 60  # dimensionless κ_s
D_CA_EFF = D_CA_FREE / (1 + BUFFER_CAPACITY)  # ~3.6 μm²/s when buffered

# Baseline
CA_BASELINE = 100e-9  # M (100 nM)


def calculate_cfl_limit(dx, D):
    """Calculate maximum stable timestep for explicit diffusion"""
    dt_max = 0.25 * dx**2 / D
    return dt_max


def test_diffusion_stability(dt, n_steps=100, show_progress=True):
    """
    Test diffusion solver stability at given timestep
    
    Returns:
        dict with max_concentration, stable (bool), and notes
    """
    # Initialize grid with small perturbation
    ca = np.ones((GRID_SIZE, GRID_SIZE)) * CA_BASELINE
    
    # Add a nanodomain spike at center (simulating channel opening)
    center = GRID_SIZE // 2
    ca[center-1:center+2, center-1:center+2] = 50e-6  # 50 μM nanodomain
    
    # Use effective D (buffered)
    D_eff = D_CA_EFF
    
    # CFL analysis
    dt_stable = calculate_cfl_limit(DX, D_eff)
    cfl_ratio = dt / dt_stable
    
    # Subcycling (as currently implemented)
    n_substeps = min(int(np.ceil(dt / dt_stable)), 100)  # Cap at 100
    dt_sub = dt / n_substeps
    
    # Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]]) / (DX**2)
    
    max_vals = [np.max(ca)]
    
    for step in range(n_steps):
        # Subcycled diffusion
        for _ in range(n_substeps):
            laplacian = ndimage.convolve(ca, laplacian_kernel, mode='constant')
            ca += dt_sub * D_eff * laplacian
            ca = np.maximum(ca, 0)  # No negative concentrations
        
        max_vals.append(np.max(ca))
        
        # Check for explosion
        if np.max(ca) > 1e-3 or not np.isfinite(np.max(ca)):  # > 1 mM or NaN
            return {
                'stable': False,
                'exploded_at_step': step,
                'max_concentration': np.max(ca),
                'cfl_ratio': cfl_ratio,
                'n_substeps': n_substeps,
                'effective_cfl': dt_sub / dt_stable,
                'max_history': max_vals
            }
    
    return {
        'stable': True,
        'max_concentration': np.max(ca),
        'final_mean': np.mean(ca),
        'cfl_ratio': cfl_ratio,
        'n_substeps': n_substeps,
        'effective_cfl': dt_sub / dt_stable,
        'max_history': max_vals
    }


def test_implicit_diffusion(dt, n_steps=100):
    """
    Test implicit (backward Euler) diffusion - unconditionally stable
    
    Uses iterative Jacobi solver for simplicity.
    """
    ca = np.ones((GRID_SIZE, GRID_SIZE)) * CA_BASELINE
    center = GRID_SIZE // 2
    ca[center-1:center+2, center-1:center+2] = 50e-6
    
    D_eff = D_CA_EFF
    alpha = D_eff * dt / (DX**2)  # Diffusion number
    
    max_vals = [np.max(ca)]
    
    for step in range(n_steps):
        # Implicit solve: (I - α∇²)c^{n+1} = c^n
        # Using Jacobi iteration (simple but slow)
        ca_new = ca.copy()
        
        for iteration in range(50):  # Fixed iterations
            ca_old = ca_new.copy()
            
            for i in range(1, GRID_SIZE-1):
                for j in range(1, GRID_SIZE-1):
                    neighbors = (ca_new[i-1,j] + ca_new[i+1,j] + 
                                ca_new[i,j-1] + ca_new[i,j+1])
                    ca_new[i,j] = (ca[i,j] + alpha * neighbors) / (1 + 4*alpha)
            
            # Check convergence
            if np.max(np.abs(ca_new - ca_old)) < 1e-12:
                break
        
        ca = ca_new
        max_vals.append(np.max(ca))
    
    return {
        'stable': True,  # Implicit is always stable
        'max_concentration': np.max(ca),
        'final_mean': np.mean(ca),
        'max_history': max_vals
    }


def print_cfl_analysis():
    """Print detailed CFL analysis for different scenarios"""
    print("=" * 70)
    print("CFL STABILITY ANALYSIS FOR MODEL 6 CALCIUM DIFFUSION")
    print("=" * 70)
    
    print(f"\nGrid Parameters:")
    print(f"  Grid size: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  dx = {DX*1e9:.1f} nm")
    print(f"  Active zone: {ACTIVE_ZONE_RADIUS*1e9:.0f} nm radius")
    
    print(f"\nDiffusion Coefficients:")
    print(f"  D_ca (free): {D_CA_FREE*1e12:.0f} μm²/s")
    print(f"  D_ca (buffered): {D_CA_EFF*1e12:.1f} μm²/s (κ_s = {BUFFER_CAPACITY})")
    
    # CFL limits
    dt_free = calculate_cfl_limit(DX, D_CA_FREE)
    dt_buff = calculate_cfl_limit(DX, D_CA_EFF)
    
    print(f"\nCFL Stability Limits:")
    print(f"  Free calcium: dt_max = {dt_free*1e9:.1f} ns")
    print(f"  Buffered calcium: dt_max = {dt_buff*1e9:.1f} ns")
    
    print("\n" + "-" * 70)
    print("TIMESTEP COMPARISON")
    print("-" * 70)
    
    timesteps = [
        (1e-6, "1 μs (original default)"),
        (1e-5, "10 μs"),
        (1e-4, "100 μs"),
        (1e-3, "1 ms (runner.py setting)"),
    ]
    
    for dt, name in timesteps:
        ratio_free = dt / dt_free
        ratio_buff = dt / dt_buff
        substeps_needed = int(np.ceil(dt / dt_buff))
        substeps_capped = min(substeps_needed, 100)
        effective_ratio = (dt / substeps_capped) / dt_buff
        
        print(f"\n  {name}:")
        print(f"    CFL ratio (free): {ratio_free:.0f}×")
        print(f"    CFL ratio (buffered): {ratio_buff:.0f}×")
        print(f"    Substeps needed: {substeps_needed}")
        print(f"    Substeps (capped at 100): {substeps_capped}")
        print(f"    Effective CFL after subcycling: {effective_ratio:.1f}×")
        
        if effective_ratio > 1:
            print(f"    ⚠️  UNSTABLE - will explode!")
        else:
            print(f"    ✓ Stable")


def run_stability_tests():
    """Run actual stability tests at different timesteps"""
    print("\n" + "=" * 70)
    print("STABILITY TESTS (100 steps each)")
    print("=" * 70)
    
    timesteps = [1e-6, 1e-5, 1e-4, 1e-3]
    
    for dt in timesteps:
        print(f"\nTesting dt = {dt*1e6:.0f} μs...")
        result = test_diffusion_stability(dt, n_steps=100)
        
        if result['stable']:
            print(f"  ✓ STABLE")
            print(f"    Peak [Ca²⁺]: {result['max_concentration']*1e6:.2f} μM")
            print(f"    Mean [Ca²⁺]: {result['final_mean']*1e6:.3f} μM")
        else:
            print(f"  ✗ EXPLODED at step {result['exploded_at_step']}")
            print(f"    Final [Ca²⁺]: {result['max_concentration']:.2e} M")
        
        print(f"    Substeps used: {result['n_substeps']}")
        print(f"    Effective CFL: {result['effective_cfl']:.1f}×")


def print_solutions():
    """Print recommended solutions"""
    print("\n" + "=" * 70)
    print("RECOMMENDED SOLUTIONS")
    print("=" * 70)
    
    print("""
1. QUICK FIX: Revert runner.py timestep
   ─────────────────────────────────────
   Change: params.simulation.dt_diffusion = 1e-3  →  1e-6
   
   Pros: Simple, model was working at 1 μs
   Cons: 1000× slower simulation (may be okay for development)

2. INCREASE GRID SPACING (Coarser Grid)
   ─────────────────────────────────────
   Current: dx = 4 nm, grid = 100×100
   Proposed: dx = 40 nm, grid = 10×10
   
   dt_max = 0.25 × (40 nm)² / (3.6 μm²/s) ≈ 1.1 μs
   
   With 1 ms timestep: need ~900 substeps
   Still problematic, but better.
   
   Better option: dx = 100 nm, grid = 4×4
   dt_max ≈ 7 μs → feasible with moderate subcycling
   
   Pros: Faster simulation
   Cons: Loses nanodomain resolution (defeats physics purpose)

3. IMPLICIT SOLVER (Crank-Nicolson or ADI)
   ───────────────────────────────────────
   Unconditionally stable at ANY timestep!
   
   Implementation:
   - Crank-Nicolson: (I - αΔ/2)c^{n+1} = (I + αΔ/2)c^n
   - Use scipy.sparse.linalg.spsolve or iterative solver
   
   Pros: Large timesteps, maintains physics
   Cons: More complex implementation, ~2-3× slower per step
   
   RECOMMENDED for production code.

4. ANALYTICAL NANODOMAIN MODEL
   ───────────────────────────────
   Skip numerical diffusion entirely!
   
   Use: Naraghi & Neher 1997 analytical solution
   [Ca](r,t) = (I/4πDr) × erfc(r/√(4Dt))
   
   Pros: Exact physics, no stability issues, FAST
   Cons: Assumes linear superposition (okay for your case)
   
   RECOMMENDED for speed-critical simulations.

5. ADAPTIVE TIMESTEPPING
   ─────────────────────────
   Small dt during channel bursts (when nanodomains matter)
   Large dt during quiescent periods
   
   Pros: Best of both worlds
   Cons: Complex to implement correctly
""")


if __name__ == "__main__":
    print_cfl_analysis()
    run_stability_tests()
    print_solutions()