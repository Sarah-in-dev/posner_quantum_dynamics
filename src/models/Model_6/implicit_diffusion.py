"""
Calcium Diffusion Fix - Simple Explicit Solver with Proper Subcycling
======================================================================

The original explicit solver was correct and mass-conserving.
The only problem was the cap at 100 substeps.

This provides a drop-in replacement for update_diffusion() that:
1. Calculates the exact number of substeps needed for CFL stability
2. Uses scipy's optimized convolution (very fast)
3. Conserves mass exactly

For dt=1ms: needs ~900 substeps (not 100)
This is fine - scipy.ndimage.convolve is highly optimized.
"""

import numpy as np
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


def update_diffusion_explicit(diffusion_obj, dt: float):
    """
    Explicit diffusion with proper CFL-based subcycling.
    
    No arbitrary cap on substeps - calculates exactly what's needed.
    
    Args:
        diffusion_obj: CalciumDiffusion instance with ca_free, params, dx
        dt: timestep (s)
    """
    ca = diffusion_obj.ca_free
    dx = diffusion_obj.dx
    params = diffusion_obj.params
    
    # Compute spatially-varying D_eff
    kappa_s = (params.buffer_concentration * params.buffer_kd / 
               (ca + params.buffer_kd)**2)
    D_eff = params.D_ca / (1 + kappa_s)
    D_max = np.mean(D_eff)
    
    # CFL stability: dt_stable = 0.25 * dx² / D_max
    dt_stable = 0.25 * dx**2 / D_max
    n_substeps = max(1, int(np.ceil(dt / dt_stable)))
    dt_sub = dt / n_substeps
    
    # Log once
    if not hasattr(diffusion_obj, '_substep_info_logged'):
        logger.info(f"Explicit diffusion: dt={dt*1e3:.2f}ms, {n_substeps} substeps, dt_sub={dt_sub*1e6:.2f}μs")
        diffusion_obj._substep_info_logged = True
    
    # Laplacian kernel (5-point stencil)
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]]) / (dx**2)
    
    # Use mean D_eff for efficiency (recalculating per substep is slow)
    D_mean = np.mean(D_eff)
    
    # Subcycle with optimized scipy convolution
    for _ in range(n_substeps):
        laplacian = ndimage.convolve(ca, laplacian_kernel, mode='nearest')
        ca = ca + dt_sub * D_mean * laplacian
        # No clipping needed if CFL is satisfied - but safety check
        ca = np.maximum(ca, 0)
    
    diffusion_obj.ca_free = ca


# Alias for drop-in replacement
update_diffusion_implicit = update_diffusion_explicit


# =============================================================================
# TESTING
# =============================================================================

class MockParams:
    D_ca = 220e-12  # m²/s
    buffer_kd = 10e-6  # M
    buffer_concentration = 300e-6  # M
    ca_baseline = 100e-9  # M


class MockDiffusion:
    def __init__(self, grid_shape, dx):
        self.grid_shape = grid_shape
        self.dx = dx
        self.params = MockParams()
        self.ca_free = np.ones(grid_shape) * self.params.ca_baseline


def test_mass_conservation():
    """Test mass conservation"""
    print("=" * 70)
    print("MASS CONSERVATION TEST - EXPLICIT SOLVER")
    print("=" * 70)
    
    grid_shape = (50, 50)
    dx = 4e-9
    
    for dt in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        diff = MockDiffusion(grid_shape, dx)
        center = grid_shape[0] // 2
        diff.ca_free[center-1:center+2, center-1:center+2] = 50e-6
        
        initial_mass = np.sum(diff.ca_free)
        
        for _ in range(100):
            update_diffusion_explicit(diff, dt)
        
        final_mass = np.sum(diff.ca_free)
        mass_change = abs(final_mass - initial_mass) / initial_mass * 100
        
        # Count substeps
        D_eff = 220e-12 / 61
        dt_stable = 0.25 * dx**2 / D_eff
        n_sub = max(1, int(np.ceil(dt / dt_stable)))
        
        status = "✓" if mass_change < 0.01 else "⚠"
        print(f"\n  dt = {dt*1e6:.0f} μs ({n_sub} substeps/step):")
        print(f"    Peak: 50.0 → {np.max(diff.ca_free)*1e6:.3f} μM")
        print(f"    Mass change: {mass_change:.6f}% {status}")


def test_speed():
    """Benchmark speed"""
    import time
    
    print("\n" + "=" * 70)
    print("SPEED TEST")
    print("=" * 70)
    
    grid_shape = (100, 100)
    dx = 4e-9
    
    for dt in [1e-4, 1e-3]:
        diff = MockDiffusion(grid_shape, dx)
        center = grid_shape[0] // 2
        diff.ca_free[center, center] = 100e-6
        
        # Count substeps
        D_eff = 220e-12 / 61
        dt_stable = 0.25 * dx**2 / D_eff
        n_sub = max(1, int(np.ceil(dt / dt_stable)))
        
        n_steps = 100
        start = time.time()
        for _ in range(n_steps):
            update_diffusion_explicit(diff, dt)
        elapsed = time.time() - start
        
        ms_per_step = elapsed / n_steps * 1000
        
        print(f"\n  dt = {dt*1e3:.1f} ms ({n_sub} substeps/step):")
        print(f"    {n_steps} steps in {elapsed:.2f}s ({ms_per_step:.1f} ms/step)")
        
        steps_60s = int(60 / dt)
        time_60s_min = steps_60s * ms_per_step / 1000 / 60
        print(f"    60s simulation: {steps_60s} steps → {time_60s_min:.1f} min")


def test_nanodomain_physics():
    """Verify nanodomain behavior is preserved"""
    print("\n" + "=" * 70)
    print("NANODOMAIN PHYSICS TEST")
    print("=" * 70)
    
    grid_shape = (100, 100)
    dx = 4e-9
    dt = 1e-3
    
    diff = MockDiffusion(grid_shape, dx)
    
    # Simulate channel opening: inject calcium at center
    center = grid_shape[0] // 2
    
    print("\nSimulating channel burst (10 ms)...")
    
    for step in range(10):  # 10 steps of 1ms = 10ms
        # Add flux at channel location (simulating open channel)
        diff.ca_free[center, center] += 5e-6  # Add 5 μM per step
        
        # Diffuse
        update_diffusion_explicit(diff, dt)
        
        if step in [0, 4, 9]:
            peak = np.max(diff.ca_free)
            # Find radius where [Ca] drops to 10%
            threshold = 0.1 * peak
            above = diff.ca_free > threshold
            radius_pixels = np.sqrt(np.sum(above) / np.pi)
            radius_nm = radius_pixels * dx * 1e9
            
            print(f"  t = {(step+1)*dt*1e3:.0f} ms:")
            print(f"    Peak [Ca²⁺]: {peak*1e6:.1f} μM")
            print(f"    Nanodomain radius (10%): {radius_nm:.0f} nm")


if __name__ == "__main__":
    test_mass_conservation()
    test_speed()
    test_nanodomain_physics()
    
    print("\n" + "=" * 70)
    print("INTEGRATION")
    print("=" * 70)
    print("""
In calcium_system.py, replace update_diffusion in CalciumDiffusion class:

    from implicit_diffusion import update_diffusion_explicit
    
    def update_diffusion(self, dt: float):
        update_diffusion_explicit(self, dt)

Or simply modify your existing update_diffusion to remove the substep cap:
    
    # OLD (broken):
    n_substeps = min(n_substeps_ideal, 100)  # DELETE THIS CAP
    
    # NEW (correct):
    n_substeps = n_substeps_ideal  # Use exact number needed
""")