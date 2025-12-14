# debug_calcium_system.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model6_parameters import Model6Parameters
import numpy as np

params = Model6Parameters()

print("=== CALCIUM PARAMETERS ===")
print(f"Baseline: {params.calcium.ca_baseline * 1e6:.3f} μM")
print(f"Channels per site: {params.calcium.n_channels_per_site}")
print(f"Single channel current: {params.calcium.single_channel_current * 1e12:.2f} pA")
print(f"Channel open time: {params.calcium.channel_open_time * 1e3:.2f} ms")
print(f"Pump Vmax: {params.calcium.pump_vmax * 1e6:.1f} μM/s")
print(f"Pump Km: {params.calcium.pump_km * 1e6:.2f} μM")
print(f"Buffer capacity: {params.calcium.buffer_capacity_kappa_s}")

# Calculate expected influx
n_ch = params.calcium.n_channels_per_site
I_ch = params.calcium.single_channel_current  # A
t_open = params.calcium.channel_open_time  # s
e = 1.6e-19  # C
z = 2  # Ca²⁺ valence

ions_per_opening = (I_ch * t_open) / (z * e)
print(f"\n=== EXPECTED CALCIUM INFLUX ===")
print(f"Ions per channel opening: {ions_per_opening:.2e}")
print(f"Total ions (50 channels): {ions_per_opening * n_ch:.2e}")

# In what volume?
# If active zone: 1.57e-18 L
az_vol_L = 1.57e-18
ions_total = ions_per_opening * n_ch
moles = ions_total / 6.022e23
conc_M = moles / az_vol_L
print(f"Expected peak [Ca²⁺]: {conc_M * 1e6:.1f} μM")

# Check extrusion rate
vmax = params.calcium.pump_vmax  # M/s
km = params.calcium.pump_km  # M
ca_high = 50e-6  # 50 μM
pump_rate = vmax * ca_high / (km + ca_high)
print(f"\n=== EXPECTED EXTRUSION ===")
print(f"At 50 μM Ca²⁺: pump rate = {pump_rate * 1e6:.1f} μM/s")
print(f"Time to clear 50 μM: {50e-6 / pump_rate * 1000:.1f} ms")