"""
Protocol 01: Baseline Statistical Characterization
===================================================

Single spike protocol to establish baseline stochastic variability.

This is the reference condition for all other protocols.

Expected Results:
- Calcium: ~5-10 μM peak (stochastic)
- Dimers: ~300-500 nM (high variability, CV > 100%)
- J-coupling: ~19 Hz (deterministic, CV < 15%)
- T2: ~40-50s (emergent from isolated dimers)

Author: Sarah Davidson
Date: November 2025
"""

from protocol_runner import ProtocolRunner
from protocol_configs import PROTOCOL_01_BASELINE

if __name__ == "__main__":
    runner = ProtocolRunner(PROTOCOL_01_BASELINE)
    results = runner.run()
    
    # Print key findings
    print("\nKEY FINDINGS:")
    stats = results['statistics']
    print(f"  • Dimer formation: {stats['dimer_accumulation_nM']['mean']:.1f} ± "
          f"{stats['dimer_accumulation_nM']['std']:.1f} nM")
    print(f"  • T2 coherence: {stats['T2_coherence_s']['mean']:.1f} ± "
          f"{stats['T2_coherence_s']['std']:.1f} s")
    print(f"  • Coefficient of variation: {stats['dimer_accumulation_nM']['cv']:.1f}% "
          "(biological realism)")