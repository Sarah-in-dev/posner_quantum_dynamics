#!/usr/bin/env python
"""Compare P31 vs P32 isotopes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_runner import ExperimentRunner
from config import get_config
import matplotlib.pyplot as plt

def main():
    """Run isotope comparison"""
    print("Comparing ³¹P vs ³²P...")
    
    runner = ExperimentRunner(get_config('physiological'))
    results = runner.run_isotope_comparison()
    
    # Display results
    print(f"\nCoherence time ratio: {results['isotope_effect']['coherence_ratio']:.1f}x")
    print(f"Enhancement ratio: {results['isotope_effect']['enhancement_ratio']:.1f}x")
    
    # Create comparison plot
    # ... plotting code ...

if __name__ == "__main__":
    main()