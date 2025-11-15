"""
Protocol Comparison Utility
============================

Compare results from two or more protocols.

Usage:
    python compare_protocols.py 01 02          # Compare baseline vs isotope
    python compare_protocols.py 01 04a 04b     # Compare baseline vs theta bursts
    python compare_protocols.py 03a 01 03b     # Temperature series

Author: Sarah Davidson
Date: November 2025
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_latest_protocol_results(protocol_number: str) -> dict:
    """Load most recent results for a protocol"""
    
    # Search for protocol directories
    protocol_dir = Path(__file__).parent / "experimental_protocols"
    
    if not protocol_dir.exists():
        raise FileNotFoundError(f"No experimental_protocols directory found")
    
    # Find all directories for this protocol
    pattern = f"protocol_{protocol_number}_*"
    matching_dirs = sorted(protocol_dir.glob(pattern))
    
    if not matching_dirs:
        raise FileNotFoundError(f"No results found for protocol {protocol_number}")
    
    # Get most recent (last in sorted list)
    latest_dir = matching_dirs[-1]
    
    # Load statistics.json
    stats_file = latest_dir / "statistics.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"No statistics.json in {latest_dir}")
    
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    return data


def compare_protocols(protocol_ids: list):
    """Compare multiple protocols"""
    
    print("="*80)
    print("PROTOCOL COMPARISON")
    print("="*80)
    print()
    
    # Load all protocol results
    all_results = {}
    for pid in protocol_ids:
        try:
            data = load_latest_protocol_results(pid)
            all_results[pid] = data
            print(f"✓ Loaded Protocol {pid}: {data['protocol']['name']}")
        except FileNotFoundError as e:
            print(f"✗ Error loading Protocol {pid}: {e}")
            return
    
    print()
    
    # Extract key metrics for comparison
    metrics_to_compare = [
        'dimer_accumulation_nM',
        'T2_coherence_s',
        'j_coupling_peak_Hz',
        'calcium_peak_uM'
    ]
    
    # Print comparison table
    print("KEY METRICS COMPARISON:")
    print("-"*80)
    print(f"{'Metric':<30} ", end="")
    for pid in protocol_ids:
        print(f"Protocol {pid:<8} ", end="")
    print()
    print("-"*80)
    
    for metric in metrics_to_compare:
        label = metric.replace('_', ' ').title()
        print(f"{label:<30} ", end="")
        
        for pid in protocol_ids:
            stats = all_results[pid]['statistics'][metric]
            mean = stats['mean']
            std = stats['std']
            
            if 'nm' in metric.lower():
                print(f"{mean:>6.1f}±{std:<5.1f} nM ", end="")
            elif 'hz' in metric.lower():
                print(f"{mean:>6.1f}±{std:<5.1f} Hz ", end="")
            elif 't2' in metric.lower():
                print(f"{mean:>6.1f}±{std:<5.1f} s  ", end="")
            else:
                print(f"{mean:>6.2f}±{std:<5.2f}    ", end="")
        
        print()
    
    print("-"*80)
    print()
    
    # Calculate fold changes if comparing exactly 2 protocols
    if len(protocol_ids) == 2:
        print("FOLD CHANGES (Protocol {} vs {})".format(protocol_ids[1], protocol_ids[0]))
        print("-"*80)
        
        for metric in metrics_to_compare:
            baseline = all_results[protocol_ids[0]]['statistics'][metric]['mean']
            test = all_results[protocol_ids[1]]['statistics'][metric]['mean']
            
            if baseline > 0:
                fold_change = test / baseline
                label = metric.replace('_', ' ').title()
                print(f"{label:<30} {fold_change:>6.2f}x")
        
        print()
        
        # Special interpretation for isotope comparison (01 vs 02)
        if protocol_ids[0] == '01' and protocol_ids[1] == '02':
            print("ISOTOPE SUBSTITUTION INTERPRETATION:")
            print("-"*80)
            T2_P31 = all_results['01']['statistics']['T2_coherence_s']['mean']
            T2_P32 = all_results['02']['statistics']['T2_coherence_s']['mean']
            ratio = T2_P31 / T2_P32
            
            print(f"T2 ratio (³¹P / ³²P): {ratio:.1f}x")
            print(f"Expected ratio: ~10x")
            
            if ratio > 5:
                print("✓ QUANTUM MECHANISM NECESSARY: Isotope effect confirms quantum coherence")
            else:
                print("✗ QUANTUM MECHANISM NOT SUPPORTED: Insufficient isotope effect")
            print()
    
    # Generate comparison plot
    _generate_comparison_plot(all_results, protocol_ids, metrics_to_compare)


def _generate_comparison_plot(all_results: dict, protocol_ids: list, metrics: list):
    """Generate side-by-side comparison plot"""
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    x = np.arange(len(protocol_ids))
    width = 0.6
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        means = [all_results[pid]['statistics'][metric]['mean'] for pid in protocol_ids]
        stds = [all_results[pid]['statistics'][metric]['std'] for pid in protocol_ids]
        
        ax.bar(x, means, width, yerr=stds, capsize=5, 
               alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Protocol')
        ax.set_xticks(x)
        ax.set_xticklabels(protocol_ids)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            if 'nm' in metric.lower() or 'um' in metric.lower():
                label = f"{mean:.0f}"
            elif 't2' in metric.lower() or 'hz' in metric.lower():
                label = f"{mean:.1f}"
            else:
                label = f"{mean:.2f}"
            ax.text(j, mean + std, label, ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f"Protocol Comparison: {', '.join(protocol_ids)}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(__file__).parent / f"comparison_{'_'.join(protocol_ids)}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_protocols.py <protocol1> <protocol2> [protocol3 ...]")
        print()
        print("Examples:")
        print("  python compare_protocols.py 01 02          # Baseline vs isotope")
        print("  python compare_protocols.py 01 04a 04b     # Baseline vs theta bursts")
        print("  python compare_protocols.py 03a 01 03b     # Temperature series")
        sys.exit(1)
    
    protocol_ids = sys.argv[1:]
    compare_protocols(protocol_ids)