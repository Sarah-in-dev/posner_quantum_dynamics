"""
TALON Permutation Engine — Core Classes
========================================
Extracted from the Murmur Health TALON framework for use in the
Model 6 quantum biology parameter sweep.

Contains only the domain-agnostic engine:
  - RefinedDimension  — one parameter axis with values/ranges
  - TestVector        — one combination of parameter values (one simulation run)
  - TestSuite         — collection of vectors with coverage stats
  - CoveringArrayGenerator — greedy N-wise covering array generator
  - TestVectorBuilder — converts covering rows to TestVector objects

No Murmur-specific imports, classes, or data.
"""

import random
import math
import itertools
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RefinedDimension:
    """A parameter axis ready for covering array generation."""
    dim_id: str                     # Unique identifier
    variable: str                   # Parameter name in the model
    category: str                   # 'threshold', 'boolean', 'enum', 'range'
    values: List[Any]               # Possible test values
    value_labels: List[str]         # Human-readable labels for each value
    source_file: str                # File where parameter lives
    source_function: str            # Class/function where parameter lives
    source_line: int                # Line number (0 if unknown)
    condition: str                  # Description of what this parameter controls
    importance: str = "high"        # 'critical', 'high', 'medium', 'low'
    notes: str = ""


@dataclass
class TestVector:
    """A single row in the covering array — one simulation run."""
    vector_id: int
    chain_id: str
    chain_name: str
    values: Dict[str, Any] = field(default_factory=dict)   # dim_id → chosen value
    labels: Dict[str, str] = field(default_factory=dict)   # dim_id → human label
    description: str = ""


@dataclass
class TestSuite:
    """Complete sweep suite — dimensions + generated vectors + coverage stats."""
    chain_id: str
    chain_name: str
    route: str
    total_branches: int
    refined_dimensions: List[RefinedDimension]
    test_vectors: List[TestVector]
    coverage_strength: int

    # Stats
    raw_dimensions: int = 0
    filtered_dimensions: int = 0
    theoretical_exhaustive: int = 0
    actual_tests: int = 0
    reduction_ratio: str = ""


# ============================================================================
# COVERING ARRAY GENERATOR — N-wise combinatorial coverage
# ============================================================================

class CoveringArrayGenerator:
    """
    Generate N-wise covering arrays using a greedy algorithm.

    For t-wise coverage (t = coverage_strength):
    - Every combination of any t dimension values appears in at least one
      test vector.
    - 2-wise (pairwise): every pair of values from any 2 dimensions.
    - 3-wise: every triple of values from any 3 dimensions.

    Uses greedy 'one-test-at-a-time' algorithm:
    1. Enumerate all t-wise tuples that need covering.
    2. Repeatedly select the test vector that covers the most uncovered tuples.
    3. Stop when all tuples are covered or max_tests is reached.
    """

    def __init__(self, coverage_strength: int = 3, max_tests: int = 5000,
                 seed: int = 42):
        self.t = coverage_strength
        self.max_tests = max_tests
        self.rng = random.Random(seed)

    def generate(self, dimensions: List[RefinedDimension]) -> List[Dict[str, Any]]:
        """
        Generate covering array rows.

        Args:
            dimensions: List of RefinedDimension instances.

        Returns:
            List of dicts mapping dim_id → chosen value, one per test vector.
        """
        if not dimensions:
            return []

        k = len(dimensions)
        t = min(self.t, k)

        dim_values_eff = [list(range(len(d.values))) for d in dimensions]

        # For very large spaces, limit to critical/high dimensions for 3-wise
        effective_dims = dimensions
        if k > 50:
            if t >= 3:
                effective_dims = [d for d in dimensions
                                  if d.importance in ('critical', 'high')]
                if len(effective_dims) < 5:
                    effective_dims = dimensions[:30]
            else:
                effective_dims = dimensions[:60]

        k_eff = len(effective_dims)
        t_eff = min(t, k_eff)

        if k_eff == 0:
            return []

        dim_values_eff = [list(range(len(d.values))) for d in effective_dims]
        dim_indices = list(range(k_eff))

        # Enumerate all t-wise tuples
        uncovered = set()
        for combo in itertools.combinations(dim_indices, t_eff):
            value_lists = [dim_values_eff[i] for i in combo]
            for val_combo in itertools.product(*value_lists):
                uncovered.add((combo, val_combo))

        total_tuples = len(uncovered)
        if total_tuples == 0:
            return []

        # Cap if tuple space is massive
        if total_tuples > 1_000_000:
            uncovered_list = list(uncovered)
            self.rng.shuffle(uncovered_list)
            uncovered = set(uncovered_list[:500_000])
            total_tuples = len(uncovered)

        # Greedy generation
        tests = []
        iteration = 0

        while uncovered and len(tests) < self.max_tests:
            iteration += 1
            best_test = None
            best_coverage = 0

            num_candidates = min(100, max(20, total_tuples // 100))

            for _ in range(num_candidates):
                candidate = tuple(
                    self.rng.choice(vals) for vals in dim_values_eff
                )
                covered = 0
                for combo in itertools.combinations(dim_indices, t_eff):
                    val_combo = tuple(candidate[i] for i in combo)
                    if (combo, val_combo) in uncovered:
                        covered += 1
                if covered > best_coverage:
                    best_coverage = covered
                    best_test = candidate

            if best_test is None or best_coverage == 0:
                break

            test_dict = {}
            for i, dim in enumerate(effective_dims):
                val_idx = best_test[i]
                test_dict[dim.dim_id] = dim.values[val_idx]
            tests.append(test_dict)

            for combo in itertools.combinations(dim_indices, t_eff):
                val_combo = tuple(best_test[i] for i in combo)
                uncovered.discard((combo, val_combo))

            if iteration % 50 == 0:
                pct = (1 - len(uncovered) / total_tuples) * 100
                print(f"  {iteration} vectors, {pct:.1f}% coverage, "
                      f"{len(uncovered)} tuples remaining")

        # Fill excluded dimensions with random values
        if k_eff < k:
            excluded_dims = [d for d in dimensions
                             if d.dim_id not in {e.dim_id for e in effective_dims}]
            for test in tests:
                for dim in excluded_dims:
                    test[dim.dim_id] = self.rng.choice(dim.values)

        return tests


# ============================================================================
# TEST VECTOR BUILDER — Convert covering array rows to TestVector objects
# ============================================================================

class TestVectorBuilder:
    """Convert raw covering array rows into descriptive TestVector objects."""

    def __init__(self, dimensions: List[RefinedDimension]):
        self.dimensions = {d.dim_id: d for d in dimensions}

    def build(self, chain_id: str, chain_name: str,
              covering_rows: List[Dict[str, Any]]) -> List[TestVector]:
        """Convert covering array rows to TestVector objects."""
        vectors = []

        for i, row in enumerate(covering_rows):
            labels = {}
            for dim_id, value in row.items():
                dim = self.dimensions.get(dim_id)
                if dim:
                    try:
                        val_idx = dim.values.index(value)
                        labels[dim_id] = dim.value_labels[val_idx]
                    except (ValueError, IndexError):
                        labels[dim_id] = str(value)
                else:
                    labels[dim_id] = str(value)

            # Build description from critical/high dimensions
            desc_parts = []
            for dim_id, value in row.items():
                dim = self.dimensions.get(dim_id)
                if dim and dim.importance in ('critical', 'high'):
                    label = labels.get(dim_id, str(value))
                    desc_parts.append(f"{dim.variable}={label}")

            description = ", ".join(desc_parts[:8])
            if len(desc_parts) > 8:
                description += f" (+{len(desc_parts) - 8} more)"

            vectors.append(TestVector(
                vector_id=i + 1,
                chain_id=chain_id,
                chain_name=chain_name,
                values=row,
                labels=labels,
                description=description,
            ))

        return vectors