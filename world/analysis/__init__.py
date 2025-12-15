"""
Analysis module for RSL World.

Provides mathematical analysis tools:
- Topology: Fundamental groups, winding numbers
- Statistics: Distributions, correlations
- Metrics: Distance measures, invariants
- Phase structure: Phase transitions, order parameters
"""

from .topology import (
    TopologyAnalyzer,
    compute_winding_number,
    find_topological_defects,
)
from .statistics import (
    compute_autocorrelation,
    compute_power_spectrum,
    StatisticsCalculator,
)
from .metrics import (
    hamming_distance,
    edit_distance,
    overlap,
    MetricsCalculator,
)
from .phase_structure import (
    PhaseAnalyzer,
    compute_order_parameter,
    find_phase_transitions,
)

# Aliases for compatibility
compute_winding_numbers = lambda lattice: {'total': compute_winding_number(lattice.sites if hasattr(lattice, 'sites') else lattice)}
count_defects = lambda lattice: {'domain_walls': len(find_topological_defects(lattice.sites if hasattr(lattice, 'sites') else lattice))}
compute_braid_invariants = lambda lattice: {}  # Placeholder
compute_susceptibility = lambda lattice: PhaseAnalyzer().susceptibility(lattice.sites if hasattr(lattice, 'sites') else lattice)
compute_binder_cumulant = lambda lattice: PhaseAnalyzer().binder_cumulant(lattice.sites if hasattr(lattice, 'sites') else lattice)

__all__ = [
    # Topology
    "TopologyAnalyzer",
    "compute_winding_number",
    "compute_winding_numbers",
    "find_topological_defects",
    "count_defects",
    "compute_braid_invariants",
    # Statistics
    "compute_autocorrelation",
    "compute_power_spectrum",
    "StatisticsCalculator",
    # Metrics
    "hamming_distance",
    "edit_distance",
    "overlap",
    "MetricsCalculator",
    # Phase structure
    "PhaseAnalyzer",
    "compute_order_parameter",
    "compute_susceptibility",
    "compute_binder_cumulant",
]
