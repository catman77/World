"""
RSL (Reversible Symmetry Lattice) module.

Contains:
- RSLFilter: F1 tension-based filters for rule activation
- TensionCalculator: H_micro(S) = J * M (domain wall energy)
- CapacityCalculator: C_i(S) = C0 - α * h_i(S)

These implement the core RSL physics for emergent dynamics.
"""

from .filters import RSLFilter, F1Filter
from .tension import TensionCalculator, local_tension
from .capacity import CapacityCalculator, local_capacity

__all__ = [
    "RSLFilter",
    "F1Filter",
    "TensionCalculator",
    "local_tension",
    "CapacityCalculator",
    "local_capacity",
]
