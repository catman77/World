"""
RSL World Simulator

A 1D reversible lattice simulator based on RSL (Reversible Symmetry Lattice) theory.
Implements symbolic string dynamics with Ω-cycles, coarse-graining, and observer interface.

Main components:
- core: String representation, rules, evolution engine
- rsl: RSL filters, tension, capacity calculations
- omega: Ω-cycles, particle types (Q/L/G), charges
- observer: Observer Π_obs, 1D→3D mapping, interface
- analysis: Topology, statistics, phase structure
- visualization: Graphs, time series
- storage: JSON/HDF5 persistence
"""

__version__ = "0.1.0"
__author__ = "RSL World Team"

from .core import Lattice, Rule, RuleSet, EvolutionEngine
from .rsl import RSLFilter, TensionCalculator, CapacityCalculator
from .omega import OmegaCycle, ParticleType, ChargeStructure
from .observer import Observer, CoordinateMapper, Interface
from .config import WorldConfig

__all__ = [
    "Lattice",
    "Rule", 
    "RuleSet",
    "EvolutionEngine",
    "RSLFilter",
    "TensionCalculator",
    "CapacityCalculator",
    "OmegaCycle",
    "ParticleType",
    "ChargeStructure",
    "Observer",
    "CoordinateMapper",
    "Interface",
    "WorldConfig",
]
