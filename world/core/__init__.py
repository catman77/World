"""
Core module for RSL World Simulator.

Contains:
- Lattice: 1D string representation [L0: s0 | s1 | ... | s_{N-1}]
- Rule: Local transformation rules T_i
- RuleSet: Collection of rules with conflict resolution
- EvolutionEngine: State evolution with deterministic dynamics
"""

from .lattice import Lattice, LatticeState
from .rules import Rule, RuleSet, RuleConflict
from .evolution import EvolutionEngine, EvolutionResult

__all__ = [
    "Lattice",
    "LatticeState",
    "Rule",
    "RuleSet", 
    "RuleConflict",
    "EvolutionEngine",
    "EvolutionResult",
]
