"""
Core module for RSL World Simulator.

Contains:
- Lattice: 1D string representation [L0: s0 | s1 | ... | s_{N-1}]
- Rule: Local transformation rules T_i
- RuleSet: Collection of rules with conflict resolution
- EvolutionEngine: State evolution with deterministic dynamics
- GraphStructure: Power-law graph for effective 3D gravity
- World: Combined spin + φ-field dynamics
"""

from .lattice import Lattice, LatticeState
from .rules import Rule, RuleSet, RuleConflict
from .evolution import EvolutionEngine, EvolutionResult
from .graph_structure import (
    GraphStructure, GraphConfig,
    build_powerlaw_edges, create_graph_for_gravity,
    theoretical_alpha_for_ds,
)
from .world import World, WorldConfig, create_world_with_gravity

__all__ = [
    "Lattice",
    "LatticeState",
    "Rule",
    "RuleSet", 
    "RuleConflict",
    "EvolutionEngine",
    "EvolutionResult",
    # Graph structure
    "GraphStructure",
    "GraphConfig",
    "build_powerlaw_edges",
    "create_graph_for_gravity",
    "theoretical_alpha_for_ds",
    # World with gravity
    "World",
    "WorldConfig",
    "create_world_with_gravity",
]
