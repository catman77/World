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
from .wormhole import WormholeLayer, WormholeConfig, create_wormhole_layer
from .omega_cycle import (
    OmegaCycleConfig, OmegaSignature, OmegaCycleDetector,
    ResonanceTrigger, create_omega_detector, create_resonance_trigger
)
from .antigravity import (
    AntigravityConfig, ChiField, GeometryInverter, AntigravityLayer,
    create_antigravity_layer
)
from .stone import (
    TargetType, TargetSpec, StoneConfig, ParameterSpace,
    WorldEvaluator, EvolutionStrategy, StoneMechanism,
    make_density_target, make_custom_target,
    create_stone_for_world
)

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
    # FTL / Wormhole layer
    "WormholeLayer",
    "WormholeConfig",
    "create_wormhole_layer",
    # Ω-cycle detection & resonance triggers
    "OmegaCycleConfig",
    "OmegaSignature", 
    "OmegaCycleDetector",
    "ResonanceTrigger",
    "create_omega_detector",
    "create_resonance_trigger",
    # Antigravity (χ-field & geometry inversion)
    "AntigravityConfig",
    "ChiField",
    "GeometryInverter",
    "AntigravityLayer",
    "create_antigravity_layer",
    # Stone mechanism (probability control via Evolution Strategy)
    "TargetType",
    "TargetSpec",
    "StoneConfig",
    "ParameterSpace",
    "WorldEvaluator",
    "EvolutionStrategy",
    "StoneMechanism",
    "make_density_target",
    "make_custom_target",
    "create_stone_for_world",
]
