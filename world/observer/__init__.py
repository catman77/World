"""
Observer module for RSL World.

Implements the observer interface Π_obs: S → Y that projects
microscopic states to macroscopic observations.

Key components:
- GlobalObserver: Main observer with semantic learning
- Observer: Coarse-graining and projection (legacy)
- CoordinateMapper: 1D→3D mapping via space-filling curves
- MortonMapper: Z-order curve for 16³ grids
- IFACE: Observable state structures (objects, fields)
- SemanticState: Observer's learned knowledge
"""

from .observer import Observer, ObserverState
from .coordinate_mapper import CoordinateMapper, HilbertMapper
from .zorder import MortonMapper, morton_encode, morton_decode, idx_to_xyz, xyz_to_idx
from .interface import Interface, Measurement
from .amplitude import AmplitudeCalculator, compute_amplitude

# New IFACE structures
from .iface import (
    IFACEObject, IFACEField, IFACEState, IFACEHistory,
    ParticleType
)

# Semantic state
from .semantics import (
    SemanticState, SemanticHistory,
    FieldEquationParams, ConservationLaw, GravityLaw, EventStatistics
)

# Global observer
from .global_observer import GlobalObserver, ObserverConfig, create_observer

__all__ = [
    # Legacy
    "Observer",
    "ObserverState",
    "CoordinateMapper",
    "HilbertMapper",
    "Interface",
    "Measurement",
    "AmplitudeCalculator",
    "compute_amplitude",
    
    # Z-order mapping
    "MortonMapper",
    "morton_encode",
    "morton_decode",
    "idx_to_xyz",
    "xyz_to_idx",
    
    # IFACE structures
    "IFACEObject",
    "IFACEField", 
    "IFACEState",
    "IFACEHistory",
    "ParticleType",
    
    # Semantic state
    "SemanticState",
    "SemanticHistory",
    "FieldEquationParams",
    "ConservationLaw",
    "GravityLaw",
    "EventStatistics",
    
    # Global observer
    "GlobalObserver",
    "ObserverConfig",
    "create_observer",
    
    # OBS Fitness (для эволюционного поиска)
    "OBSFitness",
    "OBSFitnessConfig",
    "OBSFitnessComponents",
    "CombinedFitness",
    "CombinedFitnessConfig",
    "evaluate_observer_fitness",
    
    # TDA (топологический анализ)
    "SemanticTDA",
    "TopologicalSummary",
    "PersistentHomology",
    "analyze_semantic_trajectory",
]

# OBS Fitness
from .fitness import (
    OBSFitness, OBSFitnessConfig, OBSFitnessComponents,
    CombinedFitness, CombinedFitnessConfig,
    evaluate_observer_fitness,
)

# TDA
from .tda import (
    SemanticTDA, TopologicalSummary, PersistentHomology,
    analyze_semantic_trajectory,
)
