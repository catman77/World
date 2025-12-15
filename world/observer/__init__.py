"""
Observer module for RSL World.

Implements the observer interface Π_obs: S → Y that projects
microscopic states to macroscopic observations.

Key components:
- Observer: Coarse-graining and projection
- CoordinateMapper: 1D→3D mapping via space-filling curves
- Interface: Observer-accessible measurements
"""

from .observer import Observer, ObserverState
from .coordinate_mapper import CoordinateMapper, HilbertMapper, MortonMapper
from .interface import Interface, Measurement
from .amplitude import AmplitudeCalculator, compute_amplitude

__all__ = [
    "Observer",
    "ObserverState",
    "CoordinateMapper",
    "HilbertMapper",
    "MortonMapper",
    "Interface",
    "Measurement",
    "AmplitudeCalculator",
    "compute_amplitude",
]
