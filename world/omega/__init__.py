"""
Omega (Ω) cycle module for RSL particles.

Ω-cycles are closed sequences of rule applications that:
1. Return to the original state (cyclic)
2. Maintain localized defects (particle-like)
3. Can carry charges (SM compatible)

Particle types:
- Q (quarks): Fractional charges, color Z₃
- L (leptons): Integer charges, no color
- G (gauge): Mediate interactions
"""

from .cycles import OmegaCycle, CycleDetector, find_omega_cycles
from .particles import Particle, ParticleType, ParticleRegistry
from .charges import ChargeStructure, SMCharges, ChargeConservation

__all__ = [
    "OmegaCycle",
    "CycleDetector",
    "find_omega_cycles",
    "Particle",
    "ParticleType",
    "ParticleRegistry",
    "ChargeStructure",
    "SMCharges",
    "ChargeConservation",
]
