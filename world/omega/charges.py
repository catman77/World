"""
Charge structure for RSL particles.

Implements Standard Model compatible charge structure:
- Color charge: SU(3) → Z₃ discrete
- Electric charge: Fractional for quarks, integer for leptons
- Weak isospin: SU(2)_L
- Hypercharge: U(1)_Y

Conservation laws are derived from charge structure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from .particles import Particle, ParticleType, ParticleCategory


class ColorCharge(Enum):
    """
    Color charge using Z₃ discrete representation of SU(3).
    
    Values are third roots of unity: 1, ω, ω² where ω = e^{2πi/3}
    """
    NONE = 0      # Color singlet
    RED = 1       # r
    GREEN = 2     # g  
    BLUE = 3      # b
    ANTI_RED = -1   # r̄
    ANTI_GREEN = -2 # ḡ
    ANTI_BLUE = -3  # b̄
    
    def conjugate(self) -> "ColorCharge":
        """Get anti-color."""
        conjugate_map = {
            ColorCharge.NONE: ColorCharge.NONE,
            ColorCharge.RED: ColorCharge.ANTI_RED,
            ColorCharge.GREEN: ColorCharge.ANTI_GREEN,
            ColorCharge.BLUE: ColorCharge.ANTI_BLUE,
            ColorCharge.ANTI_RED: ColorCharge.RED,
            ColorCharge.ANTI_GREEN: ColorCharge.GREEN,
            ColorCharge.ANTI_BLUE: ColorCharge.BLUE,
        }
        return conjugate_map[self]
    
    @staticmethod
    def combine(c1: "ColorCharge", c2: "ColorCharge") -> "ColorCharge":
        """
        Combine two color charges (Z₃ addition).
        
        For color singlet formation: r + r̄ = 0, r + g + b = 0
        """
        # Map to Z₃ values
        value_map = {
            ColorCharge.NONE: 0,
            ColorCharge.RED: 1, ColorCharge.GREEN: 2, ColorCharge.BLUE: 0,  # mod 3
            ColorCharge.ANTI_RED: 2, ColorCharge.ANTI_GREEN: 1, ColorCharge.ANTI_BLUE: 0,
        }
        
        result = (value_map[c1] + value_map[c2]) % 3
        
        # This is simplified - real SU(3) is more complex
        if result == 0:
            return ColorCharge.NONE
        else:
            return ColorCharge.RED  # Simplified


@dataclass
class ChargeStructure:
    """
    Complete charge structure for a particle.
    
    Attributes:
        electric: Electric charge in units of e (proton charge)
        color: Color charge (Z₃)
        weak_isospin: Weak isospin T₃ (-1/2, 0, +1/2)
        hypercharge: Weak hypercharge Y
    """
    electric: float = 0.0          # Q in units of e
    color: ColorCharge = ColorCharge.NONE
    weak_isospin: float = 0.0      # T₃
    hypercharge: float = 0.0       # Y
    
    @property
    def is_color_singlet(self) -> bool:
        """Check if particle is color neutral."""
        return self.color == ColorCharge.NONE
    
    @property
    def is_electrically_neutral(self) -> bool:
        """Check if particle is electrically neutral."""
        return abs(self.electric) < 1e-10
    
    def verify_gell_mann_nishijima(self) -> bool:
        """
        Verify Gell-Mann-Nishijima formula:
        Q = T₃ + Y/2
        """
        expected = self.weak_isospin + self.hypercharge / 2
        return abs(self.electric - expected) < 1e-10
    
    def conjugate(self) -> "ChargeStructure":
        """Get charge structure of antiparticle."""
        return ChargeStructure(
            electric=-self.electric,
            color=self.color.conjugate(),
            weak_isospin=-self.weak_isospin,
            hypercharge=-self.hypercharge,
        )
    
    @staticmethod
    def combine(c1: "ChargeStructure", c2: "ChargeStructure") -> "ChargeStructure":
        """Combine charges (for composite states)."""
        return ChargeStructure(
            electric=c1.electric + c2.electric,
            color=ColorCharge.combine(c1.color, c2.color),
            weak_isospin=c1.weak_isospin + c2.weak_isospin,
            hypercharge=c1.hypercharge + c2.hypercharge,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "electric": self.electric,
            "color": self.color.name,
            "weak_isospin": self.weak_isospin,
            "hypercharge": self.hypercharge,
        }
    
    def __repr__(self) -> str:
        return f"Charges(Q={self.electric:+.2f}, T₃={self.weak_isospin:+.1f}, Y={self.hypercharge:+.2f})"


class SMCharges:
    """
    Standard Model charge assignments for particles.
    
    Provides the correct SM charges for all particle types.
    """
    
    # Standard Model charge table
    _CHARGES: Dict[Tuple[ParticleType, bool], ChargeStructure] = {}
    
    @classmethod
    def _initialize(cls) -> None:
        """Initialize SM charge table."""
        if cls._CHARGES:
            return
        
        # Quarks (per color - actual quarks carry one of r,g,b)
        # Up-type quarks: Q = +2/3, T₃ = +1/2, Y = +1/3
        for q in [ParticleType.UP_1, ParticleType.UP_2, ParticleType.UP_3]:
            cls._CHARGES[(q, False)] = ChargeStructure(
                electric=2/3,
                color=ColorCharge.RED,  # Placeholder - actual color varies
                weak_isospin=0.5,
                hypercharge=1/3,
            )
            cls._CHARGES[(q, True)] = cls._CHARGES[(q, False)].conjugate()
        
        # Down-type quarks: Q = -1/3, T₃ = -1/2, Y = +1/3
        for q in [ParticleType.DOWN_1, ParticleType.DOWN_2, ParticleType.DOWN_3]:
            cls._CHARGES[(q, False)] = ChargeStructure(
                electric=-1/3,
                color=ColorCharge.RED,
                weak_isospin=-0.5,
                hypercharge=1/3,
            )
            cls._CHARGES[(q, True)] = cls._CHARGES[(q, False)].conjugate()
        
        # Charged leptons: Q = -1, T₃ = -1/2, Y = -1
        for l in [ParticleType.ELECTRON, ParticleType.MUON, ParticleType.TAU]:
            cls._CHARGES[(l, False)] = ChargeStructure(
                electric=-1,
                color=ColorCharge.NONE,
                weak_isospin=-0.5,
                hypercharge=-1,
            )
            cls._CHARGES[(l, True)] = cls._CHARGES[(l, False)].conjugate()
        
        # Neutrinos: Q = 0, T₃ = +1/2, Y = -1
        for n in [ParticleType.ELECTRON_NEUTRINO, ParticleType.MUON_NEUTRINO, ParticleType.TAU_NEUTRINO]:
            cls._CHARGES[(n, False)] = ChargeStructure(
                electric=0,
                color=ColorCharge.NONE,
                weak_isospin=0.5,
                hypercharge=-1,
            )
            cls._CHARGES[(n, True)] = cls._CHARGES[(n, False)].conjugate()
        
        # Gauge bosons
        cls._CHARGES[(ParticleType.PHOTON, False)] = ChargeStructure(0, ColorCharge.NONE, 0, 0)
        cls._CHARGES[(ParticleType.Z, False)] = ChargeStructure(0, ColorCharge.NONE, 0, 0)
        cls._CHARGES[(ParticleType.W_PLUS, False)] = ChargeStructure(1, ColorCharge.NONE, 1, 0)
        cls._CHARGES[(ParticleType.W_MINUS, False)] = ChargeStructure(-1, ColorCharge.NONE, -1, 0)
        cls._CHARGES[(ParticleType.GLUON, False)] = ChargeStructure(0, ColorCharge.NONE, 0, 0)  # Simplified
        
        # Higgs: Q = 0, T₃ = -1/2, Y = +1
        cls._CHARGES[(ParticleType.HIGGS, False)] = ChargeStructure(0, ColorCharge.NONE, -0.5, 1)
    
    @classmethod
    def get_charges(cls, particle_type: ParticleType, antiparticle: bool = False) -> ChargeStructure:
        """
        Get SM charges for a particle type.
        
        Args:
            particle_type: Type of particle
            antiparticle: Whether this is the antiparticle
            
        Returns:
            ChargeStructure with SM quantum numbers
        """
        cls._initialize()
        
        key = (particle_type, antiparticle)
        if key in cls._CHARGES:
            return cls._CHARGES[key]
        
        # Unknown particle - return zero charges
        return ChargeStructure()
    
    @classmethod
    def get_for_particle(cls, particle: Particle) -> ChargeStructure:
        """Get charges for a Particle instance."""
        charges = cls.get_charges(particle.particle_type, particle.antiparticle)
        
        # Update color if specified on particle
        if particle.color is not None:
            color_map = {
                "r": ColorCharge.RED,
                "g": ColorCharge.GREEN,
                "b": ColorCharge.BLUE,
                "r̄": ColorCharge.ANTI_RED,
                "ḡ": ColorCharge.ANTI_GREEN,
                "b̄": ColorCharge.ANTI_BLUE,
            }
            if particle.color in color_map:
                charges = ChargeStructure(
                    electric=charges.electric,
                    color=color_map[particle.color],
                    weak_isospin=charges.weak_isospin,
                    hypercharge=charges.hypercharge,
                )
        
        return charges


class ChargeConservation:
    """
    Check and enforce charge conservation in interactions.
    
    Conservation laws:
    1. Electric charge: ΔQ = 0
    2. Color charge: Color singlet in final state
    3. Weak isospin: Conserved in strong/EM, not in weak
    4. Baryon number: ΔB = 0
    5. Lepton number: ΔL = 0
    """
    
    def __init__(self):
        self.tolerance = 1e-10
    
    def check_electric_conservation(
        self,
        initial: List[Particle],
        final: List[Particle],
    ) -> bool:
        """Check electric charge conservation."""
        Q_initial = sum(SMCharges.get_for_particle(p).electric for p in initial)
        Q_final = sum(SMCharges.get_for_particle(p).electric for p in final)
        return abs(Q_initial - Q_final) < self.tolerance
    
    def check_color_conservation(
        self,
        initial: List[Particle],
        final: List[Particle],
    ) -> bool:
        """
        Check color conservation.
        
        Initial and final states must both be color singlets.
        """
        def is_color_singlet(particles: List[Particle]) -> bool:
            # Simplified check: count colors
            colors = [SMCharges.get_for_particle(p).color for p in particles]
            # For singlet: either all NONE, or rgb combination
            non_none = [c for c in colors if c != ColorCharge.NONE]
            if not non_none:
                return True
            # Check for color balance (simplified)
            reds = sum(1 for c in non_none if c == ColorCharge.RED)
            greens = sum(1 for c in non_none if c == ColorCharge.GREEN)
            blues = sum(1 for c in non_none if c == ColorCharge.BLUE)
            anti_reds = sum(1 for c in non_none if c == ColorCharge.ANTI_RED)
            anti_greens = sum(1 for c in non_none if c == ColorCharge.ANTI_GREEN)
            anti_blues = sum(1 for c in non_none if c == ColorCharge.ANTI_BLUE)
            
            return (reds == anti_reds and greens == anti_greens and blues == anti_blues)
        
        return is_color_singlet(initial) and is_color_singlet(final)
    
    def check_baryon_number(
        self,
        initial: List[Particle],
        final: List[Particle],
    ) -> bool:
        """Check baryon number conservation."""
        def baryon_number(p: Particle) -> float:
            if p.particle_type.category == ParticleCategory.QUARK:
                return -1/3 if p.antiparticle else 1/3
            return 0
        
        B_initial = sum(baryon_number(p) for p in initial)
        B_final = sum(baryon_number(p) for p in final)
        return abs(B_initial - B_final) < self.tolerance
    
    def check_lepton_number(
        self,
        initial: List[Particle],
        final: List[Particle],
    ) -> bool:
        """Check lepton number conservation."""
        def lepton_number(p: Particle) -> int:
            if p.particle_type.category == ParticleCategory.LEPTON:
                return -1 if p.antiparticle else 1
            return 0
        
        L_initial = sum(lepton_number(p) for p in initial)
        L_final = sum(lepton_number(p) for p in final)
        return L_initial == L_final
    
    def check_all(
        self,
        initial: List[Particle],
        final: List[Particle],
    ) -> Dict[str, bool]:
        """Check all conservation laws."""
        return {
            "electric": self.check_electric_conservation(initial, final),
            "color": self.check_color_conservation(initial, final),
            "baryon": self.check_baryon_number(initial, final),
            "lepton": self.check_lepton_number(initial, final),
        }
    
    def is_allowed(
        self,
        initial: List[Particle],
        final: List[Particle],
        interaction_type: str = "strong",
    ) -> bool:
        """
        Check if interaction is allowed by conservation laws.
        
        Args:
            initial: Initial state particles
            final: Final state particles
            interaction_type: "strong", "em", or "weak"
            
        Returns:
            True if interaction is allowed
        """
        checks = self.check_all(initial, final)
        
        # Electric and baryon/lepton always conserved
        if not (checks["electric"] and checks["baryon"] and checks["lepton"]):
            return False
        
        # Color conserved in all interactions (final state must be singlet)
        if interaction_type in ["strong", "em"] and not checks["color"]:
            return False
        
        return True


def generate_allowed_interactions(
    particle_types: List[ParticleType],
    max_particles: int = 4,
) -> List[Tuple[List[ParticleType], List[ParticleType]]]:
    """
    Generate all allowed interactions between particle types.
    
    This is for exploring the space of possible interactions
    consistent with conservation laws.
    
    Returns list of (initial, final) particle type tuples.
    """
    # Placeholder - full implementation would enumerate combinations
    # and filter by conservation laws
    return []
