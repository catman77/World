"""
Particle types and registry for RSL Ω-cycles.

Maps Ω-cycles to Standard Model particle types:
- Q (quarks): Fractional charges, carry color
- L (leptons): Integer charges, no color
- G (gauge): Force carriers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum, auto
import numpy as np

from .cycles import OmegaCycle


class ParticleCategory(Enum):
    """Broad category of particle."""
    QUARK = auto()
    LEPTON = auto()
    GAUGE_BOSON = auto()
    HIGGS = auto()
    UNKNOWN = auto()


class ParticleType(Enum):
    """
    Standard Model particle types.
    
    Naming convention: name_generation (for fermions)
    """
    # Quarks (Q)
    UP_1 = "u"
    DOWN_1 = "d"
    UP_2 = "c"       # charm
    DOWN_2 = "s"     # strange
    UP_3 = "t"       # top
    DOWN_3 = "b"     # bottom
    
    # Leptons (L)
    ELECTRON = "e"
    ELECTRON_NEUTRINO = "νe"
    MUON = "μ"
    MUON_NEUTRINO = "νμ"
    TAU = "τ"
    TAU_NEUTRINO = "ντ"
    
    # Gauge bosons (G)
    PHOTON = "γ"
    W_PLUS = "W+"
    W_MINUS = "W-"
    Z = "Z"
    GLUON = "g"
    
    # Higgs
    HIGGS = "H"
    
    # Unknown/unclassified
    UNKNOWN = "?"
    
    @property
    def category(self) -> ParticleCategory:
        """Get particle category."""
        if self.value in ["u", "d", "c", "s", "t", "b"]:
            return ParticleCategory.QUARK
        elif self.value in ["e", "νe", "μ", "νμ", "τ", "ντ"]:
            return ParticleCategory.LEPTON
        elif self.value in ["γ", "W+", "W-", "Z", "g"]:
            return ParticleCategory.GAUGE_BOSON
        elif self.value == "H":
            return ParticleCategory.HIGGS
        else:
            return ParticleCategory.UNKNOWN
    
    @property
    def is_fermion(self) -> bool:
        """Check if particle is a fermion (spin 1/2)."""
        return self.category in [ParticleCategory.QUARK, ParticleCategory.LEPTON]
    
    @property
    def is_boson(self) -> bool:
        """Check if particle is a boson (integer spin)."""
        return self.category in [ParticleCategory.GAUGE_BOSON, ParticleCategory.HIGGS]


@dataclass
class Particle:
    """
    A particle instance identified from an Ω-cycle.
    
    Attributes:
        cycle: The underlying Ω-cycle
        particle_type: Identified particle type
        antiparticle: Whether this is an antiparticle
        generation: Generation number (1, 2, 3 for fermions)
        color: Color charge ("r", "g", "b", or None)
        spin: Spin state (+1/2, -1/2 for fermions)
    """
    cycle: OmegaCycle
    particle_type: ParticleType = ParticleType.UNKNOWN
    antiparticle: bool = False
    generation: int = 1
    color: Optional[str] = None  # "r", "g", "b" for quarks
    spin: float = 0.0
    
    # Derived properties
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def position(self) -> float:
        """Current position from cycle."""
        return self.cycle.position
    
    @property
    def velocity(self) -> float:
        """Velocity from cycle."""
        return self.cycle.velocity
    
    @property
    def symbol(self) -> str:
        """Standard symbol with antiparticle bar."""
        base = self.particle_type.value
        if self.antiparticle:
            return f"ā{base}"  # Using combining overline approximation
        return base
    
    @property
    def name(self) -> str:
        """Full particle name."""
        prefix = "anti-" if self.antiparticle else ""
        name_map = {
            ParticleType.UP_1: "up quark",
            ParticleType.DOWN_1: "down quark",
            ParticleType.UP_2: "charm quark",
            ParticleType.DOWN_2: "strange quark",
            ParticleType.UP_3: "top quark",
            ParticleType.DOWN_3: "bottom quark",
            ParticleType.ELECTRON: "electron",
            ParticleType.ELECTRON_NEUTRINO: "electron neutrino",
            ParticleType.MUON: "muon",
            ParticleType.MUON_NEUTRINO: "muon neutrino",
            ParticleType.TAU: "tau",
            ParticleType.TAU_NEUTRINO: "tau neutrino",
            ParticleType.PHOTON: "photon",
            ParticleType.W_PLUS: "W+ boson",
            ParticleType.W_MINUS: "W- boson",
            ParticleType.Z: "Z boson",
            ParticleType.GLUON: "gluon",
            ParticleType.HIGGS: "Higgs boson",
        }
        return prefix + name_map.get(self.particle_type, "unknown particle")
    
    def conjugate(self) -> "Particle":
        """Create antiparticle."""
        return Particle(
            cycle=self.cycle,
            particle_type=self.particle_type,
            antiparticle=not self.antiparticle,
            generation=self.generation,
            color=self._conjugate_color(),
            spin=-self.spin,
            metadata=self.metadata.copy(),
        )
    
    def _conjugate_color(self) -> Optional[str]:
        """Get conjugate color (anti-color)."""
        if self.color is None:
            return None
        return {"r": "r̄", "g": "ḡ", "b": "b̄"}.get(self.color, self.color)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle": self.cycle.to_dict(),
            "particle_type": self.particle_type.value,
            "antiparticle": self.antiparticle,
            "generation": self.generation,
            "color": self.color,
            "spin": self.spin,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        color_str = f"[{self.color}]" if self.color else ""
        return f"Particle({self.symbol}{color_str}, pos={self.position:.1f})"


class ParticleRegistry:
    """
    Registry of detected particles in the simulation.
    
    Maintains a collection of Particle instances and provides:
    - Classification of Ω-cycles into particle types
    - Tracking of particle creation/annihilation
    - Statistics on particle populations
    """
    
    def __init__(self):
        self._particles: List[Particle] = []
        self._creation_history: List[Tuple[int, Particle]] = []  # (time, particle)
        self._annihilation_history: List[Tuple[int, Particle]] = []
        
        # Classification rules (cycle signature -> particle type)
        self._classification_rules: Dict[str, ParticleType] = {}
    
    def register(self, particle: Particle, time: int = 0) -> None:
        """Register a new particle."""
        self._particles.append(particle)
        self._creation_history.append((time, particle))
    
    def unregister(self, particle: Particle, time: int = 0) -> bool:
        """Unregister a particle (annihilation)."""
        if particle in self._particles:
            self._particles.remove(particle)
            self._annihilation_history.append((time, particle))
            return True
        return False
    
    def classify_cycle(self, cycle: OmegaCycle) -> Particle:
        """
        Classify an Ω-cycle as a particle type.
        
        Uses cycle properties to determine particle type:
        - Period -> generation/mass
        - Extent -> localization
        - Velocity -> momentum
        - Signature -> quantum numbers
        """
        # Default classification based on cycle properties
        particle_type = ParticleType.UNKNOWN
        antiparticle = False
        generation = 1
        color = None
        
        # Check explicit classification rules
        if cycle.signature in self._classification_rules:
            particle_type = self._classification_rules[cycle.signature]
        else:
            # Heuristic classification based on cycle properties
            particle_type, antiparticle, generation, color = self._heuristic_classify(cycle)
        
        return Particle(
            cycle=cycle,
            particle_type=particle_type,
            antiparticle=antiparticle,
            generation=generation,
            color=color,
        )
    
    def _heuristic_classify(
        self, 
        cycle: OmegaCycle,
    ) -> Tuple[ParticleType, bool, int, Optional[str]]:
        """
        Heuristic classification of cycle.
        
        This is a placeholder for more sophisticated classification
        based on RSL theory predictions.
        """
        # Use cycle period as proxy for mass/generation
        if cycle.period <= 5:
            generation = 1
        elif cycle.period <= 15:
            generation = 2
        else:
            generation = 3
        
        # Use extent to distinguish quarks vs leptons
        # (quarks are more localized due to confinement)
        if cycle.extent <= 3:
            # Quark-like
            particle_type = ParticleType.UP_1 if generation == 1 else ParticleType.UP_2
            color = ["r", "g", "b"][hash(cycle.signature) % 3]
        else:
            # Lepton-like
            particle_type = ParticleType.ELECTRON if generation == 1 else ParticleType.MUON
            color = None
        
        # Antiparticle from signature parity
        antiparticle = hash(cycle.signature) % 2 == 1
        
        return particle_type, antiparticle, generation, color
    
    def add_classification_rule(
        self, 
        signature: str, 
        particle_type: ParticleType,
    ) -> None:
        """Add explicit classification rule for a cycle signature."""
        self._classification_rules[signature] = particle_type
    
    @property
    def particles(self) -> List[Particle]:
        """Current particle list."""
        return list(self._particles)
    
    def count_by_type(self) -> Dict[ParticleType, int]:
        """Count particles by type."""
        counts: Dict[ParticleType, int] = {}
        for p in self._particles:
            counts[p.particle_type] = counts.get(p.particle_type, 0) + 1
        return counts
    
    def count_by_category(self) -> Dict[ParticleCategory, int]:
        """Count particles by category."""
        counts: Dict[ParticleCategory, int] = {}
        for p in self._particles:
            cat = p.particle_type.category
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    def get_quarks(self) -> List[Particle]:
        """Get all quark particles."""
        return [p for p in self._particles if p.particle_type.category == ParticleCategory.QUARK]
    
    def get_leptons(self) -> List[Particle]:
        """Get all lepton particles."""
        return [p for p in self._particles if p.particle_type.category == ParticleCategory.LEPTON]
    
    def get_gauge_bosons(self) -> List[Particle]:
        """Get all gauge boson particles."""
        return [p for p in self._particles if p.particle_type.category == ParticleCategory.GAUGE_BOSON]
    
    def particle_density(self, lattice_size: int) -> float:
        """Compute particle density."""
        return len(self._particles) / lattice_size if lattice_size > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all particles."""
        self._particles.clear()
    
    def __len__(self) -> int:
        return len(self._particles)
    
    def __iter__(self):
        return iter(self._particles)
