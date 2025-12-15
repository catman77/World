"""
Configuration module for RSL World Simulator.

Contains all configurable parameters for the simulation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal
from enum import Enum
import json
from pathlib import Path


class BoundaryCondition(Enum):
    """Boundary conditions for the lattice."""
    PERIODIC = "periodic"  # Ring/torus
    FIXED = "fixed"        # Fixed boundary values
    OPEN = "open"          # Open boundaries


class InitialCondition(Enum):
    """Initial state generation methods."""
    ALL_PLUS = "all_plus"           # All +
    ALL_MINUS = "all_minus"         # All -
    RANDOM_UNIFORM = "random"       # Random uniform
    ALTERNATING = "alternating"     # +-+-+-...
    SINGLE_DEFECT = "single_defect" # One domain wall in center
    MULTI_DEFECT = "multi_defect"   # Multiple domain walls
    CUSTOM = "custom"               # Custom user-defined


class StopCondition(Enum):
    """Stopping criteria for evolution."""
    MAX_STEPS = "max_steps"           # Fixed number of steps
    EQUILIBRIUM = "equilibrium"       # H_micro stabilized
    CYCLE_DETECTED = "cycle"          # State cycle detected
    ENTROPY_THRESHOLD = "entropy"     # Entropy threshold reached
    MANUAL = "manual"                 # Manual stop


@dataclass
class RSLParams:
    """RSL-specific parameters."""
    # Hamiltonian parameters
    J: float = 1.0              # Coupling constant for H_micro
    
    # Capacity parameters  
    C0: float = 2.0             # Base capacity
    alpha: float = 0.5          # Capacity reduction coefficient
    
    # Phase parameters
    theta_max: float = 0.15     # Maximum phase step (rad)
    
    # Tension threshold
    H_threshold: float = 0.55   # Normalized tension threshold for RA application
    
    # Rule parameters
    L_max: int = 4              # Maximum rule pattern length


@dataclass
class OmegaParams:
    """Ω-cycle parameters."""
    # Particle types enabled
    enable_quarks: bool = True
    enable_leptons: bool = True  
    enable_gauge: bool = True
    
    # SM charge structure
    color_group: str = "Z3"     # SU(3) → Z₃ discrete
    weak_isospin: bool = True   # SU(2)_L
    hypercharge: bool = True    # U(1)_Y
    
    # Cycle detection
    max_cycle_length: int = 100
    cycle_detection_window: int = 1000


@dataclass
class ObserverParams:
    """Observer interface parameters."""
    # Coarse-graining
    coarse_radius: int = 5      # R for B_R(i) averaging
    max_coarse_levels: int = 10 # Maximum L levels (unlimited in theory)
    
    # 3D mapping
    mapping_method: Literal["hilbert", "morton", "linear"] = "hilbert"
    target_dim: int = 3         # Target spatial dimension
    
    # Interface
    interface_modes: List[str] = field(default_factory=lambda: ["visual", "haptic"])


@dataclass
class EvolutionParams:
    """Evolution engine parameters."""
    # Determinism
    deterministic: bool = True          # Strict left-to-right resolution
    parallel_independent: bool = False  # Allow parallel non-overlapping rules
    
    # Performance
    use_numba: bool = True              # Use numba JIT
    chunk_size: int = 1000              # Steps between checkpoints
    
    # History
    store_history: bool = True
    history_stride: int = 1             # Store every N-th state
    max_history_states: int = 100000    # Maximum states in memory


@dataclass
class StopParams:
    """Stopping criteria parameters."""
    condition: StopCondition = StopCondition.MAX_STEPS
    max_steps: int = 10000
    
    # Equilibrium detection
    equilibrium_window: int = 100
    equilibrium_threshold: float = 1e-6
    
    # Entropy threshold
    entropy_target: float = 0.5
    
    # Cycle detection (uses OmegaParams.cycle_detection_window)


@dataclass
class StorageParams:
    """Storage parameters."""
    format: Literal["json", "hdf5", "both"] = "hdf5"
    compress: bool = True
    base_path: Path = field(default_factory=lambda: Path("./data"))
    
    # Auto-save
    auto_save: bool = True
    auto_save_interval: int = 1000  # Steps


@dataclass
class WorldConfig:
    """
    Main configuration container for the RSL World Simulator.
    
    Example:
        config = WorldConfig(
            lattice_size=1000,
            initial_condition=InitialCondition.RANDOM_UNIFORM,
        )
        config.save("my_config.json")
    """
    # Lattice parameters
    lattice_size: int = 100
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    
    # Initial state
    initial_condition: InitialCondition = InitialCondition.RANDOM_UNIFORM
    random_seed: Optional[int] = None  # None = random seed
    
    # Alphabet (extended for Ω markers)
    base_alphabet: Tuple[str, ...] = ("+", "-")
    
    # Sub-configurations
    rsl: RSLParams = field(default_factory=RSLParams)
    omega: OmegaParams = field(default_factory=OmegaParams)
    observer: ObserverParams = field(default_factory=ObserverParams)
    evolution: EvolutionParams = field(default_factory=EvolutionParams)
    stop: StopParams = field(default_factory=StopParams)
    storage: StorageParams = field(default_factory=StorageParams)
    
    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self._to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str | Path) -> "WorldConfig":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    def _to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        def convert(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dataclass_fields__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            return obj
        
        return convert(self)
    
    @classmethod
    def _from_dict(cls, data: dict) -> "WorldConfig":
        """Reconstruct from dictionary."""
        # Convert enums back
        if 'boundary' in data:
            data['boundary'] = BoundaryCondition(data['boundary'])
        if 'initial_condition' in data:
            data['initial_condition'] = InitialCondition(data['initial_condition'])
        
        # Convert sub-configs
        if 'rsl' in data:
            data['rsl'] = RSLParams(**data['rsl'])
        if 'omega' in data:
            data['omega'] = OmegaParams(**data['omega'])
        if 'observer' in data:
            data['observer'] = ObserverParams(**data['observer'])
        if 'evolution' in data:
            data['evolution'] = EvolutionParams(**data['evolution'])
        if 'stop' in data:
            if 'condition' in data['stop']:
                data['stop']['condition'] = StopCondition(data['stop']['condition'])
            data['stop'] = StopParams(**data['stop'])
        if 'storage' in data:
            if 'base_path' in data['storage']:
                data['storage']['base_path'] = Path(data['storage']['base_path'])
            data['storage'] = StorageParams(**data['storage'])
        
        # Handle tuple for base_alphabet
        if 'base_alphabet' in data:
            data['base_alphabet'] = tuple(data['base_alphabet'])
            
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings/errors."""
        issues = []
        
        if self.lattice_size < 2:
            issues.append("lattice_size must be at least 2")
        if self.lattice_size > 10**7:
            issues.append("lattice_size > 10^7 may cause memory issues")
            
        if self.rsl.C0 <= 0:
            issues.append("C0 must be positive")
        if self.rsl.alpha < 0:
            issues.append("alpha must be non-negative")
        if not 0 < self.rsl.theta_max < 3.15:
            issues.append("theta_max should be in (0, π)")
            
        if self.observer.coarse_radius < 1:
            issues.append("coarse_radius must be at least 1")
            
        return issues


# Preset configurations
def minimal_config() -> WorldConfig:
    """Minimal configuration for quick testing."""
    return WorldConfig(
        lattice_size=20,
        stop=StopParams(max_steps=100),
        evolution=EvolutionParams(store_history=True, history_stride=1),
    )


def standard_config() -> WorldConfig:
    """Standard configuration for typical simulations."""
    return WorldConfig(
        lattice_size=100,
        random_seed=42,
        stop=StopParams(max_steps=10000),
    )


def large_scale_config() -> WorldConfig:
    """Large-scale configuration with optimizations."""
    return WorldConfig(
        lattice_size=100000,
        evolution=EvolutionParams(
            use_numba=True,
            chunk_size=10000,
            history_stride=100,
        ),
        stop=StopParams(max_steps=1000000),
        storage=StorageParams(format="hdf5", compress=True),
    )


# Aliases for backward compatibility
SimConfig = WorldConfig
RSLConfig = RSLParams
ObserverConfig = ObserverParams
AnalysisConfig = StopParams  # Analysis uses stop conditions
