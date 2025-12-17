"""
IFACE - Observer Interface structures for RSL World.

IFACE represents what the observer "sees":
- IFACEObject: Particles (Ω-cycles) with mass, charge, position, velocity
- IFACEField: Continuous fields (ϕ, capacity) in 3D grid
- IFACEState: Complete snapshot of observable world at time t

This implements the projection Π_obs: S(t) → IFACE(t)
where S is the microscopic 1D state and IFACE is the macroscopic 3D view.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .zorder import MortonMapper, morton_decode, morton_encode
from .coordinate_mapper import Coordinates3D


class ParticleType(Enum):
    """Classification of Ω-particle types."""
    UNKNOWN = "unknown"
    LEPTON = "lepton"       # Light particles (short period)
    QUARK = "quark"         # Medium particles (medium period)
    BOSON = "boson"         # Gauge-like (symmetric pattern)
    RESONANCE = "resonance" # Heavy/unstable (long period)


@dataclass
class IFACEObject:
    """
    A particle-like object in the observer's 3D space.
    
    Represents an Ω-cycle mapped to 3D coordinates with
    observable properties: mass, charge, position, velocity.
    
    Attributes:
        id: Unique identifier
        omega_type: Classification of particle type
        mass: Effective mass (H_core of Ω-cycle)
        Q: Electric-like charge (-1, 0, +1)
        pos: 3D position (x, y, z)
        vel: 3D velocity (vx, vy, vz)
        period: Original Ω-cycle period
        support_size: Spatial extent in 1D
        creation_time: When first detected
    """
    id: int
    omega_type: ParticleType = ParticleType.UNKNOWN
    mass: float = 0.0
    Q: float = 0.0
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    period: int = 1
    support_size: int = 1
    creation_time: int = 0
    
    # Additional quantum numbers (for future SM-like extension)
    B: float = 0.0      # Baryon number
    L: float = 0.0      # Lepton number
    color: int = 0      # Color charge (0,1,2 for r,g,b)
    
    # Tracking
    previous_pos: Optional[Tuple[float, float, float]] = None
    
    def __post_init__(self):
        # Classify by period if not set
        if self.omega_type == ParticleType.UNKNOWN:
            self.omega_type = self._classify_by_period()
    
    def _classify_by_period(self) -> ParticleType:
        """Classify particle type based on period."""
        if self.period <= 4:
            return ParticleType.LEPTON
        elif self.period <= 12:
            return ParticleType.QUARK
        elif self.period <= 20:
            return ParticleType.BOSON
        else:
            return ParticleType.RESONANCE
    
    @property 
    def speed(self) -> float:
        """Magnitude of velocity."""
        return np.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)
    
    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy 0.5 * m * v²."""
        return 0.5 * self.mass * self.speed**2
    
    def distance_to(self, other: "IFACEObject") -> float:
        """Euclidean distance to another object."""
        dx = self.pos[0] - other.pos[0]
        dy = self.pos[1] - other.pos[1]
        dz = self.pos[2] - other.pos[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def update_velocity(self, new_pos: Tuple[float, float, float], dt: float = 1.0) -> None:
        """Update velocity from position change."""
        if self.previous_pos is not None:
            self.vel = (
                (new_pos[0] - self.previous_pos[0]) / dt,
                (new_pos[1] - self.previous_pos[1]) / dt,
                (new_pos[2] - self.previous_pos[2]) / dt,
            )
        self.previous_pos = self.pos
        self.pos = new_pos
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "omega_type": self.omega_type.value,
            "mass": self.mass,
            "Q": self.Q,
            "pos": self.pos,
            "vel": self.vel,
            "period": self.period,
            "support_size": self.support_size,
            "creation_time": self.creation_time,
            "B": self.B,
            "L": self.L,
            "color": self.color,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IFACEObject":
        """Deserialize from dictionary."""
        data = data.copy()
        data["omega_type"] = ParticleType(data.get("omega_type", "unknown"))
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"IFACEObject(id={self.id}, type={self.omega_type.value}, "
                f"m={self.mass:.2f}, Q={self.Q:+.1f}, "
                f"pos=({self.pos[0]:.2f},{self.pos[1]:.2f},{self.pos[2]:.2f}))")


@dataclass
class IFACEField:
    """
    Continuous field representation in 3D grid.
    
    Contains:
    - phi: Coarse-grained field ϕ_R (order parameter)
    - capacity: Local capacity C (relates to time dilation)
    - tension: Local tension/energy density H
    
    Grid dimensions match the Morton mapper (16³ = 4096 by default).
    """
    phi: np.ndarray          # Shape: (n, n, n), coarse field
    capacity: np.ndarray     # Shape: (n, n, n), local capacity
    tension: np.ndarray      # Shape: (n, n, n), local tension
    
    dim_size: int = 16       # Grid dimension
    scale: float = 1.0       # Physical scale
    
    def __post_init__(self):
        # Validate shapes
        n = self.dim_size
        if self.phi.shape != (n, n, n):
            raise ValueError(f"phi must be ({n},{n},{n}), got {self.phi.shape}")
        if self.capacity.shape != (n, n, n):
            raise ValueError(f"capacity must be ({n},{n},{n}), got {self.capacity.shape}")
        if self.tension.shape != (n, n, n):
            raise ValueError(f"tension must be ({n},{n},{n}), got {self.tension.shape}")
    
    @classmethod
    def zeros(cls, dim_size: int = 16, scale: float = 1.0) -> "IFACEField":
        """Create zero-initialized field."""
        n = dim_size
        return cls(
            phi=np.zeros((n, n, n), dtype=np.float64),
            capacity=np.ones((n, n, n), dtype=np.float64),  # Default capacity = 1
            tension=np.zeros((n, n, n), dtype=np.float64),
            dim_size=dim_size,
            scale=scale,
        )
    
    @classmethod
    def from_1d(
        cls,
        phi_1d: np.ndarray,
        capacity_1d: np.ndarray,
        tension_1d: np.ndarray,
        mapper: Optional[MortonMapper] = None,
    ) -> "IFACEField":
        """
        Create field from 1D arrays using Morton mapping.
        
        Args:
            phi_1d: 1D coarse field
            capacity_1d: 1D capacity values  
            tension_1d: 1D tension values
            mapper: Morton mapper (creates default if None)
        """
        if mapper is None:
            # Determine order from array size
            n = int(round(len(phi_1d) ** (1/3)))
            order = int(np.log2(n)) if n > 0 else 4
            mapper = MortonMapper(order=order)
        
        return cls(
            phi=mapper.create_3d_grid(phi_1d),
            capacity=mapper.create_3d_grid(capacity_1d),
            tension=mapper.create_3d_grid(tension_1d),
            dim_size=mapper.dim_size,
            scale=mapper.scale,
        )
    
    def value_at(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Get field values at continuous coordinates.
        
        Returns:
            (phi, capacity, tension) at the point
        """
        n = self.dim_size
        # Convert to grid indices
        xi = int(x * n / self.scale) % n
        yi = int(y * n / self.scale) % n
        zi = int(z * n / self.scale) % n
        
        return (
            float(self.phi[xi, yi, zi]),
            float(self.capacity[xi, yi, zi]),
            float(self.tension[xi, yi, zi]),
        )
    
    def gradient_phi(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient of phi field.
        
        Returns:
            (grad_x, grad_y, grad_z) arrays
        """
        # Use central differences with periodic boundaries
        grad_x = np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)
        grad_y = np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)
        grad_z = np.roll(self.phi, -1, axis=2) - np.roll(self.phi, 1, axis=2)
        
        # Normalize by 2*dx
        dx = self.scale / self.dim_size
        return grad_x / (2*dx), grad_y / (2*dx), grad_z / (2*dx)
    
    def laplacian_phi(self) -> np.ndarray:
        """Compute Laplacian of phi field."""
        dx = self.scale / self.dim_size
        lap = (
            np.roll(self.phi, -1, axis=0) + np.roll(self.phi, 1, axis=0) +
            np.roll(self.phi, -1, axis=1) + np.roll(self.phi, 1, axis=1) +
            np.roll(self.phi, -1, axis=2) + np.roll(self.phi, 1, axis=2) -
            6 * self.phi
        ) / (dx**2)
        return lap
    
    def potential_from_capacity(self, C0: float = 2.0) -> np.ndarray:
        """
        Compute gravitational potential from capacity.
        
        Φ = C0 - C (lower capacity = deeper potential well)
        """
        return C0 - self.capacity
    
    @property
    def total_energy(self) -> float:
        """Total tension/energy in the field."""
        return float(np.sum(self.tension))
    
    @property
    def mean_phi(self) -> float:
        """Mean field value."""
        return float(np.mean(self.phi))
    
    @property
    def mean_capacity(self) -> float:
        """Mean capacity."""
        return float(np.mean(self.capacity))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phi": self.phi.tolist(),
            "capacity": self.capacity.tolist(),
            "tension": self.tension.tolist(),
            "dim_size": self.dim_size,
            "scale": self.scale,
        }


@dataclass
class IFACEState:
    """
    Complete observable state at time t.
    
    This is what the observer "sees" after projection Π_obs.
    
    Attributes:
        t: Simulation time step
        tau: Observer proper time (can differ due to time dilation)
        objects: List of particle-like objects
        field: Continuous field data
        metadata: Additional information
    """
    t: int                              # Simulation time
    tau: float = 0.0                    # Observer proper time
    objects: List[IFACEObject] = dataclass_field(default_factory=list)
    field: Optional[IFACEField] = None
    
    # Aggregate observables
    total_Q: float = 0.0                # Total charge (from objects)
    total_mass: float = 0.0             # Total mass
    total_energy: float = 0.0           # Total energy
    
    # Global topological charge (lattice-wide, independent of object count)
    global_Q: float = 0.0               # Topological charge: N(+/-) - N(-/+)
    
    # Metadata
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    
    def __post_init__(self):
        self._compute_totals()
    
    def _compute_totals(self):
        """Compute aggregate quantities."""
        self.total_Q = sum(obj.Q for obj in self.objects)
        self.total_mass = sum(obj.mass for obj in self.objects)
        self.total_energy = sum(obj.mass + obj.kinetic_energy for obj in self.objects)
        # global_Q is set externally from lattice analysis
        if self.field is not None:
            self.total_energy += self.field.total_energy
    
    @property
    def num_objects(self) -> int:
        """Number of particle-like objects."""
        return len(self.objects)
    
    def get_object_by_id(self, obj_id: int) -> Optional[IFACEObject]:
        """Find object by ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None
    
    def objects_by_type(self, ptype: ParticleType) -> List[IFACEObject]:
        """Get objects of specific type."""
        return [obj for obj in self.objects if obj.omega_type == ptype]
    
    def center_of_mass(self) -> Tuple[float, float, float]:
        """Compute center of mass of all objects."""
        if not self.objects or self.total_mass == 0:
            return (0.0, 0.0, 0.0)
        
        cx = sum(obj.mass * obj.pos[0] for obj in self.objects) / self.total_mass
        cy = sum(obj.mass * obj.pos[1] for obj in self.objects) / self.total_mass
        cz = sum(obj.mass * obj.pos[2] for obj in self.objects) / self.total_mass
        return (cx, cy, cz)
    
    def field_at_object(self, obj: IFACEObject) -> Tuple[float, float, float]:
        """Get field values at object's position."""
        if self.field is None:
            return (0.0, 1.0, 0.0)
        return self.field.value_at(*obj.pos)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "t": self.t,
            "tau": self.tau,
            "objects": [obj.to_dict() for obj in self.objects],
            "field": self.field.to_dict() if self.field else None,
            "total_Q": self.total_Q,
            "total_mass": self.total_mass,
            "total_energy": self.total_energy,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IFACEState":
        """Deserialize from dictionary."""
        objects = [IFACEObject.from_dict(o) for o in data.get("objects", [])]
        field_data = data.get("field")
        field = None
        if field_data:
            field = IFACEField(
                phi=np.array(field_data["phi"]),
                capacity=np.array(field_data["capacity"]),
                tension=np.array(field_data["tension"]),
                dim_size=field_data["dim_size"],
                scale=field_data["scale"],
            )
        return cls(
            t=data["t"],
            tau=data.get("tau", 0.0),
            objects=objects,
            field=field,
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        return (f"IFACEState(t={self.t}, τ={self.tau:.2f}, "
                f"objects={self.num_objects}, Q={self.total_Q:+.1f}, "
                f"M={self.total_mass:.2f})")


@dataclass
class IFACEHistory:
    """
    History of IFACE states over time.
    
    Used for:
    - Tracking particle trajectories
    - Computing velocities
    - Fitting field equations
    - Checking conservation laws
    """
    states: List[IFACEState] = dataclass_field(default_factory=list)
    max_history: int = 1000  # Maximum states to keep
    
    def add(self, state: IFACEState) -> None:
        """Add state to history."""
        self.states.append(state)
        # Trim if exceeds max
        if len(self.states) > self.max_history:
            self.states = self.states[-self.max_history:]
    
    @property
    def length(self) -> int:
        """Number of stored states."""
        return len(self.states)
    
    def get_phi_history(self) -> np.ndarray:
        """
        Get phi field history as 4D array.
        
        Returns:
            Array of shape (T, n, n, n) where T is number of time steps
        """
        if not self.states or self.states[0].field is None:
            return np.array([])
        
        return np.array([s.field.phi for s in self.states if s.field is not None])
    
    def get_object_trajectory(self, obj_id: int) -> np.ndarray:
        """
        Get trajectory of specific object.
        
        Returns:
            Array of shape (T, 3) with positions over time
        """
        trajectory = []
        for state in self.states:
            obj = state.get_object_by_id(obj_id)
            if obj is not None:
                trajectory.append(obj.pos)
        return np.array(trajectory)
    
    def get_total_Q_history(self) -> np.ndarray:
        """Get history of total charge (sum of object charges)."""
        return np.array([s.total_Q for s in self.states])
    
    def get_global_Q_history(self) -> np.ndarray:
        """Get history of global topological charge (conserved)."""
        return np.array([s.global_Q for s in self.states])
    
    def get_total_mass_history(self) -> np.ndarray:
        """Get history of total mass."""
        return np.array([s.total_mass for s in self.states])
    
    def get_field_energy_history(self) -> np.ndarray:
        """Get history of total field energy (conserved quantity)."""
        return np.array([
            s.field.total_energy if s.field else 0.0 
            for s in self.states
        ])
    
    def clear(self) -> None:
        """Clear history."""
        self.states.clear()
