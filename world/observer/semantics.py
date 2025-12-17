"""
Semantic State (S_sem) for Observer.

S_sem represents the observer's internal "knowledge" about the world:
- Field equation parameters (κ, m², λ)
- Conservation laws (Q, mass)
- Event probabilities (decay/scattering)
- Gravity law parameters

The observer learns these by observing IFACE over time and
fitting models to the observed data.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class FieldEquationParams:
    """
    Parameters of the effective field equation:
        ∂²ϕ/∂t² = κ∇²ϕ - m²ϕ - λϕ³
    
    These are fitted from observed field evolution.
    """
    kappa: float = 0.0      # Wave propagation coefficient
    m2: float = 0.0         # Mass term (squared)
    lambda_: float = 0.0    # Nonlinear coupling
    
    # Fitting quality metrics
    r_squared: float = 0.0  # R² of the fit
    residual: float = 0.0   # Mean squared residual
    n_samples: int = 0      # Number of data points used
    
    def to_vector(self) -> np.ndarray:
        """Convert to parameter vector."""
        return np.array([self.kappa, self.m2, self.lambda_])
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "FieldEquationParams":
        """Create from parameter vector."""
        return cls(kappa=v[0], m2=v[1], lambda_=v[2] if len(v) > 2 else 0.0)
    
    def distance_to(self, other: "FieldEquationParams") -> float:
        """Euclidean distance to another parameter set."""
        return np.linalg.norm(self.to_vector() - other.to_vector())
    
    def __repr__(self) -> str:
        return f"FieldEq(κ={self.kappa:.4f}, m²={self.m2:.4f}, λ={self.lambda_:.4f}, R²={self.r_squared:.3f})"


@dataclass 
class ConservationLaw:
    """
    Represents a conservation law and its verification.
    
    Tracks whether a quantity Q is conserved: dQ/dt ≈ 0
    """
    name: str
    is_conserved: bool = False
    confidence: float = 0.0     # 0 to 1
    max_violation: float = 0.0  # Maximum observed |ΔQ|
    mean_violation: float = 0.0 # Mean |ΔQ|
    n_checks: int = 0
    
    def check(self, values: np.ndarray, tolerance: float = 0.01) -> None:
        """
        Check conservation from history of values.
        
        Args:
            values: Array of Q values over time
            tolerance: Relative tolerance for conservation
        """
        if len(values) < 2:
            return
        
        q0 = values[0] if values[0] != 0 else 1.0
        violations = np.abs(values - values[0])
        
        self.max_violation = float(np.max(violations))
        self.mean_violation = float(np.mean(violations))
        self.n_checks = len(values)
        
        # Conservation if mean violation is small relative to initial value
        relative_violation = self.mean_violation / (abs(q0) + 1e-10)
        self.is_conserved = relative_violation < tolerance
        self.confidence = max(0, 1 - relative_violation / tolerance)


@dataclass
class GravityLaw:
    """
    Parameters of effective gravity law:
        a = -γ ∇Φ
    
    Where Φ is derived from capacity: Φ = C₀ - C
    """
    gamma: float = 0.0          # Coupling constant
    correlation: float = 0.0    # Correlation of a with -∇Φ
    n_samples: int = 0
    
    def __repr__(self) -> str:
        return f"GravityLaw(γ={self.gamma:.4f}, corr={self.correlation:.3f})"


@dataclass
class EventStatistics:
    """
    Statistics of particle events (creation, annihilation, decay).
    """
    # Event counts by type
    creations: int = 0
    annihilations: int = 0
    decays: int = 0
    collisions: int = 0
    
    # Transition probabilities: (from_type, to_types) -> count
    transitions: Dict[str, int] = dataclass_field(default_factory=lambda: defaultdict(int))
    
    # Total events
    total_events: int = 0
    
    def record_creation(self, particle_type: str) -> None:
        """Record particle creation event."""
        self.creations += 1
        self.total_events += 1
        self.transitions[f"vacuum->{particle_type}"] += 1
    
    def record_annihilation(self, particle_types: List[str]) -> None:
        """Record annihilation event."""
        self.annihilations += 1
        self.total_events += 1
        key = "+".join(sorted(particle_types)) + "->vacuum"
        self.transitions[key] += 1
    
    def record_decay(self, parent: str, products: List[str]) -> None:
        """Record decay event."""
        self.decays += 1
        self.total_events += 1
        key = f"{parent}->" + "+".join(sorted(products))
        self.transitions[key] += 1
    
    def get_probability(self, pattern: str) -> float:
        """Get estimated probability for a transition pattern."""
        if self.total_events == 0:
            return 0.0
        return self.transitions[pattern] / self.total_events


@dataclass
class SemanticState:
    """
    Complete semantic state of the observer (S_sem).
    
    Contains:
    - Current estimates of physical laws
    - History of parameter estimates for convergence analysis
    - Conservation law verification
    - Event statistics
    
    This is what the observer "knows" about the world.
    """
    # Field equation
    field_eq: FieldEquationParams = dataclass_field(default_factory=FieldEquationParams)
    field_eq_history: List[FieldEquationParams] = dataclass_field(default_factory=list)
    
    # Conservation laws
    charge_conservation: ConservationLaw = dataclass_field(
        default_factory=lambda: ConservationLaw(name="Q")
    )
    mass_conservation: ConservationLaw = dataclass_field(
        default_factory=lambda: ConservationLaw(name="mass")
    )
    
    # Gravity law  
    gravity: GravityLaw = dataclass_field(default_factory=GravityLaw)
    gravity_history: List[GravityLaw] = dataclass_field(default_factory=list)
    
    # Event statistics
    events: EventStatistics = dataclass_field(default_factory=EventStatistics)
    
    # Observation time (when parameters stabilized)
    observation_time: Optional[int] = None
    
    # Update counter
    update_count: int = 0
    
    # History limits
    max_history: int = 100
    
    def update_field_eq(self, params: FieldEquationParams) -> None:
        """Update field equation parameters and record history."""
        self.field_eq = params
        self.field_eq_history.append(params)
        if len(self.field_eq_history) > self.max_history:
            self.field_eq_history = self.field_eq_history[-self.max_history:]
        self.update_count += 1
        self._check_stabilization()
    
    def update_gravity(self, params: GravityLaw) -> None:
        """Update gravity law parameters and record history."""
        self.gravity = params
        self.gravity_history.append(params)
        if len(self.gravity_history) > self.max_history:
            self.gravity_history = self.gravity_history[-self.max_history:]
        self._check_stabilization()
    
    def _check_stabilization(self, epsilon: float = 0.01, window: int = 10) -> None:
        """
        Check if parameters have stabilized (Observation Time t_OT).
        
        Parameters are considered stable if changes are < epsilon
        for the last `window` updates.
        """
        if self.observation_time is not None:
            return  # Already stabilized
        
        if len(self.field_eq_history) < window:
            return
        
        # Check field equation stability
        recent = self.field_eq_history[-window:]
        changes = [recent[i].distance_to(recent[i-1]) 
                   for i in range(1, len(recent))]
        
        if all(c < epsilon for c in changes):
            self.observation_time = self.update_count
    
    def to_vector(self) -> np.ndarray:
        """
        Convert semantic state to feature vector for analysis.
        
        Used for TDA/topology analysis of understanding trajectory.
        """
        v = []
        # Field equation params
        v.extend(self.field_eq.to_vector())
        v.append(self.field_eq.r_squared)
        
        # Conservation
        v.append(1.0 if self.charge_conservation.is_conserved else 0.0)
        v.append(1.0 if self.mass_conservation.is_conserved else 0.0)
        
        # Gravity
        v.append(self.gravity.gamma)
        v.append(self.gravity.correlation)
        
        return np.array(v)
    
    def is_stabilized(self) -> bool:
        """Check if understanding has stabilized."""
        return self.observation_time is not None
    
    def get_observation_time(self) -> int:
        """Get observation time (t_OT) or current count if not stabilized."""
        return self.observation_time if self.observation_time else self.update_count
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of current knowledge."""
        return {
            "field_equation": {
                "kappa": self.field_eq.kappa,
                "m2": self.field_eq.m2,
                "lambda": self.field_eq.lambda_,
                "r_squared": self.field_eq.r_squared,
            },
            "conservation": {
                "Q": {
                    "conserved": self.charge_conservation.is_conserved,
                    "confidence": self.charge_conservation.confidence,
                },
                "mass": {
                    "conserved": self.mass_conservation.is_conserved,
                    "confidence": self.mass_conservation.confidence,
                },
            },
            "gravity": {
                "gamma": self.gravity.gamma,
                "correlation": self.gravity.correlation,
            },
            "stabilized": self.is_stabilized(),
            "observation_time": self.get_observation_time(),
            "total_updates": self.update_count,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "field_eq": {
                "kappa": self.field_eq.kappa,
                "m2": self.field_eq.m2,
                "lambda": self.field_eq.lambda_,
                "r_squared": self.field_eq.r_squared,
                "residual": self.field_eq.residual,
            },
            "charge_conservation": {
                "is_conserved": self.charge_conservation.is_conserved,
                "confidence": self.charge_conservation.confidence,
                "max_violation": self.charge_conservation.max_violation,
            },
            "mass_conservation": {
                "is_conserved": self.mass_conservation.is_conserved,
                "confidence": self.mass_conservation.confidence,
                "max_violation": self.mass_conservation.max_violation,
            },
            "gravity": {
                "gamma": self.gravity.gamma,
                "correlation": self.gravity.correlation,
            },
            "observation_time": self.observation_time,
            "update_count": self.update_count,
        }
    
    def __repr__(self) -> str:
        status = "STABLE" if self.is_stabilized() else "LEARNING"
        return (f"SemanticState({status}, t_OT={self.get_observation_time()}, "
                f"{self.field_eq}, Q={'✓' if self.charge_conservation.is_conserved else '✗'}, "
                f"M={'✓' if self.mass_conservation.is_conserved else '✗'})")


class SemanticHistory:
    """
    History of semantic states for trajectory analysis.
    
    Used for:
    - Tracking learning progress
    - TDA analysis (β₀, β₁)
    - Visualization of understanding evolution
    """
    
    def __init__(self, max_size: int = 1000):
        self.states: List[SemanticState] = []
        self.vectors: List[np.ndarray] = []
        self.times: List[int] = []
        self.max_size = max_size
    
    def add(self, state: SemanticState, t: int) -> None:
        """Add semantic state to history."""
        self.states.append(state)
        self.vectors.append(state.to_vector())
        self.times.append(t)
        
        # Trim if needed
        if len(self.states) > self.max_size:
            self.states = self.states[-self.max_size:]
            self.vectors = self.vectors[-self.max_size:]
            self.times = self.times[-self.max_size:]
    
    def get_vectors_array(self) -> np.ndarray:
        """Get all vectors as 2D array."""
        if not self.vectors:
            return np.array([])
        return np.array(self.vectors)
    
    def get_parameter_trajectory(self, param_name: str) -> np.ndarray:
        """Get trajectory of specific parameter over time."""
        param_map = {
            "kappa": lambda s: s.field_eq.kappa,
            "m2": lambda s: s.field_eq.m2,
            "lambda": lambda s: s.field_eq.lambda_,
            "gamma": lambda s: s.gravity.gamma,
            "r_squared": lambda s: s.field_eq.r_squared,
        }
        
        if param_name not in param_map:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        return np.array([param_map[param_name](s) for s in self.states])
    
    def clear(self) -> None:
        """Clear history."""
        self.states.clear()
        self.vectors.clear()
        self.times.clear()
