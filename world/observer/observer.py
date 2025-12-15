"""
Observer implementation for RSL World.

The Observer implements the projection:
    Π_obs: S → Y

where S is the microscopic state space and Y is the observer's
macroscopic observation space.

Key property: |[Y]| >> 1 (many microscopic states map to same observation)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
import numpy as np

from ..core.lattice import Lattice, LatticeState


@dataclass
class ObserverState:
    """
    State as seen by observer (macroscopic description).
    
    Contains coarse-grained fields and derived observables.
    """
    # Coarse-grained field ϕ_R(x)
    coarse_field: np.ndarray
    
    # Coarse-graining radius
    radius: int
    
    # Observation level
    level: int = 0
    
    # Time of observation
    time: int = 0
    
    # Derived observables
    mean_field: float = 0.0
    field_variance: float = 0.0
    
    # Positions in 3D space (if mapped)
    coordinates_3d: Optional[np.ndarray] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.coarse_field is not None:
            self.mean_field = float(np.mean(self.coarse_field))
            self.field_variance = float(np.var(self.coarse_field))
    
    @property
    def size(self) -> int:
        """Number of coarse-grained sites."""
        return len(self.coarse_field) if self.coarse_field is not None else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "coarse_field": self.coarse_field.tolist() if self.coarse_field is not None else None,
            "radius": self.radius,
            "level": self.level,
            "time": self.time,
            "mean_field": self.mean_field,
            "field_variance": self.field_variance,
            "coordinates_3d": self.coordinates_3d.tolist() if self.coordinates_3d is not None else None,
            "metadata": self.metadata,
        }


class Observer:
    """
    Observer that performs coarse-graining projection Π_obs.
    
    The observer cannot see individual lattice sites, only
    averaged quantities over some resolution scale R.
    
    Parameters:
        radius: Coarse-graining radius R
        stride: Stride for coarse-grained sampling
        max_levels: Maximum coarse-graining levels
    
    The coarse field is:
        ϕ_R(i) = (1/|B_R(i)|) * Σ_{j ∈ B_R(i)} s_j
    
    where B_R(i) is the ball of radius R around site i.
    
    Example:
        observer = Observer(radius=5)
        
        # Observe microscopic state
        obs_state = observer.observe(lattice)
        
        # Multi-level coarse-graining
        levels = observer.multi_level_observe(lattice, max_level=3)
    """
    
    def __init__(
        self,
        radius: int = 5,
        stride: int = 1,
        max_levels: int = 10,
        periodic: bool = True,
    ):
        self.radius = radius
        self.stride = stride
        self.max_levels = max_levels
        self.periodic = periodic
        
        # History of observations
        self._observation_history: List[ObserverState] = []
    
    def observe(
        self, 
        lattice: Lattice,
        radius: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> ObserverState:
        """
        Perform observation (coarse-graining) of lattice state.
        
        Args:
            lattice: Microscopic lattice state
            radius: Override coarse-graining radius
            stride: Override sampling stride
            
        Returns:
            ObserverState with coarse-grained field
        """
        R = radius if radius is not None else self.radius
        S = stride if stride is not None else self.stride
        
        # Compute coarse field
        coarse_field = self._compute_coarse_field(lattice, R, S)
        
        obs_state = ObserverState(
            coarse_field=coarse_field,
            radius=R,
            level=lattice.level,
            time=lattice.time,
        )
        
        # Store in history
        self._observation_history.append(obs_state)
        
        return obs_state
    
    def _compute_coarse_field(
        self, 
        lattice: Lattice,
        radius: int,
        stride: int,
    ) -> np.ndarray:
        """
        Compute coarse-grained field.
        
        ϕ_R(i) = (1/|B_R(i)|) * Σ_{j ∈ B_R(i)} s_j
        """
        N = lattice.size
        positions = range(0, N, stride)
        n_positions = len(list(positions))
        
        coarse_field = np.zeros(n_positions, dtype=np.float64)
        
        # Kernel size
        kernel_size = 2 * radius + 1
        
        for idx, i in enumerate(range(0, N, stride)):
            # Average over ball B_R(i)
            total = 0.0
            for j in range(-radius, radius + 1):
                if self.periodic:
                    site = (i + j) % N
                else:
                    site = max(0, min(N - 1, i + j))
                total += lattice[site]
            
            coarse_field[idx] = total / kernel_size
        
        return coarse_field
    
    def multi_level_observe(
        self,
        lattice: Lattice,
        max_level: Optional[int] = None,
        block_size: int = 2,
    ) -> List[ObserverState]:
        """
        Perform multi-level coarse-graining.
        
        Creates hierarchy L0 → L1 → L2 → ... of increasingly coarse views.
        
        Args:
            lattice: Microscopic lattice state
            max_level: Maximum coarse-graining level (default: until size < block_size)
            block_size: Blocking factor at each level
            
        Returns:
            List of ObserverState at each level
        """
        if max_level is None:
            max_level = self.max_levels
        
        levels = []
        current_lattice = lattice.copy()
        
        for level in range(max_level + 1):
            # Observe at current level
            obs = self.observe(current_lattice, radius=self.radius)
            obs.level = level
            levels.append(obs)
            
            # Check if further coarse-graining possible
            if current_lattice.size < block_size * 2:
                break
            
            # Coarse-grain for next level
            current_lattice = current_lattice.coarse_grain(block_size)
        
        return levels
    
    def equivalence_class(
        self,
        obs_state: ObserverState,
        all_states: List[LatticeState],
    ) -> List[LatticeState]:
        """
        Find all microscopic states consistent with observation.
        
        This gives the "fiber" over the observer state:
        [Y] = {S : Π_obs(S) = Y}
        
        In practice, we can only sample from this set.
        """
        matching = []
        
        for state in all_states:
            # Create temporary lattice for observation
            temp_lattice = Lattice(initial=state.sites, periodic=self.periodic)
            temp_obs = self.observe(temp_lattice)
            
            # Check if observations match within tolerance
            if np.allclose(temp_obs.coarse_field, obs_state.coarse_field, rtol=0.01):
                matching.append(state)
        
        return matching
    
    def entropy_estimate(
        self,
        obs_state: ObserverState,
        lattice_size: int,
    ) -> float:
        """
        Estimate entropy of microscopic states consistent with observation.
        
        log|[Y]| ≈ (information not captured by coarse-graining)
        
        This is a rough estimate based on the number of "hidden" degrees of freedom.
        """
        # Number of observed degrees of freedom
        n_observed = len(obs_state.coarse_field)
        
        # Number of microscopic degrees of freedom
        n_micro = lattice_size
        
        # Hidden degrees of freedom
        n_hidden = n_micro - n_observed
        
        # Rough entropy estimate (assuming binary states)
        # Each hidden site contributes ~ln(2) entropy
        return n_hidden * np.log(2)
    
    @property
    def observation_history(self) -> List[ObserverState]:
        """Return history of observations."""
        return list(self._observation_history)
    
    def clear_history(self) -> None:
        """Clear observation history."""
        self._observation_history.clear()


class BlockObserver(Observer):
    """
    Observer using block averaging instead of ball averaging.
    
    Divides lattice into non-overlapping blocks of size B.
    Each block produces one coarse-grained value.
    """
    
    def __init__(
        self,
        block_size: int = 10,
        max_levels: int = 10,
    ):
        super().__init__(radius=block_size//2, stride=block_size, max_levels=max_levels)
        self.block_size = block_size
    
    def _compute_coarse_field(
        self,
        lattice: Lattice,
        radius: int,
        stride: int,
    ) -> np.ndarray:
        """Compute coarse field using block averaging."""
        N = lattice.size
        n_blocks = N // self.block_size
        
        coarse_field = np.zeros(n_blocks, dtype=np.float64)
        
        for b in range(n_blocks):
            start = b * self.block_size
            end = start + self.block_size
            coarse_field[b] = np.mean([lattice[i] for i in range(start, end)])
        
        return coarse_field


class AdaptiveObserver(Observer):
    """
    Observer with adaptive coarse-graining based on local structure.
    
    Uses finer resolution in high-activity regions.
    """
    
    def __init__(
        self,
        min_radius: int = 2,
        max_radius: int = 10,
        activity_threshold: float = 0.5,
    ):
        super().__init__(radius=max_radius)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.activity_threshold = activity_threshold
    
    def _compute_coarse_field(
        self,
        lattice: Lattice,
        radius: int,
        stride: int,
    ) -> np.ndarray:
        """
        Compute coarse field with adaptive resolution.
        
        Uses smaller radius where local tension is high.
        """
        N = lattice.size
        coarse_field = np.zeros(N, dtype=np.float64)
        local_radii = np.zeros(N, dtype=np.int32)
        
        for i in range(N):
            # Compute local activity (simple: variance in small window)
            small_window = [lattice[(i + j) % N] for j in range(-2, 3)]
            activity = np.var(small_window)
            
            # Choose radius based on activity
            if activity > self.activity_threshold:
                R = self.min_radius  # Fine resolution
            else:
                R = self.max_radius  # Coarse resolution
            
            local_radii[i] = R
            
            # Compute average
            total = sum(lattice[(i + j) % N] for j in range(-R, R + 1))
            coarse_field[i] = total / (2 * R + 1)
        
        return coarse_field
