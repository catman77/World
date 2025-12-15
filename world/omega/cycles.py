"""
Ω-cycle detection and analysis.

An Ω-cycle is a sequence of rule applications that:
1. Forms a closed loop (returns to original local state)
2. Maintains spatial localization (doesn't spread indefinitely)
3. Propagates through the lattice as a coherent structure

Ω-cycles are the RSL analogs of particles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any
from collections import defaultdict
import numpy as np


@dataclass
class OmegaCycle:
    """
    Represents a detected Ω-cycle (particle-like excitation).
    
    An Ω-cycle is characterized by:
    - Position: Center of mass location in lattice
    - Extent: Spatial spread of the cycle
    - Period: Number of steps to complete the cycle
    - Signature: Pattern that identifies the cycle type
    
    Attributes:
        position: Current center position
        extent: Width of affected region
        period: Steps to return to initial state
        rule_sequence: Sequence of rule applications in cycle
        signature: Unique pattern identifier
        velocity: Movement rate (sites per step)
        phase: Current phase within cycle [0, 2π)
        creation_time: Time step when first detected
    """
    position: float  # Can be fractional (center of mass)
    extent: int      # Width of affected region
    period: int      # Cycle length in steps
    
    rule_sequence: List[str] = field(default_factory=list)
    signature: str = ""
    
    velocity: float = 0.0  # Sites per step
    phase: float = 0.0     # [0, 2π)
    creation_time: int = 0
    
    # Additional properties
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def wavelength(self) -> float:
        """Effective wavelength λ = period * |velocity|."""
        if abs(self.velocity) > 0:
            return self.period * abs(self.velocity)
        return float('inf')
    
    @property
    def frequency(self) -> float:
        """Frequency ν = 1/period."""
        return 1.0 / self.period if self.period > 0 else 0.0
    
    def advance_phase(self, delta_t: int = 1) -> None:
        """Advance phase by time steps."""
        self.phase = (self.phase + 2 * np.pi * delta_t / self.period) % (2 * np.pi)
    
    def update_position(self, delta_t: int = 1, lattice_size: Optional[int] = None) -> None:
        """Update position based on velocity."""
        self.position += self.velocity * delta_t
        if lattice_size is not None:
            self.position = self.position % lattice_size
    
    def overlaps_with(self, other: "OmegaCycle") -> bool:
        """Check if this cycle's extent overlaps with another."""
        # Compute overlap of [pos - extent/2, pos + extent/2]
        self_left = self.position - self.extent / 2
        self_right = self.position + self.extent / 2
        other_left = other.position - other.extent / 2
        other_right = other.position + other.extent / 2
        
        return not (self_right < other_left or other_right < self_left)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "position": self.position,
            "extent": self.extent,
            "period": self.period,
            "rule_sequence": self.rule_sequence,
            "signature": self.signature,
            "velocity": self.velocity,
            "phase": self.phase,
            "creation_time": self.creation_time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OmegaCycle":
        """Deserialize from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"Ω({self.signature}, pos={self.position:.1f}, T={self.period})"


class CycleDetector:
    """
    Detector for Ω-cycles in lattice evolution.
    
    Detection methods:
    1. Local state recurrence: Track local patterns over time
    2. Domain wall tracking: Follow domain walls as quasi-particles
    3. Correlation analysis: Detect periodic correlations
    
    Parameters:
        window_size: Spatial window for local pattern matching
        max_period: Maximum cycle period to search for
        min_recurrences: Minimum recurrences to confirm cycle
    """
    
    def __init__(
        self,
        window_size: int = 5,
        max_period: int = 100,
        min_recurrences: int = 2,
    ):
        self.window_size = window_size
        self.max_period = max_period
        self.min_recurrences = min_recurrences
        
        # History for pattern matching
        self._pattern_history: Dict[int, List[Tuple[int, bytes]]] = defaultdict(list)
        # position -> [(time, pattern_bytes), ...]
        
        # Detected cycles
        self._detected_cycles: List[OmegaCycle] = []
        
        # Domain wall tracking
        self._wall_positions: List[List[int]] = []
    
    def reset(self) -> None:
        """Clear detection state."""
        self._pattern_history.clear()
        self._detected_cycles.clear()
        self._wall_positions.clear()
    
    def update(
        self, 
        sites: np.ndarray, 
        time: int,
        rules_applied: Optional[List] = None,
    ) -> List[OmegaCycle]:
        """
        Update detector with new state.
        
        Args:
            sites: Current lattice state
            time: Current time step
            rules_applied: Optional list of rules applied this step
            
        Returns:
            List of newly detected cycles
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
            
        new_cycles = []
        
        # Method 1: Local pattern recurrence
        new_cycles.extend(self._check_pattern_recurrence(sites, time))
        
        # Method 2: Domain wall tracking
        new_cycles.extend(self._track_domain_walls(sites, time))
        
        # Update history
        self._record_patterns(sites, time)
        
        # Add to detected cycles (avoid duplicates)
        for cycle in new_cycles:
            if not self._is_duplicate(cycle):
                self._detected_cycles.append(cycle)
        
        return new_cycles
    
    def _record_patterns(self, sites: np.ndarray, time: int) -> None:
        """Record local patterns at each position."""
        N = len(sites)
        half_window = self.window_size // 2
        
        for i in range(N):
            # Extract local window
            indices = np.arange(i - half_window, i + half_window + 1) % N
            pattern = sites[indices].tobytes()
            
            # Store with time
            self._pattern_history[i].append((time, pattern))
            
            # Limit history size
            if len(self._pattern_history[i]) > self.max_period * 2:
                self._pattern_history[i] = self._pattern_history[i][-self.max_period:]
    
    def _check_pattern_recurrence(
        self, 
        sites: np.ndarray, 
        time: int,
    ) -> List[OmegaCycle]:
        """Check for recurring patterns at each position."""
        cycles = []
        N = len(sites)
        half_window = self.window_size // 2
        
        for i in range(N):
            # Get current pattern
            indices = np.arange(i - half_window, i + half_window + 1) % N
            current_pattern = sites[indices].tobytes()
            
            # Look for recurrence in history
            history = self._pattern_history[i]
            
            for past_time, past_pattern in history[:-1]:  # Exclude current
                if past_pattern == current_pattern:
                    period = time - past_time
                    
                    if 2 <= period <= self.max_period:
                        # Count recurrences
                        recurrences = sum(
                            1 for t, p in history 
                            if p == current_pattern and (time - t) % period == 0
                        )
                        
                        if recurrences >= self.min_recurrences:
                            cycle = OmegaCycle(
                                position=float(i),
                                extent=self.window_size,
                                period=period,
                                signature=current_pattern.hex()[:16],
                                creation_time=past_time,
                            )
                            cycles.append(cycle)
        
        return cycles
    
    def _track_domain_walls(
        self, 
        sites: np.ndarray, 
        time: int,
    ) -> List[OmegaCycle]:
        """Track domain walls as potential particle positions."""
        cycles = []
        N = len(sites)
        
        # Find current wall positions
        current_walls = []
        for i in range(N):
            if sites[i] != sites[(i + 1) % N]:
                current_walls.append(i)
        
        self._wall_positions.append(current_walls)
        
        # Limit history
        if len(self._wall_positions) > self.max_period:
            self._wall_positions = self._wall_positions[-self.max_period:]
        
        # Look for periodic wall motion
        if len(self._wall_positions) >= 3:
            for wall in current_walls:
                velocity, period = self._estimate_wall_motion(wall)
                if period is not None and period >= 2:
                    cycle = OmegaCycle(
                        position=float(wall),
                        extent=1,  # Domain walls are point-like
                        period=period,
                        velocity=velocity,
                        signature=f"wall_{wall}",
                        creation_time=time - period,
                    )
                    cycles.append(cycle)
        
        return cycles
    
    def _estimate_wall_motion(
        self, 
        current_pos: int,
    ) -> Tuple[float, Optional[int]]:
        """Estimate velocity and period of a domain wall."""
        if len(self._wall_positions) < 2:
            return 0.0, None
        
        # Simple velocity estimate from recent history
        recent = self._wall_positions[-3:]
        
        # Try to track this wall backwards
        positions = [current_pos]
        for past_walls in reversed(recent[:-1]):
            # Find closest wall
            if past_walls:
                closest = min(past_walls, key=lambda w: abs(w - positions[-1]))
                if abs(closest - positions[-1]) <= 5:  # Max jump
                    positions.append(closest)
                else:
                    break
            else:
                break
        
        if len(positions) >= 2:
            velocity = (positions[0] - positions[-1]) / (len(positions) - 1)
            return velocity, None  # Period detection not implemented here
        
        return 0.0, None
    
    def _is_duplicate(self, cycle: OmegaCycle) -> bool:
        """Check if cycle is duplicate of existing one."""
        for existing in self._detected_cycles:
            if (abs(existing.position - cycle.position) < self.window_size and
                existing.period == cycle.period):
                return True
        return False
    
    @property
    def detected_cycles(self) -> List[OmegaCycle]:
        """Return list of all detected cycles."""
        return list(self._detected_cycles)
    
    def get_active_cycles(self, time: int, max_age: int = 1000) -> List[OmegaCycle]:
        """Get cycles that are still active (detected recently)."""
        return [
            c for c in self._detected_cycles
            if time - c.creation_time < max_age
        ]


def find_omega_cycles(
    history: List[np.ndarray],
    window_size: int = 5,
    max_period: int = 100,
) -> List[OmegaCycle]:
    """
    Analyze evolution history to find all Ω-cycles.
    
    Args:
        history: List of lattice states (np.ndarray or LatticeState)
        window_size: Local window size for pattern matching
        max_period: Maximum cycle period
        
    Returns:
        List of detected OmegaCycles
    """
    detector = CycleDetector(
        window_size=window_size,
        max_period=max_period,
    )
    
    for t, state in enumerate(history):
        if hasattr(state, 'sites'):
            state = state.sites
        detector.update(state, t)
    
    return detector.detected_cycles


def compute_cycle_spectrum(
    cycles: List[OmegaCycle],
) -> Dict[int, int]:
    """
    Compute spectrum of cycle periods.
    
    Returns dict mapping period -> count.
    """
    spectrum = defaultdict(int)
    for cycle in cycles:
        spectrum[cycle.period] += 1
    return dict(spectrum)


def compute_cycle_density(
    cycles: List[OmegaCycle],
    lattice_size: int,
) -> float:
    """
    Compute density of cycles (cycles per site).
    """
    return len(cycles) / lattice_size if lattice_size > 0 else 0.0
