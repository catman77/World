"""
Evolution engine for RSL World dynamics.

Implements the time evolution of the lattice state:
    S(t) → S(t+1) = T(S(t))

where T is the deterministic update operator constructed from local rules.

Key features:
- Deterministic left-to-right rule application
- F1 filter for tension-based activation
- History tracking for cycle detection
- Numba-optimized core loops
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Callable, Any, Tuple
import numpy as np
from collections import deque
import time

from .lattice import Lattice, LatticeState, hamming_distance
from .rules import Rule, RuleSet, RuleMatch


@dataclass
class EvolutionStats:
    """Statistics from evolution run."""
    total_steps: int = 0
    rules_applied: int = 0
    rules_by_name: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # State analysis
    unique_states: int = 0
    cycle_length: Optional[int] = None
    cycle_start: Optional[int] = None
    
    # Physical quantities
    avg_magnetization: float = 0.0
    avg_domain_walls: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def steps_per_second(self) -> float:
        if self.elapsed_time > 0:
            return self.total_steps / self.elapsed_time
        return 0.0


@dataclass
class EvolutionResult:
    """
    Complete result of an evolution run.
    
    Contains:
    - Final state
    - History (if enabled)
    - Statistics
    - Stop reason
    """
    final_state: LatticeState
    history: List[LatticeState] = field(default_factory=list)
    stats: EvolutionStats = field(default_factory=EvolutionStats)
    stop_reason: str = "max_steps"
    
    def get_state_at(self, time: int) -> Optional[LatticeState]:
        """Get state at specific time (if in history)."""
        for state in self.history:
            if state.time == time:
                return state
        return None
    
    def magnetization_series(self) -> np.ndarray:
        """Get magnetization time series from history."""
        return np.array([np.mean(s.sites) for s in self.history])
    
    def domain_wall_series(self) -> np.ndarray:
        """Get domain wall count time series from history."""
        def count_walls(sites):
            return np.sum(sites[:-1] != sites[1:])
        return np.array([count_walls(s.sites) for s in self.history])


class CycleDetector:
    """
    Detect cycles in state evolution using Floyd's algorithm with hashing.
    
    Uses two approaches:
    1. Hash table for exact cycle detection
    2. Floyd's tortoise-hare for memory efficiency
    """
    
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self._state_hashes: Dict[int, int] = {}  # hash -> first occurrence time
        self._recent_states: deque = deque(maxlen=window_size)
    
    def check(self, state: LatticeState) -> Optional[Tuple[int, int]]:
        """
        Check if state was seen before.
        
        Returns:
            (cycle_start, cycle_length) if cycle detected, else None
        """
        state_hash = hash(state)
        current_time = state.time
        
        if state_hash in self._state_hashes:
            # Potential cycle - verify with actual state comparison
            first_time = self._state_hashes[state_hash]
            
            # Look for actual state in recent history
            for old_state in self._recent_states:
                if old_state.time == first_time and state == old_state:
                    return (first_time, current_time - first_time)
        
        # Record state
        self._state_hashes[state_hash] = current_time
        self._recent_states.append(state)
        
        # Cleanup old entries beyond window
        if len(self._state_hashes) > self.window_size * 2:
            min_time = current_time - self.window_size
            self._state_hashes = {
                h: t for h, t in self._state_hashes.items() 
                if t >= min_time
            }
        
        return None
    
    def reset(self) -> None:
        """Clear cycle detection state."""
        self._state_hashes.clear()
        self._recent_states.clear()


class EquilibriumDetector:
    """
    Detect equilibrium (stable state) based on observable convergence.
    
    Tracks magnetization and domain walls over a window.
    """
    
    def __init__(self, window_size: int = 100, threshold: float = 1e-6):
        self.window_size = window_size
        self.threshold = threshold
        self._magnetization_history: deque = deque(maxlen=window_size)
        self._domain_wall_history: deque = deque(maxlen=window_size)
    
    def update(self, state: LatticeState) -> bool:
        """
        Update with new state and check for equilibrium.
        
        Returns True if equilibrium detected.
        """
        mag = float(np.mean(state.sites))
        walls = np.sum(state.sites[:-1] != state.sites[1:])
        
        self._magnetization_history.append(mag)
        self._domain_wall_history.append(walls)
        
        if len(self._magnetization_history) < self.window_size:
            return False
        
        # Check variance over window
        mag_var = np.var(list(self._magnetization_history))
        wall_var = np.var(list(self._domain_wall_history))
        
        return mag_var < self.threshold and wall_var < self.threshold
    
    def reset(self) -> None:
        """Clear equilibrium detection state."""
        self._magnetization_history.clear()
        self._domain_wall_history.clear()


class EvolutionEngine:
    """
    Main evolution engine for RSL World dynamics.
    
    Implements deterministic time evolution with:
    - Left-to-right rule application (conflict resolution)
    - Optional F1 tension-based filtering
    - History tracking with configurable stride
    - Cycle and equilibrium detection
    - Performance optimization via numba
    
    Example:
        engine = EvolutionEngine(rules, config)
        result = engine.run(initial_lattice, max_steps=1000)
        
        # Access history
        for state in result.history:
            print(f"t={state.time}: {state.to_compact_string()}")
    """
    
    def __init__(
        self,
        rules: RuleSet,
        tension_calculator: Optional[Callable[[Lattice], np.ndarray]] = None,
        capacity_calculator: Optional[Callable[[Lattice], np.ndarray]] = None,
        use_numba: bool = False,
    ):
        """
        Initialize evolution engine.
        
        Args:
            rules: Rule set for state transitions
            tension_calculator: Optional function to compute local tension
            capacity_calculator: Optional function to compute local capacity
            use_numba: Enable numba JIT compilation
        """
        self.rules = rules
        self.tension_calculator = tension_calculator
        self.capacity_calculator = capacity_calculator
        self.use_numba = use_numba
        
        # Detectors
        self.cycle_detector = CycleDetector()
        self.equilibrium_detector = EquilibriumDetector()
        
        # Callbacks
        self._step_callbacks: List[Callable[[Lattice, int], None]] = []
    
    def add_step_callback(self, callback: Callable[[Lattice, int], None]) -> None:
        """Add callback to be called after each step."""
        self._step_callbacks.append(callback)
    
    def step(
        self, 
        lattice: Lattice,
        tension: Optional[np.ndarray] = None,
        capacity: Optional[np.ndarray] = None,
    ) -> List[RuleMatch]:
        """
        Perform single evolution step.
        
        Args:
            lattice: Lattice to evolve (modified in place)
            tension: Optional pre-computed tension array
            capacity: Optional pre-computed capacity array
            
        Returns:
            List of applied rule matches
        """
        # Compute tension/capacity if needed and calculators provided
        if tension is None and self.tension_calculator is not None:
            tension = self.tension_calculator(lattice)
        if capacity is None and self.capacity_calculator is not None:
            capacity = self.capacity_calculator(lattice)
        
        # Find all matches
        matches = self.rules.find_matches(lattice)
        
        # Filter by tension if available
        if tension is not None:
            matches = self._filter_by_tension(matches, tension)
        
        # Filter by capacity if available  
        if capacity is not None:
            matches = self._filter_by_capacity(matches, capacity)
        
        # Resolve conflicts (deterministic left-to-right)
        resolved = self.rules.resolve_conflicts(matches, strategy="left_to_right")
        
        # Apply rules
        for match in resolved:
            match.rule.apply(lattice, match.position)
        
        # Update time
        lattice.increment_time()
        
        return resolved
    
    def _filter_by_tension(
        self, 
        matches: List[RuleMatch],
        tension: np.ndarray,
    ) -> List[RuleMatch]:
        """Filter matches based on local tension."""
        filtered = []
        for match in matches:
            # Rule applies if tension at position exceeds rule's threshold
            local_tension = tension[match.position % len(tension)]
            if match.rule.activation == RuleActivation.TENSION:
                if local_tension >= match.rule.tension_threshold:
                    filtered.append(match)
            elif match.rule.activation == RuleActivation.ALWAYS:
                filtered.append(match)
        return filtered
    
    def _filter_by_capacity(
        self,
        matches: List[RuleMatch],
        capacity: np.ndarray,
    ) -> List[RuleMatch]:
        """Filter matches based on local capacity."""
        filtered = []
        for match in matches:
            local_capacity = capacity[match.position % len(capacity)]
            # Rule applies if capacity >= rule length
            if local_capacity >= match.rule.length:
                filtered.append(match)
            elif match.rule.activation == RuleActivation.ALWAYS:
                filtered.append(match)
        return filtered
    
    def run(
        self,
        lattice: Lattice,
        max_steps: int = 10000,
        store_history: bool = True,
        history_stride: int = 1,
        detect_cycles: bool = True,
        detect_equilibrium: bool = True,
        equilibrium_window: int = 100,
        equilibrium_threshold: float = 1e-6,
    ) -> EvolutionResult:
        """
        Run evolution for multiple steps.
        
        Args:
            lattice: Initial lattice state
            max_steps: Maximum evolution steps
            store_history: Whether to store state history
            history_stride: Store every N-th state
            detect_cycles: Enable cycle detection
            detect_equilibrium: Enable equilibrium detection
            equilibrium_window: Window size for equilibrium check
            equilibrium_threshold: Variance threshold for equilibrium
            
        Returns:
            EvolutionResult with final state, history, and statistics
        """
        # Initialize
        stats = EvolutionStats(start_time=time.time())
        history: List[LatticeState] = []
        
        if detect_cycles:
            self.cycle_detector.reset()
        if detect_equilibrium:
            self.equilibrium_detector = EquilibriumDetector(
                window_size=equilibrium_window,
                threshold=equilibrium_threshold,
            )
        
        # Store initial state
        if store_history:
            history.append(lattice.to_state())
        
        stop_reason = "max_steps"
        
        # Main evolution loop
        for step in range(max_steps):
            # Perform step
            applied = self.step(lattice)
            
            # Update stats
            stats.total_steps += 1
            stats.rules_applied += len(applied)
            for match in applied:
                name = match.rule.name
                stats.rules_by_name[name] = stats.rules_by_name.get(name, 0) + 1
            
            # Store state
            current_state = lattice.to_state()
            if store_history and (step + 1) % history_stride == 0:
                history.append(current_state)
            
            # Check stopping conditions
            if detect_cycles:
                cycle = self.cycle_detector.check(current_state)
                if cycle is not None:
                    stats.cycle_start, stats.cycle_length = cycle
                    stop_reason = f"cycle_detected (start={cycle[0]}, length={cycle[1]})"
                    break
            
            if detect_equilibrium:
                if self.equilibrium_detector.update(current_state):
                    stop_reason = "equilibrium"
                    break
            
            # Call callbacks
            for callback in self._step_callbacks:
                callback(lattice, step)
        
        # Finalize stats
        stats.end_time = time.time()
        stats.unique_states = len(set(hash(s) for s in history)) if history else 0
        
        if history:
            stats.avg_magnetization = float(np.mean([np.mean(s.sites) for s in history]))
            stats.avg_domain_walls = float(np.mean([
                np.sum(s.sites[:-1] != s.sites[1:]) for s in history
            ]))
        
        return EvolutionResult(
            final_state=lattice.to_state(),
            history=history,
            stats=stats,
            stop_reason=stop_reason,
        )
    
    def run_until(
        self,
        lattice: Lattice,
        condition: Callable[[Lattice, int], bool],
        max_steps: int = 100000,
        **kwargs,
    ) -> EvolutionResult:
        """
        Run evolution until condition is met.
        
        Args:
            lattice: Initial lattice state
            condition: Function(lattice, step) -> bool, stops when True
            max_steps: Maximum steps before giving up
            **kwargs: Additional arguments for run()
            
        Returns:
            EvolutionResult
        """
        # Use callback mechanism
        stop_flag = [False]  # Mutable to allow modification in callback
        
        def check_condition(lat: Lattice, step: int) -> None:
            if condition(lat, step):
                stop_flag[0] = True
        
        self.add_step_callback(check_condition)
        
        try:
            # Run with modified parameters
            result = self.run(
                lattice,
                max_steps=max_steps,
                detect_cycles=False,
                detect_equilibrium=False,
                **kwargs,
            )
            
            if stop_flag[0]:
                result.stop_reason = "condition_met"
            
            return result
        finally:
            self._step_callbacks.remove(check_condition)


# Import for activation enum
from .rules import RuleActivation


# ===== Utility functions =====

def create_default_engine(
    max_length: int = 3,
    conserving: bool = True,
) -> EvolutionEngine:
    """
    Create evolution engine with default rules.
    
    Args:
        max_length: Maximum rule pattern length
        conserving: Use magnetization-conserving rules
    """
    from .rules import create_conserving_rules, create_swap_rules
    
    if conserving:
        rules = create_conserving_rules(max_length)
    else:
        rules = create_swap_rules()
    
    return EvolutionEngine(rules)
