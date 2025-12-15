"""
Capacity calculation for RSL dynamics.

Implements local capacity:
    C_i(S) = C0 - α * h_i(S)

where:
- C0: Base capacity (default 2.0)
- α: Capacity reduction coefficient (default 0.5)
- h_i(S): Local tension at site i

Capacity determines the maximum rule length that can be applied at a site.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .tension import TensionCalculator, local_tension


def local_capacity(
    sites: np.ndarray,
    C0: float = 2.0,
    alpha: float = 0.5,
    periodic: bool = True,
) -> np.ndarray:
    """
    Compute local capacity C_i(S) for all sites.
    
    C_i(S) = C0 - α * h_i(S)
    
    Args:
        sites: Array of site values (+1 or -1)
        C0: Base capacity
        alpha: Capacity reduction coefficient
        periodic: Use periodic boundary conditions
        
    Returns:
        Array of local capacity values
    """
    h = local_tension(sites, periodic)
    C = C0 - alpha * h
    return np.maximum(C, 0)  # Capacity cannot be negative


class CapacityCalculator:
    """
    Calculator for RSL local capacity.
    
    Computes:
    - Local capacity: C_i(S) = C0 - α * h_i(S)
    - Available capacity: max rule length at each site
    - Capacity profile: distribution across lattice
    
    Parameters:
        C0: Base capacity (default 2.0)
        alpha: Capacity reduction coefficient (default 0.5)
        periodic: Use periodic boundary conditions
    
    The capacity constrains rule application:
    - Rules of length L can only apply where C_i >= L
    - Higher tension → lower capacity → shorter rules
    
    Example:
        calc = CapacityCalculator(C0=2.0, alpha=0.5)
        
        C = calc.local_capacity(lattice)
        available = calc.available_for_length(lattice, L=3)
    """
    
    def __init__(
        self,
        C0: float = 2.0,
        alpha: float = 0.5,
        periodic: bool = True,
        use_numba: bool = True,
    ):
        self.C0 = C0
        self.alpha = alpha
        self.periodic = periodic
        self.tension_calc = TensionCalculator(
            periodic=periodic,
            use_numba=use_numba,
        )
    
    def local_capacity(self, sites: np.ndarray) -> np.ndarray:
        """
        Compute local capacity C_i(S) for all sites.
        
        Args:
            sites: Array of site values or Lattice object
            
        Returns:
            Array of local capacity values
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        h = self.tension_calc.local_tension(sites)
        C = self.C0 - self.alpha * h
        return np.maximum(C, 0)
    
    def mean_capacity(self, sites: np.ndarray) -> float:
        """Compute mean capacity across lattice."""
        return float(np.mean(self.local_capacity(sites)))
    
    def min_capacity(self, sites: np.ndarray) -> float:
        """Compute minimum capacity across lattice."""
        return float(np.min(self.local_capacity(sites)))
    
    def max_capacity(self, sites: np.ndarray) -> float:
        """Compute maximum capacity across lattice."""
        return float(np.max(self.local_capacity(sites)))
    
    def available_for_length(
        self, 
        sites: np.ndarray, 
        L: int,
    ) -> np.ndarray:
        """
        Find sites where rules of length L can be applied.
        
        Args:
            sites: Array of site values or Lattice object
            L: Required rule length
            
        Returns:
            Boolean array indicating available sites
        """
        C = self.local_capacity(sites)
        return C >= L
    
    def count_available(self, sites: np.ndarray, L: int) -> int:
        """Count number of sites available for rules of length L."""
        return int(np.sum(self.available_for_length(sites, L)))
    
    def capacity_constrained_positions(
        self,
        sites: np.ndarray,
        L: int,
    ) -> np.ndarray:
        """
        Get array of positions where rules of length L can apply.
        
        Returns array of indices.
        """
        available = self.available_for_length(sites, L)
        return np.where(available)[0]
    
    def effective_rule_length(self, sites: np.ndarray) -> np.ndarray:
        """
        Compute effective maximum rule length at each site.
        
        Returns floor(C_i) as the maximum rule length.
        """
        C = self.local_capacity(sites)
        return np.floor(C).astype(int)
    
    def capacity_distribution(
        self,
        sites: np.ndarray,
        bins: int = 10,
    ):
        """
        Compute capacity distribution histogram.
        
        Returns (counts, bin_edges).
        """
        C = self.local_capacity(sites)
        return np.histogram(C, bins=bins, range=(0, self.C0))
    
    def __call__(self, sites: np.ndarray) -> np.ndarray:
        """Callable interface for use with EvolutionEngine."""
        return self.local_capacity(sites)


class AdaptiveCapacity:
    """
    Adaptive capacity that evolves with the system.
    
    Implements dynamic capacity that can:
    - Increase/decrease based on system behavior
    - Have memory of past states
    - Respond to global observables
    
    This is useful for exploring different capacity regimes.
    """
    
    def __init__(
        self,
        base_calculator: CapacityCalculator,
        adaptation_rate: float = 0.01,
    ):
        self.calculator = base_calculator
        self.adaptation_rate = adaptation_rate
        self._C0_history = [base_calculator.C0]
        self._alpha_history = [base_calculator.alpha]
    
    def update(
        self, 
        sites: np.ndarray,
        target_mean_capacity: Optional[float] = None,
    ) -> None:
        """
        Update capacity parameters based on current state.
        
        Args:
            sites: Current lattice state
            target_mean_capacity: Optional target mean capacity
        """
        current_mean = self.calculator.mean_capacity(sites)
        
        if target_mean_capacity is not None:
            # Adjust C0 to approach target
            error = target_mean_capacity - current_mean
            self.calculator.C0 += self.adaptation_rate * error
            self.calculator.C0 = max(0.1, self.calculator.C0)  # Keep positive
        
        self._C0_history.append(self.calculator.C0)
        self._alpha_history.append(self.calculator.alpha)
    
    def reset(self) -> None:
        """Reset to initial parameters."""
        if self._C0_history:
            self.calculator.C0 = self._C0_history[0]
        if self._alpha_history:
            self.calculator.alpha = self._alpha_history[0]
