"""
RSL Filters for rule activation control.

Implements F1 and other filters that determine when rules can be applied
based on local state properties.

F1 Filter:
- Activates rules when local tension exceeds threshold
- Implements RA (Random Action) behavior for high-tension regions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable
from abc import ABC, abstractmethod
import numpy as np

from .tension import TensionCalculator
from .capacity import CapacityCalculator


class RSLFilter(ABC):
    """
    Abstract base class for RSL filters.
    
    Filters determine which positions in the lattice are active
    for rule application.
    """
    
    @abstractmethod
    def active_positions(self, sites: np.ndarray) -> np.ndarray:
        """
        Return array of positions where rules can be applied.
        
        Args:
            sites: Lattice site values
            
        Returns:
            Boolean array or array of indices
        """
        pass
    
    @abstractmethod
    def activation_strength(self, sites: np.ndarray) -> np.ndarray:
        """
        Return activation strength at each position.
        
        Values in [0, 1] indicating how "active" each site is.
        """
        pass
    
    def passes(self, lattice) -> bool:
        """
        Check if the lattice passes the filter (no high-tension regions).
        
        Returns True if there are no active positions (i.e., all tension is low).
        """
        if hasattr(lattice, 'sites'):
            sites = lattice.sites
        elif hasattr(lattice, '_sites'):
            sites = lattice._sites
        else:
            sites = lattice
        active = self.active_positions(sites)
        return not np.any(active)


class F1Filter(RSLFilter):
    """
    F1 Tension-based filter.
    
    Activates positions where local tension exceeds threshold.
    This implements the RSL principle that rules activate in
    high-tension regions to reduce total system tension.
    
    Parameters:
        threshold: Normalized tension threshold (default 0.5)
        soft_activation: Use soft threshold with sigmoid
        temperature: Temperature for soft activation
    
    Example:
        filter = F1Filter(threshold=0.5)
        
        # Get active positions
        active = filter.active_positions(sites)
        
        # Get activation strength
        strength = filter.activation_strength(sites)
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        soft_activation: bool = False,
        temperature: float = 0.1,
        periodic: bool = True,
    ):
        self.threshold = threshold
        self.soft_activation = soft_activation
        self.temperature = temperature
        self.tension_calc = TensionCalculator(periodic=periodic)
    
    def active_positions(self, sites: np.ndarray) -> np.ndarray:
        """
        Return positions where tension >= threshold.
        
        Returns boolean array.
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
            
        h = self.tension_calc.local_tension(sites)
        
        if self.soft_activation:
            # Soft threshold: probability increases near threshold
            prob = 1 / (1 + np.exp(-(h - self.threshold) / self.temperature))
            return np.random.random(len(h)) < prob
        else:
            return h >= self.threshold
    
    def activation_strength(self, sites: np.ndarray) -> np.ndarray:
        """
        Return activation strength at each position.
        
        For hard threshold: 0 or 1
        For soft threshold: sigmoid of (tension - threshold)
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
            
        h = self.tension_calc.local_tension(sites)
        
        if self.soft_activation:
            return 1 / (1 + np.exp(-(h - self.threshold) / self.temperature))
        else:
            return (h >= self.threshold).astype(float)
    
    def tension_excess(self, sites: np.ndarray) -> np.ndarray:
        """
        Return tension excess above threshold at each site.
        
        max(0, h_i - threshold)
        """
        h = self.tension_calc.local_tension(sites)
        return np.maximum(0, h - self.threshold)


class CapacityFilter(RSLFilter):
    """
    Filter based on local capacity.
    
    Activates positions where capacity >= minimum required.
    """
    
    def __init__(
        self,
        min_capacity: float = 1.0,
        C0: float = 2.0,
        alpha: float = 0.5,
        periodic: bool = True,
    ):
        self.min_capacity = min_capacity
        self.capacity_calc = CapacityCalculator(
            C0=C0, alpha=alpha, periodic=periodic
        )
    
    def active_positions(self, sites: np.ndarray) -> np.ndarray:
        """Return positions where capacity >= min_capacity."""
        C = self.capacity_calc.local_capacity(sites)
        return C >= self.min_capacity
    
    def activation_strength(self, sites: np.ndarray) -> np.ndarray:
        """Return normalized capacity as strength."""
        C = self.capacity_calc.local_capacity(sites)
        return C / self.capacity_calc.C0


class CombinedFilter(RSLFilter):
    """
    Combine multiple filters with AND/OR logic.
    
    Example:
        combined = CombinedFilter([
            F1Filter(threshold=0.5),
            CapacityFilter(min_capacity=2),
        ], mode="and")
    """
    
    def __init__(
        self,
        filters: List[RSLFilter],
        mode: str = "and",  # "and" or "or"
    ):
        self.filters = filters
        self.mode = mode
    
    def active_positions(self, sites: np.ndarray) -> np.ndarray:
        """Combine filter activations."""
        if not self.filters:
            return np.ones(len(sites) if not hasattr(sites, '_sites') else len(sites._sites), dtype=bool)
        
        activations = [f.active_positions(sites) for f in self.filters]
        
        if self.mode == "and":
            result = activations[0]
            for a in activations[1:]:
                result = result & a
            return result
        else:  # or
            result = activations[0]
            for a in activations[1:]:
                result = result | a
            return result
    
    def activation_strength(self, sites: np.ndarray) -> np.ndarray:
        """Combine strengths (product for AND, max for OR)."""
        if not self.filters:
            return np.ones(len(sites) if not hasattr(sites, '_sites') else len(sites._sites))
        
        strengths = [f.activation_strength(sites) for f in self.filters]
        
        if self.mode == "and":
            result = strengths[0]
            for s in strengths[1:]:
                result = result * s
            return result
        else:  # or
            result = strengths[0]
            for s in strengths[1:]:
                result = np.maximum(result, s)
            return result


@dataclass
class FilterConfig:
    """Configuration for RSL filters."""
    tension_threshold: float = 0.5
    capacity_min: float = 1.0
    C0: float = 2.0
    alpha: float = 0.5
    soft_activation: bool = False
    temperature: float = 0.1
    periodic: bool = True


def create_rsl_filter(config: FilterConfig) -> RSLFilter:
    """
    Create combined RSL filter from configuration.
    
    Creates F1 (tension) + capacity filter combination.
    """
    f1 = F1Filter(
        threshold=config.tension_threshold,
        soft_activation=config.soft_activation,
        temperature=config.temperature,
        periodic=config.periodic,
    )
    
    capacity = CapacityFilter(
        min_capacity=config.capacity_min,
        C0=config.C0,
        alpha=config.alpha,
        periodic=config.periodic,
    )
    
    return CombinedFilter([f1, capacity], mode="and")
