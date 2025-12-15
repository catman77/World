"""
Tension calculation for RSL dynamics.

Implements microscopic Hamiltonian:
    H_micro(S) = J * M

where M = number of domain walls (sites where s_i ≠ s_{i+1}).

Also provides local tension h_i(S) for each site.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


def local_tension(sites: np.ndarray, periodic: bool = True) -> np.ndarray:
    """
    Compute local tension h_i(S) for each site.
    
    Local tension at site i is based on the number of domain walls
    adjacent to that site:
    
    h_i = 0.5 * (|s_i - s_{i-1}| + |s_i - s_{i+1}|) / 2
    
    Normalized to [0, 1] where:
    - 0 = site is in uniform region
    - 1 = site is at domain wall
    
    Args:
        sites: Array of site values (+1 or -1)
        periodic: Use periodic boundary conditions
        
    Returns:
        Array of local tension values
    """
    N = len(sites)
    h = np.zeros(N, dtype=np.float64)
    
    for i in range(N):
        # Left neighbor
        if periodic:
            left = sites[(i - 1) % N]
        elif i > 0:
            left = sites[i - 1]
        else:
            left = sites[i]  # Boundary: reflect
        
        # Right neighbor
        if periodic:
            right = sites[(i + 1) % N]
        elif i < N - 1:
            right = sites[i + 1]
        else:
            right = sites[i]  # Boundary: reflect
        
        # Count walls adjacent to this site
        walls = 0
        if sites[i] != left:
            walls += 1
        if sites[i] != right:
            walls += 1
        
        h[i] = walls / 2.0  # Normalize to [0, 1]
    
    return h


if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _local_tension_numba(sites: np.ndarray, periodic: bool) -> np.ndarray:
        """Numba-optimized local tension calculation."""
        N = len(sites)
        h = np.zeros(N, dtype=np.float64)
        
        for i in range(N):
            if periodic:
                left = sites[(i - 1) % N]
                right = sites[(i + 1) % N]
            else:
                left = sites[max(0, i - 1)]
                right = sites[min(N - 1, i + 1)]
            
            walls = 0
            if sites[i] != left:
                walls += 1
            if sites[i] != right:
                walls += 1
            
            h[i] = walls / 2.0
        
        return h


class TensionCalculator:
    """
    Calculator for RSL tension (microscopic Hamiltonian).
    
    Computes:
    - Global tension: H_micro(S) = J * M
    - Local tension: h_i(S) at each site
    - Normalized tension: H_norm = H_micro / N
    
    Parameters:
        J: Coupling constant (default 1.0)
        periodic: Use periodic boundary conditions
        use_numba: Enable numba optimization
    
    Example:
        calc = TensionCalculator(J=1.0)
        
        H = calc.global_tension(lattice)
        h = calc.local_tension(lattice)
        H_norm = calc.normalized_tension(lattice)
    """
    
    def __init__(
        self,
        J: float = 1.0,
        periodic: bool = True,
        use_numba: bool = True,
    ):
        self.J = J
        self.periodic = periodic
        self.use_numba = use_numba and HAS_NUMBA
    
    def domain_wall_count(self, sites: np.ndarray) -> int:
        """
        Count domain walls M.
        
        M = number of positions where s_i ≠ s_{i+1}
        """
        M = int(np.sum(sites[:-1] != sites[1:]))
        
        if self.periodic:
            if sites[-1] != sites[0]:
                M += 1
        
        return M
    
    def global_tension(self, sites: np.ndarray) -> float:
        """
        Compute global Hamiltonian H_micro(S) = J * M.
        
        Args:
            sites: Array of site values or Lattice object
            
        Returns:
            H_micro value
        """
        # Handle Lattice object
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        M = self.domain_wall_count(sites)
        return self.J * M
    
    def normalized_tension(self, sites: np.ndarray) -> float:
        """
        Compute normalized tension H_norm = H_micro / N.
        
        Normalized to [0, 1] for easy threshold comparison.
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        H = self.global_tension(sites)
        N = len(sites)
        
        # Maximum domain walls in periodic lattice is N (alternating)
        return H / N
    
    def local_tension(self, sites: np.ndarray) -> np.ndarray:
        """
        Compute local tension h_i(S) for all sites.
        
        Args:
            sites: Array of site values or Lattice object
            
        Returns:
            Array of local tension values in [0, 1]
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        if self.use_numba and HAS_NUMBA:
            return _local_tension_numba(sites.astype(np.int8), self.periodic)
        else:
            return local_tension(sites, self.periodic)
    
    def tension_gradient(self, sites: np.ndarray) -> np.ndarray:
        """
        Compute tension gradient Δh_i = h_i - mean(h).
        
        Positive gradient indicates higher-than-average tension.
        """
        h = self.local_tension(sites)
        return h - np.mean(h)
    
    def find_high_tension_sites(
        self, 
        sites: np.ndarray, 
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Find sites with tension above threshold.
        
        Returns array of indices.
        """
        h = self.local_tension(sites)
        return np.where(h >= threshold)[0]
    
    def tension_distribution(
        self, 
        sites: np.ndarray,
        bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tension distribution histogram.
        
        Returns (counts, bin_edges).
        """
        h = self.local_tension(sites)
        return np.histogram(h, bins=bins, range=(0, 1))
    
    def __call__(self, sites: np.ndarray) -> np.ndarray:
        """Callable interface for use with EvolutionEngine."""
        return self.local_tension(sites)


# Convenience function
def compute_hamiltonian(
    sites: np.ndarray,
    J: float = 1.0,
    periodic: bool = True,
) -> float:
    """
    Compute H_micro(S) = J * M.
    
    Quick function for one-off calculations.
    """
    calc = TensionCalculator(J=J, periodic=periodic, use_numba=False)
    return calc.global_tension(sites)
