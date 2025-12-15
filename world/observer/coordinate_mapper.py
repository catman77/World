"""
Coordinate mapping from 1D lattice to 3D observer space.

Implements space-filling curves for mapping 1D → 3D:
- Hilbert curve: Best locality preservation
- Morton (Z-order) curve: Simpler computation
- Linear: Simple row-major mapping

The mapping preserves locality: nearby 1D sites map to nearby 3D positions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class Coordinates3D:
    """3D coordinates for a point."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: "Coordinates3D") -> float:
        """Euclidean distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __add__(self, other: "Coordinates3D") -> "Coordinates3D":
        return Coordinates3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Coordinates3D") -> "Coordinates3D":
        return Coordinates3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "Coordinates3D":
        return Coordinates3D(self.x * scalar, self.y * scalar, self.z * scalar)


class CoordinateMapper(ABC):
    """
    Abstract base class for 1D → 3D coordinate mapping.
    """
    
    @abstractmethod
    def map_index(self, index: int) -> Coordinates3D:
        """Map 1D index to 3D coordinates."""
        pass
    
    @abstractmethod
    def inverse_map(self, coords: Coordinates3D) -> int:
        """Map 3D coordinates back to 1D index."""
        pass
    
    def map_all(self, size: int) -> np.ndarray:
        """Map all indices to 3D coordinates array."""
        coords = np.zeros((size, 3), dtype=np.float64)
        for i in range(size):
            c = self.map_index(i)
            coords[i] = [c.x, c.y, c.z]
        return coords
    
    def locality_score(self, size: int) -> float:
        """
        Compute locality preservation score.
        
        Measures how well the mapping preserves neighbor relationships.
        Score near 1 = good locality, near 0 = poor locality.
        """
        coords = self.map_all(size)
        
        # Compare 1D distance to 3D distance for neighbors
        locality_sum = 0.0
        count = 0
        
        for i in range(size - 1):
            # 1D neighbors should be close in 3D
            d3d = np.linalg.norm(coords[i+1] - coords[i])
            # Good locality: d3d close to 1
            locality_sum += 1 / (1 + d3d)
            count += 1
        
        return locality_sum / count if count > 0 else 0.0


class HilbertMapper(CoordinateMapper):
    """
    Hilbert curve mapping from 1D to 3D.
    
    The Hilbert curve provides excellent locality preservation:
    nearby points in 1D remain nearby in 3D.
    
    Implementation uses recursive definition of the Hilbert curve.
    """
    
    def __init__(self, order: int = 4, scale: float = 1.0):
        """
        Initialize Hilbert mapper.
        
        Args:
            order: Hilbert curve order (determines resolution)
            scale: Scaling factor for coordinates
        """
        self.order = order
        self.scale = scale
        self._size = (1 << order) ** 3  # 2^order cubed
        
        # Precompute mapping tables for efficiency
        self._precompute_tables()
    
    def _precompute_tables(self) -> None:
        """Precompute Hilbert curve lookup tables."""
        # For efficiency, we'll precompute the mapping
        n = 1 << self.order  # 2^order
        self._forward_table = {}
        self._inverse_table = {}
        
        for d in range(n**3):
            x, y, z = self._d2xyz(d, self.order)
            self._forward_table[d] = (x, y, z)
            self._inverse_table[(x, y, z)] = d
    
    def _d2xyz(self, d: int, order: int) -> Tuple[int, int, int]:
        """
        Convert Hilbert index to 3D coordinates.
        
        Implementation of the 3D Hilbert curve algorithm.
        """
        x = y = z = 0
        s = 1
        
        for i in range(order):
            rx = 1 & (d // 4)
            ry = 1 & (d // 2)
            rz = 1 & d
            
            # Rotation
            if rz == 0:
                if ry == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y, z = z, x, y
            
            x += s * rx
            y += s * ry
            z += s * rz
            
            d //= 8
            s *= 2
        
        return x, y, z
    
    def _xyz2d(self, x: int, y: int, z: int, order: int) -> int:
        """Convert 3D coordinates to Hilbert index."""
        d = 0
        s = 1 << (order - 1)
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            rz = 1 if (z & s) > 0 else 0
            
            d += s * s * s * (4 * rx + 2 * ry + rz)
            
            # Rotation
            if rz == 0:
                if ry == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y, z = y, z, x
            
            s //= 2
        
        return d
    
    def map_index(self, index: int) -> Coordinates3D:
        """Map 1D Hilbert index to 3D coordinates."""
        n = 1 << self.order
        # Wrap index to valid range
        index = index % (n**3)
        
        if index in self._forward_table:
            x, y, z = self._forward_table[index]
        else:
            x, y, z = self._d2xyz(index, self.order)
        
        return Coordinates3D(
            x * self.scale / n,
            y * self.scale / n,
            z * self.scale / n,
        )
    
    def inverse_map(self, coords: Coordinates3D) -> int:
        """Map 3D coordinates to 1D Hilbert index."""
        n = 1 << self.order
        
        # Convert continuous coordinates to discrete
        x = int(coords.x * n / self.scale) % n
        y = int(coords.y * n / self.scale) % n
        z = int(coords.z * n / self.scale) % n
        
        key = (x, y, z)
        if key in self._inverse_table:
            return self._inverse_table[key]
        
        return self._xyz2d(x, y, z, self.order)


class MortonMapper(CoordinateMapper):
    """
    Morton (Z-order) curve mapping from 1D to 3D.
    
    Simpler than Hilbert but still preserves locality reasonably well.
    Uses bit interleaving.
    """
    
    def __init__(self, bits: int = 10, scale: float = 1.0):
        """
        Initialize Morton mapper.
        
        Args:
            bits: Number of bits per dimension
            scale: Scaling factor for coordinates
        """
        self.bits = bits
        self.scale = scale
        self._max_coord = (1 << bits) - 1
    
    def _split_bits(self, x: int) -> int:
        """Spread bits of x for Morton interleaving."""
        x = x & self._max_coord
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x
    
    def _compact_bits(self, x: int) -> int:
        """Compact interleaved bits back to single dimension."""
        x = x & 0x09249249
        x = (x | (x >> 2)) & 0x030C30C3
        x = (x | (x >> 4)) & 0x0300F00F
        x = (x | (x >> 8)) & 0x030000FF
        x = (x | (x >> 16)) & 0x000003FF
        return x
    
    def map_index(self, index: int) -> Coordinates3D:
        """Map 1D Morton index to 3D coordinates."""
        # Extract interleaved bits
        x = self._compact_bits(index)
        y = self._compact_bits(index >> 1)
        z = self._compact_bits(index >> 2)
        
        # Normalize to [0, scale]
        max_val = 1 << self.bits
        return Coordinates3D(
            x * self.scale / max_val,
            y * self.scale / max_val,
            z * self.scale / max_val,
        )
    
    def inverse_map(self, coords: Coordinates3D) -> int:
        """Map 3D coordinates to 1D Morton index."""
        max_val = 1 << self.bits
        
        # Convert to discrete coordinates
        x = int(coords.x * max_val / self.scale) & self._max_coord
        y = int(coords.y * max_val / self.scale) & self._max_coord
        z = int(coords.z * max_val / self.scale) & self._max_coord
        
        # Interleave bits
        return self._split_bits(x) | (self._split_bits(y) << 1) | (self._split_bits(z) << 2)


class LinearMapper(CoordinateMapper):
    """
    Simple linear (row-major) mapping from 1D to 3D.
    
    Maps 1D index to 3D grid: i → (x, y, z) where x varies fastest.
    Poor locality preservation but simple.
    """
    
    def __init__(self, dims: Tuple[int, int, int] = (10, 10, 10), scale: float = 1.0):
        """
        Initialize linear mapper.
        
        Args:
            dims: Grid dimensions (nx, ny, nz)
            scale: Scaling factor for coordinates
        """
        self.nx, self.ny, self.nz = dims
        self.scale = scale
    
    def map_index(self, index: int) -> Coordinates3D:
        """Map 1D index to 3D grid position."""
        total = self.nx * self.ny * self.nz
        index = index % total
        
        x = index % self.nx
        y = (index // self.nx) % self.ny
        z = index // (self.nx * self.ny)
        
        return Coordinates3D(
            x * self.scale / self.nx,
            y * self.scale / self.ny,
            z * self.scale / self.nz,
        )
    
    def inverse_map(self, coords: Coordinates3D) -> int:
        """Map 3D coordinates to 1D index."""
        x = int(coords.x * self.nx / self.scale) % self.nx
        y = int(coords.y * self.ny / self.scale) % self.ny
        z = int(coords.z * self.nz / self.scale) % self.nz
        
        return x + y * self.nx + z * self.nx * self.ny


def create_mapper(
    method: str,
    lattice_size: int,
    **kwargs,
) -> CoordinateMapper:
    """
    Create coordinate mapper by method name.
    
    Args:
        method: "hilbert", "morton", or "linear"
        lattice_size: Size of lattice for dimension calculation
        **kwargs: Additional arguments for mapper
        
    Returns:
        CoordinateMapper instance
    """
    if method == "hilbert":
        # Calculate order from lattice size
        order = max(1, int(np.ceil(np.log2(lattice_size ** (1/3)))))
        return HilbertMapper(order=order, **kwargs)
    
    elif method == "morton":
        bits = max(1, int(np.ceil(np.log2(lattice_size ** (1/3)))))
        return MortonMapper(bits=bits, **kwargs)
    
    elif method == "linear":
        # Compute cubic grid dimensions
        dim = int(np.ceil(lattice_size ** (1/3)))
        return LinearMapper(dims=(dim, dim, dim), **kwargs)
    
    else:
        raise ValueError(f"Unknown mapping method: {method}")


def compare_locality(lattice_size: int) -> dict:
    """
    Compare locality preservation of different mappers.
    
    Returns dict with scores for each method.
    """
    results = {}
    
    for method in ["hilbert", "morton", "linear"]:
        mapper = create_mapper(method, lattice_size)
        results[method] = mapper.locality_score(lattice_size)
    
    return results
