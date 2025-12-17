"""
Z-order (Morton code) coordinate mapping for 1D → 3D.

Morton codes provide good locality preservation with simple computation.
For N = 16³ = 4096 sites, each coordinate is 4 bits.

The mapping interleaves bits:
    i = ...z2y2x2z1y1x1z0y0x0
    x = ...x2x1x0
    y = ...y2y1y0  
    z = ...z2z1z0
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .coordinate_mapper import CoordinateMapper, Coordinates3D


def _split_by_3(x: int) -> int:
    """Spread bits of x to every third bit position."""
    x = x & 0x000003ff  # Keep only 10 bits
    x = (x | (x << 16)) & 0x030000ff
    x = (x | (x << 8))  & 0x0300f00f
    x = (x | (x << 4))  & 0x030c30c3
    x = (x | (x << 2))  & 0x09249249
    return x


def _compact_by_3(x: int) -> int:
    """Compact every third bit into contiguous bits."""
    x = x & 0x09249249
    x = (x | (x >> 2))  & 0x030c30c3
    x = (x | (x >> 4))  & 0x0300f00f
    x = (x | (x >> 8))  & 0x030000ff
    x = (x | (x >> 16)) & 0x000003ff
    return x


def morton_encode(x: int, y: int, z: int) -> int:
    """
    Encode 3D coordinates into Morton code (Z-order index).
    
    Args:
        x, y, z: 3D integer coordinates (each 0 to N-1 where N is power of 2)
        
    Returns:
        Morton code (1D index)
    """
    return _split_by_3(x) | (_split_by_3(y) << 1) | (_split_by_3(z) << 2)


def morton_decode(m: int) -> Tuple[int, int, int]:
    """
    Decode Morton code into 3D coordinates.
    
    Args:
        m: Morton code (1D index)
        
    Returns:
        (x, y, z) 3D integer coordinates
    """
    x = _compact_by_3(m)
    y = _compact_by_3(m >> 1)
    z = _compact_by_3(m >> 2)
    return x, y, z


@dataclass
class GridConfig:
    """Configuration for 3D grid."""
    order: int = 4        # 2^order per dimension
    scale: float = 1.0    # Physical scale factor
    
    @property
    def dim_size(self) -> int:
        """Size in each dimension."""
        return 1 << self.order  # 2^order
    
    @property
    def total_size(self) -> int:
        """Total number of sites."""
        return self.dim_size ** 3
    
    def __post_init__(self):
        if self.order < 1 or self.order > 10:
            raise ValueError(f"Order must be 1-10, got {self.order}")


class MortonMapper(CoordinateMapper):
    """
    Morton (Z-order) curve mapping from 1D to 3D.
    
    Uses bit-interleaving for efficient encoding/decoding.
    
    For N = 16³ (order=4):
        - dim_size = 16
        - total_size = 4096
        - Each coordinate uses 4 bits
    
    Example:
        mapper = MortonMapper(order=4)  # 16³ grid
        
        # 1D → 3D
        coords = mapper.map_index(1000)
        print(coords)  # Coordinates3D(x, y, z)
        
        # 3D → 1D
        idx = mapper.inverse_map(coords)
        assert idx == 1000
    """
    
    def __init__(self, order: int = 4, scale: float = 1.0):
        """
        Initialize Morton mapper.
        
        Args:
            order: Grid order (dimension = 2^order). 
                   For 16³: order=4. For 32³: order=5.
            scale: Physical scale factor for coordinates
        """
        self.config = GridConfig(order=order, scale=scale)
        self._cache_enabled = True
        self._forward_cache: dict = {}
        self._inverse_cache: dict = {}
        
    @property
    def order(self) -> int:
        return self.config.order
    
    @property
    def dim_size(self) -> int:
        return self.config.dim_size
    
    @property
    def total_size(self) -> int:
        return self.config.total_size
    
    @property
    def scale(self) -> float:
        return self.config.scale
    
    def map_index(self, index: int) -> Coordinates3D:
        """
        Map 1D Morton index to 3D coordinates.
        
        Args:
            index: 1D index (0 to total_size-1)
            
        Returns:
            3D coordinates normalized to [0, scale)
        """
        # Wrap to valid range
        index = index % self.total_size
        
        # Check cache
        if self._cache_enabled and index in self._forward_cache:
            return self._forward_cache[index]
        
        # Decode Morton code
        x, y, z = morton_decode(index)
        
        # Normalize to [0, scale)
        n = self.dim_size
        coords = Coordinates3D(
            x * self.scale / n,
            y * self.scale / n,
            z * self.scale / n,
        )
        
        # Cache result
        if self._cache_enabled:
            self._forward_cache[index] = coords
            
        return coords
    
    def map_index_discrete(self, index: int) -> Tuple[int, int, int]:
        """
        Map 1D index to discrete 3D grid coordinates.
        
        Args:
            index: 1D index (0 to total_size-1)
            
        Returns:
            (x, y, z) integer grid coordinates (0 to dim_size-1)
        """
        index = index % self.total_size
        return morton_decode(index)
    
    def inverse_map(self, coords: Coordinates3D) -> int:
        """
        Map 3D coordinates to 1D Morton index.
        
        Args:
            coords: 3D coordinates (normalized or grid)
            
        Returns:
            1D Morton index
        """
        n = self.dim_size
        
        # Convert to discrete grid coordinates
        x = int(coords.x * n / self.scale) % n
        y = int(coords.y * n / self.scale) % n
        z = int(coords.z * n / self.scale) % n
        
        return morton_encode(x, y, z)
    
    def inverse_map_discrete(self, x: int, y: int, z: int) -> int:
        """
        Map discrete 3D grid coordinates to 1D index.
        
        Args:
            x, y, z: Integer grid coordinates
            
        Returns:
            1D Morton index
        """
        n = self.dim_size
        return morton_encode(x % n, y % n, z % n)
    
    def get_neighbors_1d(self, index: int) -> list[int]:
        """
        Get 6 face-neighbors in 1D index space.
        
        Returns neighbors in order: +x, -x, +y, -y, +z, -z
        """
        x, y, z = morton_decode(index % self.total_size)
        n = self.dim_size
        
        neighbors = []
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            nx = (x + dx) % n
            ny = (y + dy) % n
            nz = (z + dz) % n
            neighbors.append(morton_encode(nx, ny, nz))
        
        return neighbors
    
    def map_all(self, size: Optional[int] = None) -> np.ndarray:
        """
        Map all indices to 3D coordinates array.
        
        Args:
            size: Number of indices (defaults to total_size)
            
        Returns:
            Array of shape (size, 3) with coordinates
        """
        if size is None:
            size = self.total_size
        
        coords = np.zeros((size, 3), dtype=np.float64)
        for i in range(size):
            c = self.map_index(i)
            coords[i] = [c.x, c.y, c.z]
        return coords
    
    def map_all_discrete(self, size: Optional[int] = None) -> np.ndarray:
        """
        Map all indices to discrete 3D grid coordinates.
        
        Returns:
            Array of shape (size, 3) with integer coordinates
        """
        if size is None:
            size = self.total_size
        
        coords = np.zeros((size, 3), dtype=np.int32)
        for i in range(size):
            coords[i] = morton_decode(i % self.total_size)
        return coords
    
    def create_3d_grid(self, values_1d: np.ndarray) -> np.ndarray:
        """
        Convert 1D array to 3D grid using Morton mapping.
        
        Args:
            values_1d: 1D array of values (length <= total_size)
            
        Returns:
            3D array of shape (dim_size, dim_size, dim_size)
        """
        n = self.dim_size
        grid_3d = np.zeros((n, n, n), dtype=values_1d.dtype)
        
        for i, val in enumerate(values_1d):
            if i >= self.total_size:
                break
            x, y, z = morton_decode(i)
            grid_3d[x, y, z] = val
            
        return grid_3d
    
    def flatten_3d_grid(self, grid_3d: np.ndarray) -> np.ndarray:
        """
        Convert 3D grid to 1D array using Morton order.
        
        Args:
            grid_3d: 3D array of shape (n, n, n)
            
        Returns:
            1D array of length n³
        """
        n = grid_3d.shape[0]
        values_1d = np.zeros(n**3, dtype=grid_3d.dtype)
        
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    idx = morton_encode(x, y, z)
                    values_1d[idx] = grid_3d[x, y, z]
                    
        return values_1d
    
    def precompute_cache(self) -> None:
        """Precompute all mappings for fast lookup."""
        for i in range(self.total_size):
            _ = self.map_index(i)
    
    def clear_cache(self) -> None:
        """Clear mapping caches."""
        self._forward_cache.clear()
        self._inverse_cache.clear()


# Convenience functions for default 16³ grid
_default_mapper: Optional[MortonMapper] = None


def get_default_mapper() -> MortonMapper:
    """Get default Morton mapper (16³ grid)."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = MortonMapper(order=4, scale=1.0)
    return _default_mapper


def idx_to_xyz(index: int) -> Tuple[int, int, int]:
    """Convert 1D index to 3D coordinates (default 16³ grid)."""
    return morton_decode(index)


def xyz_to_idx(x: int, y: int, z: int) -> int:
    """Convert 3D coordinates to 1D index (default 16³ grid)."""
    return morton_encode(x, y, z)
