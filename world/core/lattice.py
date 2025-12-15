"""
Lattice representation for 1D RSL world.

The lattice is a 1D array of symbols representing the state string:
    [L0: s0 | s1 | ... | s_{N-1}]

where s_i ∈ Σ (alphabet, typically {+, -} at base level).

Key concepts:
- State: Current configuration of all sites
- Domain wall: Boundary between + and - regions  
- Coarse-graining level: L0, L1, L2, ... (observer-dependent)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator, Sequence
import numpy as np
from enum import IntEnum
import copy


class Symbol(IntEnum):
    """
    Base alphabet symbols encoded as integers for efficient computation.
    
    Using IntEnum allows both numeric operations and readable names.
    """
    MINUS = -1  # "-" symbol
    PLUS = +1   # "+" symbol
    
    def __str__(self) -> str:
        return "+" if self == Symbol.PLUS else "-"
    
    def __repr__(self) -> str:
        return f"Symbol.{'PLUS' if self == Symbol.PLUS else 'MINUS'}"
    
    @classmethod
    def from_char(cls, char: str) -> "Symbol":
        """Convert character to Symbol."""
        if char in ("+", "1", "↑"):
            return cls.PLUS
        elif char in ("-", "0", "↓"):
            return cls.MINUS
        else:
            raise ValueError(f"Unknown symbol character: {char}")
    
    def flip(self) -> "Symbol":
        """Return the opposite symbol."""
        return Symbol.MINUS if self == Symbol.PLUS else Symbol.PLUS


@dataclass
class LatticeState:
    """
    Immutable snapshot of lattice state at a given time.
    
    Attributes:
        sites: Array of site values (+1 or -1)
        time: Evolution step when this state was captured
        level: Coarse-graining level (0 = microscopic)
        metadata: Optional additional information
    """
    sites: np.ndarray  # dtype=np.int8, shape=(N,)
    time: int = 0
    level: int = 0  # Coarse-graining level
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure sites is a numpy array with correct dtype
        if not isinstance(self.sites, np.ndarray):
            self.sites = np.array(self.sites, dtype=np.int8)
        elif self.sites.dtype != np.int8:
            self.sites = self.sites.astype(np.int8)
        # Make immutable
        self.sites.flags.writeable = False
    
    @property
    def size(self) -> int:
        """Number of sites."""
        return len(self.sites)
    
    @property
    def N(self) -> int:
        """Alias for size."""
        return len(self.sites)
    
    def __len__(self) -> int:
        return len(self.sites)
    
    def __getitem__(self, index: int) -> int:
        """Get site value at index (supports periodic boundaries)."""
        return int(self.sites[index % self.size])
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticeState):
            return False
        return (self.level == other.level and 
                np.array_equal(self.sites, other.sites))
    
    def __hash__(self) -> int:
        return hash((self.sites.tobytes(), self.level))
    
    def to_string(self) -> str:
        """
        Convert to standard notation: [L{level}: s0 | s1 | ... | s_{N-1}]
        """
        symbols = " | ".join("+" if s > 0 else "-" for s in self.sites)
        return f"[L{self.level}: {symbols}]"
    
    def to_compact_string(self) -> str:
        """Compact notation: +--+-+..."""
        return "".join("+" if s > 0 else "-" for s in self.sites)
    
    @classmethod
    def from_string(cls, s: str, time: int = 0) -> "LatticeState":
        """
        Parse from string notation.
        
        Accepts:
        - Standard: "[L0: + | - | + | - ]"
        - Compact: "+--+-+"
        """
        s = s.strip()
        level = 0
        
        # Parse standard notation
        if s.startswith("["):
            # Extract level
            if s[1] == "L":
                level_end = s.index(":")
                level = int(s[2:level_end])
                s = s[level_end+1:-1]  # Remove brackets and level marker
            else:
                s = s[1:-1]  # Just remove brackets
            
            # Parse | separated symbols
            parts = s.split("|")
            sites = []
            for part in parts:
                part = part.strip()
                if part:
                    sites.append(1 if part == "+" else -1)
        else:
            # Compact notation
            sites = [1 if c == "+" else -1 for c in s if c in "+-"]
        
        return cls(
            sites=np.array(sites, dtype=np.int8),
            time=time,
            level=level,
        )
    
    def copy(self) -> "LatticeState":
        """Create a mutable copy."""
        new_sites = self.sites.copy()
        new_sites.flags.writeable = True
        state = LatticeState(
            sites=new_sites,
            time=self.time,
            level=self.level,
            metadata=copy.copy(self.metadata),
        )
        return state


class Lattice:
    """
    Mutable 1D lattice with periodic boundaries.
    
    Represents the fundamental 1D world string with operations for:
    - Local symbol access and modification
    - Domain wall counting
    - Coarse-graining (block averaging)
    - State snapshots
    
    The lattice uses periodic boundary conditions by default (ring topology).
    
    Example:
        lattice = Lattice(size=10)
        lattice.randomize(seed=42)
        print(lattice.to_state().to_string())
        # [L0: + | - | + | - | + | - | + | + | - | + ]
        
        # Apply local flip
        lattice[3] = Symbol.PLUS
        
        # Count domain walls
        print(lattice.domain_wall_count())  # Number of +/- boundaries
    """
    
    def __init__(
        self,
        size: int = 100,
        initial: Optional[np.ndarray | Sequence[int] | str] = None,
        periodic: bool = True,
    ):
        """
        Initialize lattice.
        
        Args:
            size: Number of sites (if initial is None)
            initial: Initial configuration (array, sequence, or string)
            periodic: Use periodic boundary conditions
        """
        self.periodic = periodic
        self._time = 0
        self._level = 0
        
        if initial is not None:
            if isinstance(initial, str):
                state = LatticeState.from_string(initial)
                self._sites = state.sites.copy()
                self._sites.flags.writeable = True
                self._level = state.level
            elif isinstance(initial, np.ndarray):
                self._sites = initial.astype(np.int8).copy()
            else:
                self._sites = np.array(initial, dtype=np.int8)
        else:
            # Default: all plus
            self._sites = np.ones(size, dtype=np.int8)
    
    @classmethod
    def from_array(cls, array: np.ndarray, periodic: bool = True) -> "Lattice":
        """Create lattice from numpy array."""
        return cls(initial=array, periodic=periodic)
    
    @classmethod
    def random(cls, size: int, p_plus: float = 0.5, seed: Optional[int] = None, 
               periodic: bool = True) -> "Lattice":
        """Create random lattice."""
        lattice = cls(size=size, periodic=periodic)
        lattice.randomize(p_plus=p_plus, seed=seed)
        return lattice
    
    @property
    def size(self) -> int:
        """Number of sites."""
        return len(self._sites)
    
    @property
    def N(self) -> int:
        """Alias for size."""
        return len(self._sites)
    
    @property
    def time(self) -> int:
        """Current evolution step."""
        return self._time
    
    @property
    def level(self) -> int:
        """Coarse-graining level."""
        return self._level
    
    @property
    def sites(self) -> np.ndarray:
        """Raw site values as numpy array (read-only copy)."""
        return self._sites.copy()
    
    @property
    def phases(self) -> np.ndarray:
        """
        Compute phase array from sites.
        
        Phase φ(i) = π * (1 - s_i) / 2
        So: s=+1 → φ=0, s=-1 → φ=π
        """
        return np.pi * (1 - self._sites) / 2
    
    def __len__(self) -> int:
        return len(self._sites)
    
    def __getitem__(self, index: int) -> int:
        """Get site value with periodic boundary."""
        if self.periodic:
            return int(self._sites[index % self.size])
        else:
            if index < 0 or index >= self.size:
                raise IndexError(f"Index {index} out of bounds [0, {self.size})")
            return int(self._sites[index])
    
    def __setitem__(self, index: int, value: int | Symbol):
        """Set site value with periodic boundary."""
        if isinstance(value, Symbol):
            value = int(value)
        if self.periodic:
            self._sites[index % self.size] = value
        else:
            if index < 0 or index >= self.size:
                raise IndexError(f"Index {index} out of bounds [0, {self.size})")
            self._sites[index] = value
    
    def get_window(self, center: int, radius: int) -> np.ndarray:
        """
        Get sites in window [center-radius, center+radius] with periodic wrapping.
        
        Returns array of length 2*radius + 1.
        """
        indices = np.arange(center - radius, center + radius + 1)
        if self.periodic:
            indices = indices % self.size
        return self._sites[indices].copy()
    
    def flip(self, index: int) -> None:
        """Flip symbol at index."""
        if self.periodic:
            index = index % self.size
        self._sites[index] *= -1
    
    def swap(self, i: int, j: int) -> None:
        """Swap symbols at positions i and j."""
        if self.periodic:
            i = i % self.size
            j = j % self.size
        self._sites[i], self._sites[j] = self._sites[j], self._sites[i]
    
    # ===== Initialization methods =====
    
    def fill(self, value: int | Symbol) -> "Lattice":
        """Fill all sites with given value."""
        if isinstance(value, Symbol):
            value = int(value)
        self._sites.fill(value)
        return self
    
    def all_plus(self) -> "Lattice":
        """Set all sites to +."""
        return self.fill(Symbol.PLUS)
    
    def all_minus(self) -> "Lattice":
        """Set all sites to -."""
        return self.fill(Symbol.MINUS)
    
    def randomize(self, seed: Optional[int] = None, p_plus: float = 0.5) -> "Lattice":
        """
        Fill with random values.
        
        Args:
            seed: Random seed (None for random)
            p_plus: Probability of + symbol
        """
        rng = np.random.default_rng(seed)
        self._sites = np.where(
            rng.random(self.size) < p_plus, 
            Symbol.PLUS, 
            Symbol.MINUS
        ).astype(np.int8)
        return self
    
    def alternating(self, start: int = 1) -> "Lattice":
        """Fill with alternating pattern: +-+-+- or -+-+-+"""
        for i in range(self.size):
            self._sites[i] = start if i % 2 == 0 else -start
        return self
    
    def single_defect(self, position: Optional[int] = None) -> "Lattice":
        """
        Create single domain wall (defect) at position.
        
        Results in: +++...+++---...--- with wall at position.
        """
        if position is None:
            position = self.size // 2
        self._sites[:position] = Symbol.PLUS
        self._sites[position:] = Symbol.MINUS
        return self
    
    def multi_defect(self, positions: Sequence[int]) -> "Lattice":
        """Create multiple domain walls at specified positions."""
        current = Symbol.PLUS
        prev_pos = 0
        for pos in sorted(positions):
            self._sites[prev_pos:pos] = current
            current = Symbol.MINUS if current == Symbol.PLUS else Symbol.PLUS
            prev_pos = pos
        self._sites[prev_pos:] = current
        return self
    
    # ===== Analysis methods =====
    
    def domain_wall_count(self) -> int:
        """
        Count number of domain walls (boundaries between + and - regions).
        
        For periodic boundaries, counts walls including wrap-around.
        Domain wall at position i exists when s[i] ≠ s[i+1].
        
        This is M in the formula H_micro(S) = J * M.
        """
        # Count where consecutive symbols differ
        M = np.sum(self._sites[:-1] != self._sites[1:])
        
        # For periodic, also check wrap-around
        if self.periodic and self._sites[-1] != self._sites[0]:
            M += 1
        
        return int(M)
    
    def domain_walls(self) -> List[int]:
        """
        Return positions of all domain walls.
        
        Position i is a wall if s[i] ≠ s[i+1].
        """
        walls = []
        for i in range(self.size - 1):
            if self._sites[i] != self._sites[i + 1]:
                walls.append(i)
        
        if self.periodic and self._sites[-1] != self._sites[0]:
            walls.append(self.size - 1)
        
        return walls
    
    def magnetization(self) -> float:
        """
        Total magnetization: m = (1/N) * Σ s_i
        
        Returns value in [-1, 1].
        """
        return float(np.mean(self._sites))
    
    def local_magnetization(self, index: int, radius: int) -> float:
        """
        Local magnetization in window around index.
        
        m_R(i) = (1/|B_R(i)|) * Σ_{j ∈ B_R(i)} s_j
        
        This is the coarse field ϕ_R(i).
        """
        window = self.get_window(index, radius)
        return float(np.mean(window))
    
    # ===== Coarse-graining =====
    
    def coarse_grain(self, block_size: int) -> "Lattice":
        """
        Create coarse-grained version using block averaging.
        
        Each block of `block_size` sites is replaced by its majority symbol.
        The new lattice has size N/block_size (rounded).
        
        Args:
            block_size: Number of sites per block
            
        Returns:
            New Lattice at level L+1
        """
        # Compute block averages
        n_blocks = self.size // block_size
        new_sites = np.zeros(n_blocks, dtype=np.int8)
        
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block_sum = np.sum(self._sites[start:end])
            # Majority vote (+ for tie)
            new_sites[i] = 1 if block_sum >= 0 else -1
        
        # Create new lattice at higher level
        new_lattice = Lattice(size=n_blocks, periodic=self.periodic)
        new_lattice._sites = new_sites
        new_lattice._level = self._level + 1
        new_lattice._time = self._time
        
        return new_lattice
    
    def coarse_field(self, radius: int) -> np.ndarray:
        """
        Compute coarse field ϕ_R(i) for all sites.
        
        ϕ_R(i) = (1/|B_R(i)|) * Σ_{j ∈ B_R(i)} s_j
        
        Uses convolution for efficiency.
        """
        kernel_size = 2 * radius + 1
        kernel = np.ones(kernel_size) / kernel_size
        
        if self.periodic:
            # Pad for periodic convolution
            padded = np.concatenate([
                self._sites[-radius:],
                self._sites,
                self._sites[:radius]
            ])
            result = np.convolve(padded.astype(float), kernel, mode='valid')
        else:
            result = np.convolve(self._sites.astype(float), kernel, mode='same')
        
        return result
    
    # ===== State management =====
    
    def to_state(self) -> LatticeState:
        """Create immutable snapshot of current state."""
        return LatticeState(
            sites=self._sites.copy(),
            time=self._time,
            level=self._level,
        )
    
    def from_state(self, state: LatticeState) -> "Lattice":
        """Restore lattice from state snapshot."""
        self._sites = state.sites.copy()
        self._sites.flags.writeable = True
        self._time = state.time
        self._level = state.level
        return self
    
    def increment_time(self, steps: int = 1) -> None:
        """Advance time counter."""
        self._time += steps
    
    def copy(self) -> "Lattice":
        """Create a copy of this lattice."""
        new_lattice = Lattice(size=self.size, periodic=self.periodic)
        new_lattice._sites = self._sites.copy()
        new_lattice._time = self._time
        new_lattice._level = self._level
        return new_lattice
    
    # ===== String representations =====
    
    def to_string(self) -> str:
        """Standard notation: [L{level}: s0 | s1 | ... ]"""
        return self.to_state().to_string()
    
    def __str__(self) -> str:
        if self.size <= 20:
            return self.to_state().to_compact_string()
        else:
            # Truncate for large lattices
            start = "".join("+" if s > 0 else "-" for s in self._sites[:10])
            end = "".join("+" if s > 0 else "-" for s in self._sites[-10:])
            return f"{start}...{end} (N={self.size})"
    
    def __repr__(self) -> str:
        return f"Lattice(size={self.size}, time={self._time}, level={self._level})"
    
    # ===== Iteration =====
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over site values."""
        return iter(self._sites)
    
    def enumerate(self) -> Iterator[Tuple[int, int]]:
        """Iterate over (index, value) pairs."""
        return enumerate(self._sites)


# ===== Utility functions =====

def hamming_distance(s1: LatticeState, s2: LatticeState) -> int:
    """Count number of differing sites between two states."""
    return int(np.sum(s1.sites != s2.sites))


def state_overlap(s1: LatticeState, s2: LatticeState) -> float:
    """
    Compute overlap between states: (1/N) * Σ s1_i * s2_i
    
    Returns 1.0 for identical, -1.0 for opposite, 0.0 for uncorrelated.
    """
    return float(np.mean(s1.sites * s2.sites))
