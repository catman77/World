"""
Topological analysis for RSL World.

Analyzes topological properties of 1D lattice configurations:
- Winding numbers
- Fundamental group π₁
- Topological defects (domain walls as solitons)
- Homotopy classes

For 1D periodic lattice (ring/S¹), the relevant topology is:
- π₁(S¹) = ℤ (fundamental group is integers)
- Winding number counts net "rotations" around the ring

Domain walls act as topological defects that cannot be removed
by continuous deformation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import Counter


@dataclass
class TopologicalDefect:
    """
    A topological defect (domain wall) in the lattice.
    
    Attributes:
        position: Location in lattice
        charge: Topological charge (+1 or -1)
        type: Type of defect ("wall", "kink", "antikink")
    """
    position: int
    charge: int  # +1 for + → -, -1 for - → +
    type: str = "wall"
    
    def __repr__(self) -> str:
        return f"Defect({self.type}, pos={self.position}, q={self.charge})"


@dataclass
class FundamentalGroupElement:
    """
    Element of the fundamental group π₁.
    
    For 1D ring (S¹), π₁(S¹) = ℤ, so elements are integers.
    The element n represents n windings around the ring.
    """
    winding_number: int
    path_signature: str = ""  # Sequence of defect crossings
    
    def __add__(self, other: "FundamentalGroupElement") -> "FundamentalGroupElement":
        """Group operation (addition of winding numbers)."""
        return FundamentalGroupElement(
            winding_number=self.winding_number + other.winding_number,
            path_signature=self.path_signature + other.path_signature,
        )
    
    def inverse(self) -> "FundamentalGroupElement":
        """Inverse element."""
        return FundamentalGroupElement(
            winding_number=-self.winding_number,
            path_signature=self.path_signature[::-1],
        )
    
    def __repr__(self) -> str:
        return f"π₁({self.winding_number})"


class TopologyAnalyzer:
    """
    Analyzer for topological properties of lattice configurations.
    
    Main analyses:
    1. Winding number: Net "rotation" of the field
    2. Defect counting: Number and types of domain walls
    3. Fundamental group: Homotopy classification
    4. Topological invariants: Quantities preserved under evolution
    
    Example:
        analyzer = TopologyAnalyzer()
        
        # Find defects
        defects = analyzer.find_defects(lattice)
        
        # Compute winding number
        w = analyzer.winding_number(lattice)
        
        # Get fundamental group element
        g = analyzer.fundamental_group_element(lattice)
    """
    
    def __init__(self):
        self._defect_history: List[List[TopologicalDefect]] = []
    
    def find_defects(self, sites: np.ndarray, periodic: bool = True) -> List[TopologicalDefect]:
        """
        Find all topological defects (domain walls) in configuration.
        
        A domain wall exists at position i when s[i] ≠ s[i+1].
        
        Args:
            sites: Lattice site values or Lattice object
            periodic: Use periodic boundary conditions
            
        Returns:
            List of TopologicalDefect objects
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        N = len(sites)
        defects = []
        
        for i in range(N):
            next_i = (i + 1) % N if periodic else min(i + 1, N - 1)
            
            if sites[i] != sites[next_i]:
                # Determine charge based on transition direction
                if sites[i] > sites[next_i]:
                    charge = +1  # + → - transition
                    defect_type = "kink"
                else:
                    charge = -1  # - → + transition  
                    defect_type = "antikink"
                
                defects.append(TopologicalDefect(
                    position=i,
                    charge=charge,
                    type=defect_type,
                ))
        
        self._defect_history.append(defects)
        return defects
    
    def winding_number(self, sites: np.ndarray, periodic: bool = True) -> int:
        """
        Compute winding number of configuration.
        
        For 1D binary field, winding number is:
        W = (1/2) * Σ (s[i] - s[i+1])
        
        This counts the net number of domain walls.
        For periodic boundaries, W must be even (walls come in pairs).
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        N = len(sites)
        W = 0
        
        for i in range(N):
            next_i = (i + 1) % N if periodic else min(i + 1, N - 1)
            W += (sites[i] - sites[next_i])
        
        return W // 2  # Normalize
    
    def total_charge(self, defects: List[TopologicalDefect]) -> int:
        """
        Compute total topological charge.
        
        For periodic boundaries, total charge must be zero
        (kinks and antikinks must balance).
        """
        return sum(d.charge for d in defects)
    
    def fundamental_group_element(
        self, 
        sites: np.ndarray,
        periodic: bool = True,
    ) -> FundamentalGroupElement:
        """
        Compute fundamental group element for configuration.
        
        For S¹, this is the winding number.
        """
        w = self.winding_number(sites, periodic)
        
        # Build path signature from defects
        defects = self.find_defects(sites, periodic)
        signature = "".join("+" if d.charge > 0 else "-" for d in defects)
        
        return FundamentalGroupElement(
            winding_number=w,
            path_signature=signature,
        )
    
    def homotopy_class(self, sites: np.ndarray) -> int:
        """
        Determine homotopy class of configuration.
        
        Configurations in the same homotopy class can be
        continuously deformed into each other.
        
        For S¹ → {+,-}, the homotopy class is determined by
        the number of domain walls (mod 2 for periodic).
        """
        defects = self.find_defects(sites)
        return len(defects) % 2
    
    def are_homotopic(
        self, 
        sites1: np.ndarray, 
        sites2: np.ndarray,
    ) -> bool:
        """
        Check if two configurations are homotopic.
        
        They are homotopic if they have the same homotopy class.
        """
        return self.homotopy_class(sites1) == self.homotopy_class(sites2)
    
    def defect_density(self, sites: np.ndarray) -> float:
        """
        Compute density of topological defects.
        
        ρ = (number of defects) / N
        """
        if hasattr(sites, '_sites'):
            N = len(sites._sites)
        else:
            N = len(sites)
        
        defects = self.find_defects(sites)
        return len(defects) / N if N > 0 else 0.0
    
    def defect_correlation(
        self, 
        sites: np.ndarray,
        max_r: int = 20,
    ) -> np.ndarray:
        """
        Compute defect-defect correlation function.
        
        C(r) = <n(0) * n(r)> where n(i) = 1 if defect at i, 0 otherwise.
        """
        defects = self.find_defects(sites)
        if hasattr(sites, '_sites'):
            N = len(sites._sites)
        else:
            N = len(sites)
        
        # Create defect indicator array
        n = np.zeros(N, dtype=np.float64)
        for d in defects:
            n[d.position] = 1
        
        # Compute correlation
        correlation = np.zeros(max_r + 1)
        for r in range(max_r + 1):
            for i in range(N):
                j = (i + r) % N
                correlation[r] += n[i] * n[j]
            correlation[r] /= N
        
        return correlation
    
    def track_defect_motion(
        self, 
        history: List[np.ndarray],
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Track motion of defects through evolution history.
        
        Returns dict mapping defect_id -> [(time, position), ...]
        """
        trajectories: Dict[int, List[Tuple[int, int]]] = {}
        next_id = 0
        
        prev_defects: List[TopologicalDefect] = []
        prev_positions: Dict[int, int] = {}  # position -> id
        
        for t, state in enumerate(history):
            if hasattr(state, 'sites'):
                state = state.sites
            
            current_defects = self.find_defects(state)
            current_positions = {d.position: d for d in current_defects}
            
            # Match current defects to previous
            matched = set()
            
            for d in current_defects:
                # Look for nearby defect in previous step
                best_id = None
                best_dist = float('inf')
                
                for prev_pos, prev_id in prev_positions.items():
                    dist = min(
                        abs(d.position - prev_pos),
                        abs(d.position - prev_pos + len(state)),
                        abs(d.position - prev_pos - len(state)),
                    )
                    if dist < best_dist and prev_id not in matched:
                        best_dist = dist
                        best_id = prev_id
                
                if best_id is not None and best_dist <= 3:
                    # Match found
                    matched.add(best_id)
                    trajectories[best_id].append((t, d.position))
                else:
                    # New defect
                    trajectories[next_id] = [(t, d.position)]
                    best_id = next_id
                    next_id += 1
            
            # Update previous positions
            prev_positions = {d.position: i for i, d in enumerate(current_defects)}
            prev_defects = current_defects
        
        return trajectories
    
    def topological_entropy(
        self,
        history: List[np.ndarray],
    ) -> float:
        """
        Compute topological entropy from evolution history.
        
        Measures the complexity of defect dynamics.
        """
        # Count defect configurations
        config_counts = Counter()
        
        for state in history:
            defects = self.find_defects(state)
            config = tuple(d.position for d in sorted(defects, key=lambda d: d.position))
            config_counts[config] += 1
        
        # Shannon entropy
        total = sum(config_counts.values())
        entropy = 0.0
        for count in config_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)
        
        return entropy


def compute_winding_number(sites: np.ndarray, periodic: bool = True) -> int:
    """Convenience function for winding number calculation."""
    analyzer = TopologyAnalyzer()
    return analyzer.winding_number(sites, periodic)


def compute_fundamental_group(sites: np.ndarray) -> FundamentalGroupElement:
    """Convenience function for fundamental group element."""
    analyzer = TopologyAnalyzer()
    return analyzer.fundamental_group_element(sites)


def find_topological_defects(sites: np.ndarray) -> List[TopologicalDefect]:
    """Convenience function for defect finding."""
    analyzer = TopologyAnalyzer()
    return analyzer.find_defects(sites)


class BraidGroup:
    """
    Braid group analysis for defect worldlines.
    
    When defects move through time, their worldlines can braid,
    giving rise to non-abelian statistics in 2+1D.
    
    For 1D + time, this gives a simplified picture of how
    defect exchanges affect the system.
    """
    
    def __init__(self):
        self._generators: List[str] = []  # σ₁, σ₂, ...
    
    def compute_braid_word(
        self,
        trajectories: Dict[int, List[Tuple[int, int]]],
    ) -> str:
        """
        Compute braid word from defect trajectories.
        
        A braid word encodes the sequence of crossings.
        """
        # Find crossing events
        crossings = []
        
        defect_ids = list(trajectories.keys())
        for i, id1 in enumerate(defect_ids):
            for id2 in defect_ids[i+1:]:
                traj1 = dict(trajectories[id1])
                traj2 = dict(trajectories[id2])
                
                # Find times when both defects exist
                common_times = set(traj1.keys()) & set(traj2.keys())
                sorted_times = sorted(common_times)
                
                for t in sorted_times[:-1]:
                    t_next = sorted_times[sorted_times.index(t) + 1]
                    
                    # Check for crossing
                    pos1_t = traj1.get(t, 0)
                    pos2_t = traj2.get(t, 0)
                    pos1_next = traj1.get(t_next, 0)
                    pos2_next = traj2.get(t_next, 0)
                    
                    # Simple crossing detection
                    if (pos1_t < pos2_t) != (pos1_next < pos2_next):
                        # Crossing occurred
                        sign = "+" if pos1_t < pos2_t else "-"
                        crossings.append((t, min(i, len(defect_ids)-1), sign))
        
        # Build braid word
        crossings.sort()
        word = "".join(f"σ{c[1]}{c[2]}" for c in crossings)
        return word if word else "e"  # Identity element
