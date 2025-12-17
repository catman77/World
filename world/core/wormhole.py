"""
Wormhole Layer for FTL Physics in RSL World.

Implements dynamic wormhole edges H(t) that can be activated
to create FTL effects while maintaining compatibility with
base RSL physics.

Key concepts from New_Physics_v1.md:
- FTL is NOT "exceeding c" but "shortening the path"
- Wormhole edges are activated DETERMINISTICALLY based on:
  1. Resonance between Ω-configurations
  2. Meaning density Q
  3. Available capacity/resource
- When context=0 (no wormholes), reduces to standard RSL physics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Set, Tuple, List, Optional
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from .world import World


@dataclass
class WormholeConfig:
    """Configuration for wormhole activation."""
    
    # Activation thresholds
    resonance_threshold: float = 0.7    # Min resonance for activation
    meaning_threshold: float = 0.3      # Min meaning density Q
    
    # Geometric constraints
    min_hop_distance: int = 50          # Min graph distance for wormhole
    max_wormholes: int = 10             # Max simultaneous wormholes
    
    # Resource management
    initial_capacity: float = 10.0      # Starting capacity
    activation_cost: float = 1.0        # Cost per activation
    
    # Resonance computation
    resonance_window: int = 5           # Window size for pattern comparison


@dataclass
class WormholeLayer:
    """
    Dynamic wormhole layer H(t) for FTL physics.
    
    Adds deterministic non-local edges to the base graph,
    enabling FTL effects while preserving local causality.
    
    Usage:
        from world.core.world import World, WorldConfig
        from world.core.wormhole import WormholeLayer, WormholeConfig
        
        world = World(WorldConfig(N=512), ruleset)
        wh = WormholeLayer(world)
        
        # Try to activate wormhole
        if wh.try_activate(node_A, node_B):
            print(f"FTL path: {wh.get_effective_distance(node_A, node_B)} hops")
    
    The wormhole does NOT break causality:
    - Signal still travels at 1 hop/step through edges
    - FTL emerges from path shortening, not speed increase
    """
    
    world: "World"
    config: WormholeConfig = field(default_factory=WormholeConfig)
    
    # Active wormhole edges: Set of (i, j) tuples where i < j
    active_edges: Set[Tuple[int, int]] = field(default_factory=set)
    
    # Resource tracking
    capacity: float = field(default=0.0)
    
    # History for analysis
    activation_history: List[dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.capacity == 0.0:
            self.capacity = self.config.initial_capacity
    
    def compute_resonance(self, i: int, j: int) -> float:
        """
        Compute resonance between nodes i and j.
        
        Resonance measures similarity of local spin configurations,
        analogous to "matching Ω-cycle signatures".
        
        Args:
            i: First node index
            j: Second node index
            
        Returns:
            Resonance value in [0, 1], higher = more similar
        """
        s = self.world.s
        N = self.world.N
        w = self.config.resonance_window
        
        # Extract local patterns
        i_start, i_end = max(0, i - w), min(N, i + w + 1)
        j_start, j_end = max(0, j - w), min(N, j + w + 1)
        
        pattern_i = s[i_start:i_end].astype(float)
        pattern_j = s[j_start:j_end].astype(float)
        
        # Handle edge cases
        min_len = min(len(pattern_i), len(pattern_j))
        if min_len < 2:
            return 0.0
        
        # Compute correlation
        pi = pattern_i[:min_len]
        pj = pattern_j[:min_len]
        
        # Avoid division by zero
        std_i = np.std(pi)
        std_j = np.std(pj)
        if std_i < 1e-10 or std_j < 1e-10:
            # Constant patterns - check if equal
            return 1.0 if np.allclose(pi, pj) else 0.0
        
        corr = np.corrcoef(pi, pj)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0
    
    def compute_meaning_density(self, region_start: int, region_end: int) -> float:
        """
        Compute "meaning density" Q in a region.
        
        High Q indicates structured Ω-cycles present.
        Simple proxy: density of domain walls (spin transitions).
        
        Args:
            region_start: Start of region
            region_end: End of region
            
        Returns:
            Meaning density Q in [0, 1]
        """
        s = self.world.s
        region = s[max(0, region_start):min(len(s), region_end)]
        
        if len(region) < 2:
            return 0.0
        
        # Count transitions (domain walls)
        transitions = np.sum(np.abs(np.diff(region)))
        
        # Normalize by max possible transitions
        Q = transitions / (len(region) - 1)
        return Q
    
    def check_activation_conditions(self, i: int, j: int) -> dict:
        """
        Check all activation conditions for wormhole between i and j.
        
        Returns:
            Dictionary with condition checks and values
        """
        result = {
            'can_activate': True,
            'reasons': [],
            'resonance': 0.0,
            'Q_avg': 0.0,
            'base_distance': 0,
        }
        
        # Check capacity
        if len(self.active_edges) >= self.config.max_wormholes:
            result['can_activate'] = False
            result['reasons'].append('max_wormholes_reached')
        
        if self.capacity < self.config.activation_cost:
            result['can_activate'] = False
            result['reasons'].append('insufficient_capacity')
        
        # Check base distance
        base_dist = self.world.graph.compute_graph_distance(i, j)
        result['base_distance'] = base_dist
        
        if base_dist < self.config.min_hop_distance:
            result['can_activate'] = False
            result['reasons'].append('distance_too_small')
        
        # Check resonance
        resonance = self.compute_resonance(i, j)
        result['resonance'] = resonance
        
        if resonance < self.config.resonance_threshold:
            result['can_activate'] = False
            result['reasons'].append('low_resonance')
        
        # Check meaning density
        w = self.config.resonance_window * 2
        Q_i = self.compute_meaning_density(i - w, i + w)
        Q_j = self.compute_meaning_density(j - w, j + w)
        Q_avg = (Q_i + Q_j) / 2
        result['Q_avg'] = Q_avg
        
        if Q_avg < self.config.meaning_threshold:
            result['can_activate'] = False
            result['reasons'].append('low_meaning_density')
        
        # Check if already exists
        edge = (min(i, j), max(i, j))
        if edge in self.active_edges:
            result['can_activate'] = False
            result['reasons'].append('already_active')
        
        return result
    
    def try_activate(self, i: int, j: int, force: bool = False) -> bool:
        """
        Try to activate wormhole between nodes i and j.
        
        Args:
            i: First node index
            j: Second node index
            force: If True, bypass resonance/meaning checks (keep capacity/distance)
            
        Returns:
            True if activation successful, False otherwise
        """
        conditions = self.check_activation_conditions(i, j)
        
        if force:
            # Only check hard constraints
            if len(self.active_edges) >= self.config.max_wormholes:
                return False
            if self.capacity < self.config.activation_cost:
                return False
            if conditions['base_distance'] < self.config.min_hop_distance:
                return False
        else:
            if not conditions['can_activate']:
                return False
        
        # Activate
        edge = (min(i, j), max(i, j))
        self.active_edges.add(edge)
        self.capacity -= self.config.activation_cost
        
        # Record history
        self.activation_history.append({
            'time': self.world.t,
            'edge': edge,
            'action': 'activate',
            'resonance': conditions['resonance'],
            'Q_avg': conditions['Q_avg'],
            'base_distance': conditions['base_distance'],
        })
        
        return True
    
    def deactivate(self, i: int, j: int) -> bool:
        """Deactivate wormhole between i and j."""
        edge = (min(i, j), max(i, j))
        if edge in self.active_edges:
            self.active_edges.remove(edge)
            self.activation_history.append({
                'time': self.world.t,
                'edge': edge,
                'action': 'deactivate',
            })
            return True
        return False
    
    def clear_all(self):
        """Deactivate all wormholes."""
        for edge in list(self.active_edges):
            self.deactivate(edge[0], edge[1])
    
    def get_neighbors_with_wormholes(self, node: int) -> List[int]:
        """
        Get all neighbors of node including wormhole connections.
        
        Args:
            node: Node index
            
        Returns:
            List of neighbor indices
        """
        # Base graph neighbors
        neighbors = list(self.world.graph.neighbors(node))
        
        # Add wormhole neighbors
        for edge in self.active_edges:
            if edge[0] == node:
                neighbors.append(edge[1])
            elif edge[1] == node:
                neighbors.append(edge[0])
        
        return neighbors
    
    def get_effective_distance(self, i: int, j: int) -> int:
        """
        Compute shortest path distance using base graph + wormholes.
        
        Args:
            i: Start node
            j: End node
            
        Returns:
            Shortest path length in hops
        """
        if i == j:
            return 0
        
        # Direct wormhole check
        edge = (min(i, j), max(i, j))
        if edge in self.active_edges:
            return 1
        
        # BFS on extended graph
        visited = {i}
        queue = deque([(i, 0)])
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.get_neighbors_with_wormholes(node):
                if neighbor == j:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return float('inf')
    
    def compute_ftl_factor(self, i: int, j: int) -> float:
        """
        Compute FTL factor: ratio of base distance to wormhole distance.
        
        Returns:
            FTL factor >= 1.0 (1.0 means no speedup)
        """
        base_dist = self.world.graph.compute_graph_distance(i, j)
        wh_dist = self.get_effective_distance(i, j)
        
        if wh_dist == 0 or wh_dist == float('inf'):
            return 1.0
        
        return base_dist / wh_dist
    
    @property
    def n_active(self) -> int:
        """Number of active wormholes."""
        return len(self.active_edges)
    
    def summary(self) -> str:
        """Return summary string."""
        return (f"WormholeLayer(active={self.n_active}, "
                f"capacity={self.capacity:.1f}/{self.config.initial_capacity:.1f})")
    
    def __repr__(self) -> str:
        return self.summary()


def create_wormhole_layer(
    world: "World",
    resonance_threshold: float = 0.5,
    min_distance: int = 30,
    initial_capacity: float = 10.0,
) -> WormholeLayer:
    """
    Convenience function to create a WormholeLayer.
    
    Args:
        world: World instance to attach to
        resonance_threshold: Min resonance for activation
        min_distance: Min graph distance for wormhole
        initial_capacity: Starting capacity
        
    Returns:
        Configured WormholeLayer
    """
    config = WormholeConfig(
        resonance_threshold=resonance_threshold,
        min_hop_distance=min_distance,
        initial_capacity=initial_capacity,
    )
    return WormholeLayer(world=world, config=config)
