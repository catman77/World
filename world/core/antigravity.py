"""
Antigravity Module: χ-field and Geometry Inversion.

Implements antigravity through two mechanisms:
1. Second scalar field χ with Φ_eff = φ - η·χ
2. Local geometry inversion (metric sign change)

Key physics from New_Physics_v1.md:
- Normal: φ from Lφ = ρ_m gives attraction
- Antigravity: χ field or inverted graph geometry gives repulsion
- Compatibility: η=0 reduces to standard RSL gravity
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Set, Dict, Tuple, List
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg

if TYPE_CHECKING:
    from .world import World
    from .graph_structure import GraphStructure


@dataclass
class AntigravityConfig:
    """Configuration for antigravity field."""
    
    # χ-field parameters
    chi_coupling: float = 0.0            # η: strength of χ contribution (0 = no antigrav)
    chi_source_scale: float = 1.0        # Scale factor for χ sources
    
    # Geometry inversion
    inversion_enabled: bool = False      # Enable metric inversion mode
    inversion_radius: int = 5            # Radius of inverted region
    
    # Resource management
    antigrav_cost: float = 2.0           # Cost to activate antigravity
    initial_capacity: float = 10.0       # Starting capacity
    
    # Constraints
    max_inversion_regions: int = 3       # Max simultaneous inversions


@dataclass
class ChiField:
    """
    Second scalar field χ for antigravity.
    
    The effective gravitational potential is:
        Φ_eff = φ - η·χ
    
    Where:
        - φ: Standard gravity potential (from mass sources)
        - χ: Antigravity potential (from χ-sources)
        - η: Coupling parameter (device control)
    
    When η=0: Standard gravity (compatibility)
    When η>0: Partial cancellation (reduced gravity)
    When η·χ > φ locally: Effective repulsion (antigravity)
    """
    
    graph: "GraphStructure"
    config: AntigravityConfig = field(default_factory=AntigravityConfig)
    
    # χ-field values
    chi: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # χ-sources (node -> source strength)
    sources: Dict[int, float] = field(default_factory=dict)
    
    # Resource tracking
    capacity: float = field(default=0.0)
    
    def __post_init__(self):
        N = self.graph.N
        if len(self.chi) == 0:
            self.chi = np.zeros(N)
        if self.capacity == 0.0:
            self.capacity = self.config.initial_capacity
    
    def add_chi_source(self, node: int, strength: float = 1.0) -> bool:
        """
        Add χ-source at a node (creates antigravity "charge").
        
        Args:
            node: Node index for χ-source
            strength: Source strength (positive = repulsive center)
            
        Returns:
            True if successful
        """
        cost = self.config.antigrav_cost
        if self.capacity < cost:
            return False
        
        self.sources[node] = strength * self.config.chi_source_scale
        self.capacity -= cost
        return True
    
    def remove_chi_source(self, node: int):
        """Remove χ-source at node."""
        if node in self.sources:
            del self.sources[node]
    
    def solve_chi_field(self) -> np.ndarray:
        """
        Solve Laplacian equation for χ-field: Lχ = ρ_χ
        
        Uses same graph Laplacian as φ-field.
        """
        N = self.graph.N
        
        if not self.sources:
            self.chi = np.zeros(N)
            return self.chi
        
        # Build source vector
        rho_chi = np.zeros(N)
        for node, strength in self.sources.items():
            rho_chi[node] = strength
        
        # Solve Lχ = ρ_χ (same as gravity but different source)
        L = self.graph.laplacian
        
        # Add small regularization for stability
        L_reg = L + 1e-6 * diags([1]*N)
        
        # Solve using conjugate gradient
        self.chi, _ = cg(L_reg, rho_chi)
        
        return self.chi
    
    def compute_effective_potential(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute effective gravitational potential.
        
        Φ_eff = φ - η·χ
        
        Args:
            phi: Standard gravity potential
            
        Returns:
            Effective potential
        """
        eta = self.config.chi_coupling
        if eta == 0 or len(self.chi) == 0:
            return phi
        
        return phi - eta * self.chi
    
    def compute_effective_force(self, phi: np.ndarray, node: int) -> float:
        """
        Compute effective radial force at a node.
        
        Positive = outward (repulsion/antigravity)
        Negative = inward (attraction/gravity)
        """
        Phi_eff = self.compute_effective_potential(phi)
        
        # Compute gradient (discrete: difference with neighbors)
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0
        
        # Average gradient toward neighbors
        grad = np.mean([Phi_eff[n] - Phi_eff[node] for n in neighbors])
        return grad  # Positive gradient = repulsion
    
    @property
    def n_sources(self) -> int:
        """Number of active χ-sources."""
        return len(self.sources)
    
    def summary(self) -> str:
        return f"ChiField(sources={self.n_sources}, η={self.config.chi_coupling:.2f})"


@dataclass 
class GeometryInverter:
    """
    Local geometry inversion for antigravity.
    
    Instead of adding χ-field, this modifies the graph structure
    locally to create "inverted" geodesics.
    
    Physical analogy: Warping spacetime to reverse geodesic curvature.
    """
    
    graph: "GraphStructure"
    config: AntigravityConfig = field(default_factory=AntigravityConfig)
    
    # Inverted regions: center -> (radius, strength)
    inverted_regions: Dict[int, Tuple[int, float]] = field(default_factory=dict)
    
    # Modified edge weights for inverted geometry
    _modified_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Resource tracking
    capacity: float = field(default=0.0)
    
    def __post_init__(self):
        if self.capacity == 0.0:
            self.capacity = self.config.initial_capacity
    
    def create_inversion_region(
        self, 
        center: int, 
        radius: Optional[int] = None,
        strength: float = 1.0
    ) -> bool:
        """
        Create inverted geometry region around a center node.
        
        In this region, effective distances are inverted:
        particles that would fall toward center now move away.
        
        Args:
            center: Center node of inversion
            radius: Radius in hops (default: config.inversion_radius)
            strength: Inversion strength (1.0 = full inversion)
            
        Returns:
            True if successful
        """
        if not self.config.inversion_enabled:
            return False
        
        if len(self.inverted_regions) >= self.config.max_inversion_regions:
            return False
        
        if self.capacity < self.config.antigrav_cost:
            return False
        
        radius = radius or self.config.inversion_radius
        
        self.inverted_regions[center] = (radius, strength)
        self.capacity -= self.config.antigrav_cost
        
        # Compute modified weights
        self._update_modified_weights()
        
        return True
    
    def remove_inversion_region(self, center: int):
        """Remove inversion region."""
        if center in self.inverted_regions:
            del self.inverted_regions[center]
            self._update_modified_weights()
    
    def _update_modified_weights(self):
        """Update edge weights for all inverted regions."""
        self._modified_weights.clear()
        
        for center, (radius, strength) in self.inverted_regions.items():
            # BFS to find all nodes within radius
            nodes_in_region = self._get_nodes_in_radius(center, radius)
            
            # Modify weights: edges pointing away from center get lower weight
            # (making "moving away" energetically favorable)
            for node in nodes_in_region:
                dist_from_center = self.graph.compute_graph_distance(center, node)
                if dist_from_center == 0:
                    continue
                
                for neighbor in self.graph.neighbors(node):
                    dist_neighbor = self.graph.compute_graph_distance(center, neighbor)
                    
                    edge = (min(node, neighbor), max(node, neighbor))
                    
                    if dist_neighbor > dist_from_center:
                        # Edge moving away: reduce weight (easier to traverse)
                        factor = 1.0 - strength * 0.5
                    elif dist_neighbor < dist_from_center:
                        # Edge moving toward center: increase weight (harder)
                        factor = 1.0 + strength * 2.0
                    else:
                        factor = 1.0
                    
                    self._modified_weights[edge] = factor
    
    def _get_nodes_in_radius(self, center: int, radius: int) -> Set[int]:
        """Get all nodes within radius hops of center."""
        from collections import deque
        
        visited = {center}
        queue = deque([(center, 0)])
        
        while queue:
            node, dist = queue.popleft()
            if dist >= radius:
                continue
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return visited
    
    def get_effective_distance(self, i: int, j: int) -> float:
        """
        Compute effective distance considering inverted regions.
        
        Returns modified distance that reflects antigravity geometry.
        """
        # Basic BFS with modified weights
        from collections import deque
        
        if i == j:
            return 0.0
        
        visited = {i: 0.0}
        queue = deque([(i, 0.0)])
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.graph.neighbors(node):
                edge = (min(node, neighbor), max(node, neighbor))
                weight = self._modified_weights.get(edge, 1.0)
                new_dist = dist + weight
                
                if neighbor == j:
                    return new_dist
                
                if neighbor not in visited or visited[neighbor] > new_dist:
                    visited[neighbor] = new_dist
                    queue.append((neighbor, new_dist))
        
        return float('inf')
    
    def compute_antigrav_acceleration(self, node: int, phi: np.ndarray) -> float:
        """
        Compute effective acceleration at a node.
        
        In inverted region: acceleration is outward (antigravity)
        Outside: normal gravity
        """
        # Check if node is in any inverted region
        for center, (radius, strength) in self.inverted_regions.items():
            dist = self.graph.compute_graph_distance(center, node)
            if dist <= radius and dist > 0:
                # In inverted region: reverse gradient
                grad = self._compute_gradient(node, phi)
                return -strength * grad  # Negative = outward
        
        # Normal gravity
        return self._compute_gradient(node, phi)
    
    def _compute_gradient(self, node: int, phi: np.ndarray) -> float:
        """Compute discrete gradient of φ at node."""
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0
        return np.mean([phi[n] - phi[node] for n in neighbors])
    
    @property 
    def n_inversions(self) -> int:
        return len(self.inverted_regions)
    
    def summary(self) -> str:
        return f"GeometryInverter(regions={self.n_inversions})"


@dataclass
class AntigravityLayer:
    """
    Unified antigravity layer combining χ-field and geometry inversion.
    
    Provides a single interface for antigravity effects.
    
    Usage:
        from world.core.antigravity import AntigravityLayer, AntigravityConfig
        
        config = AntigravityConfig(chi_coupling=0.5)
        antigrav = AntigravityLayer(world, config)
        
        # Add χ-source
        antigrav.add_source(node=100)
        antigrav.update()
        
        # Check effective force
        F_eff = antigrav.get_effective_force(node=50)
    """
    
    world: "World"
    config: AntigravityConfig = field(default_factory=AntigravityConfig)
    
    # Sub-modules
    chi_field: ChiField = field(default=None)
    geometry: GeometryInverter = field(default=None)
    
    # History
    activation_history: List[dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.chi_field is None:
            self.chi_field = ChiField(self.world.graph, self.config)
        if self.geometry is None:
            self.geometry = GeometryInverter(self.world.graph, self.config)
    
    def add_source(self, node: int, strength: float = 1.0, method: str = 'chi') -> bool:
        """
        Add antigravity source.
        
        Args:
            node: Node index
            strength: Source strength
            method: 'chi' for χ-field, 'inversion' for geometry inversion
            
        Returns:
            True if successful
        """
        if method == 'chi':
            success = self.chi_field.add_chi_source(node, strength)
        elif method == 'inversion':
            success = self.geometry.create_inversion_region(node, strength=strength)
        else:
            return False
        
        if success:
            self.activation_history.append({
                'time': self.world.t,
                'action': 'add_source',
                'node': node,
                'method': method,
                'strength': strength
            })
        
        return success
    
    def update(self):
        """Update χ-field solution."""
        if self.config.chi_coupling > 0:
            self.chi_field.solve_chi_field()
    
    def get_effective_potential(self) -> np.ndarray:
        """Get effective gravitational potential Φ_eff = φ - η·χ."""
        return self.chi_field.compute_effective_potential(self.world.phi)
    
    def get_effective_force(self, node: int) -> float:
        """
        Get effective gravitational force at node.
        
        Combines χ-field and geometry effects.
        
        Returns:
            Effective force (positive = repulsion/antigravity)
        """
        # χ-field contribution
        chi_force = self.chi_field.compute_effective_force(self.world.phi, node)
        
        # Geometry inversion contribution  
        geom_force = self.geometry.compute_antigrav_acceleration(node, self.world.phi)
        
        # Combine (geometry takes precedence in inverted regions)
        if self.geometry.n_inversions > 0:
            return geom_force
        return chi_force
    
    def measure_antigrav_effect(self, test_node: int, source_node: int) -> dict:
        """
        Measure antigravity effect between test and source nodes.
        
        Returns:
            Dictionary with force measurements and comparison to normal gravity
        """
        # Normal gravity (φ only)
        phi = self.world.phi
        normal_grad = np.mean([phi[n] - phi[test_node] 
                              for n in self.world.graph.neighbors(test_node)])
        
        # Effective force with antigravity
        eff_force = self.get_effective_force(test_node)
        
        # Distance
        dist = self.world.graph.compute_graph_distance(test_node, source_node)
        
        return {
            'test_node': test_node,
            'source_node': source_node,
            'distance': dist,
            'normal_force': normal_grad,
            'effective_force': eff_force,
            'force_ratio': eff_force / normal_grad if abs(normal_grad) > 1e-10 else 0,
            'is_antigrav': eff_force > 0  # Positive = repulsion
        }
    
    @property
    def capacity(self) -> float:
        """Combined capacity."""
        return self.chi_field.capacity + self.geometry.capacity
    
    def summary(self) -> str:
        return (f"AntigravityLayer(χ={self.chi_field.n_sources}, "
                f"inv={self.geometry.n_inversions}, "
                f"η={self.config.chi_coupling:.2f})")
    
    def __repr__(self) -> str:
        return self.summary()


def create_antigravity_layer(
    world: "World",
    chi_coupling: float = 0.5,
    enable_inversion: bool = False,
    initial_capacity: float = 10.0
) -> AntigravityLayer:
    """
    Convenience function to create AntigravityLayer.
    
    Args:
        world: World instance
        chi_coupling: η parameter (0 = no antigravity)
        enable_inversion: Enable geometry inversion mode
        initial_capacity: Starting resource capacity
        
    Returns:
        Configured AntigravityLayer
    """
    config = AntigravityConfig(
        chi_coupling=chi_coupling,
        inversion_enabled=enable_inversion,
        initial_capacity=initial_capacity
    )
    return AntigravityLayer(world=world, config=config)
