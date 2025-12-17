"""
Graph Structure for 1D RSL World with Power-Law Connections.

Implements deterministic power-law graph G over 1D lattice indices
to achieve effective 3D gravity (d_s ≈ 3).

Key insight from graph_gravity_model.ipynb:
- Power-law connections P(edge at distance d) ~ d^(-α)
- For α ≈ 2.0: spectral dimension d_s ≈ 3
- This gives φ ~ 1/r and F ~ 1/r² in graph metric

The graph is built DETERMINISTICALLY (no randomness).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


@dataclass
class GraphConfig:
    """Configuration for power-law graph construction."""
    
    N: int = 4096                  # Number of nodes (lattice size)
    alpha: float = 2.0             # Power-law exponent P(d) ~ d^(-α)
    c: float = 1.0                 # Scaling constant for edge count
    d_max: Optional[int] = None   # Maximum distance for long-range edges
    include_1d_chain: bool = True  # Include nearest-neighbor 1D edges
    periodic: bool = True          # Periodic boundary conditions
    
    def __post_init__(self):
        if self.d_max is None:
            self.d_max = self.N // 2


def build_powerlaw_edges(config: GraphConfig) -> List[Tuple[int, int]]:
    """
    Build deterministic power-law graph edges.
    
    For each distance d, we add floor(c / d^α) edges distributed
    evenly across the lattice. This is DETERMINISTIC.
    
    Args:
        config: Graph configuration
        
    Returns:
        List of edges (i, j) where i < j
    """
    N = config.N
    alpha = config.alpha
    c = config.c
    d_max = config.d_max or N // 2
    
    edges: Set[Tuple[int, int]] = set()
    
    # 1. Local 1D chain edges (always include)
    if config.include_1d_chain:
        for i in range(N - 1):
            edges.add((i, i + 1))
        # Periodic boundary
        if config.periodic:
            edges.add((0, N - 1))
    
    # 2. Long-range power-law edges
    for d in range(2, d_max + 1):
        # Number of edges at distance d
        n_edges_at_d = int(c / (d ** alpha) * N)
        
        if n_edges_at_d < 1:
            # Below threshold: still add some edges to maintain connectivity
            # Add one edge per d spacing
            n_edges_at_d = max(1, N // (d * 10))
        
        # Distribute edges evenly across lattice
        if n_edges_at_d > 0:
            step = max(1, N // n_edges_at_d)
            for start in range(0, N, step):
                i = start
                j = (i + d) % N if config.periodic else min(i + d, N - 1)
                
                if i != j:
                    edge = (min(i, j), max(i, j))
                    edges.add(edge)
    
    return sorted(edges)


def build_powerlaw_edges_v2(config: GraphConfig) -> List[Tuple[int, int]]:
    """
    Alternative construction: exact control over edge density.
    
    For each node i, connect to nodes at distances d where
    the cumulative count satisfies n_edges(≤d) ~ d^(1-α) * c.
    
    This gives more precise control over the spectral dimension.
    """
    N = config.N
    alpha = config.alpha
    c = config.c
    d_max = config.d_max or N // 2
    
    edges: Set[Tuple[int, int]] = set()
    
    # 1D chain
    if config.include_1d_chain:
        for i in range(N - 1):
            edges.add((i, i + 1))
        if config.periodic:
            edges.add((0, N - 1))
    
    # For α = 2: n_edges(d) ~ 1/d
    # Total edges at distance d for entire graph: ~ N/d
    
    for d in range(2, d_max + 1):
        # Target: c * N / d^α total edges at distance d
        target_edges = int(c * N / (d ** alpha))
        
        if target_edges >= 1:
            # Choose starting positions to be evenly spaced
            spacing = N / target_edges
            for k in range(target_edges):
                i = int(k * spacing) % N
                j = (i + d) % N if config.periodic else min(i + d, N - 1)
                
                if i != j:
                    edge = (min(i, j), max(i, j))
                    edges.add(edge)
    
    return sorted(edges)


@dataclass 
class GraphStructure:
    """
    Graph structure over 1D lattice with power-law connections.
    
    Provides:
    - Edge list and neighbor lookup
    - Graph Laplacian for φ-field evolution
    - Spectral embedding for IFACE coordinates
    - Distance computations
    """
    
    config: GraphConfig
    edges: List[Tuple[int, int]] = field(default_factory=list)
    _neighbors: Dict[int, List[int]] = field(default_factory=dict, repr=False)
    _laplacian: Optional[sparse.csr_matrix] = field(default=None, repr=False)
    _embedding_3d: Optional[np.ndarray] = field(default=None, repr=False)
    _distances: Optional[Dict[int, Dict[int, int]]] = field(default=None, repr=False)
    
    def __post_init__(self):
        if not self.edges:
            self.edges = build_powerlaw_edges_v2(self.config)
        self._build_neighbors()
    
    def _build_neighbors(self):
        """Build neighbor lookup from edge list."""
        self._neighbors = {i: [] for i in range(self.config.N)}
        for i, j in self.edges:
            self._neighbors[i].append(j)
            self._neighbors[j].append(i)
    
    @property
    def N(self) -> int:
        """Number of nodes."""
        return self.config.N
    
    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)
    
    @property
    def avg_degree(self) -> float:
        """Average node degree."""
        return 2 * len(self.edges) / self.config.N
    
    def neighbors(self, i: int) -> List[int]:
        """Get neighbors of node i."""
        return self._neighbors.get(i, [])
    
    def degree(self, i: int) -> int:
        """Get degree of node i."""
        return len(self._neighbors.get(i, []))
    
    @property
    def laplacian(self) -> sparse.csr_matrix:
        """
        Graph Laplacian matrix L = D - A.
        
        For Poisson equation: L·φ = -ρ
        """
        if self._laplacian is None:
            self._laplacian = self._build_laplacian()
        return self._laplacian
    
    def _build_laplacian(self) -> sparse.csr_matrix:
        """Build sparse Laplacian matrix."""
        N = self.config.N
        row, col, data = [], [], []
        
        # Off-diagonal: -1 for each edge
        for i, j in self.edges:
            row.extend([i, j])
            col.extend([j, i])
            data.extend([-1.0, -1.0])
        
        # Diagonal: degree of each node
        for i in range(N):
            row.append(i)
            col.append(i)
            data.append(float(self.degree(i)))
        
        L = sparse.coo_matrix((data, (row, col)), shape=(N, N))
        return L.tocsr()
    
    def laplacian_phi(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute graph Laplacian of field φ.
        
        Lφ[i] = Σ_j (φ[j] - φ[i]) for j ∈ neighbors(i)
        
        Note: Returns -L·φ to match convention ∇²φ = -ρ
        """
        L_phi = np.zeros_like(phi)
        for i in range(self.N):
            for j in self._neighbors[i]:
                L_phi[i] += phi[j] - phi[i]
        return L_phi
    
    @property
    def embedding_3d(self) -> np.ndarray:
        """
        3D spectral embedding of graph.
        
        Uses smallest non-zero eigenvectors of Laplacian.
        Preserves graph distances in Euclidean space.
        
        Returns:
            Array of shape (N, 3) with 3D coordinates
        """
        if self._embedding_3d is None:
            self._embedding_3d = self._compute_spectral_embedding()
        return self._embedding_3d
    
    def _compute_spectral_embedding(self, dim: int = 3) -> np.ndarray:
        """
        Compute spectral embedding using Laplacian eigenvectors.
        
        The smallest non-zero eigenvectors of L give coordinates
        that preserve graph structure in Euclidean space.
        """
        L = self.laplacian
        
        # Find smallest eigenvalues/vectors
        # k = dim + 1 to skip the constant eigenvector (λ=0)
        try:
            vals, vecs = sp_linalg.eigsh(L, k=dim + 2, which='SM')
        except Exception:
            # Fallback: use dense solver for small graphs
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Skip first eigenvector (constant, λ≈0), take next 'dim'
        # Usually indices 1, 2, 3
        start_idx = 1
        while start_idx < len(vals) and vals[start_idx] < 1e-10:
            start_idx += 1
        
        coords = vecs[:, start_idx:start_idx + dim]
        
        # Normalize to reasonable scale
        scale = np.std(coords)
        if scale > 0:
            coords = coords / scale
        
        return coords
    
    def get_iface_coords(self, i: int) -> Tuple[float, float, float]:
        """Get 3D IFACE coordinates for node i."""
        coords = self.embedding_3d[i]
        return (coords[0], coords[1], coords[2])
    
    def compute_graph_distance(self, i: int, j: int) -> int:
        """
        Compute shortest path distance between nodes i and j.
        
        Uses BFS for exact distance.
        """
        if i == j:
            return 0
        
        from collections import deque
        visited = {i}
        queue = deque([(i, 0)])
        
        while queue:
            node, dist = queue.popleft()
            for neighbor in self._neighbors[node]:
                if neighbor == j:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1  # Not connected
    
    def compute_all_distances_from(self, source: int) -> Dict[int, int]:
        """BFS to compute distances from source to all nodes."""
        from collections import deque
        
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            node = queue.popleft()
            for neighbor in self._neighbors[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        return distances
    
    def diameter(self) -> int:
        """Compute graph diameter (maximum shortest path)."""
        # Sample a few nodes for efficiency
        sample_size = min(10, self.N)
        sample_nodes = np.linspace(0, self.N - 1, sample_size, dtype=int)
        
        max_dist = 0
        for node in sample_nodes:
            distances = self.compute_all_distances_from(node)
            max_dist = max(max_dist, max(distances.values()))
        
        return max_dist
    
    def spectral_dimension(self, n_walks: int = 500, max_steps: int = 100) -> float:
        """
        Estimate spectral dimension via random walk return probability.
        
        P(return at step t) ~ t^(-d_s/2)
        """
        return_counts = np.zeros(max_steps)
        nodes = list(range(self.N))
        
        for _ in range(n_walks):
            start = np.random.choice(nodes)
            current = start
            
            for t in range(max_steps):
                neighbors = self._neighbors[current]
                if neighbors:
                    current = np.random.choice(neighbors)
                if current == start:
                    return_counts[t] += 1
        
        # Fit P(t) ~ t^(-d_s/2)
        return_probs = return_counts / n_walks
        t_vals = np.arange(2, max_steps)
        p_vals = return_probs[2:]
        
        valid = p_vals > 0
        if np.sum(valid) < 5:
            return np.nan
        
        log_t = np.log(t_vals[valid])
        log_p = np.log(p_vals[valid])
        
        slope, _ = np.polyfit(log_t, log_p, 1)
        d_s = -2 * slope
        
        return d_s
    
    def summary(self) -> str:
        """Return summary of graph structure."""
        return (f"GraphStructure(N={self.N}, edges={self.n_edges}, "
                f"avg_degree={self.avg_degree:.2f}, "
                f"diameter≈{self.diameter()}, α={self.config.alpha})")


def create_graph_for_gravity(N: int, target_d_s: float = 3.0) -> GraphStructure:
    """
    Create graph with target spectral dimension for gravity simulation.
    
    Uses theoretical relationship: d_s ≈ 2α/(α-1) for 1 < α < 2
    
    For d_s = 3: α = d_s / (d_s - 2) = 3 / 1 = 3 (theory)
    But empirically α ≈ 2.0 works better.
    
    Args:
        N: Number of nodes
        target_d_s: Target spectral dimension (3.0 for 1/r² gravity)
        
    Returns:
        GraphStructure configured for target gravity law
    """
    # Empirical finding: α ≈ 2.0 gives d_s ≈ 3
    # and φ ~ 1/d, F ~ 1/d² in graph metric
    alpha = 2.0
    
    config = GraphConfig(
        N=N,
        alpha=alpha,
        c=1.0,
        include_1d_chain=True,
        periodic=True
    )
    
    return GraphStructure(config=config)


# Convenience function
def theoretical_alpha_for_ds(d_s: float) -> float:
    """
    Compute theoretical α for target spectral dimension.
    
    Formula: d_s ≈ 2α/(α-1) for 1 < α < 2
    Solving: α = d_s / (d_s - 2)
    """
    if d_s <= 2:
        return float('inf')
    return d_s / (d_s - 2)
