"""
Extended World with Graph-based φ-field for Gravity.

Combines:
- 1D spin lattice with SM-rules (++- ↔ -++)
- Power-law graph G for long-range connections
- φ-field evolution via graph Laplacian

The φ-field acts as gravitational potential:
    ∂φ/∂t = D·Lφ + β·ρ(s) - γ·φ

where L is the graph Laplacian, ρ(s) is source from spins/defects.

With α=2 power-law graph, this gives φ ~ 1/r and F ~ 1/r² in IFACE.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

from .lattice import Lattice, LatticeState
from .rules import RuleSet
from .evolution import EvolutionEngine
from .graph_structure import GraphStructure, GraphConfig, create_graph_for_gravity

if TYPE_CHECKING:
    from ..omega.cycles import OmegaCycle


@dataclass
class WorldConfig:
    """Configuration for World with graph-based gravity."""
    
    # Lattice
    N: int = 4096                  # Lattice size (16³ or other)
    initial_state: str = "vacuum"  # "vacuum", "random", "defects"
    defect_density: float = 0.05   # For "random" initial state
    
    # Graph structure
    graph_alpha: float = 2.0       # Power-law exponent (2.0 for 3D gravity)
    graph_c: float = 1.0           # Edge density scale
    
    # φ-field dynamics
    D_phi: float = 0.1             # Diffusion coefficient
    beta_source: float = 0.01      # Source coupling to spins
    gamma_decay: float = 0.001     # φ decay rate (regularization)
    
    # Evolution
    spin_rule_priority: bool = True  # Spins update before φ


class World:
    """
    RSL World with graph-based gravitational field.
    
    Layers:
    1. Spin layer s[i] ∈ {+1, -1}: Matter (SM-rules)
    2. φ layer φ[i] ∈ R: Gravitational potential (graph Laplacian)
    
    The graph structure G creates effective 3D geometry:
    - Power-law connections P(d) ~ d^(-α)
    - Spectral dimension d_s ≈ 3 for α ≈ 2
    - φ evolves by ∂φ/∂t = D·Lφ + source(s)
    
    Example:
        config = WorldConfig(N=4096, graph_alpha=2.0)
        ruleset = create_sm_ruleset()  # ++- ↔ -++
        world = World(config, ruleset)
        
        for t in range(1000):
            world.step()
            
        print(f"φ range: {world.phi.min():.3f} to {world.phi.max():.3f}")
    """
    
    def __init__(
        self, 
        config: WorldConfig,
        ruleset: RuleSet,
        graph: Optional[GraphStructure] = None,
    ):
        self.config = config
        self.ruleset = ruleset
        self.N = config.N
        
        # Build graph structure (deterministic)
        if graph is None:
            graph_config = GraphConfig(
                N=config.N,
                alpha=config.graph_alpha,
                c=config.graph_c,
                include_1d_chain=True,
                periodic=True
            )
            self.graph = GraphStructure(config=graph_config)
        else:
            self.graph = graph
        
        # Initialize spin layer
        self._init_spins()
        
        # Initialize φ-field (gravitational potential)
        self.phi = np.zeros(self.N, dtype=np.float64)
        
        # Evolution engine for spin rules
        self.lattice = Lattice(size=self.N)
        self.lattice.sites[:] = self.s
        self.engine = EvolutionEngine(self.ruleset)
        
        # Time tracking
        self.t = 0
        
        # Cache for Ω-cycles (set externally by detector)
        self.omega_cycles: List["OmegaCycle"] = []
    
    def _init_spins(self):
        """Initialize spin configuration."""
        cfg = self.config
        
        if cfg.initial_state == "vacuum":
            self.s = np.ones(self.N, dtype=np.int8)
        elif cfg.initial_state == "random":
            p_minus = cfg.defect_density
            self.s = np.random.choice(
                [-1, 1], 
                size=self.N, 
                p=[p_minus, 1 - p_minus]
            ).astype(np.int8)
        elif cfg.initial_state == "defects":
            # Create two well-separated defect clusters
            self.s = np.ones(self.N, dtype=np.int8)
            # Defect 1 at position N//4
            pos1 = self.N // 4
            self.s[pos1-5:pos1+5] = -1
            # Defect 2 at position 3*N//4
            pos2 = 3 * self.N // 4
            self.s[pos2-5:pos2+5] = -1
        else:
            raise ValueError(f"Unknown initial state: {cfg.initial_state}")
    
    def step(self):
        """
        One evolution step E_τ.
        
        Updates both spin layer (SM-rules) and φ-field (graph Laplacian).
        """
        if self.config.spin_rule_priority:
            self._step_spins()
            self._step_phi()
        else:
            self._step_phi()
            self._step_spins()
        
        self.t += 1
    
    def _step_spins(self):
        """Update spins via SM-rules (++- ↔ -++)."""
        # Sync lattice with current state
        self.lattice.sites[:] = self.s
        
        # Apply rules via evolution engine
        result = self.engine.step(self.lattice)
        
        # Get updated state
        self.s = self.lattice.sites.copy()
    
    def _step_phi(self):
        """
        Update φ-field via graph Laplacian dynamics.
        
        φ(t+1) = φ(t) + D·Lφ + β·ρ(s) - γ·φ
        
        where ρ(s) is source term from defects/spins.
        """
        cfg = self.config
        
        # Graph Laplacian: Lφ[i] = Σ_j (φ[j] - φ[i])
        L_phi = self.graph.laplacian_phi(self.phi)
        
        # Source from spins: defects (s=-1) are sources
        # ρ[i] = (1 - s[i]) / 2, so ρ=1 for s=-1, ρ=0 for s=+1
        rho = (1 - self.s) / 2.0
        
        # Update φ
        self.phi += (
            cfg.D_phi * L_phi +       # Diffusion via graph
            cfg.beta_source * rho -    # Source from defects
            cfg.gamma_decay * self.phi # Decay/regularization
        )
    
    def get_phi_gradient(self, i: int) -> np.ndarray:
        """
        Compute gradient of φ at node i in IFACE coordinates.
        
        ∇φ[i] ≈ average of (φ[j] - φ[i]) * (r_j - r_i) / |r_j - r_i|²
        
        Returns:
            3D gradient vector in IFACE coordinates
        """
        grad = np.zeros(3)
        r_i = self.graph.embedding_3d[i]
        
        neighbors = self.graph.neighbors(i)
        if not neighbors:
            return grad
        
        for j in neighbors:
            r_j = self.graph.embedding_3d[j]
            dr = r_j - r_i
            dist_sq = np.dot(dr, dr)
            
            if dist_sq > 1e-10:
                dphi = self.phi[j] - self.phi[i]
                grad += dphi * dr / dist_sq
        
        grad /= len(neighbors)
        return grad
    
    def get_state(self) -> dict:
        """Return current world state for observation."""
        return {
            't': self.t,
            's': self.s.copy(),
            'phi': self.phi.copy(),
            'graph': self.graph,
        }
    
    def to_lattice_state(self) -> LatticeState:
        """Convert current state to LatticeState."""
        return LatticeState(
            sites=self.s.copy(),
            time=self.t,
            level=0,
            metadata={'phi_max': float(self.phi.max())}
        )
    
    @property
    def iface_coords(self) -> np.ndarray:
        """Get 3D IFACE coordinates for all nodes."""
        return self.graph.embedding_3d
    
    @property
    def topological_charge(self) -> int:
        """
        Compute global topological charge Q.
        
        Q = Σ_i (1 - s[i]) / 2 = number of s=-1 sites
        """
        return int(np.sum((1 - self.s) // 2))
    
    @property
    def total_mass(self) -> float:
        """
        Compute total "mass" (proxy for defect count).
        
        Could be refined with Ω-cycle detection.
        """
        return float(self.topological_charge)
    
    def solve_poisson(self) -> np.ndarray:
        """
        Solve Poisson equation L·φ = -ρ on graph.
        
        Returns stationary φ-field for current spin configuration.
        Useful for measuring F(r) directly.
        """
        from scipy.sparse.linalg import spsolve
        
        L = self.graph.laplacian
        rho = (1 - self.s) / 2.0
        
        # Regularize for solvability (L is singular)
        L_reg = L + 0.01 * sparse.eye(self.N)
        
        phi_stationary = spsolve(L_reg.tocsr(), rho)
        return phi_stationary
    
    def summary(self) -> str:
        """Return summary of world state."""
        return (f"World(N={self.N}, t={self.t}, "
                f"Q={self.topological_charge}, "
                f"φ_range=[{self.phi.min():.3f}, {self.phi.max():.3f}], "
                f"graph: {self.graph.n_edges} edges, α={self.config.graph_alpha})")


# Need to import sparse for solve_poisson
from scipy import sparse


def create_world_with_gravity(
    N: int = 4096,
    ruleset: Optional[RuleSet] = None,
    initial_state: str = "defects",
) -> World:
    """
    Convenience function to create World configured for gravity experiments.
    
    Args:
        N: Lattice size
        ruleset: Spin rules (default: SM-rules)
        initial_state: Initial spin configuration
        
    Returns:
        World with power-law graph (α=2) for 1/r² gravity
    """
    # Use SM-rules if not specified
    if ruleset is None:
        from .rules import RuleSet, Rule
        ruleset = RuleSet(rules=[
            Rule(name="sm_r", pattern=[1, 1, -1], replacement=[-1, 1, 1]),  # ++- → -++
            Rule(name="sm_l", pattern=[-1, 1, 1], replacement=[1, 1, -1]),  # -++ → ++-
        ])
    
    config = WorldConfig(
        N=N,
        graph_alpha=2.0,  # For d_s ≈ 3 and F ~ 1/r²
        initial_state=initial_state,
    )
    
    return World(config, ruleset)
