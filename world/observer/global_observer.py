"""
Global Observer for RSL World.

Implements the complete observer loop:
    E_τ: World evolution
    O: Observation (Π_obs: S → IFACE)  
    M: Materialization (passive for now)

The observer:
1. Observes the world through IFACE projection
2. Updates semantic state (learns physical laws)
3. Records history for trajectory analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np
from collections import defaultdict

from .zorder import MortonMapper, morton_decode
from .iface import IFACEObject, IFACEField, IFACEState, IFACEHistory, ParticleType
from .semantics import (
    SemanticState, SemanticHistory, FieldEquationParams, 
    GravityLaw, ConservationLaw
)

if TYPE_CHECKING:
    from ..core.lattice import Lattice
    from ..omega.cycles import OmegaCycle


@dataclass
class ObserverConfig:
    """Configuration for GlobalObserver."""
    # Grid configuration
    lattice_size: int = 4096        # N = 16³
    grid_order: int = 4             # Morton order (16³)
    scale: float = 1.0              # Physical scale
    
    # Coarse-graining
    coarse_radius: int = 3          # R for ϕ_R
    
    # Capacity parameters
    C0: float = 2.0                 # Base capacity
    alpha: float = 0.5              # Tension-capacity coupling
    
    # Learning parameters
    min_history_for_fit: int = 10   # Minimum samples for regression
    fit_every: int = 5              # Fit field equation every N steps
    
    # History limits
    max_iface_history: int = 500
    max_semantic_history: int = 1000
    
    # Graph-based IFACE coordinates (NEW)
    use_graph_embedding: bool = False  # Use spectral embedding from graph
    graph_coords: Optional[np.ndarray] = None  # Pre-computed 3D coords (N,3)


class GlobalObserver:
    """
    Global observer that sees the entire IFACE.
    
    Implements:
    - Π_obs: Projection from 1D state to 3D IFACE
    - update_semantics: Learn physical laws from observations
    - Conservation law verification
    - Field equation fitting
    
    Example:
        observer = GlobalObserver()
        
        # Main loop
        for t in range(T):
            world.step()
            iface = observer.observe(world)
            observer.update_semantics()
            
        # Check what was learned
        print(observer.semantic_state.summary())
    """
    
    def __init__(self, config: Optional[ObserverConfig] = None):
        self.config = config or ObserverConfig()
        
        # Morton mapper for 1D → 3D
        self.mapper = MortonMapper(
            order=self.config.grid_order,
            scale=self.config.scale
        )
        
        # Semantic state (what observer "knows")
        self.semantic_state = SemanticState()
        self.semantic_history = SemanticHistory(max_size=self.config.max_semantic_history)
        
        # IFACE history for fitting
        self.iface_history = IFACEHistory(max_history=self.config.max_iface_history)
        
        # Object tracking (for velocity computation)
        self._prev_objects: Dict[int, IFACEObject] = {}
        self._object_id_counter = 0
        
        # Current time
        self.t: int = 0
        self.tau: float = 0.0  # Proper time
        
    def observe(
        self, 
        lattice: "Lattice",
        omega_cycles: Optional[List["OmegaCycle"]] = None,
    ) -> IFACEState:
        """
        Perform observation: Π_obs(S) → IFACE.
        
        Args:
            lattice: Current lattice state
            omega_cycles: Detected Ω-cycles (particles)
            
        Returns:
            IFACEState with objects and fields
        """
        # Get microscopic state
        state = lattice.to_state()
        S = state.sites
        N = len(S)
        
        # Ensure lattice size matches configuration
        if N != self.config.lattice_size:
            # Adjust if needed (pad or truncate)
            if N < self.config.lattice_size:
                S = np.pad(S, (0, self.config.lattice_size - N), constant_values=1)
            else:
                S = S[:self.config.lattice_size]
        
        # Compute fields
        phi_1d = self._compute_coarse_field(S)
        capacity_1d = self._compute_capacity(S)
        tension_1d = self._compute_tension(S)
        
        # Create 3D field
        iface_field = IFACEField.from_1d(
            phi_1d, capacity_1d, tension_1d,
            mapper=self.mapper
        )
        
        # Convert Ω-cycles to IFACE objects
        objects = []
        if omega_cycles:
            objects = self._cycles_to_objects(omega_cycles, S)
        
        # Compute global topological charge (conserved under SM rules)
        global_Q = self._compute_global_topological_charge(S)
        
        # Update proper time (average dτ/dt from capacity)
        mean_capacity = np.mean(capacity_1d)
        d_tau = mean_capacity / self.config.C0
        self.tau += d_tau
        
        # Create IFACE state
        iface_state = IFACEState(
            t=self.t,
            tau=self.tau,
            objects=objects,
            field=iface_field,
        )
        # Set global topological charge
        iface_state.global_Q = global_Q
        
        # Store in history
        self.iface_history.add(iface_state)
        
        # Update time
        self.t += 1
        
        return iface_state
    
    def update_semantics(self) -> None:
        """
        Update semantic state based on IFACE history.
        
        This is where the observer "learns" physical laws.
        """
        # Check conservation laws
        self._check_conservation()
        
        # Fit field equation periodically
        if self.t % self.config.fit_every == 0:
            self._fit_field_equation()
        
        # Update gravity law if we have particle trajectories
        self._fit_gravity_law()
        
        # Record semantic state in history
        self.semantic_history.add(self.semantic_state, self.t)
    
    def _compute_coarse_field(self, S: np.ndarray) -> np.ndarray:
        """
        Compute coarse-grained field ϕ_R.
        
        ϕ_R(i) = (1/|B_R(i)|) Σ_{j ∈ B_R(i)} s_j
        """
        R = self.config.coarse_radius
        N = len(S)
        phi = np.zeros(N, dtype=np.float64)
        
        # Use convolution for efficiency
        kernel = np.ones(2*R + 1) / (2*R + 1)
        
        # Pad for periodic boundary
        S_padded = np.concatenate([S[-R:], S, S[:R]])
        phi = np.convolve(S_padded, kernel, mode='valid')[:N]
        
        return phi
    
    def _compute_global_topological_charge(self, S: np.ndarray) -> float:
        """
        Compute global topological charge of the lattice.
        
        This is the net number of domain walls:
        Q_global = N(+→-) - N(-→+)
        
        For periodic boundary, this is ALWAYS ZERO because every +→- wall
        must be matched by a -→+ wall to close the ring.
        
        For conservation analysis, we track the number of domain walls instead.
        The total count should be conserved under SM rules.
        
        Returns:
            Global topological charge (should be conserved)
        """
        N = len(S)
        n_plus_minus = 0  # +→- transitions
        n_minus_plus = 0  # -→+ transitions
        
        for i in range(N):
            next_i = (i + 1) % N  # Periodic boundary
            if S[i] == 1 and S[next_i] == -1:
                n_plus_minus += 1
            elif S[i] == -1 and S[next_i] == 1:
                n_minus_plus += 1
        
        # Net charge (for periodic: always 0)
        # But we return the count for analysis
        # Return wall count as "mass-like" conserved quantity
        return float(n_plus_minus + n_minus_plus)  # Total domain walls = conserved
    
    def _compute_tension(self, S: np.ndarray) -> np.ndarray:
        """
        Compute local tension H_i.
        
        H_i = number of domain walls at site i
        """
        N = len(S)
        H = np.zeros(N, dtype=np.float64)
        
        # Left boundary
        H[1:] += (S[1:] != S[:-1]).astype(float)
        # Right boundary  
        H[:-1] += (S[:-1] != S[1:]).astype(float)
        
        # Periodic boundary
        H[0] += (S[0] != S[-1])
        H[-1] += (S[-1] != S[0])
        
        return H
    
    def _compute_capacity(self, S: np.ndarray) -> np.ndarray:
        """
        Compute local capacity C_i = C0 - α*H_i.
        """
        H = self._compute_tension(S)
        C = self.config.C0 - self.config.alpha * H
        return np.maximum(C, 0.1)  # Ensure positive capacity
    
    def _cycles_to_objects(
        self, 
        cycles: List["OmegaCycle"],
        S: np.ndarray
    ) -> List[IFACEObject]:
        """
        Convert Ω-cycles to IFACE objects with 3D coordinates.
        """
        objects = []
        current_ids = set()
        
        for cycle in cycles:
            # Get 1D position (center of support)
            pos_1d = int(cycle.position) % self.config.lattice_size
            
            # Convert to 3D
            coords = self.mapper.map_index(pos_1d)
            pos_3d = (coords.x, coords.y, coords.z)
            
            # Compute mass from H_core (or use period as proxy)
            mass = self._estimate_mass(cycle, S)
            
            # Compute charge from pattern
            Q = self._estimate_charge(cycle, S)
            
            # Find matching previous object for velocity
            obj_id = self._match_object(cycle, pos_3d)
            current_ids.add(obj_id)
            
            # Create object
            obj = IFACEObject(
                id=obj_id,
                mass=mass,
                Q=Q,
                pos=pos_3d,
                period=cycle.period,
                support_size=cycle.extent,
                creation_time=cycle.creation_time,
            )
            
            # Update velocity from previous position
            if obj_id in self._prev_objects:
                prev = self._prev_objects[obj_id]
                obj.update_velocity(pos_3d)
                obj.previous_pos = prev.pos
            
            objects.append(obj)
        
        # Update tracking
        self._prev_objects = {obj.id: obj for obj in objects}
        
        return objects
    
    def _estimate_mass(self, cycle: "OmegaCycle", S: np.ndarray) -> float:
        """Estimate mass from Ω-cycle properties."""
        # Simple estimate: period inversely related to mass (E ~ ħω)
        # Or use extent as proxy for localization energy
        return float(cycle.extent) / cycle.period if cycle.period > 0 else 1.0
    
    def _estimate_charge(self, cycle: "OmegaCycle", S: np.ndarray) -> float:
        """
        Estimate charge from Ω-cycle pattern based on domain wall topology.
        
        For SM-like rules (++- ↔ -++):
        - +/- domain wall → particle (Q = +1)
        - -/+ domain wall → antiparticle (Q = -1)
        - Neutral cycles have balanced walls (Q = 0)
        
        Total charge is conserved: sum(Q) = const
        """
        # Get local pattern around cycle
        pos = int(cycle.position) % len(S)
        half_extent = max(cycle.extent // 2, 2)
        left = max(0, pos - half_extent)
        right = min(len(S), pos + half_extent + 1)
        
        local_pattern = S[left:right]
        
        if len(local_pattern) < 2:
            return 0.0
        
        # Count domain wall transitions (topological charge)
        # +→- transition: contributes +1
        # -→+ transition: contributes -1
        charge = 0.0
        for i in range(len(local_pattern) - 1):
            if local_pattern[i] == 1 and local_pattern[i+1] == -1:
                charge += 1.0  # +/- wall
            elif local_pattern[i] == -1 and local_pattern[i+1] == 1:
                charge -= 1.0  # -/+ wall
        
        # Normalize to [-1, 0, +1] for particle classification
        if charge > 0.5:
            return 1.0
        elif charge < -0.5:
            return -1.0
        return 0.0
    
    def _match_object(
        self, 
        cycle: "OmegaCycle",
        pos_3d: Tuple[float, float, float]
    ) -> int:
        """Match cycle to existing object or create new ID."""
        # Find closest previous object
        min_dist = float('inf')
        best_id = None
        
        for obj_id, obj in self._prev_objects.items():
            dx = pos_3d[0] - obj.pos[0]
            dy = pos_3d[1] - obj.pos[1]
            dz = pos_3d[2] - obj.pos[2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < min_dist and dist < 0.2:  # Threshold for matching
                min_dist = dist
                best_id = obj_id
        
        if best_id is not None:
            return best_id
        
        # New object
        self._object_id_counter += 1
        return self._object_id_counter
    
    def _check_conservation(self) -> None:
        """Check conservation laws from IFACE history."""
        if self.iface_history.length < 3:
            return
        
        # Get history using global topological charge (conserved)
        # NOT total_Q which depends on object detection count
        Q_history = self.iface_history.get_global_Q_history()
        
        # Use field energy for mass conservation (independent of object count)
        field_energy = self.iface_history.get_field_energy_history()
        
        # Check charge conservation (using global topological charge = domain walls)
        self.semantic_state.charge_conservation.check(Q_history)
        
        # Check energy/mass conservation using field energy
        self.semantic_state.mass_conservation.check(field_energy)
    
    def _fit_field_equation(self) -> None:
        """
        Fit field equation parameters κ, m², λ from ϕ(x,t) history.
        
        Equation: ∂²ϕ/∂t² = κ∇²ϕ - m²ϕ - λϕ³
        """
        if self.iface_history.length < self.config.min_history_for_fit:
            return
        
        # Get phi history
        phi_history = self.iface_history.get_phi_history()
        if len(phi_history) < 3:
            return
        
        T, n, _, _ = phi_history.shape
        
        # Build regression data
        X_data = []
        Y_data = []
        
        for t in range(1, T - 1):
            phi_t = phi_history[t]
            phi_tm1 = phi_history[t-1]
            phi_tp1 = phi_history[t+1]
            
            # Second time derivative
            phi_tt = phi_tp1 - 2*phi_t + phi_tm1
            
            # Laplacian (using finite differences)
            lap_phi = (
                np.roll(phi_t, -1, axis=0) + np.roll(phi_t, 1, axis=0) +
                np.roll(phi_t, -1, axis=1) + np.roll(phi_t, 1, axis=1) +
                np.roll(phi_t, -1, axis=2) + np.roll(phi_t, 1, axis=2) -
                6 * phi_t
            )
            
            # Sample subset of points (for efficiency)
            sample_indices = np.random.choice(n**3, min(1000, n**3), replace=False)
            
            for idx in sample_indices:
                i, j, k = np.unravel_index(idx, (n, n, n))
                
                # Features: [∇²ϕ, -ϕ, -ϕ³]
                X_data.append([
                    lap_phi[i,j,k],
                    -phi_t[i,j,k],
                    -phi_t[i,j,k]**3
                ])
                Y_data.append(phi_tt[i,j,k])
        
        if len(X_data) < 10:
            return
        
        X = np.array(X_data)
        Y = np.array(Y_data)
        
        # Linear regression: Y = X @ θ
        try:
            theta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
            
            # Compute R²
            Y_pred = X @ theta
            ss_res = np.sum((Y - Y_pred)**2)
            ss_tot = np.sum((Y - np.mean(Y))**2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)
            
            params = FieldEquationParams(
                kappa=float(theta[0]),
                m2=float(theta[1]),
                lambda_=float(theta[2]) if len(theta) > 2 else 0.0,
                r_squared=float(r_squared),
                residual=float(np.sqrt(ss_res / len(Y))),
                n_samples=len(Y)
            )
            
            self.semantic_state.update_field_eq(params)
            
        except Exception as e:
            # Regression failed
            pass
    
    def _fit_gravity_law(self) -> None:
        """
        Fit gravity law: a = -γ∇Φ
        
        Uses particle trajectories and local potential from capacity.
        """
        if self.iface_history.length < 5:
            return
        
        # Collect acceleration and gradient data
        a_data = []
        grad_phi_data = []
        
        # Get recent states
        states = self.iface_history.states[-20:]
        
        for i in range(1, len(states) - 1):
            state_prev = states[i-1]
            state = states[i]
            state_next = states[i+1]
            
            if state.field is None:
                continue
            
            # Compute potential from capacity
            potential = state.field.potential_from_capacity(self.config.C0)
            grad_x, grad_y, grad_z = self._gradient_3d(potential)
            
            for obj in state.objects:
                obj_prev = state_prev.get_object_by_id(obj.id)
                obj_next = state_next.get_object_by_id(obj.id)
                
                if obj_prev is None or obj_next is None:
                    continue
                
                # Acceleration
                ax = obj_next.pos[0] - 2*obj.pos[0] + obj_prev.pos[0]
                ay = obj_next.pos[1] - 2*obj.pos[1] + obj_prev.pos[1]
                az = obj_next.pos[2] - 2*obj.pos[2] + obj_prev.pos[2]
                
                # Get gradient at position
                xi = int(obj.pos[0] * state.field.dim_size) % state.field.dim_size
                yi = int(obj.pos[1] * state.field.dim_size) % state.field.dim_size
                zi = int(obj.pos[2] * state.field.dim_size) % state.field.dim_size
                
                gx = grad_x[xi, yi, zi]
                gy = grad_y[xi, yi, zi]
                gz = grad_z[xi, yi, zi]
                
                a_data.extend([ax, ay, az])
                grad_phi_data.extend([-gx, -gy, -gz])
        
        if len(a_data) < 10:
            return
        
        a = np.array(a_data)
        g = np.array(grad_phi_data)
        
        # Fit a = γ * (-∇Φ)
        try:
            # Simple linear fit
            gamma = np.dot(a, g) / (np.dot(g, g) + 1e-10)
            
            # Correlation
            if len(a) > 1:
                corr = np.corrcoef(a, g)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
            
            self.semantic_state.update_gravity(GravityLaw(
                gamma=float(gamma),
                correlation=float(corr),
                n_samples=len(a)
            ))
            
        except Exception:
            pass
    
    def _gradient_3d(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D gradient using central differences."""
        grad_x = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / 2
        grad_y = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / 2
        grad_z = (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / 2
        return grad_x, grad_y, grad_z
    
    def reset(self) -> None:
        """Reset observer state."""
        self.semantic_state = SemanticState()
        self.semantic_history.clear()
        self.iface_history.clear()
        self._prev_objects.clear()
        self._object_id_counter = 0
        self.t = 0
        self.tau = 0.0
    
    def set_graph_embedding(self, coords_3d: np.ndarray) -> None:
        """
        Set graph-based 3D coordinates for IFACE.
        
        This replaces Morton mapping with spectral embedding from
        the power-law graph, enabling correct 1/r² gravity.
        
        Args:
            coords_3d: Array of shape (N, 3) with 3D coordinates
        """
        self.config.use_graph_embedding = True
        self.config.graph_coords = coords_3d
    
    def get_iface_position(self, idx_1d: int) -> Tuple[float, float, float]:
        """
        Get 3D IFACE coordinates for 1D index.
        
        Uses either Morton mapping or graph spectral embedding.
        """
        if self.config.use_graph_embedding and self.config.graph_coords is not None:
            coords = self.config.graph_coords[idx_1d % len(self.config.graph_coords)]
            return (coords[0], coords[1], coords[2])
        else:
            coords = self.mapper.map_index(idx_1d)
            return (coords.x, coords.y, coords.z)
    
    def compute_graph_gradient_phi(
        self, 
        phi: np.ndarray, 
        idx: int,
        graph_neighbors: List[int]
    ) -> Tuple[float, float, float]:
        """
        Compute gradient of φ at node idx using graph structure.
        
        ∇φ[idx] ≈ Σ_j (φ[j] - φ[idx]) * (r_j - r_idx) / |r_j - r_idx|²
        
        Args:
            phi: Field values at all nodes
            idx: Node index
            graph_neighbors: List of neighbor indices
            
        Returns:
            3D gradient vector in IFACE coordinates
        """
        if not self.config.use_graph_embedding or self.config.graph_coords is None:
            return (0.0, 0.0, 0.0)
        
        r_i = self.config.graph_coords[idx]
        grad = np.zeros(3)
        
        for j in graph_neighbors:
            r_j = self.config.graph_coords[j]
            dr = r_j - r_i
            dist_sq = np.dot(dr, dr)
            
            if dist_sq > 1e-10:
                dphi = phi[j] - phi[idx]
                grad += dphi * dr / dist_sq
        
        if graph_neighbors:
            grad /= len(graph_neighbors)
        
        return (grad[0], grad[1], grad[2])
    
    @property
    def knowledge(self) -> Dict[str, Any]:
        """Get summary of observer's knowledge."""
        return self.semantic_state.summary()
    
    def __repr__(self) -> str:
        return f"GlobalObserver(t={self.t}, τ={self.tau:.2f}, {self.semantic_state})"


# Convenience function for quick setup
def create_observer(lattice_size: int = 4096) -> GlobalObserver:
    """Create observer with default configuration for given lattice size."""
    order = int(np.log2(int(round(lattice_size ** (1/3)))))
    config = ObserverConfig(
        lattice_size=lattice_size,
        grid_order=order
    )
    return GlobalObserver(config)
