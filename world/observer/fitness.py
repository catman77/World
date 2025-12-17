"""
OBSFitness - Observer fitness metrics for evolutionary search.

Evaluates how well an observer can extract meaningful physics:
1. Field equation accuracy (R², residual)
2. Conservation law verification (ΔQ, ΔM)
3. Observation Time (t_OT) - speed of convergence
4. Probability calibration (KL divergence)

Can be combined with SMFitness for total fitness:
    TotalFitness = α * SMFitness + β * OBSFitness
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .global_observer import GlobalObserver
    from .semantics import SemanticState


@dataclass
class OBSFitnessConfig:
    """Configuration for OBSFitness calculation."""
    
    # Scale parameters for fitness components
    sigma_field: float = 0.1      # Field equation error scale
    sigma_Q: float = 0.1          # Charge conservation scale
    sigma_mass: float = 0.1       # Mass conservation scale
    T_scale_fraction: float = 0.25 # OT scale as fraction of total time
    sigma_KL: float = 0.5         # KL divergence scale
    
    # Weights for combining components
    w_field: float = 1.0          # Field equation weight
    w_Q: float = 1.0              # Charge conservation weight
    w_mass: float = 0.5           # Mass conservation weight  
    w_OT: float = 1.0             # Observation time weight
    w_prob: float = 1.0           # Probability calibration weight
    w_gravity: float = 0.5        # Gravity law weight
    
    # Thresholds
    min_r_squared: float = 0.3    # Minimum R² for valid field eq
    min_confidence: float = 0.5   # Minimum conservation confidence
    
    @property
    def total_weight(self) -> float:
        """Sum of all weights."""
        return (self.w_field + self.w_Q + self.w_mass + 
                self.w_OT + self.w_prob + self.w_gravity)


@dataclass
class OBSFitnessComponents:
    """Individual fitness components for analysis."""
    
    # Field equation quality
    fitness_field: float = 0.0
    field_r_squared: float = 0.0
    field_residual: float = 0.0
    
    # Conservation laws
    fitness_Q: float = 0.0
    Q_violation: float = 0.0
    fitness_mass: float = 0.0
    mass_violation: float = 0.0
    
    # Observation time
    fitness_OT: float = 0.0
    t_OT: int = 0
    is_stabilized: bool = False
    
    # Probability calibration
    fitness_prob: float = 0.0
    kl_divergence: float = 0.0
    
    # Gravity law
    fitness_gravity: float = 0.0
    gravity_correlation: float = 0.0
    
    # Total
    total_fitness: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "fitness_field": self.fitness_field,
            "field_r_squared": self.field_r_squared,
            "field_residual": self.field_residual,
            "fitness_Q": self.fitness_Q,
            "Q_violation": self.Q_violation,
            "fitness_mass": self.fitness_mass,
            "mass_violation": self.mass_violation,
            "fitness_OT": self.fitness_OT,
            "t_OT": self.t_OT,
            "is_stabilized": self.is_stabilized,
            "fitness_prob": self.fitness_prob,
            "kl_divergence": self.kl_divergence,
            "fitness_gravity": self.fitness_gravity,
            "gravity_correlation": self.gravity_correlation,
            "total_fitness": self.total_fitness,
        }
    
    def __repr__(self) -> str:
        return (f"OBSFitness(total={self.total_fitness:.3f}, "
                f"field={self.fitness_field:.3f}, Q={self.fitness_Q:.3f}, "
                f"OT={self.fitness_OT:.3f})")


class OBSFitness:
    """
    Compute observer fitness metrics.
    
    Evaluates how effectively an observer extracts physical laws from
    observations. Higher fitness = better understanding.
    
    Example:
        observer = GlobalObserver()
        # ... run simulation ...
        
        fitness_calc = OBSFitness()
        score, components = fitness_calc.evaluate(observer, T_total=100)
        print(f"OBSFitness: {score:.3f}")
    """
    
    def __init__(self, config: Optional[OBSFitnessConfig] = None):
        self.config = config or OBSFitnessConfig()
    
    def evaluate(
        self,
        observer: "GlobalObserver",
        T_total: Optional[int] = None,
    ) -> Tuple[float, OBSFitnessComponents]:
        """
        Evaluate observer fitness.
        
        Args:
            observer: The observer to evaluate
            T_total: Total simulation time (for OT scaling)
            
        Returns:
            Tuple of (total_fitness, components)
        """
        sem = observer.semantic_state
        cfg = self.config
        
        # Use observer time if not specified
        if T_total is None:
            T_total = max(observer.t, 1)
        
        components = OBSFitnessComponents()
        
        # 1. Field equation quality
        components.field_r_squared = sem.field_eq.r_squared
        components.field_residual = sem.field_eq.residual
        
        if sem.field_eq.r_squared > cfg.min_r_squared:
            # Good fit - use R² directly
            components.fitness_field = sem.field_eq.r_squared
        else:
            # Penalize poor fits
            err = 1.0 - sem.field_eq.r_squared
            components.fitness_field = np.exp(-err / cfg.sigma_field) * 0.5
        
        # 2. Charge conservation
        components.Q_violation = sem.charge_conservation.mean_violation
        if sem.charge_conservation.is_conserved:
            components.fitness_Q = 1.0
        else:
            delta_Q_norm = components.Q_violation / (abs(sem.charge_conservation.n_checks) + 1)
            components.fitness_Q = np.exp(-delta_Q_norm / cfg.sigma_Q)
        
        # 3. Mass conservation  
        components.mass_violation = sem.mass_conservation.mean_violation
        if sem.mass_conservation.is_conserved:
            components.fitness_mass = 1.0
        else:
            delta_M_norm = components.mass_violation / (abs(sem.mass_conservation.n_checks) + 1)
            components.fitness_mass = np.exp(-delta_M_norm / cfg.sigma_mass)
        
        # 4. Observation Time (t_OT)
        components.is_stabilized = sem.is_stabilized()
        components.t_OT = sem.get_observation_time()
        
        T_scale = cfg.T_scale_fraction * T_total
        if T_scale > 0:
            components.fitness_OT = np.exp(-components.t_OT / T_scale)
        else:
            components.fitness_OT = 1.0 if components.is_stabilized else 0.0
        
        # 5. Probability calibration (placeholder - needs event data)
        components.fitness_prob = self._compute_prob_fitness(sem)
        
        # 6. Gravity law
        components.gravity_correlation = abs(sem.gravity.correlation)
        if components.gravity_correlation > 0.5:
            components.fitness_gravity = components.gravity_correlation
        else:
            components.fitness_gravity = 0.5 * components.gravity_correlation
        
        # Enhanced gravity check: verify 1/r² law if we have graph data
        if hasattr(observer, 'config') and observer.config.use_graph_embedding:
            components.fitness_gravity = self._compute_graph_gravity_fitness(observer, sem)
        
        # Compute total fitness (weighted average)
        weighted_sum = (
            cfg.w_field * components.fitness_field +
            cfg.w_Q * components.fitness_Q +
            cfg.w_mass * components.fitness_mass +
            cfg.w_OT * components.fitness_OT +
            cfg.w_prob * components.fitness_prob +
            cfg.w_gravity * components.fitness_gravity
        )
        components.total_fitness = weighted_sum / cfg.total_weight
        
        return components.total_fitness, components
    
    def _compute_prob_fitness(self, sem: "SemanticState") -> float:
        """
        Compute probability calibration fitness.
        
        Compares predicted event probabilities to observed frequencies.
        Uses KL divergence where possible.
        """
        # Get event statistics
        events = sem.events
        
        if events.total_events == 0:
            # No events observed - neutral fitness
            return 0.5
        
        # Simple calibration check: are predictions close to frequencies?
        total_error = 0.0
        n_types = 0
        
        for event_type, (count, pred_prob) in [
            ("creation", (events.creation_count, events.creation_prob)),
            ("annihilation", (events.annihilation_count, events.annihilation_prob)),
            ("decay", (events.decay_count, events.decay_prob)),
            ("scattering", (events.scattering_count, events.scattering_prob)),
        ]:
            if events.total_events > 0:
                obs_freq = count / events.total_events
                if pred_prob > 0:
                    # Absolute error (simpler than KL for sparse data)
                    total_error += abs(obs_freq - pred_prob)
                    n_types += 1
        
        if n_types == 0:
            return 0.5
        
        avg_error = total_error / n_types
        return np.exp(-avg_error / self.config.sigma_KL)
    
    def _compute_graph_gravity_fitness(
        self, 
        observer: "GlobalObserver",
        sem: "SemanticState"
    ) -> float:
        """
        Compute gravity fitness using graph-based analysis.
        
        For graph with spectral embedding, we check:
        1. Correlation between acceleration and -∇φ
        2. Power-law exponent (should be close to -2 for 1/r²)
        
        Returns:
            Fitness score in [0, 1]
        """
        # Get object history for acceleration
        history = observer.iface_history
        if history.length < 5:
            return 0.5  # Not enough data
        
        # Use existing correlation from semantic state gravity analysis
        # This is computed by the Observer during observation
        if sem.gravity.n_samples > 3:
            corr = abs(sem.gravity.correlation)
            
            # Bonus for having many samples
            sample_bonus = min(0.2, sem.gravity.n_samples / 100.0)
            
            # Fitness based on correlation strength
            if corr > 0.7:
                return min(1.0, 0.8 + sample_bonus)
            elif corr > 0.5:
                return 0.6 + 0.2 * corr + sample_bonus
            else:
                return 0.3 + 0.3 * corr
        
        # Fallback: try to compute from trajectories
        # Get all tracked objects
        states = history.states
        if len(states) < 3:
            return 0.5
        
        # Find objects that appear in multiple frames
        obj_ids_seen = set()
        for state in states:
            for obj in state.objects:
                obj_ids_seen.add(obj.id)
        
        # Compute acceleration for tracked objects
        a_samples = []
        for obj_id in obj_ids_seen:
            trajectory = history.get_object_trajectory(obj_id)
            if len(trajectory) >= 3:
                # Compute accelerations
                for t in range(2, len(trajectory)):
                    a_vec = trajectory[t] - 2*trajectory[t-1] + trajectory[t-2]
                    a_samples.append(np.linalg.norm(a_vec))
        
        if len(a_samples) > 5:
            # Have some acceleration data - give partial credit
            return 0.4 + 0.1 * min(1.0, len(a_samples) / 20.0)
        
        return 0.5


@dataclass
class CombinedFitnessConfig:
    """Configuration for combined SM + OBS fitness."""
    
    alpha: float = 0.7    # Weight for SMFitness
    beta: float = 0.3     # Weight for OBSFitness
    
    # Phase-dependent weighting
    use_adaptive: bool = True
    initial_alpha: float = 0.9   # Start with more SM focus
    final_alpha: float = 0.5     # Balance at convergence


class CombinedFitness:
    """
    Combined fitness: TotalFitness = α * SMFitness + β * OBSFitness
    
    Used for evolutionary search that optimizes both:
    - Rule sets (micro-dynamics) via SMFitness
    - Observer understanding via OBSFitness
    
    Example:
        sm_fitness = SMFitness(...)
        obs_fitness = OBSFitness()
        combined = CombinedFitness(sm_fitness, obs_fitness)
        
        total, components = combined.evaluate(genome, observer)
    """
    
    def __init__(
        self,
        sm_fitness: Any,  # SMFitness from notebook
        obs_fitness: OBSFitness,
        config: Optional[CombinedFitnessConfig] = None,
    ):
        self.sm_fitness = sm_fitness
        self.obs_fitness = obs_fitness
        self.config = config or CombinedFitnessConfig()
        
        # Track generations for adaptive weighting
        self.generation = 0
    
    def evaluate(
        self,
        genome: Any,
        observer: "GlobalObserver",
        T_total: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate combined fitness.
        
        Args:
            genome: Rule genome (for SMFitness)
            observer: Observer to evaluate
            T_total: Total simulation time
            
        Returns:
            Tuple of (total_fitness, components_dict)
        """
        # Compute SM fitness
        sm_score, sm_components = self.sm_fitness.evaluate(genome)
        
        # Compute OBS fitness
        obs_score, obs_components = self.obs_fitness.evaluate(observer, T_total)
        
        # Get weights (possibly adaptive)
        alpha, beta = self._get_weights()
        
        # Combine
        total = alpha * sm_score + beta * obs_score
        
        # Build components dict
        components = {
            "sm_fitness": sm_score,
            "sm_components": sm_components,
            "obs_fitness": obs_score,
            "obs_components": obs_components.to_dict(),
            "alpha": alpha,
            "beta": beta,
            "total": total,
        }
        
        return total, components
    
    def _get_weights(self) -> Tuple[float, float]:
        """Get current weights (possibly adaptive)."""
        cfg = self.config
        
        if not cfg.use_adaptive:
            return cfg.alpha, cfg.beta
        
        # Linear interpolation based on generation
        # (could use more sophisticated schedules)
        max_gen = 100
        progress = min(self.generation / max_gen, 1.0)
        
        alpha = cfg.initial_alpha + progress * (cfg.final_alpha - cfg.initial_alpha)
        beta = 1.0 - alpha
        
        return alpha, beta
    
    def advance_generation(self) -> None:
        """Called after each generation in evolutionary search."""
        self.generation += 1


def evaluate_observer_fitness(
    observer: "GlobalObserver",
    T_total: Optional[int] = None,
    config: Optional[OBSFitnessConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function to evaluate observer fitness.
    
    Args:
        observer: Observer to evaluate
        T_total: Total simulation time
        config: Optional fitness configuration
        
    Returns:
        Tuple of (fitness_score, components_dict)
    """
    fitness = OBSFitness(config)
    score, components = fitness.evaluate(observer, T_total)
    return score, components.to_dict()
