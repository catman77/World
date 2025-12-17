"""
Stone Mechanism: Probability Control via Evolution Strategy
============================================================

Implements the "Philosophical Stone" mechanism for finding optimal
configurations that guarantee target outcomes.

Core Algorithm (from Algorithm_Final.md):
-----------------------------------------
1. Define target predicate: hit(world_state) -> {0, 1}
2. Use Evolution Strategy to find theta* such that P(hit|theta*) ~ 1
3. Train policy pi: input -> theta* (constant output)
4. Result: deterministic 100% hit rate

Key Insight:
------------
The World is DETERMINISTIC. We don't "control randomness",
we SEARCH for parameters where the target is guaranteed.

Improvement Ratio:
-----------------
- P0 (random theta): typically << 1% 
- P_Phi (with Stone): ~ 100%
- Ratio: can exceed 1000x or more
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from .world import World


class TargetType(Enum):
    """Types of targets for Stone mechanism."""
    FTL = "ftl"
    ANTIGRAV = "antigravity"
    RESONANCE = "resonance"
    GRAVITY_LAW = "gravity_law"
    CUSTOM = "custom"


@dataclass
class TargetSpec:
    """Target specification for Stone mechanism."""
    target_type: TargetType
    predicate: Callable[[Dict[str, Any]], bool]
    reward_fn: Optional[Callable[[Dict[str, Any]], float]] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    name: str = "unnamed_target"
    
    def evaluate(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        hit = self.predicate(world_state)
        reward = self.reward_fn(world_state) if self.reward_fn else (1.0 if hit else 0.0)
        return {'hit': hit, 'reward': reward, 'target_type': self.target_type.value}


@dataclass
class StoneConfig:
    """Configuration for Stone mechanism."""
    population_size: int = 20
    num_generations: int = 50
    initial_noise: float = 0.1
    noise_decay: float = 0.95
    elite_ratio: float = 0.2
    samples_per_eval: int = 30
    baseline_samples: int = 1000
    final_eval_samples: int = 500
    seed: Optional[int] = None


class ParameterSpace:
    """Parameter space theta for optimization."""
    
    def __init__(self, dim: int, bounds: Optional[List[Tuple[float, float]]] = None, names: Optional[List[str]] = None):
        self.dim = dim
        self.bounds = bounds if bounds else [(0.0, 1.0)] * dim
        self.names = names if names else [f"theta_{i}" for i in range(dim)]
    
    def sample_random(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.array([rng.uniform(lo, hi) for lo, hi in self.bounds])
    
    def clip(self, theta: np.ndarray) -> np.ndarray:
        return np.array([np.clip(theta[i], lo, hi) for i, (lo, hi) in enumerate(self.bounds)])
    
    def sample_around(self, center: np.ndarray, noise_std: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return self.clip(center + rng.standard_normal(self.dim) * noise_std)


class WorldEvaluator:
    """Evaluates theta by running World simulations."""
    
    def __init__(self, world_factory: Callable[[np.ndarray], Any], target_spec: TargetSpec, steps_per_eval: int = 50):
        self.world_factory = world_factory
        self.target_spec = target_spec
        self.steps_per_eval = steps_per_eval
    
    def evaluate(self, theta: np.ndarray, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        world = self.world_factory(theta)
        for _ in range(self.steps_per_eval):
            world.step()
        world_state = self._extract_state(world)
        result = self.target_spec.evaluate(world_state)
        result['theta'] = theta.copy()
        return result
    
    def _extract_state(self, world) -> Dict[str, Any]:
        state = {
            'spins': world.spins.copy(),
            'total_spins': int(world.spins.sum()),
            'density': float(world.spins.mean()),
            'step': getattr(world, 'step_count', 0),
        }
        if hasattr(world, 'observable') and world.observable and hasattr(world.observable, 'history'):
            state['observable_final'] = world.observable.history[-1] if world.observable.history else None
        return state
    
    def evaluate_batch(self, theta: np.ndarray, num_samples: int, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
        hits, total_reward = 0, 0.0
        for _ in range(num_samples):
            result = self.evaluate(theta, rng)
            hits += int(result['hit'])
            total_reward += result['reward']
        return hits / num_samples, total_reward / num_samples


class EvolutionStrategy:
    """ES optimizer for finding theta*."""
    
    def __init__(self, param_space: ParameterSpace, evaluator: WorldEvaluator, config: StoneConfig):
        self.param_space = param_space
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.center = None
        self.best_theta = None
        self.best_fitness = -np.inf
        self.generation = 0
        self.history = []
    
    def initialize(self, initial_theta: Optional[np.ndarray] = None):
        self.center = self.param_space.clip(initial_theta) if initial_theta is not None else self.param_space.sample_random(self.rng)
        self.best_theta = self.center.copy()
        self.best_fitness = -np.inf
        self.generation = 0
        self.history = []
    
    def step(self, verbose: bool = False) -> Dict[str, Any]:
        pop_size = self.config.population_size
        noise_std = self.config.initial_noise * (self.config.noise_decay ** self.generation)
        
        population = [self.param_space.sample_around(self.center, noise_std, self.rng) for _ in range(pop_size)]
        fitness_values = []
        for theta in population:
            hit_rate, mean_reward = self.evaluator.evaluate_batch(theta, self.config.samples_per_eval, self.rng)
            fitness_values.append(hit_rate + 0.1 * mean_reward)
        
        fitness_values = np.array(fitness_values)
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_theta = population[best_idx].copy()
        
        weights = np.exp((fitness_values - fitness_values.mean()) * 5.0)
        weights /= weights.sum()
        self.center = self.param_space.clip(sum(w * p for w, p in zip(weights, population)))
        
        stats = {'generation': self.generation, 'best_fitness': self.best_fitness, 'mean_fitness': fitness_values.mean()}
        self.history.append(stats)
        if verbose:
            print(f"  Gen {self.generation}: best={self.best_fitness:.3f}, mean={stats['mean_fitness']:.3f}")
        self.generation += 1
        return stats
    
    def run(self, verbose: bool = True) -> np.ndarray:
        if self.center is None:
            self.initialize()
        if verbose:
            print(f"Starting ES ({self.config.num_generations} generations)")
        for _ in range(self.config.num_generations):
            self.step(verbose=verbose)
            if self.best_fitness >= 1.0:
                if verbose:
                    print("  Early stop: perfect fitness!")
                break
        return self.best_theta


class StoneMechanism:
    """Main Stone mechanism: find theta* such that P(hit|theta*) ~ 1."""
    
    def __init__(self, target_spec: TargetSpec, config: Optional[StoneConfig] = None):
        self.target_spec = target_spec
        self.config = config or StoneConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.param_space = None
        self.evaluator = None
        self.es = None
        self.theta_star = None
        self.p0 = None
        self.p_phi = None
        self.improvement = None
    
    def setup(self, world_factory: Callable[[np.ndarray], Any], param_space: ParameterSpace, steps_per_eval: int = 50):
        self.param_space = param_space
        self.evaluator = WorldEvaluator(world_factory, self.target_spec, steps_per_eval)
        self.es = EvolutionStrategy(param_space, self.evaluator, self.config)
    
    def estimate_baseline(self, verbose: bool = True) -> float:
        if verbose:
            print(f"Estimating P0 ({self.config.baseline_samples} samples)...")
        hits = sum(1 for _ in range(self.config.baseline_samples) 
                   if self.evaluator.evaluate(self.param_space.sample_random(self.rng), self.rng)['hit'])
        self.p0 = hits / self.config.baseline_samples
        if verbose:
            print(f"  P0 = {hits}/{self.config.baseline_samples} = {self.p0:.4%}")
        return self.p0
    
    def find_optimal(self, initial_theta: Optional[np.ndarray] = None, verbose: bool = True) -> np.ndarray:
        if verbose:
            print("\n" + "="*50 + "\nFINDING theta* VIA ES\n" + "="*50)
        self.es.initialize(initial_theta)
        self.theta_star = self.es.run(verbose=verbose)
        hit_rate, _ = self.evaluator.evaluate_batch(self.theta_star, self.config.final_eval_samples, self.rng)
        if verbose:
            print(f"\nFound theta* with hit rate: {hit_rate:.2%}")
        return self.theta_star
    
    def evaluate_improvement(self, verbose: bool = True) -> Dict[str, float]:
        if self.p0 is None:
            self.estimate_baseline(verbose)
        hit_rate, _ = self.evaluator.evaluate_batch(self.theta_star, self.config.final_eval_samples, self.rng)
        self.p_phi = hit_rate
        self.improvement = self.p_phi / self.p0 if self.p0 > 0 else self.p_phi / (3.0 / self.config.baseline_samples)
        if verbose:
            print(f"\nP_Phi = {self.p_phi:.2%}, Improvement = {self.improvement:.1f}x")
        return {'p0': self.p0, 'p_phi': self.p_phi, 'improvement': self.improvement, 'theta_star': self.theta_star.tolist()}
    
    def run_full(self, world_factory, param_space, steps_per_eval=50, verbose=True) -> Dict[str, Any]:
        self.setup(world_factory, param_space, steps_per_eval)
        self.estimate_baseline(verbose)
        self.find_optimal(verbose=verbose)
        return self.evaluate_improvement(verbose)


def make_density_target(density_min: float = 0.4, density_max: float = 0.6) -> TargetSpec:
    """Target: spin density in range."""
    def predicate(state):
        d = state.get('density', 0)
        return density_min <= d <= density_max
    def reward_fn(state):
        d = state.get('density', 0)
        center = (density_min + density_max) / 2
        return max(0, 1 - abs(d - center) / ((density_max - density_min) / 2))
    return TargetSpec(TargetType.CUSTOM, predicate, reward_fn, name=f"Density_{density_min}_{density_max}")


def make_custom_target(name: str, predicate: Callable, reward_fn: Optional[Callable] = None) -> TargetSpec:
    """Create custom target."""
    return TargetSpec(TargetType.CUSTOM, predicate, reward_fn, name=name)


def create_stone_for_world(world_class, target_spec: TargetSpec, N: int = 128, param_dim: int = 10, config: Optional[StoneConfig] = None) -> StoneMechanism:
    """Convenience: create Stone for World class."""
    def world_factory(theta):
        world = world_class(N=N)
        if len(theta) >= 1:
            world.spins = (np.random.default_rng(int(theta[0] * 1000)).random((N, N)) < theta[0]).astype(np.int8)
        return world
    stone = StoneMechanism(target_spec, config)
    stone.setup(world_factory, ParameterSpace(dim=param_dim))
    return stone
