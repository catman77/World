"""
Parameter space exploration for RSL World.

Systematic scanning of rule space and physical parameters
to find configurations that produce SM-like physics.
"""

from __future__ import annotations
import itertools
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

import numpy as np

from world.config import SimConfig, RSLConfig, ObserverConfig
from world.core import Lattice, Rule, RuleSet, EvolutionEngine
from world.rsl import TensionCalculator, F1Filter
from world.omega import CycleDetector, ChargeConservation
from world.analysis import (
    compute_winding_numbers,
    count_defects,
    compute_order_parameter,
    autocorrelation,
)
from world.storage import JSONStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Result of a single parameter exploration."""
    
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    score: float = 0.0
    is_physical: bool = False
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': self.parameters,
            'metrics': self.metrics,
            'score': self.score,
            'is_physical': self.is_physical,
            'notes': self.notes,
        }


def generate_rule_variations(base_length: int = 2) -> List[Rule]:
    """
    Generate all possible rules of given length.
    
    Args:
        base_length: Pattern length
        
    Returns:
        List of all possible rules
    """
    symbols = [-1, 0, 1]
    patterns = list(itertools.product(symbols, repeat=base_length))
    
    rules = []
    for pattern in patterns:
        for replacement in patterns:
            if pattern != replacement:  # Non-trivial rule
                rules.append(Rule(tuple(pattern), tuple(replacement)))
    
    return rules


def evaluate_ruleset(
    rules: RuleSet,
    config: SimConfig,
    rsl_config: RSLConfig,
    num_runs: int = 3,
) -> Dict[str, float]:
    """
    Evaluate a ruleset by running simulations.
    
    Args:
        rules: RuleSet to evaluate
        config: Simulation config
        rsl_config: RSL config
        num_runs: Number of independent runs
        
    Returns:
        Dictionary of averaged metrics
    """
    all_metrics = []
    
    for run in range(num_runs):
        # Create lattice
        lattice = Lattice.random(config.size, p_plus=config.initial_p_plus)
        lattice.phases[:] = np.random.uniform(0, 2*np.pi, config.size)
        
        # Create engine
        engine = EvolutionEngine(rules, record_history=True)
        tension_calc = TensionCalculator(J=rsl_config.J)
        
        # Run simulation
        engine.run(lattice, steps=config.max_steps)
        
        # Compute metrics
        history = engine.history
        
        # Extract time series
        magnetizations = [state.magnetization() for state in history]
        tensions = [tension_calc.micro_tension(state) for state in history]
        
        # Metrics
        metrics = {
            'final_magnetization': abs(lattice.magnetization()),
            'magnetization_std': np.std(magnetizations),
            'final_tension': tensions[-1],
            'tension_mean': np.mean(tensions),
            'tension_std': np.std(tensions),
        }
        
        # Topological metrics
        winding = compute_winding_numbers(lattice)
        defects = count_defects(lattice)
        
        metrics['winding_total'] = abs(winding.get('total', 0))
        metrics['num_defects'] = defects.get('domain_walls', 0)
        
        # Cycle detection
        detector = CycleDetector()
        cycles = detector.detect(lattice)
        metrics['num_cycles'] = len(cycles)
        
        # Correlation time
        if len(magnetizations) > 10:
            acf = autocorrelation(np.array(magnetizations), max_lag=min(100, len(magnetizations)//2))
            # Find correlation time (where ACF drops below 1/e)
            tau = 1
            for i, c in enumerate(acf):
                if c < np.exp(-1):
                    tau = i
                    break
            metrics['correlation_time'] = tau
        else:
            metrics['correlation_time'] = 1
        
        all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(values))
        avg_metrics[f'{key}_err'] = float(np.std(values) / np.sqrt(num_runs))
    
    return avg_metrics


def compute_physicality_score(metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Compute score indicating how "physical" the dynamics are.
    
    Physical systems should have:
    - Non-trivial dynamics (not frozen or chaotic)
    - Topological structure (winding, defects)
    - Conservation-like behavior
    - Emergent structures (cycles)
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Tuple of (score, notes)
    """
    score = 0.0
    notes = []
    
    # 1. Non-trivial magnetization (not all same)
    m_std = metrics.get('magnetization_std', 0)
    if 0.1 < m_std < 0.8:
        score += 1.0
        notes.append("Good magnetization dynamics")
    
    # 2. Moderate tension (not zero, not maximal)
    h_mean = metrics.get('tension_mean', 0)
    h_std = metrics.get('tension_std', 0)
    if h_mean > 0.1 and h_std > 0.05:
        score += 1.0
        notes.append("Non-trivial tension dynamics")
    
    # 3. Topological structure
    if metrics.get('num_defects', 0) > 0:
        score += 0.5
        notes.append("Has topological defects")
    
    if metrics.get('winding_total', 0) > 0:
        score += 0.5
        notes.append("Non-trivial winding number")
    
    # 4. Emergent cycles
    if metrics.get('num_cycles', 0) > 0:
        score += 1.0
        notes.append(f"Has {int(metrics['num_cycles'])} omega cycles")
    
    # 5. Correlation time (not too short, not too long)
    tau = metrics.get('correlation_time', 1)
    if 5 < tau < 100:
        score += 1.0
        notes.append(f"Good correlation time: {tau}")
    
    # Normalize score
    score = score / 5.0  # Max 5 points
    
    return score, notes


def explore_parameter_space(
    param_ranges: Dict[str, List],
    base_config: SimConfig,
    base_rsl_config: RSLConfig,
    rules: Optional[RuleSet] = None,
    output_dir: Optional[Path] = None,
    max_workers: int = 1,
) -> List[ExplorationResult]:
    """
    Explore parameter space systematically.
    
    Args:
        param_ranges: Dict mapping parameter name to list of values
        base_config: Base simulation config
        base_rsl_config: Base RSL config
        rules: RuleSet to use (or generate variations)
        output_dir: Directory for results
        max_workers: Number of parallel workers
        
    Returns:
        List of exploration results
    """
    if output_dir is None:
        output_dir = Path("./exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    logger.info(f"Exploring {len(combinations)} parameter combinations")
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        logger.info(f"[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Update configs
        config = SimConfig(
            size=params.get('size', base_config.size),
            max_steps=params.get('max_steps', base_config.max_steps),
            initial_p_plus=params.get('p_plus', base_config.initial_p_plus),
        )
        
        rsl_config = RSLConfig(
            J=params.get('J', base_rsl_config.J),
            C0=params.get('C0', base_rsl_config.C0),
            alpha=params.get('alpha', base_rsl_config.alpha),
            H_threshold=params.get('H_threshold', base_rsl_config.H_threshold),
        )
        
        # Use provided rules or create default
        if rules is None:
            from world.main import create_default_rules
            current_rules = create_default_rules()
        else:
            current_rules = rules
        
        try:
            # Evaluate
            metrics = evaluate_ruleset(current_rules, config, rsl_config)
            score, notes = compute_physicality_score(metrics)
            
            result = ExplorationResult(
                parameters=params,
                metrics=metrics,
                score=score,
                is_physical=(score > 0.5),
                notes=notes,
            )
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            result = ExplorationResult(
                parameters=params,
                metrics={},
                score=0.0,
                is_physical=False,
                notes=[f"Error: {str(e)}"],
            )
        
        results.append(result)
    
    # Save results
    storage = JSONStorage(output_dir)
    storage.save(
        [r.to_dict() for r in results],
        "exploration_results",
        compress=True,
    )
    
    # Report best results
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    logger.info("\nTop 5 configurations:")
    for i, r in enumerate(sorted_results[:5]):
        logger.info(f"  {i+1}. Score={r.score:.3f}, Params={r.parameters}")
        for note in r.notes:
            logger.info(f"     - {note}")
    
    return results


def explore_rule_space(
    rule_length: int = 2,
    num_rules_per_set: int = 5,
    num_combinations: int = 100,
    config: Optional[SimConfig] = None,
    rsl_config: Optional[RSLConfig] = None,
    output_dir: Optional[Path] = None,
) -> List[ExplorationResult]:
    """
    Explore space of possible rulesets.
    
    Args:
        rule_length: Length of rule patterns
        num_rules_per_set: Number of rules per set
        num_combinations: Number of random combinations to try
        config: Simulation config
        rsl_config: RSL config
        output_dir: Output directory
        
    Returns:
        List of exploration results
    """
    if config is None:
        config = SimConfig(size=50, max_steps=500)
    if rsl_config is None:
        rsl_config = RSLConfig()
    if output_dir is None:
        output_dir = Path("./rule_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all possible rules
    all_rules = generate_rule_variations(rule_length)
    logger.info(f"Generated {len(all_rules)} possible rules of length {rule_length}")
    
    results = []
    
    for i in range(num_combinations):
        # Sample random ruleset
        selected = np.random.choice(len(all_rules), size=num_rules_per_set, replace=False)
        ruleset = RuleSet([all_rules[j] for j in selected])
        
        # Evaluate
        params = {
            'rules': [(r.pattern, r.replacement) for r in ruleset.rules],
            'num_rules': len(ruleset),
        }
        
        logger.info(f"[{i+1}/{num_combinations}] Testing ruleset with {len(ruleset)} rules")
        
        try:
            metrics = evaluate_ruleset(ruleset, config, rsl_config, num_runs=2)
            score, notes = compute_physicality_score(metrics)
            
            result = ExplorationResult(
                parameters=params,
                metrics=metrics,
                score=score,
                is_physical=(score > 0.5),
                notes=notes,
            )
        except Exception as e:
            result = ExplorationResult(
                parameters=params,
                metrics={},
                score=0.0,
                is_physical=False,
                notes=[f"Error: {str(e)}"],
            )
        
        results.append(result)
    
    # Save and report
    storage = JSONStorage(output_dir)
    storage.save([r.to_dict() for r in results], "rule_exploration", compress=True)
    
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    logger.info("\nTop 5 rulesets:")
    for i, r in enumerate(sorted_results[:5]):
        logger.info(f"  {i+1}. Score={r.score:.3f}")
        logger.info(f"     Rules: {r.parameters['rules'][:3]}...")
        for note in r.notes[:3]:
            logger.info(f"     - {note}")
    
    return results


def main():
    """CLI for parameter exploration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RSL World Parameter Explorer")
    parser.add_argument('--mode', choices=['params', 'rules'], default='params',
                       help='Exploration mode')
    parser.add_argument('--output', type=str, default='./exploration',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'params':
        # Explore physical parameters
        param_ranges = {
            'J': [0.5, 1.0, 2.0],
            'C0': [1.0, 2.0, 3.0],
            'alpha': [0.25, 0.5, 1.0],
            'p_plus': [0.3, 0.5, 0.7],
        }
        
        explore_parameter_space(
            param_ranges=param_ranges,
            base_config=SimConfig(size=50, max_steps=500),
            base_rsl_config=RSLConfig(),
            output_dir=Path(args.output),
        )
    
    elif args.mode == 'rules':
        # Explore rule space
        explore_rule_space(
            rule_length=2,
            num_rules_per_set=5,
            num_combinations=50,
            output_dir=Path(args.output),
        )


if __name__ == "__main__":
    main()
