"""
World Simulator - RSL-based 1D universe with 4D projection.

Main entry point for simulations.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np

from world.config import WorldConfig, RSLParams, ObserverParams, StopParams
from world.core import Lattice, Rule, RuleSet, EvolutionEngine
from world.rsl import TensionCalculator, CapacityCalculator, F1Filter
from world.omega import CycleDetector
from world.observer import Observer, HilbertMapper
from world.analysis import (
    compute_winding_numbers,
    count_defects,
    compute_autocorrelation,
    compute_order_parameter,
)
from world.storage import ExperimentRecorder
from world.visualization import plot_evolution_summary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple config wrappers for CLI
@dataclass
class SimConfig:
    """Simple simulation config for CLI."""
    size: int = 100
    max_steps: int = 1000
    initial_p_plus: float = 0.5
    record_interval: int = 10
    metric_interval: int = 10


@dataclass
class RSLConfig:
    """Simple RSL config for CLI."""
    J: float = 1.0
    C0: float = 2.0
    alpha: float = 0.5
    H_threshold: float = 0.5


@dataclass
class ObserverConfig:
    """Simple observer config for CLI."""
    L_max: int = 3


def create_default_rules() -> RuleSet:
    """
    Create default rule set for RSL evolution.
    
    Returns:
        RuleSet with basic rules
    """
    rules = RuleSet()
    
    # Basic exchange rules (reversible pairs)
    rules.add(Rule(name="swap_pm", pattern=[1, -1], replacement=[-1, 1]))   # Swap +- to -+
    rules.add(Rule(name="swap_mp", pattern=[-1, 1], replacement=[1, -1]))   # Swap -+ to +-
    
    # Domain growth rules
    rules.add(Rule(name="grow_p0", pattern=[1, 0], replacement=[1, 1]))     # + absorbs 0
    rules.add(Rule(name="grow_0p", pattern=[0, 1], replacement=[1, 1]))     # + absorbs 0
    rules.add(Rule(name="grow_m0", pattern=[-1, 0], replacement=[-1, -1])) # - absorbs 0
    rules.add(Rule(name="grow_0m", pattern=[0, -1], replacement=[-1, -1])) # - absorbs 0
    
    # Domain wall annihilation
    rules.add(Rule(name="wall_decay_p", pattern=[1, -1, 1], replacement=[1, 0, 1]))   # wall can decay
    rules.add(Rule(name="wall_decay_m", pattern=[-1, 1, -1], replacement=[-1, 0, -1])) # wall can decay
    
    return rules


def run_simulation(
    config: SimConfig,
    rsl_config: RSLConfig,
    observer_config: ObserverConfig,
    rules: Optional[RuleSet] = None,
    output_dir: Optional[Path] = None,
    experiment_name: str = "simulation",
) -> dict:
    """
    Run a complete simulation.
    
    Args:
        config: Simulation configuration
        rsl_config: RSL parameters
        observer_config: Observer parameters
        rules: Rule set (uses default if None)
        output_dir: Directory for output files
        experiment_name: Name for this experiment
        
    Returns:
        Dictionary with results
    """
    logger.info(f"Starting simulation: {experiment_name}")
    logger.info(f"Size: {config.size}, Steps: {config.max_steps}")
    
    # Setup output
    if output_dir is None:
        output_dir = Path("./experiments")
    output_dir = Path(output_dir)
    
    # Initialize experiment recorder
    recorder = ExperimentRecorder(
        name=experiment_name,
        base_path=output_dir,
        config={
            'sim': config.__dict__,
            'rsl': rsl_config.__dict__,
            'observer': observer_config.__dict__,
        },
        description="RSL World simulation",
    )
    
    with recorder:
        # Create lattice
        lattice = Lattice(size=config.size)
        lattice.randomize(p_plus=config.initial_p_plus)
        
        logger.info(f"Initial magnetization: {lattice.magnetization():.4f}")
        
        # Create rules
        if rules is None:
            rules = create_default_rules()
        
        logger.info(f"Using {len(rules)} rules")
        
        # Create RSL components
        tension_calc = TensionCalculator(J=rsl_config.J)
        capacity_calc = CapacityCalculator(C0=rsl_config.C0, alpha=rsl_config.alpha)
        f1_filter = F1Filter(threshold=rsl_config.H_threshold)
        
        # Create evolution engine
        engine = EvolutionEngine(rules)
        
        # Initialize history storage
        recorder.init_history(config.size, config.max_steps)
        
        # Store history locally
        history = [lattice.to_state()]
        
        # Initial state
        recorder.record_state(lattice.sites, 0)
        recorder.record_metrics({
            'magnetization': lattice.magnetization(),
            'tension': tension_calc.global_tension(lattice.sites),
            'order_param': compute_order_parameter(lattice),
        }, step=0)
        
        # Save initial snapshot
        recorder.save_snapshot(lattice, "initial")
        
        # Track metrics for logging
        last_m = lattice.magnetization()
        rules_applied_total = 0
        
        # Evolution loop
        for step in range(1, config.max_steps + 1):
            # Apply rules
            applied = engine.step(lattice)
            rules_applied_total += len(applied)
            
            # Apply F1 filter (for now just check, don't reject)
            passes_filter = f1_filter.passes(lattice)
            
            # Record state (every N steps to save memory)
            if step % config.record_interval == 0:
                recorder.record_state(lattice.sites, step)
                history.append(lattice.to_state())
            
            # Record metrics
            if step % config.metric_interval == 0:
                h = tension_calc.global_tension(lattice.sites)
                m = lattice.magnetization()
                phi = compute_order_parameter(lattice)
                
                recorder.record_metrics({
                    'magnetization': m,
                    'tension': h,
                    'order_param': phi,
                    'f1_pass': 1 if passes_filter else 0,
                }, step=step)
            
            # Progress logging
            if step % max(1, config.max_steps // 10) == 0:
                m = lattice.magnetization()
                logger.info(f"Step {step}/{config.max_steps} - M={m:.4f}, rules applied: {rules_applied_total}")
        
        # Final state
        logger.info(f"Final magnetization: {lattice.magnetization():.4f}")
        logger.info(f"Total rules applied: {rules_applied_total}")
        recorder.save_snapshot(lattice, "final")
        
        # Analysis
        logger.info("Running analysis...")
        
        # Topological analysis
        winding = compute_winding_numbers(lattice)
        defects = count_defects(lattice)
        
        # Cycle detection (update with history)
        cycle_detector = CycleDetector()
        for i, state in enumerate(history):
            cycle_detector.update(state.sites, i)
        cycles = cycle_detector.detected_cycles
        
        # Observer projection
        observer = Observer(max_levels=observer_config.L_max)
        observation = observer.observe(lattice)
        
        # Save analysis results
        analysis_results = {
            'winding_numbers': winding,
            'defects': defects,
            'num_cycles': len(cycles),
            'final_magnetization': lattice.magnetization(),
            'final_tension': tension_calc.global_tension(lattice.sites),
        }
        recorder.save_analysis(analysis_results)
        
        logger.info(f"Experiment saved to: {recorder.path}")
        
        return {
            'lattice': lattice,
            'history': history,
            'analysis': analysis_results,
            'experiment_path': recorder.path,
        }


def main():
    """Command-line interface for running simulations."""
    parser = argparse.ArgumentParser(description="RSL World Simulator")
    
    parser.add_argument('--size', type=int, default=100,
                       help='Lattice size (default: 100)')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of evolution steps (default: 1000)')
    parser.add_argument('--p-plus', type=float, default=0.5,
                       help='Initial probability of +1 spin (default: 0.5)')
    parser.add_argument('--J', type=float, default=1.0,
                       help='Coupling constant (default: 1.0)')
    parser.add_argument('--C0', type=float, default=2.0,
                       help='Base capacity (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Capacity reduction factor (default: 0.5)')
    parser.add_argument('--H-threshold', type=float, default=0.5,
                       help='Tension threshold for F1 filter (default: 0.5)')
    parser.add_argument('--L-max', type=int, default=3,
                       help='Maximum coarse-graining level (default: 3)')
    parser.add_argument('--output', type=str, default='./experiments',
                       help='Output directory (default: ./experiments)')
    parser.add_argument('--name', type=str, default='simulation',
                       help='Experiment name (default: simulation)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: None)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization after simulation')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Create configs
    sim_config = SimConfig(
        size=args.size,
        max_steps=args.steps,
        initial_p_plus=args.p_plus,
    )
    
    rsl_config = RSLConfig(
        J=args.J,
        C0=args.C0,
        alpha=args.alpha,
        H_threshold=args.H_threshold,
    )
    
    observer_config = ObserverConfig(
        L_max=args.L_max,
    )
    
    # Run simulation
    results = run_simulation(
        config=sim_config,
        rsl_config=rsl_config,
        observer_config=observer_config,
        output_dir=Path(args.output),
        experiment_name=args.name,
    )
    
    # Visualization
    if args.visualize:
        logger.info("Generating visualization...")
        import matplotlib.pyplot as plt
        
        fig = plot_evolution_summary(results['history'])
        
        viz_path = results['experiment_path'] / "evolution_summary.png"
        fig.savefig(viz_path, dpi=150)
        plt.close(fig)
        
        logger.info(f"Visualization saved to: {viz_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
