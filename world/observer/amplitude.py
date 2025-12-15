"""
Amplitude calculation for RSL quantum-like behavior.

Implements the amplitude formula:
    ψ_{Y→Y_j} = (1/√N_j) * Σ_k exp(iθ_k)

where:
- N_j = number of microscopic states in [Y_j]
- θ_k = phase accumulated by state k

The phases encode the microscopic dynamics invisible to observer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from ..core.lattice import LatticeState


@dataclass
class Phase:
    """
    Phase information for a microscopic state.
    
    Tracks accumulated phase θ from evolution history.
    """
    state_hash: int
    value: float  # θ in [0, 2π)
    time: int = 0
    
    def advance(self, delta: float) -> None:
        """Advance phase by delta, keeping in [0, 2π)."""
        self.value = (self.value + delta) % (2 * np.pi)


@dataclass
class Amplitude:
    """
    Complex amplitude result.
    
    ψ = |ψ| * exp(iφ) = re + i*im
    """
    real: float
    imag: float
    
    @property
    def magnitude(self) -> float:
        """Amplitude magnitude |ψ|."""
        return np.sqrt(self.real**2 + self.imag**2)
    
    @property
    def phase(self) -> float:
        """Amplitude phase arg(ψ)."""
        return np.arctan2(self.imag, self.real)
    
    @property
    def probability(self) -> float:
        """Probability |ψ|²."""
        return self.real**2 + self.imag**2
    
    def __add__(self, other: "Amplitude") -> "Amplitude":
        return Amplitude(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, scalar: float) -> "Amplitude":
        return Amplitude(self.real * scalar, self.imag * scalar)
    
    def conjugate(self) -> "Amplitude":
        return Amplitude(self.real, -self.imag)
    
    @classmethod
    def from_polar(cls, magnitude: float, phase: float) -> "Amplitude":
        """Create amplitude from polar form."""
        return cls(
            real=magnitude * np.cos(phase),
            imag=magnitude * np.sin(phase),
        )
    
    def __repr__(self) -> str:
        return f"({self.real:.4f} + {self.imag:.4f}i)"


class AmplitudeCalculator:
    """
    Calculator for RSL transition amplitudes.
    
    Computes:
        ψ_{Y→Y_j} = (1/√N_j) * Σ_k exp(iθ_k)
    
    The phases θ_k are accumulated during evolution and encode
    the microscopic history invisible to the observer.
    
    Parameters:
        theta_max: Maximum phase increment per step
        use_rule_phases: Derive phases from rule applications
    
    Example:
        calc = AmplitudeCalculator(theta_max=0.1)
        
        # Track phases during evolution
        for step in evolution:
            calc.update_phases(state, rules_applied)
        
        # Compute transition amplitude
        amp = calc.transition_amplitude(initial_obs, final_obs, states)
    """
    
    def __init__(
        self,
        theta_max: float = 0.15,
        use_rule_phases: bool = True,
    ):
        self.theta_max = theta_max
        self.use_rule_phases = use_rule_phases
        
        # Phase tracking
        self._phases: Dict[int, Phase] = {}  # state_hash -> Phase
        
        # Rule-specific phase assignments
        self._rule_phases: Dict[str, float] = {}
    
    def assign_rule_phase(self, rule_name: str, phase: float) -> None:
        """Assign a specific phase to a rule."""
        self._rule_phases[rule_name] = phase
    
    def get_phase(self, state: LatticeState) -> float:
        """Get current phase for a state."""
        state_hash = hash(state)
        if state_hash in self._phases:
            return self._phases[state_hash].value
        return 0.0
    
    def update_phases(
        self,
        state: LatticeState,
        rules_applied: Optional[List] = None,
        tension: Optional[float] = None,
    ) -> None:
        """
        Update phase for state based on evolution.
        
        Phase increment methods:
        1. From applied rules (if use_rule_phases)
        2. From local tension
        3. Random within theta_max
        """
        state_hash = hash(state)
        
        # Create phase entry if new
        if state_hash not in self._phases:
            self._phases[state_hash] = Phase(
                state_hash=state_hash,
                value=0.0,
                time=state.time,
            )
        
        phase = self._phases[state_hash]
        
        # Compute phase increment
        delta = 0.0
        
        if self.use_rule_phases and rules_applied:
            # Sum phases from applied rules
            for match in rules_applied:
                rule_name = match.rule.name if hasattr(match, 'rule') else str(match)
                if rule_name in self._rule_phases:
                    delta += self._rule_phases[rule_name]
                else:
                    # Default: small random phase
                    delta += np.random.uniform(0, self.theta_max)
        elif tension is not None:
            # Phase proportional to tension
            delta = self.theta_max * tension
        else:
            # Default: small random phase
            delta = np.random.uniform(0, self.theta_max)
        
        phase.advance(delta)
        phase.time = state.time
    
    def transition_amplitude(
        self,
        initial_class: List[LatticeState],
        final_class: List[LatticeState],
    ) -> Amplitude:
        """
        Compute transition amplitude between observation classes.
        
        ψ_{Y→Y'} = (1/√N') * Σ_{k ∈ [Y']} exp(iθ_k)
        
        Args:
            initial_class: States in initial observation equivalence class
            final_class: States in final observation equivalence class
            
        Returns:
            Complex amplitude
        """
        if not final_class:
            return Amplitude(0.0, 0.0)
        
        N = len(final_class)
        norm = 1.0 / np.sqrt(N)
        
        # Sum over phases in final class
        total_real = 0.0
        total_imag = 0.0
        
        for state in final_class:
            theta = self.get_phase(state)
            total_real += np.cos(theta)
            total_imag += np.sin(theta)
        
        return Amplitude(
            real=norm * total_real,
            imag=norm * total_imag,
        )
    
    def interference(
        self,
        paths: List[List[LatticeState]],
    ) -> Amplitude:
        """
        Compute interference between multiple paths.
        
        Total amplitude is sum over all paths.
        """
        total = Amplitude(0.0, 0.0)
        
        for path in paths:
            if path:
                # Amplitude for this path
                theta_total = sum(self.get_phase(s) for s in path)
                path_amp = Amplitude.from_polar(1.0 / len(paths), theta_total)
                total = total + path_amp
        
        return total
    
    def decoherence_factor(
        self,
        states: List[LatticeState],
    ) -> float:
        """
        Compute decoherence factor.
        
        D = |Σ exp(iθ_k)|² / N
        
        D = 1: Full coherence (all phases aligned)
        D → 0: Decoherence (phases random)
        """
        if not states:
            return 0.0
        
        N = len(states)
        total_real = sum(np.cos(self.get_phase(s)) for s in states)
        total_imag = sum(np.sin(self.get_phase(s)) for s in states)
        
        return (total_real**2 + total_imag**2) / (N * N)
    
    def phase_distribution(
        self,
        states: List[LatticeState],
        bins: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get distribution of phases.
        
        Returns (counts, bin_edges) for histogram.
        """
        phases = [self.get_phase(s) for s in states]
        return np.histogram(phases, bins=bins, range=(0, 2*np.pi))
    
    def reset(self) -> None:
        """Reset all phase tracking."""
        self._phases.clear()


def compute_amplitude(
    phases: List[float],
) -> Amplitude:
    """
    Compute amplitude from list of phases.
    
    ψ = (1/√N) * Σ exp(iθ_k)
    """
    if not phases:
        return Amplitude(0.0, 0.0)
    
    N = len(phases)
    norm = 1.0 / np.sqrt(N)
    
    real = norm * sum(np.cos(theta) for theta in phases)
    imag = norm * sum(np.sin(theta) for theta in phases)
    
    return Amplitude(real, imag)


def compute_born_probability(amplitude: Amplitude) -> float:
    """Compute Born probability |ψ|²."""
    return amplitude.probability


def interference_pattern(
    phases1: List[float],
    phases2: List[float],
) -> float:
    """
    Compute interference term between two sets of phases.
    
    Returns: 2 * Re(ψ₁* ψ₂)
    """
    amp1 = compute_amplitude(phases1)
    amp2 = compute_amplitude(phases2)
    
    # ψ₁* ψ₂
    real_part = amp1.real * amp2.real + amp1.imag * amp2.imag
    
    return 2 * real_part
