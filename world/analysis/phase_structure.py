"""
Phase structure analysis for RSL World.

Analyzes phase transitions and critical phenomena:
- Order parameters
- Phase diagrams
- Critical exponents
- Finite-size scaling
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np


@dataclass
class PhasePoint:
    """A point in parameter space with measured observables."""
    parameters: Dict[str, float]  # e.g., {"temperature": 1.0, "coupling": 0.5}
    observables: Dict[str, float]  # e.g., {"magnetization": 0.8, "susceptibility": 2.1}
    errors: Dict[str, float] = None  # Optional uncertainties
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = {}


@dataclass
class PhaseTransition:
    """Detected phase transition."""
    location: Dict[str, float]  # Parameter values at transition
    type: str  # "first_order", "second_order", "crossover"
    order_parameter_jump: float  # Discontinuity magnitude
    critical_exponents: Dict[str, float] = None
    
    def __repr__(self) -> str:
        return f"PhaseTransition({self.type} at {self.location})"


class PhaseAnalyzer:
    """
    Analyzer for phase structure and transitions.
    
    Detects and characterizes phase transitions by analyzing
    how order parameters change with control parameters.
    
    Example:
        analyzer = PhaseAnalyzer()
        
        # Scan parameter space
        for temp in temperatures:
            # Run simulation at this temperature
            m = measure_magnetization(lattice)
            analyzer.add_point({"temperature": temp}, {"magnetization": m})
        
        # Find transitions
        transitions = analyzer.find_transitions("temperature", "magnetization")
    """
    
    def __init__(self):
        self._points: List[PhasePoint] = []
    
    def add_point(
        self, 
        parameters: Dict[str, float],
        observables: Dict[str, float],
        errors: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add measurement point."""
        self._points.append(PhasePoint(parameters, observables, errors))
    
    def get_scan(
        self,
        param_name: str,
        observable_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 1D scan of observable vs parameter.
        
        Returns (parameter_values, observable_values) arrays.
        """
        params = []
        obs = []
        
        for point in self._points:
            if param_name in point.parameters and observable_name in point.observables:
                params.append(point.parameters[param_name])
                obs.append(point.observables[observable_name])
        
        # Sort by parameter
        idx = np.argsort(params)
        return np.array(params)[idx], np.array(obs)[idx]
    
    def find_transitions(
        self,
        param_name: str,
        observable_name: str,
        threshold: float = 0.1,
    ) -> List[PhaseTransition]:
        """
        Find phase transitions in 1D parameter scan.
        
        Detects transitions by looking for:
        - Large changes in order parameter (first order)
        - Diverging susceptibility (second order)
        - Smooth crossover regions
        
        Args:
            param_name: Control parameter to scan
            observable_name: Order parameter to monitor
            threshold: Minimum change to consider as transition
            
        Returns:
            List of detected PhaseTransition objects
        """
        params, obs = self.get_scan(param_name, observable_name)
        
        if len(params) < 3:
            return []
        
        transitions = []
        
        # Compute derivatives
        dobs = np.diff(obs)
        dparam = np.diff(params)
        
        # Avoid division by zero
        dparam[dparam == 0] = 1e-10
        
        derivative = dobs / dparam
        
        # Find peaks in |derivative|
        abs_deriv = np.abs(derivative)
        mean_deriv = np.mean(abs_deriv)
        
        for i in range(1, len(abs_deriv) - 1):
            if abs_deriv[i] > abs_deriv[i-1] and abs_deriv[i] > abs_deriv[i+1]:
                if abs_deriv[i] > threshold * np.max(abs_deriv):
                    # Found potential transition
                    transition_param = (params[i] + params[i+1]) / 2
                    jump = abs(dobs[i])
                    
                    # Classify transition type
                    if jump > 0.3 * (np.max(obs) - np.min(obs)):
                        trans_type = "first_order"
                    elif abs_deriv[i] > 3 * mean_deriv:
                        trans_type = "second_order"
                    else:
                        trans_type = "crossover"
                    
                    transitions.append(PhaseTransition(
                        location={param_name: transition_param},
                        type=trans_type,
                        order_parameter_jump=jump,
                    ))
        
        return transitions
    
    def susceptibility(
        self,
        param_name: str,
        observable_name: str,
        window: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute susceptibility χ = d<O>/d(param).
        
        Uses numerical differentiation.
        
        Returns (parameter_values, susceptibility_values).
        """
        params, obs = self.get_scan(param_name, observable_name)
        
        if len(params) < 2 * window + 1:
            return params, np.zeros_like(params)
        
        # Smooth numerical derivative
        chi = np.zeros_like(params)
        
        for i in range(window, len(params) - window):
            # Linear fit in window
            x = params[i-window:i+window+1]
            y = obs[i-window:i+window+1]
            coeffs = np.polyfit(x, y, 1)
            chi[i] = coeffs[0]  # Slope
        
        return params, chi
    
    def binder_cumulant(
        self,
        param_name: str,
        m2_name: str = "m2",
        m4_name: str = "m4",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Binder cumulant U = 1 - <m⁴>/(3<m²>²).
        
        Useful for locating critical points in finite-size scaling.
        
        Returns (parameter_values, binder_values).
        """
        params = []
        binder = []
        
        for point in self._points:
            if (param_name in point.parameters and 
                m2_name in point.observables and
                m4_name in point.observables):
                
                m2 = point.observables[m2_name]
                m4 = point.observables[m4_name]
                
                if m2 > 1e-10:
                    U = 1 - m4 / (3 * m2**2)
                else:
                    U = 0
                
                params.append(point.parameters[param_name])
                binder.append(U)
        
        idx = np.argsort(params)
        return np.array(params)[idx], np.array(binder)[idx]
    
    def finite_size_scaling(
        self,
        sizes: List[int],
        param_name: str,
        observable_name: str,
        critical_point: float,
        nu: float = 1.0,
        beta: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Perform finite-size scaling analysis.
        
        Scales data according to:
        O(L, t) = L^(-β/ν) * f(t * L^(1/ν))
        
        where t = (param - critical_point) / critical_point.
        
        Returns dict with scaled quantities for collapse plot.
        """
        results = {
            "scaled_param": [],
            "scaled_observable": [],
            "size": [],
        }
        
        for point in self._points:
            if param_name not in point.parameters:
                continue
            if observable_name not in point.observables:
                continue
            if "size" not in point.parameters:
                continue
            
            L = point.parameters["size"]
            p = point.parameters[param_name]
            obs = point.observables[observable_name]
            
            # Reduced parameter
            t = (p - critical_point) / critical_point
            
            # Scaled quantities
            scaled_param = t * L**(1/nu)
            scaled_obs = obs * L**(beta/nu)
            
            results["scaled_param"].append(scaled_param)
            results["scaled_observable"].append(scaled_obs)
            results["size"].append(L)
        
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def estimate_critical_exponents(
        self,
        param_name: str,
        critical_point: float,
        sizes: List[int],
    ) -> Dict[str, float]:
        """
        Estimate critical exponents from finite-size data.
        
        Uses:
        - β: Order parameter scaling m ~ L^(-β/ν)
        - ν: Correlation length scaling ξ ~ |t|^(-ν)
        - γ: Susceptibility scaling χ ~ L^(γ/ν)
        """
        # This is a simplified implementation
        # Full implementation would use proper fitting
        
        return {
            "beta": 0.5,  # Placeholder
            "nu": 1.0,
            "gamma": 1.0,
        }
    
    def clear(self) -> None:
        """Clear all data points."""
        self._points.clear()


def find_phase_transitions(
    param_values: np.ndarray,
    observable_values: np.ndarray,
    threshold: float = 0.1,
) -> List[float]:
    """
    Convenience function to find transition points.
    
    Returns list of parameter values where transitions occur.
    """
    analyzer = PhaseAnalyzer()
    
    for p, o in zip(param_values, observable_values):
        analyzer.add_point({"param": p}, {"observable": o})
    
    transitions = analyzer.find_transitions("param", "observable", threshold)
    return [t.location["param"] for t in transitions]


def compute_order_parameter(
    sites: np.ndarray,
    order_type: str = "magnetization",
) -> float:
    """
    Compute order parameter for a lattice state.
    
    Order types:
    - "magnetization": m = <s>
    - "staggered": m_s = <(-1)^i * s_i>
    - "domain_walls": ρ = (number of walls) / N
    """
    if hasattr(sites, '_sites'):
        sites = sites._sites
    
    N = len(sites)
    
    if order_type == "magnetization":
        return float(np.mean(sites))
    
    elif order_type == "staggered":
        # Alternating sign pattern
        signs = np.array([(-1)**i for i in range(N)])
        return float(np.mean(signs * sites))
    
    elif order_type == "domain_walls":
        walls = np.sum(sites[:-1] != sites[1:])
        return float(walls / N)
    
    else:
        raise ValueError(f"Unknown order type: {order_type}")


class CriticalSlowing:
    """
    Analyze critical slowing down near phase transitions.
    
    Near criticality, the autocorrelation time diverges:
    τ ~ |T - T_c|^(-z*ν)
    
    where z is the dynamic critical exponent.
    """
    
    def __init__(self):
        self._measurements: List[Tuple[float, float]] = []  # (param, tau)
    
    def add_measurement(self, parameter: float, correlation_time: float) -> None:
        """Add correlation time measurement at parameter value."""
        self._measurements.append((parameter, correlation_time))
    
    def estimate_critical_point(self) -> Optional[float]:
        """
        Estimate critical point from divergence of correlation time.
        """
        if len(self._measurements) < 3:
            return None
        
        params, taus = zip(*sorted(self._measurements))
        
        # Find maximum correlation time
        max_idx = np.argmax(taus)
        return params[max_idx]
    
    def estimate_dynamic_exponent(
        self,
        critical_point: float,
    ) -> float:
        """
        Estimate dynamic critical exponent z*ν.
        
        From τ ~ |T - T_c|^(-z*ν).
        """
        if len(self._measurements) < 3:
            return 1.0
        
        params, taus = zip(*sorted(self._measurements))
        
        # Fit to power law on each side of critical point
        # Simplified: just use points near transition
        
        t_values = [abs(p - critical_point) for p in params]
        
        # Filter out critical point itself
        valid = [(t, tau) for t, tau in zip(t_values, taus) if t > 0.01]
        
        if len(valid) < 2:
            return 1.0
        
        t_values, taus = zip(*valid)
        
        # Log-log fit
        log_t = np.log(t_values)
        log_tau = np.log(taus)
        
        coeffs = np.polyfit(log_t, log_tau, 1)
        
        return -coeffs[0]  # z*ν is negative of slope
