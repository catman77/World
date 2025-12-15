"""
Observer interface (IFACE) for RSL World.

Provides the measurements available to the observer:
- Field measurements (continuous)
- Particle detection
- Energy/momentum
- Correlation functions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
import numpy as np

from .observer import Observer, ObserverState
from .coordinate_mapper import CoordinateMapper, Coordinates3D


@dataclass
class Measurement:
    """
    A single measurement result.
    
    Attributes:
        name: Name of measured quantity
        value: Measured value
        uncertainty: Measurement uncertainty (if applicable)
        time: Time of measurement
        position: Position of measurement (if localized)
    """
    name: str
    value: Any
    uncertainty: Optional[float] = None
    time: int = 0
    position: Optional[Coordinates3D] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if self.uncertainty is not None:
            return f"{self.name} = {self.value} ± {self.uncertainty}"
        return f"{self.name} = {self.value}"


class Interface:
    """
    Observer interface (IFACE) for measurements.
    
    Provides:
    - Field measurements: Local and global field values
    - Particle measurements: Detection, counting
    - Statistical measurements: Correlations, distributions
    - Energy measurements: Total energy, local energy density
    
    The interface implements the principle that observers can only
    access coarse-grained information, not microscopic details.
    
    Example:
        iface = Interface(observer, mapper)
        
        # Measure field at position
        field = iface.measure_field(lattice, x=0.5, y=0.5, z=0.5)
        
        # Measure correlation function
        corr = iface.measure_correlation(lattice, r_max=10)
    """
    
    def __init__(
        self,
        observer: Observer,
        mapper: Optional[CoordinateMapper] = None,
    ):
        self.observer = observer
        self.mapper = mapper
        self._measurements: List[Measurement] = []
    
    def measure_field(
        self,
        lattice,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        index: Optional[int] = None,
    ) -> Measurement:
        """
        Measure field value at specified position.
        
        Can specify either:
        - 3D coordinates (x, y, z) if mapper available
        - 1D index directly
        """
        # Observe lattice
        obs_state = self.observer.observe(lattice)
        
        if index is not None:
            # Direct 1D measurement
            if index < len(obs_state.coarse_field):
                value = obs_state.coarse_field[index]
            else:
                value = None
            position = self.mapper.map_index(index) if self.mapper else None
        elif x is not None and self.mapper:
            # 3D position measurement
            position = Coordinates3D(x, y or 0, z or 0)
            index = self.mapper.inverse_map(position)
            # Map to coarse field index
            stride = self.observer.stride
            coarse_idx = index // stride
            if coarse_idx < len(obs_state.coarse_field):
                value = obs_state.coarse_field[coarse_idx]
            else:
                value = None
        else:
            # Global average
            value = obs_state.mean_field
            position = None
        
        measurement = Measurement(
            name="field",
            value=value,
            uncertainty=np.sqrt(obs_state.field_variance) if obs_state.field_variance else None,
            time=lattice.time,
            position=position,
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_magnetization(self, lattice) -> Measurement:
        """Measure total magnetization."""
        obs_state = self.observer.observe(lattice)
        
        measurement = Measurement(
            name="magnetization",
            value=obs_state.mean_field,
            uncertainty=np.sqrt(obs_state.field_variance / len(obs_state.coarse_field)),
            time=lattice.time,
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_energy(self, lattice, tension_calc=None) -> Measurement:
        """
        Measure total energy (tension).
        
        Uses H_micro(S) = J * M (domain wall count).
        """
        if tension_calc is None:
            # Simple domain wall counting
            M = lattice.domain_wall_count()
            energy = float(M)
        else:
            energy = tension_calc.global_tension(lattice._sites)
        
        measurement = Measurement(
            name="energy",
            value=energy,
            time=lattice.time,
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_correlation(
        self,
        lattice,
        r_max: int = 10,
    ) -> Measurement:
        """
        Measure two-point correlation function.
        
        C(r) = <s_i * s_{i+r}> - <s_i>^2
        
        Returns array of correlations for r = 0, 1, ..., r_max.
        """
        sites = lattice._sites
        N = len(sites)
        
        mean = np.mean(sites)
        correlations = np.zeros(r_max + 1)
        
        for r in range(r_max + 1):
            # Compute correlation at distance r
            if r == 0:
                correlations[r] = 1.0  # Self-correlation
            else:
                corr_sum = 0.0
                for i in range(N):
                    j = (i + r) % N
                    corr_sum += sites[i] * sites[j]
                correlations[r] = corr_sum / N - mean**2
        
        measurement = Measurement(
            name="correlation",
            value=correlations,
            time=lattice.time,
            metadata={"r_max": r_max},
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_structure_factor(
        self,
        lattice,
        k_max: int = 20,
    ) -> Measurement:
        """
        Measure structure factor S(k).
        
        S(k) = |ρ(k)|² where ρ(k) = Σ s_j * exp(-ikj)
        
        The structure factor reveals periodicity in the system.
        """
        sites = lattice._sites.astype(float)
        N = len(sites)
        
        k_values = np.arange(k_max + 1) * 2 * np.pi / N
        S_k = np.zeros(k_max + 1)
        
        for idx, k in enumerate(k_values):
            # Compute Fourier component
            rho_k = np.sum(sites * np.exp(-1j * k * np.arange(N)))
            S_k[idx] = np.abs(rho_k)**2 / N
        
        measurement = Measurement(
            name="structure_factor",
            value=S_k,
            time=lattice.time,
            metadata={"k_values": k_values.tolist(), "k_max": k_max},
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_domain_statistics(self, lattice) -> Measurement:
        """
        Measure domain statistics.
        
        Returns:
        - Number of domains
        - Mean domain size
        - Domain size distribution
        """
        sites = lattice._sites
        N = len(sites)
        
        # Find domain boundaries
        boundaries = [0]
        for i in range(N - 1):
            if sites[i] != sites[i + 1]:
                boundaries.append(i + 1)
        
        # Compute domain sizes
        if len(boundaries) == 1:
            # Single domain
            domain_sizes = [N]
        else:
            domain_sizes = []
            for i in range(len(boundaries)):
                if i == len(boundaries) - 1:
                    size = N - boundaries[i]
                else:
                    size = boundaries[i + 1] - boundaries[i]
                domain_sizes.append(size)
        
        measurement = Measurement(
            name="domain_statistics",
            value={
                "num_domains": len(domain_sizes),
                "mean_size": np.mean(domain_sizes),
                "std_size": np.std(domain_sizes),
                "sizes": domain_sizes,
            },
            time=lattice.time,
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_entropy(self, lattice) -> Measurement:
        """
        Measure entropy of site distribution.
        
        S = -Σ p_i * log(p_i)
        
        For binary sites, this is based on fraction of + and -.
        """
        sites = lattice._sites
        N = len(sites)
        
        # Count +1 and -1
        n_plus = np.sum(sites == 1)
        n_minus = N - n_plus
        
        # Probabilities
        p_plus = n_plus / N
        p_minus = n_minus / N
        
        # Entropy
        entropy = 0.0
        if p_plus > 0:
            entropy -= p_plus * np.log(p_plus)
        if p_minus > 0:
            entropy -= p_minus * np.log(p_minus)
        
        measurement = Measurement(
            name="entropy",
            value=entropy,
            time=lattice.time,
            metadata={"p_plus": p_plus, "p_minus": p_minus},
        )
        
        self._measurements.append(measurement)
        return measurement
    
    def measure_all(self, lattice) -> Dict[str, Measurement]:
        """Perform all standard measurements."""
        return {
            "magnetization": self.measure_magnetization(lattice),
            "energy": self.measure_energy(lattice),
            "entropy": self.measure_entropy(lattice),
            "domain_statistics": self.measure_domain_statistics(lattice),
        }
    
    @property
    def measurements(self) -> List[Measurement]:
        """Return all recorded measurements."""
        return list(self._measurements)
    
    def clear_measurements(self) -> None:
        """Clear measurement history."""
        self._measurements.clear()
    
    def get_time_series(self, name: str) -> List[Tuple[int, Any]]:
        """Get time series of measurements with given name."""
        return [
            (m.time, m.value) 
            for m in self._measurements 
            if m.name == name
        ]
