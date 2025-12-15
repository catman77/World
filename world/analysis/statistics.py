"""
Statistical analysis for RSL World.

Provides statistical measures and distributions:
- Autocorrelation functions
- Power spectra
- Distributions and moments
- Fluctuation analysis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class StatisticalSummary:
    """Summary statistics for a quantity."""
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    median: float
    
    @classmethod
    def from_data(cls, data: np.ndarray) -> "StatisticalSummary":
        """Compute all statistics from data array."""
        data = np.asarray(data)
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        
        # Higher moments
        if std > 0:
            centered = data - mean
            skewness = np.mean(centered**3) / std**3
            kurtosis = np.mean(centered**4) / std**4 - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        return cls(
            mean=float(mean),
            std=float(std),
            variance=float(variance),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            min=float(np.min(data)),
            max=float(np.max(data)),
            median=float(np.median(data)),
        )


class StatisticsCalculator:
    """
    Calculator for statistical properties of lattice evolution.
    
    Provides:
    - Time series analysis (autocorrelation, power spectrum)
    - Spatial statistics (correlation functions)
    - Distribution analysis
    - Fluctuation scaling
    
    Example:
        calc = StatisticsCalculator()
        
        # Compute autocorrelation
        ac = calc.autocorrelation(magnetization_series)
        
        # Compute power spectrum
        ps = calc.power_spectrum(magnetization_series)
    """
    
    def autocorrelation(
        self, 
        data: np.ndarray, 
        max_lag: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute autocorrelation function.
        
        C(τ) = <x(t) * x(t+τ)> - <x>²
        
        Args:
            data: Time series data
            max_lag: Maximum lag (default: len(data)//4)
            
        Returns:
            Array of autocorrelation values C(0), C(1), ...
        """
        data = np.asarray(data)
        N = len(data)
        
        if max_lag is None:
            max_lag = N // 4
        
        mean = np.mean(data)
        var = np.var(data)
        
        if var < 1e-10:
            return np.zeros(max_lag + 1)
        
        autocorr = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[0] = 1.0
            else:
                c = np.mean((data[:-lag] - mean) * (data[lag:] - mean)) / var
                autocorr[lag] = c
        
        return autocorr
    
    def correlation_time(
        self, 
        autocorr: np.ndarray,
        threshold: float = np.exp(-1),
    ) -> float:
        """
        Estimate correlation time from autocorrelation.
        
        τ_c = smallest lag where C(τ) < threshold * C(0)
        """
        for i, c in enumerate(autocorr):
            if c < threshold:
                return float(i)
        return float(len(autocorr))
    
    def power_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectrum via FFT.
        
        Returns (frequencies, power) arrays.
        """
        data = np.asarray(data)
        N = len(data)
        
        # FFT
        fft = np.fft.fft(data)
        power = np.abs(fft)**2 / N
        
        # Frequencies
        freqs = np.fft.fftfreq(N)
        
        # Return positive frequencies only
        pos_mask = freqs >= 0
        return freqs[pos_mask], power[pos_mask]
    
    def spatial_correlation(
        self,
        sites: np.ndarray,
        max_r: int = 20,
    ) -> np.ndarray:
        """
        Compute spatial correlation function.
        
        C(r) = <s(0) * s(r)> - <s>²
        """
        if hasattr(sites, '_sites'):
            sites = sites._sites
        
        N = len(sites)
        mean = np.mean(sites)
        
        correlation = np.zeros(max_r + 1)
        
        for r in range(max_r + 1):
            corr_sum = 0.0
            for i in range(N):
                j = (i + r) % N
                corr_sum += sites[i] * sites[j]
            correlation[r] = corr_sum / N - mean**2
        
        return correlation
    
    def correlation_length(
        self,
        spatial_corr: np.ndarray,
    ) -> float:
        """
        Estimate correlation length from spatial correlation.
        
        ξ = Σ r * |C(r)| / Σ |C(r)|
        """
        r = np.arange(len(spatial_corr))
        abs_c = np.abs(spatial_corr)
        
        total = np.sum(abs_c)
        if total < 1e-10:
            return 0.0
        
        return float(np.sum(r * abs_c) / total)
    
    def histogram(
        self,
        data: np.ndarray,
        bins: int = 50,
        density: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of data.
        
        Returns (counts_or_density, bin_centers).
        """
        counts, bin_edges = np.histogram(data, bins=bins, density=density)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return counts, bin_centers
    
    def fluctuation_scaling(
        self,
        data: np.ndarray,
        window_sizes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fluctuation scaling (DFA-like analysis).
        
        F(L) = std of averages over windows of size L
        
        Returns (window_sizes, fluctuations).
        """
        data = np.asarray(data)
        N = len(data)
        
        if window_sizes is None:
            max_window = N // 4
            window_sizes = [2**i for i in range(2, int(np.log2(max_window)) + 1)]
        
        fluctuations = []
        valid_sizes = []
        
        for L in window_sizes:
            if L > N // 2:
                continue
            
            n_windows = N // L
            window_means = []
            
            for i in range(n_windows):
                window = data[i*L:(i+1)*L]
                window_means.append(np.mean(window))
            
            if len(window_means) > 1:
                fluctuations.append(np.std(window_means))
                valid_sizes.append(L)
        
        return np.array(valid_sizes), np.array(fluctuations)
    
    def fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Fit power law y = a * x^b using log-log regression.
        
        Returns (exponent_b, prefactor_a).
        """
        # Filter positive values
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return 0.0, 1.0
        
        # Log-log regression
        log_x = np.log(x)
        log_y = np.log(y)
        
        coeffs = np.polyfit(log_x, log_y, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        
        return float(b), float(a)


def compute_autocorrelation(data: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Convenience function for autocorrelation."""
    calc = StatisticsCalculator()
    return calc.autocorrelation(data, max_lag)


def compute_power_spectrum(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for power spectrum."""
    calc = StatisticsCalculator()
    return calc.power_spectrum(data)


class TimeSeriesAnalyzer:
    """
    Analyzer for time series from lattice evolution.
    
    Tracks multiple quantities over time and provides analysis.
    """
    
    def __init__(self):
        self._series: Dict[str, List[float]] = {}
        self._times: List[int] = []
        self._calc = StatisticsCalculator()
    
    def add_point(self, time: int, **quantities) -> None:
        """Add data point with multiple quantities."""
        self._times.append(time)
        
        for name, value in quantities.items():
            if name not in self._series:
                self._series[name] = []
            self._series[name].append(value)
    
    def get_series(self, name: str) -> np.ndarray:
        """Get time series for quantity."""
        return np.array(self._series.get(name, []))
    
    def get_times(self) -> np.ndarray:
        """Get time array."""
        return np.array(self._times)
    
    def summary(self, name: str) -> StatisticalSummary:
        """Get statistical summary for quantity."""
        data = self.get_series(name)
        return StatisticalSummary.from_data(data)
    
    def autocorrelation(self, name: str, max_lag: int = 100) -> np.ndarray:
        """Compute autocorrelation for quantity."""
        data = self.get_series(name)
        return self._calc.autocorrelation(data, max_lag)
    
    def cross_correlation(
        self,
        name1: str,
        name2: str,
        max_lag: int = 100,
    ) -> np.ndarray:
        """
        Compute cross-correlation between two quantities.
        
        C_{12}(τ) = <x₁(t) * x₂(t+τ)>
        """
        data1 = self.get_series(name1)
        data2 = self.get_series(name2)
        
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1)
        std2 = np.std(data2)
        
        if std1 < 1e-10 or std2 < 1e-10:
            return np.zeros(2 * max_lag + 1)
        
        N = min(len(data1), len(data2))
        cross_corr = np.zeros(2 * max_lag + 1)
        
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                c = np.mean((data1[:N-lag] - mean1) * (data2[lag:N] - mean2))
            else:
                c = np.mean((data1[-lag:N] - mean1) * (data2[:N+lag] - mean2))
            cross_corr[lag + max_lag] = c / (std1 * std2)
        
        return cross_corr
    
    def detect_stationarity(self, name: str, window: int = 100) -> bool:
        """
        Check if time series is approximately stationary.
        
        Compares mean/std in first and second half.
        """
        data = self.get_series(name)
        N = len(data)
        
        if N < 2 * window:
            return True
        
        first_half = data[:N//2]
        second_half = data[N//2:]
        
        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        std_diff = abs(np.std(first_half) - np.std(second_half))
        
        # Relative differences
        mean_rel = mean_diff / (abs(np.mean(data)) + 1e-10)
        std_rel = std_diff / (np.std(data) + 1e-10)
        
        return mean_rel < 0.1 and std_rel < 0.2
    
    def quantities(self) -> List[str]:
        """List tracked quantities."""
        return list(self._series.keys())
    
    def clear(self) -> None:
        """Clear all data."""
        self._series.clear()
        self._times.clear()
