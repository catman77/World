"""
Tests for analysis module.
"""

import pytest
import numpy as np
from world.core import Lattice
from world.analysis import (
    compute_autocorrelation,
    compute_power_spectrum,
    hamming_distance,
    edit_distance,
    compute_order_parameter,
)


class TestStatistics:
    """Tests for statistical functions."""
    
    def test_autocorrelation(self):
        """Test autocorrelation calculation."""
        # Periodic signal should have periodic autocorrelation
        signal = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        acf = compute_autocorrelation(signal, max_lag=4)
        
        assert len(acf) == 5  # lag 0 to 4
        assert acf[0] == pytest.approx(1.0)  # Self-correlation
    
    def test_power_spectrum(self):
        """Test power spectrum calculation."""
        # Periodic signal
        signal = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        freq, power = compute_power_spectrum(signal)
        
        # Should have peak at Nyquist frequency (period 2)
        assert len(freq) == len(power)


class TestMetrics:
    """Tests for distance metrics."""
    
    def test_hamming_identical(self):
        """Identical sequences should have zero distance."""
        s1 = np.array([1, -1, 1, -1])
        s2 = np.array([1, -1, 1, -1])
        
        assert hamming_distance(s1, s2) == 0
    
    def test_hamming_all_different(self):
        """Completely different sequences should have maximum distance."""
        s1 = np.array([1, 1, 1, 1])
        s2 = np.array([-1, -1, -1, -1])
        
        assert hamming_distance(s1, s2) == 4
    
    def test_edit_distance(self):
        """Test edit distance calculation."""
        s1 = np.array([1, -1, 1])
        s2 = np.array([1, 1, 1])
        
        # One position differs
        d = edit_distance(s1, s2)
        assert d >= 1


class TestPhaseStructure:
    """Tests for phase structure analysis."""
    
    def test_order_parameter(self):
        """Test order parameter calculation."""
        # Uniform lattice should have |φ| = 1
        lattice = Lattice.from_array(np.ones(10))
        phi = compute_order_parameter(lattice)
        assert abs(phi) == pytest.approx(1.0)
        
        # Mixed lattice should have |φ| < 1
        lattice = Lattice.from_array(np.array([1, 1, 1, -1, -1, -1]))
        phi = compute_order_parameter(lattice)
        assert abs(phi) < 1.0


class TestAnalysisIntegration:
    """Integration tests."""
    
    def test_full_analysis_pipeline(self):
        """Test running full analysis on a lattice."""
        lattice = Lattice.random(size=100, p_plus=0.5, seed=42)
        
        # Order parameter
        phi = compute_order_parameter(lattice)
        assert -1 <= phi <= 1
        
        # Autocorrelation
        acf = compute_autocorrelation(lattice.sites.astype(float), max_lag=10)
        assert len(acf) == 11
        
        # Power spectrum
        freq, power = compute_power_spectrum(lattice.sites.astype(float))
        assert len(freq) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
