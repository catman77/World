"""
Tests for RSL module.
"""

import pytest
import numpy as np
from world.core import Lattice
from world.rsl import TensionCalculator, CapacityCalculator, F1Filter


class TestTensionCalculator:
    """Tests for TensionCalculator."""
    
    def test_uniform_lattice(self):
        """Uniform lattice should have zero tension."""
        sites = np.ones(10, dtype=np.int8)
        calc = TensionCalculator(J=1.0)
        
        h = calc.global_tension(sites)
        assert h == 0.0
    
    def test_alternating_lattice(self):
        """Alternating lattice should have maximum tension."""
        # Alternating pattern: all domain walls
        sites = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
        calc = TensionCalculator(J=1.0)
        
        h = calc.global_tension(sites)
        # 10 domain walls (including periodic boundary)
        assert h == 10.0
    
    def test_local_tension(self):
        """Test local tension calculation."""
        sites = np.array([1, 1, -1, -1, 1, 1], dtype=np.int8)
        calc = TensionCalculator()
        
        local_h = calc.local_tension(sites)
        
        # Domain walls at positions 1-2, 3-4 (plus periodic 5-0)
        assert local_h.shape == (6,)
    
    def test_scaling_with_J(self):
        """Test tension scales with J."""
        sites = np.array([1, -1, 1, -1], dtype=np.int8)
        
        calc1 = TensionCalculator(J=1.0)
        calc2 = TensionCalculator(J=2.0)
        
        h1 = calc1.global_tension(sites)
        h2 = calc2.global_tension(sites)
        
        assert h2 == 2 * h1


class TestCapacityCalculator:
    """Tests for CapacityCalculator."""
    
    def test_uniform_lattice(self):
        """Uniform lattice should have constant capacity."""
        sites = np.ones(10, dtype=np.int8)
        calc = CapacityCalculator(C0=2.0, alpha=0.5)
        
        capacities = calc.local_capacity(sites)
        
        # No domain walls, so C = C0 everywhere
        assert np.all(capacities == 2.0)
    
    def test_domain_wall(self):
        """Capacity should decrease at domain walls."""
        sites = np.array([1, 1, -1, -1], dtype=np.int8)
        calc = CapacityCalculator(C0=2.0, alpha=0.5)
        
        capacities = calc.local_capacity(sites)
        
        # Capacities should be calculated
        assert len(capacities) == 4
    
    def test_capacity_bounds(self):
        """Capacity should not go below zero."""
        # Create high tension lattice
        sites = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)
        calc = CapacityCalculator(C0=2.0, alpha=10.0)  # High alpha
        
        capacities = calc.local_capacity(sites)
        
        # All capacities should be non-negative
        assert np.all(capacities >= 0)


class TestF1Filter:
    """Tests for F1Filter."""
    
    def test_low_tension_passes(self):
        """Low tension state should pass filter."""
        lattice = Lattice.from_array(np.ones(10))  # Uniform
        f1 = F1Filter(threshold=0.5)
        
        assert f1.passes(lattice) is True
    
    def test_high_tension_blocked(self):
        """High tension state should be blocked."""
        lattice = Lattice.from_array(np.array([1, -1, 1, -1, 1, -1]))  # Alternating
        f1 = F1Filter(threshold=0.1)  # Low threshold
        
        assert f1.passes(lattice) is False
    
    def test_active_positions(self):
        """Test active positions detection."""
        lattice = Lattice.from_array(np.array([1, 1, -1, -1, 1, 1]))
        f1 = F1Filter(threshold=0.1)
        
        active = f1.active_positions(lattice.sites)
        
        # Should detect positions near domain walls
        assert isinstance(active, np.ndarray)


class TestRSLIntegration:
    """Integration tests for RSL module."""
    
    def test_tension_capacity_consistency(self):
        """High tension should correspond to low capacity."""
        # Use a lattice with varying tension
        sites = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
        
        tension_calc = TensionCalculator(J=1.0)
        capacity_calc = CapacityCalculator(C0=2.0, alpha=0.5)
        
        local_h = tension_calc.local_tension(sites)
        capacities = capacity_calc.local_capacity(sites)
        
        # At domain walls, tension is high (1) and capacity is low
        # Check that capacity formula is applied correctly: C = C0 - alpha * h
        for i in range(len(sites)):
            expected_capacity = max(0, 2.0 - 0.5 * local_h[i])
            assert capacities[i] == pytest.approx(expected_capacity, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
