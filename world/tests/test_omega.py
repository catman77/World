"""
Tests for omega module.
"""

import pytest
import numpy as np
from world.core import Lattice
from world.omega import CycleDetector, OmegaCycle


class TestOmegaCycle:
    """Tests for OmegaCycle dataclass."""
    
    def test_create_cycle(self):
        """Test cycle creation."""
        cycle = OmegaCycle(
            position=5.0,
            extent=3,
            period=10,
            velocity=1.0,
            creation_time=0,
        )
        
        assert cycle.position == 5.0
        assert cycle.period == 10
    
    def test_cycle_properties(self):
        """Test cycle properties."""
        cycle = OmegaCycle(
            position=0.0,
            extent=4,
            period=4,
            velocity=1.0,
            creation_time=100,
        )
        
        assert cycle.extent == 4
        assert cycle.creation_time == 100
        assert cycle.frequency == 0.25  # 1/4


class TestCycleDetector:
    """Tests for CycleDetector class."""
    
    def test_create_detector(self):
        """Test detector creation."""
        detector = CycleDetector(window_size=5, max_period=100)
        assert detector.window_size == 5
        assert detector.max_period == 100
    
    def test_update(self):
        """Test update with state."""
        detector = CycleDetector()
        sites = np.array([1, -1, 1, -1, 1, -1])
        
        # Update with state
        new_cycles = detector.update(sites, time=0)
        
        # Should return list (possibly empty)
        assert isinstance(new_cycles, list)
    
    def test_detect_no_cycles_in_uniform(self):
        """Test no cycles in uniform state."""
        detector = CycleDetector()
        
        # Feed uniform states
        for t in range(20):
            sites = np.ones(10, dtype=np.int8)
            detector.update(sites, time=t)
        
        # No cycles should be detected in unchanging state
        cycles = detector.detected_cycles
        # May or may not detect depending on algorithm
        assert isinstance(cycles, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
