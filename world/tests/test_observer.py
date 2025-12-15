"""
Tests for observer module.
"""

import pytest
import numpy as np
from world.core import Lattice
from world.observer import Observer


class TestObserver:
    """Tests for Observer class."""
    
    def test_create_observer(self):
        """Test observer creation."""
        observer = Observer(radius=5, max_levels=3)
        assert observer.radius == 5
        assert observer.max_levels == 3
    
    def test_observe(self):
        """Test observation of lattice."""
        lattice = Lattice.random(size=100, p_plus=0.5, seed=42)
        observer = Observer(radius=5)
        
        state = observer.observe(lattice)
        
        # Should have coarse-grained field
        assert state.coarse_field is not None
        assert len(state.coarse_field) > 0
    
    def test_observation_history(self):
        """Test observation history."""
        observer = Observer(radius=5)
        lattice = Lattice.random(size=50, p_plus=0.5, seed=42)
        
        # Make multiple observations
        for _ in range(5):
            observer.observe(lattice)
        
        # History is stored internally
        assert len(observer._observation_history) == 5


class TestProjection:
    """Tests for projection functions."""
    
    def test_coarse_field_values(self):
        """Test coarse field calculation."""
        # Uniform lattice should give uniform coarse field
        lattice = Lattice.from_array(np.ones(100))
        observer = Observer(radius=3)
        
        state = observer.observe(lattice)
        
        # Coarse field should be close to 1 (all +1)
        assert np.mean(state.coarse_field) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
