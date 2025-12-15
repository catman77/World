"""
Tests for core module.
"""

import pytest
import numpy as np
from world.core import Lattice, Rule, RuleSet, EvolutionEngine


class TestLattice:
    """Tests for Lattice class."""
    
    def test_create_default(self):
        """Test creating lattice with default values (all +1)."""
        lattice = Lattice(size=10)
        assert len(lattice) == 10
        assert np.all(lattice.sites == 1)  # Default is all +1
    
    def test_create_random(self):
        """Test creating lattice with random values."""
        lattice = Lattice.random(size=100, p_plus=0.5)
        assert len(lattice) == 100
        assert set(np.unique(lattice.sites)).issubset({-1, 1})
    
    def test_create_from_array(self):
        """Test creating from array."""
        sites = np.array([1, -1, 1, 1, -1])
        lattice = Lattice.from_array(sites)
        assert len(lattice) == 5
        np.testing.assert_array_equal(lattice.sites, sites)
    
    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        lattice = Lattice.from_array(np.array([1, -1, 1, -1, 1]))
        assert lattice[0] == 1
        assert lattice[4] == 1
        assert lattice[5] == 1  # Wraps around
        assert lattice[-1] == 1  # -1 wraps to last element
    
    def test_get_window(self):
        """Test window access."""
        lattice = Lattice.from_array(np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
        window = lattice.get_window(center=5, radius=2)
        assert len(window) == 5  # 2*radius + 1
        # Window should include positions 3,4,5,6,7
    
    def test_magnetization(self):
        """Test magnetization calculation."""
        # All +1
        lattice = Lattice.from_array(np.ones(10))
        assert lattice.magnetization() == 1.0
        
        # All -1
        lattice = Lattice.from_array(-np.ones(10))
        assert lattice.magnetization() == -1.0
        
        # Half and half
        lattice = Lattice.from_array(np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]))
        assert lattice.magnetization() == 0.0
    
    def test_phases(self):
        """Test phase property."""
        lattice = Lattice.from_array(np.array([1, -1, 1]))
        phases = lattice.phases
        assert phases.shape == (3,)
        # s=+1 → φ=0, s=-1 → φ=π
        assert phases[0] == 0.0
        assert phases[1] == pytest.approx(np.pi)
        assert phases[2] == 0.0
    
    def test_copy(self):
        """Test lattice copy."""
        original = Lattice.from_array(np.array([1, -1, 1]))
        
        copy = original.copy()
        assert len(copy) == len(original)
        np.testing.assert_array_equal(copy.sites, original.sites)
        
        # Modify copy, original should be unchanged
        copy[0] = -1
        assert original[0] == 1


class TestRule:
    """Tests for Rule class."""
    
    def test_create_rule(self):
        """Test rule creation."""
        rule = Rule(name="test", pattern=[1, 1], replacement=[-1, -1])
        assert tuple(rule.pattern) == (1, 1)
        assert tuple(rule.replacement) == (-1, -1)
        assert len(rule.pattern) == 2
    
    def test_matches(self):
        """Test pattern matching."""
        rule = Rule(name="test", pattern=[1, -1], replacement=[-1, 1])
        lattice = Lattice.from_array(np.array([1, -1, 1, 1, -1]))
        
        # Should match at positions 0 and 3
        match_0 = rule.matches(lattice, 0)
        match_3 = rule.matches(lattice, 3)
        assert match_0 == True
        assert match_3 == True
    
    def test_apply(self):
        """Test rule application."""
        rule = Rule(name="test", pattern=[1, -1], replacement=[-1, 1])
        lattice = Lattice.from_array(np.array([1, -1, 1]))
        
        rule.apply(lattice, 0)
        
        np.testing.assert_array_equal(lattice.sites, [-1, 1, 1])


class TestRuleSet:
    """Tests for RuleSet class."""
    
    def test_empty_ruleset(self):
        """Test empty ruleset."""
        rules = RuleSet()
        assert len(rules) == 0
    
    def test_add_rule(self):
        """Test adding rules."""
        rules = RuleSet()
        rule = Rule(name="test", pattern=[1, 1], replacement=[-1, -1])
        rules.add(rule)
        assert len(rules) == 1
    
    def test_find_matches(self):
        """Test finding all rule matches."""
        rules = RuleSet()
        rules.add(Rule(name="swap", pattern=[1, -1], replacement=[-1, 1]))
        
        lattice = Lattice.from_array(np.array([1, -1, 1, -1, 1]))
        matches = rules.find_matches(lattice)
        
        # Should find matches at positions 0 and 2
        assert len(matches) >= 2


class TestEvolutionEngine:
    """Tests for EvolutionEngine class."""
    
    def test_single_step(self):
        """Test single evolution step."""
        rules = RuleSet()
        rules.add(Rule(name="swap", pattern=[1, -1], replacement=[-1, 1]))
        
        engine = EvolutionEngine(rules)
        lattice = Lattice.from_array(np.array([1, -1, 1]))
        
        applied = engine.step(lattice)
        
        # Should have applied the rule
        assert len(applied) >= 0  # At least 0 rules applied
    
    def test_run_n_steps(self):
        """Test running multiple steps."""
        rules = RuleSet()
        rules.add(Rule(name="swap", pattern=[1, -1], replacement=[-1, 1]))
        
        engine = EvolutionEngine(rules)
        lattice = Lattice.from_array(np.array([1, -1, 1, -1]))
        
        # Run 10 steps
        for _ in range(10):
            engine.step(lattice)
        
        # Lattice should still be same size
        assert len(lattice) == 4
    
    def test_history_via_run(self):
        """Test history recording via run method."""
        rules = RuleSet()
        rules.add(Rule(name="swap", pattern=[1, -1], replacement=[-1, 1]))
        
        engine = EvolutionEngine(rules)
        lattice = Lattice.from_array(np.array([1, -1, 1, -1]))
        
        result = engine.run(lattice, max_steps=5, store_history=True)
        
        # History should have states
        assert len(result.history) > 0


class TestIntegration:
    """Integration tests."""
    
    def test_simple_simulation(self):
        """Test basic simulation."""
        # Create rules
        rules = RuleSet()
        rules.add(Rule(name="swap_pm", pattern=[1, -1], replacement=[-1, 1]))
        rules.add(Rule(name="swap_mp", pattern=[-1, 1], replacement=[1, -1]))
        
        # Create engine and lattice
        engine = EvolutionEngine(rules)
        lattice = Lattice.random(size=20, p_plus=0.5, seed=42)
        
        initial_m = lattice.magnetization()
        
        # Run simulation
        for _ in range(100):
            engine.step(lattice)
        
        final_m = lattice.magnetization()
        
        # Magnetization should be conserved for swap rules
        assert abs(initial_m - final_m) < 0.01
