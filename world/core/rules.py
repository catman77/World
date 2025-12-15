"""
Rule system for RSL World evolution.

Rules are local involutions T_i that transform patterns:
    T_i: pattern → transformed_pattern
    
Key properties:
- Reversibility: T_i² = id (applying twice restores original)
- Locality: Only affects sites within pattern
- Deterministic resolution: Left-to-right when conflicts

Rules can be:
- F1 filter rules (tension-based activation)
- RA rules (random action for H ≥ threshold)  
- Conservation rules (preserve charges)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Callable, Dict, Any
from enum import Enum
import numpy as np
from .lattice import Lattice, LatticeState


class RuleType(Enum):
    """Types of rules in the system."""
    FLIP = "flip"           # Single site flip
    SWAP = "swap"           # Swap adjacent sites
    PATTERN = "pattern"     # General pattern replacement
    INVOLUTION = "involution"  # Self-inverse pattern (T² = id)


class RuleActivation(Enum):
    """Rule activation conditions."""
    ALWAYS = "always"       # Always applicable if pattern matches
    TENSION = "tension"     # Apply if local tension exceeds threshold
    RANDOM = "random"       # Apply with probability based on tension
    CAPACITY = "capacity"   # Apply if local capacity allows


@dataclass
class Rule:
    """
    Local transformation rule T_i.
    
    A rule specifies:
    - Pattern to match (input)
    - Replacement pattern (output)
    - Activation condition
    - Conservation constraints
    
    For involutions (T² = id), the inverse is computed automatically.
    
    Example:
        # Single flip rule
        rule = Rule(
            name="flip",
            pattern=[1],      # Match +
            replacement=[-1], # Replace with -
            rule_type=RuleType.FLIP,
        )
        
        # Domain wall hop
        rule = Rule(
            name="wall_hop",
            pattern=[1, -1],      # Match +-
            replacement=[-1, 1],  # Replace with -+
            rule_type=RuleType.SWAP,
        )
    """
    name: str
    pattern: np.ndarray | List[int]  # Values to match
    replacement: np.ndarray | List[int]  # Values after transformation
    
    rule_type: RuleType = RuleType.PATTERN
    activation: RuleActivation = RuleActivation.ALWAYS
    
    # Activation parameters
    tension_threshold: float = 0.0  # For TENSION activation
    probability: float = 1.0        # For RANDOM activation
    
    # Conservation
    conserves_magnetization: bool = False  # Sum of symbols preserved
    conserves_domain_walls: bool = False   # Number of walls preserved
    
    # Metadata
    priority: int = 0              # Higher = applied first
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert to numpy arrays
        if not isinstance(self.pattern, np.ndarray):
            self.pattern = np.array(self.pattern, dtype=np.int8)
        if not isinstance(self.replacement, np.ndarray):
            self.replacement = np.array(self.replacement, dtype=np.int8)
        
        # Validate lengths match
        if len(self.pattern) != len(self.replacement):
            raise ValueError(
                f"Pattern length {len(self.pattern)} != replacement length {len(self.replacement)}"
            )
    
    @property
    def length(self) -> int:
        """Pattern length (L in the theory)."""
        return len(self.pattern)
    
    @property
    def L(self) -> int:
        """Alias for length."""
        return len(self.pattern)
    
    def matches(self, lattice: Lattice, position: int) -> bool:
        """
        Check if pattern matches at position.
        
        Uses periodic boundary conditions if lattice is periodic.
        """
        for i, expected in enumerate(self.pattern):
            if lattice[position + i] != expected:
                return False
        return True
    
    def apply(self, lattice: Lattice, position: int) -> bool:
        """
        Apply rule at position (in place).
        
        Returns True if applied, False if pattern didn't match.
        """
        if not self.matches(lattice, position):
            return False
        
        for i, value in enumerate(self.replacement):
            lattice[position + i] = value
        
        return True
    
    def is_involution(self) -> bool:
        """
        Check if rule is self-inverse (T² = id).
        
        A rule is an involution if applying it twice restores the original.
        """
        # For a pattern replacement rule to be an involution,
        # applying the rule to the replacement must give back the pattern
        if self.rule_type == RuleType.FLIP:
            return True  # Flip is always self-inverse
        if self.rule_type == RuleType.SWAP:
            return True  # Swap is always self-inverse
            
        # Check if applying to replacement gives pattern
        return np.array_equal(self.pattern, self.replacement)
    
    def inverse(self) -> "Rule":
        """
        Get inverse rule (swap pattern and replacement).
        
        For involutions, returns self.
        """
        if self.is_involution():
            return self
        
        return Rule(
            name=f"{self.name}_inv",
            pattern=self.replacement.copy(),
            replacement=self.pattern.copy(),
            rule_type=self.rule_type,
            activation=self.activation,
            tension_threshold=self.tension_threshold,
            probability=self.probability,
            conserves_magnetization=self.conserves_magnetization,
            conserves_domain_walls=self.conserves_domain_walls,
            priority=self.priority,
            metadata=self.metadata.copy(),
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return False
        return (np.array_equal(self.pattern, other.pattern) and 
                np.array_equal(self.replacement, other.replacement))
    
    def __hash__(self) -> int:
        return hash((self.pattern.tobytes(), self.replacement.tobytes()))
    
    def __repr__(self) -> str:
        p = "".join("+" if x > 0 else "-" for x in self.pattern)
        r = "".join("+" if x > 0 else "-" for x in self.replacement)
        return f"Rule('{self.name}': {p} → {r})"


@dataclass
class RuleMatch:
    """A rule that matched at a specific position."""
    rule: Rule
    position: int
    
    @property
    def end_position(self) -> int:
        """End position (exclusive) of match."""
        return self.position + self.rule.length


class RuleConflict(Exception):
    """Raised when rules conflict and cannot be resolved."""
    def __init__(self, matches: List[RuleMatch], message: str = ""):
        self.matches = matches
        super().__init__(message or f"Conflicting rules at positions: {[m.position for m in matches]}")


class RuleSet:
    """
    Collection of rules with conflict resolution.
    
    Provides:
    - Pattern matching across lattice
    - Conflict detection (overlapping matches)
    - Deterministic resolution (left-to-right)
    - Conservation law checking
    
    Resolution strategy (deterministic, left-to-right):
    1. Find all matches sorted by position (ascending)
    2. For ties, sort by priority (descending)
    3. Apply first match
    4. Skip any matches that overlap with applied match
    5. Continue with remaining matches
    
    Example:
        rules = RuleSet()
        rules.add(Rule("flip_plus", [1], [-1]))
        rules.add(Rule("flip_minus", [-1], [1]))
        
        # Find all applicable rules
        matches = rules.find_matches(lattice)
        
        # Apply with conflict resolution
        applied = rules.apply_all(lattice)
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = list(rules) if rules else []
        self._by_length: Dict[int, List[Rule]] = {}
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild rule index by pattern length."""
        self._by_length.clear()
        for rule in self.rules:
            length = rule.length
            if length not in self._by_length:
                self._by_length[length] = []
            self._by_length[length].append(rule)
    
    def add(self, rule: Rule) -> "RuleSet":
        """Add a rule to the set."""
        self.rules.append(rule)
        length = rule.length
        if length not in self._by_length:
            self._by_length[length] = []
        self._by_length[length].append(rule)
        return self
    
    def remove(self, rule: Rule) -> bool:
        """Remove a rule from the set. Returns True if found."""
        if rule in self.rules:
            self.rules.remove(rule)
            self._rebuild_index()
            return True
        return False
    
    def find_matches(
        self, 
        lattice: Lattice,
        positions: Optional[List[int]] = None,
    ) -> List[RuleMatch]:
        """
        Find all rule matches in the lattice.
        
        Args:
            lattice: Lattice to search
            positions: Optional list of positions to check (default: all)
            
        Returns:
            List of RuleMatch objects sorted by (position, -priority)
        """
        matches = []
        
        if positions is None:
            positions = range(lattice.size)
        
        for pos in positions:
            for rule in self.rules:
                if rule.matches(lattice, pos):
                    matches.append(RuleMatch(rule=rule, position=pos))
        
        # Sort by position (ascending), then priority (descending)
        matches.sort(key=lambda m: (m.position, -m.rule.priority))
        
        return matches
    
    def find_conflicts(self, matches: List[RuleMatch]) -> List[Tuple[RuleMatch, RuleMatch]]:
        """
        Find pairs of overlapping matches.
        
        Two matches conflict if their affected regions overlap.
        """
        conflicts = []
        
        for i, m1 in enumerate(matches):
            for m2 in matches[i+1:]:
                # Check overlap
                if m1.position < m2.end_position and m2.position < m1.end_position:
                    conflicts.append((m1, m2))
        
        return conflicts
    
    def resolve_conflicts(
        self, 
        matches: List[RuleMatch],
        strategy: str = "left_to_right",
    ) -> List[RuleMatch]:
        """
        Resolve conflicts between matches.
        
        Strategy:
        - "left_to_right": Take leftmost match, skip overlapping (deterministic)
        - "priority": Take highest priority, skip overlapping
        - "random": Random selection among conflicts
        
        Returns list of non-overlapping matches to apply.
        """
        if not matches:
            return []
        
        if strategy == "left_to_right":
            # Matches are already sorted by position
            resolved = []
            last_end = -1
            
            for match in matches:
                if match.position >= last_end:
                    resolved.append(match)
                    last_end = match.end_position
            
            return resolved
        
        elif strategy == "priority":
            # Sort by priority first
            sorted_matches = sorted(matches, key=lambda m: -m.rule.priority)
            resolved = []
            used_positions: Set[int] = set()
            
            for match in sorted_matches:
                match_positions = set(range(match.position, match.end_position))
                if not match_positions & used_positions:
                    resolved.append(match)
                    used_positions |= match_positions
            
            return resolved
        
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")
    
    def apply_all(
        self, 
        lattice: Lattice,
        strategy: str = "left_to_right",
    ) -> List[RuleMatch]:
        """
        Find all matches, resolve conflicts, and apply rules.
        
        Returns list of applied matches.
        """
        matches = self.find_matches(lattice)
        resolved = self.resolve_conflicts(matches, strategy)
        
        # Apply in order
        for match in resolved:
            match.rule.apply(lattice, match.position)
        
        return resolved
    
    def check_conservation(
        self, 
        before: LatticeState, 
        after: LatticeState,
    ) -> Dict[str, bool]:
        """
        Check if conservation laws are satisfied.
        
        Returns dict mapping conservation law name to whether it holds.
        """
        results = {}
        
        # Magnetization conservation
        mag_before = np.sum(before.sites)
        mag_after = np.sum(after.sites)
        results["magnetization"] = (mag_before == mag_after)
        
        # Domain wall conservation (approximate - count walls)
        walls_before = np.sum(before.sites[:-1] != before.sites[1:])
        walls_after = np.sum(after.sites[:-1] != after.sites[1:])
        results["domain_walls"] = (walls_before == walls_after)
        
        return results
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __iter__(self):
        return iter(self.rules)
    
    def __repr__(self) -> str:
        return f"RuleSet({len(self.rules)} rules)"


# ===== Predefined rule generators =====

def create_flip_rules() -> RuleSet:
    """Create basic flip rules: + ↔ -"""
    return RuleSet([
        Rule("flip_plus", [1], [-1], RuleType.FLIP),
        Rule("flip_minus", [-1], [1], RuleType.FLIP),
    ])


def create_swap_rules() -> RuleSet:
    """Create adjacent swap rules: +- ↔ -+"""
    return RuleSet([
        Rule("swap", [1, -1], [-1, 1], RuleType.SWAP),
    ])


def create_wall_hop_rules() -> RuleSet:
    """
    Create domain wall hopping rules.
    
    A domain wall is where +/- meet. These rules let walls move:
    +-- → -+-  (wall hops right)
    --+ → -+-  (wall hops left)
    """
    return RuleSet([
        Rule("wall_right", [1, -1, -1], [-1, 1, -1], RuleType.PATTERN, priority=1),
        Rule("wall_left", [-1, -1, 1], [-1, 1, -1], RuleType.PATTERN, priority=1),
        Rule("wall_right_inv", [-1, 1, 1], [1, -1, 1], RuleType.PATTERN, priority=1),
        Rule("wall_left_inv", [1, 1, -1], [1, -1, 1], RuleType.PATTERN, priority=1),
    ])


def create_conserving_rules(max_length: int = 3) -> RuleSet:
    """
    Generate all rules that conserve magnetization.
    
    For conservation, sum(pattern) must equal sum(replacement).
    """
    rules = RuleSet()
    
    # For each length, generate all conserving transformations
    for L in range(2, max_length + 1):
        # Generate all patterns of length L
        for p in range(2**L):
            pattern = np.array([(1 if (p >> i) & 1 else -1) for i in range(L)], dtype=np.int8)
            pattern_sum = np.sum(pattern)
            
            # Find all replacements with same sum
            for r in range(2**L):
                if r == p:  # Skip identity
                    continue
                replacement = np.array([(1 if (r >> i) & 1 else -1) for i in range(L)], dtype=np.int8)
                if np.sum(replacement) == pattern_sum:
                    # Create rule (only need one direction for involutions)
                    if r > p:  # Avoid duplicates
                        name = f"conserve_L{L}_{p:0{L}b}_{r:0{L}b}"
                        rules.add(Rule(
                            name=name,
                            pattern=pattern,
                            replacement=replacement,
                            rule_type=RuleType.INVOLUTION,
                            conserves_magnetization=True,
                        ))
    
    return rules
