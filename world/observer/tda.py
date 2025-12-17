"""
TDA (Topological Data Analysis) for semantic trajectory analysis.

Analyzes the topology of the observer's understanding trajectory in
semantic space using persistent homology:

- β₀: Number of connected components (concepts/clusters)
- β₁: Number of loops (cyclic dependencies/patterns)

The semantic trajectory xsem(t) forms a point cloud in high-dimensional
space. TDA reveals the structure of this trajectory:

- Many β₀ components → fragmented understanding
- Large β₁ loops → recurring patterns in learning
- Stable topology → convergent understanding

Uses Vietoris-Rips complex for computational tractability.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from .semantics import SemanticHistory


@dataclass
class PersistenceInterval:
    """
    A persistence interval (birth, death) in the persistence diagram.
    
    Represents a topological feature that appears at 'birth' scale
    and disappears at 'death' scale.
    """
    birth: float
    death: float
    dimension: int  # 0 for components, 1 for loops
    
    @property
    def persistence(self) -> float:
        """Lifetime of the feature (death - birth)."""
        return self.death - self.birth
    
    @property
    def midpoint(self) -> float:
        """Midpoint of the interval."""
        return (self.birth + self.death) / 2
    
    def __repr__(self) -> str:
        return f"[{self.birth:.3f}, {self.death:.3f}]_{self.dimension}"


@dataclass
class TopologicalSummary:
    """
    Summary of topological features from persistent homology.
    """
    # Betti numbers at reference scale
    beta_0: int = 0  # Connected components
    beta_1: int = 0  # Loops/cycles
    
    # Persistence statistics
    total_persistence_0: float = 0.0
    total_persistence_1: float = 0.0
    max_persistence_0: float = 0.0
    max_persistence_1: float = 0.0
    
    # Number of significant features
    n_significant_0: int = 0
    n_significant_1: int = 0
    
    # Entropy of persistence (complexity measure)
    persistence_entropy: float = 0.0
    
    # Reference scale used
    scale: float = 0.0
    
    def __repr__(self) -> str:
        return (f"TDA(β₀={self.beta_0}, β₁={self.beta_1}, "
                f"pers₀={self.total_persistence_0:.3f}, "
                f"pers₁={self.total_persistence_1:.3f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "beta_0": self.beta_0,
            "beta_1": self.beta_1,
            "total_persistence_0": self.total_persistence_0,
            "total_persistence_1": self.total_persistence_1,
            "max_persistence_0": self.max_persistence_0,
            "max_persistence_1": self.max_persistence_1,
            "n_significant_0": self.n_significant_0,
            "n_significant_1": self.n_significant_1,
            "persistence_entropy": self.persistence_entropy,
            "scale": self.scale,
        }


class VietorisRipsComplex:
    """
    Vietoris-Rips complex construction for point cloud.
    
    For a point cloud X and scale ε, the VR complex includes:
    - A vertex for each point
    - An edge between points with distance ≤ ε
    - A triangle if all three pairwise distances ≤ ε
    - etc.
    
    This is a computational simplification (compared to Čech complex)
    that still captures homological features.
    """
    
    def __init__(self, points: np.ndarray):
        """
        Initialize with point cloud.
        
        Args:
            points: (n_points, n_dims) array
        """
        self.points = np.asarray(points)
        self.n_points = len(points)
        
        # Compute pairwise distances
        if self.n_points > 1:
            self.distances = squareform(pdist(points))
        else:
            self.distances = np.zeros((1, 1))
    
    def get_edges_at_scale(self, epsilon: float) -> List[Tuple[int, int]]:
        """Get all edges (pairs) with distance ≤ epsilon."""
        edges = []
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                if self.distances[i, j] <= epsilon:
                    edges.append((i, j))
        return edges
    
    def count_components_at_scale(self, epsilon: float) -> int:
        """
        Count connected components at given scale using Union-Find.
        """
        if self.n_points == 0:
            return 0
        
        # Union-Find
        parent = list(range(self.n_points))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union points within distance epsilon
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                if self.distances[i, j] <= epsilon:
                    union(i, j)
        
        # Count unique roots
        return len(set(find(i) for i in range(self.n_points)))


class PersistentHomology:
    """
    Compute persistent homology for point cloud data.
    
    Uses a simplified approach:
    - H₀ (components): Track component merging via Union-Find
    - H₁ (loops): Approximate via Euler characteristic
    
    For full computation, consider using external libraries like
    ripser, gudhi, or dionysus.
    """
    
    def __init__(
        self,
        max_dimension: int = 1,
        n_scales: int = 50,
    ):
        """
        Initialize persistent homology computation.
        
        Args:
            max_dimension: Maximum homology dimension (0 or 1)
            n_scales: Number of scale values for filtration
        """
        self.max_dimension = max_dimension
        self.n_scales = n_scales
    
    def compute(
        self,
        points: np.ndarray,
        max_scale: Optional[float] = None,
    ) -> Tuple[List[PersistenceInterval], TopologicalSummary]:
        """
        Compute persistent homology.
        
        Args:
            points: (n_points, n_dims) array
            max_scale: Maximum filtration scale (auto if None)
            
        Returns:
            Tuple of (persistence_intervals, summary)
        """
        points = np.asarray(points)
        
        if len(points) < 2:
            return [], TopologicalSummary()
        
        # Build VR complex
        vr = VietorisRipsComplex(points)
        
        # Determine scale range
        if max_scale is None:
            max_scale = np.max(vr.distances) * 1.1
        
        scales = np.linspace(0, max_scale, self.n_scales)
        
        # Track H₀ (connected components)
        intervals_0 = self._compute_h0_persistence(vr, scales)
        
        # Track H₁ (loops) - simplified approach
        intervals_1 = []
        if self.max_dimension >= 1:
            intervals_1 = self._compute_h1_persistence(vr, scales)
        
        all_intervals = intervals_0 + intervals_1
        
        # Compute summary at median scale
        ref_scale = scales[len(scales) // 2]
        summary = self._compute_summary(all_intervals, ref_scale, max_scale)
        
        return all_intervals, summary
    
    def _compute_h0_persistence(
        self,
        vr: VietorisRipsComplex,
        scales: np.ndarray,
    ) -> List[PersistenceInterval]:
        """
        Compute H₀ persistence (connected components).
        
        Each point is born at scale 0. Components merge at the scale
        where they first become connected.
        """
        n = vr.n_points
        if n == 0:
            return []
        
        # All points born at 0
        birth_times = [0.0] * n
        death_times = [float('inf')] * n  # Infinity until merged
        
        # Track which components have merged
        parent = list(range(n))
        active = [True] * n  # Component still exists
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # Process edges in order of distance
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((vr.distances[i, j], i, j))
        edges.sort()
        
        for dist, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                # Merge components - one dies
                parent[pi] = pj
                death_times[pi] = dist
                active[pi] = False
        
        # Create intervals
        intervals = []
        for i in range(n):
            if active[i] or death_times[i] < float('inf'):
                death = death_times[i] if death_times[i] < float('inf') else scales[-1]
                intervals.append(PersistenceInterval(
                    birth=birth_times[i],
                    death=death,
                    dimension=0
                ))
        
        return intervals
    
    def _compute_h1_persistence(
        self,
        vr: VietorisRipsComplex,
        scales: np.ndarray,
    ) -> List[PersistenceInterval]:
        """
        Compute H₁ persistence (loops).
        
        Uses Euler characteristic approximation:
            χ = V - E + F (for simplicial complex)
            β₀ - β₁ + β₂ - ... = χ
        
        For VR complex up to dimension 1:
            β₁ ≈ E - V + β₀
        """
        n = vr.n_points
        if n < 3:
            return []
        
        # Track β₁ at each scale
        prev_beta1 = 0
        intervals = []
        pending_loops = []  # Loops waiting to close
        
        for idx, eps in enumerate(scales[1:], 1):
            n_edges = len(vr.get_edges_at_scale(eps))
            n_components = vr.count_components_at_scale(eps)
            
            # Euler characteristic: χ = V - E + F (F=0 for 1-skeleton)
            # β₀ - β₁ = χ = V - E
            # β₁ = β₀ - V + E = n_components - n + n_edges
            beta1 = max(0, n_components - n + n_edges)
            
            # New loops appeared
            if beta1 > prev_beta1:
                for _ in range(beta1 - prev_beta1):
                    pending_loops.append(scales[idx-1])  # Birth at previous scale
            
            # Loops closed
            elif beta1 < prev_beta1:
                for _ in range(prev_beta1 - beta1):
                    if pending_loops:
                        birth = pending_loops.pop(0)
                        intervals.append(PersistenceInterval(
                            birth=birth,
                            death=eps,
                            dimension=1
                        ))
            
            prev_beta1 = beta1
        
        # Close remaining loops at max scale
        for birth in pending_loops:
            intervals.append(PersistenceInterval(
                birth=birth,
                death=scales[-1],
                dimension=1
            ))
        
        return intervals
    
    def _compute_summary(
        self,
        intervals: List[PersistenceInterval],
        ref_scale: float,
        max_scale: float,
    ) -> TopologicalSummary:
        """Compute summary statistics from persistence intervals."""
        summary = TopologicalSummary(scale=ref_scale)
        
        # Separate by dimension
        intervals_0 = [i for i in intervals if i.dimension == 0]
        intervals_1 = [i for i in intervals if i.dimension == 1]
        
        # Betti numbers at reference scale
        summary.beta_0 = sum(1 for i in intervals_0 
                           if i.birth <= ref_scale < i.death)
        summary.beta_1 = sum(1 for i in intervals_1 
                           if i.birth <= ref_scale < i.death)
        
        # Persistence statistics for H₀
        if intervals_0:
            persistences_0 = [i.persistence for i in intervals_0]
            summary.total_persistence_0 = sum(persistences_0)
            summary.max_persistence_0 = max(persistences_0)
            threshold = 0.1 * max_scale
            summary.n_significant_0 = sum(1 for p in persistences_0 if p > threshold)
        
        # Persistence statistics for H₁
        if intervals_1:
            persistences_1 = [i.persistence for i in intervals_1]
            summary.total_persistence_1 = sum(persistences_1)
            summary.max_persistence_1 = max(persistences_1)
            threshold = 0.1 * max_scale
            summary.n_significant_1 = sum(1 for p in persistences_1 if p > threshold)
        
        # Persistence entropy
        all_pers = [i.persistence for i in intervals if i.persistence > 0]
        if all_pers:
            total = sum(all_pers)
            probs = [p / total for p in all_pers]
            summary.persistence_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        
        return summary


class SemanticTDA:
    """
    TDA analysis specialized for semantic trajectory.
    
    Analyzes the observer's learning trajectory in semantic space
    to characterize understanding structure and convergence.
    """
    
    def __init__(
        self,
        n_scales: int = 50,
        significance_threshold: float = 0.1,
    ):
        """
        Initialize semantic TDA analyzer.
        
        Args:
            n_scales: Number of filtration scales
            significance_threshold: Threshold for significant features (relative to max scale)
        """
        self.homology = PersistentHomology(max_dimension=1, n_scales=n_scales)
        self.significance_threshold = significance_threshold
    
    def analyze(
        self,
        history: "SemanticHistory",
        normalize: bool = True,
    ) -> Tuple[TopologicalSummary, Dict[str, Any]]:
        """
        Analyze semantic trajectory topology.
        
        Args:
            history: SemanticHistory with trajectory vectors
            normalize: Whether to normalize point cloud
            
        Returns:
            Tuple of (summary, detailed_analysis)
        """
        # Get vectors from history
        vectors = history.get_vectors_array()
        
        if len(vectors) < 3:
            return TopologicalSummary(), {"error": "Insufficient data"}
        
        # Normalize if requested
        if normalize:
            mean = np.mean(vectors, axis=0)
            std = np.std(vectors, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            vectors = (vectors - mean) / std
        
        # Compute persistent homology
        intervals, summary = self.homology.compute(vectors)
        
        # Additional analysis
        analysis = self._interpret_topology(summary, vectors)
        analysis["n_points"] = len(vectors)
        analysis["n_dimensions"] = vectors.shape[1] if len(vectors.shape) > 1 else 1
        analysis["intervals"] = [(i.birth, i.death, i.dimension) for i in intervals[:20]]  # Top 20
        
        return summary, analysis
    
    def _interpret_topology(
        self,
        summary: TopologicalSummary,
        vectors: np.ndarray,
    ) -> Dict[str, Any]:
        """Interpret topological features for semantic meaning."""
        interpretation = {}
        
        # β₀ interpretation
        if summary.beta_0 == 1:
            interpretation["connectivity"] = "unified"
            interpretation["connectivity_desc"] = "Single connected understanding"
        elif summary.beta_0 <= 3:
            interpretation["connectivity"] = "clustered"
            interpretation["connectivity_desc"] = f"{summary.beta_0} concept clusters"
        else:
            interpretation["connectivity"] = "fragmented"
            interpretation["connectivity_desc"] = f"Fragmented into {summary.beta_0} components"
        
        # β₁ interpretation
        if summary.beta_1 == 0:
            interpretation["structure"] = "tree-like"
            interpretation["structure_desc"] = "Hierarchical, no cyclic patterns"
        elif summary.beta_1 <= 2:
            interpretation["structure"] = "simple_loops"
            interpretation["structure_desc"] = f"{summary.beta_1} recurring pattern(s)"
        else:
            interpretation["structure"] = "complex"
            interpretation["structure_desc"] = f"Complex structure with {summary.beta_1} loops"
        
        # Convergence assessment
        if summary.n_significant_0 <= 1 and summary.n_significant_1 == 0:
            interpretation["convergence"] = "converged"
        elif summary.total_persistence_0 > summary.total_persistence_1:
            interpretation["convergence"] = "converging"
        else:
            interpretation["convergence"] = "exploring"
        
        # Complexity score
        interpretation["complexity_score"] = (
            0.3 * summary.beta_0 +
            0.5 * summary.beta_1 +
            0.2 * summary.persistence_entropy
        )
        
        return interpretation
    
    def compare_trajectories(
        self,
        histories: List["SemanticHistory"],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple semantic trajectories.
        
        Useful for comparing observers with different configurations
        or rule sets.
        
        Args:
            histories: List of SemanticHistory objects
            labels: Optional labels for each trajectory
            
        Returns:
            Comparison results
        """
        if labels is None:
            labels = [f"traj_{i}" for i in range(len(histories))]
        
        results = {}
        for label, history in zip(labels, histories):
            summary, analysis = self.analyze(history)
            results[label] = {
                "summary": summary.to_dict(),
                "analysis": analysis,
            }
        
        # Find best by various criteria
        comparisons = {}
        
        # Most unified (lowest β₀)
        min_beta0 = min(results[l]["summary"]["beta_0"] for l in labels)
        comparisons["most_unified"] = [l for l in labels 
                                       if results[l]["summary"]["beta_0"] == min_beta0]
        
        # Simplest (lowest complexity)
        complexities = {l: results[l]["analysis"].get("complexity_score", float('inf')) 
                       for l in labels}
        min_complexity = min(complexities.values())
        comparisons["simplest"] = [l for l in labels 
                                  if complexities[l] == min_complexity]
        
        # Most converged (lowest total persistence)
        persistences = {l: results[l]["summary"]["total_persistence_0"] 
                       for l in labels}
        min_pers = min(persistences.values())
        comparisons["most_converged"] = [l for l in labels 
                                        if persistences[l] == min_pers]
        
        return {
            "trajectories": results,
            "comparisons": comparisons,
        }


def analyze_semantic_trajectory(
    history: "SemanticHistory",
    n_scales: int = 50,
) -> Dict[str, Any]:
    """
    Convenience function for TDA analysis of semantic trajectory.
    
    Args:
        history: SemanticHistory to analyze
        n_scales: Number of filtration scales
        
    Returns:
        Analysis results dictionary
    """
    analyzer = SemanticTDA(n_scales=n_scales)
    summary, analysis = analyzer.analyze(history)
    
    return {
        "summary": summary.to_dict(),
        "interpretation": analysis,
        "beta_0": summary.beta_0,
        "beta_1": summary.beta_1,
    }
