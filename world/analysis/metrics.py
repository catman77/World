"""
Metrics and distance measures for RSL World.

Provides various metrics for comparing lattice states:
- Hamming distance
- Edit distance
- Correlation-based distances
- Information-theoretic measures
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


def hamming_distance(s1: np.ndarray, s2: np.ndarray) -> int:
    """
    Compute Hamming distance between two states.
    
    Number of positions where symbols differ.
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    return int(np.sum(s1 != s2))


def normalized_hamming(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute normalized Hamming distance in [0, 1].
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    N = min(len(s1), len(s2))
    return hamming_distance(s1[:N], s2[:N]) / N if N > 0 else 0.0


def edit_distance(s1: np.ndarray, s2: np.ndarray) -> int:
    """
    Compute edit (Levenshtein) distance between two states.
    
    Minimum number of insertions, deletions, substitutions
    to transform s1 into s2.
    
    Note: For same-length lattices, this equals Hamming distance.
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    m, n = len(s1), len(s2)
    
    # Same length: use Hamming
    if m == n:
        return hamming_distance(s1, s2)
    
    # DP for edit distance
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    return int(dp[m, n])


def overlap(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute overlap between states.
    
    O = (1/N) * Σ s1_i * s2_i
    
    Returns 1.0 for identical, -1.0 for opposite, 0 for uncorrelated.
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    N = min(len(s1), len(s2))
    return float(np.mean(s1[:N] * s2[:N]))


def mutual_information(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute mutual information between two states.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    For binary states, computes based on joint distribution.
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    N = min(len(s1), len(s2))
    s1, s2 = s1[:N], s2[:N]
    
    # Joint distribution
    pp = np.sum((s1 == 1) & (s2 == 1)) / N  # P(+,+)
    pm = np.sum((s1 == 1) & (s2 == -1)) / N  # P(+,-)
    mp = np.sum((s1 == -1) & (s2 == 1)) / N  # P(-,+)
    mm = np.sum((s1 == -1) & (s2 == -1)) / N  # P(-,-)
    
    # Marginals
    p1_plus = np.sum(s1 == 1) / N
    p1_minus = 1 - p1_plus
    p2_plus = np.sum(s2 == 1) / N
    p2_minus = 1 - p2_plus
    
    # Entropies
    def entropy(ps):
        return -sum(p * np.log(p + 1e-10) for p in ps if p > 0)
    
    H1 = entropy([p1_plus, p1_minus])
    H2 = entropy([p2_plus, p2_minus])
    H12 = entropy([pp, pm, mp, mm])
    
    return H1 + H2 - H12


def jensen_shannon_divergence(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between states.
    
    A symmetric measure based on KL divergence.
    """
    if hasattr(s1, 'sites'):
        s1 = s1.sites
    if hasattr(s2, 'sites'):
        s2 = s2.sites
    
    # Convert to probability distributions
    # (histogram over coarse-grained values)
    def to_prob(s, bins=10):
        hist, _ = np.histogram(s, bins=bins, range=(-1, 1))
        return hist / np.sum(hist) + 1e-10  # Avoid zeros
    
    p = to_prob(s1)
    q = to_prob(s2)
    m = (p + q) / 2
    
    def kl_div(p, q):
        return np.sum(p * np.log(p / q))
    
    return (kl_div(p, m) + kl_div(q, m)) / 2


class MetricsCalculator:
    """
    Calculator for various metrics between lattice states.
    
    Provides:
    - Distance metrics (Hamming, edit, correlation)
    - Similarity metrics (overlap, mutual information)
    - Distance matrices for state collections
    
    Example:
        calc = MetricsCalculator()
        
        # Distance between two states
        d = calc.distance(state1, state2, metric="hamming")
        
        # Distance matrix for history
        D = calc.distance_matrix(history)
    """
    
    def __init__(self, default_metric: str = "hamming"):
        self.default_metric = default_metric
        
        self._metrics = {
            "hamming": hamming_distance,
            "normalized_hamming": normalized_hamming,
            "edit": edit_distance,
            "overlap": lambda s1, s2: 1 - (overlap(s1, s2) + 1) / 2,
            "mutual_information": lambda s1, s2: -mutual_information(s1, s2),
            "jensen_shannon": jensen_shannon_divergence,
        }
    
    def distance(
        self, 
        s1: np.ndarray, 
        s2: np.ndarray,
        metric: Optional[str] = None,
    ) -> float:
        """
        Compute distance between two states.
        
        Args:
            s1, s2: States to compare
            metric: Distance metric to use
            
        Returns:
            Distance value
        """
        metric = metric or self.default_metric
        
        if metric not in self._metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(self._metrics[metric](s1, s2))
    
    def distance_matrix(
        self,
        states: List[np.ndarray],
        metric: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix for list of states.
        
        Returns NxN symmetric matrix where D[i,j] = distance(states[i], states[j]).
        """
        N = len(states)
        D = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                d = self.distance(states[i], states[j], metric)
                D[i, j] = d
                D[j, i] = d
        
        return D
    
    def nearest_neighbors(
        self,
        state: np.ndarray,
        states: List[np.ndarray],
        k: int = 5,
        metric: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to state in collection.
        
        Returns list of (index, distance) pairs.
        """
        distances = [(i, self.distance(state, s, metric)) for i, s in enumerate(states)]
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def centroid(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid (mean) state.
        
        For binary states, returns continuous values that can be
        thresholded to get a representative binary state.
        """
        states_arr = [s.sites if hasattr(s, 'sites') else s for s in states]
        return np.mean(states_arr, axis=0)
    
    def dispersion(
        self,
        states: List[np.ndarray],
        metric: Optional[str] = None,
    ) -> float:
        """
        Compute dispersion (average distance from centroid).
        """
        if not states:
            return 0.0
        
        centroid = self.centroid(states)
        
        # For binary metrics, threshold centroid
        if metric in ["hamming", "edit"]:
            centroid = np.sign(centroid)
            centroid[centroid == 0] = 1  # Arbitrary choice for ties
        
        total = sum(self.distance(s, centroid, metric) for s in states)
        return total / len(states)
    
    def clustering_coefficient(
        self,
        D: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        Compute clustering coefficient from distance matrix.
        
        Based on graph where edges connect states with distance < threshold.
        """
        N = len(D)
        
        # Build adjacency from distance matrix
        A = (D < threshold).astype(int)
        np.fill_diagonal(A, 0)
        
        # Count triangles
        triangles = 0
        triples = 0
        
        for i in range(N):
            neighbors = np.where(A[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            
            triples += k * (k - 1) / 2
            
            # Count edges among neighbors
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if A[neighbors[j], neighbors[l]] > 0:
                        triangles += 1
        
        return triangles / triples if triples > 0 else 0.0


class RecurrenceAnalyzer:
    """
    Recurrence analysis for lattice evolution.
    
    Builds recurrence plots and computes recurrence quantifiers.
    """
    
    def __init__(self, threshold: float = 0.1, metric: str = "normalized_hamming"):
        self.threshold = threshold
        self.metric = metric
        self._calculator = MetricsCalculator(metric)
    
    def recurrence_matrix(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Compute recurrence matrix.
        
        R[i,j] = 1 if distance(states[i], states[j]) < threshold
        """
        D = self._calculator.distance_matrix(states, self.metric)
        return (D < self.threshold).astype(int)
    
    def recurrence_rate(self, R: np.ndarray) -> float:
        """
        Compute recurrence rate.
        
        RR = (1/N²) * Σ R[i,j]
        """
        N = len(R)
        return np.sum(R) / (N * N)
    
    def determinism(self, R: np.ndarray, min_length: int = 2) -> float:
        """
        Compute determinism (fraction of recurrence points forming diagonals).
        
        DET = (points in diagonals of length >= min_length) / (total recurrence points)
        """
        N = len(R)
        total = np.sum(R)
        
        if total == 0:
            return 0.0
        
        diagonal_points = 0
        
        # Count diagonal lines
        for offset in range(-(N-1), N):
            diag = np.diag(R, offset)
            # Find runs of 1s
            run_length = 0
            for val in diag:
                if val == 1:
                    run_length += 1
                else:
                    if run_length >= min_length:
                        diagonal_points += run_length
                    run_length = 0
            if run_length >= min_length:
                diagonal_points += run_length
        
        return diagonal_points / total
    
    def laminarity(self, R: np.ndarray, min_length: int = 2) -> float:
        """
        Compute laminarity (fraction of recurrence points forming vertical lines).
        
        LAM = (points in vertical lines of length >= min_length) / (total recurrence points)
        """
        N = len(R)
        total = np.sum(R)
        
        if total == 0:
            return 0.0
        
        vertical_points = 0
        
        for j in range(N):
            col = R[:, j]
            run_length = 0
            for val in col:
                if val == 1:
                    run_length += 1
                else:
                    if run_length >= min_length:
                        vertical_points += run_length
                    run_length = 0
            if run_length >= min_length:
                vertical_points += run_length
        
        return vertical_points / total
