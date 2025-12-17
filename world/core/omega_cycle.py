"""
Ω-cycle Detector and Resonance Triggers.

Implements deterministic activation conditions for wormholes
based on Ω-cycle resonance patterns in spin lattice.

Key concepts:
- Ω-cycles: Periodic/quasi-periodic patterns in spin strings
- Resonance: Correlation between Ω-signatures at distant nodes
- Deterministic triggers: NO randomness, only structural conditions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
import numpy as np
from collections import deque

if TYPE_CHECKING:
    from .world import World


@dataclass
class OmegaCycleConfig:
    """Configuration for Ω-cycle detection."""
    
    # Pattern detection
    min_pattern_length: int = 3          # Minimum L for pattern
    max_pattern_length: int = 12         # Maximum L for pattern search
    
    # Resonance thresholds
    resonance_threshold: float = 0.7     # Min correlation for "resonance"
    phase_tolerance: float = 0.2         # Phase alignment tolerance
    
    # Domain wall detection
    domain_wall_penalty: float = 0.1     # Penalty per domain wall mismatch
    
    # Spectral analysis
    use_fft: bool = True                 # Use FFT for period detection
    fft_significance: float = 0.1        # Min spectral peak significance


@dataclass
class OmegaSignature:
    """
    Ω-signature of a local spin region.
    
    Encodes the periodic structure / pattern at a given location.
    """
    center: int
    window_size: int
    
    # Pattern data
    dominant_period: int = 0             # Primary period (0 if aperiodic)
    spectral_peaks: List[Tuple[int, float]] = field(default_factory=list)
    pattern_entropy: float = 0.0         # Shannon entropy of pattern
    domain_wall_density: float = 0.0     # Density of spin transitions
    
    # Extracted pattern (if periodic)
    pattern: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def is_periodic(self) -> bool:
        """Check if this signature represents a periodic pattern."""
        return self.dominant_period > 0 and self.dominant_period < self.window_size // 2


class OmegaCycleDetector:
    """
    Detector for Ω-cycles in spin lattice.
    
    An Ω-cycle is a repeating structural pattern in the spin string
    that defines a "resonance signature" for that region.
    
    Usage:
        detector = OmegaCycleDetector(world, OmegaCycleConfig())
        sig_i = detector.extract_signature(node_i)
        sig_j = detector.extract_signature(node_j)
        resonance = detector.compute_resonance(sig_i, sig_j)
    """
    
    def __init__(self, world: "World", config: Optional[OmegaCycleConfig] = None):
        self.world = world
        self.config = config or OmegaCycleConfig()
        
        # Cache for signatures (cleared on world update)
        self._cache: Dict[int, OmegaSignature] = {}
        self._cache_t: int = -1
    
    def _invalidate_cache(self):
        """Invalidate cache if world time changed."""
        if self.world.t != self._cache_t:
            self._cache.clear()
            self._cache_t = self.world.t
    
    def extract_signature(self, center: int, window_size: Optional[int] = None) -> OmegaSignature:
        """
        Extract Ω-signature at a given center node.
        
        Args:
            center: Center node index
            window_size: Window size (default: 2 * max_pattern_length)
            
        Returns:
            OmegaSignature for this location
        """
        self._invalidate_cache()
        
        if center in self._cache:
            return self._cache[center]
        
        if window_size is None:
            window_size = 2 * self.config.max_pattern_length
        
        s = self.world.s
        N = len(s)
        
        # Extract local region
        start = max(0, center - window_size // 2)
        end = min(N, center + window_size // 2 + 1)
        region = s[start:end].astype(float)
        
        sig = OmegaSignature(center=center, window_size=window_size)
        
        if len(region) < self.config.min_pattern_length:
            self._cache[center] = sig
            return sig
        
        # Compute domain wall density
        transitions = np.sum(np.abs(np.diff(region)))
        sig.domain_wall_density = transitions / (len(region) - 1)
        
        # Compute pattern entropy (via histogram)
        unique, counts = np.unique(region, return_counts=True)
        probs = counts / counts.sum()
        sig.pattern_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Find dominant period via autocorrelation or FFT
        if self.config.use_fft and len(region) >= 8:
            sig.dominant_period, sig.spectral_peaks = self._fft_period_detection(region)
        else:
            sig.dominant_period = self._autocorr_period_detection(region)
        
        # Extract pattern if periodic
        if sig.dominant_period > 0 and sig.dominant_period <= len(region) // 2:
            pattern_len = sig.dominant_period
            sig.pattern = region[:pattern_len].copy()
        
        self._cache[center] = sig
        return sig
    
    def _fft_period_detection(self, region: np.ndarray) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Detect dominant period using FFT.
        
        Returns:
            (dominant_period, list of (period, magnitude) peaks)
        """
        n = len(region)
        
        # Zero-mean for better spectral analysis
        region_centered = region - np.mean(region)
        
        # FFT
        fft = np.fft.rfft(region_centered)
        magnitudes = np.abs(fft)
        
        # Find peaks (excluding DC component)
        if len(magnitudes) < 2:
            return 0, []
        
        # Frequencies to periods
        freqs = np.fft.rfftfreq(n)
        
        # Find significant peaks
        max_mag = np.max(magnitudes[1:]) if len(magnitudes) > 1 else 0
        if max_mag < 1e-10:
            return 0, []
        
        threshold = self.config.fft_significance * max_mag
        peaks = []
        
        for i in range(1, len(magnitudes)):
            if magnitudes[i] > threshold:
                if freqs[i] > 0:
                    period = int(round(1.0 / freqs[i]))
                    if self.config.min_pattern_length <= period <= self.config.max_pattern_length:
                        peaks.append((period, magnitudes[i]))
        
        # Sort by magnitude
        peaks.sort(key=lambda x: -x[1])
        
        dominant_period = peaks[0][0] if peaks else 0
        return dominant_period, peaks[:5]  # Top 5 peaks
    
    def _autocorr_period_detection(self, region: np.ndarray) -> int:
        """
        Detect dominant period using autocorrelation.
        
        Returns:
            Dominant period (0 if no clear period)
        """
        n = len(region)
        region_centered = region - np.mean(region)
        
        # Autocorrelation
        autocorr = np.correlate(region_centered, region_centered, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Find first significant peak after lag 0
        for lag in range(self.config.min_pattern_length, min(len(autocorr), self.config.max_pattern_length + 1)):
            if autocorr[lag] > 0.5:  # High correlation at this lag
                # Check if it's a local maximum
                if lag > 0 and lag < len(autocorr) - 1:
                    if autocorr[lag] >= autocorr[lag-1] and autocorr[lag] >= autocorr[lag+1]:
                        return lag
        
        return 0
    
    def compute_resonance(self, sig_i: OmegaSignature, sig_j: OmegaSignature) -> float:
        """
        Compute resonance between two Ω-signatures.
        
        Resonance is high when:
        - Both have similar dominant periods
        - Patterns are correlated (phase-aligned or anti-aligned)
        - Domain wall densities are similar
        
        Returns:
            Resonance value in [0, 1]
        """
        # Period match component
        period_match = 0.0
        if sig_i.dominant_period > 0 and sig_j.dominant_period > 0:
            ratio = min(sig_i.dominant_period, sig_j.dominant_period) / max(sig_i.dominant_period, sig_j.dominant_period)
            # Harmonic matching: periods 2:1, 3:2, etc. also count
            if ratio > 0.9:
                period_match = 1.0
            elif ratio > 0.45 and ratio < 0.55:  # 1:2 ratio
                period_match = 0.7
            elif ratio > 0.3 and ratio < 0.4:    # 1:3 ratio
                period_match = 0.5
        
        # Pattern correlation (if both have patterns)
        pattern_corr = 0.0
        if len(sig_i.pattern) > 0 and len(sig_j.pattern) > 0:
            # Align patterns to same length
            min_len = min(len(sig_i.pattern), len(sig_j.pattern))
            pi = sig_i.pattern[:min_len]
            pj = sig_j.pattern[:min_len]
            
            # Try both alignments (same phase and anti-phase)
            std_i = np.std(pi)
            std_j = np.std(pj)
            if std_i > 1e-10 and std_j > 1e-10:
                corr = np.corrcoef(pi, pj)[0, 1]
                if not np.isnan(corr):
                    pattern_corr = abs(corr)  # Accept anti-correlation too
        
        # Domain wall density similarity
        dw_diff = abs(sig_i.domain_wall_density - sig_j.domain_wall_density)
        dw_match = max(0, 1.0 - 2.0 * dw_diff)
        
        # Entropy similarity (similar complexity)
        ent_diff = abs(sig_i.pattern_entropy - sig_j.pattern_entropy)
        ent_match = max(0, 1.0 - ent_diff)
        
        # Combine components
        # Weights: period is most important for resonance
        resonance = (
            0.4 * period_match +
            0.3 * pattern_corr +
            0.2 * dw_match +
            0.1 * ent_match
        )
        
        return resonance
    
    def find_resonant_pairs(
        self, 
        min_distance: int = 50,
        max_pairs: int = 10,
        force_recompute: bool = False
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of nodes with high Ω-resonance.
        
        This is DETERMINISTIC: same world state → same pairs.
        
        Args:
            min_distance: Minimum lattice distance between nodes
            max_pairs: Maximum number of pairs to return
            force_recompute: Recompute all signatures
            
        Returns:
            List of (node_i, node_j, resonance) tuples, sorted by resonance
        """
        if force_recompute:
            self._cache.clear()
        
        N = self.world.N
        stride = max(1, N // 50)  # Sample ~50 points
        
        # Extract signatures at sample points
        sample_nodes = list(range(0, N, stride))
        signatures = {i: self.extract_signature(i) for i in sample_nodes}
        
        # Find resonant pairs
        pairs = []
        for i, idx_i in enumerate(sample_nodes):
            for idx_j in sample_nodes[i+1:]:
                # Check minimum distance
                if abs(idx_j - idx_i) < min_distance:
                    continue
                
                res = self.compute_resonance(signatures[idx_i], signatures[idx_j])
                if res >= self.config.resonance_threshold:
                    pairs.append((idx_i, idx_j, res))
        
        # Sort by resonance (descending) and limit
        pairs.sort(key=lambda x: -x[2])
        return pairs[:max_pairs]


@dataclass 
class ResonanceTrigger:
    """
    Deterministic trigger for wormhole activation based on Ω-resonance.
    
    Checks activation conditions without any randomness.
    """
    detector: OmegaCycleDetector
    
    # Trigger thresholds
    resonance_min: float = 0.7
    period_match_required: bool = True
    min_meaning_density: float = 0.2
    
    def check_trigger(self, node_i: int, node_j: int) -> dict:
        """
        Check if trigger conditions are met for nodes i,j.
        
        Returns:
            Dictionary with trigger decision and diagnostics
        """
        result = {
            'triggered': False,
            'resonance': 0.0,
            'period_i': 0,
            'period_j': 0,
            'meaning_density': 0.0,
            'reasons': []
        }
        
        # Extract signatures
        sig_i = self.detector.extract_signature(node_i)
        sig_j = self.detector.extract_signature(node_j)
        
        result['period_i'] = sig_i.dominant_period
        result['period_j'] = sig_j.dominant_period
        
        # Compute resonance
        resonance = self.detector.compute_resonance(sig_i, sig_j)
        result['resonance'] = resonance
        
        # Check resonance threshold
        if resonance < self.resonance_min:
            result['reasons'].append(f'low_resonance ({resonance:.2f} < {self.resonance_min})')
            return result
        
        # Check period match if required
        if self.period_match_required:
            if sig_i.dominant_period == 0 or sig_j.dominant_period == 0:
                result['reasons'].append('no_periodic_structure')
                return result
            
            ratio = min(sig_i.dominant_period, sig_j.dominant_period) / max(sig_i.dominant_period, sig_j.dominant_period)
            if ratio < 0.45:  # Allow 1:1 and 1:2 harmonics
                result['reasons'].append(f'period_mismatch (ratio={ratio:.2f})')
                return result
        
        # Check meaning density (average domain wall density)
        meaning = (sig_i.domain_wall_density + sig_j.domain_wall_density) / 2
        result['meaning_density'] = meaning
        
        if meaning < self.min_meaning_density:
            result['reasons'].append(f'low_meaning_density ({meaning:.2f} < {self.min_meaning_density})')
            return result
        
        # All conditions met
        result['triggered'] = True
        return result
    
    def find_active_triggers(self, candidate_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, dict]]:
        """
        Check triggers for all candidate pairs.
        
        Returns:
            List of (i, j, trigger_info) for triggered pairs
        """
        active = []
        for i, j in candidate_pairs:
            info = self.check_trigger(i, j)
            if info['triggered']:
                active.append((i, j, info))
        return active


def create_omega_detector(world: "World", **kwargs) -> OmegaCycleDetector:
    """Create Ω-cycle detector with custom configuration."""
    config = OmegaCycleConfig(**kwargs)
    return OmegaCycleDetector(world, config)


def create_resonance_trigger(
    world: "World",
    resonance_min: float = 0.7,
    period_match_required: bool = True,
    min_meaning_density: float = 0.2
) -> ResonanceTrigger:
    """Create resonance trigger for wormhole activation."""
    detector = create_omega_detector(world)
    return ResonanceTrigger(
        detector=detector,
        resonance_min=resonance_min,
        period_match_required=period_match_required,
        min_meaning_density=min_meaning_density
    )
