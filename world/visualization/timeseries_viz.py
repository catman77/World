"""
Time series visualization functions.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Dict
import numpy as np

_plt = None
def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def plot_timeseries(
    times: np.ndarray,
    values: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "",
    ylabel: str = "Value",
    color: str = "blue",
    **kwargs,
) -> Any:
    """
    Plot time series data.
    
    Args:
        times: Time values
        values: Observable values
        ax: Matplotlib axis
        title: Plot title
        ylabel: Y-axis label
        color: Line color
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(times, values, color=color, **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_multiple_timeseries(
    times: np.ndarray,
    series: Dict[str, np.ndarray],
    ax: Optional[Any] = None,
    title: str = "",
    **kwargs,
) -> Any:
    """
    Plot multiple time series on same axis.
    
    Args:
        times: Time values
        series: Dict mapping name -> values
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    for name, values in series.items():
        ax.plot(times, values, label=name, **kwargs)
    
    ax.set_xlabel('Time')
    ax.legend()
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_correlations(
    autocorr: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Autocorrelation Function",
    show_exponential_fit: bool = True,
) -> Any:
    """
    Plot autocorrelation function.
    
    Args:
        autocorr: Autocorrelation values C(0), C(1), ...
        ax: Matplotlib axis
        title: Plot title
        show_exponential_fit: Show exponential decay fit
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    lags = np.arange(len(autocorr))
    ax.plot(lags, autocorr, 'b-', linewidth=2, label='Data')
    
    # Show e^(-1) line for correlation time
    ax.axhline(np.exp(-1), color='gray', linestyle='--', alpha=0.5, 
               label=f'e⁻¹ = {np.exp(-1):.3f}')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Exponential fit
    if show_exponential_fit and len(autocorr) > 5:
        # Find correlation time (where C drops below 1/e)
        tau = None
        for i, c in enumerate(autocorr):
            if c < np.exp(-1):
                tau = i
                break
        
        if tau and tau > 0:
            fit = np.exp(-lags / tau)
            ax.plot(lags, fit, 'r--', alpha=0.5, label=f'Fit: τ = {tau}')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('C(τ)')
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_power_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Power Spectrum",
    log_scale: bool = True,
) -> Any:
    """
    Plot power spectrum.
    
    Args:
        freqs: Frequency values
        power: Power values
        ax: Matplotlib axis
        title: Plot title
        log_scale: Use log scale for y-axis
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter out zero frequency
    mask = freqs > 0
    f = freqs[mask]
    p = power[mask]
    
    if log_scale:
        ax.loglog(f, p, 'b-', linewidth=1)
    else:
        ax.plot(f, p, 'b-', linewidth=1)
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(title)
    
    return ax


def plot_histogram(
    data: np.ndarray,
    bins: int = 50,
    ax: Optional[Any] = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    density: bool = True,
    show_mean: bool = True,
) -> Any:
    """
    Plot histogram of data.
    
    Args:
        data: Data values
        bins: Number of bins
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        density: Normalize to probability density
        show_mean: Show mean line
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(data, bins=bins, density=density, alpha=0.7, edgecolor='black')
    
    if show_mean:
        mean = np.mean(data)
        ax.axvline(mean, color='red', linestyle='--', 
                   label=f'Mean = {mean:.4f}')
        ax.legend()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density' if density else 'Count')
    ax.set_title(title)
    
    return ax


def plot_evolution_summary(
    history: List,
    figsize: Tuple[int, int] = (14, 10),
) -> Any:
    """
    Create summary figure with multiple panels.
    
    Panels:
    1. Space-time diagram
    2. Magnetization time series
    3. Domain wall count
    4. Magnetization distribution
    
    Args:
        history: List of LatticeState objects
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    plt = _get_plt()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    states = []
    magnetizations = []
    domain_walls = []
    times = []
    
    for i, s in enumerate(history):
        if hasattr(s, 'sites'):
            sites = s.sites
            time = s.time if hasattr(s, 'time') else i
        elif hasattr(s, '_sites'):
            sites = s._sites
            time = s._time if hasattr(s, '_time') else i
        else:
            sites = np.asarray(s)
            time = i
        
        states.append(sites)
        magnetizations.append(np.mean(sites))
        domain_walls.append(np.sum(sites[:-1] != sites[1:]))
        times.append(time)
    
    times = np.array(times)
    
    # Panel 1: Space-time
    if states:
        data = np.vstack(states[:100])  # Limit to 100 steps
        axes[0, 0].imshow(data, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Time')
        axes[0, 0].set_title('Space-Time Diagram')
    
    # Panel 2: Magnetization
    axes[0, 1].plot(times, magnetizations, 'b-', linewidth=0.5)
    axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Magnetization')
    axes[0, 1].set_title('Magnetization vs Time')
    axes[0, 1].set_ylim(-1.1, 1.1)
    
    # Panel 3: Domain walls
    axes[1, 0].plot(times, domain_walls, 'r-', linewidth=0.5)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Domain Wall Count')
    axes[1, 0].set_title('Domain Walls vs Time')
    
    # Panel 4: Magnetization distribution
    axes[1, 1].hist(magnetizations, bins=50, density=True, alpha=0.7)
    axes[1, 1].set_xlabel('Magnetization')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Magnetization Distribution')
    
    plt.tight_layout()
    return fig
