"""
Phase diagram visualization.
"""

from __future__ import annotations
from typing import Optional, Tuple, Any, Dict, List
import numpy as np

_plt = None

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def plot_phase_diagram(
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    order_param: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Phase Diagram",
    xlabel: str = "Parameter 1",
    ylabel: str = "Parameter 2",
    cmap: str = "viridis",
) -> Any:
    """
    Plot 2D phase diagram.
    
    Args:
        param1_values: Values for parameter 1 (x-axis)
        param2_values: Values for parameter 2 (y-axis)
        order_param: 2D array of order parameter values
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    extent = [
        param1_values.min(), param1_values.max(),
        param2_values.min(), param2_values.max(),
    ]
    
    im = ax.imshow(order_param.T, origin='lower', aspect='auto',
                   extent=extent, cmap=cmap)
    
    plt.colorbar(im, ax=ax, label='Order Parameter')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def plot_phase_boundaries(
    param1_values: np.ndarray,
    param2_values: np.ndarray,
    susceptibility: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Phase Boundaries",
    threshold: float = None,
) -> Any:
    """
    Plot phase boundaries from susceptibility peaks.
    
    Args:
        param1_values: Values for parameter 1
        param2_values: Values for parameter 2
        susceptibility: 2D susceptibility array
        ax: Matplotlib axis
        title: Plot title
        threshold: Peak detection threshold
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    extent = [
        param1_values.min(), param1_values.max(),
        param2_values.min(), param2_values.max(),
    ]
    
    # Show susceptibility
    im = ax.imshow(susceptibility.T, origin='lower', aspect='auto',
                   extent=extent, cmap='hot')
    plt.colorbar(im, ax=ax, label='Susceptibility')
    
    # Find boundaries (peaks in susceptibility)
    if threshold is None:
        threshold = susceptibility.mean() + 2 * susceptibility.std()
    
    # Contour at threshold
    ax.contour(param1_values, param2_values, susceptibility.T,
               levels=[threshold], colors='white', linewidths=2)
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_title(title)
    
    return ax


def plot_order_parameter_scaling(
    L_values: np.ndarray,
    order_params: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Finite Size Scaling",
    fit: bool = True,
) -> Any:
    """
    Plot finite size scaling of order parameter.
    
    Args:
        L_values: System sizes
        order_params: Order parameter values for each size
        ax: Matplotlib axis
        title: Plot title
        fit: Show power law fit
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(L_values, order_params, 'bo-', markersize=8, label='Data')
    
    if fit and len(L_values) > 2:
        # Power law fit: phi ~ L^(-beta/nu)
        log_L = np.log(L_values)
        log_phi = np.log(order_params)
        
        mask = np.isfinite(log_phi)
        if np.sum(mask) > 2:
            slope, intercept = np.polyfit(log_L[mask], log_phi[mask], 1)
            fit_phi = np.exp(intercept + slope * log_L)
            ax.loglog(L_values, fit_phi, 'r--', 
                     label=f'Fit: φ ~ L^{{{slope:.3f}}}')
    
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Order Parameter φ')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_susceptibility_scaling(
    L_values: np.ndarray,
    susceptibilities: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Susceptibility Scaling",
) -> Any:
    """
    Plot susceptibility vs system size.
    
    Args:
        L_values: System sizes
        susceptibilities: Susceptibility values
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(L_values, susceptibilities, 'ro-', markersize=8, label='Data')
    
    # Power law fit: χ ~ L^(γ/ν)
    if len(L_values) > 2:
        log_L = np.log(L_values)
        log_chi = np.log(susceptibilities)
        
        mask = np.isfinite(log_chi)
        if np.sum(mask) > 2:
            slope, intercept = np.polyfit(log_L[mask], log_chi[mask], 1)
            fit_chi = np.exp(intercept + slope * log_L)
            ax.loglog(L_values, fit_chi, 'b--', 
                     label=f'Fit: χ ~ L^{{{slope:.3f}}}')
    
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Susceptibility χ')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_binder_cumulant(
    param_values: np.ndarray,
    binder: Dict[int, np.ndarray],
    ax: Optional[Any] = None,
    title: str = "Binder Cumulant",
) -> Any:
    """
    Plot Binder cumulant vs parameter for different sizes.
    
    Args:
        param_values: Control parameter values
        binder: Dict mapping size -> Binder cumulant array
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    for L, U in sorted(binder.items()):
        ax.plot(param_values, U, '-o', markersize=4, label=f'L={L}')
    
    ax.set_xlabel('Control Parameter')
    ax.set_ylabel('Binder Cumulant U')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_correlation_length(
    param_values: np.ndarray,
    xi: np.ndarray,
    L: int,
    ax: Optional[Any] = None,
    title: str = "Correlation Length",
) -> Any:
    """
    Plot correlation length vs parameter.
    
    Args:
        param_values: Control parameter values
        xi: Correlation length values
        L: System size
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogy(param_values, xi, 'b-o', markersize=4)
    ax.axhline(L / 2, color='red', linestyle='--', 
               alpha=0.5, label=f'L/2 = {L/2}')
    
    ax.set_xlabel('Control Parameter')
    ax.set_ylabel('Correlation Length ξ')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_critical_exponents_summary(
    exponents: Dict[str, float],
    ax: Optional[Any] = None,
    title: str = "Critical Exponents",
) -> Any:
    """
    Plot summary of critical exponents.
    
    Args:
        exponents: Dict of exponent name -> value
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(exponents.keys())
    values = list(exponents.values())
    
    # Known values for 1D Ising
    known_1d = {'beta': 0, 'gamma': None, 'nu': None, 'alpha': None, 'eta': 1}
    # Known values for 2D Ising
    known_2d = {'beta': 1/8, 'gamma': 7/4, 'nu': 1, 'alpha': 0, 'eta': 1/4}
    
    x = np.arange(len(names))
    width = 0.25
    
    ax.bar(x - width, values, width, label='Measured', color='blue', alpha=0.7)
    
    # Compare with known if available
    if any(n in known_1d for n in names):
        known_vals = [known_1d.get(n, None) for n in names]
        valid = [v if v is not None else 0 for v in known_vals]
        ax.bar(x, valid, width, label='1D Ising', color='orange', alpha=0.7)
    
    if any(n in known_2d for n in names):
        known_vals = [known_2d.get(n, None) for n in names]
        valid = [v if v is not None else 0 for v in known_vals]
        ax.bar(x + width, valid, width, label='2D Ising', color='green', alpha=0.7)
    
    ax.set_xlabel('Exponent')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_phase_diagram_full(
    results: Dict,
    figsize: Tuple[int, int] = (16, 10),
) -> Any:
    """
    Create comprehensive phase diagram figure.
    
    Args:
        results: Dict with keys:
            - 'param1': parameter 1 values
            - 'param2': parameter 2 values  
            - 'order_param': 2D order parameter
            - 'susceptibility': 2D susceptibility
            - 'binder': Binder cumulant data
            
    Returns:
        Matplotlib figure
    """
    plt = _get_plt()
    
    fig = plt.figure(figsize=figsize)
    
    # Phase diagram
    ax1 = fig.add_subplot(2, 2, 1)
    if 'order_param' in results:
        plot_phase_diagram(
            results['param1'], results['param2'], results['order_param'],
            ax=ax1, title='Order Parameter'
        )
    
    # Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    if 'susceptibility' in results:
        plot_phase_boundaries(
            results['param1'], results['param2'], results['susceptibility'],
            ax=ax2, title='Susceptibility & Phase Boundaries'
        )
    
    # Binder cumulant
    ax3 = fig.add_subplot(2, 2, 3)
    if 'binder' in results:
        plot_binder_cumulant(
            results.get('param1', np.array([0])),
            results['binder'],
            ax=ax3
        )
    
    # Critical exponents
    ax4 = fig.add_subplot(2, 2, 4)
    if 'exponents' in results:
        plot_critical_exponents_summary(results['exponents'], ax=ax4)
    
    plt.tight_layout()
    return fig
