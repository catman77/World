"""
Lattice visualization functions.

Provides visualizations for 1D lattice states and evolution.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any
import numpy as np

# Lazy import for matplotlib
_plt = None
_animation = None

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

def _get_animation():
    global _animation
    if _animation is None:
        from matplotlib import animation
        _animation = animation
    return _animation


def plot_lattice(
    sites: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "",
    cmap: str = "RdBu",
    show_walls: bool = True,
) -> Any:
    """
    Plot a single lattice state as a 1D strip.
    
    Args:
        sites: Array of site values or Lattice/LatticeState object
        ax: Matplotlib axis (created if None)
        title: Plot title
        cmap: Colormap for +/- sites
        show_walls: Mark domain wall positions
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if hasattr(sites, 'sites'):
        sites = sites.sites
    elif hasattr(sites, '_sites'):
        sites = sites._sites
    
    sites = np.asarray(sites)
    N = len(sites)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 1))
    
    # Plot as image
    data = sites.reshape(1, -1)
    ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1,
              extent=[0, N, 0, 1])
    
    # Mark domain walls
    if show_walls:
        for i in range(N - 1):
            if sites[i] != sites[i + 1]:
                ax.axvline(i + 0.5, color='black', linewidth=1, alpha=0.7)
    
    ax.set_xlim(0, N)
    ax.set_yticks([])
    ax.set_xlabel('Site')
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_lattice_evolution(
    history: List[np.ndarray],
    max_states: int = 100,
    ax: Optional[Any] = None,
    title: str = "Lattice Evolution",
    cmap: str = "RdBu",
    aspect: str = "auto",
) -> Any:
    """
    Plot space-time diagram of lattice evolution.
    
    Args:
        history: List of lattice states
        max_states: Maximum number of time steps to show
        ax: Matplotlib axis
        title: Plot title
        cmap: Colormap
        aspect: Aspect ratio ('auto' or 'equal')
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    # Extract site arrays
    states = []
    for s in history[:max_states]:
        if hasattr(s, 'sites'):
            states.append(s.sites)
        elif hasattr(s, '_sites'):
            states.append(s._sites)
        else:
            states.append(np.asarray(s))
    
    if not states:
        return None
    
    # Stack into 2D array (time x space)
    data = np.vstack(states)
    T, N = data.shape
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect=aspect, vmin=-1, vmax=1,
                   extent=[0, N, T, 0], origin='upper')
    
    ax.set_xlabel('Site')
    ax.set_ylabel('Time')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Site value')
    
    return ax


def plot_coarse_graining_levels(
    levels: List[np.ndarray],
    ax: Optional[Any] = None,
    title: str = "Coarse-Graining Hierarchy",
) -> Any:
    """
    Plot multiple coarse-graining levels side by side.
    
    Args:
        levels: List of coarse-grained states at different levels
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    n_levels = len(levels)
    
    if ax is None:
        fig, axes = plt.subplots(n_levels, 1, figsize=(12, 2 * n_levels))
        if n_levels == 1:
            axes = [axes]
    else:
        axes = [ax]
    
    for i, (level, ax) in enumerate(zip(levels, axes)):
        if hasattr(level, 'coarse_field'):
            data = level.coarse_field
        else:
            data = np.asarray(level)
        
        ax.imshow(data.reshape(1, -1), cmap='RdBu', aspect='auto', 
                  vmin=-1, vmax=1)
        ax.set_yticks([])
        ax.set_ylabel(f'L{i}')
        ax.set_xlabel('Position' if i == n_levels - 1 else '')
    
    if title:
        axes[0].set_title(title)
    
    plt.tight_layout()
    return axes


def plot_domain_walls(
    history: List[np.ndarray],
    ax: Optional[Any] = None,
    title: str = "Domain Wall Trajectories",
) -> Any:
    """
    Plot domain wall positions over time.
    
    Shows worldlines of domain walls as particles.
    
    Args:
        history: List of lattice states
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find domain walls at each time
    for t, state in enumerate(history):
        if hasattr(state, 'sites'):
            sites = state.sites
        elif hasattr(state, '_sites'):
            sites = state._sites
        else:
            sites = np.asarray(state)
        
        # Find wall positions
        for i in range(len(sites) - 1):
            if sites[i] != sites[i + 1]:
                # Determine wall type
                if sites[i] > sites[i + 1]:
                    color = 'red'
                    marker = '^'
                else:
                    color = 'blue'
                    marker = 'v'
                
                ax.scatter(i + 0.5, t, c=color, marker=marker, s=10, alpha=0.7)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Time')
    ax.set_title(title)
    ax.invert_yaxis()
    
    return ax


def animate_evolution(
    history: List[np.ndarray],
    interval: int = 100,
    cmap: str = "RdBu",
    save_path: Optional[str] = None,
) -> Any:
    """
    Create animation of lattice evolution.
    
    Args:
        history: List of lattice states
        interval: Milliseconds between frames
        cmap: Colormap
        save_path: Optional path to save animation
        
    Returns:
        Matplotlib animation object
    """
    plt = _get_plt()
    animation = _get_animation()
    
    # Extract site arrays
    states = []
    for s in history:
        if hasattr(s, 'sites'):
            states.append(s.sites)
        elif hasattr(s, '_sites'):
            states.append(s._sites)
        else:
            states.append(np.asarray(s))
    
    if not states:
        return None
    
    N = len(states[0])
    
    fig, ax = plt.subplots(figsize=(12, 2))
    
    data = states[0].reshape(1, -1)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1,
                   extent=[0, N, 0, 1])
    
    ax.set_xlim(0, N)
    ax.set_yticks([])
    ax.set_xlabel('Site')
    title = ax.set_title('t = 0')
    
    def update(frame):
        data = states[frame].reshape(1, -1)
        im.set_array(data)
        title.set_text(f't = {frame}')
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(states),
        interval=interval, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    return anim


def plot_magnetization_profile(
    sites: np.ndarray,
    radius: int = 5,
    ax: Optional[Any] = None,
    title: str = "Local Magnetization Profile",
) -> Any:
    """
    Plot local magnetization (coarse field) profile.
    
    Args:
        sites: Lattice state
        radius: Averaging radius
        ax: Matplotlib axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    
    if hasattr(sites, '_sites'):
        sites = sites._sites
    sites = np.asarray(sites)
    N = len(sites)
    
    # Compute local magnetization
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    padded = np.concatenate([sites[-radius:], sites, sites[:radius]])
    local_mag = np.convolve(padded.astype(float), kernel, mode='valid')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(range(N), local_mag, 'b-', linewidth=1)
    ax.fill_between(range(N), local_mag, 0, alpha=0.3)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Local magnetization')
    ax.set_title(title)
    ax.set_xlim(0, N)
    ax.set_ylim(-1.1, 1.1)
    
    return ax
