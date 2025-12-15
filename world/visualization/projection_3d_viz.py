"""
3D projection visualization for observer view.
"""

from __future__ import annotations
from typing import Optional, Tuple, Any, List
import numpy as np

_plt = None
_Axes3D = None

def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt

def _get_3d():
    global _Axes3D
    if _Axes3D is None:
        from mpl_toolkits.mplot3d import Axes3D
        _Axes3D = Axes3D
    return _Axes3D


def plot_3d_projection(
    positions: np.ndarray,
    values: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "3D Projection",
    cmap: str = "viridis",
    size: float = 20.0,
    alpha: float = 0.8,
) -> Any:
    """
    Plot 3D projection of 1D lattice.
    
    Uses coordinate mapping from 1D index to 3D position
    and colors by field value.
    
    Args:
        positions: Nx3 array of 3D coordinates
        values: N array of field values for coloring
        ax: Matplotlib 3D axis
        title: Plot title
        cmap: Colormap name
        size: Marker size
        alpha: Marker transparency
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    _get_3d()  # Import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Color by values
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=values,
        cmap=cmap,
        s=size,
        alpha=alpha,
    )
    
    plt.colorbar(scatter, ax=ax, label='Field Value')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return ax


def plot_hilbert_curve(
    order: int = 3,
    ax: Optional[Any] = None,
    color: str = 'blue',
) -> Any:
    """
    Plot 3D Hilbert curve.
    
    Args:
        order: Hilbert curve order
        ax: Matplotlib 3D axis
        color: Line color
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    _get_3d()
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Generate Hilbert curve points
    n = 2 ** order
    points = []
    
    for i in range(n ** 3):
        x, y, z = _d2xyz(i, order)
        points.append((x, y, z))
    
    points = np.array(points)
    
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 
            color=color, linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Hilbert Curve (order {order})')
    
    return ax


def _d2xyz(d: int, order: int) -> Tuple[int, int, int]:
    """Convert Hilbert index to 3D coordinates (simplified)."""
    n = 2 ** order
    x = y = z = 0
    s = 1
    
    while s < n:
        rx = 1 & (d // 4)
        ry = 1 & ((d ^ rx) // 2)
        rz = 1 & (d ^ rx ^ ry)
        
        # Rotate
        if ry == 0:
            if rz == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        
        x += s * rx
        y += s * ry
        z += s * rz
        
        d //= 8
        s *= 2
    
    return x, y, z


def plot_cycles_3d(
    cycle_positions: List[np.ndarray],
    cycle_values: List[np.ndarray],
    ax: Optional[Any] = None,
    title: str = "Omega Cycles in 3D",
) -> Any:
    """
    Plot omega cycles as colored tubes in 3D.
    
    Args:
        cycle_positions: List of Mx3 position arrays per cycle
        cycle_values: List of M value arrays per cycle
        ax: Matplotlib 3D axis
        title: Plot title
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    _get_3d()
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10.colors
    
    for i, (pos, val) in enumerate(zip(cycle_positions, cycle_values)):
        color = colors[i % len(colors)]
        
        # Plot cycle as line
        if len(pos) > 1:
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                   color=color, linewidth=2, label=f'Cycle {i}')
            
            # Close the cycle
            ax.plot([pos[-1, 0], pos[0, 0]], 
                   [pos[-1, 1], pos[0, 1]],
                   [pos[-1, 2], pos[0, 2]],
                   color=color, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if len(cycle_positions) <= 10:
        ax.legend()
    
    return ax


def plot_field_slices(
    field_3d: np.ndarray,
    slices: Tuple[int, int, int] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> Any:
    """
    Plot 2D slices through 3D field.
    
    Args:
        field_3d: 3D numpy array of field values
        slices: Tuple of (x_slice, y_slice, z_slice) indices
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    plt = _get_plt()
    
    nx, ny, nz = field_3d.shape
    
    if slices is None:
        slices = (nx // 2, ny // 2, nz // 2)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # XY slice at z
    ax = axes[0]
    im = ax.imshow(field_3d[:, :, slices[2]], cmap='RdBu', origin='lower')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'XY slice at Z={slices[2]}')
    plt.colorbar(im, ax=ax)
    
    # XZ slice at y
    ax = axes[1]
    im = ax.imshow(field_3d[:, slices[1], :], cmap='RdBu', origin='lower')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f'XZ slice at Y={slices[1]}')
    plt.colorbar(im, ax=ax)
    
    # YZ slice at x
    ax = axes[2]
    im = ax.imshow(field_3d[slices[0], :, :], cmap='RdBu', origin='lower')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title(f'YZ slice at X={slices[0]}')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_isosurface(
    field_3d: np.ndarray,
    level: float = 0.0,
    ax: Optional[Any] = None,
    title: str = "Isosurface",
    color: str = 'cyan',
    alpha: float = 0.5,
) -> Any:
    """
    Plot isosurface of 3D field using marching cubes.
    
    Args:
        field_3d: 3D field values
        level: Isosurface level
        ax: Matplotlib 3D axis
        title: Plot title
        color: Surface color
        alpha: Surface transparency
        
    Returns:
        Matplotlib axis
    """
    plt = _get_plt()
    _get_3d()
    
    try:
        from skimage import measure
        has_skimage = True
    except ImportError:
        has_skimage = False
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    if has_skimage:
        # Use marching cubes for smooth surface
        verts, faces, _, _ = measure.marching_cubes(field_3d, level=level)
        
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces,
            color=color,
            alpha=alpha,
        )
    else:
        # Fallback: plot points at threshold
        indices = np.argwhere(np.abs(field_3d - level) < 0.1)
        if len(indices) > 0:
            ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2],
                      color=color, alpha=alpha, s=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return ax


def animate_3d_evolution(
    history: List[Tuple[np.ndarray, np.ndarray]],
    interval: int = 100,
    output_path: Optional[str] = None,
) -> Any:
    """
    Create 3D animation of evolution.
    
    Args:
        history: List of (positions, values) tuples
        interval: Frame interval in ms
        output_path: Path to save animation
        
    Returns:
        Matplotlib animation
    """
    plt = _get_plt()
    _get_3d()
    from matplotlib.animation import FuncAnimation
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get ranges
    all_pos = np.vstack([h[0] for h in history])
    all_val = np.concatenate([h[1] for h in history])
    
    xmin, xmax = all_pos[:, 0].min(), all_pos[:, 0].max()
    ymin, ymax = all_pos[:, 1].min(), all_pos[:, 1].max()
    zmin, zmax = all_pos[:, 2].min(), all_pos[:, 2].max()
    vmin, vmax = all_val.min(), all_val.max()
    
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', vmin=vmin, vmax=vmax)
    
    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return scatter,
    
    def update(frame):
        pos, val = history[frame]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter.set_array(val)
        ax.set_title(f'Frame {frame}')
        return scatter,
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(history), interval=interval, blit=False)
    
    if output_path:
        anim.save(output_path, writer='pillow')
    
    return anim
