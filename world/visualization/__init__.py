"""
Visualization module for RSL World.

Provides visualization tools:
- Lattice state visualization
- Time series plots
- Phase space plots
- 3D projection views
"""

from .lattice_viz import (
    plot_lattice,
    plot_lattice_evolution as plot_evolution,
    animate_evolution,
)
from .timeseries_viz import (
    plot_timeseries,
    plot_multiple_timeseries,
    plot_correlations,
    plot_power_spectrum,
    plot_histogram,
    plot_evolution_summary,
)
from .projection_3d_viz import (
    plot_3d_projection,
    plot_hilbert_curve,
    plot_cycles_3d,
    plot_field_slices,
    plot_isosurface,
    animate_3d_evolution,
)
from .phase_viz import (
    plot_phase_diagram,
    plot_phase_boundaries,
    plot_order_parameter_scaling,
    plot_susceptibility_scaling,
    plot_binder_cumulant,
    plot_correlation_length,
    plot_critical_exponents_summary,
    plot_phase_diagram_full,
)

__all__ = [
    # Lattice
    'plot_lattice',
    'plot_evolution',
    'animate_evolution',
    # Time series
    'plot_timeseries',
    'plot_multiple_timeseries',
    'plot_correlations',
    'plot_power_spectrum',
    'plot_histogram',
    'plot_evolution_summary',
    # 3D projection
    'plot_3d_projection',
    'plot_hilbert_curve',
    'plot_cycles_3d',
    'plot_field_slices',
    'plot_isosurface',
    'animate_3d_evolution',
    # Phase diagrams
    'plot_phase_diagram',
    'plot_phase_boundaries',
    'plot_order_parameter_scaling',
    'plot_susceptibility_scaling',
    'plot_binder_cumulant',
    'plot_correlation_length',
    'plot_critical_exponents_summary',
    'plot_phase_diagram_full',
]
