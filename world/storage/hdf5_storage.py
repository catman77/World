"""
HDF5 storage for large-scale simulation data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from contextlib import contextmanager

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class HDF5Storage:
    """
    HDF5-based storage for efficient handling of large arrays.
    
    Features:
    - Efficient storage of numpy arrays
    - Chunked storage for evolution history
    - Compression support
    - Lazy loading
    """
    
    def __init__(self, filepath: Union[str, Path], mode: str = 'a'):
        """
        Initialize HDF5 storage.
        
        Args:
            filepath: Path to HDF5 file
            mode: File mode ('r', 'r+', 'w', 'w-', 'a')
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 storage. "
                            "Install with: pip install h5py")
        
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._file = None
    
    @contextmanager
    def open(self):
        """Context manager for file access."""
        f = h5py.File(self.filepath, self.mode)
        try:
            yield f
        finally:
            f.close()
    
    def save_state(
        self,
        sites: np.ndarray,
        phases: Optional[np.ndarray] = None,
        time: int = 0,
        group: str = "state",
        attrs: Optional[Dict] = None,
    ) -> None:
        """
        Save a single lattice state.
        
        Args:
            sites: Site values array
            phases: Phase values array (optional)
            time: Simulation time
            group: Group name in HDF5 file
            attrs: Additional attributes to store
        """
        with self.open() as f:
            g = f.require_group(group)
            
            # Save arrays
            if 'sites' in g:
                del g['sites']
            g.create_dataset('sites', data=sites, compression='gzip')
            
            if phases is not None:
                if 'phases' in g:
                    del g['phases']
                g.create_dataset('phases', data=phases, compression='gzip')
            
            g.attrs['time'] = time
            
            if attrs:
                for k, v in attrs.items():
                    if isinstance(v, np.ndarray):
                        if k in g:
                            del g[k]
                        g.create_dataset(k, data=v, compression='gzip')
                    else:
                        g.attrs[k] = v
    
    def load_state(self, group: str = "state") -> Dict[str, Any]:
        """
        Load a lattice state.
        
        Args:
            group: Group name in HDF5 file
            
        Returns:
            Dict with sites, phases, time, and any attributes
        """
        with h5py.File(self.filepath, 'r') as f:
            g = f[group]
            
            result = {
                'sites': g['sites'][:],
                'time': g.attrs.get('time', 0),
            }
            
            if 'phases' in g:
                result['phases'] = g['phases'][:]
            
            # Load additional attributes
            for k in g.attrs.keys():
                if k not in result:
                    result[k] = g.attrs[k]
            
            # Load additional datasets
            for k in g.keys():
                if k not in result:
                    result[k] = g[k][:]
            
            return result
    
    def init_history(
        self,
        size: int,
        max_steps: int,
        include_phases: bool = True,
        group: str = "history",
    ) -> None:
        """
        Initialize chunked storage for evolution history.
        
        Args:
            size: Lattice size
            max_steps: Maximum number of time steps
            include_phases: Store phase information
            group: Group name
        """
        with self.open() as f:
            g = f.require_group(group)
            
            # Create resizable datasets
            g.create_dataset(
                'sites',
                shape=(0, size),
                maxshape=(max_steps, size),
                dtype=np.int8,
                chunks=(min(100, max_steps), size),
                compression='gzip',
            )
            
            if include_phases:
                g.create_dataset(
                    'phases',
                    shape=(0, size),
                    maxshape=(max_steps, size),
                    dtype=np.float64,
                    chunks=(min(100, max_steps), size),
                    compression='gzip',
                )
            
            g.create_dataset(
                'times',
                shape=(0,),
                maxshape=(max_steps,),
                dtype=np.int64,
                chunks=(min(100, max_steps),),
            )
            
            g.attrs['size'] = size
            g.attrs['max_steps'] = max_steps
            g.attrs['current_step'] = 0
    
    def append_history(
        self,
        sites: np.ndarray,
        time: int,
        phases: Optional[np.ndarray] = None,
        group: str = "history",
    ) -> None:
        """
        Append state to history.
        
        Args:
            sites: Site values
            time: Simulation time
            phases: Phase values (optional)
            group: Group name
        """
        with self.open() as f:
            g = f[group]
            
            idx = g.attrs['current_step']
            
            # Resize and append
            g['sites'].resize((idx + 1, g.attrs['size']))
            g['sites'][idx] = sites
            
            g['times'].resize((idx + 1,))
            g['times'][idx] = time
            
            if phases is not None and 'phases' in g:
                g['phases'].resize((idx + 1, g.attrs['size']))
                g['phases'][idx] = phases
            
            g.attrs['current_step'] = idx + 1
    
    def load_history(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
        group: str = "history",
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load history slice.
        
        Args:
            start: Start index
            end: End index (None = all)
            step: Step size
            group: Group name
            
        Returns:
            Tuple of (sites, times, phases)
        """
        with h5py.File(self.filepath, 'r') as f:
            g = f[group]
            
            if end is None:
                end = g.attrs['current_step']
            
            sites = g['sites'][start:end:step]
            times = g['times'][start:end:step]
            phases = g['phases'][start:end:step] if 'phases' in g else None
            
            return sites, times, phases
    
    def save_analysis(
        self,
        results: Dict[str, Any],
        group: str = "analysis",
    ) -> None:
        """
        Save analysis results.
        
        Args:
            results: Dict of results (arrays and scalars)
            group: Group name
        """
        with self.open() as f:
            g = f.require_group(group)
            
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    if k in g:
                        del g[k]
                    g.create_dataset(k, data=v, compression='gzip')
                elif isinstance(v, (int, float, str, bool)):
                    g.attrs[k] = v
                elif isinstance(v, dict):
                    # Nested dict -> subgroup
                    self.save_analysis(v, f"{group}/{k}")
    
    def load_analysis(self, group: str = "analysis") -> Dict[str, Any]:
        """
        Load analysis results.
        
        Args:
            group: Group name
            
        Returns:
            Dict of results
        """
        with h5py.File(self.filepath, 'r') as f:
            if group not in f:
                return {}
            
            g = f[group]
            result = {}
            
            # Load attributes
            for k in g.attrs.keys():
                result[k] = g.attrs[k]
            
            # Load datasets
            for k in g.keys():
                if isinstance(g[k], h5py.Dataset):
                    result[k] = g[k][:]
                elif isinstance(g[k], h5py.Group):
                    result[k] = self.load_analysis(f"{group}/{k}")
            
            return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the HDF5 file."""
        info = {'filepath': str(self.filepath)}
        
        with h5py.File(self.filepath, 'r') as f:
            info['groups'] = list(f.keys())
            
            for group_name in f.keys():
                g = f[group_name]
                info[group_name] = {
                    'datasets': list(g.keys()),
                    'attrs': dict(g.attrs),
                }
        
        return info
    
    def close(self):
        """Ensure file is closed."""
        if self._file is not None:
            self._file.close()
            self._file = None


class LazyHDF5Array:
    """
    Lazy-loading wrapper for HDF5 datasets.
    
    Allows array-like access without loading entire dataset.
    """
    
    def __init__(self, filepath: Union[str, Path], dataset_path: str):
        """
        Initialize lazy array.
        
        Args:
            filepath: Path to HDF5 file
            dataset_path: Path to dataset within file
        """
        self.filepath = Path(filepath)
        self.dataset_path = dataset_path
        
        # Get shape without loading data
        with h5py.File(self.filepath, 'r') as f:
            ds = f[dataset_path]
            self._shape = ds.shape
            self._dtype = ds.dtype
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    def __len__(self) -> int:
        return self._shape[0]
    
    def __getitem__(self, key) -> np.ndarray:
        """Load slice from HDF5."""
        with h5py.File(self.filepath, 'r') as f:
            return f[self.dataset_path][key]
    
    def load(self) -> np.ndarray:
        """Load entire array."""
        with h5py.File(self.filepath, 'r') as f:
            return f[self.dataset_path][:]
