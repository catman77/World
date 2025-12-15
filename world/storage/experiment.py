"""
Experiment recording and management.
"""

from __future__ import annotations
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

from .json_storage import JSONStorage, NumpyEncoder, numpy_decoder
from .hdf5_storage import HDF5Storage, HAS_H5PY


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run."""
    
    experiment_id: str
    name: str
    description: str
    created_at: str
    config: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentMetadata":
        return cls(**d)


class ExperimentRecorder:
    """
    Records experiment data including:
    - Configuration
    - Evolution history
    - Snapshots at checkpoints
    - Analysis results
    - Metrics over time
    """
    
    def __init__(
        self,
        name: str,
        base_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        use_hdf5: bool = True,
    ):
        """
        Initialize experiment recorder.
        
        Args:
            name: Experiment name
            base_path: Base directory for experiments
            config: Experiment configuration
            description: Experiment description
            tags: List of tags for categorization
            use_hdf5: Use HDF5 for large data (if available)
        """
        self.experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.base_path = Path(base_path) / self.experiment_id
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.use_hdf5 = use_hdf5 and HAS_H5PY
        
        # Initialize storage
        self.json_storage = JSONStorage(self.base_path)
        if self.use_hdf5:
            self.hdf5_storage = HDF5Storage(self.base_path / "data.h5")
        
        # Create metadata
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            config=config or {},
            tags=tags or [],
            status="running",
        )
        
        # Save initial metadata
        self._save_metadata()
        
        # Tracking
        self.start_time = time.time()
        self._metrics: Dict[str, List[tuple]] = {}  # metric_name -> [(time, value), ...]
        self._step = 0
    
    def _save_metadata(self):
        """Save metadata to file."""
        self.json_storage.save(self.metadata.to_dict(), "metadata", compress=False)
    
    def set_config(self, config: Dict[str, Any]):
        """Update experiment configuration."""
        self.metadata.config.update(config)
        self._save_metadata()
    
    def init_history(
        self,
        size: int,
        max_steps: int,
        include_phases: bool = True,
    ):
        """
        Initialize history storage.
        
        Args:
            size: Lattice size
            max_steps: Maximum simulation steps
            include_phases: Include phase data
        """
        if self.use_hdf5:
            self.hdf5_storage.init_history(size, max_steps, include_phases)
        else:
            # For JSON, we'll accumulate in memory and save periodically
            self._history_buffer = {
                'sites': [],
                'times': [],
                'phases': [] if include_phases else None,
            }
            self._history_flush_interval = min(1000, max_steps // 10)
    
    def record_state(
        self,
        sites: np.ndarray,
        time: int,
        phases: Optional[np.ndarray] = None,
    ):
        """
        Record a simulation state.
        
        Args:
            sites: Site values
            time: Simulation time
            phases: Phase values
        """
        self._step = time
        
        if self.use_hdf5:
            self.hdf5_storage.append_history(sites, time, phases)
        else:
            self._history_buffer['sites'].append(sites.tolist())
            self._history_buffer['times'].append(time)
            if phases is not None and self._history_buffer['phases'] is not None:
                self._history_buffer['phases'].append(phases.tolist())
            
            # Periodic flush
            if len(self._history_buffer['times']) >= self._history_flush_interval:
                self._flush_history()
    
    def _flush_history(self):
        """Flush history buffer to disk."""
        if hasattr(self, '_history_buffer') and self._history_buffer['times']:
            chunk_id = len(list(self.base_path.glob("history_*.json*")))
            self.json_storage.save(self._history_buffer, f"history_{chunk_id:04d}", compress=True)
            
            # Reset buffer
            include_phases = self._history_buffer['phases'] is not None
            self._history_buffer = {
                'sites': [],
                'times': [],
                'phases': [] if include_phases else None,
            }
    
    def record_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Time step (uses current step if None)
        """
        if step is None:
            step = self._step
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append((step, value))
    
    def record_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Record multiple metrics at once."""
        for name, value in metrics.items():
            self.record_metric(name, value, step)
    
    def save_snapshot(
        self,
        state: Any,
        name: str,
        additional_data: Optional[Dict] = None,
    ):
        """
        Save a named snapshot.
        
        Args:
            state: State to save
            name: Snapshot name
            additional_data: Additional data to include
        """
        snapshot_dir = self.base_path / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        data = {
            'step': self._step,
            'timestamp': datetime.now().isoformat(),
        }
        
        if hasattr(state, 'to_dict'):
            data['state'] = state.to_dict()
        elif hasattr(state, '_sites'):
            data['state'] = {
                'sites': state._sites,
                'phases': getattr(state, '_phases', None),
                'time': getattr(state, '_time', self._step),
            }
        else:
            data['state'] = state
        
        if additional_data:
            data.update(additional_data)
        
        storage = JSONStorage(snapshot_dir)
        storage.save(data, name, compress=True)
    
    def save_analysis(self, results: Dict[str, Any], name: str = "analysis"):
        """
        Save analysis results.
        
        Args:
            results: Analysis results dict
            name: Results name
        """
        if self.use_hdf5:
            self.hdf5_storage.save_analysis(results, name)
        else:
            self.json_storage.save(results, name, compress=True)
    
    def finalize(self, status: str = "completed"):
        """
        Finalize experiment and save all data.
        
        Args:
            status: Final status (completed, failed)
        """
        # Flush any buffered data
        if hasattr(self, '_history_buffer'):
            self._flush_history()
        
        # Save metrics
        metrics_data = {
            name: {'steps': [m[0] for m in values], 'values': [m[1] for m in values]}
            for name, values in self._metrics.items()
        }
        self.json_storage.save(metrics_data, "metrics", compress=True)
        
        # Update metadata
        self.metadata.status = status
        self.metadata.duration_seconds = time.time() - self.start_time
        self._save_metadata()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type is not None else "completed"
        self.finalize(status)
        return False
    
    @property
    def path(self) -> Path:
        """Get experiment directory path."""
        return self.base_path


def load_experiment(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved experiment.
    
    Args:
        path: Path to experiment directory
        
    Returns:
        Dict containing:
        - metadata: ExperimentMetadata
        - metrics: Dict of recorded metrics
        - snapshots: List of snapshot names
        - has_hdf5: Whether HDF5 data is available
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Experiment not found: {path}")
    
    storage = JSONStorage(path)
    
    # Load metadata
    metadata_dict = storage.load("metadata")
    metadata = ExperimentMetadata.from_dict(metadata_dict)
    
    # Load metrics
    metrics = {}
    if storage.exists("metrics"):
        metrics = storage.load("metrics")
    
    # List snapshots
    snapshots_dir = path / "snapshots"
    snapshots = []
    if snapshots_dir.exists():
        snapshots = [f.stem for f in snapshots_dir.glob("*.json*")]
    
    # Check for HDF5 data
    has_hdf5 = (path / "data.h5").exists()
    
    return {
        'metadata': metadata,
        'metrics': metrics,
        'snapshots': snapshots,
        'has_hdf5': has_hdf5,
        'path': path,
    }


def load_experiment_history(
    path: Union[str, Path],
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Load evolution history from experiment.
    
    Args:
        path: Experiment path
        start: Start index
        end: End index
        step: Step size
        
    Returns:
        Dict with sites, times, phases arrays
    """
    path = Path(path)
    
    # Try HDF5 first
    hdf5_path = path / "data.h5"
    if hdf5_path.exists() and HAS_H5PY:
        storage = HDF5Storage(hdf5_path, mode='r')
        sites, times, phases = storage.load_history(start, end, step)
        return {'sites': sites, 'times': times, 'phases': phases}
    
    # Fall back to JSON
    all_sites = []
    all_times = []
    all_phases = []
    
    for chunk_file in sorted(path.glob("history_*.json*")):
        storage = JSONStorage(path)
        data = storage.load(chunk_file.stem)
        all_sites.extend(data['sites'])
        all_times.extend(data['times'])
        if data.get('phases'):
            all_phases.extend(data['phases'])
    
    sites = np.array(all_sites)[start:end:step]
    times = np.array(all_times)[start:end:step]
    phases = np.array(all_phases)[start:end:step] if all_phases else None
    
    return {'sites': sites, 'times': times, 'phases': phases}


def load_experiment_snapshot(
    path: Union[str, Path],
    name: str,
) -> Dict[str, Any]:
    """
    Load a specific snapshot.
    
    Args:
        path: Experiment path
        name: Snapshot name
        
    Returns:
        Snapshot data
    """
    path = Path(path) / "snapshots"
    storage = JSONStorage(path)
    return storage.load(name)


def list_experiments(base_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    List all experiments in a directory.
    
    Args:
        base_path: Base experiments directory
        
    Returns:
        List of experiment summaries
    """
    base_path = Path(base_path)
    experiments = []
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir() and (exp_dir / "metadata.json").exists():
            try:
                info = load_experiment(exp_dir)
                experiments.append({
                    'id': info['metadata'].experiment_id,
                    'name': info['metadata'].name,
                    'status': info['metadata'].status,
                    'created_at': info['metadata'].created_at,
                    'duration': info['metadata'].duration_seconds,
                    'path': str(exp_dir),
                })
            except Exception:
                pass
    
    return sorted(experiments, key=lambda x: x['created_at'], reverse=True)
