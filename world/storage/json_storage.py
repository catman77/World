"""
JSON storage for simple serialization.
"""

from __future__ import annotations
import json
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict, is_dataclass
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'shape': obj.shape,
                'data': obj.tolist(),
            }
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        return super().default(obj)


def numpy_decoder(dct):
    """JSON decoder hook for numpy arrays."""
    if '__numpy__' in dct:
        return np.array(dct['data'], dtype=dct['dtype']).reshape(dct['shape'])
    return dct


class JSONStorage:
    """
    JSON-based storage backend.
    
    Supports:
    - Plain JSON files
    - Gzipped JSON files
    - Automatic numpy array handling
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize JSON storage.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        data: Any,
        filename: str,
        compress: bool = False,
    ) -> Path:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filename: Filename (without extension)
            compress: Use gzip compression
            
        Returns:
            Path to saved file
        """
        if compress:
            filepath = self.base_path / f"{filename}.json.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
        else:
            filepath = self.base_path / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
        
        return filepath
    
    def load(self, filename: str) -> Any:
        """
        Load data from JSON file.
        
        Args:
            filename: Filename (with or without extension)
            
        Returns:
            Loaded data
        """
        # Try different extensions
        for ext in ['', '.json', '.json.gz']:
            filepath = self.base_path / f"{filename}{ext}"
            if filepath.exists():
                break
        else:
            raise FileNotFoundError(f"No JSON file found for {filename}")
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                return json.load(f, object_hook=numpy_decoder)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f, object_hook=numpy_decoder)
    
    def list_files(self, pattern: str = "*.json*") -> List[Path]:
        """List all JSON files in storage."""
        return list(self.base_path.glob(pattern))
    
    def exists(self, filename: str) -> bool:
        """Check if file exists."""
        for ext in ['', '.json', '.json.gz']:
            if (self.base_path / f"{filename}{ext}").exists():
                return True
        return False
    
    def delete(self, filename: str) -> bool:
        """Delete file if exists."""
        for ext in ['', '.json', '.json.gz']:
            filepath = self.base_path / f"{filename}{ext}"
            if filepath.exists():
                filepath.unlink()
                return True
        return False


def save_state(
    state: Any,
    filepath: Union[str, Path],
    compress: bool = True,
) -> Path:
    """
    Convenience function to save lattice state.
    
    Args:
        state: State object (Lattice or dict)
        filepath: Full path to save file
        compress: Use compression
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert state to dict if needed
    if hasattr(state, 'to_dict'):
        data = state.to_dict()
    elif hasattr(state, '__dict__'):
        data = {
            'sites': getattr(state, '_sites', getattr(state, 'sites', None)),
            'phases': getattr(state, '_phases', getattr(state, 'phases', None)),
            'time': getattr(state, '_time', getattr(state, 'time', 0)),
        }
    else:
        data = state
    
    if compress or str(filepath).endswith('.gz'):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
    
    return filepath


def load_state(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load lattice state.
    
    Args:
        filepath: Path to state file
        
    Returns:
        State dict with numpy arrays
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.gz' or str(filepath).endswith('.json.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f, object_hook=numpy_decoder)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f, object_hook=numpy_decoder)
