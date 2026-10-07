"""
Config Loader Utility for EYEON.

Provides robust configuration loading, validation, and path resolution
relative to the project root regardless of the execution working directory.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

_CACHED_CONFIG: Optional[Dict[str, Any]] = None
_PROJECT_ROOT: Optional[Path] = None


def get_project_root() -> Path:
    """
    Locates the project root directory by traversing upwards until config.yaml is found.
    Defaults to the parent directory of this file if not explicitly found.
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # Start from current directory of this file (src/utils/)
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").is_file():
            _PROJECT_ROOT = parent
            return _PROJECT_ROOT

    # Fallback to grandparent of src/utils (i.e., project root)
    _PROJECT_ROOT = current.parent.parent
    return _PROJECT_ROOT


def load_config(config_path: Optional[str or Path] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Loads and caches the YAML configuration.

    Args:
        config_path: Optional explicit path to config.yaml. If None, auto-locates from project root.
        reload: If True, forces re-reading from disk.

    Returns:
        Dict containing full project configuration.
    """
    global _CACHED_CONFIG

    if _CACHED_CONFIG is not None and not reload and config_path is None:
        return _CACHED_CONFIG

    if config_path is None:
        resolved_path = get_project_root() / "config.yaml"
    else:
        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = get_project_root() / resolved_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML structure in: {resolved_path}")

    _CACHED_CONFIG = config
    return _CACHED_CONFIG


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Retrieves a nested value from the config using dot-notation.
    Example: get_config_value("detection.confidence.person", 0.40)
    """
    config = load_config()
    keys = key_path.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def get_path(key_path: str) -> Path:
    """
    Retrieves a path string from config and resolves it relative to the project root.
    Example: get_path("video.source") -> Path(".../EYEON/data/videos")
    """
    val = get_config_value(key_path)
    if val is None:
        raise KeyError(f"Configuration key '{key_path}' not found.")
    p = Path(val)
    if p.is_absolute():
        return p
    return get_project_root() / p


if __name__ == "__main__":
    print(f"Project Root: {get_project_root()}")
    cfg = load_config()
    print("Configuration loaded successfully!")
    print(f"Project Name: {cfg.get('project', {}).get('name')}")
    print(f"Model Path:   {get_path('detection.model')}")
    print(f"Video Source: {get_path('video.source')}")
    print(f"Database:     {get_path('database.path')}")
