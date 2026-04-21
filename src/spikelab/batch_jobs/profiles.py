"""Load cluster profile presets for job execution."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Dict

import yaml

from .models import ClusterProfile


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid profile file: {path}")
    return data


def load_cluster_profile(path: str) -> ClusterProfile:
    """Load a profile from an explicit YAML path."""
    payload = _read_yaml(Path(path))
    return ClusterProfile.model_validate(payload)


def load_profile_from_name(name: str) -> ClusterProfile:
    """Load one of the built-in profile files by name."""
    normalized = name.strip().lower()
    if normalized in {"nrp", "nautilus"}:
        filename = "nrp.yaml"
    else:
        filename = "defaults.yaml"
    base = files("spikelab.batch_jobs").joinpath("profiles")
    payload = _read_yaml(Path(str(base.joinpath(filename))))
    return ClusterProfile.model_validate(payload)
