from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Literal
import yaml

from src.config.base_config import ensure_core_dirs

ensure_core_dirs()

# root of all preset YAMLs
PRESETS_ROOT = Path(__file__).resolve().parents[1] / "config" / "profiles"

PresetType = Literal["mesh", "clouds_filtering", "points_filtering"]


def _dir_for(kind: PresetType) -> Path:
    """Return directory path for the given preset kind."""
    return PRESETS_ROOT / kind


def list_presets(kind: PresetType) -> list[str]:
    """
    Return all available preset names (without .yaml extension)
    for the given kind.
    """
    d = _dir_for(kind)
    if not d.exists():
        raise FileNotFoundError(f"Preset directory not found: {d}")
    return sorted([p.stem for p in d.glob("*.yaml") if p.is_file()])


def load_preset(kind: PresetType, name: str) -> Dict[str, Any]:
    """
    Load a preset YAML of the given kind ("mesh", "clouds", "points_filtering")
    and return it as a dict.
    """
    path = _dir_for(kind) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset '{name}' not found in {path.parent}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid preset structure in {path}")
    return data
