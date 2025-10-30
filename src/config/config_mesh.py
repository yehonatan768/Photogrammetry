# src/config/config_mesh.py
from pathlib import Path
import yaml
from src.config.base_config import EXPORTS_DIR, ensure_core_dirs

# === Directory for mesh YAML profiles ===
MESH_PRESETS_DIR = Path(__file__).parent / "mesh_profiles"
ensure_core_dirs()

def load_mesh_preset(name: str) -> dict:
    """
    Load a mesh reconstruction preset YAML file.

    Args:
        name (str): Profile name, e.g. 'balanced', 'crisp', 'ultra_preserve'.

    Returns:
        dict: Mesh parameters such as depth, crop, smooth, adaptive, etc.
    """
    path = MESH_PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Mesh preset '{name}' not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
