from pathlib import Path
import yaml

# ---- Project paths ----
PROJECT_DIR = Path(r"D:\Projects\NerfStudio")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"

# Where the point prefilter YAML presets live
POINTS_FILTER_PRESETS_DIR = Path(__file__).parent / "points_filtering"


def load_points_filter_preset(name: str) -> dict:
    """
    Load a point prefilter preset YAML by name.

    Args:
        name (str): preset name, e.g. "light", "medium", "ultra".

    Returns:
        dict: keys like radius_keep_q, density_keep_q, composite_keep_q, sor_std.
    """
    preset_path = POINTS_FILTER_PRESETS_DIR / f"{name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Points filter preset '{name}' not found at {preset_path}")
    with preset_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
