from pathlib import Path
import yaml

# ---- Project paths ----
PROJECT_DIR = Path(r"D:\Projects\NerfStudio")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"

# Where the cloud filtering YAML presets live
CLOUD_FILTER_PRESETS_DIR = Path(__file__).parent / "cloud_filtering"


def load_cloud_filter_preset(name: str) -> dict:
    """
    Load a cloud filtering preset YAML by name.

    Args:
        name (str): preset name, e.g. "medium_plus", "hard", "ultra_light".

    Returns:
        dict: keys like radius_keep_q, density_keep_q, hsv_v_min, etc.
    """
    preset_path = CLOUD_FILTER_PRESETS_DIR / f"{name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Cloud filter preset '{name}' not found at {preset_path}")
    with preset_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
