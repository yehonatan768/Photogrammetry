from __future__ import annotations

from pathlib import Path
from typing import Dict

# -------------------------------------------------
# Project root discovery
# -------------------------------------------------

PROJECT_DIR: Path = Path(__file__).resolve().parents[2]

# -------------------------------------------------
# Canonical subdirectories
# -------------------------------------------------
VIDEOS_DIR: Path = PROJECT_DIR / "videos"

OUTPUTS_DIR: Path = PROJECT_DIR / "outputs"

# Nerfstudio dataset output (ns-process-data step)
DATASET_DIR: Path = OUTPUTS_DIR / "dataset"

# Nerfstudio training runs (ns-train nerfacto ... writes here)
EXPERIMENTS_DIR: Path = OUTPUTS_DIR / "experiments"

# Exported artifacts (point clouds, filtered clouds, meshes, etc.)
EXPORTS_DIR: Path = OUTPUTS_DIR / "exports"


# -------------------------------------------------
# Optional helpers for downstream modules
# -------------------------------------------------
def ensure_core_dirs() -> None:
    """
    Create core output directories if they don't exist yet.
    Safe to call at startup of any script.
    """
    for p in [VIDEOS_DIR, OUTPUTS_DIR, DATASET_DIR, EXPERIMENTS_DIR, EXPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def as_dict() -> Dict[str, str]:
    """
    Convenience: return main paths as plain strings (good for printing / logging).
    """
    return {
        "PROJECT_DIR": str(PROJECT_DIR),
        "VIDEOS_DIR": str(VIDEOS_DIR),
        "OUTPUTS_DIR": str(OUTPUTS_DIR),
        "DATASET_DIR": str(DATASET_DIR),
        "EXPERIMENTS_DIR": str(EXPERIMENTS_DIR),
        "EXPORTS_DIR": str(EXPORTS_DIR),
    }


def debug_print_layout() -> None:
    """
    Print the resolved layout (useful for sanity-check when moving machines).
    """
    print("=== Project Layout ===")
    for k, v in as_dict().items():
        print(f"{k:15} -> {v}")
    print("======================")
