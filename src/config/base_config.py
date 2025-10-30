"""
base_config.py
==============

Central/shared project layout config.

This module defines the canonical directory structure for the project
WITHOUT hardcoding a specific absolute path like "D:\\Projects\\NerfStudio".
Instead, it infers the project root dynamically from its own location.

All other modules (training, export, filtering, mesh, etc.) should import
paths from here instead of redefining them.

Example usage:
    from src.config.base_config import (
        PROJECT_DIR,
        VIDEOS_DIR,
        OUTPUTS_DIR,
        DATASET_DIR,
        EXPERIMENTS_DIR,
        EXPORTS_DIR,
        ensure_core_dirs,
    )

    ensure_core_dirs()  # make sure required folders exist
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


# -------------------------------------------------
# Project root discovery
# -------------------------------------------------
# We assume this file lives at:
#   <project_root>/src/config/base_config.py
# So project_root is 2 levels up from here.
PROJECT_DIR: Path = Path(__file__).resolve().parents[2]

# If for some reason you want to override PROJECT_DIR manually (for example
# run this code as a standalone script from elsewhere), you *could* replace
# PROJECT_DIR at runtime. But by default we trust the repo layout.


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
