from __future__ import annotations
import argparse
import sys
from pathlib import Path
import subprocess

from src.config.base_config import EXPERIMENTS_DIR, EXPORTS_DIR, ensure_core_dirs
from src.utils.menu import choose_export_folder, resolve_export_and_input
from src.utils.common import header

ensure_core_dirs()


# ============================================================
# Helpers
# ============================================================

def list_experiments() -> list[Path]:
    """Return all experiment folders under outputs/experiments."""
    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(f"No experiments directory found at: {EXPERIMENTS_DIR}")
    exps = [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()]
    exps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return exps


def choose_experiment() -> Path:
    """Interactive menu to choose an experiment."""
    exps = list_experiments()
    if not exps:
        raise FileNotFoundError("No experiment folders found.")

    print("\nAvailable experiments:\n")
    for i, exp in enumerate(exps, 1):
        print(f"  {i}) {exp.name}")

    while True:
        sel = input(f"\nChoose experiment [1-{len(exps)} or name]: ").strip()
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(exps):
                return exps[i - 1]
        low = sel.lower()
        # exact/prefix matching
        matches = [p for p in exps if p.name.lower().startswith(low)]
        if len(matches) == 1:
            return matches[0]
        print("Invalid choice. Try again.")


def run_ns_render(config_path: Path, output_dir: Path,
                  trajectory: str = "orbit", fmt: str = "video",
                  width: int = 1920, height: int = 1080,
                  seconds: int = 10) -> None:
    """
    Run ns-render with the provided parameters.
    """
    cmd = [
        "ns-render",
        "--load-config", str(config_path),
        "--output-path", str(output_dir),
        "--trajectory", trajectory,
        "--output-format", fmt,
        "--width", str(width),
        "--height", str(height),
        "--seconds", str(seconds)
    ]

    print(f"\nRunning render:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ============================================================
# Main pipeline
# ============================================================

def pipeline() -> int:
    """
    Interactive or CLI-based rendering pipeline:
      1. Choose experiment (auto or menu)
      2. Choose render trajectory
      3. Render video or frames via ns-render
    """
    try:
        parser = argparse.ArgumentParser(description="Render novel views from trained Nerfstudio models.")
        parser.add_argument("--exp", type=str, help="Experiment folder name or 'auto' for latest.")
        parser.add_argument("--trajectory", type=str, default="orbit",
                            choices=["orbit", "spiral", "figure_eight", "interpolate"],
                            help="Camera path type.")
        parser.add_argument("--output-format", type=str, default="video",
                            choices=["video", "images"], help="Render ou
