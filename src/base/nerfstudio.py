r"""
nerfstudio.py — Steps 1 & 2 (dataset → train) with explicit project layout
===========================================================================

Purpose (unchanged)
-------------------
Automate the first two stages of a Nerfstudio workflow:

1) **Build dataset** from a video using ``ns-process-data video``.
2) **Train nerfacto** using ``ns-train`` with a named experiment under
   ``<PROJECT_DIR>/outputs/experiments``.

What stayed the same
--------------------
- Same globals (paths, flags, names) and same CLI tools (``ns-process-data``, ``ns-train``).
- Same default behavior for frames target, viewer port, and environment variable
  ``CUDA_VISIBLE_DEVICES``.
- Skips dataset/training conditionally using ``SKIP_DATASET`` and ``SKIP_TRAIN``.
- Produces identical folders/files as your original script.

What's improved
---------------
- Smaller, single-purpose helpers with clear docstrings.
- Safer path handling and explicit, readable printing.
- Command construction is centralized and easy to tweak.

Requirements
------------
- Python 3.9+
- Nerfstudio CLIs available on PATH: ``ns-process-data``, ``ns-train``
- (Optional) ``ffmpeg`` improves video handling

Usage
-----
    py nerfstudio.py

Then open the viewer when prompted:
    http://127.0.0.1:7007
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from src.config.base_config import (
    PROJECT_DIR,
    VIDEOS_DIR,
    DATASET_DIR,
    EXPERIMENTS_DIR,
    ensure_core_dirs,
)

# =========================
# Global configuration
# =========================

# Absolute path to your Nerfstudio project root (adjust if needed).
PROJECT_DIR: Path = Path(r"D:\Projects\NerfStudio")

# Name of the video file located in <PROJECT_DIR>/videos/
VIDEO_FILE: str = "DJI0450.mp4"

# Logical name for the training run; this becomes the folder under output/experiments/
EXPERIMENT_NAME: str = "DJI0450_high_quality"

# Training length; higher = better quality but longer time.
MAX_ITERS: int = 80_000

# Controls which steps to run when re-running the script.
SKIP_DATASET: bool = False
SKIP_TRAIN: bool = False

# Select which GPU to use (e.g., "0" for first GPU, "0,1" for multi-GPU if supported).
CUDA_VISIBLE_DEVICES: str = "0"

# Target number of frames Nerfstudio should sample from the video (None = let Nerfstudio decide).
NUM_FRAMES_TARGET: int | None = 450

# -------------------------
# Directory layout (important)
# -------------------------
# We keep datasets under the original "<project>/outputs/dataset/...", but
# **experiments** now go to "<project>/outputs/experiments/...".
DATASET_ROOT: Path = PROJECT_DIR / "outputs" / "dataset"
EXPERIMENT_ROOT: Path = PROJECT_DIR / "outputs" / "experiments"   # ← as requested



# =========================
# Pretty printing
# =========================
def header(title: str) -> None:
    """Print a section header for readability."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


# =========================
# Shell helpers
# =========================
def run(cmd: str) -> None:
    """
    Execute a shell command and raise on failure with a clear header.

    Parameters
    ----------
    cmd : str
        Fully-formed command line string (already quoted as needed).

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a non-zero return code.
    """
    header("Running command")
    print(cmd + "\n")
    # shell=True is used to allow one-line strings; Nerfstudio CLIs are console commands.
    subprocess.run(cmd, shell=True, check=True)


def exists_on_path(tool: str) -> bool:
    """Return True if an executable is discoverable on PATH (via shutil.which)."""
    return shutil.which(tool) is not None


def ensure_exists(path: Path, kind: str = "dir") -> None:
    """
    Ensure a path exists (for directories) or fail if a required file is missing.

    Parameters
    ----------
    path : Path
        The file or directory path to validate.
    kind : {"dir", "file"}
        - "dir": creates the directory (parents included) if it does not exist.
        - "file": raises FileNotFoundError if the file does not exist.
    """
    if kind == "dir":
        path.mkdir(parents=True, exist_ok=True)
    elif kind == "file":
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    else:
        raise ValueError("kind must be 'dir' or 'file'")


def check_tools() -> None:
    """
    Validate that required CLIs are on PATH and emit a warning for optional tools.
    """
    required = ["ns-process-data", "ns-train"]
    missing = [t for t in required if not exists_on_path(t)]
    if missing:
        raise RuntimeError("Missing Nerfstudio CLI(s): " + ", ".join(missing))

    if not exists_on_path("ffmpeg"):
        # ffmpeg is not strictly required by Nerfstudio, but often improves handling of videos.
        print("⚠️  ffmpeg not found on PATH (recommended but not strictly required).")


# =========================
# Command builders
# =========================
def build_ns_process_cmd(video_path: Path, dataset_dir: Path) -> str:
    """
    Construct the ``ns-process-data video`` command for dataset building.
    Honors ``NUM_FRAMES_TARGET`` if provided.
    """
    parts = [
        "ns-process-data video",
        f'--data "{str(video_path)}"',
        f'--output-dir "{str(dataset_dir)}"',
    ]
    if NUM_FRAMES_TARGET is not None:
        parts.append(f"--num-frames-target {int(NUM_FRAMES_TARGET)}")
    return " ".join(parts)


def build_ns_train_cmd(dataset_dir: Path, experiments_dir: Path,
                       exp_name: str, max_iters: int) -> str:
    """
    Construct the ``ns-train nerfacto`` command for training the model.
    """
    parts = [
        "ns-train nerfacto",
        f'--data "{str(dataset_dir)}"',
        f'--output-dir "{str(experiments_dir)}"',
        f'--experiment-name "{exp_name}"',
        f"--max-num-iterations {max_iters}",
        "--vis viewer",
        "--viewer.websocket-port 7007",
        "--viewer.quit-on-train-completion True",
    ]
    return " ".join(parts)


# =========================
# Step 1 – Build dataset
# =========================
def step1_build_dataset(video_path: Path, dataset_dir: Path) -> None:
    """
    Create a Nerfstudio dataset from a video using ``ns-process-data video``.

    Notes
    -----
    - If ``dataset_dir`` is **not empty**, the step is skipped to avoid overwriting.
    - This step extracts frames and runs COLMAP to estimate camera poses.
    """
    ensure_exists(dataset_dir, "dir")

    # Safety: if the dataset directory is not empty, assume it already exists and skip.
    if any(dataset_dir.iterdir()):
        print(f"ℹ️  Dataset already exists: {dataset_dir}  (skipping)")
        return

    cmd = build_ns_process_cmd(video_path, dataset_dir)
    run(cmd)


# =========================
# Step 2 – Train NeRF
# =========================
def step2_train_nerf(dataset_dir: Path, experiments_dir: Path,
                     exp_name: str, max_iters: int) -> None:
    """
    Train a nerfacto model using ``ns-train``, writing results under
    ``<experiments_dir>/<exp_name>/nerfacto/<timestamp>/``.
    """
    ensure_exists(experiments_dir, "dir")

    cmd = build_ns_train_cmd(dataset_dir, experiments_dir, exp_name, max_iters)

    print("🌐 Open the training viewer at: http://127.0.0.1:7007")
    run(cmd)


# =========================
# Main
# =========================
def main() -> None:
    """Wire together environment, validation, and the 2-stage pipeline."""
    # Make sure the selected GPU(s) are used by downstream libraries (PyTorch, etc.).
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # Early check for required tools
    check_tools()

    # Ensure expected directories and files exist
    videos_dir = PROJECT_DIR / "videos"
    ensure_exists(videos_dir, "dir")
    ensure_exists(DATASET_ROOT, "dir")
    ensure_exists(EXPERIMENT_ROOT, "dir")

    video_path = videos_dir / VIDEO_FILE
    ensure_exists(video_path, "file")

    # Dataset path for this video:
    #   <PROJECT_DIR>/outputs/dataset/<video_stem>/
    dataset_dir = DATASET_ROOT / video_path.stem

    # ---- Step 1: Build dataset (unless skipped) ----
    if not SKIP_DATASET:
        step1_build_dataset(video_path, dataset_dir)
    else:
        print("⏭️  Skipping Step 1 (dataset prep)")

    # ---- Step 2: Train nerfacto (unless skipped) ----
    if not SKIP_TRAIN:
        step2_train_nerf(dataset_dir, EXPERIMENT_ROOT, EXPERIMENT_NAME, MAX_ITERS)
    else:
        print("⏭️  Skipping Step 2 (training)")

    # Wrap-up
    print("\nDone ✅ – Steps 1 & 2 completed.")
    print(f"• Dataset path:     {dataset_dir}")
    print(f"• Experiments root: {EXPERIMENT_ROOT}")
    print(f"• Experiment name:  {EXPERIMENT_NAME}")
    print("Next (optional): ns-export pointcloud → Poisson meshing → ns-render turntable.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        # Surface the called command's exit code and textual form for quick debugging.
        print(f"\n❌ Command failed with exit code {e.returncode}\n{e}\n")
        sys.exit(e.returncode)
    except Exception as e:
        # Any other exception is printed plainly and returned as exit code 1.
        print(f"\n❌ {e}\n")
        sys.exit(1)
