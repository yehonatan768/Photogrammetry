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
from src.config.config_nerfstudio import (
    VIDEO_FILE,
    EXPERIMENT_NAME,
    MAX_ITERS,
    SKIP_DATASET,
    SKIP_TRAIN,
    CUDA_VISIBLE_DEVICES,
    NUM_FRAMES_TARGET,
)


# =========================
# Pretty printing
# =========================
def header(title: str) -> None:
    """Human-friendly section header."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


# =========================
# Shell helpers
# =========================
def run(cmd: str) -> None:
    """
    Execute a shell command and raise if it fails.

    We keep shell=True because nerfstudio exposes console CLIs and we
    want to pass them as one string.
    """
    header("Running command")
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)


def exists_on_path(tool: str) -> bool:
    """Return True if `tool` is discoverable on PATH."""
    return shutil.which(tool) is not None


def ensure_exists(path: Path, kind: str = "dir") -> None:
    """
    Ensure path exists (for dirs) or complain if file is missing.

    kind = "dir": create if missing
    kind = "file": raise if missing
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
    Sanity-check that nerfstudio CLIs are installed and reachable.
    Warn if ffmpeg is missing.
    """
    required = ["ns-process-data", "ns-train"]
    missing = [t for t in required if not exists_on_path(t)]
    if missing:
        raise RuntimeError(
            "Missing Nerfstudio CLI(s) on PATH: " + ", ".join(missing)
        )

    if not exists_on_path("ffmpeg"):
        print("⚠️  ffmpeg not found on PATH (recommended but not strictly required).")


# =========================
# Command builders
# =========================
def build_ns_process_cmd(video_path: Path, dataset_dir: Path) -> str:
    """
    Build the 'ns-process-data video' CLI command to create a Nerfstudio dataset.
    """
    parts = [
        "ns-process-data video",
        f'--data "{str(video_path)}"',
        f'--output-dir "{str(dataset_dir)}"',
    ]
    if NUM_FRAMES_TARGET is not None:
        parts.append(f"--num-frames-target {int(NUM_FRAMES_TARGET)}")
    return " ".join(parts)


def build_ns_train_cmd(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> str:
    """
    Build the 'ns-train nerfacto' CLI command to train NeRF.
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
    Run ns-process-data to generate a Nerfstudio dataset from a video.

    Will skip if dataset_dir is already non-empty, to avoid overwriting.
    """
    ensure_exists(dataset_dir, "dir")

    if any(dataset_dir.iterdir()):
        print(f"ℹ️  Dataset already exists: {dataset_dir}  (skipping)")
        return

    cmd = build_ns_process_cmd(video_path, dataset_dir)
    run(cmd)


# =========================
# Step 2 – Train NeRF
# =========================
def step2_train_nerf(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> None:
    """
    Run ns-train nerfacto and start training.
    """
    ensure_exists(experiments_dir, "dir")

    cmd = build_ns_train_cmd(dataset_dir, experiments_dir, exp_name, max_iters)

    print("🌐 Viewer will be at: http://127.0.0.1:7007")
    run(cmd)


# =========================
# Main pipeline
# =========================
def main() -> None:
    """
    1. Ensure folders exist (videos/, outputs/dataset, outputs/experiments).
    2. Check required tools.
    3. Build dataset from VIDEO_FILE (unless skipped).
    4. Train NeRF into experiments/ (unless skipped).
    """
    # make sure outputs/* exists according to base_config
    ensure_core_dirs()  # creates VIDEOS_DIR, DATASET_DIR, EXPERIMENTS_DIR, etc. :contentReference[oaicite:2]{index=2}

    # GPU selection for downstream libs (torch etc.)
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # confirm scripts we need are installed
    check_tools()

    # --- resolve key paths based on config + user choices ---
    ensure_exists(VIDEOS_DIR, "dir")
    ensure_exists(DATASET_DIR, "dir")
    ensure_exists(EXPERIMENTS_DIR, "dir")

    video_path = VIDEOS_DIR / VIDEO_FILE
    ensure_exists(video_path, "file")

    # dataset for this video goes under outputs/dataset/<video_stem>/
    dataset_dir = DATASET_DIR / video_path.stem

    # ---- Step 1: build dataset ----
    if not SKIP_DATASET:
        step1_build_dataset(video_path, dataset_dir)
    else:
        print("⏭️  Skipping Step 1 (dataset prep)")

    # ---- Step 2: train nerfacto ----
    if not SKIP_TRAIN:
        step2_train_nerf(dataset_dir, EXPERIMENTS_DIR, EXPERIMENT_NAME, MAX_ITERS)
    else:
        print("⏭️  Skipping Step 2 (training)")

    # ---- summary ----
    print("\nDone ✅ – Steps 1 & 2 completed.")
    print(f"• Project root:     {PROJECT_DIR}")
    print(f"• Video file:       {video_path.name}")
    print(f"• Dataset path:     {dataset_dir}")
    print(f"• Experiments dir:  {EXPERIMENTS_DIR}")
    print(f"• Experiment name:  {EXPERIMENT_NAME}")
    print("Next step: ns-export pointcloud → filtering → meshing → renders.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}\n{e}\n")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)
