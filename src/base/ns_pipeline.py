from __future__ import annotations

import os
import sys
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
)
from src.utils.ns_utils import (
    ensure_exists,
    exists_on_path,
)
from src.base.nerfstudio_interface.ns_dataset import build_dataset_from_video
from src.base.nerfstudio_interface.ns_train import train_nerf_model


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


def main() -> None:
    """
    Full pipeline:
    1. Ensure folders exist (videos/, outputs/dataset, outputs/experiments).
    2. Check required tools.
    3. Build dataset from VIDEO_FILE (unless skipped).
    4. Train NeRF into experiments/ (unless skipped).
    """
    # 1. Ensure core dir structure exists
    ensure_core_dirs()  # creates VIDEOS_DIR, DATASET_DIR, EXPERIMENTS_DIR, etc. :contentReference[oaicite:0]{index=0}

    # 2. Pick GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # 3. Check CLIs exist
    check_tools()

    # 4. Resolve paths
    ensure_exists(VIDEOS_DIR, "dir")
    ensure_exists(DATASET_DIR, "dir")
    ensure_exists(EXPERIMENTS_DIR, "dir")

    video_path = VIDEOS_DIR / VIDEO_FILE
    ensure_exists(video_path, "file")

    # dataset for this video goes under outputs/dataset/<video_stem>/
    dataset_dir = DATASET_DIR / video_path.stem

    # 5. Step 1: dataset build
    if not SKIP_DATASET:
        build_dataset_from_video(video_path, dataset_dir)
    else:
        print("⏭️  Skipping Step 1 (dataset prep)")

    # 6. Step 2: training
    if not SKIP_TRAIN:
        train_nerf_model(dataset_dir, EXPERIMENTS_DIR, EXPERIMENT_NAME, MAX_ITERS)
    else:
        print("⏭️  Skipping Step 2 (training)")

    # 7. Summary
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
