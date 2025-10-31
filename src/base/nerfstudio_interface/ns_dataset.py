from __future__ import annotations

from pathlib import Path
from src.utils.ns_utils import run, ensure_exists
from src.config.config_nerfstudio import NUM_FRAMES_TARGET


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


def build_dataset_from_video(video_path: Path, dataset_dir: Path) -> None:
    """
    Run ns-process-data to generate a Nerfstudio dataset from a video.

    Will skip if dataset_dir is already non-empty, to avoid overwriting.
    """
    ensure_exists(dataset_dir, "dir")

    # Don't overwrite an existing processed dataset
    if any(dataset_dir.iterdir()):
        print(f"ℹ️  Dataset already exists: {dataset_dir}  (skipping)")
        return

    cmd = build_ns_process_cmd(video_path, dataset_dir)
    run(cmd)
