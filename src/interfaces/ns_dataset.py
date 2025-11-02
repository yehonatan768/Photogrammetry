from __future__ import annotations

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, Any

from src.config.base_config import (
    PROJECT_DIR,
    VIDEOS_DIR,
    DATASET_DIR,
    ensure_core_dirs,
)
from src.config.config_nerfstudio import (
    CUDA_VISIBLE_DEVICES,
    VIDEO_EXTS,
)
from src.config.config_ffmpeg import NUM_FRAMES_TARGET, USE_CUSTOM_FFMPEG
from src.utils.ns_utils import (
    run,
    ensure_exists,
    exists_on_path,
)
from src.utils.menu import _ask_choice  # adjust if menu is elsewhere
from src.interfaces.ffmpeg_extract import extract_frames_ffmpeg


# =========================
# internal helpers
# =========================

def _check_tools() -> None:
    """
    Verify required external tools exist before we start:
    - ns-process-data (Nerfstudio dataset builder)
    - ffmpeg (must exist if USE_CUSTOM_FFMPEG=True, optional otherwise)

    Raises:
        RuntimeError: if required CLI tools are missing.
    """
    # ns-process-data is ALWAYS required
    required = ["ns-process-data"]
    missing = [t for t in required if not exists_on_path(t)]
    if missing:
        raise RuntimeError(
            "Missing Nerfstudio CLI(s) on PATH: " + ", ".join(missing)
        )

    # ffmpeg warning / requirement
    if not exists_on_path("ffmpeg"):
        if USE_CUSTOM_FFMPEG:
            raise RuntimeError("ffmpeg not found on PATH but USE_CUSTOM_FFMPEG=True.")
        print("⚠️  ffmpeg not found on PATH (recommended for video decoding).")


def _list_videos(videos_dir: Path) -> list[Path]:
    """
    Return all allowed video files under `videos_dir`, newest first.
    Allowed extensions are defined in VIDEO_EXTS.
    """
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos folder not found: {videos_dir}")

    vids = [
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    ]
    if not vids:
        raise FileNotFoundError(
            f"No video files found in {videos_dir}. "
            f"Put e.g. DJI0001.mp4 there."
        )

    vids.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return vids


def _choose_video(videos_dir: Path) -> Path:
    """
    Interactive picker (by number / name / prefix).
    If only one video exists, auto-select it.
    """
    vids = _list_videos(videos_dir)

    if len(vids) == 1:
        only = vids[0]
        print(f"\nFound single video: {only.name} (auto-selected)")
        return only

    print("\nAvailable videos:\n")
    names: list[str] = []
    for i, v in enumerate(vids, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  {i}) {v.name}  ({size_mb:.1f} MB)")
        names.append(v.name)

    idx = _ask_choice(
        f"\nChoose video [1-{len(vids)} or name/prefix]: ",
        names,
    )
    return vids[idx]


# ---------- dataset version naming ----------

_version_regex = re.compile(r"^(?P<base>.+?)_ver(?P<num>\d+)$")


def _split_ver(name: str) -> tuple[str, int | None]:
    """
    "Barn_ver3" -> ("Barn", 3)
    "Barn"      -> ("Barn", None)
    """
    m = _version_regex.match(name)
    if not m:
        return name, None
    return m.group("base"), int(m.group("num"))


def _make_unique_dataset_dir(base_stem: str) -> Path:
    """
    Decide where to write the *next* dataset for this video.

    Strategy:
    - Prefer DATASET_DIR/<base_stem> if not used yet.
    - Otherwise create DATASET_DIR/<base_stem>_verK with next available K.

    Example:
        base_stem = "Barn"
        Existing: Barn/, Barn_ver1/, Barn_ver2/
        New run:  Barn_ver3/
    """
    ensure_exists(DATASET_DIR, "dir")

    existing: list[str] = []
    for p in DATASET_DIR.iterdir():
        if p.is_dir() and (p.name == base_stem or p.name.startswith(base_stem + "_ver")):
            existing.append(p.name)

    # First run -> plain "<base_stem>"
    candidate = DATASET_DIR / base_stem
    if base_stem not in existing:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    # Subsequent runs -> "<base_stem>_verK"
    max_ver = 0
    for name in existing:
        stem, ver = _split_ver(name)
        if stem == base_stem and ver is not None and ver > max_ver:
            max_ver = ver

    new_name = f"{base_stem}_ver{max_ver + 1}"
    candidate = DATASET_DIR / new_name
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


# ---------- ns-process-data command builders ----------

def _build_ns_process_cmd_video(video_path: Path, dataset_dir: Path) -> str:
    """
    Nerfstudio handles:
    - extracting frames internally from the video
    - running COLMAP
    - generating transforms.json

    This path is used if USE_CUSTOM_FFMPEG == False.
    """
    parts = [
        "ns-process-data video",
        f'--data "{str(video_path)}"',
        f'--output-dir "{str(dataset_dir)}"',
    ]
    if NUM_FRAMES_TARGET is not None:
        parts.append(f"--num-frames-target {int(NUM_FRAMES_TARGET)}")
    return " ".join(parts)


def _build_ns_process_cmd_images(frames_dir: Path, dataset_dir: Path) -> str:
    """
    Build the ns-process-data command for image mode.

    We:
    - point Nerfstudio at the extracted frames directory
    - tell it where to write the dataset
    - force a matching method that does NOT rely on vocab_tree
      to avoid the FAISS/FLANN vocab tree incompatibility in new COLMAP.

    Options for --matching-method that are common:
      - sequential  (good for video trajectories)
      - exhaustive  (heavier, all-vs-all)
      - vocab_tree  (fast, but currently breaking for you)

    We'll use 'sequential' here.
    """
    parts = [
        "ns-process-data images",
        f'--data "{str(frames_dir)}"',
        f'--output-dir "{str(dataset_dir)}"',
        "--matching-method sequential",
    ]
    return " ".join(parts)


def _run_ns_process(video_path: Path, dataset_dir: Path) -> tuple[int, str]:
    """
    Create the dataset in dataset_dir.

    Branch A (USE_CUSTOM_FFMPEG == True):
        1. Extract 'valuable' frames with our ffmpeg pipeline (scene detection,
           fps limit, scaling, etc.) into dataset_dir/images_raw
        2. Call `ns-process-data images ...` to run COLMAP + build transforms.json

    Branch B (USE_CUSTOM_FFMPEG == False):
        1. Call `ns-process-data video ...` and let Nerfstudio handle everything.

    Returns:
        (frame_count, mode)
        frame_count: how many frames we extracted (0 if we let nerfstudio do it internally)
        mode: "custom_ffmpeg" or "nerfstudio_video"
    """
    ensure_exists(dataset_dir, "dir")

    if USE_CUSTOM_FFMPEG:
        frames_dir = dataset_dir / "images_raw"
        frame_count = extract_frames_ffmpeg(video_path, frames_dir, verbose=True)

        cmd = _build_ns_process_cmd_images(frames_dir, dataset_dir)
        run(cmd)

        return frame_count, "custom_ffmpeg"

    # else: let nerfstudio do the extraction from the video
    cmd = _build_ns_process_cmd_video(video_path, dataset_dir)
    run(cmd)
    return 0, "nerfstudio_video"


# =========================
# public API for pipeline
# =========================

def run_dataset() -> Dict[str, Any]:
    """
    Build a Nerfstudio-ready dataset from a chosen video.

    Steps:
    1. Ensure project dirs (videos/, outputs/dataset/, outputs/experiments/, etc.).
    2. Set CUDA_VISIBLE_DEVICES (GPU selection).
    3. Verify external tools (ns-process-data, ffmpeg if needed).
    4. Ask user which video to process.
    5. Create a NEW UNIQUE dataset directory under outputs/dataset/...:
          <video_stem>/, <video_stem>_ver1/, <video_stem>_ver2/, ...
    6. If USE_CUSTOM_FFMPEG == True:
         - extract only "valuable" frames (scene change, etc.) via ffmpeg
         - run `ns-process-data images ...`
       Else:
         - run `ns-process-data video ...`
    7. Print summary and return metadata for next pipeline stage (training).

    Returns:
        {
            "video_path": Path,
            "dataset_dir": Path,
            "video_stem": str,
            "frame_count": int,
            "mode": str,
        }
    """
    # dirs and GPU
    ensure_core_dirs()
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # sanity check tools
    _check_tools()

    # choose source video
    ensure_exists(VIDEOS_DIR, "dir")
    video_path = _choose_video(VIDEOS_DIR)
    ensure_exists(video_path, "file")

    # make new dataset version directory
    base_stem = video_path.stem
    dataset_dir = _make_unique_dataset_dir(base_stem)

    # generate dataset with ns-process-data (and maybe ffmpeg first)
    frame_count, mode = _run_ns_process(video_path, dataset_dir)

    print("\n[DATASET STEP DONE ✅]")
    print(f"• Video file:        {video_path.name}")
    print(f"• Dataset directory: {dataset_dir}")
    print(f"• Custom ffmpeg:     {USE_CUSTOM_FFMPEG}")
    if USE_CUSTOM_FFMPEG:
        print(f"• Frames kept:       {frame_count} (scene-filtered)")
    else:
        print(f"• Frames kept:       handled internally by nerfstudio")

    return {
        "video_path": video_path,
        "dataset_dir": dataset_dir,
        "video_stem": base_stem,
        "frame_count": frame_count,
        "mode": mode,
    }


def main() -> None:
    """
    Standalone runner.
    Usage:
        python -m src.interfaces.ns_dataset
    or:
        python path/to/ns_dataset.py

    This will:
    - ask you which video to use,
    - create a new unique dataset folder,
    - extract good frames (scene-based) if USE_CUSTOM_FFMPEG=True,
    - run COLMAP + transforms,
    - print summary.
    """
    try:
        info = run_dataset()
        print("\nSummary:")
        print(f"- Project root:     {PROJECT_DIR}")
        print(f"- Dataset dir:      {info['dataset_dir']}")
        print(f"- Video stem:       {info['video_stem']}")
        print(f"- Frames kept:      {info['frame_count']}")
        print(f"- Mode:             {info['mode']}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess failed with exit {e.returncode}\n{e}\n")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
