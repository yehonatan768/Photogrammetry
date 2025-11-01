from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Optional

from src.config.config_nerfstudio import (
    CUSTOM_FFMPEG_FPS,
    CUSTOM_FFMPEG_SCALE,
    CUSTOM_FFMPEG_TRIM_START,
    CUSTOM_FFMPEG_TRIM_END,
)


def build_ffmpeg_command(
    video_path: Path,
    frames_dir: Path,
) -> list[str]:
    """
    Build the ffmpeg command according to config knobs.

    The idea:
    ffmpeg [trim] -i <video> [-vf "...scale...,fps=..."] -vsync 0 frames_dir/frame_%05d.png

    Returns:
        list[str]: argv-style ffmpeg command for subprocess.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []

    # Optional resize filter
    if CUSTOM_FFMPEG_SCALE:
        vf_parts.append(CUSTOM_FFMPEG_SCALE)

    # Optional FPS limiter
    if CUSTOM_FFMPEG_FPS is not None:
        vf_parts.append(f"fps={CUSTOM_FFMPEG_FPS}")

    vf_arg: list[str] = []
    if vf_parts:
        vf_arg = ["-vf", ",".join(vf_parts)]

    # Optional trimming:
    #   -ss <start>  ... seek to start
    #   -to <end>    ... stop at end
    # If you don't want trimming, leave them None.
    trim_args: list[str] = []
    if CUSTOM_FFMPEG_TRIM_START:
        trim_args += ["-ss", CUSTOM_FFMPEG_TRIM_START]
    if CUSTOM_FFMPEG_TRIM_END:
        trim_args += ["-to", CUSTOM_FFMPEG_TRIM_END]

    output_pattern = str(frames_dir / "frame_%05d.png")

    cmd = [
        "ffmpeg",
        *trim_args,
        "-i", str(video_path),
        *vf_arg,
        "-vsync", "0",        # don't duplicate/drop frames
        "-qscale:v", "2",     # good quality (mostly relevant for jpeg, harmless for png)
        output_pattern,
    ]

    return cmd


def extract_frames_ffmpeg(
    video_path: Path,
    frames_dir: Path,
    *,
    verbose: bool = True,
) -> None:
    """
    Extract frames from a source video using ffmpeg into frames_dir.

    This respects config values like CUSTOM_FFMPEG_SCALE, CUSTOM_FFMPEG_FPS,
    CUSTOM_FFMPEG_TRIM_START, CUSTOM_FFMPEG_TRIM_END.

    After running, frames_dir should contain frame_00001.png, frame_00002.png, ...

    Raises:
        RuntimeError: if ffmpeg command fails or produced 0 frames.
    """
    cmd = build_ffmpeg_command(video_path, frames_dir)

    if verbose:
        printable = " ".join(shlex.quote(part) for part in cmd)
        print("\n[FFMPEG EXTRACT]")
        print(printable)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

    produced = list(frames_dir.glob("frame_*.png"))
    if not produced:
        raise RuntimeError(
            f"ffmpeg produced 0 frames in {frames_dir}. "
            "Check FPS / trim / scale settings."
        )

    if verbose:
        print(f"✓ Extracted {len(produced)} frames to {frames_dir}")
