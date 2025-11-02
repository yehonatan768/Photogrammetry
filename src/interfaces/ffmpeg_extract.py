from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from src.config.config_ffmpeg import (
    CUSTOM_FFMPEG_FPS,
    CUSTOM_FFMPEG_SCALE,
    CUSTOM_FFMPEG_TRIM_START,
    CUSTOM_FFMPEG_TRIM_END,
    CUSTOM_SCENE_THRESHOLD,
    USE_SCENE_FILTER,
    NUM_FRAMES_TARGET,
    ALLOW_DYNAMIC_FPS,
)


# ============================================================
# ffprobe helpers
# ============================================================

def _parse_fraction(frac: str) -> Optional[float]:
    """
    Convert strings like "24000/1001" or "30/1" into float FPS.

    Args:
        frac: A string in the form "num/den" or plain number "30".

    Returns:
        float or None if cannot parse.
    """
    if not frac:
        return None
    if "/" in frac:
        num_str, den_str = frac.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None
    # already looks like "29.97" etc
    try:
        return float(frac)
    except ValueError:
        return None


def _probe_video_info(video_path: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """
    Use ffprobe to get:
      - estimated FPS of the source video
      - total frame count (nb_frames)
      - duration in seconds

    We try to read two kinds of metadata:
      1. stream info (nb_frames, r_frame_rate)
      2. container format info (duration)

    Notes:
    - nb_frames may be "N/A" on some codecs/containers.
    - r_frame_rate is usually something like "24000/1001".
    - duration may come from format layer.

    Args:
        video_path: path to the input video.

    Returns:
        (fps, total_frames, duration_sec)
        Any item can be None if not discoverable.
    """
    # Ask ffprobe for stream-level info (first video stream v:0)
    # and container-level duration.
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        raw = subprocess.check_output(cmd, text=True).strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffprobe missing or failed
        return (None, None, None)

    # We expect up to 3 lines, but formats vary. We'll try to be robust.
    # Typical order we get is:
    #   nb_frames
    #   r_frame_rate
    #   duration
    #
    # We'll try to coerce them safely.
    nb_frames_val: Optional[int] = None
    fps_val: Optional[float] = None
    duration_val: Optional[float] = None

    # Heuristic parse: walk through lines and assign by pattern.
    for line in raw:
        # duration is often like "12.345678"
        if "." in line and ":" not in line:
            # might be either duration OR just some float-ish single value
            try:
                as_float = float(line)
            except ValueError:
                as_float = None

            # assign duration if not set yet and looks >0
            if as_float is not None and as_float > 0 and duration_val is None:
                duration_val = as_float
                continue

        # frame rate "24000/1001" or "30/1"
        if "/" in line or line.isdigit():
            maybe_fps = _parse_fraction(line)
            if maybe_fps is not None and maybe_fps > 0:
                # prefer first valid fps only if not already taken
                if fps_val is None:
                    fps_val = maybe_fps
                continue

        # integer number? could be nb_frames
        if line.isdigit():
            try:
                cand_int = int(line)
            except ValueError:
                cand_int = None
            if cand_int is not None and cand_int > 0:
                if nb_frames_val is None:
                    nb_frames_val = cand_int
                continue

    # Fallback: if nb_frames is missing but we have duration and fps,
    # estimate nb_frames ≈ duration * fps
    if nb_frames_val is None and duration_val is not None and fps_val is not None:
        est = int(round(duration_val * fps_val))
        if est > 0:
            nb_frames_val = est

    return (fps_val, nb_frames_val, duration_val)


def _compute_effective_fps(
    video_path: Path,
    base_fps: Optional[float],
    max_frames: Optional[int],
    allow_dynamic: bool,
) -> Optional[float]:
    """
    Decide what FPS we will actually ask ffmpeg to sample.

    Logic:
    1. Determine the "starting" fps (base_fps):
       - If CUSTOM_FFMPEG_FPS is set, that's our base.
       - Else we try to read the source fps using ffprobe.

    2. If ALLOW_DYNAMIC_FPS == True and MAX_FRAMES is set:
       Try to reduce the fps so we won't exceed MAX_FRAMES
       for the ENTIRE clip.
       Idea:
         effective_fps = base_fps / ratio
         ratio = (total_frames / MAX_FRAMES)
       So if total_frames is 2000 and we want 400,
       ratio = 2000/400 = 5  -> effective_fps = base_fps / 5.

       That way ffmpeg never even writes the "extra" frames.

    If we cannot probe frame count / fps / duration, we just
    return base_fps as-is (best effort fallback).

    Args:
        video_path: input video.
        base_fps: starting desired fps (may be None).
        max_frames: max frames desired overall (may be None).
        allow_dynamic: if False, do not downscale fps automatically.

    Returns:
        effective_fps: float or None. If None, we won't put "fps=" in -vf.
    """
    # If no fps hint at all, probe from video.
    probe_fps, probe_total, _ = _probe_video_info(video_path)

    # 1. pick initial fps
    effective_fps = base_fps if base_fps is not None else probe_fps

    # can't do anything if we still don't know fps
    if effective_fps is None:
        return None

    # 2. optionally downscale to respect MAX_FRAMES
    if allow_dynamic and max_frames and probe_total and probe_total > max_frames:
        ratio = probe_total / max_frames
        if ratio > 1.0:
            new_fps = effective_fps / ratio
            # clamp to something >0
            new_fps = max(0.1, new_fps)
            effective_fps = new_fps

    return effective_fps


# ============================================================
# ffmpeg command builders
# ============================================================

def _build_vf_filter(
    effective_fps: Optional[float],
) -> str:
    """
    Build the ffmpeg -vf filter chain string.

    We support two high-level modes:

    MODE A: USE_SCENE_FILTER == False
        "fps=EFFECTIVE_FPS,scale=..."
        We keep a steady temporal sample (e.g. ~5 fps across the clip).
        We do NOT drop by scene-change. This is best for COLMAP,
        because COLMAP likes smooth viewpoint changes and overlap.

    MODE B: USE_SCENE_FILTER == True
        "fps=EFFECTIVE_FPS,select='gt(scene,THRESH)',scale=..."
        We first downsample in time, then keep only frames where the scene
        changed a lot. This reduces similar/near-duplicate views, but can
        become too aggressive and break COLMAP if you remove all the
        gradual parallax.

    Why we always put `fps` first:
        - Putting fps first ensures we are sampling *real* timestamps.
        - If `select` ran first and `fps` after,
          ffmpeg would try to enforce constant FPS by DUPLICATING frames.
          That's exactly the "same frame repeated" bug you had before.

    Args:
        effective_fps: The fps we actually want ffmpeg to sample at.
            If None, we omit the `fps=` filter and let ffmpeg keep full rate.

    Returns:
        Full "-vf" filter chain string for ffmpeg.
    """
    vf_parts: list[str] = []

    # 1. Temporal sampling first.
    if effective_fps is not None:
        # We format with up to ~4 decimal places to avoid too many digits.
        vf_parts.append(f"fps={effective_fps:.4f}")

    # 2. Optional scene-change based thinning.
    if USE_SCENE_FILTER:
        vf_parts.append(f"select='gt(scene,{CUSTOM_SCENE_THRESHOLD})'")

    # 3. Optional spatial scaling last.
    if CUSTOM_FFMPEG_SCALE:
        vf_parts.append(CUSTOM_FFMPEG_SCALE)

    return ",".join(vf_parts)


def _build_ffmpeg_command(
    video_path: Path,
    frames_dir: Path,
    effective_fps: Optional[float],
) -> list[str]:
    """
    Construct the full ffmpeg command to extract frames.

    Output files:
        frames_dir/frame_00001.png
        frames_dir/frame_00002.png
        ...

    Key flags:
    - "-vf <filter_chain>":
        Uses the chain from _build_vf_filter().
        Depending on USE_SCENE_FILTER this either keeps dense adjacent
        frames (good for COLMAP) or aggressively filters (good for disk).

    - "-vsync vfr":
        Variable Frame Rate output.
        This stops ffmpeg from inventing duplicate frames, which means
        each PNG we emit is an actual source frame, not a synthetic repeat.

    - "-ss", "-to":
        Optional trim start/end based on CUSTOM_FFMPEG_TRIM_START / _END.
        If unset, we keep the full clip.

    Args:
        video_path: Source video file.
        frames_dir: Directory where extracted PNGs will be written.
        effective_fps: The FPS we chose after dynamic downscaling.

    Returns:
        ffmpeg command (list[str]) for subprocess.run().
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    vf_filter = _build_vf_filter(effective_fps)
    output_pattern = str(frames_dir / "frame_%05d.png")

    trim_args: list[str] = []
    if CUSTOM_FFMPEG_TRIM_START:
        trim_args += ["-ss", CUSTOM_FFMPEG_TRIM_START]
    if CUSTOM_FFMPEG_TRIM_END:
        trim_args += ["-to", CUSTOM_FFMPEG_TRIM_END]

    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-i", str(video_path),
        *trim_args,
        "-vf", vf_filter,
        "-vsync", "vfr",
        output_pattern,
    ]

    return cmd


def _list_extracted_frames(frames_dir: Path) -> list[Path]:
    """
    Return a sorted list of all extracted frame PNGs.

    Args:
        frames_dir: directory where frames were extracted.

    Returns:
        Sorted list of Paths like frame_00001.png, frame_00002.png, ...
    """
    return sorted(frames_dir.glob("frame_*.png"))


def _prune_to_max_frames(
    frames: list[Path],
    max_frames: Optional[int],
    verbose: bool,
) -> int:
    """
    Safety net AFTER extraction:
    Even though we try to keep frame count low by lowering fps,
    rounding can still overshoot. Also, ALLOW_DYNAMIC_FPS could be False.

    This function downsamples evenly by deleting extras.

    Example:
        we got 2047 frames
        max_frames = 400
        keep_every = floor(2047 / 400) ~= 5
        => keep indices {0,5,10,15,...}

    Args:
        frames: sorted list of Path objects to PNG frames.
        max_frames: desired max frames.
        verbose: print information.

    Returns:
        int: final number of frames left on disk.
    """
    if not max_frames or len(frames) <= max_frames:
        return len(frames)

    total = len(frames)
    # evenly spaced indices to keep
    step = max(1, total // max_frames)
    keep_indices = set(range(0, total, step))

    for idx, path in enumerate(frames):
        if idx not in keep_indices:
            # remove frame from disk
            path.unlink(missing_ok=True)

    # recount what's left
    final_count = len(list(path for idx, path in enumerate(frames) if idx in keep_indices))

    if verbose:
        print(f"⚠️  Limited to {final_count} frames (MAX_FRAMES={max_frames}).")

    return final_count


# ============================================================
# Public API
# ============================================================

def extract_frames_ffmpeg(
    video_path: Path,
    frames_dir: Path,
    verbose: bool = True,
) -> int:
    """
    Extract frames from a video into individual PNG images using ffmpeg.

    Workflow:
    1. Probe the video with ffprobe (fps, frame count, duration).
    2. Decide an effective FPS:
        - Start from CUSTOM_FFMPEG_FPS (or the source fps if CUSTOM_FFMPEG_FPS is None).
        - If ALLOW_DYNAMIC_FPS and MAX_FRAMES are set, lower this fps so
          we theoretically won't exceed MAX_FRAMES over the full clip.
        - This makes extraction much faster on huge videos because we
          never write thousands of PNGs just to delete them.
    3. Build and run ffmpeg with:
        - `fps=...`
        - optional `select='gt(scene,THRESH)'` if USE_SCENE_FILTER is True
        - optional `scale=...`
        - `-vsync vfr` to avoid duplicated frames.
    4. After ffmpeg finishes, list all frames we actually wrote.
    5. As a final safety net:
        - If we STILL exceed MAX_FRAMES (rounding, no dynamic fps, etc.),
          evenly prune frames on disk to end up near MAX_FRAMES.

    Args:
        video_path: Input video file.
        frames_dir: Output directory for the extracted PNGs.
        verbose: If True, print diagnostics, command, and counts.

    Returns:
        int: Final number of frame PNGs left in frames_dir.
    """
    # Decide dynamic FPS before extraction
    effective_fps = _compute_effective_fps(
        video_path=video_path,
        base_fps=CUSTOM_FFMPEG_FPS,
        max_frames=NUM_FRAMES_TARGET,
        allow_dynamic=ALLOW_DYNAMIC_FPS,
    )

    # Build ffmpeg command
    cmd = _build_ffmpeg_command(video_path, frames_dir, effective_fps)

    if verbose:
        print("=" * 70)
        print(">>> Running ffmpeg frame extraction")
        print("=" * 70)
        print("Video:               ", video_path)
        print("Output frames dir:   ", frames_dir)
        print("USE_SCENE_FILTER:    ", USE_SCENE_FILTER)
        print("ALLOW_DYNAMIC_FPS:   ", ALLOW_DYNAMIC_FPS)
        print("MAX_FRAMES:          ", NUM_FRAMES_TARGET)
        print("Chosen effective_fps:", effective_fps)
        print("Command:")
        print(" ".join(cmd))
        print()

    # Run ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if verbose:
            print("ffmpeg failed!")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
        raise RuntimeError(f"ffmpeg extraction failed for {video_path}")

    # List frames actually extracted
    all_frames = _list_extracted_frames(frames_dir)
    frame_count_before_prune = len(all_frames)

    # Final safety pruning if still above MAX_FRAMES
    final_count = _prune_to_max_frames(
        frames=all_frames,
        max_frames=NUM_FRAMES_TARGET,
        verbose=verbose,
    )

    if verbose:
        print(f"Initial extracted frames: {frame_count_before_prune}")
        print(f"Final kept frames:       {final_count}")
        print()

    return final_count
