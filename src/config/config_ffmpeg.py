from __future__ import annotations

# ============================================================
# Frame extraction / ffmpeg configuration
# ============================================================

# Target FPS for sampling.
# - If this is not None, we START from this fps.
# - If this is None, we will try to estimate the real video fps via ffprobe.
CUSTOM_FFMPEG_FPS: float | None = 5.0

# Optional scaling to apply in ffmpeg.
# Example: "scale=1920:-1" keeps width 1920px and preserves aspect ratio.
# Set to "" or None to disable scaling.
CUSTOM_FFMPEG_SCALE: str | None = "scale=1920:-1"

# Optional trimming of the source video.
# Examples: "00:00:03.0" to start from 3s, "00:00:20.0" to stop at 20s.
# Set to None to keep full clip.
CUSTOM_FFMPEG_TRIM_START: str | None = None
CUSTOM_FFMPEG_TRIM_END: str | None = None

# Scene-change filtering.
# If True: drop frames unless the scene has changed a lot.
# If False: keep all temporally sampled frames.
USE_SCENE_FILTER: bool = False
CUSTOM_SCENE_THRESHOLD: float = 0.2  # used only if USE_SCENE_FILTER is True

# ============================================================
# NEW: dataset size / downsampling controls
# ============================================================

# Hard cap on how many frames we want in the final dataset.
# None  -> unlimited
# 400   -> try to end up with ~600 frames total
NUM_FRAMES_TARGET: int = 800

# If True:
#   We will dynamically LOWER the effective fps that ffmpeg uses
#   so that (roughly) we never exceed MAX_FRAMES in the first place.
#   This is faster than extracting thousands of frames and deleting them.
#
# If False:
#   We'll just use CUSTOM_FFMPEG_FPS (or source fps), extract everything,
#   and only afterwards prune down to ~MAX_FRAMES.
ALLOW_DYNAMIC_FPS: bool = False
USE_CUSTOM_FFMPEG = True