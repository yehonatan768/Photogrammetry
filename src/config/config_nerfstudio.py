# =========================================
# Nerfstudio / Dataset configuration knobs
# =========================================

# How many frames nerfstudio should *try* to sample (only used in built-in video mode).
# If you are doing custom ffmpeg extraction, this is ignored.
NUM_FRAMES_TARGET: int | None = 600

# Which GPU(s) to use, passed as CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES: str = "0"

# Which video file extensions count as valid input videos
VIDEO_EXTS: list[str] = [".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI"]


# =========================================
# Custom ffmpeg extraction pipeline knobs
# =========================================

# If True:
#   1. We run ffmpeg ourselves (with our settings below),
#   2. Then we call `ns-process-data images ...`
# If False:
#   We just call `ns-process-data video ...` and let nerfstudio handle it.
USE_CUSTOM_FFMPEG: bool = True

# Limit FPS when extracting frames. None = don't force FPS.
# Example: 2 means "take ~2 frames per second"
CUSTOM_FFMPEG_FPS: int | None = 2

# Optional scaling filter for ffmpeg (-vf).
# Example: "scale=1920:-1" means resize width to 1920px and keep aspect ratio.
# Set to None to disable scaling.
CUSTOM_FFMPEG_SCALE: str | None = "scale=1920:-1"

# Optional trim:
# Start timestamp in the source video (string "HH:MM:SS.mmm"), or None for start.
CUSTOM_FFMPEG_TRIM_START: str | None = None  # e.g. "00:00:05.000"

# End timestamp in the source video, or None for "until end".
CUSTOM_FFMPEG_TRIM_END: str | None = None    # e.g. "00:00:17.000"

# Training length; larger -> better quality but longer runtime
MAX_ITERS: int = 100_000

# Should we skip dataset building / training if we're rerunning?
SKIP_DATASET: bool = False
SKIP_TRAIN: bool = False