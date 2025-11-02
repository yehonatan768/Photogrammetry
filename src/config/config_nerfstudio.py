# =========================================
# Nerfstudio / Dataset configuration knobs
# =========================================
from typing import Optional

# How many frames nerfstudio should *try* to sample (only used in built-in video mode).
# If you are doing custom ffmpeg extraction, this is ignored.

# Which GPU(s) to use, passed as CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES: str = "0"

# Which video file extensions count as valid input videos
VIDEO_EXTS: list[str] = [".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI"]

# Training length; larger -> better quality but longer runtime
MAX_ITERS: int = 100_000

# Should we skip dataset building / training if we're rerunning?
SKIP_DATASET: bool = True
SKIP_TRAIN: bool = True
NUM_RAYS_PER_BATCH: int = 4096
