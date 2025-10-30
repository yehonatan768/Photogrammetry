# =========================
# User-tunable run config
# =========================

# Name of the source video inside <PROJECT_DIR>/videos/
VIDEO_FILE: str = "DJI0450.mp4"

# Logical name for this training run (used as <EXPERIMENTS_DIR>/<EXPERIMENT_NAME>/...)
EXPERIMENT_NAME: str = "DJI0450_high_quality"

# Training length; larger -> better quality but longer runtime
MAX_ITERS: int = 80_000

# Should we skip dataset building / training if we're rerunning?
SKIP_DATASET: bool = False
SKIP_TRAIN: bool = False

# Which GPU(s) to expose to nerfstudio / torch
CUDA_VISIBLE_DEVICES: str = "0"

# How many frames to sample from video when building dataset
# (None -> let ns-process-data decide)
NUM_FRAMES_TARGET: int | None = 450