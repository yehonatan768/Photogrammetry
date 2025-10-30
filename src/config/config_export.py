# =========================
# Configuration
# =========================
import shutil

NS_EXPORT = shutil.which("ns-export") or "ns-export"
NORMAL_METHOD = "open3d"
NUM_POINTS_DEFAULT = 6_000_000
REMOVE_OUTLIERS_DEFAULT = False

SUPPRESS_CHILD_WARNINGS = True
CHILD_WARNING_FILTER = "ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning"
NOISY_PATTERNS = [
    r"^WARNING: Using a slow implementation",
    r"FutureWarning:",
    r"RuntimeWarning:",
]