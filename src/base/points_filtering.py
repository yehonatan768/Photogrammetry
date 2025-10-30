"""
points_filtering.py — Core-preserving prefilter for Nerfstudio point clouds
=================================================================

What this does (unchanged behavior)
-----------------------------------
- Lets you select an **export folder** under ``<PROJECT_DIR>/outputs/exports``.
- Always reads the **base** point cloud: ``point_cloud.ply``.
- Asks you to pick a **prefilter mode** from a menu (same keys and thresholds).
- Computes three signals per point:
    1) **radius** from a robust center (median of XYZ)
    2) **local density proxy** = 1 / k-distance (kNN with k=8)
    3) **composite** = z(density) + z(-radius)
- Keeps points that pass **all three** quantile thresholds from the chosen mode.
- If too few points survive, **fallback**: radius-only keep at 99.5% quantile.
- Optional **SOR** outlier removal per mode's ``sor_std``.
- Writes to ``<mode>_filtered.ply`` in the same export folder.

Goal of this refactor
---------------------
- Improve **readability** with small, single-purpose helpers and thorough docstrings.
- **No changes** to thresholds, logic, menu behavior, or outputs.

Dependencies
------------
- Python 3.9+
- open3d, numpy, tqdm

Usage
-----
    py points_filtering.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
from src.config.base_config import EXPORTS_DIR, ensure_core_dirs
ensure_core_dirs()


# =========================
# Configuration (edit here)
# =========================


# Prefilter modes (same thresholds as provided)
PREFILTER_MODES: Dict[str, Dict[str, float | None]] = {
    "off": {},

    # Very permissive (keep nearly everything)
    "ultra_light": {
        "radius_keep_q": 0.9999,
        "density_keep_q": 0.10,
        "composite_keep_q": 0.15,
        "sor_std": None,
    },
    "extra_light": {
        "radius_keep_q": 0.9997,
        "density_keep_q": 0.15,
        "composite_keep_q": 0.20,
        "sor_std": None,
    },

    # Middle ground (renamed "light" -> "medium")
    "light": {
        "radius_keep_q": 0.9995,
        "density_keep_q": 0.20,
        "composite_keep_q": 0.25,
        "sor_std": None,
    },
    "medium_light": {
        "radius_keep_q": 0.998,
        "density_keep_q": 0.30,
        "composite_keep_q": 0.35,
        "sor_std": 2.8,
    },
    "medium": {  # ← your original "light"
        "radius_keep_q": 0.995,
        "density_keep_q": 0.40,
        "composite_keep_q": 0.45,
        "sor_std": 2.5,
    },
    "medium_strong": {
        "radius_keep_q": 0.990,
        "density_keep_q": 0.55,
        "composite_keep_q": 0.50,
        "sor_std": 2.2,
    },

    # Stronger pruning
    "strong": {
        "radius_keep_q": 0.975,
        "density_keep_q": 0.60,
        "composite_keep_q": 0.55,
        "sor_std": 2.0,
    },
    "extra_strong": {
        "radius_keep_q": 0.970,
        "density_keep_q": 0.65,
        "composite_keep_q": 0.58,
        "sor_std": 1.9,
    },

    # Maximum strictness
    "ultra": {
        "radius_keep_q": 0.960,
        "density_keep_q": 0.70,
        "composite_keep_q": 0.60,
        "sor_std": 1.8,
    },
}


# =========================
# Data structures
# =========================
@dataclass
class PrefilterParams:
    """Mode hyperparameters derived from PREFILTER_MODES."""
    radius_keep_q: float
    density_keep_q: float
    composite_keep_q: float
    sor_std: Optional[float]


# =========================
# Small utilities
# =========================
def header(title: str) -> None:
    """Print a pretty section header to the console."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def _o3d() -> "module":
    """Local import to keep import cost low and ease test/mocking."""
    import open3d as o3d  # type: ignore
    return o3d


def robust_center(points: np.ndarray) -> np.ndarray:
    """Return a robust center estimate: per-axis median of points."""
    return np.median(points, axis=0)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize to z-scores with a tiny epsilon for stability."""
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-9)
    return (x - mu) / sd


def load_point_cloud(path: Path) -> "open3d.geometry.PointCloud":
    """Read a point cloud from disk and validate it is non-empty."""
    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty")
    return pcd


def compute_features(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-point:
      - radius from robust center
      - kNN density proxy (1 / k-distance, k=8)
      - composite = z(density) + z(-radius)
    """
    c = robust_center(points)
    radius = np.linalg.norm(points - c, axis=1)

    d_knn = knn_k_distance(points, k=8)
    s_dens = 1.0 / (1e-9 + d_knn)

    composite = _zscore(s_dens) + _zscore(-radius)
    return radius, s_dens, composite


def thresholds_from_quantiles(radius: np.ndarray,
                              s_dens: np.ndarray,
                              composite: np.ndarray,
                              params: PrefilterParams) -> Tuple[float, float, float]:
    """Return (radius_thr, density_thr, composite_thr) as quantiles from params."""
    r_thr = float(np.quantile(radius, params.radius_keep_q))
    d_thr = float(np.quantile(s_dens, params.density_keep_q))
    s_thr = float(np.quantile(composite, params.composite_keep_q))
    return r_thr, d_thr, s_thr


def diagnostic_counts(mask: np.ndarray, total: int, label: str) -> None:
    """Print a one-line diagnostic for a selection mask."""
    print(f"  • pass {label:<9}: {int(mask.sum()):,} / {total:,} ({mask.mean():.1%})")


def knn_k_distance(points: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Return the distance to the k-th nearest neighbor for each point.
    Matches original logic (KDTreeFlann, using sqrt of last squared distance).
    """
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    n = len(points)
    dists = np.empty(n, dtype=np.float32)

    # Chunk to keep UI snappy on big clouds
    chunk = max(10_000, n // 200)
    for start in tqdm(range(0, n, chunk), desc="Compute kNN distances", unit="pts"):
        end = min(start + chunk, n)
        for i in range(start, end):
            ok, _, sq = kdt.search_knn_vector_3d(pcd.points[i], k + 1)
            dists[i] = float(np.sqrt(sq[-1])) if ok >= k + 1 else np.inf
    return dists


def apply_joint_selection(radius: np.ndarray,
                          s_dens: np.ndarray,
                          composite: np.ndarray,
                          r_thr: float,
                          d_thr: float,
                          s_thr: float) -> np.ndarray:
    """
    Apply per-criterion masks and AND them together.
    Returns the final boolean mask and prints diagnostics.
    """
    n = len(radius)
    mask_r = (radius <= r_thr)
    mask_d = (s_dens >= d_thr)
    mask_s = (composite >= s_thr)

    diagnostic_counts(mask_r, n, "radius")
    diagnostic_counts(mask_d, n, "density")
    diagnostic_counts(mask_s, n, "composite")

    return mask_r & mask_d & mask_s


def fallback_if_too_few(keep_mask: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """
    If too few points remain, relax to radius-only keep at 99.5%.
    Mirrors original behavior exactly.
    """
    n = len(radius)
    keep_count = int(keep_mask.sum())
    if keep_count >= max(1000, int(0.03 * n)):
        return keep_mask

    print("⚠️  Kept too few points; relaxing to radius-only crop at 99.5% …")
    r_thr2 = float(np.quantile(radius, 0.995))
    relax_mask = (radius <= r_thr2)

    keep_count = int(relax_mask.sum())
    keep_ratio = keep_count / max(1, n)
    print(f"Radius-only keep: {keep_count:,} points ({keep_ratio:.1%})")

    if keep_count == 0:
        print("⚠️  Still empty after fallback. Copying input as filtered output.")
    return relax_mask


def maybe_run_sor(pcd: "open3d.geometry.PointCloud", sor_std: Optional[float]) -> "open3d.geometry.PointCloud":
    """Run Statistical Outlier Removal if a std_ratio is provided and there are points."""
    if sor_std is None or len(pcd.points) == 0:
        return pcd
    o3d = _o3d()
    print("Run light SOR …")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=float(sor_std))
    return pcd


# =========================
# Interactive menus
# =========================
def choose_export_folder() -> Path:
    """
    Let the user pick an **export folder** under EXPORTS_DIR.
    Newest (by mtime) appears first.
    """
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory at: {EXPORTS_DIR}")

    candidates = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folders found under: {EXPORTS_DIR}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    print("\nAvailable export folders:\n")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}) {p.name}")

    while True:
        sel = input(f"\nChoose export folder [1-{len(candidates)}]: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(candidates):
            return candidates[int(sel) - 1]
        print("Invalid choice. Try again.")


def choose_mode() -> str:
    """
    Dynamic menu built from ``PREFILTER_MODES`` keys (in insertion order).
    Accepts number (1..N) or name/prefix (case-insensitive). Ensures 'off' is first.
    """
    keys = list(PREFILTER_MODES.keys())
    if "off" in keys:
        keys = ["off"] + [k for k in keys if k != "off"]

    print("\nPrefilter modes:\n")
    for i, name in enumerate(keys, 1):
        print(f"  {i}) {name}")
    prompt = f"\nChoose mode [1-{len(keys)} or name]: "

    def resolve(sel: str) -> Optional[str]:
        sel = sel.strip()
        if not sel:
            return None
        if sel.isdigit():
            idx = int(sel)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
            return None
        low = sel.lower()
        exact = [k for k in keys if k.lower() == low]
        if exact:
            return exact[0]
        pref = [k for k in keys if k.lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]
        return None

    while True:
        sel = input(prompt)
        mode = resolve(sel)
        if mode is not None:
            return mode
        print("Invalid choice. Try again.")


# =========================
# Main prefilter routine
# =========================
def prefilter_keep_core(in_ply: Path, out_ply: Path, mode: str) -> Path:
    """
    Apply the prefilter in the chosen mode to ``in_ply`` and write ``out_ply``.
    Behavior matches original script:
      - If mode == "off": copy input to output.
      - Else: compute features → quantile thresholds → AND masks → fallback if needed → optional SOR → save.
    """
    o3d = _o3d()

    if mode not in PREFILTER_MODES:
        raise ValueError(f"Unknown prefilter mode: {mode}")

    if mode == "off":
        o3d.io.write_point_cloud(str(out_ply), o3d.io.read_point_cloud(str(in_ply)))
        print("Prefilter: OFF (copied)")
        return out_ply

    params = PrefilterParams(**PREFILTER_MODES[mode])

    header(f"Prefilter (mode={mode})")
    pcd = load_point_cloud(in_ply)
    P = np.asarray(pcd.points)
    n = len(P)
    print(f"Points in: {n:,}")

    # Features
    radius, s_dens, composite = compute_features(P)

    # Thresholds
    r_thr, d_thr, s_thr = thresholds_from_quantiles(radius, s_dens, composite, params)
    print(f"Thresholds -> radius≤{r_thr:.4g}, density≥{d_thr:.4g}, composite≥{s_thr:.4g}")

    # Selection (with diagnostics)
    keep_mask = apply_joint_selection(radius, s_dens, composite, r_thr, d_thr, s_thr)
    keep_count = int(keep_mask.sum())
    keep_ratio = keep_count / max(1, n)
    print(f"Prefilter selection: {keep_count:,} points ({keep_ratio:.1%})")

    # Fallback (radius-only 99.5%) if needed
    final_mask = fallback_if_too_few(keep_mask, radius)

    # Handle "empty after fallback": copy input (original behavior)
    if int(final_mask.sum()) == 0:
        o3d.io.write_point_cloud(str(out_ply), pcd)
        return out_ply

    # Save selected points
    kept_idx = np.where(final_mask)[0]
    pcd = pcd.select_by_index(kept_idx)

    # Optional SOR
    pcd = maybe_run_sor(pcd, params.sor_std)

    # Write
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False, compressed=False)
    print(f"✅ Done prefilter → {out_ply} ({len(pcd.points):,} pts)")
    return out_ply


# =========================
# CLI pipeline
# =========================
def pipeline() -> int:
    """
    Full interactive flow (unchanged I/O behavior):
      1) Choose export folder
      2) Read ``point_cloud.ply``
      3) Choose prefilter mode
      4) Write ``<mode>_filtered.ply``
    """
    try:
        export_dir = choose_export_folder()

        in_ply = export_dir / "point_cloud.ply"
        if not in_ply.exists():
            raise FileNotFoundError(
                f"Expected base file not found: {in_ply}\n"
                f"Make sure you exported a point cloud (ns-export pointcloud) into this folder."
            )

        mode = choose_mode()
        out_ply = export_dir / f"{mode}_filtered.ply"

        _ = prefilter_keep_core(in_ply, out_ply, mode)

        print("\n=== SUMMARY ===")
        print(f"Export dir:  {export_dir}")
        print(f"Input:       {in_ply.name}")
        print(f"Output:      {out_ply.name}")
        print("Done ✅")
        return 0

    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
