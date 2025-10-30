#!/usr/bin/env python3
r"""
clouds_filtering.py — Cloud/noise reduction for photogrammetry point clouds
========================================================================

What this does (same behavior, clearer structure)
-------------------------------------------------
1) Pick an **export folder** under ``<PROJECT_DIR>/outputs/exports``.
2) Pick an **input .ply** in that folder (``point_cloud.ply``, ``*_filtered.ply``, etc.).
3) Apply a filtering **mode** (off + 10 profiles, light → hard):
   • Center **radius** crop (dense central core)
   • Local **density** gate using kNN distances
   • Optional **SOR** (Statistical Outlier Removal)
   • Optional **color‑based sky removal** (HSV high‑V + low‑S) on medium+ profiles
   • **Fallback** if selection too small: radius‑only keep @ 99.5%
4) Save to ``<mode>_cloudfree.ply`` in the same folder.

Also supports **batch/CLI**:
    py clouds_filtering.py --mode medium --folder <export_dir_name> --input some_cloud.ply
    py clouds_filtering.py --mode hard   --folder auto             --input auto

Design goals
------------
- Keep your **existing logic & thresholds** intact.
- Improve **readability** via small helpers and explicit docstrings.
- Preserve the same CLI flags and interactive menus.

Requirements
------------
- Python 3.9+
- open3d, numpy, tqdm
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


# =========================
# Global Configuration
# =========================
PROJECT_DIR = Path(r"D:\Projects\NerfStudio")
OUTPUTS_DIR = PROJECT_DIR / "outputs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"





# =========================
# UI helpers
# =========================
def header(title: str) -> None:
    """Print a nice section header in the console."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def list_export_folders() -> list[Path]:
    """Return export directories sorted by mtime (newest first)."""
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory at: {EXPORTS_DIR}")
    candidates = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def choose_export_folder_interactive() -> Path:
    """Interactive picker for export directory (newest first)."""
    candidates = list_export_folders()
    if not candidates:
        raise FileNotFoundError(f"No export folders found under: {EXPORTS_DIR}")
    print("\nAvailable export folders:\n")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}) {p.name}")
    while True:
        sel = input(f"\nChoose export folder [1-{len(candidates)}]: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(candidates):
            return candidates[int(sel) - 1]
        print("Invalid choice. Try again.")


def choose_ply_file(export_dir: Path) -> Path:
    """Let the user pick a .ply file inside export_dir (prefers filtered → latest)."""
    ply_files = sorted([p for p in export_dir.glob("*.ply") if p.is_file()],
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {export_dir}")

    if len(ply_files) == 1:
        print(f"\nFound single .ply: {ply_files[0].name} (auto-selected)")
        return ply_files[0]

    print("\nPLY files in folder:\n")
    priority = {"filtered.ply": 0, "light_filtered.ply": 1, "point_cloud.ply": 2, "mesh.ply": 99}
    ply_files.sort(key=lambda p: (priority.get(p.name, 10), -p.stat().st_mtime))

    for i, f in enumerate(ply_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}) {f.name}  ({size_mb:.1f} MB)")

    while True:
        sel = input(f"\nChoose input .ply [1-{len(ply_files)}]: ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(ply_files):
            return ply_files[int(sel) - 1]
        print("Invalid choice. Try again.")


# =========================
# Core filtering – small helpers
# =========================
@dataclass
class FilterParams:
    """Mode hyperparameters (mirrors FILTER_MODES structure)."""
    radius_keep_q: float
    density_keep_q: float
    composite_keep_q: float
    sor_std: Optional[float]
    hsv_v_min: Optional[float] = None
    hsv_s_max: Optional[float] = None


def _o3d() -> "module":
    """Lazy import of open3d to keep import time down and ease testing."""
    import open3d as o3d  # type: ignore
    return o3d


def robust_center(points: np.ndarray) -> np.ndarray:
    """Robust per-axis center using the median (less sensitive to outliers)."""
    return np.median(points, axis=0)


def _z(x: np.ndarray) -> np.ndarray:
    """z-score with tiny epsilon for stability."""
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-9)
    return (x - mu) / sd


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB∈[0,1] to HSV∈[0,1].
    Hue is unused for sky removal, but we keep the signature for clarity.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    v = maxc
    s = np.where(maxc == 0, 0.0, (maxc - minc) / np.clip(maxc, 1e-8, None))
    h = np.zeros_like(v)  # not used
    return np.stack([h, s, v], axis=1)


def _color_sky_mask(colors_01: np.ndarray, hsv_v_min: float, hsv_s_max: float) -> np.ndarray:
    """Detect likely sky/cloud: high‑V (bright) & low‑S (desaturated)."""
    hsv = _rgb_to_hsv(colors_01)
    S, V = hsv[:, 1], hsv[:, 2]
    return (V >= hsv_v_min) & (S <= hsv_s_max)


def load_point_cloud(path: Path) -> "open3d.geometry.PointCloud":
    """Read point cloud and assert it's non-empty."""
    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty")
    return pcd


def knn_k_distance(points: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Distance to the k‑th NN per point (KDTreeFlann), matches your original logic.
    """
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    n = len(points)
    dists = np.empty(n, dtype=np.float32)
    chunk = max(10_000, n // 200)
    for start in tqdm(range(0, n, chunk), desc="Compute kNN distances", unit="pts"):
        end = min(start + chunk, n)
        for i in range(start, end):
            ok, _, sq = kdt.search_knn_vector_3d(pcd.points[i], k + 1)
            dists[i] = float(np.sqrt(sq[-1])) if ok >= k + 1 else np.inf
    return dists


def compute_signals(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per‑point:
      - radius from robust center
      - density proxy = 1 / k‑distance (k=8)
      - composite = z(density) + z(‑radius)
    """
    c = robust_center(points)
    radius = np.linalg.norm(points - c, axis=1)
    d_knn = knn_k_distance(points, k=8)
    s_dens = 1.0 / (1e-9 + d_knn)
    composite = _z(s_dens) + _z(-radius)
    return radius, s_dens, composite


def thresholds_from_quantiles(radius: np.ndarray,
                              s_dens: np.ndarray,
                              composite: np.ndarray,
                              params: FilterParams) -> Tuple[float, float, float]:
    """Return (r_thr, d_thr, s_thr) by applying mode quantiles."""
    r_thr = float(np.quantile(radius, params.radius_keep_q))
    d_thr = float(np.quantile(s_dens, params.density_keep_q))
    s_thr = float(np.quantile(composite, params.composite_keep_q))
    return r_thr, d_thr, s_thr


def apply_selection(radius: np.ndarray,
                    s_dens: np.ndarray,
                    composite: np.ndarray,
                    r_thr: float,
                    d_thr: float,
                    s_thr: float) -> np.ndarray:
    """AND all masks (radius≤, density≥, composite≥)."""
    return (radius <= r_thr) & (s_dens >= d_thr) & (composite >= s_thr)


def fallback_if_too_few(mask: np.ndarray,
                        radius: np.ndarray,
                        n0: int) -> np.ndarray:
    """
    If too few points remain, relax to **radius-only** keep at 99.5%.
    Uses the exact same policy as the original script.
    """
    keep_count = int(mask.sum())
    if keep_count >= max(1000, int(0.03 * n0)):
        return mask
    print("⚠️  Too few kept; fallback to radius-only crop at 99.5% …")
    r_thr2 = float(np.quantile(radius, 0.995))
    return (radius <= r_thr2)


def maybe_color_sky_removal(pcd: "open3d.geometry.PointCloud",
                            params: FilterParams) -> "open3d.geometry.PointCloud":
    """Apply HSV-based sky removal if colors & thresholds exist; otherwise no-op."""
    if params.hsv_v_min is None or params.hsv_s_max is None:
        return pcd
    if not pcd.has_colors():
        return pcd

    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    if len(C) != len(P):
        return pcd

    colors01 = np.clip(C, 0.0, 1.0)
    sky = _color_sky_mask(colors01, params.hsv_v_min, params.hsv_s_max)
    if sky.any():
        print(f"Color-sky removal: dropping {int(sky.sum()):,} pts ({100.0 * sky.mean():.1f}%)")
        keep_idx = np.where(~sky)[0]
        pcd = pcd.select_by_index(keep_idx)
    return pcd


def maybe_run_sor(pcd: "open3d.geometry.PointCloud", sor_std: Optional[float]) -> "open3d.geometry.PointCloud":
    """Run SOR if std_ratio is provided and there are points."""
    if sor_std is None or len(pcd.points) == 0:
        return pcd
    o3d = _o3d()
    print("Run SOR …")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=float(sor_std))
    return pcd


def write_ply(path: Path, pcd: "open3d.geometry.PointCloud") -> Path:
    """Write PLY to disk (binary, uncompressed)."""
    o3d = _o3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    return path


# =========================
# Public filtering routine
# =========================
def filter_clouds(in_ply: Path, out_ply: Path, mode: str) -> Tuple[Path, int, int]:
    """
    Apply the chosen filter **mode** to ``in_ply`` and write ``out_ply``.
    Behavior preserved:
      - mode == "off" → copy
      - color sky removal on medium+ (based on thresholds present)
      - AND of (radius, density, composite); fallback; optional SOR; write
    Returns (out_path, n_in, n_out).
    """
    o3d = _o3d()

    if mode not in FILTER_MODES:
        raise ValueError(f"Unknown filter mode: {mode}")

    if mode == "off":
        pcd = o3d.io.read_point_cloud(str(in_ply))
        write_ply(out_ply, pcd)
        print("Filter: OFF (copied)")
        return out_ply, len(pcd.points), len(pcd.points)

    params = FilterParams(**FILTER_MODES[mode])
    header(f"Cloud filter (mode={mode})")

    # Load cloud
    pcd = load_point_cloud(in_ply)
    n0 = len(pcd.points)
    print(f"Points in: {n0:,}")

    # Optional color sky removal
    pcd = maybe_color_sky_removal(pcd, params)
    if len(pcd.points) == 0:
        print("⚠️  Empty after color-sky removal; copying input as output.")
        pcd = load_point_cloud(in_ply)  # reload original
        write_ply(out_ply, pcd)
        return out_ply, n0, 0

    # Compute signals & thresholds
    P = np.asarray(pcd.points)
    radius, s_dens, composite = compute_signals(P)
    r_thr, d_thr, s_thr = thresholds_from_quantiles(radius, s_dens, composite, params)
    print(f"Thresholds -> radius≤{r_thr:.4g}, density≥{d_thr:.4g}, composite≥{s_thr:.4g}")

    # Selection and fallback
    mask = apply_selection(radius, s_dens, composite, r_thr, d_thr, s_thr)
    print(f"Selected (pre-SOR): {int(mask.sum()):,} / {len(P):,} ({mask.mean():.1%})")

    mask = fallback_if_too_few(mask, radius, n0)
    kept_idx = np.where(mask)[0]
    if len(kept_idx) == 0:
        print("⚠️  Empty after fallback; copying input as output.")
        pcd = load_point_cloud(in_ply)  # reload original
        write_ply(out_ply, pcd)
        return out_ply, n0, 0

    pcd = pcd.select_by_index(kept_idx)

    # Optional SOR and write
    pcd = maybe_run_sor(pcd, params.sor_std)
    write_ply(out_ply, pcd)
    print(f"✅ Done → {out_ply} ({len(pcd.points):,} pts)")
    return out_ply, n0, len(pcd.points)


# =========================
# CLI / Pipeline
# =========================
def resolve_export_and_input(folder_arg: Optional[str], input_arg: Optional[str]) -> Tuple[Path, Path]:
    """
    Resolve export folder and input .ply (supports interactive choice and 'auto').
      - folder_arg: name of folder under EXPORTS_DIR, or 'auto'
      - input_arg : filename within folder, or 'auto' / interactive
    """
    # Resolve export folder
    if folder_arg is None:
        export_dir = choose_export_folder_interactive()
    else:
        if folder_arg.lower() == "auto":
            candidates = list_export_folders()
            if not candidates:
                raise FileNotFoundError(f"No export folders under {EXPORTS_DIR}")
            export_dir = candidates[0]
            print(f"Auto-selected export folder: {export_dir.name}")
        else:
            export_dir = EXPORTS_DIR / folder_arg
            if not export_dir.exists():
                raise FileNotFoundError(f"Export folder not found: {export_dir}")

    # Resolve input .ply
    if input_arg is None:
        in_ply = choose_ply_file(export_dir)
    else:
        if input_arg.lower() == "auto":
            ply_files = sorted([p for p in export_dir.glob("*.ply")], key=lambda p: p.stat().st_mtime, reverse=True)
            priority = {"filtered.ply": 0, "light_filtered.ply": 1, "point_cloud.ply": 2, "mesh.ply": 99}
            ply_files.sort(key=lambda p: (priority.get(p.name, 10), -p.stat().st_mtime))
            if not ply_files:
                raise FileNotFoundError(f"No .ply files in {export_dir}")
            in_ply = ply_files[0]
            print(f"Auto-selected input: {in_ply.name}")
        else:
            in_ply = export_dir / input_arg
            if not in_ply.exists():
                raise FileNotFoundError(f"Input .ply not found: {in_ply}")

    return export_dir, in_ply


def pipeline() -> int:
    """
    Full interactive/CLI flow (unchanged flags & outputs):
      1) --folder / menu → export folder
      2) --input  / menu → input PLY
      3) --mode   / menu → filter profile
      4) Apply filter → write ``<mode>_cloudfree.ply`` (or dry-run printout)
    """
    try:
        parser = argparse.ArgumentParser(description="Cloud/noise reduction for photogrammetry point clouds")
        parser.add_argument("--mode", type=str, default=None, choices=list(FILTER_MODES.keys()),
                            help="Filter mode. If omitted, a menu will appear.")
        parser.add_argument("--folder", type=str, default=None,
                            help="Export folder name under exports, or 'auto' for latest. If omitted, a menu will appear.")
        parser.add_argument("--input", type=str, default=None,
                            help="Input .ply filename in the folder, or 'auto' to auto-pick. If omitted, a menu will appear.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Compute stats and selection ratio without writing output.")
        args = parser.parse_args()

        export_dir, in_ply = resolve_export_and_input(args.folder, args.input)

        # Resolve mode (interactive if needed)
        if args.mode is None:
            print("\nFilter modes:\n")
            keys = list(FILTER_MODES.keys())
            for i, name in enumerate(keys, 1):
                print(f"  {i}) {name}")
            while True:
                sel = input(f"\nChoose mode [1-{len(keys)} or name]: ").strip().lower()
                if sel.isdigit() and 1 <= int(sel) <= len(keys):
                    mode = keys[int(sel) - 1]
                    break
                matches = [k for k in keys if k.startswith(sel)]
                if len(matches) == 1:
                    mode = matches[0]; break
                if sel in keys:
                    mode = sel; break
                print("Invalid choice. Try again.")
        else:
            mode = args.mode

        out_ply = export_dir / f"{mode}_cloudfree.ply"
        if args.dry_run:
            print("DRY RUN: Filtering but not writing output…")
        out_path, n_in, n_out = filter_clouds(in_ply, out_ply, mode)
        if args.dry_run:
            print(f"[DRY] Would write: {out_path}  (kept {n_out:,}/{n_in:,})")
            try:
                out_path.unlink(missing_ok=True)  # remove if written
            except Exception:
                pass

        print("\n=== SUMMARY ===")
        print(f"Export dir : {export_dir}")
        print(f"Input      : {in_ply.name}")
        print(f"Mode       : {mode}")
        print(f"Output     : {out_ply.name}")
        print("Done ✅")
        return 0

    except SystemExit:
        return 1  # argparse already printed help/error
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
