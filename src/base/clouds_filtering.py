from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.config.base_config import EXPORTS_DIR, ensure_core_dirs
from src.utils.menu import (
    choose_export_folder,
    choose_ply_file,
    choose_preset,
)
from src.utils.preset_loader import (
    load_preset,
    list_presets,
)

ensure_core_dirs()


# =========================
# UI helpers
# =========================
def header(title: str) -> None:
    """Pretty console section header."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


# =========================
# Filtering parameter struct
# =========================
@dataclass
class FilterParams:
    """
    Parameters for the clouds/noise cleanup stage.

    Matches the YAML structure under profiles/clouds_filtering/*.yaml:
      radius_keep_q: float
      density_keep_q: float
      composite_keep_q: float
      sor_std: Optional[float]
      hsv_v_min: Optional[float]
      hsv_s_max: Optional[float]
    """
    radius_keep_q: float
    density_keep_q: float
    composite_keep_q: float
    sor_std: Optional[float]
    hsv_v_min: Optional[float] = None
    hsv_s_max: Optional[float] = None


# =========================
# Low-level helpers
# =========================
def _o3d() -> "module":
    """
    Lazy import of open3d so importing this file doesn't eagerly require it.
    """
    import open3d as o3d  # type: ignore
    return o3d


def robust_center(points: np.ndarray) -> np.ndarray:
    """Median per axis, more stable than mean with outliers."""
    return np.median(points, axis=0)


def _z(x: np.ndarray) -> np.ndarray:
    """z-score normalize with tiny epsilon."""
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-9)
    return (x - mu) / sd


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0,1] -> HSV [0,1].
    We only care about S (saturation) and V (value/brightness) for sky removal.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.max(rgb, axis=1)
    minc = np.min(rgb, axis=1)
    v = maxc
    s = np.where(
        maxc == 0.0,
        0.0,
        (maxc - minc) / np.clip(maxc, 1e-8, None),
    )
    h = np.zeros_like(v)  # hue not used here
    return np.stack([h, s, v], axis=1)


def _color_sky_mask(
    colors_01: np.ndarray,
    hsv_v_min: float,
    hsv_s_max: float,
) -> np.ndarray:
    """
    Heuristic sky detector:
    Bright (V >= hsv_v_min) and low saturation (S <= hsv_s_max).
    """
    hsv = _rgb_to_hsv(colors_01)
    S, V = hsv[:, 1], hsv[:, 2]
    return (V >= hsv_v_min) & (S <= hsv_s_max)


def load_point_cloud(path: Path) -> "open3d.geometry.PointCloud":
    """Read PLY and make sure it's not empty."""
    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty")
    return pcd


def knn_k_distance(points: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Approximate density via k-th neighbor distance.
    For each point:
      - build KDTreeFlann
      - query k+1 because first neighbor is itself
      - store sqrt(last_dist)
    """
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    n = len(points)
    dists = np.empty(n, dtype=np.float32)

    # process in chunks to avoid super-slow loops on huge clouds
    chunk = max(10_000, n // 200)
    for start in tqdm(range(0, n, chunk), desc="Compute kNN distances", unit="pts"):
        end = min(start + chunk, n)
        for i in range(start, end):
            ok, _, sq = kdt.search_knn_vector_3d(pcd.points[i], k + 1)
            dists[i] = float(np.sqrt(sq[-1])) if ok >= k + 1 else np.inf
    return dists


def compute_signals(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-point signals:
      radius     = distance from robust center
      s_dens     = density proxy (1 / k-dist)
      composite  = z(s_dens) + z(-radius)
    """
    c = robust_center(points)
    radius = np.linalg.norm(points - c, axis=1)

    d_knn = knn_k_distance(points, k=8)
    s_dens = 1.0 / (1e-9 + d_knn)

    composite = _z(s_dens) + _z(-radius)
    return radius, s_dens, composite


def thresholds_from_quantiles(
    radius: np.ndarray,
    s_dens: np.ndarray,
    composite: np.ndarray,
    params: FilterParams,
) -> Tuple[float, float, float]:
    """
    Turn quantiles from preset YAML into concrete numeric thresholds.
    """
    r_thr = float(np.quantile(radius, params.radius_keep_q))
    d_thr = float(np.quantile(s_dens, params.density_keep_q))
    s_thr = float(np.quantile(composite, params.composite_keep_q))
    return r_thr, d_thr, s_thr


def apply_selection(
    radius: np.ndarray,
    s_dens: np.ndarray,
    composite: np.ndarray,
    r_thr: float,
    d_thr: float,
    s_thr: float,
) -> np.ndarray:
    """
    Boolean mask:
      keep if radius <= r_thr
          AND density >= d_thr
          AND composite >= s_thr
    """
    return (radius <= r_thr) & (s_dens >= d_thr) & (composite >= s_thr)


def fallback_if_too_few(
    mask: np.ndarray,
    radius: np.ndarray,
    n0: int,
) -> np.ndarray:
    """
    Safety valve:
    If we kept almost nothing, relax to "radius only"
    with a very loose 99.5% quantile cutoff.
    """
    keep_count = int(mask.sum())
    if keep_count >= max(1000, int(0.03 * n0)):
        return mask

    print("⚠️  Too few kept; fallback to radius-only crop at 99.5% …")
    r_thr2 = float(np.quantile(radius, 0.995))
    return (radius <= r_thr2)


def maybe_color_sky_removal(
    pcd: "open3d.geometry.PointCloud",
    params: FilterParams,
) -> "open3d.geometry.PointCloud":
    """
    Optional color-based "sky" cleanup:
    remove bright-low-sat points before geometric filtering.
    """
    if params.hsv_v_min is None or params.hsv_s_max is None:
        return pcd
    if not pcd.has_colors():
        return pcd

    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    if len(C) != len(P):
        return pcd

    colors01 = np.clip(C, 0.0, 1.0)
    sky_mask = _color_sky_mask(colors01, params.hsv_v_min, params.hsv_s_max)
    if sky_mask.any():
        print(
            f"Color-sky removal: dropping {int(sky_mask.sum()):,} pts "
            f"({100.0 * sky_mask.mean():.1f}%)"
        )
        keep_idx = np.where(~sky_mask)[0]
        pcd = pcd.select_by_index(keep_idx)
    return pcd


def maybe_run_sor(
    pcd: "open3d.geometry.PointCloud",
    sor_std: Optional[float],
) -> "open3d.geometry.PointCloud":
    """
    Optional Statistical Outlier Removal (SOR).
    """
    if sor_std is None or len(pcd.points) == 0:
        return pcd
    o3d = _o3d()
    print("Run SOR …")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=float(sor_std),
    )
    return pcd


def write_ply(path: Path, pcd: "open3d.geometry.PointCloud") -> Path:
    """Save point cloud to PLY (binary)."""
    o3d = _o3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(
        str(path),
        pcd,
        write_ascii=False,
        compressed=False,
    )
    return path


# =========================
# Public filtering routine
# =========================
def filter_clouds(
    in_ply: Path,
    out_ply: Path,
    params: FilterParams,
) -> Tuple[Path, int, int]:
    """
    Apply full cloud filtering pipeline and write result to out_ply.

    Steps:
    1. Load point cloud.
    2. Optional color-based sky removal.
    3. Compute geometric/density signals.
    4. Select core points by quantiles (radius, density, composite).
    5. Fallback if too tiny.
    6. Optional SOR.
    7. Save.

    Returns
    -------
    (out_path, n_in, n_out)
    """
    o3d = _o3d()

    header("Cloud filter")

    # 1. load
    pcd = load_point_cloud(in_ply)
    n0 = len(pcd.points)
    print(f"Points in: {n0:,}")

    # 2. optional color-sky cleanup
    pcd = maybe_color_sky_removal(pcd, params)
    if len(pcd.points) == 0:
        print("⚠️  Empty after color-sky removal; copying input as output.")
        pcd = load_point_cloud(in_ply)  # revert to original
        write_ply(out_ply, pcd)
        return out_ply, n0, 0

    # 3. compute signals
    P = np.asarray(pcd.points)
    radius, s_dens, composite = compute_signals(P)

    # 4. thresholds + selection
    r_thr, d_thr, s_thr = thresholds_from_quantiles(radius, s_dens, composite, params)
    print(
        f"Thresholds -> "
        f"radius≤{r_thr:.4g}, density≥{d_thr:.4g}, composite≥{s_thr:.4g}"
    )

    mask = apply_selection(radius, s_dens, composite, r_thr, d_thr, s_thr)
    print(
        f"Selected (pre-SOR): {int(mask.sum()):,} / {len(P):,} "
        f"({mask.mean():.1%})"
    )

    # 5. fallback in case of too few
    mask = fallback_if_too_few(mask, radius, n0)
    kept_idx = np.where(mask)[0]
    if len(kept_idx) == 0:
        print("⚠️  Empty after fallback; copying input as output.")
        pcd = load_point_cloud(in_ply)
        write_ply(out_ply, pcd)
        return out_ply, n0, 0

    pcd = pcd.select_by_index(kept_idx)

    # 6. optional SOR
    pcd = maybe_run_sor(pcd, params.sor_std)

    # 7. save
    write_ply(out_ply, pcd)
    print(f"✅ Done → {out_ply} ({len(pcd.points):,} pts)")
    return out_ply, n0, len(pcd.points)


# =========================
# Helpers for CLI
# =========================
def _auto_pick_latest_export_dir() -> Path:
    """
    Fallback for --folder auto.
    Take newest folder under EXPORTS_DIR.
    """
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory at: {EXPORTS_DIR}")
    cands = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No export folders under {EXPORTS_DIR}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _auto_pick_input_ply(export_dir: Path) -> Path:
    """
    Fallback for --input auto.
    Priority order:
      filtered.ply -> light_filtered.ply -> point_cloud.ply -> mesh.ply
      then newest.
    """
    ply_files = [p for p in export_dir.glob("*.ply") if p.is_file()]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {export_dir}")

    priority = {
        "filtered.ply": 0,
        "light_filtered.ply": 1,
        "point_cloud.ply": 2,
        "mesh.ply": 99,
    }
    ply_files.sort(key=lambda p: (priority.get(p.name, 10), -p.stat().st_mtime))
    return ply_files[0]


def resolve_export_and_input(
    folder_arg: Optional[str],
    input_arg: Optional[str],
) -> Tuple[Path, Path]:
    """
    Resolve which export folder and which .ply to use, based on CLI args.
    """
    # folder
    if folder_arg is None:
        export_dir = choose_export_folder()
    else:
        if folder_arg.lower() == "auto":
            export_dir = _auto_pick_latest_export_dir()
            print(f"Auto-selected export folder: {export_dir.name}")
        else:
            export_dir = EXPORTS_DIR / folder_arg
            if not export_dir.exists():
                raise FileNotFoundError(f"Export folder not found: {export_dir}")

    # input
    if input_arg is None:
        in_ply = choose_ply_file(export_dir)
    else:
        if input_arg.lower() == "auto":
            in_ply = _auto_pick_input_ply(export_dir)
            print(f"Auto-selected input: {in_ply.name}")
        else:
            in_ply = export_dir / input_arg
            if not in_ply.exists():
                raise FileNotFoundError(f"Input .ply not found: {in_ply}")

    return export_dir, in_ply


# =========================
# Main CLI / pipeline
# =========================
def pipeline() -> int:
    """
    Flow:
      1. Resolve export dir (--folder or menu)
      2. Resolve input .ply (--input or menu)
      3. Resolve preset (--mode or menu of YAML presets)
      4. Run filter_clouds()
      5. Write <preset>_cloudfree.ply (unless --dry-run)
    """
    try:
        parser = argparse.ArgumentParser(
            description="Cloud/noise reduction for photogrammetry point clouds"
        )

        # name of preset == yaml filename (without .yaml)
        parser.add_argument(
            "--mode",
            type=str,
            default=None,
            help="Filter preset name (from profiles/clouds_filtering). If omitted, you'll get a menu.",
        )

        parser.add_argument(
            "--folder",
            type=str,
            default=None,
            help="Export folder name under exports, or 'auto' for latest. If omitted, menu.",
        )

        parser.add_argument(
            "--input",
            type=str,
            default=None,
            help="Input .ply filename in that folder, or 'auto'. If omitted, menu.",
        )

        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run filtering and report stats, but don't keep the written file.",
        )

        args = parser.parse_args()

        # 1 & 2: figure out where/which PLY
        export_dir, in_ply = resolve_export_and_input(args.folder, args.input)

        # 3: pick preset
        if args.mode is None:
            # interactive menu from available YAML presets in clouds_filtering/
            preset_name = choose_preset("clouds")
        else:
            preset_name = args.mode
            available = list_presets("clouds")
            if preset_name not in available:
                raise ValueError(
                    f"Unknown preset '{preset_name}'. "
                    f"Available: {', '.join(available)}"
                )
        # load YAML for that preset into params
        preset_dict = load_preset("clouds", preset_name)

        params = FilterParams(
            radius_keep_q=float(preset_dict["radius_keep_q"]),
            density_keep_q=float(preset_dict["density_keep_q"]),
            composite_keep_q=float(preset_dict["composite_keep_q"]),
            sor_std=(
                float(preset_dict["sor_std"])
                if preset_dict.get("sor_std") is not None
                else None
            ),
            hsv_v_min=(
                float(preset_dict["hsv_v_min"])
                if preset_dict.get("hsv_v_min") is not None
                else None
            ),
            hsv_s_max=(
                float(preset_dict["hsv_s_max"])
                if preset_dict.get("hsv_s_max") is not None
                else None
            ),
        )

        # 4: output path
        out_ply = export_dir / f"{preset_name}_cloudfree.ply"

        # 5: run filter
        out_path, n_in, n_out = filter_clouds(in_ply, out_ply, params)

        # if dry-run -> delete the file we just wrote
        if args.dry_run:
            print("DRY RUN: keeping stats only, removing output file…")
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

        # summary
        print("\n=== SUMMARY ===")
        print(f"Export dir : {export_dir}")
        print(f"Input      : {in_ply.name}")
        print(f"Preset     : {preset_name}")
        print(f"Output     : {out_ply.name}")
        print(f"Kept       : {n_out:,} / {n_in:,} points")
        print("Done ✅")
        return 0

    except SystemExit:
        # argparse error/--help already printed
        return 1
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
