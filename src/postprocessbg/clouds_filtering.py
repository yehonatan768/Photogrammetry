from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d
from src.utils.filters import thresholds_from_quantiles, fallback_if_too_few, maybe_run_sor
from src.config.base_config import ensure_core_dirs
from src.utils.pcd import compute_signals, load_point_cloud
from src.utils.menu import (
    choose_preset,
    resolve_export_and_input,
)

from src.utils.preset_loader import (
    load_preset,
    list_presets,
)
from src.utils.common import header, o3d_module, write_ply

ensure_core_dirs()


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
# Low-level helpers (color / z only)
# =========================
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
    """
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

    # 3. compute signals (shared util)
    P = np.asarray(pcd.points)
    radius, s_dens, composite = compute_signals(P)

    # 4. thresholds + selection
    r_thr, d_thr, s_thr = thresholds_from_quantiles(
        radius, s_dens, composite,
        (params.radius_keep_q, params.density_keep_q, params.composite_keep_q))

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
            preset_name = choose_preset("clouds_filtering")
        else:
            preset_name = args.mode
            available = list_presets("clouds_filtering")
            if preset_name not in available:
                raise ValueError(
                    f"Unknown preset '{preset_name}'. "
                    f"Available: {', '.join(available)}"
                )
        # load YAML for that preset into params
        preset_dict = load_preset("clouds_filtering", preset_name)

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
