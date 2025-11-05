from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from src.utils.pcd import compute_signals, load_point_cloud
from src.config.base_config import ensure_core_dirs
from src.utils.menu import (
    choose_export_folder,
    choose_ply_file,
    choose_preset,
)
from src.utils.filters import thresholds_from_quantiles, fallback_if_too_few, maybe_run_sor
from src.utils.preset_loader import load_preset
from src.utils.common import header, o3d_module

ensure_core_dirs()


# =========================
# Data structures
# =========================
@dataclass
class PrefilterParams:
    """
    Parameters controlling which points we keep.

    These correspond to keys in the YAML profile:
      radius_keep_q: float
      density_keep_q: float
      composite_keep_q: float
      sor_std: Optional[float]
    """
    radius_keep_q: float
    density_keep_q: float
    composite_keep_q: float
    sor_std: Optional[float]


def diagnostic_counts(mask: np.ndarray, total: int, label: str) -> None:
    """Print a one-line diagnostic for a selection mask."""
    print(f"  • pass {label:<9}: {int(mask.sum()):,} / {total:,} ({mask.mean():.1%})")


def apply_joint_selection(
    radius: np.ndarray,
    s_dens: np.ndarray,
    composite: np.ndarray,
    r_thr: float,
    d_thr: float,
    s_thr: float,
) -> np.ndarray:
    """
    Apply all 3 criteria and AND them.
    """
    n = len(radius)
    mask_r = (radius <= r_thr)
    mask_d = (s_dens >= d_thr)
    mask_s = (composite >= s_thr)

    diagnostic_counts(mask_r, n, "radius")
    diagnostic_counts(mask_d, n, "density")
    diagnostic_counts(mask_s, n, "composite")

    return mask_r & mask_d & mask_s


def prefilter_keep_core(
    in_ply: Path,
    out_ply: Path,
    params: PrefilterParams,
) -> Path:
    """
    Filter the point cloud `in_ply` using selection logic derived from params,
    then save it to `out_ply`.
    """
    o3d = o3d_module()

    header("Prefilter")

    pcd = load_point_cloud(in_ply)
    P = np.asarray(pcd.points)
    n = len(P)
    print(f"Points in: {n:,}")

    # 1. features (shared util)
    radius, s_dens, composite = compute_signals(P)

    # 2. thresholds
    r_thr, d_thr, s_thr = thresholds_from_quantiles(
        radius, s_dens, composite,
        (params.radius_keep_q, params.density_keep_q, params.composite_keep_q))
    print(
        f"Thresholds -> "
        f"radius≤{r_thr:.4g}, density≥{d_thr:.4g}, composite≥{s_thr:.4g}"
    )

    # 3. selection
    keep_mask = apply_joint_selection(radius, s_dens, composite, r_thr, d_thr, s_thr)

    keep_count = int(keep_mask.sum())
    keep_ratio = keep_count / max(1, n)
    print(f"Prefilter selection: {keep_count:,} points ({keep_ratio:.1%})")

    # 4. fallback if needed
    final_mask = fallback_if_too_few(keep_mask, radius)

    # 5. empty case after fallback → copy input
    if int(final_mask.sum()) == 0:
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(out_ply), pcd)
        print(f"⚠️ Empty keep_mask. Copied input to {out_ply}.")
        return out_ply

    # 6. subset and optional SOR
    kept_idx = np.where(final_mask)[0]
    pcd_kept = pcd.select_by_index(kept_idx)
    pcd_kept = maybe_run_sor(pcd_kept, params.sor_std)

    # 7. save
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(
        str(out_ply),
        pcd_kept,
        write_ascii=False,
        compressed=False,
    )
    print(f"✅ Done prefilter → {out_ply} ({len(pcd_kept.points):,} pts)")
    return out_ply


def pipeline() -> int:
    """
    Flow:
      1. Pick export folder (under EXPORTS_DIR)
      2. Pick input .ply in that folder
      3. Pick a preset from profiles/points_filtering/*.yaml
      4. Run prefilter_keep_core() and save <presetname>_filtered.ply
    """
    try:
        export_dir = choose_export_folder()
        in_ply = choose_ply_file(export_dir)
        preset_name = choose_preset("points")

        preset_dict = load_preset("points_filtering", preset_name)
        params = PrefilterParams(
            radius_keep_q=float(preset_dict["radius_keep_q"]),
            density_keep_q=float(preset_dict["density_keep_q"]),
            composite_keep_q=float(preset_dict["composite_keep_q"]),
            sor_std=(
                float(preset_dict["sor_std"])
                if preset_dict.get("sor_std") is not None
                else None
            ),
        )

        out_ply = export_dir / f"{preset_name}_filtered.ply"
        prefilter_keep_core(in_ply, out_ply, params)

        print("\n=== SUMMARY ===")
        print(f"Export dir:  {export_dir}")
        print(f"Input:       {in_ply.name}")
        print(f"Preset:      {preset_name}")
        print(f"Output:      {out_ply.name}")
        print("Done ✅")
        return 0

    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
