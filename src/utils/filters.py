# src/utils/filters.py
from __future__ import annotations
import numpy as np
import open3d

from src.utils.common import o3d_module


def thresholds_from_quantiles(radius: np.ndarray, density: np.ndarray,
                              composite: np.ndarray, qs: tuple[float, float, float]):
    """
    Turn quantiles (radius_q, density_q, composite_q) into numeric thresholds.
    """
    r_thr = float(np.quantile(radius, qs[0]))
    d_thr = float(np.quantile(density, qs[1]))
    s_thr = float(np.quantile(composite, qs[2]))
    return r_thr, d_thr, s_thr


def fallback_if_too_few(mask: np.ndarray, radius: np.ndarray, *,
                        min_ratio: float = 0.03) -> np.ndarray:
    """
    If we kept too few (<3% or <1000), relax to radius-only at 99.5% quantile.
    """
    n = len(radius)
    keep_count = int(mask.sum())
    if keep_count >= max(1000, int(n * min_ratio)):
        return mask
    print("⚠️  Too few kept; fallback to radius-only crop (99.5%)")
    r_thr2 = float(np.quantile(radius, 0.995))
    return radius <= r_thr2


def maybe_run_sor(pcd, std_ratio: float | None) -> "open3d.geometry.PointCloud":
    """
    Optional Statistical Outlier Removal.
    """
    if std_ratio is None or len(pcd.points) == 0:
        return pcd
    o3d = o3d_module()
    print("Run SOR …")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20,
                                            std_ratio=float(std_ratio))
    return pcd
