from __future__ import annotations

import numpy as np
from tqdm import tqdm


def o3d_module():
    """
    Lazy import to avoid hard dependency during module import time.
    """
    import open3d as o3d  # type: ignore
    return o3d


def robust_center(points: np.ndarray) -> np.ndarray:
    """Median per axis, robust to outliers."""
    return np.median(points, axis=0)


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize with epsilon stability."""
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-9)
    return (x - mu) / sd


def knn_k_distance(points: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Distance to the k-th nearest neighbor for each point using Open3D KDTree.
    Queries k+1 neighbors because the first neighbor is the point itself.
    """
    o3d = o3d_module()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    n = len(points)
    dists = np.empty(n, dtype=np.float32)

    # chunking for speed on large clouds
    chunk = max(10_000, n // 200)
    for start in tqdm(range(0, n, chunk), desc="Compute kNN distances", unit="pts"):
        end = min(start + chunk, n)
        for i in range(start, end):
            ok, _, sq = kdt.search_knn_vector_3d(pcd.points[i], k + 1)
            dists[i] = float(np.sqrt(sq[-1])) if ok >= k + 1 else np.inf
    return dists


def compute_signals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    composite = zscore(s_dens) + zscore(-radius)
    return radius, s_dens, composite


def load_point_cloud(path) -> "open3d.geometry.PointCloud":
    """Read a point cloud from disk and validate it is non-empty."""
    o3d = o3d_module()
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise RuntimeError("Input point cloud is empty")
    return pcd
