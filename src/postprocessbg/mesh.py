from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.pcd import o3d_module as _o3d, robust_center
from src.config.base_config import ensure_core_dirs
from src.utils.menu import choose_export_folder, choose_ply_file, choose_preset
from src.utils.preset_loader import load_preset

ensure_core_dirs()


def header(title: str) -> None:
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


@dataclass
class MeshParams:
    depth: int
    crop_low_q: float
    smooth_iters: int
    adaptive: bool
    adapt_iters: int
    adapt_low_q: float
    adapt_high_q: float
    adapt_gamma: float
    cloud_cleanup: bool | dict
    # Performance/quality knobs
    voxel_size: float = 0.0        # >0 → voxel downsample before normals/poisson
    downsample_if_over: int = 2_000_000  # apply voxel only if point count exceeds this
    # Coloring & topology
    colorize: bool = True          # transfer colors from point cloud to mesh
    color_k: int = 1               # NN count for color transfer (1 is faster)
    decimate_target: int = 0       # 0 = off, else target #triangles
    method: str = "poisson"        # "poisson" or "bpa"
    # Saving
    save_obj: bool = False         # disabled by request: save ONLY PLY


def _estimate_normals_scaled(pcd: "open3d.geometry.PointCloud") -> None:
    """
    Estimate normals with a search radius scaled to the scene size
    (derived from the AABB diagonal). Uses fewer neighbors for speed.
    """
    o3d = _o3d()
    aabb = pcd.get_axis_aligned_bounding_box()
    mins = np.asarray(aabb.get_min_bound())
    maxs = np.asarray(aabb.get_max_bound())
    diag = float(np.linalg.norm(maxs - mins))
    # Radius at ~1% of scene diagonal (tweak as needed).
    search_r = max(diag * 0.01, 1e-4)
    max_nn = 30

    for _ in tqdm(range(1), desc="Estimate normals"):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_r,
                max_nn=max_nn,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(max_nn)


def _poisson(pcd: "open3d.geometry.PointCloud", depth: int):
    o3d = _o3d()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth)
    )
    return mesh, np.asarray(densities)


def _bpa(pcd: "open3d.geometry.PointCloud"):
    """Ball Pivoting as an alternative for wire-like / thin parts."""
    o3d = _o3d()
    R = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(R) == 0 or not np.isfinite(R).any():
        raise RuntimeError("Cannot estimate distances for BPA radii.")
    avg = float(np.mean(R[np.isfinite(R)]))
    radii = o3d.utility.DoubleVector([avg * 1.2, avg * 2.0])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )
    densities = np.ones(len(mesh.vertices), dtype=np.float32)
    return mesh, densities


def _crop_by_density(mesh: "open3d.geometry.TriangleMesh",
                     densities: np.ndarray,
                     q: float) -> np.ndarray:
    keep_mask = densities >= np.quantile(densities, float(q))
    mesh.remove_vertices_by_mask(~keep_mask)
    return densities[keep_mask]


def _principal_axis(points: np.ndarray) -> np.ndarray:
    c = robust_center(points)
    X = points - c
    cov = (X.T @ X) / max(1, len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)
    up = evecs[:, np.argmax(evals)]
    norm = np.linalg.norm(up)
    return up if norm == 0 else (up / norm)


def _cloud_cleanup(mesh: "open3d.geometry.TriangleMesh",
                   pcd_points: np.ndarray,
                   dens_kept: np.ndarray,
                   height_q: float,
                   low_density_q: float) -> None:
    c = robust_center(pcd_points)
    up = _principal_axis(pcd_points)

    V = np.asarray(mesh.vertices)
    heights = (V - c) @ up

    h_thr = float(np.quantile(heights, height_q))
    d_thr = float(np.quantile(dens_kept, low_density_q)) if len(dens_kept) else np.inf

    sky_mask = (heights > h_thr) & (dens_kept < d_thr)
    if sky_mask.any():
        mesh.remove_vertices_by_mask(sky_mask)


def _simple_smooth(mesh: "open3d.geometry.TriangleMesh",
                   iters: int) -> "open3d.geometry.TriangleMesh":
    o3d = _o3d()
    if iters <= 0:
        return mesh
    for _ in tqdm(range(iters), desc="Simple smoothing"):
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
    return mesh


def _adaptive_smooth(mesh: "open3d.geometry.TriangleMesh",
                     dens_kept: np.ndarray,
                     iters: int,
                     low_q: float,
                     high_q: float,
                     gamma: float) -> "open3d.geometry.TriangleMesh":
    if iters <= 0:
        return mesh

    o3d = _o3d()
    for _ in tqdm(range(1), desc="Adaptive smoothing"):
        m_s = mesh.filter_smooth_taubin(number_of_iterations=max(1), lambda_filter=0.5, mu_filter=-0.53)

        v0 = np.asarray(mesh.vertices)
        v1 = np.asarray(m_s.vertices)

        try:
            import scipy.spatial as sps  # type: ignore
            kd = sps.cKDTree(v0)
            _, idx = kd.query(v0, k=1)
            dens_for_v = dens_kept[idx] if len(dens_kept) else np.ones(len(v0))
        except Exception:
            dens_for_v = np.ones(len(v0))

        if high_q != low_q:
            lo = np.quantile(dens_for_v, float(low_q))
            hi = np.quantile(dens_for_v, float(high_q))
        else:
            lo, hi = float(dens_for_v.min()), float(dens_for_v.max())

        if hi > lo:
            w = (dens_for_v - lo) / (hi - lo)
            w = np.clip(w, 0.0, 1.0) ** float(gamma)
        else:
            w = np.zeros_like(dens_for_v)

        V = v0 * (1.0 - w)[:, None] + v1 * w[:, None]
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.compute_vertex_normals()

    return mesh


def _final_cleanup(mesh: "open3d.geometry.TriangleMesh") -> None:
    for _ in tqdm(range(1), desc="Mesh cleanup"):
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()


def _colorize_from_pcd(mesh: "open3d.geometry.TriangleMesh",
                       pcd: "open3d.geometry.PointCloud",
                       k: int = 1) -> None:
    """Transfer colors to mesh vertices from nearest k PCD points (average)."""
    if not pcd.has_colors():
        return
    o3d = _o3d()
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) == 0:
        return
    try:
        import scipy.spatial as sps  # type: ignore
        kd = sps.cKDTree(pts)
        V = np.asarray(mesh.vertices)
        _, idx = kd.query(V, k=max(1, int(k)))
        if k == 1:
            C = cols[idx]
        else:
            C = cols[idx].mean(axis=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0.0, 1.0))
    except Exception:
        # Fallback to 1-NN using Open3D KDTree
        o3d_kd = o3d.geometry.KDTreeFlann(pcd)
        V = np.asarray(mesh.vertices)
        out_cols = np.zeros((len(V), 3), dtype=np.float32)
        for i in tqdm(range(len(V)), desc="Colorize (fallback 1-NN)"):
            ok, idx, _ = o3d_kd.search_knn_vector_3d(V[i], 1)
            if ok > 0:
                out_cols[i] = cols[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(out_cols)


def _decimate(mesh: "open3d.geometry.TriangleMesh",
              target_tris: int) -> "open3d.geometry.TriangleMesh":
    if int(target_tris) <= 0:
        return mesh
    o3d = _o3d()
    cur = np.asarray(mesh.triangles).shape[0]
    if cur <= target_tris:
        return mesh
    print(f"Decimate: {cur} → {target_tris} triangles (quadric) …")
    mesh = mesh.simplify_quadric_decimation(target_tris)
    mesh.compute_vertex_normals()
    return mesh


def _save_ply_only(mesh: "open3d.geometry.TriangleMesh",
                   out_ply: Path) -> None:
    """Save ONLY PLY (no OBJ)."""
    o3d = _o3d()
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(
        str(out_ply),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"PLY saved: {out_ply}")


def build_mesh(pointcloud_ply: Path, out_mesh_ply: Path, params: MeshParams) -> Path:
    o3d = _o3d()
    header("Surface Reconstruction")

    pcd = o3d.io.read_point_cloud(str(pointcloud_ply))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty")

    # Optional voxel downsample (only if large enough)
    if params.voxel_size and params.voxel_size > 0.0:
        n0 = len(pcd.points)
        if n0 > int(params.downsample_if_over):
            print(f"Voxel downsample ({params.voxel_size}) on {n0:,} pts …")
            pcd = pcd.voxel_down_sample(voxel_size=float(params.voxel_size))
            print(f"→ {len(pcd.points):,} pts after voxel")

    # Scaled normals
    _estimate_normals_scaled(pcd)

    # Reconstruction
    if params.method.lower() == "bpa":
        mesh, densities = _bpa(pcd)
        dens_kept = densities  # uniform placeholder for later steps
    else:
        mesh, densities = _poisson(pcd, depth=params.depth)
        dens_kept = _crop_by_density(mesh, densities, q=params.crop_low_q)

    # Optional cloud/sky cleanup
    if params.cloud_cleanup:
        if isinstance(params.cloud_cleanup, dict):
            height_q = float(params.cloud_cleanup.get("height_q", 0.90))
            low_density_q = float(params.cloud_cleanup.get("low_density_q", 0.30))
        else:
            height_q, low_density_q = 0.90, 0.30
        _cloud_cleanup(
            mesh,
            np.asarray(pcd.points),
            dens_kept,
            height_q,
            low_density_q,
        )

    # Smoothing
    mesh = _simple_smooth(mesh, iters=int(params.smooth_iters))
    if params.adaptive:
        mesh = _adaptive_smooth(
            mesh,
            dens_kept,
            int(params.adapt_iters),
            float(params.adapt_low_q),
            float(params.adapt_high_q),
            float(params.adapt_gamma),
        )

    # Decimate BEFORE colorization (faster coloring on fewer vertices)
    mesh = _decimate(mesh, int(params.decimate_target))

    # Color transfer from PCD
    if params.colorize:
        _colorize_from_pcd(mesh, pcd, k=int(params.color_k))

    # Cleanup & normals
    _final_cleanup(mesh)
    mesh.compute_vertex_normals()

    # Save ONLY PLY
    _save_ply_only(mesh, out_mesh_ply)
    return out_mesh_ply


def pipeline() -> int:
    try:
        export_dir = choose_export_folder()
        in_ply = choose_ply_file(export_dir)

        preset_name = choose_preset("mesh")
        preset = load_preset("mesh", preset_name)

        params = MeshParams(
            depth=int(preset["depth"]),
            crop_low_q=float(preset["crop"]),
            smooth_iters=int(preset["smooth"]),
            adaptive=bool(preset["adaptive"]),
            adapt_iters=int(preset["adapt_iters"]),
            adapt_low_q=float(preset["adapt_low_q"]),
            adapt_high_q=float(preset["adapt_high_q"]),
            adapt_gamma=float(preset["adapt_gamma"]),
            cloud_cleanup=preset.get("cloud_cleanup", False),
            # Optional/new keys with safe defaults:
            voxel_size=float(preset.get("voxel_size", 0.0)),
            downsample_if_over=int(preset.get("downsample_if_over", 2_000_000)),
            colorize=bool(preset.get("colorize", True)),
            color_k=int(preset.get("color_k", 1)),
            decimate_target=int(preset.get("decimate_target", 0)),
            method=str(preset.get("method", "poisson")),
            save_obj=bool(preset.get("save_obj", False)),  # ignored; always PLY only
        )

        out_mesh = export_dir / "mesh.ply"
        build_mesh(in_ply, out_mesh, params)

        print("\n=== SUMMARY ===")
        print(f"Export dir:  {export_dir}")
        print(f"Input PLY:   {in_ply.name}")
        print(f"Preset:      {preset_name}")
        print(f"Output:      {out_mesh.name}")
        print("Done ✅")
        return 0

    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
