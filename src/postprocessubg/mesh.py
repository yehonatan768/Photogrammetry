from __future__ import annotations

import sys
import numpy as np
from tqdm import tqdm
from src.config.base_config import ensure_core_dirs
from src.utils.menu import (
    choose_export_folder,
    choose_ply_file,
    choose_preset,
)
from src.utils.preset_loader import load_preset

ensure_core_dirs()


def header(title: str) -> None:
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def _o3d() -> "module":
    import open3d as o3d
    return o3d


def _estimate_normals(pcd: "open3d.geometry.PointCloud") -> None:
    o3d = _o3d()
    for _ in tqdm(range(1), desc="Estimate normals"):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.05,
                max_nn=50,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(50)


def _poisson(pcd: "open3d.geometry.PointCloud", depth: int):
    o3d = _o3d()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(depth),
    )
    return mesh, np.asarray(densities)


def _crop_by_density(mesh: "open3d.geometry.TriangleMesh",
                     densities: np.ndarray,
                     q: float) -> np.ndarray:
    keep_mask = densities >= np.quantile(densities, float(q))
    mesh.remove_vertices_by_mask(~keep_mask)
    return densities[keep_mask]


def _principal_axis(points: np.ndarray) -> np.ndarray:
    c = np.median(points, axis=0)
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
    c = np.median(pcd_points, axis=0)
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
        m_s = mesh.filter_smooth_taubin(number_of_iterations=max(1, int(iters)))

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


def _save_mesh(mesh: "open3d.geometry.TriangleMesh",
               out_path: Path) -> Path:
    o3d = _o3d()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(
        str(out_path),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"✅ Mesh saved: {out_path}")
    return out_path


def build_poisson_mesh(pointcloud_ply: Path,
                       out_mesh_ply: Path,
                       *,
                       depth: int,
                       crop_low_q: float,
                       smooth_iters: int,
                       adaptive: bool,
                       adapt_iters: int,
                       adapt_low_q: float,
                       adapt_high_q: float,
                       adapt_gamma: float,
                       cloud_cleanup: bool | dict = False) -> Path:
    o3d = _o3d()

    header("Poisson Mesh")

    pcd = o3d.io.read_point_cloud(str(pointcloud_ply))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty")

    _estimate_normals(pcd)

    mesh, densities = _poisson(pcd, depth=depth)

    dens_kept = _crop_by_density(mesh, densities, q=crop_low_q)

    if cloud_cleanup:
        if isinstance(cloud_cleanup, dict):
            height_q = float(cloud_cleanup.get("height_q", 0.90))
            low_density_q = float(cloud_cleanup.get("low_density_q", 0.30))
        else:
            height_q, low_density_q = 0.90, 0.30
        _cloud_cleanup(
            mesh,
            np.asarray(pcd.points),
            dens_kept,
            height_q,
            low_density_q,
        )

    mesh = _simple_smooth(mesh, iters=int(smooth_iters))
    if adaptive:
        mesh = _adaptive_smooth(
            mesh,
            dens_kept,
            int(adapt_iters),
            float(adapt_low_q),
            float(adapt_high_q),
            float(adapt_gamma),
        )

    _final_cleanup(mesh)
    return _save_mesh(mesh, out_mesh_ply)


def pipeline() -> int:
    try:
        export_dir = choose_export_folder()
        in_ply = choose_ply_file(export_dir)

        preset_name = choose_preset("mesh")

        preset = load_preset("mesh", preset_name)

        out_mesh = export_dir / "mesh.ply"

        build_poisson_mesh(
            in_ply,
            out_mesh,
            depth=int(preset["depth"]),
            crop_low_q=float(preset["crop"]),
            smooth_iters=int(preset["smooth"]),
            adaptive=bool(preset["adaptive"]),
            adapt_iters=int(preset["adapt_iters"]),
            adapt_low_q=float(preset["adapt_low_q"]),
            adapt_high_q=float(preset["adapt_high_q"]),
            adapt_gamma=float(preset["adapt_gamma"]),
            cloud_cleanup=preset.get("cloud_cleanup", False),
        )

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
