"""
mesh.py — Poisson mesh builder from exported Nerfstudio point clouds
===========================================================================

Why this script?
----------------
You already export point clouds from Nerfstudio (e.g., ``point_cloud.ply`` or ``filtered.ply``).
This script **turns that point cloud into a watertight Poisson mesh** while staying
**gentle** on geometry by default. It offers a **menu** to pick:

1) An **export folder** under ``<PROJECT_DIR>/outputs/exports``.
2) A **.ply** file inside that folder (prefers ``filtered.ply``, then ``light_filtered.ply``, then ``point_cloud.ply``).
3) A **mesh preset** (10 pre-tuned profiles ranging from ultra-preserve to focused/high_detail).

It then runs a tight pipeline:
- Estimate normals → Poisson reconstruction → Density crop
- Optional "sky" cleanup (height + low-density gate)
- Optional simple smoothing and/or adaptive density-aware smoothing
- Mesh cleanup + save as ``mesh.ply`` in the same export folder

Design goals
------------
- **Do not change your existing workflow/purpose** — only improve readability & documentation.
- Small, single-purpose helpers with clear docstrings.
- No external behavior changes: same menus, same outputs, same defaults.

Dependencies
------------
- Python 3.9+
- ``open3d`` (mesh ops), ``numpy``, ``tqdm``
- Optional: ``scipy`` for faster nearest-neighbor lookup in adaptive smoothing (falls back if missing).

Usage
-----
PowerShell / CMD:
    py mesh.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm
from src.config.base_config import EXPORTS_DIR, ensure_core_dirs
ensure_core_dirs()

# =========================
# Configuration
# =========================


POISSON_DEPTH_DEFAULT = 12
CROP_LOW_QUANTILE_DEFAULT = 0.02
SMOOTH_ITERS_DEFAULT = 0
ADAPTIVE_SMOOTH_DEFAULT = True
ADAPT_ITERS_DEFAULT = 5
ADAPT_LOW_Q_DEFAULT = 0.30
ADAPT_HIGH_Q_DEFAULT = 0.90
ADAPT_GAMMA_DEFAULT = 1.0


# =========================
# Mesh Presets (10 profiles)
# =========================
MESH_PRESETS = {
    # Super gentle — almost no touching of the cloud
    "ultra_light": dict(
        depth=POISSON_DEPTH_DEFAULT - 2,
        crop=0.002,        # keep ~99.8% of points
        smooth=0,
        adaptive=False,
        adapt_iters=0,
        adapt_low_q=0.50, adapt_high_q=0.90, adapt_gamma=1.0,
        cloud_cleanup=False
    ),

    # Gentle
    "gentle": dict(
        depth=POISSON_DEPTH_DEFAULT - 1,
        crop=0.004,
        smooth=0,
        adaptive=True,     # very mild
        adapt_iters=2,
        adapt_low_q=0.42, adapt_high_q=0.88, adapt_gamma=0.9,
        cloud_cleanup=False
    ),

    # Balanced–gentle (new default flavor)
    "balanced": dict(
        depth=POISSON_DEPTH_DEFAULT - 1,
        crop=0.006,
        smooth=0,
        adaptive=True,
        adapt_iters=3,
        adapt_low_q=0.38, adapt_high_q=0.90, adapt_gamma=0.9,
        cloud_cleanup=False
    ),

    # Conservative geometry changes
    "default": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=max(0.0, CROP_LOW_QUANTILE_DEFAULT * 0.3),  # soften original crop
        smooth=0,
        adaptive=True,
        adapt_iters=max(1, ADAPT_ITERS_DEFAULT // 3),
        adapt_low_q=max(0.30, ADAPT_LOW_Q_DEFAULT),
        adapt_high_q=min(0.90, ADAPT_HIGH_Q_DEFAULT),
        adapt_gamma=0.9,
        cloud_cleanup=False
    ),

    # Crisp but gentle (limits bubbles by avoiding heavy depth/smoothing)
    "crisp": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=0.010,
        smooth=0,
        adaptive=True,
        adapt_iters=3,
        adapt_low_q=0.35, adapt_high_q=0.90, adapt_gamma=0.95,
        cloud_cleanup=False
    ),

    # Focused without aggression
    "focused": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=0.015,
        smooth=0,
        adaptive=True,
        adapt_iters=4,
        adapt_low_q=0.30, adapt_high_q=0.92, adapt_gamma=0.95,
        cloud_cleanup=False
    ),

    # High detail but still restrained
    "high_detail": dict(
        depth=POISSON_DEPTH_DEFAULT + 1,
        crop=0.012,
        smooth=0,
        adaptive=True,
        adapt_iters=3,
        adapt_low_q=0.32, adapt_high_q=0.92, adapt_gamma=0.95,
        cloud_cleanup=False
    ),

    # Artifact cleanup — gentle version
    "artifact_clean": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=0.016,
        smooth=0,
        adaptive=True,
        adapt_iters=3,
        adapt_low_q=0.30, adapt_high_q=0.92, adapt_gamma=0.95,
        cloud_cleanup=dict(  # very light cleanup
            height_q=0.96,     # remove only ~top 4% by height
            low_density_q=0.15
        )
    ),

    # Used to be aggressive — now “ultra” gentle :)
    "ultra": dict(
        depth=POISSON_DEPTH_DEFAULT + 1,
        crop=0.020,
        smooth=0,
        adaptive=True,
        adapt_iters=4,
        adapt_low_q=0.28, adapt_high_q=0.93, adapt_gamma=0.95,
        cloud_cleanup=dict(
            height_q=0.95, low_density_q=0.20
        )
    ),

    # Preserve knowledge — practically no deletion
    "custom": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=0.008,
        smooth=0,
        adaptive=True,
        adapt_iters=3,
        adapt_low_q=0.32, adapt_high_q=0.93, adapt_gamma=0.95,
        cloud_cleanup=False
    ),

    # Maximum preservation — minimal intervention
    "ultra_preserve": dict(
        depth=POISSON_DEPTH_DEFAULT,  # slightly lower than before to reduce bubbles
        crop=0.0,
        smooth=0,
        adaptive=False,
        adapt_iters=0,
        adapt_low_q=0.50, adapt_high_q=0.90, adapt_gamma=1.0,
        cloud_cleanup=False
    ),

    # Soft preservation — tiniest touch
    "preserve_soft": dict(
        depth=POISSON_DEPTH_DEFAULT,
        crop=0.003,
        smooth=0,
        adaptive=True,
        adapt_iters=1,
        adapt_low_q=0.45, adapt_high_q=0.88, adapt_gamma=0.9,
        cloud_cleanup=False
    ),
}


# =========================
# Small utilities
# =========================
def header(title: str) -> None:
    """Print a pretty section header to the console."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def ask_choice(prompt: str, valid_range: range) -> int:
    """Ask the user for a numeric choice in a given range (1-based inclusive)."""
    while True:
        s = input(prompt).strip()
        if s.isdigit():
            i = int(s)
            if i in valid_range:
                return i
        print("Invalid choice. Try again.")


# =========================
# Selection menus
# =========================
def choose_export_folder() -> Path:
    """
    Let the user pick an **export folder** under ``EXPORTS_DIR``.
    The newest folders (by mtime) appear first to reduce scrolling.
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

    idx = ask_choice(f"\nChoose export folder [1-{len(candidates)}]: ", range(1, len(candidates) + 1))
    return candidates[idx - 1]


def choose_ply_file(export_dir: Path) -> Path:
    """
    Let the user pick a **.ply** inside ``export_dir``.

    Priority for convenience:
        filtered.ply → light_filtered.ply → point_cloud.ply → mesh.ply → others by mtime.
    If only one .ply exists, it's auto-selected.
    """
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

    idx = ask_choice(f"\nChoose input .ply [1-{len(ply_files)}]: ", range(1, len(ply_files) + 1))
    return ply_files[idx - 1]


def choose_mesh_preset() -> dict:
    """
    Menu sourced from ``MESH_PRESETS`` (in insertion order).
    Accepts number (1..N) or name/prefix (case-insensitive).
    """
    keys = list(MESH_PRESETS.keys())

    print("\nMesh presets:\n")
    for i, name in enumerate(keys, 1):
        print(f"  {i}) {name}")

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
        sel = input(f"\nChoose preset [1-{len(keys)} or name]: ")
        key = resolve(sel)
        if key is not None:
            return MESH_PRESETS[key]
        print("Invalid choice. Try again.")


# =========================
# Poisson pipeline helpers
# =========================
def _o3d() -> "module":
    """Local import to keep module import cost low when the file is scanned, and to ease mocking."""
    import open3d as o3d  # type: ignore
    return o3d


def _estimate_normals(pcd: "open3d.geometry.PointCloud") -> None:
    """Estimate and orient normals once. Gentle defaults that work for most scenes."""
    o3d = _o3d()
    for _ in tqdm(range(1), desc="Estimate normals"):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
        pcd.orient_normals_consistent_tangent_plane(50)


def _poisson(pcd: "open3d.geometry.PointCloud", depth: int):
    """Run Poisson surface reconstruction and return (mesh, densities array)."""
    o3d = _o3d()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    return mesh, np.asarray(densities)


def _crop_by_density(mesh: "open3d.geometry.TriangleMesh", densities: np.ndarray, q: float) -> np.ndarray:
    """
    Keep vertices whose reconstructed **density** is above the q-quantile.
    Returns the kept densities for downstream use.
    """
    keep_mask = densities >= np.quantile(densities, float(q))
    mesh.remove_vertices_by_mask(~keep_mask)
    return densities[keep_mask]


def _principal_axis(points: np.ndarray) -> np.ndarray:
    """
    Estimate a rough 'up' axis via PCA of the point cloud.
    Returns a unit vector along the largest-variance direction.
    """
    c = np.median(points, axis=0)
    X = points - c
    cov = (X.T @ X) / max(1, len(X) - 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    up = evecs[:, np.argmax(evals)]
    return up / (np.linalg.norm(up) + 1e-12)


def _cloud_cleanup(mesh: "open3d.geometry.TriangleMesh",
                   pcd_points: np.ndarray,
                   dens_kept: np.ndarray,
                   height_q: float,
                   low_density_q: float) -> None:
    """
    Optionally delete 'sky-like' vertices: high along PCA 'up' & under-density.
    Mutates ``mesh`` in-place.
    """
    c = np.median(pcd_points, axis=0)
    up = _principal_axis(pcd_points)
    V = np.asarray(mesh.vertices)
    heights = (V - c) @ up

    h_thr = float(np.quantile(heights, height_q))
    d_thr = float(np.quantile(dens_kept, low_density_q)) if len(dens_kept) else np.inf
    sky_mask = (heights > h_thr) & (dens_kept < d_thr)
    if sky_mask.any():
        mesh.remove_vertices_by_mask(sky_mask)


def _simple_smooth(mesh: "open3d.geometry.TriangleMesh", iters: int) -> "open3d.geometry.TriangleMesh":
    """Run a few iterations of simple smoothing (in-place-like)."""
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
    """
    Density-aware Taubin smoothing:
      - Re-smooth to m_s.
      - Compute per-vertex density weights in [0,1] from quantiles.
      - Blend vertices V = (1-w)*V0 + w*V1.
    """
    if iters <= 0:
        return mesh

    o3d = _o3d()
    for _ in tqdm(range(1), desc="Adaptive smoothing"):
        m_s = mesh.filter_smooth_taubin(number_of_iterations=max(1, int(iters)))
        v0 = np.asarray(mesh.vertices)
        v1 = np.asarray(m_s.vertices)

        # Map 'dens_kept' to current vertices (cheap nearest-neighbor)
        try:
            import scipy.spatial as sps  # type: ignore
            kd = sps.cKDTree(v0)
            _, idx = kd.query(v0, k=1)
            dens_for_v = dens_kept[idx] if len(dens_kept) else np.ones(len(v0))
        except Exception:
            dens_for_v = np.ones(len(v0))  # fallback: uniform weights

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
    """Remove degeneracies and fix non-manifold issues (mutates mesh)."""
    for _ in tqdm(range(1), desc="Mesh cleanup"):
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()


def _save_mesh(mesh: "open3d.geometry.TriangleMesh", out_path: Path) -> Path:
    """Write the mesh to disk (PLY with colors & normals)."""
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


# =========================
# Public build function
# =========================
def build_poisson_mesh(pointcloud_ply: Path, out_mesh_ply: Path, *,
                       depth: int,
                       crop_low_q: float,
                       smooth_iters: int,
                       adaptive: bool,
                       adapt_iters: int,
                       adapt_low_q: float,
                       adapt_high_q: float,
                       adapt_gamma: float,
                       cloud_cleanup: bool | dict = False) -> Path:
    """
    Build a Poisson mesh from a given point cloud file.

    Parameters
    ----------
    pointcloud_ply : Path
        Input .ply point cloud path (exported by Nerfstudio).
    out_mesh_ply : Path
        Destination .ply mesh path (usually ``export_dir / "mesh.ply"``).
    depth : int
        Poisson reconstruction depth (controls detail & runtime).
    crop_low_q : float
        Keep vertices with densities >= this **quantile** (0..1). Higher → stricter crop.
    smooth_iters : int
        Number of simple-smoothing iterations. 0 = none.
    adaptive : bool
        Whether to apply adaptive (density-aware) smoothing.
    adapt_iters : int
        Iterations for the Taubin adaptive smoother.
    adapt_low_q, adapt_high_q : float
        Quantiles to normalize density weights.
    adapt_gamma : float
        Exponent for weight sharpening (>=0). 1.0 = linear, <1 softer, >1 sharper.
    cloud_cleanup : bool | dict
        If truthy, remove sky-like vertices using height & low-density thresholds.
        If a dict, may include ``height_q`` and ``low_density_q`` keys.

    Returns
    -------
    Path
        The path to the written mesh file.
    """
    o3d = _o3d()

    header("Poisson Mesh")
    pcd = o3d.io.read_point_cloud(str(pointcloud_ply))
    if pcd.is_empty():
        raise RuntimeError("Loaded point cloud is empty")

    _estimate_normals(pcd)

    # Poisson reconstruction
    mesh, densities = _poisson(pcd, depth=depth)

    # Density-based crop
    dens_kept = _crop_by_density(mesh, densities, q=crop_low_q)

    # Optional sky cleanup
    if cloud_cleanup:
        if isinstance(cloud_cleanup, dict):
            height_q = float(cloud_cleanup.get("height_q", 0.90))
            low_density_q = float(cloud_cleanup.get("low_density_q", 0.30))
        else:
            height_q, low_density_q = 0.90, 0.30
        _cloud_cleanup(mesh, np.asarray(pcd.points), dens_kept, height_q, low_density_q)

    # Smoothing
    mesh = _simple_smooth(mesh, iters=int(smooth_iters))
    if adaptive:
        mesh = _adaptive_smooth(mesh, dens_kept, int(adapt_iters), float(adapt_low_q), float(adapt_high_q), float(adapt_gamma))

    # Final cleanup & save
    _final_cleanup(mesh)
    return _save_mesh(mesh, out_mesh_ply)


# =========================
# Top-level pipeline
# =========================
def pipeline() -> int:
    """
    Entire interactive flow (print-only I/O):
      1) Pick export folder
      2) Pick .ply
      3) Pick preset
      4) Build Poisson mesh to ``mesh.ply``
    """
    try:
        export_dir = choose_export_folder()
        in_ply = choose_ply_file(export_dir)
        preset = choose_mesh_preset()

        out_mesh = export_dir / "mesh.ply"
        _ = build_poisson_mesh(
            in_ply, out_mesh,
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
        print(f"Input:       {in_ply.name}")
        print(f"Preset:      {next(k for k,v in MESH_PRESETS.items() if v is preset)}")
        print(f"Output:      {out_mesh.name}")
        print("Done ✅")
        return 0
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
