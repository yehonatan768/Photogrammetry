#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Export a textured mesh or point cloud from a Nerfstudio run (project-integrated),
# calling the Nerfstudio exporter **module** with the correct **subcommands**:
#   python -m nerfstudio.scripts.exporter <subcommand> --load-config ... [options]
#
# Mesh subcommands we support: poisson (default), marching-cubes, tsdf
# Point cloud: pointcloud
#
# Examples:
#   python export_mesh.py --experiment DJI0004_ver1 --textured --texture-size 4096
#   python export_mesh.py --config <.../config.yml> --exporter pointcloud
#   python export_mesh.py --exporter mesh --mesh-algo marching-cubes --extra "--decimate-target 0.5"
#
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _fallback_header(title: str) -> None:
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


# ---------------- Project imports (safe) ----------------
try:
    from src.config.base_config import ensure_core_dirs as _ensure_core_dirs  # type: ignore

    try:
        from src.config.base_config import EXPERIMENTS_DIR as _EXPERIMENTS_DIR  # type: ignore
    except Exception:
        _EXPERIMENTS_DIR = None
    try:
        from src.config.base_config import EXPORTS_DIR as _EXPORTS_DIR  # type: ignore
    except Exception:
        _EXPORTS_DIR = None
except Exception:
    def _ensure_core_dirs() -> None:
        pass


    _EXPERIMENTS_DIR = None
    _EXPORTS_DIR = None

try:
    from src.utils.common import header as _header  # type: ignore
except Exception:
    _header = _fallback_header

_choose_experiment = None
_choose_run_for_experiment = None
try:
    from src.utils.menu import choose_experiment as _choose_experiment  # type: ignore
except Exception:
    pass
try:
    from src.utils.menu import choose_run_for_experiment as _choose_run_for_experiment  # type: ignore
except Exception:
    pass


# ---------------- Project root & venv python ----------------
def _is_project_root(p: Path) -> bool:
    return (p / ".venv").exists() and ((p / "outputs").exists() or (p / "src").exists())


def _find_project_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if _is_project_root(p):
            return p
    env_root = os.getenv("NS_PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    return start


def _venv_python(project_root: Path) -> Path:
    win = project_root / ".venv" / "Scripts" / "python.exe"
    if win.exists():
        return win
    posix = project_root / ".venv" / "bin" / "python"
    if posix.exists():
        return posix
    env_py = os.getenv("NS_VENV_PYTHON")
    if env_py and Path(env_py).exists():
        return Path(env_py)
    raise SystemExit(
        f"Could not locate venv python. Expected at:\n  {win}\n  {posix}\n"
        "Create/activate the project's .venv or set NS_VENV_PYTHON to your venv python."
    )


# ---------------- Discover runs (fallback if no menu helpers) ----------------
@dataclass
class RunInfo:
    experiment: str
    trainer: str
    timestamp: str
    config: Path


def _discover_runs(experiments_dir: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    if not experiments_dir.exists():
        return runs
    for exp_dir in sorted((p for p in experiments_dir.iterdir() if p.is_dir())):
        for trainer_dir in sorted((p for p in exp_dir.iterdir() if p.is_dir())):
            for ts_dir in sorted((p for p in trainer_dir.iterdir() if p.is_dir())):
                cfg = ts_dir / "config.yml"
                if cfg.exists():
                    runs.append(RunInfo(exp_dir.name, trainer_dir.name, ts_dir.name, cfg))
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


def _simple_choose(items: List[str], prompt: str) -> int:
    print("\n" + prompt)
    for i, n in enumerate(items, 1):
        print(f"  {i}) {n}")
    while True:
        val = input(f"Choose [1-{len(items)} or name]: ").strip()
        if val.isdigit():
            i = int(val)
            if 1 <= i <= len(items):
                return i - 1
        low = val.lower()
        exact = [i for i, n in enumerate(items) if n.lower() == low]
        if len(exact) == 1:
            return exact[0]
        pref = [i for i, n in enumerate(items) if n.lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]
        print("Invalid choice. Try again.")


# ---------------- Build command with proper subcommand ----------------
def _mesh_subcommand(algo: str) -> str:
    # Map friendly algo names to exporter subcommands
    m = {
        "poisson": "poisson",
        "marching-cubes": "marching-cubes",
        "tsdf": "tsdf",
    }
    if algo not in m:
        raise SystemExit(f"Unsupported mesh algorithm '{algo}'. Choose from: {', '.join(m)}")
    return m[algo]


def _build_ns_export_cmd(
        venv_python: Path,
        *,
        config_path: Path,
        output_dir: Path,
        exporter: str = "mesh",
        mesh_algo: str = "poisson",
        textured: bool = False,
        texture_size: int = 4096,
        extra: str = "",
) -> List[str]:
    # Subcommand must immediately follow the module
    base = [str(venv_python), "-m", "nerfstudio.scripts.exporter"]
    if exporter == "pointcloud":
        sub = ["pointcloud"]
    else:
        sub = [_mesh_subcommand(mesh_algo)]  # e.g., "poisson"
    cmd = base + sub + [
        "--load-config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    if exporter != "pointcloud" and textured:
        cmd += ["--use-texture", "--texture-size", str(texture_size)]
    if extra:
        cmd += shlex.split(extra)
    return cmd


# ---------------- Main ----------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a mesh or point cloud from a Nerfstudio run using your project's venv and utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Project root (auto-detected if not supplied).")
    parser.add_argument("--experiments-dir", type=Path, default=None,
                        help="Override experiments dir (defaults to base_config.EXPERIMENTS_DIR or <project>/outputs/experiments).")
    parser.add_argument("--exports-dir", type=Path, default=None,
                        help="Override exports dir (defaults to base_config.EXPORTS_DIR or <project>/outputs/exports).")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, help="Path to a Nerfstudio config.yml (non-interactive).")
    group.add_argument("--experiment", type=str, help="Experiment name (latest run is auto-selected).")

    parser.add_argument("--exporter", choices=["mesh", "pointcloud"], default="mesh",
                        help="Export a watertight mesh (poisson/marching-cubes/tsdf) or a colored point cloud.")
    parser.add_argument("--mesh-algo", choices=["poisson", "marching-cubes", "tsdf"], default="poisson",
                        help="Mesh extraction algorithm (only used if --exporter mesh).")
    parser.add_argument("--textured", action="store_true",
                        help="For mesh: bake UV texture (OBJ/GLB). Ignored for pointcloud.")
    parser.add_argument("--texture-size", type=int, default=4096,
                        help="Texture resolution for baked UVs.")
    parser.add_argument("--extra", type=str, default="",
                        help="Passthrough flags for Nerfstudio exporter (e.g., '--decimate-target 0.5 --mesh-format ply').")

    args = parser.parse_args()

    project_root = args.project_root or _find_project_root(Path(__file__).resolve().parent)
    vpy = _venv_python(project_root)
    _ensure_core_dirs()

    experiments_dir = args.experiments_dir or (
        _EXPERIMENTS_DIR if _EXPERIMENTS_DIR else project_root / "outputs" / "experiments")
    exports_dir = args.exports_dir or (_EXPORTS_DIR if _EXPORTS_DIR else project_root / "outputs" / "exports")
    exports_dir.mkdir(parents=True, exist_ok=True)

    # Resolve config path
    if args.config:
        cfg = args.config
        if not cfg.exists():
            raise SystemExit(f"Config not found: {cfg}")
        try:
            exp_name = cfg.parents[3].name
            run_stamp = cfg.parent.name
        except Exception:
            exp_name = cfg.parent.parent.name
            run_stamp = cfg.parent.name
    else:
        if _choose_experiment and _choose_run_for_experiment:
            exp_name = _choose_experiment(experiments_dir)
            run_stamp, cfg = _choose_run_for_experiment(experiments_dir, exp_name)
        else:
            runs = _discover_runs(experiments_dir)
            if not runs:
                raise SystemExit(f"No Nerfstudio runs found under: {experiments_dir}")
            exps = sorted({r.experiment for r in runs})
            idx = _simple_choose(exps, "Available experiments:")
            exp_name = exps[idx]
            latest = next(r for r in runs if r.experiment == exp_name)
            cfg = latest.config
            run_stamp = latest.timestamp

    suffix = ("textured" if (args.exporter == "mesh" and args.textured) else "raw")
    out_dir = exports_dir / f"{exp_name}_{args.exporter}_{suffix}_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _header("Export settings")
    print(f"Project:     {project_root}")
    print(f"Venv Python: {vpy}")
    print(f"Config:      {cfg}")
    print(f"Exporter:    {args.exporter}")
    if args.exporter == "mesh":
        print(f"Mesh algo:   {args.mesh_algo}")
        print(f"Textured:    {args.textured}")
        if args.textured:
            print(f"TextureSize: {args.texture_size}")
    print(f"Output Dir:  {out_dir}")
    if args.extra:
        print(f"Extra:       {args.extra}")

    cmd = _build_ns_export_cmd(
        venv_python=vpy,
        config_path=cfg,
        output_dir=out_dir,
        exporter=args.exporter,
        mesh_algo=args.mesh_algo,
        textured=args.textured,
        texture_size=args.texture_size,
        extra=args.extra,
    )

    _header("Nerfstudio exporter command")
    print(" ".join(shlex.quote(c) for c in cmd))

    _header("Running exporter")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Exporter failed with exit code {e.returncode}. See the above nerfstudio error message.")
    except FileNotFoundError:
        raise SystemExit(f"Could not execute: {cmd[0]}\nMake sure your venv exists and contains Python: {vpy}")

    _header("Done")
    print(f"Exported to: {out_dir}")
    if args.exporter == "mesh" and args.textured:
        print("Viewer tip: if your viewer ignores textures, export GLB or ensure OBJ+MTL+PNG are together.")


if __name__ == "__main__":
    main()
