from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.config.base_config import (
    EXPERIMENTS_DIR,
    EXPORTS_DIR,
    ensure_core_dirs,
)
from src.config.config_export import (
    NS_EXPORT,
    NORMAL_METHOD,
    NUM_POINTS_DEFAULT,
    REMOVE_OUTLIERS_DEFAULT,
    SUPPRESS_CHILD_WARNINGS,
    CHILD_WARNING_FILTER,
    NOISY_PATTERNS,
)


# =========================
# Pretty printing
# =========================

def _header(title: str) -> None:
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def _now_str_for_folder() -> str:
    dt = datetime.now()
    return f"{dt.day:02d}-{dt.month:02d}-{dt.year:04d}"


def _nice_experiment_name(exp: str) -> str:
    base = exp.split("_", 1)[0]
    return base.capitalize()


# =========================
# Process execution
# =========================

def _run_streamed(cmd: List[str], filter_patterns: List[str] | None = None) -> int:
    env = os.environ.copy()
    if SUPPRESS_CHILD_WARNINGS:
        env["PYTHONWARNINGS"] = CHILD_WARNING_FILTER
        env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if filter_patterns and any(re.search(p, line) for p in filter_patterns):
            continue
        print(line)
    return proc.wait()


# =========================
# Experiments / runs helpers
# =========================

def _find_experiments(experiments_root: Path) -> Dict[str, Path]:
    """
    Returns {experiment_name -> experiment_path} for valid nerfacto runs.
    """
    exps: Dict[str, Path] = {}
    if not experiments_root.exists():
        return exps

    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        nerfacto_dir = child / "nerfacto"
        if not nerfacto_dir.is_dir():
            continue
        has_run = any(p.name == "config.yml" for p in nerfacto_dir.rglob("config.yml"))
        if has_run:
            exps[child.name] = child

    return dict(sorted(exps.items(), key=lambda kv: kv[0].lower()))


def _list_runs_for_experiment(exp_dir: Path) -> List[Path]:
    """
    From an experiment dir, collect run subfolders that contain config.yml,
    newest first.
    """
    runs: List[Path] = []
    nerfacto_dir = exp_dir / "nerfacto"
    if nerfacto_dir.is_dir():
        for cfg in nerfacto_dir.rglob("config.yml"):
            rd = cfg.parent
            # we only accept dirs directly under nerfacto/<timestamp>/
            if rd.parent.name.lower() == "nerfacto":
                runs.append(rd)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _interactive_choose_experiment_and_run() -> Tuple[str, Path]:
    """
    Old behavior: ask user which experiment and pick newest run.
    """
    exps = _find_experiments(EXPERIMENTS_DIR)
    if not exps:
        raise FileNotFoundError(f"No experiments found under: {EXPERIMENTS_DIR}")

    print("\nAvailable experiments:\n")
    exp_names = list(exps.keys())
    for i, s in enumerate(exp_names, 1):
        print(f"  {i}) {s}")

    while True:
        sel = input(f"\nChoose an experiment [1-{len(exp_names)}] or type name: ").strip()
        if sel in exps:
            exp_name = sel
            break
        if sel.isdigit() and 1 <= int(sel) <= len(exp_names):
            exp_name = exp_names[int(sel) - 1]
            break
        print("Invalid choice. Try again.")

    runs = _list_runs_for_experiment(exps[exp_name])
    if not runs:
        raise FileNotFoundError(f"No nerfacto runs found in experiment '{exp_name}'.")
    chosen_run = runs[0]
    print(f"• Using run: {chosen_run}")
    return exp_name, chosen_run


def _latest_run_for_experiment(exp_name: str) -> Path:
    """
    Non-interactive: given an experiment name (e.g. from training step),
    pick its newest run folder.
    """
    exp_dir = EXPERIMENTS_DIR / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment '{exp_name}' not found at {exp_dir}")

    runs = _list_runs_for_experiment(exp_dir)
    if not runs:
        raise FileNotFoundError(f"No nerfacto runs found in experiment '{exp_name}'.")
    return runs[0]


# =========================
# Export folder versioning
# =========================

def _next_versioned_export_dir(exp_name: str) -> Path:
    """
    Creates exports/<Pretty>_verK-DD-MM-YYYY and returns that path.
    """
    ensure_core_dirs()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pretty = _nice_experiment_name(exp_name)
    date_str = _now_str_for_folder()

    pattern = re.compile(rf'^{re.escape(pretty)}_ver(\d+)-', re.IGNORECASE)
    max_ver = 0
    for child in EXPORTS_DIR.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            try:
                v = int(m.group(1))
                max_ver = max(max_ver, v)
            except ValueError:
                pass

    k = max_ver + 1
    while True:
        cand = EXPORTS_DIR / f"{pretty}_ver{k}-{date_str}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        k += 1


# =========================
# Nerfstudio export logic
# =========================

def _export_pointcloud_to_temp(config_yml: Path, temp_dir: Path, *,
                               num_points: int,
                               remove_outliers: bool) -> Path:
    _header("Export Dense Point Cloud")
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        NS_EXPORT, "pointcloud",
        "--load-config", str(config_yml),
        "--output-dir", str(temp_dir),
        "--normal-method", NORMAL_METHOD,
        "--num-points", str(num_points),
        "--remove-outliers", str(bool(remove_outliers)),
    ]

    ret = _run_streamed(cmd, filter_patterns=NOISY_PATTERNS)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

    ply_files = sorted(temp_dir.rglob("*.ply"), key=lambda p: p.stat().st_size, reverse=True)
    if not ply_files:
        raise FileNotFoundError(f"No .ply produced in {temp_dir}")
    print("✅ Done export")
    return ply_files[0]


# =========================
# public API for pipeline
# =========================

def run_export(experiment_name: Optional[str] = None) -> Path:
    """
    Pipeline API:
    - If experiment_name is given: use that experiment automatically.
    - If not: ask user which experiment to export.
    - Then export point cloud into a new versioned export dir.
    - Returns the final export directory.
    """
    if experiment_name is None:
        # interactive mode
        exp_name, run_dir = _interactive_choose_experiment_and_run()
    else:
        exp_name = experiment_name
        run_dir = _latest_run_for_experiment(exp_name)
        print(f"• Using experiment '{exp_name}'")
        print(f"• Latest run dir: {run_dir}")

    cfg_path = run_dir / "config.yml"

    export_dir = _next_versioned_export_dir(exp_name)
    print(f"• Export folder: {export_dir}")

    temp_export = export_dir / "_tmp_export"
    largest_ply = _export_pointcloud_to_temp(
        cfg_path, temp_export,
        num_points=NUM_POINTS_DEFAULT,
        remove_outliers=REMOVE_OUTLIERS_DEFAULT,
    )

    final_point = export_dir / "point_cloud.ply"
    shutil.copy2(largest_ply, final_point)
    shutil.rmtree(temp_export, ignore_errors=True)
    print(f"Saved: {final_point}")

    print("\n=== EXPORT SUMMARY ===")
    print(f"Experiment:  {exp_name}")
    print(f"Run:         {run_dir}")
    print(f"Output dir:  {export_dir}")
    print(f"Files:       {final_point.name}")
    print("Done ✅")

    return export_dir


def main() -> None:
    """
    Standalone mode:
    `python -m src.interfaces.ns_export`
    or `python ns_export.py`
    -> will ask you which experiment to export.
    """
    try:
        run_export(experiment_name=None)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess failed with exit {e.returncode}\n{e}\n")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
