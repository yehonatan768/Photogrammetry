r"""
export.py — Export dense point cloud (menu), aligned with new experiment layout
===============================================================================

What this does (same purpose, cleaner structure)
------------------------------------------------
1) Lists **experiments** under ``<PROJECT_DIR>/outputs/experiments`` (nerfacto-based).
2) Lets you choose an experiment; **auto-picks the newest run** (``nerfacto/<RUN>/config.yml``).
3) Calls ``ns-export pointcloud`` and writes a **versioned folder** under
   ``<PROJECT_DIR>/outputs/exports/<PrettyName>_verK-DD-MM-YYYY/point_cloud.ply``.

Notes
-----
- Keeps your original behavior and defaults. Only readability & structure improved.
- Still uses: ``NS_EXPORT``, ``NORMAL_METHOD``, ``NUM_POINTS_DEFAULT``, ``REMOVE_OUTLIERS_DEFAULT``.
- Suppresses noisy child warnings when streaming output (configurable).

Dependencies
------------
- Python 3.9+
- Nerfstudio CLI: ``ns-export`` available on PATH (or set ``NS_EXPORT`` below).

Tested on Windows + PowerShell with Nerfstudio 1.x.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# =========================
# Configuration (edit here)
# =========================
PROJECT_DIR = Path(r"D:\Projects\NerfStudio")

# New roots (match nerfstudio.py changes):
EXPERIMENTS_ROOT = PROJECT_DIR / "outputs/experiments"  # where ns-train writes
EXPORTS_DIR = PROJECT_DIR / "outputs/exports"           # where this script writes

NS_EXPORT = shutil.which("ns-export") or "ns-export"
NORMAL_METHOD = "open3d"
NUM_POINTS_DEFAULT = 6_000_000
REMOVE_OUTLIERS_DEFAULT = False

# Stream output settings
SUPPRESS_CHILD_WARNINGS = True
CHILD_WARNING_FILTER = "ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning"
NOISY_PATTERNS = [
    r"^WARNING: Using a slow implementation",
    r"FutureWarning:",
    r"RuntimeWarning:",
]


# =========================
# Console helpers
# =========================
def header(title: str) -> None:
    """Pretty section header."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def _now_str_for_folder() -> str:
    """Return DD-MM-YYYY for folder naming (keeps your original pattern)."""
    dt = datetime.now()
    return f"{dt.day:02d}-{dt.month:02d}-{dt.year:04d}"


def _nice_experiment_name(exp: str) -> str:
    """
    Convert an experiment folder to a 'pretty' prefix for export folder names.
    Example: 'barn_high_quality' -> 'Barn'.
    """
    base = exp.split("_", 1)[0]
    return base.capitalize()


# =========================
# Child process execution
# =========================
def run_streamed(cmd: List[str], filter_patterns: List[str] | None = None) -> int:
    """
    Run a child process and stream its stdout in real time.
    Optionally filter out noisy lines that clutter the console.
    Returns the process return code.
    """
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
# Experiments & runs
# =========================
def find_experiments(experiments_root: Path) -> Dict[str, Path]:
    """
    Discover experiments containing a 'nerfacto' subfolder with at least one run
    (i.e., a 'config.yml' under 'nerfacto/<RUN>/config.yml').
    Returns a dict: experiment_name -> absolute path.
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
        # any config.yml under nerfacto/*/
        has_run = any(p.name == "config.yml" for p in nerfacto_dir.rglob("config.yml"))
        if has_run:
            exps[child.name] = child

    # Sort by name for stable menu order
    return dict(sorted(exps.items(), key=lambda kv: kv[0].lower()))


def list_runs_for_experiment(exp_dir: Path) -> List[Path]:
    """
    Return run directories under '<exp>/nerfacto/<RUN>' that contain a config.yml.
    Sorted by mtime descending (newest first).
    """
    runs: List[Path] = []
    nerfacto_dir = exp_dir / "nerfacto"
    if nerfacto_dir.is_dir():
        for cfg in nerfacto_dir.rglob("config.yml"):
            rd = cfg.parent
            if rd.parent.name.lower() == "nerfacto":
                runs.append(rd)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def choose_experiment_and_run() -> Tuple[str, Path]:
    """
    Interactive menu to choose an experiment, then auto-pick the newest run.
    Returns (experiment_name, run_dir).
    """
    exps = find_experiments(EXPERIMENTS_ROOT)
    if not exps:
        raise FileNotFoundError(f"No experiments found under: {EXPERIMENTS_ROOT}")

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

    runs = list_runs_for_experiment(exps[exp_name])
    if not runs:
        raise FileNotFoundError(f"No nerfacto runs found in experiment '{exp_name}'.")
    chosen_run = runs[0]
    print(f"• Using run: {chosen_run}")
    return exp_name, chosen_run


# =========================
# Export folder versioning
# =========================
def next_versioned_export_dir(exp_name: str) -> Path:
    """
    Create a unique, **versioned** export directory for the experiment.

    Strategy
    --------
    - Pretty name = ``_nice_experiment_name(exp_name)``.
    - Scan **all existing versions** for that pretty name across *any* date:
        Pretty_ver<k>-<DD-MM-YYYY>
    - Choose next k = max(existing) + 1.
    - Ensure uniqueness for today's date as well.

    Returns the newly created folder's Path.
    """
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
    # Ensure uniqueness in the unlikely event of collision for today's date
    while True:
        cand = EXPORTS_DIR / f"{pretty}_ver{k}-{date_str}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        k += 1


# =========================
# Nerfstudio export
# =========================
def export_pointcloud_to_temp(config_yml: Path, temp_dir: Path, *,
                              num_points: int,
                              remove_outliers: bool) -> Path:
    """
    Invoke ``ns-export pointcloud`` writing to a temporary folder, and return
    the **largest** .ply produced (your original policy).
    """
    header("Export Dense Point Cloud")
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        NS_EXPORT, "pointcloud",
        "--load-config", str(config_yml),
        "--output-dir", str(temp_dir),
        "--normal-method", NORMAL_METHOD,
        "--num-points", str(num_points),
        "--remove-outliers", str(bool(remove_outliers)),
    ]

    ret = run_streamed(cmd, filter_patterns=NOISY_PATTERNS)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)

    ply_files = sorted(temp_dir.rglob("*.ply"), key=lambda p: p.stat().st_size, reverse=True)
    if not ply_files:
        raise FileNotFoundError(f"No .ply produced in {temp_dir}")
    print("✅ Done export")
    return ply_files[0]


# =========================
# Top-level pipeline
# =========================
def pipeline() -> int:
    """
    Entire interactive flow:
      1) Choose experiment (menu)
      2) Auto-pick newest run (nerfacto/<RUN>)
      3) Create versioned export folder
      4) ``ns-export pointcloud`` → copy largest .ply as ``point_cloud.ply``
    """
    try:
        # 1) Choose experiment + newest run
        exp_name, run_dir = choose_experiment_and_run()
        cfg_path = run_dir / "config.yml"

        # 2) Create versioned export folder
        export_dir = next_versioned_export_dir(exp_name)
        print(f"• Export folder: {export_dir}")

        # 3) Export to temp, then save as point_cloud.ply
        temp_export = export_dir / "_tmp_export"
        largest_ply = export_pointcloud_to_temp(
            cfg_path, temp_export,
            num_points=NUM_POINTS_DEFAULT,
            remove_outliers=REMOVE_OUTLIERS_DEFAULT,
        )
        final_point = export_dir / "point_cloud.ply"
        shutil.copy2(largest_ply, final_point)
        shutil.rmtree(temp_export, ignore_errors=True)
        print(f"Saved: {final_point}")

        print("\n=== SUMMARY ===")
        print(f"Experiment:  {exp_name}")
        print(f"Run:         {run_dir}")
        print(f"Output dir:  {export_dir}")
        print(f"Files:       {final_point.name}")
        print("Done ✅")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess failed with exit {e.returncode}\n{e}\n")
        return e.returncode
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
