from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, Any

from src.config.base_config import (
    EXPERIMENTS_DIR,
)
from src.config.config_nerfstudio import (
    MAX_ITERS,
)
from src.utils.ns_utils import ensure_exists, run
# train_nerf_model logic is basically build_ns_train_cmd+run in your file. :contentReference[oaicite:10]{index=10}


def _parse_ver_suffix(name: str) -> tuple[str, int | None]:
    """
    "Barn_ver3" -> ("Barn", 3)
    "Barn"      -> ("Barn", None)
    same as ns_pipeline.py logic. :contentReference[oaicite:11]{index=11}
    """
    m = re.match(r"^(?P<base>.+?)_ver(?P<num>\d+)$", name)
    if not m:
        return name, None
    return m.group("base"), int(m.group("num"))


def _make_unique_experiment_name(base_name: str, experiments_dir: Path) -> str:
    """
    Same logic taken from ns_pipeline.py: create <base> or <base>_ver2 ... :contentReference[oaicite:12]{index=12}
    """
    ensure_exists(experiments_dir, "dir")

    existing = [
        p.name for p in experiments_dir.iterdir()
        if p.is_dir() and (
                p.name == base_name or p.name.startswith(base_name + "_ver")
        )
    ]

    if base_name not in existing:
        return base_name

    max_ver = 0
    for name in existing:
        b, ver = _parse_ver_suffix(name)
        if b == base_name and ver is not None and ver > max_ver:
            max_ver = ver

    return f"{base_name}_ver{max_ver + 1}"


def _build_ns_train_cmd(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> str:
    """
    Mirrors build_ns_train_cmd from your ns_train.py. :contentReference[oaicite:13]{index=13}
    """
    parts = [
        "ns-train nerfacto",
        f'--data "{str(dataset_dir)}"',
        f'--output-dir "{str(experiments_dir)}"',
        f'--experiment-name "{exp_name}"',
        f"--max-num-iterations {max_iters}",
        "--train-camera-optimizer True",
        "--vis viewer",
        "--viewer.websocket-port 7007",
        "--viewer.quit-on-train-completion True",
        "--pipeline.datamanager.train-num-rays-per-batch 8192",
    ]

    return " ".join(parts)


def _run_ns_train(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> None:
    """
    Actually runs ns-train nerfacto. :contentReference[oaicite:14]{index=14}
    """
    ensure_exists(experiments_dir, "dir")
    cmd = _build_ns_train_cmd(dataset_dir, experiments_dir, exp_name, max_iters)
    print("🌐 Viewer will be at: http://127.0.0.1:7007")
    run(cmd)


def run_train(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public pipeline API:
    - dataset_info is what run_dataset() returned
    - generate experiment name
    - train nerf
    - return info about experiment (for export)

    Returns:
      {
        "experiment_name": str,
        "experiment_dir": Path,
    }
    """
    dataset_dir: Path = dataset_info["dataset_dir"]
    video_stem: str = dataset_info["video_stem"]

    experiment_name = _make_unique_experiment_name(video_stem, EXPERIMENTS_DIR)

    _run_ns_train(
        dataset_dir=dataset_dir,
        experiments_dir=EXPERIMENTS_DIR,
        exp_name=experiment_name,
        max_iters=MAX_ITERS,
    )

    print("\n[TRAIN STEP DONE ✅]")
    print(f"• Experiment name: {experiment_name}")
    print(f"• Experiments dir: {EXPERIMENTS_DIR}")

    return {
        "experiment_name": experiment_name,
        "experiment_dir": EXPERIMENTS_DIR / experiment_name,
    }


def main() -> None:
    """
    Standalone mode:
    This version of main is INTERACTIVE-LESS unless you wrap it yourself.
    Why? Training alone doesn't know which dataset to use unless we tell it.
    For standalone usage you'll usually just call ns_pipeline, or
    import run_train() in your own script and give it dataset_info.

    We still provide main() so the file is runnable, but we explain the usage.
    """
    print(
        "ns_train.py standalone usage:\n"
        "  This module expects dataset_info from ns_dataset.run_dataset().\n"
        "  Example (Python REPL):\n"
        "    from src.interfaces import ns_dataset, ns_train\n"
        "    d = ns_dataset.run_dataset()\n"
        "    ns_train.run_train(d)\n"
    )
    # We exit 0 so running `python ns_train.py` won't crash.
    sys.exit(0)


if __name__ == "__main__":
    main()
