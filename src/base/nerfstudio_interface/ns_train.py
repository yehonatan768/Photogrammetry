from __future__ import annotations

from pathlib import Path
from src.utils.ns_utils import run, ensure_exists


def build_ns_train_cmd(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> str:
    """
    Build the 'ns-train nerfacto' CLI command to train NeRF.
    """
    parts = [
        "ns-train nerfacto",
        f'--data "{str(dataset_dir)}"',
        f'--output-dir "{str(experiments_dir)}"',
        f'--experiment-name "{exp_name}"',
        f"--max-num-iterations {max_iters}",
        "--vis viewer",
        "--viewer.websocket-port 7007",
        "--viewer.quit-on-train-completion True",
    ]
    return " ".join(parts)


def train_nerf_model(
    dataset_dir: Path,
    experiments_dir: Path,
    exp_name: str,
    max_iters: int,
) -> None:
    """
    Run ns-train nerfacto and start training.
    """
    ensure_exists(experiments_dir, "dir")

    cmd = build_ns_train_cmd(dataset_dir, experiments_dir, exp_name, max_iters)

    print("🌐 Viewer will be at: http://127.0.0.1:7007")
    run(cmd)
