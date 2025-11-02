from __future__ import annotations

import sys
import subprocess
import argparse
from pathlib import Path

from src.interfaces.ns_dataset import run_dataset
from src.interfaces.ns_train import run_train
from src.interfaces.ns_export import run_export


def pipeline(
    dataset_path: Path | None = None,
    experiment_name: str | None = None,
) -> int:
    """
    Full pipeline:
    Mode A (default): build dataset (COLMAP etc.) -> train -> export
    Mode B (direct):  use existing dataset (--dataset-path + --experiment-name) -> train -> export

    Args:
        dataset_path (Path | None): If provided together with experiment_name,
            we SKIP run_dataset() and go straight to training using that dataset.
        experiment_name (str | None): Name to use for training / experiment folder.
    """
    try:
        # -----------------
        # Step 1: dataset
        # -----------------
        if dataset_path is None or experiment_name is None:
            # Mode A: normal flow, create dataset now (runs COLMAP etc.)
            # run_dataset() is expected to return a dict with keys like:
            # {
            #   "dataset_dir": Path(...),
            #   "experiment_name_suggestion": "DJI0004",
            #   "video_stem": "DJI0004",
            #   ...
            # }
            dataset_info = run_dataset()
        else:
            # Mode B: user gave us dataset + experiment explicitly
            ds_path = Path(dataset_path)
            if not ds_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {ds_path}")

            dataset_info = {
                "dataset_dir": ds_path,
                "experiment_name_suggestion": experiment_name,
                "video_stem": experiment_name,
            }

        # -----------------
        # Step 2: training
        # -----------------
        train_info = run_train(dataset_info)
        # train_info is expected to include:
        # {
        #   "experiment_name": "<final_experiment_name_we_trained>",
        #   ...
        # }

        # -----------------
        # Step 3: export
        # -----------------
        run_export(experiment_name=train_info["experiment_name"])

        print("\n=== PIPELINE COMPLETE ✅ ===")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess failed with exit {e.returncode}\n{e}\n")
        return e.returncode
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse CLI args for this pipeline script.

    Usage patterns:

    1) Full pipeline (build dataset -> train -> export):
        python pipeline.py

    2) Skip dataset step (use existing prepared dataset directly):
        python pipeline.py --dataset-path "D:\\...\\DJI0004_ver2" --experiment-name "DJI0004"

    Notes:
    - If BOTH --dataset-path and --experiment-name are provided, we skip run_dataset().
    - If either is missing, we run full pipeline.
    """
    parser = argparse.ArgumentParser(
        description="End-to-end NeRF pipeline (dataset -> train -> export)"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to an existing ns-process-data dataset folder "
             "(the one you'd pass to ns-train --data ...). "
             "If provided together with --experiment-name, we skip run_dataset().",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name to train/export under (e.g. DJI0004). "
             "Used together with --dataset-path to skip run_dataset().",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    # Decide mode based on provided flags:
    # - If both dataset-path and experiment-name are given -> direct train mode.
    # - Else -> full pipeline mode.
    if args.dataset_path and args.experiment_name:
        code = pipeline(
            dataset_path=Path(args.dataset_path),
            experiment_name=args.experiment_name,
        )
    else:
        code = pipeline(
            dataset_path=None,
            experiment_name=None,
        )

    sys.exit(code)


if __name__ == "__main__":
    main()
