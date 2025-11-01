from __future__ import annotations

import sys
import subprocess

from src.interfaces.ns_dataset import run_dataset
from src.interfaces.ns_train import run_train
from src.interfaces.ns_export import run_export


def pipeline() -> int:
    """
    Full pipeline:
    1. Build dataset from chosen video
    2. Train nerf model on that dataset
    3. Export pointcloud from the trained model run
    """
    try:
        # Step 1: dataset
        dataset_info = run_dataset()

        # Step 2: training
        train_info = run_train(dataset_info)

        # Step 3: export (non-interactive, export same experiment we just trained)
        run_export(experiment_name=train_info["experiment_name"])

        print("\n=== PIPELINE COMPLETE ✅ ===")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Subprocess failed with exit {e.returncode}\n{e}\n")
        return e.returncode
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


def main() -> None:
    code = pipeline()
    sys.exit(code)


if __name__ == "__main__":
    main()
