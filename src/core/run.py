from __future__ import annotations

import argparse
import sys
from importlib import import_module
from types import ModuleType
from typing import Callable, Optional, Iterable

# ---- Pipelines ----
GEOMETRY_PIPELINE: list[str] = [
    "points_filtering",
    "clouds_filtering",
    "mesh",
]

NERF_PIPELINE: list[str] = [
    "ns_dataset",
    "ns_train",
    "ns_export",
    "ns_render",
]

# Default when user asks for "all"
DEFAULT_ORDER: list[str] = GEOMETRY_PIPELINE + NERF_PIPELINE

# Candidate entrypoints to call inside each module
ENTRY_FUNCS: tuple[str, ...] = ("main", "pipeline", "cli", "run")


def _find_entry(mod: ModuleType) -> Optional[Callable[[], int]]:
    for name in ENTRY_FUNCS:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn  # type: ignore[return-value]
    return None


def _run_module(name: str, strict: bool) -> int:
    try:
        mod = import_module(name)
    except Exception as e:
        msg = f"… skipping '{name}' (import failed: {e.__class__.__name__})"
        if strict:
            print(msg)
            return 1
        print(msg)
        return 0

    fn = _find_entry(mod)
    if fn is None:
        msg = f"… skipping '{name}' (no entry: {', '.join(ENTRY_FUNCS)})"
        if strict:
            print(msg)
            return 1
        print(msg)
        return 0

    print(f"\n=== RUN → {name}.{fn.__name__}() ===")
    try:
        rc = int(fn())
    except SystemExit as se:
        rc = int(getattr(se, "code", 1) or 0)
    except Exception as e:
        print(f"✖ Exception in {name}: {e}")
        return 1
    if rc != 0:
        print(f"✖ Step '{name}' exited with code {rc}")
    return rc


def _expand_steps(steps_arg: str) -> list[str]:
    s = steps_arg.strip().lower()
    if s in ("all", "both"):
        return DEFAULT_ORDER
    if s in ("geom", "geometry"):
        return GEOMETRY_PIPELINE
    if s in ("nerf", "ns", "nerfstudio"):
        return NERF_PIPELINE
    # Otherwise, treat as comma-separated module names
    return [part.strip() for part in steps_arg.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project runner with separate geometry and NeRF pipelines."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help=(
            "What to run: 'geometry' | 'nerf' | 'all' | comma-separated modules. "
            "geometry = points_filtering,clouds_filtering,mesh ; "
            "nerf = ns_dataset,ns_train,ns_export,ns_render ; "
            "all = both pipelines in sequence."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a step is missing or has no entry function (otherwise skipped).",
    )
    args = parser.parse_args()

    order = _expand_steps(args.steps)
    if not order:
        print("✖ No steps resolved from --steps argument.")
        return 1

    ran_any = False
    for name in order:
        rc = _run_module(name, strict=args.strict)
        if rc != 0:
            return rc
        ran_any = True

    if not ran_any and args.strict:
        print("✖ No steps executed and --strict specified.")
        return 1

    print("\n✔ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
