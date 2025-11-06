# src/core/render.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from src.config.base_config import EXPERIMENTS_DIR, EXPORTS_DIR, ensure_core_dirs
from src.utils.common import header

ensure_core_dirs()


# ======================================================================
# Discovery utilities
# ======================================================================

def _has_nerfacto_run(exp_dir: Path) -> bool:
    """
    True if the experiment contains at least one nerfacto run:
      <exp_dir>/nerfacto/<timestamp>/config.yml
    """
    nerfacto_dir = exp_dir / "nerfacto"
    if not nerfacto_dir.is_dir():
        return False
    for p in nerfacto_dir.iterdir():
        if p.is_dir() and (p / "config.yml").is_file():
            return True
    return False


def _discover_experiments() -> Dict[str, Path]:
    """
    Return {name -> path} for experiments that have at least one nerfacto run.
    Sorted by name for stable menus.
    """
    if not EXPERIMENTS_DIR.exists():
        return {}
    mapping: Dict[str, Path] = {}
    for child in EXPERIMENTS_DIR.iterdir():
        if child.is_dir() and _has_nerfacto_run(child):
            mapping[child.name] = child
    return dict(sorted(mapping.items(), key=lambda kv: kv[0].lower()))


def _list_runs(exp_dir: Path) -> List[Path]:
    """
    All valid runs under nerfacto/, newest first.
    """
    runs: List[Path] = []
    nerfacto_dir = exp_dir / "nerfacto"
    if nerfacto_dir.is_dir():
        for p in nerfacto_dir.iterdir():
            if p.is_dir() and (p / "config.yml").is_file():
                runs.append(p)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _latest_run(exp_dir: Path) -> Path:
    runs = _list_runs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"No nerfacto runs found in: {exp_dir / 'nerfacto'}")
    return runs[0]


# ======================================================================
# Simple interactive menus (like other pipelines)
# ======================================================================

def _ask_choice(prompt: str, options: List[str]) -> int:
    """
    Return 0-based index from a numeric, exact-name, or unique-prefix choice.
    """
    while True:
        sel = input(prompt).strip()
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(options):
                return i - 1
        low = sel.lower()
        exact = [i for i, name in enumerate(options) if name.lower() == low]
        if len(exact) == 1:
            return exact[0]
        pref = [i for i, name in enumerate(options) if name.lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]
        print("Invalid choice. Try again.")


def _choose_experiment_interactive() -> Tuple[str, Path]:
    exps = _discover_experiments()
    if not exps:
        raise FileNotFoundError(f"No experiments with nerfacto runs under: {EXPERIMENTS_DIR}")

    names = list(exps.keys())
    print("\nAvailable experiments:\n")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {name}")

    idx = _ask_choice(f"\nChoose experiment [1-{len(names)} or name]: ", names)
    return names[idx], exps[names[idx]]


def _choose_run_interactive(exp_dir: Path) -> Path:
    runs = _list_runs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"No nerfacto runs in: {exp_dir / 'nerfacto'}")

    if len(runs) == 1:
        print(f"\nFound single run: {runs[0].name} (auto-selected)")
        return runs[0]

    print("\nAvailable runs (newest first):\n")
    for i, r in enumerate(runs, 1):
        print(f"  {i}) {r.name}")

    idx = _ask_choice(f"\nChoose run [1-{len(runs)} or name]: ", [r.name for r in runs])
    return runs[idx]


# ======================================================================
# ns-render runner (correct subcommand ordering)
# ======================================================================

def _trajectory_to_subcmd(trajectory: str, camera_path: Path | None) -> str:
    """
    Map our 'trajectory' flag to ns-render subcommand.
    - 'orbit' is an alias we map to 'spiral'
    - if a camera_path is provided, we must use 'camera-path'
    """
    if camera_path is not None or trajectory == "camera_path":
        return "camera-path"
    if trajectory in ("orbit", "spiral"):
        return "spiral"
    if trajectory == "interpolate":
        return "interpolate"
    # sensible default
    return "spiral"


def run_ns_render(
    config_yml: Path,
    output_path: Path,
    *,
    trajectory: str = "orbit",           # orbit -> spiral
    fmt: str = "video",                  # {"video","images"}
    seconds: int = 10,
    fps: int | None = None,
    extra_outputs: str | None = None,    # e.g. "rgb,depth,normals"
    camera_path: Path | None = None,     # viewer-exported camera path JSON
) -> None:
    """
    Build and execute a valid ns-render command. The subcommand must precede options.

    NOTE: width/height flags are intentionally omitted because your ns-render
    build does not accept them for spiral/interpolate. If you need resolution
    control, export a camera-path JSON from the viewer which encodes intrinsics.
    """
    subcmd = _trajectory_to_subcmd(trajectory, camera_path)

    cmd: List[str] = [
        "ns-render",
        subcmd,
        "--load-config", str(config_yml),
        "--output-format", fmt,
        "--output-path", str(output_path),
    ]

    # Animated subcommands accept seconds/fps
    if subcmd in ("spiral", "interpolate"):
        cmd += ["--seconds", str(seconds)]
        if fps is not None:
            cmd += ["--fps", str(fps)]

    if extra_outputs:
        cmd += ["--rendered-output-names", extra_outputs]

    if subcmd == "camera-path":
        if camera_path is None:
            raise ValueError("camera-path subcommand requires a --camera-path JSON file.")
        cmd += ["--camera-path-filename", str(camera_path)]

    header("Run ns-render")
    print(" ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


# ======================================================================
# Pipeline
# ======================================================================

def pipeline() -> int:
    """
    Flow:
      1) Choose experiment (interactive or --exp/--exp auto)
      2) Choose run (interactive or --run/--run latest)
      3) Find config.yml and render using a valid subcommand
    """
    try:
        parser = argparse.ArgumentParser(description="Render novel views from a trained Nerfstudio model.")
        parser.add_argument("--exp", type=str, default=None,
                            help="Experiment name or 'auto' for newest; if omitted, shows a menu.")
        parser.add_argument("--run", type=str, default="latest",
                            help="Timestamped run folder under nerfacto (default: latest).")
        parser.add_argument("--trajectory", type=str, default="orbit",
                            choices=["orbit", "spiral", "interpolate", "camera_path"],
                            help="Camera path: 'orbit' maps to 'spiral' subcommand.")
        parser.add_argument("--camera-path", type=str, default=None,
                            help="JSON exported from Nerfstudio viewer (implies camera-path subcommand).")
        parser.add_argument("--output-format", type=str, default="video",
                            choices=["video", "images"], help="Save video or frames.")
        parser.add_argument("--seconds", type=int, default=10)
        parser.add_argument("--fps", type=int, default=None,
                            help="Override FPS for animated renders (spiral/interpolate).")
        parser.add_argument("--extra", type=str, default=None,
                            help="Comma list of extra outputs, e.g., 'rgb,depth,normals'.")

        args = parser.parse_args()
        header("Render Nerfstudio Experiment")

        # 1) experiment
        if args.exp is None:
            exp_name, exp_dir = _choose_experiment_interactive()
        elif args.exp.lower() == "auto":
            exps = _discover_experiments()
            if not exps:
                raise FileNotFoundError(f"No experiments under: {EXPERIMENTS_DIR}")
            # newest by mtime is better for 'auto'
            newest = max(exps.values(), key=lambda p: p.stat().st_mtime)
            exp_name, exp_dir = newest.name, newest
            print(f"Auto-selected experiment: {exp_name}")
        else:
            exp_name = args.exp
            exp_dir = EXPERIMENTS_DIR / exp_name
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment not found: {exp_dir}")
            if not _has_nerfacto_run(exp_dir):
                raise FileNotFoundError(f"No nerfacto runs found under: {exp_dir / 'nerfacto'}")

        # 2) run
        if args.run and args.run.lower() != "latest":
            run_dir = exp_dir / "nerfacto" / args.run
            if not (run_dir / "config.yml").is_file():
                raise FileNotFoundError(f"Run not found or missing config.yml: {run_dir}")
        else:
            run_dir = _latest_run(exp_dir)

        print(f"Using run: {run_dir.name}")
        config_yml = run_dir / "config.yml"

        # 3) output path
        out_base = EXPORTS_DIR / f"{exp_name}_renders"
        out_base.mkdir(parents=True, exist_ok=True)
        subname = "camera_path" if (args.camera_path or args.trajectory == "camera_path") else args.trajectory
        output_path = (out_base / f"{subname}.mp4") if args.output_format == "video" else (out_base / subname)

        # 4) render
        camera_path = Path(args.camera_path) if args.camera_path else None
        run_ns_render(
            config_yml=config_yml,
            output_path=output_path,
            trajectory=args.trajectory,
            fmt=args.output_format,
            seconds=args.seconds,
            fps=args.fps,
            extra_outputs=args.extra,
            camera_path=camera_path,
        )

        print("\n✅ Render complete")
        print(f"• Experiment : {exp_name}")
        print(f"• Run        : {run_dir.name}")
        print(f"• Output     : {output_path}\n")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ns-render failed with exit code {e.returncode}\n")
        return e.returncode
    except Exception as e:
        print(f"\n❌ {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(pipeline())
