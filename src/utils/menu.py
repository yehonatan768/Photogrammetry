from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from src.config.base_config import EXPORTS_DIR
from src.utils.preset_loader import list_presets, load_preset, PresetType


# -----------------------------
# Internal helpers
# -----------------------------

def _ask_choice(prompt: str, options: List[str]) -> int:
    """
    Show a prompt and return index (0-based) of the chosen option.
    Allows entering either the number or exact/prefix text.
    """
    if not options:
        raise ValueError("Options list is empty.")

    while True:
        sel = input(prompt).strip()
        if not sel:
            print("Please enter a number or a name/prefix.")
            continue

        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(options):
                return i - 1

        low = sel.lower()

        # exact match
        exact = [idx for idx, name in enumerate(options) if name.lower() == low]
        if len(exact) == 1:
            return exact[0]

        # unique prefix match
        pref = [idx for idx, name in enumerate(options) if name.lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]

        print("Invalid choice. Try again.")


def _normalize_kind(kind: str) -> PresetType:
    aliases = {
        "points": "points_filtering",
        "points_filtering": "points_filtering",
        "clouds": "clouds_filtering",
        "clouds_filtering": "clouds_filtering",
        "mesh": "mesh",
    }
    k = aliases.get(kind.lower())
    if k is None:
        raise ValueError(
            "Unknown preset kind: {kind} (expected: points, points_filtering, clouds, clouds_filtering, mesh)"
        )
    return k  # type: ignore[return-value]


# -----------------------------
# Public API
# -----------------------------

def choose_export_folder() -> Path:
    """
    Pick an export folder under EXPORTS_DIR (newest first).
    """
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory at: {EXPORTS_DIR}")

    candidates = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folders found under: {EXPORTS_DIR}")

    # newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    print("\nAvailable export folders:\n")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}) {p.name}")

    idx = _ask_choice(
        f"\nChoose export folder [1-{len(candidates)} or name]: ",
        [c.name for c in candidates],
    )
    return candidates[idx]


def choose_ply_file(export_dir: Path) -> Path:
    """
    Choose a .ply from export_dir.
    Priority: filtered.ply -> light_filtered.ply -> point_cloud.ply -> mesh.ply -> rest by mtime.
    """
    if not export_dir.exists():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")

    ply_files = [p for p in export_dir.glob("*.ply") if p.is_file()]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {export_dir}")

    # priority ranking (lower is better)
    priority = {
        "filtered.ply": 0,
        "light_filtered.ply": 1,
        "point_cloud.ply": 2,
        "mesh.ply": 99,
    }

    # Sort by (priority, -mtime) so higher mtime (newer) comes first within same priority
    ply_files.sort(key=lambda p: (priority.get(p.name, 10), -p.stat().st_mtime))

    if len(ply_files) == 1:
        print(f"\nFound single .ply: {ply_files[0].name} (auto-selected)")
        return ply_files[0]

    print("\nPLY files in folder:\n")
    for i, f in enumerate(ply_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}) {f.name}  ({size_mb:.1f} MB)")

    idx = _ask_choice(
        f"\nChoose input .ply [1-{len(ply_files)} or name]: ",
        [p.name for p in ply_files],
    )
    return ply_files[idx]


def choose_preset(
        kind: str,
        *,
        return_config: bool = False,
) -> Union[str, Tuple[str, dict]]:
    """
    Interactively choose a preset by kind, returning the name, or (name, config dict)
    if return_config=True.

    kind accepts {"points", "points_filtering", "clouds", "mesh"}.
    "points" is kept for backwards compatibility and maps to "points_filtering".
    """
    k: PresetType = _normalize_kind(kind)

    presets = list_presets(k)
    if not presets:
        raise RuntimeError(f"No presets found for kind='{k}'")

    print(f"\nAvailable {k} presets:\n")
    for i, name in enumerate(presets, 1):
        print(f"  {i}) {name}")

    idx = _ask_choice(
        f"\nChoose preset [1-{len(presets)} or name]: ",
        presets,
    )
    name = presets[idx]

    if return_config:
        cfg = load_preset(k, name)  # returns dict from your YAML
        return name, cfg

    return name


__all__ = [
    "choose_export_folder",
    "choose_ply_file",
    "choose_preset",
]
