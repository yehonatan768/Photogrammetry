from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from src.config.base_config import EXPORTS_DIR
from src.utils.preset_loader import list_presets, load_preset, PresetType


# =============================================================
# Internal helpers
# =============================================================

def _ask_choice(prompt: str, options: List[str]) -> int:
    """
    Show a prompt and return the 0-based index of the chosen option.
    Allows selecting by number, exact name, or prefix.
    """
    if not options:
        raise ValueError("Options list is empty.")

    while True:
        sel = input(prompt).strip()
        if not sel:
            print("Please enter a number or a name/prefix.")
            continue

        # numeric input
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(options):
                return i - 1

        low = sel.lower()
        # exact match
        for idx, name in enumerate(options):
            if name.lower() == low:
                return idx
        # unique prefix match
        prefix_matches = [idx for idx, name in enumerate(options) if name.lower().startswith(low)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        print("Invalid choice. Try again.")


def _normalize_kind(kind: str) -> PresetType:
    """Normalize user-friendly kind aliases to canonical preset folder names."""
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
            f"Unknown preset kind '{kind}' (expected one of: points, points_filtering, "
            f"clouds, clouds_filtering, mesh)"
        )
    return k  # type: ignore[return-value]


# =============================================================
# Public API — Interactive Menus
# =============================================================

def choose_export_folder() -> Path:
    """
    Pick an export folder under EXPORTS_DIR (newest first, interactive).
    """
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory found at {EXPORTS_DIR}")

    folders = sorted(
        (p for p in EXPORTS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not folders:
        raise FileNotFoundError(f"No export folders under: {EXPORTS_DIR}")

    print("\nAvailable export folders:\n")
    for i, f in enumerate(folders, 1):
        print(f"  {i}) {f.name}")

    idx = _ask_choice(f"\nChoose export folder [1-{len(folders)} or name]: ",
                      [f.name for f in folders])
    return folders[idx]


def choose_ply_file(export_dir: Path) -> Path:
    """
    Choose a .ply file from export_dir (interactive).
    Priority order:
        filtered.ply → light_filtered.ply → point_cloud.ply → mesh.ply → newest.
    """
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    ply_files = sorted(
        (p for p in export_dir.glob("*.ply") if p.is_file()),
        key=lambda p: ({"filtered.ply": 0, "light_filtered.ply": 1,
                        "point_cloud.ply": 2, "mesh.ply": 99}.get(p.name, 10),
                       -p.stat().st_mtime),
    )
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {export_dir}")

    if len(ply_files) == 1:
        print(f"\nFound single .ply: {ply_files[0].name} (auto-selected)")
        return ply_files[0]

    print("\nPLY files in folder:\n")
    for i, f in enumerate(ply_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}) {f.name:25} ({size_mb:.1f} MB)")

    idx = _ask_choice(f"\nChoose input .ply [1-{len(ply_files)} or name]: ",
                      [f.name for f in ply_files])
    return ply_files[idx]


def choose_preset(
    kind: str,
    *,
    return_config: bool = False,
) -> Union[str, Tuple[str, dict]]:
    """
    Interactively choose a preset name or (name, config dict) if return_config=True.
    kind ∈ {"points", "points_filtering", "clouds", "clouds_filtering", "mesh"}.
    """
    k: PresetType = _normalize_kind(kind)
    presets = list_presets(k)
    if not presets:
        raise RuntimeError(f"No presets found for kind='{k}'")

    print(f"\nAvailable {k} presets:\n")
    for i, name in enumerate(presets, 1):
        print(f"  {i}) {name}")

    idx = _ask_choice(f"\nChoose preset [1-{len(presets)} or name]: ", presets)
    name = presets[idx]

    if return_config:
        return name, load_preset(k, name)
    return name


# =============================================================
# Helpers for automated (non-interactive) selections
# =============================================================

def _get_export_folders() -> list[Path]:
    """Return sorted export folders (newest first)."""
    if not EXPORTS_DIR.exists():
        return []
    return sorted(
        (p for p in EXPORTS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def auto_pick_latest_export_dir() -> Path:
    """Return the newest export folder (non-interactive)."""
    folders = _get_export_folders()
    if not folders:
        raise FileNotFoundError(f"No export folders found under {EXPORTS_DIR}")
    return folders[0]


def auto_pick_input_ply(export_dir: Path) -> Path:
    """Pick the best .ply file automatically, using the same priority as the menu."""
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    ply_files = sorted(
        (p for p in export_dir.glob("*.ply") if p.is_file()),
        key=lambda p: ({"filtered.ply": 0, "light_filtered.ply": 1,
                        "point_cloud.ply": 2, "mesh.ply": 99}.get(p.name, 10),
                       -p.stat().st_mtime),
    )
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {export_dir}")
    return ply_files[0]


def resolve_export_and_input(
    folder_arg: str | None,
    input_arg: str | None,
) -> Tuple[Path, Path]:
    """
    Unified resolver that supports both CLI args and interactive menus.
    - If folder_arg/input_arg is None → use menu()
    - If "auto" → auto-pick newest/most relevant
    - Else → interpret as exact name/path
    """
    # Folder selection
    if folder_arg is None:
        export_dir = choose_export_folder()
    elif folder_arg.lower() == "auto":
        export_dir = auto_pick_latest_export_dir()
        print(f"Auto-selected export folder: {export_dir.name}")
    else:
        export_dir = EXPORTS_DIR / folder_arg
        if not export_dir.exists():
            raise FileNotFoundError(f"Export folder not found: {export_dir}")

    # Input selection
    if input_arg is None:
        in_ply = choose_ply_file(export_dir)
    elif input_arg.lower() == "auto":
        in_ply = auto_pick_input_ply(export_dir)
        print(f"Auto-selected input: {in_ply.name}")
    else:
        in_ply = export_dir / input_arg
        if not in_ply.exists():
            raise FileNotFoundError(f"Input .ply not found: {in_ply}")

    return export_dir, in_ply


__all__ = [
    "choose_export_folder",
    "choose_ply_file",
    "choose_preset",
    "resolve_export_and_input",
    "auto_pick_latest_export_dir",
    "auto_pick_input_ply",
]
