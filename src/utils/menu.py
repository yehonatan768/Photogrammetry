from pathlib import Path
from typing import List
from src.config.base_config import EXPORTS_DIR


def _ask_choice(prompt: str, options: List[str]) -> int:
    """
    Show a prompt and return index (0-based) of the chosen option.
    Allows entering either the number or exact/prefix text.
    """
    while True:
        sel = input(prompt).strip()
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(options):
                return i - 1
        # also allow prefix match by text
        low = sel.lower()
        exact = [idx for idx, name in enumerate(options) if name.lower() == low]
        if len(exact) == 1:
            return exact[0]
        pref = [idx for idx, name in enumerate(options) if name.lower().startswith(low)]
        if len(pref) == 1:
            return pref[0]

        print("Invalid choice. Try again.")


def choose_export_folder() -> Path:
    """
    Pick an export folder under EXPORTS_DIR (newest first).
    """
    if not EXPORTS_DIR.exists():
        raise FileNotFoundError(f"No exports directory at: {EXPORTS_DIR}")

    candidates = [p for p in EXPORTS_DIR.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folders found under: {EXPORTS_DIR}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    print("\nAvailable export folders:\n")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}) {p.name}")

    idx = _ask_choice(f"\nChoose export folder [1-{len(candidates)} or name]: ", [c.name for c in candidates])
    return candidates[idx]


def choose_ply_file(export_dir: Path) -> Path:
    """
    Choose a .ply from export_dir.
    Priority: filtered.ply -> light_filtered.ply -> point_cloud.ply -> mesh.ply -> rest by mtime.
    """
    ply_files = [p for p in export_dir.glob("*.ply") if p.is_file()]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {export_dir}")

    # sorting policy from your code
    priority = {"filtered.ply": 0, "light_filtered.ply": 1, "point_cloud.ply": 2, "mesh.ply": 99}
    ply_files.sort(key=lambda p: (priority.get(p.name, 10), -p.stat().st_mtime))

    if len(ply_files) == 1:
        print(f"\nFound single .ply: {ply_files[0].name} (auto-selected)")
        return ply_files[0]

    print("\nPLY files in folder:\n")
    for i, f in enumerate(ply_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}) {f.name}  ({size_mb:.1f} MB)")

    idx = _ask_choice(f"\nChoose input .ply [1-{len(ply_files)} or name]: ", [p.name for p in ply_files])
    return ply_files[idx]


def choose_preset(kind: str) -> str:
    """
    kind in {"points","clouds","mesh"}.
    Returns the chosen preset NAME.
    """
    presets = get_presets(kind)
    if not presets:
        raise RuntimeError(f"No presets found for kind='{kind}'")

    print(f"\nAvailable {kind} presets:\n")
    for i, name in enumerate(presets, 1):
        print(f"  {i}) {name}")

    idx = _ask_choice(
        f"\nChoose preset [1-{len(presets)} or name]: ",
        presets
    )
    return presets[idx]
