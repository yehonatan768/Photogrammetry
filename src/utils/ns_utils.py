from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def header(title: str) -> None:
    """Human-friendly section header."""
    bar = "=" * max(64, len(title) + 6)
    print(f"\n{bar}\n>>> {title}\n{bar}\n")


def run(cmd: str) -> None:
    """
    Execute a shell command and raise if it fails.

    We keep shell=True because nerfstudio exposes console CLIs and we
    want to pass them as one string.
    """
    header("Running command")
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)


def exists_on_path(tool: str) -> bool:
    """Return True if `tool` is discoverable on PATH."""
    return shutil.which(tool) is not None


def ensure_exists(path: Path, kind: str = "dir") -> None:
    """
    Ensure path exists (for dirs) or complain if file is missing.

    kind = "dir": create if missing
    kind = "file": raise if missing
    """
    if kind == "dir":
        path.mkdir(parents=True, exist_ok=True)
    elif kind == "file":
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    else:
        raise ValueError("kind must be 'dir' or 'file'")
