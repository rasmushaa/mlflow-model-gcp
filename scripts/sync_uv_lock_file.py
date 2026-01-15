#!/usr/bin/env python3
"""
Sync uv lock file when pyproject.toml changes.
This ensures the lock file is always up to date before committing.
If pyproject.toml is modified manually, the lock file will be updated accordingly.
"""
import subprocess
import sys
from pathlib import Path


def run_uv_sync() -> int:
    """
    Run uv sync to update lock file and sync dependencies.

    Returns:
        0 if successful or no changes needed, 1 if lock file was updated
    """
    try:
        # Check if pyproject.toml exists
        if not Path("pyproject.toml").exists():
            print("No pyproject.toml found, skipping uv sync")
            return 0

        print("Running uv sync to update lock file...")
        result = subprocess.run(
            ["uv", "sync", "--frozen"], capture_output=True, text=True, check=False
        )

        # If frozen mode fails, the lock file is out of sync
        if result.returncode != 0:
            print("Lock file out of sync, updating...")
            result = subprocess.run(
                ["uv", "sync"], capture_output=True, text=True, check=True
            )
            print("✓ Lock file updated successfully")
            return 1  # Signal that files were modified
        else:
            print("✓ Lock file is up to date")
            return 0

    except subprocess.CalledProcessError as e:
        print(f"Error running uv sync: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("Error: uv command not found. Please install uv.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_uv_sync())
