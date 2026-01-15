#!/usr/bin/env python3
"""
Remove outputs from Jupyter notebooks.
This script is designed to be used as a pre-commit hook to keep notebooks clean.
"""
import json
import sys
from pathlib import Path


def clear_notebook_outputs(notebook_path: Path) -> bool:
    """
    Clear all outputs from a Jupyter notebook.

    Args:
        notebook_path: Path to the notebook file

    Returns:
        True if the notebook was modified, False otherwise
    """
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {notebook_path}: {e}", file=sys.stderr)
        return False

    modified = False

    # Clear outputs and execution counts from all cells
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                modified = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                modified = True

    # Clear notebook-level execution count if present
    if notebook.get("metadata", {}).get("execution"):
        notebook["metadata"].pop("execution", None)
        modified = True

    # Write back only if modified
    if modified:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
            f.write("\n")  # Add trailing newline
        print(f"Cleared outputs from {notebook_path}")

    return modified


def main():
    """Process all notebook files passed as arguments."""
    if len(sys.argv) < 2:
        print(
            "Usage: remove_notebook_outputs.py <notebook1.ipynb> [notebook2.ipynb ...]"
        )
        sys.exit(0)

    modified_count = 0

    for notebook_file in sys.argv[1:]:
        notebook_path = Path(notebook_file)

        if not notebook_path.suffix == ".ipynb":
            continue

        if clear_notebook_outputs(notebook_path):
            modified_count += 1

    if modified_count > 0:
        print(f"\nModified {modified_count} notebook(s)")
        sys.exit(1)  # Exit with 1 to signal pre-commit that files were modified
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
