#!/usr/bin/env python3
"""Update MLflow version badge in README.md based on polymodel/pyproject.toml."""

import re
import sys
import tomllib
from pathlib import Path


def get_mlflow_version_from_polymodel() -> str:
    """Extract MLflow version from polymodel dependencies."""
    toml_path = Path("polymodel/pyproject.toml")
    if not toml_path.exists():
        print("❌ polymodel/pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    dependencies = data["project"].get("dependencies", [])

    # Find MLflow dependency
    mlflow_dep = None
    for dep in dependencies:
        if dep.startswith("mlflow"):
            mlflow_dep = dep
            break

    if not mlflow_dep:
        print(
            "❌ MLflow dependency not found in polymodel/pyproject.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract version from patterns like "mlflow==3.6.0" or "mlflow>=3.6.0"
    version_match = re.search(r"==(\d+\.\d+\.\d+)", mlflow_dep)
    if not version_match:
        # Try other version specifiers
        version_match = re.search(r">=(\d+\.\d+\.\d+)", mlflow_dep)

    if not version_match:
        print(f"❌ Could not parse MLflow version from: {mlflow_dep}", file=sys.stderr)
        sys.exit(1)

    return version_match.group(1)


def update_readme(version: str) -> bool:
    """Update the MLflow version badge in README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ README.md not found", file=sys.stderr)
        sys.exit(1)

    content = readme_path.read_text()

    # Pattern to match the MLflow version badge
    pattern = (
        r"\[!\[MLflow Version\]\(https://img\.shields\.io/badge/mlflow-v?[^\)]+\)\]"
    )
    new_badge = f"[![MLflow Version](https://img.shields.io/badge/mlflow-v{version}-orange.svg)]"

    new_content = re.sub(pattern, new_badge, content)

    if new_content == content:
        print("✓ MLflow version badge already up to date")
        return False

    readme_path.write_text(new_content)
    print(f"✓ Updated MLflow version badge to: v{version}")
    return True


def main():
    """Main entry point."""
    version = get_mlflow_version_from_polymodel()
    changed = update_readme(version)

    if changed:
        sys.exit(1)  # Signal pre-commit that file was modified
    sys.exit(0)


if __name__ == "__main__":
    main()
