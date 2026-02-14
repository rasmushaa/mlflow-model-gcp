#!/usr/bin/env python3
"""Update polymodel version badge in README.md based on polymodel/pyproject.toml."""

import re
import sys
import tomllib
from pathlib import Path


def get_polymodel_version() -> str:
    """Extract polymodel version from polymodel/pyproject.toml."""
    toml_path = Path("polymodel/pyproject.toml")
    if not toml_path.exists():
        print("❌ polymodel/pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    version = data["project"].get("version")
    if not version:
        print("❌ No version found in polymodel/pyproject.toml", file=sys.stderr)
        sys.exit(1)

    return version


def update_readme(version: str) -> bool:
    """Update the polymodel version badge in README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ README.md not found", file=sys.stderr)
        sys.exit(1)

    content = readme_path.read_text()

    # Pattern to match the polymodel version badge
    pattern = r"\[!\[Polymodel Version\]\(https://img\.shields\.io/badge/polymodel-v?[^\)]+\)\]\(https://github\.com/[^/]+/[^/]+\)"
    new_badge = f"[![Polymodel Version](https://img.shields.io/badge/polymodel-v{version}-green.svg)](https://github.com/rasmus/mlflow-model-gcp)"

    # Try to replace existing badge
    new_content = re.sub(pattern, new_badge, content)

    # If no badge exists, add it after the title (first line)
    if new_content == content:
        lines = content.split("\n")
        if len(lines) > 0 and lines[0].startswith("#"):
            # Find the line after title and description
            insert_index = 1
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == "":
                    insert_index = i + 1
                    break
                if line.startswith("#") or line.startswith("A "):
                    insert_index = i
                    break

            # Insert badge before the first content section
            lines.insert(insert_index, "")
            lines.insert(insert_index + 1, new_badge)
            lines.insert(insert_index + 2, "")
            new_content = "\n".join(lines)
            print(f"✓ Added polymodel version badge: v{version}")
        else:
            print("⚠ Could not find appropriate location to insert badge")
            return False
    else:
        print(f"✓ Updated polymodel version badge to: v{version}")

    readme_path.write_text(new_content)
    return True


def main():
    """Main entry point."""
    version = get_polymodel_version()
    changed = update_readme(version)

    if changed:
        sys.exit(1)  # Signal pre-commit that file was modified
    sys.exit(0)


if __name__ == "__main__":
    main()
