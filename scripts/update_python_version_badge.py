#!/usr/bin/env python3
"""Update Python version badge in README.md based on pyproject.toml requires-python."""

import re
import sys
import tomllib
from pathlib import Path


def get_python_version_from_toml() -> str:
    """Extract Python version from pyproject.toml requires-python field."""
    toml_path = Path("pyproject.toml")
    if not toml_path.exists():
        print("❌ pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    requires_python = data["project"].get("requires-python")
    if not requires_python:
        print("❌ No requires-python found in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    # Extract version from patterns like ">=3.11", ">=3.11,<4.0", "==3.11.*"
    version_match = re.search(r"(\d+\.\d+)", requires_python)
    if not version_match:
        print(f"❌ Could not parse version from: {requires_python}", file=sys.stderr)
        sys.exit(1)

    return version_match.group(1)


def format_version_badge(version: str) -> str:
    """Format version into badge URL format."""
    # URL-encode the version with a plus sign: "3.11+" for ">=3.11"
    return f"{version}%2B"  # %2B is URL-encoded +


def update_readme(version: str) -> bool:
    """Update the Python version badge in README.md."""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("❌ README.md not found", file=sys.stderr)
        sys.exit(1)

    content = readme_path.read_text()
    version_badge = format_version_badge(version)

    # Pattern to match the Python version badge
    pattern = r"!\[Python Version\]\(https://img\.shields\.io/badge/python-[^\)]+\)"
    new_badge = f"![Python Version](https://img.shields.io/badge/python-{version_badge}-blue.svg)"

    new_content, count = re.subn(pattern, new_badge, content, count=1)

    # If no badge exists, add it after the title
    if count == 0:
        lines = content.split("\n")
        if len(lines) > 0 and lines[0].startswith("#"):
            # Find first empty line after title or first badge
            insert_index = 2  # Default after title and empty line
            for i, line in enumerate(lines[1:], start=1):
                if "![" in line:  # Found another badge
                    insert_index = i
                    break
                if line.strip() != "" and not line.startswith("#"):
                    break
            lines.insert(insert_index, new_badge)
            new_content = "\n".join(lines)
            print(f"✓ Added Python version badge: {version}+")
        else:
            print("⚠ Could not find appropriate location to insert badge")
            return False
    else:
        print(f"✓ Updated Python version badge to: {version}+")
        if new_content == content:
            print("✓ Python version badge already up to date")
            return False

    readme_path.write_text(new_content)
    print(f"✓ Updated Python version badge to: {version}+")
    return True


def main():
    """Main entry point."""
    version = get_python_version_from_toml()
    changed = update_readme(version)

    if changed:
        sys.exit(1)  # Signal pre-commit that file was modified
    sys.exit(0)


if __name__ == "__main__":
    main()
