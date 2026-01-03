"""
Pre-commit hook to ensure version is bumped when source code changes.
The script checks if any source files in a specified directory have been modified in the staged changes.
The current version is extracted from a staged `polymodel/pyproject.toml`,
and compared to the version in the latest remote main commit.
"""

import re
import subprocess
import sys
from typing import Optional, Tuple

# ANSI color codes
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

# Configuration
SOURCE_PATTERNS = [
    "polymodel/",
]
VERSION_FILE = "polymodel/pyproject.toml"


def run_git_command(command: list) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"{RED}Error running git command: {e}{NC}", file=sys.stderr)
        sys.exit(1)


def get_staged_files() -> list:
    """Get list of staged files."""
    output = run_git_command(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    )
    return output.split("\n") if output else []


def get_committed_diff_files_against_remote_branch(branch: str) -> list:
    """Get files changed on the current branch.

    Parameters
    ----------
    branch : str
        The remote branch to compare against (e.g., 'main').
    """
    try:
        output = run_git_command(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACM",
                f"origin/{branch}...HEAD",
            ]
        )
        return output.split("\n") if output else []
    except Exception:
        return []


def source_files_changed(staged_files: list) -> bool:
    """Check if any source files have been modified."""
    for file in staged_files:
        for pattern in SOURCE_PATTERNS:
            if file.startswith(pattern):
                return True
    return False


def extract_version_from_pyproject(content: str) -> Optional[str]:
    """Extract version from pyproject.toml content."""
    # Match version = "x.y.z" or version = 'x.y.z'
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def get_current_version() -> Optional[str]:
    """Get the current staged version from pyproject.toml."""
    try:
        # Get the staged content of pyproject.toml
        content = run_git_command(["git", "show", f":{VERSION_FILE}"])
        return extract_version_from_pyproject(content)
    except Exception:
        return None


def get_previous_version(branch: str) -> Optional[str]:
    """Get the version from remote branch.

    The function tries to fetch the specified remote branch and read the version from it.
    If the file is not found on the remote branch, it falls back to HEAD.

    Parameters
    ----------
    branch : str
        The remote branch to compare against (e.g., 'main').
    """
    try:
        # Try fetching remote main to ensure origin/main exists locally
        subprocess.run(
            ["git", "fetch", "origin", branch],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        content = run_git_command(["git", "show", f"origin/{branch}:{VERSION_FILE}"])
        return extract_version_from_pyproject(content)
    except Exception:
        try:
            content = run_git_command(["git", "show", f"HEAD:{VERSION_FILE}"])
            return extract_version_from_pyproject(content)
        except Exception:
            return None


def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse a semantic version string into a tuple of integers.
    Supports formats like:  1.2.3, 1.2.3a1, 1.2.3.dev0, 1.2.3-alpha, etc.
    """
    # Extract the numeric part (major. minor.patch) from the version
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    return tuple(int(x) for x in match.groups())


def is_version_increased(old_version: str, new_version: str) -> bool:
    """
    Check if new_version is greater than old_version.
    Returns True if new_version > old_version.
    """
    try:
        old_tuple = parse_version(old_version)
        new_tuple = parse_version(new_version)

        return new_tuple > old_tuple
    except ValueError as e:
        print(f"{YELLOW}Warning: Could not parse versions: {e}{NC}", file=sys.stderr)
        return False


def main():

    # Validate we're in a git repository
    try:
        run_git_command(["git", "rev-parse", "--git-dir"])
    except SystemExit:
        print(f"{RED}Not in a git repository{NC}", file=sys.stderr)
        sys.exit(1)

    # Get staged files and committed diffs vs origin/main
    staged_files = get_staged_files()
    committed_files = get_committed_diff_files_against_remote_branch(branch="main")

    # Union of changed files we care about (staged + committed-but-not-in-main)
    changed_files = set(f for f in (staged_files + committed_files) if f)

    # Nothing changed, exit successfully
    if not changed_files:
        print(f"{GREEN}No files staged or changed against origin/main{NC}")
        sys.exit(0)

    # No source code changes, exit successfully
    if not source_files_changed(changed_files):
        print(f"{GREEN}✓ No source code changes detected against origin/main{NC}")
        sys.exit(0)

    print(
        f"{YELLOW}Polymodel source code changes detected vs origin/main, checking version...{NC}"
    )

    # Get versions: staged version (index) and remote main version
    current_version = get_current_version()
    previous_version = get_previous_version(branch="main")

    # If version file wasn't staged, warn but still compare working/index version if present
    if VERSION_FILE not in staged_files:
        print(
            f"{YELLOW}Note: {VERSION_FILE} was not staged. Using staged/index version if available.{NC}"
        )

    if not current_version:
        print(
            f"{RED}ERROR: Could not extract version from staged {VERSION_FILE}{NC}",
            file=sys.stderr,
        )
        sys.exit(1)

    # If there's no previous version on remote/main, this might be the first main commit
    if not previous_version:
        print(
            f"{GREEN}✓ No version found on origin/main; current version is {current_version}{NC}"
        )
        sys.exit(0)

    # Check if version was actually increased above remote main
    if current_version == previous_version:
        print(
            f"{RED}ERROR: Version has not changed relative to origin/main! {NC}",
            file=sys.stderr,
        )
        print(
            f"{RED}Current staged/index version: {current_version}{NC}", file=sys.stderr
        )
        print(
            f"{RED}Remote origin/main version: {previous_version}{NC}", file=sys.stderr
        )
        print(
            f"{RED}Please increment the version in {VERSION_FILE} to be greater than origin/main{NC}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not is_version_increased(previous_version, current_version):
        print(
            f"{RED}ERROR: Version on this branch is not greater than origin/main!{NC}",
            file=sys.stderr,
        )
        print(
            f"{RED}Remote origin/main version: {previous_version}{NC}", file=sys.stderr
        )
        print(f"{RED}Your staged/index version: {current_version}{NC}", file=sys.stderr)
        print(
            f"{RED}New version must be greater than the version on origin/main{NC}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"{GREEN}✓ Version properly bumped {previous_version} → {current_version}{NC} against origin/main"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
