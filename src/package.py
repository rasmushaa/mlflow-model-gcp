import logging
import re
from importlib.metadata import version

logger = logging.getLogger(__name__)

PACKAGE = "polymodel"


def get_installed_polymodel_version() -> str:
    """Get the installed runtime version of the polymodel package.

    Returns
    -------
    version: str
        The installed version string, e.g., "1.0.1"
    """
    return version(PACKAGE)


def get_next_polymodel_major_version() -> str:
    """Get the next major version string for the polymodel package.

    If the current version is "1.2.3", the next major version will be "2.0.0".
    This is useful for setting version constraints in dependencies.

    Returns
    -------
    next_version: str
        The next major version string, e.g., "2.0.0"
    """
    current_version = get_installed_polymodel_version()
    major_version_match = re.match(r"(\d+)\..*", current_version)

    if major_version_match:
        major_version = int(major_version_match.group(1))
        return f"{major_version + 1}.0.0"

    raise ValueError(f"Invalid version format: {current_version}")
