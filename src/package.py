import logging
import pathlib
import re

logger = logging.getLogger(__name__)

# The model wheel source directory
MODEL_SRC = pathlib.Path("./polymodel")
TOML_PATH = MODEL_SRC / "pyproject.toml"


def get_current_version() -> str:
    """Get the current version of the model package from pyproject.toml.

    Returns
    -------
    version: str
        The current version string, e.g., "1.0.1"
    """
    content = TOML_PATH.read_text()
    match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', content)
    if match:
        major, minor, patch = match.groups()
        return f"{major}.{minor}.{patch}"

    return "0.0.0"


def get_wheel_path() -> pathlib.Path:
    """Get the path to matching wheel file in the dist/ directory.

    Returns
    -------
    wheel_path: pathlib.Path
        The path to the wheel file
    """
    dist_dir = MODEL_SRC / "dist"
    version = get_current_version()
    wheel_pattern = f"polymodel-{version}-py3-none-any.whl"
    wheel_path = dist_dir / wheel_pattern
    return wheel_path


def wheel_exists() -> bool:
    """Check if a matching wheel file exists in the dist/ directory.

    The wheel should always exist since it is built during the Docker build process.
    This function mainly serves as a sanity check before logging the model to MLflow,
    and to handle local development scenarios where the wheel might not have been built manually.

    Returns
    -------
    exists: bool
        True if the wheel file exists, False otherwise
    """
    dist_dir = MODEL_SRC / "dist"
    version = get_current_version()
    wheel_pattern = f"polymodel-{version}-py3-none-any.whl"
    wheel_path = dist_dir / wheel_pattern
    exists = wheel_path.exists()
    if not exists:
        logger.warning(f"Wheel file not found: {wheel_path} at {dist_dir}")
    return exists
