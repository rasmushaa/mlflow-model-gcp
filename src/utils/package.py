import hashlib
import logging
import pathlib
import re
import subprocess

logger = logging.getLogger(__name__)

# The model wheel source directory
MODEL_SRC = pathlib.Path("./polymodel")
DIST_DIR = MODEL_SRC.joinpath("dist")
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


def get_hash_of_file() -> str:
    return hashlib.sha256(TOML_PATH.read_bytes()).hexdigest()


def build_wheel() -> pathlib.Path:
    """Build the model package wheel

    Builds the package wheel using 'uv build', and returns the
    path to the built wheel file.

    Returns
    -------
    wheel_path: pathlib.Path
        The path to the built wheel file, e.g., dist/polymodel-1.0.1-py3-none-any.whl
    """

    if DIST_DIR.exists():
        for f in DIST_DIR.glob("*"):
            f.unlink()

    subprocess.run(["uv", "build", str(MODEL_SRC), "-o", str(DIST_DIR)])

    whl_path = next(DIST_DIR.glob("*.whl"))

    logger.info(f"Built wheel: {whl_path.name}")
    return whl_path
