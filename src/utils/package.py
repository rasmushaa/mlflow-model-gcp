import pathlib
import subprocess
import re

# The model wheel source directory
MODEL_SRC = pathlib.Path("./polymodel")
DIST_DIR = MODEL_SRC.joinpath("dist")
TOML_PATH = MODEL_SRC / "pyproject.toml"


def get_next_version():
    content = TOML_PATH.read_text()
    match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', content)
    if match:
        major, minor, patch = match.groups()
        new_patch = int(patch) + 1
        return f"{major}.{minor}.{new_patch}"
    
    return "1.0.0"
    

def update_version_in_pyproject(new_version):
    content = TOML_PATH.read_text()
    updated = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    TOML_PATH.write_text(updated)


def build_wheel() -> pathlib.Path:
    """ Build the model package wheel
    
    Automatically increments the patch version in pyproject.toml,
    builds the package wheel using 'uv build', and returns the
    path to the built wheel file.

    Returns
    -------
    wheel_path: pathlib.Path
        The path to the built wheel file, e.g., dist/polymodel-1.0.1-py3-none-any.whl
    """

    new_version = get_next_version()
    update_version_in_pyproject(new_version)
    
    if DIST_DIR.exists():
        for f in DIST_DIR.glob("*"): 
            f.unlink()

    subprocess.run(["uv", "build", str(MODEL_SRC), "-o", str(DIST_DIR)])

    return next(DIST_DIR.glob("*.whl"))