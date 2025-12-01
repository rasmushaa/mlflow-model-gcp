"""
Update model requirements.txt to pin versions based on global requirements.txt
You have first updateh the global requirements.txt using `uv pip compile pyproject.toml -o requirements.txt`,
and then use `uv run pipreqs src/models --force --encoding=utf-8 --mode no-pin` to generate unpinned model requirements,
before running this script to sync the versions.
"""
from pathlib import Path

# Global uv pyproject requirements
freeze_map = {}
freeze = Path("requirements.txt").read_text().splitlines()
for line in freeze:
    if "==" in line:
        pkg, ver = line.split("==")
        freeze_map[pkg.lower()] = ver

# Pin versions in model requirements, if found in global requirements
out = []
reqs = Path("src/models/requirements.txt").read_text().splitlines()
for req in reqs:
    name = req.lower().replace("_", "-")
    if name in freeze_map:
        out.append(f"{req}=={freeze_map[name]}")
    else:
        out.append(req)  # no version found -> leave it unpinned

Path("src/models/requirements.txt").write_text("\n".join(out) + "\n")
