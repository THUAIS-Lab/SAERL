"""Detect an incompatible or locally shadowed verl before starting Ray."""

import hashlib
import importlib.metadata
import importlib.util
import json
from importlib.resources import files
from pathlib import Path


def verify_verl():
    audit = json.loads(files(__package__).joinpath("upstream.json").read_text())
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install SAERL's train extra with the pinned official verl dependency first")
    root = Path(spec.origin).resolve().parent
    mismatches = []
    for name, expected in audit["git_blobs"].items():
        path = root / name
        if not path.is_file():
            mismatches.append(name)
            continue
        content = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != expected:
            mismatches.append(name)
    try:
        version = importlib.metadata.version("verl")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if mismatches or version != audit["version"]:
        raise RuntimeError(
            f"SAERL requires official verl {audit['commit']} ({audit['version']}); found {root} "
            f"version={version}, mismatched files={mismatches}. Install in a separate environment and run "
            "outside the old verl checkout; remove old editable installations/PYTHONPATH entries."
        )
    return root


if __name__ == "__main__":
    print(verify_verl())
