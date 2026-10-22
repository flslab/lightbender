#!/usr/bin/env python3
"""Build the standalone legacy add-on ZIP without importing Blender.

Run ``python authoring/build_addon.py`` from any working directory. Only the
files listed here enter the archive; local data, environments, and credentials
are never collected by walking the repository.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


PACKAGE_NAME = "lightbender_swarm"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED_FILES = {
    "__init__.py": "blender_addon.py",
    "addon_runtime.py": "addon_runtime.py",
    "bootstrap_runtime.py": "bootstrap_runtime.py",
    "requirements-addon.txt": "requirements-addon.txt",
    "README.md": "README.md",
    "BLENDER_ADDON_RELEASE.md": "BLENDER_ADDON_RELEASE.md",
    "authoring/transform.py": "transform.py",
    "authoring/place.py": "place.py",
    "authoring/set_cover_gurobi_f.py": "set_cover_gurobi_f.py",
    "authoring/deconflict.py": "deconflict.py",
    "authoring/morph_match.py": "morph_match.py",
    "authoring/camera_facing.py": "camera_facing.py",
}
OPTIONAL_FILES = {
    "requirements-addon-gurobi.txt": "requirements-addon-gurobi.txt",
}


def addon_version(source):
    """Read the literal bl_info version without executing add-on code."""
    module = ast.parse(source, filename="blender_addon.py")
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "bl_info"
            for target in node.targets
        ):
            try:
                version = ast.literal_eval(node.value)["version"]
            except (KeyError, TypeError, ValueError, SyntaxError) as error:
                raise ValueError("bl_info must contain a literal version tuple") from error
            if (
                not isinstance(version, tuple)
                or not 2 <= len(version) <= 3
                or any(type(part) is not int or part < 0 for part in version)
            ):
                raise ValueError("bl_info.version must have two or three nonnegative integers")
            return ".".join(str(part) for part in (*version, *([0] if len(version) == 2 else [])))
    raise ValueError("blender_addon.py does not define bl_info")


def release_files(source_dir):
    """Read and validate an allowlisted source tree into archive-relative files."""
    source_dir = Path(source_dir).resolve()
    missing = [source for source in REQUIRED_FILES.values() if not (source_dir / source).is_file()]
    if missing:
        raise FileNotFoundError("Missing add-on release files: " + ", ".join(sorted(missing)))
    files = {
        destination: (source_dir / source).read_bytes()
        for destination, source in REQUIRED_FILES.items()
    }
    for destination, source in OPTIONAL_FILES.items():
        path = source_dir / source
        if path.is_file():
            files[destination] = path.read_bytes()
    for name, content in files.items():
        if name.endswith(".py"):
            # Compile to catch broken source before producing an installable ZIP.
            compile(content, name, "exec", dont_inherit=True)
    version = addon_version(files["__init__.py"].decode("utf-8"))
    manifest = {
        "package": PACKAGE_NAME,
        "version": version,
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
    }
    files["release_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return version, files


def build_release(source_dir=None, output=None):
    """Return the archive path after writing it and a SHA-256 checksum file."""
    source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parent
    version, files = release_files(source_dir)
    output = Path(output) if output is not None else source_dir / "dist" / f"{PACKAGE_NAME}-{version}.zip"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".addon-", suffix=".zip", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                entry = zipfile.ZipInfo(f"{PACKAGE_NAME}/{name}", date_time=FIXED_ZIP_TIME)
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, content, compresslevel=9)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="ZIP destination (default: authoring/dist/lightbender_swarm-VERSION.zip)")
    args = parser.parse_args()
    output = build_release(output=args.output)
    print(output)
    print(output.with_suffix(output.suffix + ".sha256"))


if __name__ == "__main__":
    main()
