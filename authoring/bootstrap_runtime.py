"""Run outside Blender to create/repair the private authoring environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import venv


def install(environment, requirements, fingerprint):
    if sys.version_info < (3, 9):
        raise RuntimeError("LightBender tools require Python 3.9 or newer")
    environment = Path(environment)
    environment.parent.mkdir(parents=True, exist_ok=True)
    lock = environment.with_suffix(".lock")
    # Multiple Blender windows can enable the same add-on simultaneously.
    deadline = time.monotonic() + 1200
    while True:
        try:
            descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Setup is locked by another process. If no setup is running, remove " + str(lock))
            time.sleep(1)
    marker = environment / "lightbender-runtime.json"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if marker.is_file():
            try:
                data = json.loads(marker.read_text())
            except (OSError, ValueError):
                data = None
            if (isinstance(data, dict) and data.get("fingerprint") == fingerprint
                    and python.is_file()):
                return
        marker.unlink(missing_ok=True)
        print("Creating virtual environment:", environment, flush=True)
        # Repair an interrupted creation without wiping arbitrary directories.
        venv.EnvBuilder(with_pip=True, clear=False).create(str(environment))
        command = [str(python), "-I", "-m", "pip", "--isolated", "--require-virtualenv",
                   "--no-input", "install", "--disable-pip-version-check", "--only-binary=:all:"]
        for requirement in requirements:
            command.extend(("-r", str(requirement)))
        subprocess.run(command, check=True)
        # Never declare success just because pip exited: exercise the actual
        # imports used by transform/place/stagger/morph/camera-facing.
        imports = "import numpy, scipy.optimize, scipy.spatial, yaml, svg.path"
        if len(requirements) > 1:
            imports += ", gurobipy"
        subprocess.run([str(python), "-I", "-c", imports], check=True)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"fingerprint": fingerprint,
                                         "python": sys.version,
                                         "requirements": [str(p) for p in requirements]}), encoding="utf-8")
        temporary.replace(marker)
        print("LightBender Python tools ready", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--requirements", action="append", required=True)
    parser.add_argument("--fingerprint", required=True)
    args = parser.parse_args()
    install(args.environment, args.requirements, args.fingerprint)


if __name__ == "__main__":
    main()
