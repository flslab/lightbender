"""Portable, isolated subprocess runtime for the LightBender Blender add-on.

Only the standard library is imported in Blender. Scientific dependencies live
in a private venv and are never added to Blender's sys.path.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def authoring_directory():
    bundled = ROOT / "authoring"
    return bundled if bundled.is_dir() else ROOT


def clean_environment():
    env = os.environ.copy()
    # Blender, activated environments and macOS launchers can otherwise redirect
    # a child interpreter back to Blender's own standard library/site-packages.
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def venv_python(directory):
    directory = Path(directory)
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_python(override=""):
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError("The Python executable selected in Preferences does not exist")
        return path
    # sys.executable may be the Blender application, never execute it as Python.
    candidates = []
    if Path(sys.executable).name.lower().startswith("python"):
        candidates.append(Path(sys.executable))
    for prefix in (sys.prefix, sys.base_prefix):
        root = Path(prefix)
        candidates.extend((root / "bin" / ("python%d.%d" % sys.version_info[:2]),
                           root / "bin/python3", root / "bin/python",
                           root / "bin/python.exe", root / "python.exe"))
    for command in ("python3", "python"):
        found = shutil.which(command)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("No Python 3.9+ executable found. Select one in add-on Preferences")


def launch_terminal(command, directory):
    """Keep the external hardware stack's input() confirmations interactive."""
    command = list(map(str, command))
    if sys.platform == "win32":
        return subprocess.Popen(command, cwd=directory, env=clean_environment(),
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
    if sys.platform == "darwin":
        # Shell quoting is needed only at Terminal's text-command boundary.
        # JSON quoting safely embeds that entire command in AppleScript.
        shell_command = "cd " + shlex.quote(str(directory)) + " && " + shlex.join([
            "env", "-u", "PYTHONHOME", "-u", "PYTHONPATH", "-u", "VIRTUAL_ENV",
            "-u", "__PYVENV_LAUNCHER__", *command])
        script = ('tell application "Terminal"\nactivate\ndo script '
                  + json.dumps(shell_command, ensure_ascii=False) + '\nend tell')
        return subprocess.Popen(["osascript", "-e", script], env=clean_environment())
    terminals = (("x-terminal-emulator", ["-e"]),
                 ("gnome-terminal", ["--working-directory", str(directory), "--"]),
                 ("konsole", ["--workdir", str(directory), "-e"]),
                 ("xfce4-terminal", ["--working-directory", str(directory), "--execute"]),
                 ("xterm", ["-e"]))
    for executable, arguments in terminals:
        found = shutil.which(executable)
        if found:
            return subprocess.Popen([found, *arguments, *command], cwd=directory,
                                    env=clean_environment())
    raise RuntimeError("Live hardware needs an interactive terminal (xterm, GNOME Terminal, Konsole, or equivalent)")


class RuntimeManager:
    def __init__(self, storage):
        self.storage = Path(storage)
        self.process = None
        self.log_handle = None
        self.environment = None
        self.fingerprint = None
        self.message = "Python tools have not been set up yet"

    @property
    def log_path(self):
        return self.storage / "setup.log"

    def _configuration(self, override, gurobi):
        python = find_python(override)
        files = [ROOT / "requirements-addon.txt"]
        if gurobi:
            files.append(ROOT / "requirements-addon-gurobi.txt")
        digest = hashlib.sha256()
        digest.update(str(python).encode())
        digest.update((sys.platform + platform.machine()).encode())
        stat = python.stat()
        digest.update(str((stat.st_mtime_ns, stat.st_size)).encode())
        for path in files:
            digest.update(path.read_bytes())
        fingerprint = digest.hexdigest()
        return python, files, fingerprint, self.storage / ("venv-" + fingerprint[:16])

    def _ready(self):
        if self.environment is None:
            return False
        try:
            marker = json.loads((self.environment / "lightbender-runtime.json").read_text())
            return (isinstance(marker, dict) and marker.get("fingerprint") == self.fingerprint
                    and venv_python(self.environment).is_file())
        except (OSError, ValueError):
            return False

    def poll(self):
        if self.process is not None:
            result = self.process.poll()
            if result is None:
                return False
            self.process = None
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None
            if result != 0 or not self._ready():
                self.message = "Setup failed. See setup.log, then retry in Preferences"
                return False
        if self._ready():
            self.message = "Python tools ready"
            return True
        return False

    def start(self, override="", gurobi=False, allow_install=True):
        if self.process is not None and self.process.poll() is None:
            self.message = "Installing Python tools (see setup.log)"
            return False
        self.poll()
        try:
            python, files, fingerprint, directory = self._configuration(override, gurobi)
            self.environment, self.fingerprint = directory, fingerprint
            if self._ready():
                self.message = "Python tools ready"
                return True
            if not allow_install:
                self.message = "Python setup needs online access. Enable it, then retry in Preferences"
                return False
            self.storage.mkdir(parents=True, exist_ok=True)
            command = [str(python), "-I", str(ROOT / "bootstrap_runtime.py"),
                       "--environment", str(directory), "--fingerprint", fingerprint]
            for path in files:
                command.extend(("--requirements", str(path)))
            self.log_handle = self.log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(command, stdout=self.log_handle,
                                            stderr=subprocess.STDOUT,
                                            env=clean_environment())
            self.message = "Installing Python tools (see setup.log)"
        except (OSError, RuntimeError) as error:
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None
            self.environment = None
            self.fingerprint = None
            self.message = "Python setup failed: " + str(error)
        return False

    def python(self):
        if not self.poll():
            raise RuntimeError(self.message + ". Open LightBender add-on Preferences")
        return str(venv_python(self.environment))

    def run(self, arguments, **kwargs):
        return subprocess.run([self.python(), *map(str, arguments)],
                              env=clean_environment(), capture_output=True,
                              text=True, **kwargs)

    def load_yaml(self, path):
        result = self.run(["-c", "import json,sys,yaml; "
                           "f=open(sys.argv[1],encoding='utf-8'); "
                           "print(json.dumps(yaml.safe_load(f)))", path], check=True)
        return json.loads(result.stdout)

    def close(self):
        # Setup can finish after disabling the add-on; no Python callbacks or
        # Blender state remain in the child process.
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None
