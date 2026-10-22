"""Exercise the add-on installer without Blender, pip, or network access."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import venv


AUTHORING = Path(__file__).resolve().parents[1] / "authoring"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, AUTHORING / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module("lightbender_addon_runtime_test", "addon_runtime.py")
bootstrap = load_module("lightbender_bootstrap_runtime_test", "bootstrap_runtime.py")


class EnvironmentTests(unittest.TestCase):
    def test_children_do_not_inherit_blender_or_activated_python(self):
        original = {
            "PYTHONHOME": "/blender/python",
            "PYTHONPATH": "/blender/site-packages",
            "VIRTUAL_ENV": "/another/venv",
            "__PYVENV_LAUNCHER__": "/another/python",
            "PATH": "/normal/search/path",
        }
        with mock.patch.dict(os.environ, original, clear=True):
            cleaned = runtime.clean_environment()
            self.assertEqual(dict(os.environ), original)
        self.assertEqual(cleaned, {
            "PATH": "/normal/search/path", "PYTHONNOUSERSITE": "1",
        })

    def test_environment_python_location_on_both_platforms(self):
        for platform, expected in (("posix", "bin/python"),
                                   ("nt", "Scripts/python.exe")):
            with self.subTest(platform=platform):
                with mock.patch.object(runtime, "os", SimpleNamespace(name=platform)):
                    self.assertEqual(runtime.venv_python("tools"), Path("tools") / expected)

    def test_find_python_never_returns_the_blender_application(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            blender = folder / "Blender"
            blender.touch()
            interpreter = folder / "python3"
            interpreter.touch()
            with mock.patch.object(runtime.sys, "executable", str(blender)), \
                    mock.patch.object(runtime.sys, "prefix", str(folder / "missing-prefix")), \
                    mock.patch.object(runtime.sys, "base_prefix", str(folder / "missing-base")), \
                    mock.patch.object(runtime.shutil, "which", return_value=str(interpreter)):
                self.assertEqual(runtime.find_python(), interpreter.resolve())

    def test_invalid_explicit_python_does_not_silently_use_another(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                runtime.find_python(str(Path(directory) / "missing-python"))

    def test_mac_terminal_preserves_unicode_and_quoted_argument_boundaries(self):
        command = ["/Users/caf\N{LATIN SMALL LETTER E WITH ACUTE}/Python Tools/python3",
                   "/tmp/author's repo/orchestrator.py", "--blender", "127.0.0.1:5598"]
        directory = "/tmp/author's repo"
        with mock.patch.object(runtime.sys, "platform", "darwin"), \
                mock.patch.object(runtime.subprocess, "Popen") as launch:
            runtime.launch_terminal(command, directory)
        arguments = launch.call_args.args[0]
        self.assertEqual(arguments[:2], ["osascript", "-e"])
        script = arguments[2]
        # AppleScript has no JSON-style Unicode escapes (\\u00e9).
        self.assertIn("caf\N{LATIN SMALL LETTER E WITH ACUTE}", script)
        literal = next(line[len("do script "):] for line in script.splitlines()
                       if line.startswith("do script "))
        shell_arguments = shlex.split(json.loads(literal))
        self.assertEqual(shell_arguments[:3], ["cd", directory, "&&"])
        self.assertEqual(shell_arguments[-len(command):], command)

    def test_windows_hardware_has_an_interactive_console_without_a_shell(self):
        command = [r"C:\Tools With Spaces\python.exe", r"C:\User's Tools\orchestrator.py"]
        with mock.patch.object(runtime.sys, "platform", "win32"), \
                mock.patch.object(runtime.subprocess, "CREATE_NEW_CONSOLE", 16, create=True), \
                mock.patch.object(runtime.subprocess, "Popen") as launch:
            runtime.launch_terminal(command, r"C:\User's Tools")
        self.assertEqual(launch.call_args.args[0], command)
        self.assertEqual(launch.call_args.kwargs["creationflags"], 16)
        self.assertNotIn("shell", launch.call_args.kwargs)

    def test_linux_hardware_uses_an_available_interactive_terminal(self):
        command = ["/tmp/tools/python", "/tmp/author's repo/orchestrator.py"]
        with mock.patch.object(runtime.sys, "platform", "linux"), \
                mock.patch.object(runtime.shutil, "which", side_effect=lambda name:
                                  "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None), \
                mock.patch.object(runtime.subprocess, "Popen") as launch:
            runtime.launch_terminal(command, "/tmp/author's repo")
        self.assertEqual(launch.call_args.args[0], [
            "/usr/bin/gnome-terminal", "--working-directory", "/tmp/author's repo", "--", *command,
        ])
        self.assertNotIn("shell", launch.call_args.kwargs)


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="LightBender runtime '")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "requirements-addon.txt").write_text("numpy>=1.26.4,<3\n")
        (self.root / "requirements-addon-gurobi.txt").write_text("gurobipy>=12,<14\n")
        self.interpreter = self.root / "python3"
        self.interpreter.touch()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(runtime, "ROOT", self.root).start()
        mock.patch.object(runtime, "find_python", return_value=self.interpreter).start()
        self.manager = runtime.RuntimeManager(self.root / "user data")
        self.addCleanup(self.manager.close)

    def mark_ready(self, fingerprint=None, directory=None):
        if fingerprint is None:
            _, _, fingerprint, directory = self.manager._configuration("", False)
        directory.mkdir(parents=True, exist_ok=True)
        python = runtime.venv_python(directory)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        (directory / "lightbender-runtime.json").write_text(
            json.dumps({"fingerprint": fingerprint}))
        return directory

    def test_completed_install_is_reused_without_online_access_or_processes(self):
        expected = self.mark_ready()
        with mock.patch.object(runtime.subprocess, "Popen") as launch:
            self.assertTrue(self.manager.start(allow_install=False))
            self.assertTrue(self.manager.start(allow_install=False))
        launch.assert_not_called()
        self.assertEqual(self.manager.python(), str(runtime.venv_python(expected)))

    def test_requirements_and_optional_solver_changes_invalidate_installation(self):
        self.mark_ready()
        self.assertTrue(self.manager.start(allow_install=False))
        initial_environment = self.manager.environment
        (self.root / "requirements-addon.txt").write_text("numpy>=2,<3\n")
        self.assertFalse(self.manager.start(allow_install=False))
        self.assertNotEqual(self.manager.environment, initial_environment)
        self.mark_ready(self.manager.fingerprint, self.manager.environment)
        self.assertTrue(self.manager.start(allow_install=False))
        self.assertFalse(self.manager.start(gurobi=True, allow_install=False))

    def test_invalid_marker_never_declares_python_ready(self):
        environment = self.mark_ready()
        self.manager.environment = environment
        self.manager.fingerprint = "expected"
        marker = environment / "lightbender-runtime.json"
        for contents in ("broken json", "[]", "null", '{"fingerprint":"different"}'):
            with self.subTest(contents=contents):
                marker.write_text(contents)
                self.assertFalse(self.manager.poll())

    def test_invalid_new_configuration_does_not_hide_failure_behind_old_ready_state(self):
        self.mark_ready()
        self.assertTrue(self.manager.start(allow_install=False))
        with mock.patch.object(runtime, "find_python", side_effect=RuntimeError("missing executable")):
            self.assertFalse(self.manager.start(override="/missing"))
        self.assertFalse(self.manager.poll())
        self.assertIn("missing executable", self.manager.message)

    def test_background_setup_is_single_and_failure_is_reported(self):
        child = mock.Mock()
        child.poll.return_value = None
        with mock.patch.object(runtime.subprocess, "Popen", return_value=child) as launch:
            self.assertFalse(self.manager.start())
            self.assertFalse(self.manager.start())
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertEqual(command[:3], [str(self.interpreter), "-I",
                                          str(self.root / "bootstrap_runtime.py")])
            self.assertIn(str(self.root / "requirements-addon.txt"), command)
            self.assertNotIn("shell", launch.call_args.kwargs)
            child.poll.return_value = 1
            self.assertFalse(self.manager.poll())
        self.assertIsNone(self.manager.log_handle)
        self.assertIn("Setup failed", self.manager.message)
        with self.assertRaisesRegex(RuntimeError, "Preferences"):
            self.manager.python()

    def test_helper_paths_and_arguments_are_passed_without_a_shell(self):
        executable = r"C:\Users\Example User\tools\Scripts\python.exe"
        arguments = [self.root / "shape's graph.yaml", "--threshold", "0.16"]
        with mock.patch.object(self.manager, "python", return_value=executable), \
                mock.patch.object(runtime.subprocess, "run") as run:
            self.manager.run(arguments, check=True)
        self.assertEqual(run.call_args.args[0], [executable, *map(str, arguments)])
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")

    def test_helper_actually_runs_in_private_venv_without_blender_python_paths(self):
        _, _, fingerprint, directory = self.manager._configuration("", False)
        venv.EnvBuilder(with_pip=False).create(str(directory))
        self.mark_ready(fingerprint, directory)
        self.assertTrue(self.manager.start(allow_install=False))
        with mock.patch.dict(os.environ, {
            "PYTHONHOME": "/blender/python", "PYTHONPATH": "/blender/packages",
            "VIRTUAL_ENV": "/blender/venv",
        }):
            result = self.manager.run([
                "-c", "import json,sys; print(json.dumps({"
                "'prefix':sys.prefix, 'base':sys.base_prefix, 'path':sys.path}))",
            ], check=True)
        child = json.loads(result.stdout)
        self.assertEqual(Path(child["prefix"]).resolve(), directory.resolve())
        self.assertNotEqual(child["prefix"], child["base"])
        self.assertNotIn("/blender/packages", child["path"])
        self.assertFalse(any("/blender/" in path for path in child["path"]))

    def test_a_successful_exit_without_a_ready_marker_is_a_failure(self):
        child = mock.Mock()
        child.poll.return_value = None
        with mock.patch.object(runtime.subprocess, "Popen", return_value=child):
            self.manager.start()
            child.poll.return_value = 0
            self.assertFalse(self.manager.poll())
        self.assertIn("Setup failed", self.manager.message)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="LightBender install '")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = self.root / "venv with spaces"
        self.marker = self.environment / "lightbender-runtime.json"
        self.lock = self.environment.with_suffix(".lock")
        self.requirement = self.root / "core requirements.txt"
        self.requirement.write_text("numpy>=1.26.4,<3\n")
        self.addCleanup(mock.patch.stopall)
        self.builder = mock.patch.object(bootstrap.venv, "EnvBuilder").start()
        self.builder.return_value.create.side_effect = self.create_environment
        self.run = mock.patch.object(bootstrap.subprocess, "run").start()

    def create_environment(self, path):
        environment = Path(path)
        environment.mkdir(parents=True, exist_ok=True)
        python = runtime.venv_python(environment)
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()

    def install(self):
        bootstrap.install(self.environment, [self.requirement], "test-fingerprint")

    def test_publishes_success_only_after_dependencies_are_importable(self):
        def check_incomplete(command, **kwargs):
            self.assertFalse(self.marker.exists())
            self.assertTrue(self.lock.exists())
            return subprocess.CompletedProcess(command, 0)
        self.run.side_effect = check_incomplete
        self.install()
        self.assertEqual(json.loads(self.marker.read_text())["fingerprint"], "test-fingerprint")
        self.assertFalse(self.lock.exists())
        self.assertEqual(self.run.call_count, 2)
        pip = self.run.call_args_list[0].args[0]
        self.assertEqual(pip[-2:], ["-r", str(self.requirement)])
        self.assertIn("--only-binary=:all:", pip)
        self.assertIn("--isolated", pip)
        self.assertIn("--require-virtualenv", pip)
        self.assertIn("--no-input", pip)
        self.assertEqual(self.run.call_args_list[1].args[0][1:3], ["-I", "-c"])

    def test_pip_or_import_failure_does_not_leave_a_success_marker_or_lock(self):
        for failure_at in (0, 1):
            with self.subTest(failure_at=failure_at):
                self.run.side_effect = ([subprocess.CalledProcessError(1, "pip")]
                                        if failure_at == 0 else
                                        [None, subprocess.CalledProcessError(1, "import")])
                with self.assertRaises(subprocess.CalledProcessError):
                    self.install()
                self.assertFalse(self.marker.exists())
                self.assertFalse(self.lock.exists())
        self.run.side_effect = None
        self.install()
        self.assertTrue(self.marker.exists())

    def test_a_ready_installation_is_idempotent(self):
        self.install()
        self.builder.reset_mock()
        self.run.reset_mock()
        self.install()
        self.builder.assert_not_called()
        self.run.assert_not_called()
        self.assertFalse(self.lock.exists())

    def test_damaged_markers_are_repaired_instead_of_failing_every_retry(self):
        for contents in ("broken json", "[]", "null"):
            with self.subTest(contents=contents):
                self.environment.mkdir(exist_ok=True)
                self.marker.write_text(contents)
                self.install()
                self.assertEqual(json.loads(self.marker.read_text())["fingerprint"], "test-fingerprint")

    def test_a_matching_marker_with_a_missing_python_is_repaired(self):
        self.environment.mkdir()
        self.marker.write_text(json.dumps({"fingerprint": "test-fingerprint"}))
        self.install()
        self.assertTrue(runtime.venv_python(self.environment).is_file())
        self.builder.assert_called_once()

    def test_concurrent_setup_timeout_does_not_remove_another_process_lock(self):
        self.lock.write_text("another process")
        with mock.patch.object(bootstrap.time, "monotonic", side_effect=[0, 1201]), \
                mock.patch.object(bootstrap.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "locked by another process"):
                self.install()
        sleep.assert_not_called()
        self.assertEqual(self.lock.read_text(), "another process")
        self.builder.assert_not_called()
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
