"""Exercise startup shell lifetime without opening hardware or network links."""

import ast
import io
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import yaml


def boot_method(connection):
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SwarmOrchestrator')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_boot_remote_node')
    namespace = {'Connection': Mock(return_value=connection), 'io': io,
                 'yaml': yaml, 'shlex': shlex, 'CommandTimedOut': TimeoutError}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace['_boot_remote_node']


def close_ssh_output(command, **kwargs):
    # SSH can finish after the outer shell exits while its background setup
    # is still running. Closing its output must not kill the remaining chain.
    parent = subprocess.Popen(['bash', '-c', command], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        parent.wait(timeout=2)
    finally:
        parent.stdout.close()
        parent.stderr.close()
    return SimpleNamespace(exited=parent.returncode)


class RemoteDroneStartupTests(unittest.TestCase):
    def boot(self, work, command, node_type='Drone', node_id='lb11'):
        connection = SimpleNamespace(put=Mock(), run=Mock(side_effect=close_ssh_output))
        owner = SimpleNamespace(logger=Mock(), manifest={},
                                common_cfg={'work_dir': str(work)},
                                _prepare_drone_localizer=Mock())
        result = boot_method(connection)(owner, {'id': node_id, 'ip': 'unused',
                                                'user': 'unused'}, command, node_type)
        self.assertTrue(result)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.02)
        return False

    def test_delayed_setup_survives_closed_ssh_output(self):
        with tempfile.TemporaryDirectory(prefix='drone startup ') as directory:
            work = Path(directory)
            marker = work / 'controller.started'
            command = ('sleep .15 && printf "setup finished\\n" && '
                       f'printf started > {shlex.quote(str(marker))} &')
            self.boot(work, command)
            self.assertTrue(self.wait_for(marker.exists))
            self.assertEqual(marker.read_text(), 'started')
            self.assertIn('setup finished', (work / 'drone_lb11_boot.log').read_text())

    def test_failed_setup_still_blocks_controller_and_records_error(self):
        with tempfile.TemporaryDirectory(prefix='drone startup ') as directory:
            work = Path(directory)
            marker = work / 'controller.started'
            failed_setup = shlex.quote('printf "setup failed\\n" >&2; exit 42')
            command = (f'sleep .15 && sh -c {failed_setup} && '
                       f'printf started > {shlex.quote(str(marker))} &')
            self.boot(work, command)
            log = work / 'drone_lb11_boot.log'
            self.assertTrue(self.wait_for(lambda: log.exists() and
                                         'setup failed' in log.read_text()))
            self.assertFalse(marker.exists())

    def test_camera_nested_shell_survives_closed_ssh_output(self):
        with tempfile.TemporaryDirectory(prefix='camera startup ') as directory:
            work = Path(directory)
            marker = work / 'camera.started'
            payload = ('sleep .15 && printf "camera setup finished\\n" && '
                       f'printf started > {shlex.quote(str(marker))} &')
            self.boot(work, f'bash -c {shlex.quote(payload)}', 'Camera', 'CAM')
            self.assertTrue(self.wait_for(marker.exists))
            self.assertEqual(marker.read_text(), 'started')
            self.assertIn('camera setup finished',
                          (work / 'camera_CAM_boot.log').read_text())

    def test_second_camera_setup_failure_records_its_own_error(self):
        with tempfile.TemporaryDirectory(prefix='camera startup ') as directory:
            work = Path(directory)
            marker = work / 'camera.started'
            failed_setup = shlex.quote('printf "camera setup failed\\n" >&2; exit 42')
            payload = (f'sleep .15 && sh -c {failed_setup} && '
                       f'printf started > {shlex.quote(str(marker))} &')
            self.boot(work, f'bash -c {shlex.quote(payload)}', 'Camera', 'CAM_2')
            log = work / 'camera_CAM_2_boot.log'
            self.assertTrue(self.wait_for(lambda: log.exists() and
                                         'camera setup failed' in log.read_text()))
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
