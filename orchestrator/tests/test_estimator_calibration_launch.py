"""No-hardware tests of the interactive orchestrator IMU calibration branch."""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from estimator_calibration_launch import (  # noqa: E402
    ARTIFACTS, calibration_command, run_estimator_calibration, select_drone,
    validate_estimator_calibration_options,
)


def main_ast():
    tree = ast.parse((ROOT / 'orchestrator.py').read_text())
    return next(node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare))


def cli_parser():
    statements = [node for node in main_ast().body if (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'parser'
                for target in node.targets)
    ) or (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'parser'
        and node.value.func.attr == 'add_argument'
    )]
    namespace = {'argparse': argparse}
    exec(compile(ast.Module(body=statements, type_ignores=[]), 'orchestrator.py', 'exec'), namespace)
    return namespace['parser']


def options(*extra):
    return cli_parser().parse_args(['--calibrate-estimator-imu', '--drone-id', 'lb11',
        '--imu-firmware-id', 'flashed-build-1', '--imu-fixture-id', 'fixture-1',
        '--imu-reference-note', 'independent fixture', *extra])


def manifest():
    return {'common': {'work_dir': '/home/fls/controller repo', 'venv_path': '/home/fls/env'},
            'controller': {'mission_path': 'SFL', 'mission_file': 'one.yaml'},
            'drones': [{'id': 'lb11', 'ip': '192.0.2.11', 'user': 'fls',
                        'uri': 'radio://0/100/2M/E7E7E7E706'},
                       {'id': 'lb2', 'ip': '192.0.2.2', 'user': 'fls'}]}


class FakeConnection:
    def __init__(self, *, code=0, files=None, error=None):
        self.code, self.error = code, error
        self.files = files if files is not None else {name: '{}' for name in ARTIFACTS}
        if files is None:
            self.files['fit/calibration.json'] = json.dumps({
                'schema': 'estimator_processed_imu_calibration_v1',
                'accepted': True, 'drone_id': 'lb11', 'firmware_id': 'flashed-build-1',
                'dataset_sha256': hashlib.sha256(self.files['dataset.json'].encode()).hexdigest()})
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        kwargs['out_stream'].write('Remote pose prompt\n')
        if self.error:
            raise self.error
        return SimpleNamespace(exited=self.code)

    def get(self, remote, *, local):
        relative = next(name for name in ARTIFACTS if remote.endswith('/' + name))
        if relative not in self.files:
            raise FileNotFoundError(relative)
        Path(local).write_text(self.files[relative])


class LaunchTests(unittest.TestCase):
    def test_single_flag_registered_and_old_calibration_stays_distinct(self):
        args = cli_parser().parse_args(['--calibrate-estimator-imu'])
        validate_estimator_calibration_options(args)
        self.assertFalse(getattr(args, 'calibrate', False))
        self.assertFalse(args.interaction)
        self.assertIsNone(args.imu_duration_s)
        defaults = cli_parser().parse_args([])
        validate_estimator_calibration_options(defaults)
        self.assertFalse(defaults.calibrate_estimator_imu)
        if hasattr(defaults, 'calibrate'):
            old = cli_parser().parse_args(['--calibrate'])
            validate_estimator_calibration_options(old)
            self.assertFalse(old.calibrate_estimator_imu)

    def test_conflicts_fail_before_any_remote_or_flight_setup(self):
        for field in ('calibrate', 'interaction', 'mpc', 'hover', 'baseline',
                      'braking_test', 'sense', 'droneless', 'ground', 'record',
                      'skip_confirm', 'active_septic_brake_calibration',
                      'vicon_source_metadata', 'test_marker_grid', 'kill', 'off'):
            args = options()
            setattr(args, field, [] if field in ('kill', 'off') else True)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_estimator_calibration_options(args)
        for tokens in (['--imu-duration-s', 'nan'], ['--imu-duration-s', '2'],
                       ['--imu-settle-s', '0'], ['--drone-id', '../bad']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_estimator_calibration_options(options(*tokens))

    def test_imu_auxiliary_flags_require_imu_mode(self):
        with self.assertRaises(ValueError):
            validate_estimator_calibration_options(cli_parser().parse_args(['--imu-fixture-id', 'x']))

    def test_real_entry_branches_before_constructing_swarm_orchestrator(self):
        run = Mock(return_value=0)
        swarm = Mock(side_effect=AssertionError('must not construct flight orchestrator'))
        namespace = {'argparse': argparse, 'sys': sys,
            'validate_estimator_calibration_options': validate_estimator_calibration_options,
            'validate_interaction_modes': Mock(),
            'run_estimator_calibration': run, 'SwarmOrchestrator': swarm,
            'MANIFEST_FILE': Path('swarm_manifest.yaml'), 'BASE_DIR': ROOT}
        with patch('sys.argv', ['orchestrator.py', '--calibrate-estimator-imu']), \
                self.assertRaises(SystemExit) as stopped:
            exec(compile(ast.Module(body=main_ast().body, type_ignores=[]), 'orchestrator.py', 'exec'), namespace)
        self.assertEqual(stopped.exception.code, 0)
        run.assert_called_once()
        swarm.assert_not_called()

    def test_select_explicit_aircraft_without_loading_any_mission(self):
        selected = select_drone(manifest(), options(), '/nonexistent')
        self.assertEqual(selected['id'], 'lb11')

    def test_single_mission_identity_is_selected_by_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'SFL').mkdir()
            (root / 'SFL/one.yaml').write_text('drones:\n  lb11: {}\n')
            args = options()
            args.drone_id = None
            self.assertEqual(select_drone(manifest(), args, root)['id'], 'lb11')
            (root / 'SFL/one.yaml').write_text('drones:\n  lb11: {}\n  lb2: {}\n')
            with self.assertRaisesRegex(ValueError, 'one aircraft'):
                select_drone(manifest(), args, root)

    def test_usb_radio_and_literal_shell_quoting(self):
        config = manifest()
        drone = config['drones'][0]
        provenance = {'firmware_id': 'build $(touch /tmp/no)', 'fixture_id': "fixture ' one",
                      'reference_note': 'checked; literal $HOME'}
        command, remote = calibration_command(config, drone, options(), 'session_1', provenance)
        tokens = shlex.split(command)
        self.assertEqual(tokens[1], config['common']['work_dir'])
        self.assertEqual(tokens[tokens.index('--uri') + 1], 'usb://0')
        self.assertEqual(tokens[tokens.index('--firmware-id') + 1], provenance['firmware_id'])
        self.assertEqual(tokens[tokens.index('--reference-note') + 1], provenance['reference_note'])
        self.assertEqual(remote.name, 'session_1')
        for forbidden in ('controller.py', 'nohup', '--orchestrated', '--vicon', '--sense', '--interaction'):
            self.assertNotIn(forbidden, tokens)
        radio, _ = calibration_command(config, drone, options('--radio'), 's1', provenance)
        self.assertIn(drone['uri'], shlex.split(radio))

    def launch(self, tmp, connection, *, args=None, prompt=None):
        path = Path(tmp) / 'manifest.yaml'
        path.write_text(yaml.safe_dump(manifest()))
        factory = Mock(return_value=connection)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = run_estimator_calibration(args or options(), path, tmp,
                connection_factory=factory, session='test_session', prompt=prompt)
        directory = Path(tmp) / 'logs/estimator_imu_lb11_test_session'
        return code, directory, factory

    def test_interactive_pty_and_all_artifacts_are_retrieved(self):
        with TemporaryDirectory() as tmp:
            connection = FakeConnection()
            code, directory, factory = self.launch(tmp, connection)
            self.assertEqual(code, 0)
            factory.assert_called_once_with(host='192.0.2.11', user='fls', connect_timeout=5)
            self.assertTrue(connection.calls[0][1]['pty'])
            self.assertIs(connection.calls[0][1]['in_stream'], sys.stdin)
            self.assertEqual((directory / 'terminal.txt').read_text(), 'Remote pose prompt\n')
            self.assertTrue(all((directory / name).exists() for name in ARTIFACTS))
            result = json.loads((directory / 'launch_result.json').read_text())
            self.assertFalse(result['flight_started'])
            self.assertFalse(result['firmware_applied'])
            self.assertFalse(result['missing'])

    def test_failed_or_interrupted_session_still_retrieves_available_raw_data(self):
        for error, expected in ((None, 1), (KeyboardInterrupt(), 130)):
            with self.subTest(error=error), TemporaryDirectory() as tmp:
                conn = FakeConnection(code=1, files={'dataset.json': '{"status":"incomplete"}',
                                                    'packets.jsonl': ''}, error=error)
                code, directory, _ = self.launch(tmp, conn)
                self.assertEqual(code, expected)
                self.assertTrue((directory / 'dataset.json').exists())
                self.assertFalse((directory / 'fit/calibration.json').exists())
                result = json.loads((directory / 'launch_result.json').read_text())
                self.assertIn('fit/calibration.json', result['missing'])

    def test_success_without_downloaded_accepted_calibration_is_not_reported_as_pass(self):
        for files in ({'dataset.json': '{}'},
                      {name: '{"accepted":false}' for name in ARTIFACTS}):
            with self.subTest(files=files), TemporaryDirectory() as tmp:
                code, _, _ = self.launch(tmp, FakeConnection(files=files))
                self.assertEqual(code, 1)

    def test_missing_provenance_is_prompted_before_connection(self):
        with TemporaryDirectory() as tmp:
            args = options()
            args.imu_firmware_id = args.imu_fixture_id = args.imu_reference_note = None
            prompt = Mock(side_effect=['real-build', 'checked-fixture', 'fixture checked independently'])
            connection = FakeConnection()
            fit = json.loads(connection.files['fit/calibration.json'])
            fit['firmware_id'] = 'real-build'
            connection.files['fit/calibration.json'] = json.dumps(fit)
            code, _, _ = self.launch(tmp, connection, args=args, prompt=prompt)
            self.assertEqual(code, 0)
            self.assertEqual(prompt.call_count, 3)

    def test_mismatched_downloaded_dataset_is_not_accepted(self):
        with TemporaryDirectory() as tmp:
            connection = FakeConnection()
            connection.files['dataset.json'] = '{"different":"dataset"}'
            code, directory, _ = self.launch(tmp, connection)
            self.assertEqual(code, 1)
            result = json.loads((directory / 'launch_result.json').read_text())
            self.assertIn('fingerprint', result['error'])


if __name__ == '__main__':
    unittest.main()
