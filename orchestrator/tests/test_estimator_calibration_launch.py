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
    ARTIFACTS, FLIGHT_ARTIFACTS, calibration_command, run_estimator_calibration, select_drone,
    validate_estimator_calibration_options, run_offline_replay, latest_imu_calibration,
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
                        'uri': 'radio://0/100/2M/E7E7E7E706', 'init_pos':[0,-1,.24]},
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
        if 'out_stream' in kwargs:
            kwargs['out_stream'].write('Remote pose prompt\n')
        if self.error:
            raise self.error
        return SimpleNamespace(exited=self.code)

    def get(self, remote, *, local):
        relative = next(name for name in set(ARTIFACTS + FLIGHT_ARTIFACTS) if remote.endswith('/' + name))
        if relative not in self.files:
            raise FileNotFoundError(relative)
        Path(local).write_text(self.files[relative])


class LaunchTests(unittest.TestCase):
    def test_cage_gravity_method_is_default_and_exact_fixture_remains_selectable(self):
        for selection, expected in (([], 'gravity_norm'),
                (['--imu-calibration-method','six_face'], 'six_face')):
            args=options(*selection)
            command,_=calibration_command(manifest(),manifest()['drones'][0],args,'test',
                {'firmware_id':'f','fixture_id':'f','reference_note':'f'})
            tokens=shlex.split(command)
            self.assertEqual(tokens[tokens.index('--calibration-method')+1],expected)

    def test_latest_imu_ignores_failed_other_aircraft_and_flight_copies(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stamp, valid, drone, copy in [('001',True,'lb11',False),
                    ('002',False,'lb11',False),('003',True,'other',False),('004',True,'lb11',True)]:
                source = root/'logs'/('estimator_imu_lb11_'+stamp)
                (source/'fit').mkdir(parents=True)
                (source/'dataset.json').write_text('{}')
                (source/'fit/candidate.json').write_text(json.dumps({'fit_passed':valid,'drone_id':drone,
                    'dataset_sha256':hashlib.sha256(b'{}').hexdigest()}))
                if copy:
                    (source/'launch_result.json').write_text(json.dumps({'source_calibration':'original'}))
            self.assertEqual(latest_imu_calibration(root,'lb11').name,'estimator_imu_lb11_001')
            args = cli_parser().parse_args(['--validate-estimator-imu','--drone-id','lb11'])
            self.assertEqual(args.validate_estimator_imu,'latest')
            with self.assertRaises(ValueError):
                latest_imu_calibration(root,'missing')

    def test_saved_session_uploads_original_data_and_downloads_new_flight(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'saved'
            (source/'fit').mkdir(parents=True)
            files = {name:'{}' for name in FLIGHT_ARTIFACTS}
            files['fit/candidate.json'] = json.dumps({'schema':'estimator_processed_imu_calibration_v1',
                'fit_passed':True,'drone_id':'lb11','firmware_id':'actual-saved-build',
                'dataset_sha256':hashlib.sha256(files['dataset.json'].encode()).hexdigest()})
            files['flight/report.json'] = json.dumps({'capture_completed':True,'flight_started':True,
                'candidate_applied':False,'drone_id':'lb11','firmware_id':'actual-saved-build',
                'candidate_sha256':hashlib.sha256(files['fit/candidate.json'].encode()).hexdigest(),
                'packets_sha256':hashlib.sha256(files['flight/packets.jsonl'].encode()).hexdigest()})
            for name in ('dataset.json','packets.jsonl','fit/report.json','fit/report.md','fit/candidate.json'):
                (source/name).write_text(files[name])
            manifest_path = root/'manifest.yaml'
            manifest_path.write_text(yaml.safe_dump(manifest()))
            connection = FakeConnection(files=files)
            uploaded = {}
            connection.put = lambda stream, remote: uploaded.update({remote:stream.read()})
            replay = Mock()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = run_estimator_calibration(options('--imu-auto-flight','--validate-estimator-imu',str(source)),
                    manifest_path,root,connection_factory=lambda **kw:connection,session='retry',replay_runner=replay)
            self.assertEqual(code,0)
            self.assertEqual(len(uploaded),5)
            self.assertTrue(connection.calls[0][0].startswith('mkdir -p '))
            self.assertIn('controller.py',connection.calls[1][0])
            self.assertIn('--imu-validation-session',connection.calls[1][0])
            self.assertNotIn('Interaction.calibrate_estimator_imu',connection.calls[1][0])
            self.assertEqual((source/'flight').exists(),False)
            self.assertEqual((root/'logs/estimator_imu_lb11_retry/dataset.json').read_bytes(),(source/'dataset.json').read_bytes())
            replay.assert_called_once()

    def test_saved_session_command_only_runs_flight(self):
        args = options('--imu-auto-flight', '--validate-estimator-imu', '/saved/session')
        command, directory = calibration_command(manifest(), manifest()['drones'][0], args,
            'retry_1', {'firmware_id': 'f', 'fixture_id': 'f', 'reference_note': 'f'})
        self.assertIn('controller.py', command)
        self.assertIn('--calibrate', command)
        self.assertNotIn('Interaction.calibrate_estimator_imu', command)
        self.assertNotIn('Interaction.estimator_validation_flight', command)
        self.assertIn('--imu-validation-session', command)
        self.assertIn('--init-pos', command)
        self.assertNotIn('--gyro-calibration', command)

    def test_saved_wrong_aircraft_rejected_before_connection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'saved'
            (source/'fit').mkdir(parents=True)
            for name in ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md'):
                (source/name).write_text('{}')
            (source/'fit/candidate.json').write_text(json.dumps({'fit_passed': True,
                'drone_id': 'other', 'dataset_sha256': hashlib.sha256(b'{}').hexdigest()}))
            manifest_path = root/'manifest.yaml'
            manifest_path.write_text(yaml.safe_dump(manifest()))
            factory = Mock()
            with self.assertRaisesRegex(ValueError, 'matching static IMU candidate'):
                run_estimator_calibration(options('--imu-auto-flight', '--validate-estimator-imu', str(source)),
                    manifest_path, root, connection_factory=factory)
            factory.assert_not_called()

    def test_auto_flight_needs_no_yaml_and_override_requires_auto_flag(self):
        validate_estimator_calibration_options(options('--imu-auto-flight'))
        for tokens in (['--imu-flight-config', 'flight.yaml'],):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_estimator_calibration_options(options(*tokens))
        args = options('--imu-auto-flight', '--imu-flight-config', 'flight.yaml')
        validate_estimator_calibration_options(args)
        command, directory = calibration_command(manifest(), manifest()['drones'][0], args, 'session',
            {'firmware_id': 'f', 'fixture_id': 'fixture', 'reference_note': 'checked'})
        self.assertIn('--auto-flight', shlex.split(command))
        self.assertIn('/tmp/fls_estimator_flight_session.yaml', shlex.split(command))
        self.assertNotIn('controller.py', command)
        self.assertFalse(args.interaction)
        gyro_args = options('--imu-calibrate-gyro')
        validate_estimator_calibration_options(gyro_args)
        command, _ = calibration_command(manifest(), manifest()['drones'][0], gyro_args, 'session',
            {'firmware_id': 'f', 'fixture_id': 'fixture', 'reference_note': 'checked'})
        self.assertIn('--gyro-calibration', shlex.split(command))

    def test_builtin_command_uses_manifest_position_and_no_config_file(self):
        args=options('--imu-auto-flight')
        command,_=calibration_command(manifest(),manifest()['drones'][0],args,'session',
            {'firmware_id':'f','fixture_id':'fixture','reference_note':'checked'})
        tokens=shlex.split(command)
        self.assertNotIn('--flight-config',tokens)
        i=tokens.index('--flight-initial-position')
        self.assertEqual(list(map(float,tokens[i+1:i+4])),[0,-1,.24])
        bad=manifest()['drones'][0].copy();bad.pop('init_pos')
        with self.assertRaises(ValueError):
            calibration_command(manifest(),bad,args,'session',{'firmware_id':'f','fixture_id':'fixture','reference_note':'checked'})

    def test_builtin_launch_has_no_flight_file_upload_or_mission_read(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);manifest_path=root/'manifest.yaml'
            manifest_path.write_text(yaml.safe_dump(manifest()))
            files={name:'{}' for name in FLIGHT_ARTIFACTS}
            files['fit/candidate.json']=json.dumps({'schema':'estimator_processed_imu_calibration_v1',
                'fit_passed':True,'accepted':False,'drone_id':'lb11','firmware_id':'flashed-build-1',
                'dataset_sha256':hashlib.sha256(files['dataset.json'].encode()).hexdigest()})
            files['flight/report.json']=json.dumps({'capture_completed':True,'flight_started':True,'candidate_applied':False,
                'drone_id':'lb11','firmware_id':'flashed-build-1',
                'candidate_sha256':hashlib.sha256(files['fit/candidate.json'].encode()).hexdigest(),
                'packets_sha256':hashlib.sha256(files['flight/packets.jsonl'].encode()).hexdigest()})
            connection=FakeConnection(files=files);connection.put=Mock();connection.sftp=Mock()
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=run_estimator_calibration(options('--imu-auto-flight'),manifest_path,root,
                    connection_factory=lambda **kw:connection,session='builtin_test',replay_runner=Mock())
            self.assertEqual(code,0)
            connection.put.assert_not_called();connection.sftp.assert_not_called()
            self.assertFalse((root/'SFL/one.yaml').exists())
            self.assertNotIn('--flight-config',connection.calls[0][0])

    def test_auto_flight_uploads_config_and_downloads_hash_matched_capture(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            flight_config = root/'flight.yaml'
            flight_config.write_text(yaml.safe_dump({'rigidbody': 'lb11'}))
            manifest_path = root/'manifest.yaml'
            manifest_path.write_text(yaml.safe_dump(manifest()))
            args = options('--imu-auto-flight', '--imu-flight-config', str(flight_config))
            files = {name: '{}' for name in FLIGHT_ARTIFACTS}
            files['fit/candidate.json'] = json.dumps({'schema': 'estimator_processed_imu_calibration_v1',
                'fit_passed': True, 'accepted': False, 'drone_id': 'lb11', 'firmware_id': 'flashed-build-1',
                'dataset_sha256': hashlib.sha256(files['dataset.json'].encode()).hexdigest()})
            files['flight/report.json'] = json.dumps({'capture_completed': True, 'flight_started': True,
                'candidate_applied': False, 'drone_id': 'lb11', 'firmware_id': 'flashed-build-1',
                'candidate_sha256': hashlib.sha256(files['fit/candidate.json'].encode()).hexdigest(),
                'packets_sha256': hashlib.sha256(files['flight/packets.jsonl'].encode()).hexdigest()})
            connection = FakeConnection(files=files)
            connection.put = Mock()
            connection.sftp = Mock()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = run_estimator_calibration(args, manifest_path, root,
                    connection_factory=lambda **kw: connection, session='flight_test', replay_runner=Mock())
            self.assertEqual(code, 0)
            connection.put.assert_called_once()
            connection.sftp().remove.assert_called_once_with('/tmp/fls_estimator_flight_flight_test.yaml')
            local = root/'logs/estimator_imu_lb11_flight_test'
            result = json.loads((local/'launch_result.json').read_text())
            self.assertTrue(result['flight_started'])
            self.assertTrue(result['flight_capture_completed'])
            self.assertFalse(result['firmware_applied'])
            self.assertFalse((local/'fit/calibration.json').exists())

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

    def test_missing_provenance_defaults_to_unknown_without_prompt(self):
        with TemporaryDirectory() as tmp:
            args = options()
            args.imu_firmware_id = args.imu_fixture_id = args.imu_reference_note = None
            prompt = Mock(side_effect=AssertionError('metadata must not prompt'))
            connection = FakeConnection()
            fit = json.loads(connection.files['fit/calibration.json'])
            fit['firmware_id'] = 'unknown'
            connection.files['fit/calibration.json'] = json.dumps(fit)
            code, directory, _ = self.launch(tmp, connection, args=args, prompt=prompt)
            self.assertEqual(code, 0)
            prompt.assert_not_called()
            result=json.loads((directory/'launch_result.json').read_text())
            self.assertEqual(result['provenance'],dict.fromkeys(('firmware_id','fixture_id','reference_note'),'unknown'))

    def test_mismatched_downloaded_dataset_is_not_accepted(self):
        with TemporaryDirectory() as tmp:
            connection = FakeConnection()
            connection.files['dataset.json'] = '{"different":"dataset"}'
            code, directory, _ = self.launch(tmp, connection)
            self.assertEqual(code, 1)
            result = json.loads((directory / 'launch_result.json').read_text())
            self.assertIn('fingerprint', result['error'])


class LocalRamReplayTests(unittest.TestCase):
    def test_ram_binary_is_locally_decoded_then_replayed_without_corrupting_cli(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp)/'lightbender/orchestrator'
            base.mkdir(parents=True)
            repo=Path(tmp)/'fls-cf-offboard-controller/Interaction'
            repo.mkdir(parents=True)
            (repo/'replay_estimator3.py').write_text('')
            session=base/'logs/test'
            (session/'flight/ram_trace').mkdir(parents=True)
            (session/'flight/ram_trace/packets.jsonl').write_text('')
            ok=SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch('estimator_calibration_launch.subprocess.run',return_value=ok) as run:
                run_offline_replay(session,base)
            commands=[call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands),3)
            self.assertIn('Interaction.estimator_ram_trace',commands[1])
            self.assertIn('decode',commands[1])
            self.assertTrue(commands[2][4].endswith('/ram_trace/decoded/packets.jsonl'))
            self.assertEqual(commands[2][5],'--candidate')
            self.assertTrue(commands[2][-1].endswith('/ram_trace/replay'))


if __name__ == '__main__':
    unittest.main()
