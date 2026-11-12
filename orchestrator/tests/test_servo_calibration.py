"""Closed-loop calibration, bounded failures, and launch ordering."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import servo_calibration as calibration
from servo_calibration import DEFAULTS, RodAdjustment, ServoCalibrationMixin, configure, participants


def configuration():
    manifest = dict(mission={'require_handshake': True},
        servo_calibration={'enabled': True, 'calibration_file': '/camera/rgb.json'},
        drones=[dict(id='lb1', type='V', servo_offsets=[0., 0.])])
    missions = [dict(name='calibration-test', camera=[2., 0., 1.], drones={'lb1': dict(target=[0., 0., 1., 0., 0.],
                      servos=[[180., 360.], [180., 360.]], waypoints=[[0., 0., 1., 0., 0.], [0., 0., 1., 0., 5.]])})]
    cameras = [dict(id='CAM', run_id='camera-run')]
    return manifest, missions, cameras


class ConfigurationTests(unittest.TestCase):
    def test_disabled_preserves_existing_missions_and_no_camera_requirement(self):
        self.assertFalse(configure({}, SimpleNamespace(), [], [])['enabled'])
        self.assertFalse(configure({'servo_calibration': {'enabled': True}},
                                   SimpleNamespace(kill=[]), [], [])['enabled'])

    def test_enable_cli_and_validate_before_launch(self):
        manifest, missions, cameras = configuration()
        manifest['servo_calibration']['enabled'] = False
        config = configure(manifest, SimpleNamespace(illumination=True, calibrate_servos=True), cameras, missions)
        self.assertTrue(config['enabled'])
        for field, value in [('timeout_s', 0), ('gain', 2), ('max_iterations', 2.5),
                             ('max_step_deg', float('nan')), ('failure_policy', 'ignore')]:
            manifest, missions, cameras = configuration()
            manifest['servo_calibration'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                configure(manifest, SimpleNamespace(illumination=True), cameras, missions)

    def test_missing_camera_no_handshake_morphing_and_missing_samples(self):
        for change in ('camera', 'handshake', 'morphing', 'samples'):
            manifest, missions, cameras = configuration()
            args = SimpleNamespace(illumination=True)
            if change == 'camera':
                cameras = []
            elif change == 'handshake':
                manifest['mission']['require_handshake'] = False
            elif change == 'morphing':
                args.morphing = True
            else:
                missions[0]['drones']['lb1']['servos'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                configure(manifest, args, cameras, missions)

    def test_offset_feasibility_covers_entire_sfl_batch(self):
        manifest, missions, _ = configuration()
        missions.append(copy.deepcopy(missions[0]))
        missions[1]['drones']['lb1']['servos'] = [[90., 450.], [270., 270.]]
        entry = participants(manifest, missions, DEFAULTS)[0]
        self.assertEqual(entry['limits'], [(0., 0.), (0., 0.)])


class SimulatedCalibration(ServoCalibrationMixin):
    def __init__(self, bias=(-8., 5.), stalled=False, invisible=False, policy='abort'):
        self.manifest, self.missions, self.cameras = configuration()
        self.servo_calibration_cfg = dict(DEFAULTS, enabled=True, max_iterations=8,
            tolerance_deg=.7, settle_s=.001, failure_policy=policy)
        self.running = threading.Event()
        self.running.set()
        self.emergency = False
        self.logger = Mock()
        self.tag = 'test'
        self.offsets = [0., 0.]
        self.bias, self.stalled, self.invisible = bias, stalled, invisible
        self.commands = []
        self.drop_measurement = False
        self.fail_cleanup = False

    def _active_cameras(self):
        return self.cameras

    def _camera_id(self, camera):
        return camera['id']

    def _servo_exchange(self, command, expected_id, status, deadline):
        self.commands.append(copy.deepcopy(command))
        if self.fail_cleanup and command.get('finish'):
            raise TimeoutError('cleanup ACK missing')
        if command['cmd'] == 'SERVO_CALIBRATE':
            self.offsets = list(command['offsets'])
            return dict(offsets=self.offsets)
        if command['cmd'] == 'MEASURE_SERVOS':
            if self.drop_measurement:
                raise TimeoutError('camera timeout')
            return dict(angles=[None, None] if self.invisible else
                [target+bias+(0. if self.stalled and i == 0 else offset)
                 for i, (target, bias, offset) in enumerate(zip(command['reference']['angles'], self.bias, self.offsets))])
        return {}


class ClosedLoopTests(unittest.TestCase):
    def run_simulation(self, simulation, failure=False):
        with tempfile.TemporaryDirectory() as temp, patch.object(calibration, '__file__', str(Path(temp)/'servo_calibration.py')):
            if failure:
                with self.assertRaises(RuntimeError):
                    simulation._calibrate_illumination_servos()
            else:
                simulation._calibrate_illumination_servos()
            return json.loads((Path(temp)/'logs/test/servo_calibration.json').read_text())

    def test_correct_sign_closed_loop_converges_and_retains_offsets(self):
        simulation = SimulatedCalibration()
        report = self.run_simulation(simulation)
        self.assertEqual([r['status'] for r in report['drones']['lb1']['rods']], ['converged']*2)
        self.assertAlmostEqual(simulation.offsets[0], 8., delta=.7)
        self.assertAlmostEqual(simulation.offsets[1], -5., delta=.7)
        self.assertTrue(simulation.commands[-2]['finish'])
        self.assertEqual(simulation.commands[-1]['cmd'], 'STOP_SERVO_CAMERA')
        self.assertFalse(any(command['cmd'] == 'START' for command in simulation.commands))

    def test_stalled_servo_rolls_back_and_other_rod_can_converge(self):
        simulation = SimulatedCalibration(stalled=True, policy='continue')
        report = self.run_simulation(simulation)
        self.assertEqual(report['drones']['lb1']['rods'][0]['status'], 'stalled')
        self.assertEqual(simulation.offsets[0], 0.)
        self.assertAlmostEqual(simulation.offsets[1], -5., delta=.7)
        self.assertLessEqual(report['drones']['lb1']['iterations'], 8)

    def test_hardware_limit_exits_and_restores_offset(self):
        simulation = SimulatedCalibration(bias=(8., 0.), policy='continue')
        simulation.missions[0]['drones']['lb1']['servos'] = [[90., 270.]]*2
        report = self.run_simulation(simulation)
        self.assertEqual(report['drones']['lb1']['rods'][0]['status'], 'limit_reached')
        self.assertEqual(report['drones']['lb1']['iterations'], 2)
        self.assertEqual(simulation.offsets, [0., 0.])

    def test_invisible_servo_is_bounded_and_aborts_by_default(self):
        simulation = SimulatedCalibration(invisible=True)
        report = self.run_simulation(simulation, failure=True)
        self.assertEqual(report['drones']['lb1']['iterations'], 8)
        self.assertEqual([r['status'] for r in report['drones']['lb1']['rods']], ['not_detected']*2)
        self.assertEqual(simulation.offsets, [0., 0.])

    def test_timeout_restores_offsets_and_cleanup_failure_always_aborts(self):
        for cleanup_failure in (False, True):
            simulation = SimulatedCalibration(policy='continue')
            simulation.drop_measurement = True
            simulation.fail_cleanup = cleanup_failure
            report = self.run_simulation(simulation, failure=cleanup_failure)
            self.assertIn('error', report)
            self.assertIn('cleanup_error' if cleanup_failure else 'applied_offsets', report)
            self.assertEqual(simulation.offsets, [0., 0.])

    def test_converged_rod_refreshes_reading_and_reopens_after_drift(self):
        rod = RodAdjustment(180., 0., (-15., 15.), DEFAULTS)
        rod.observe(180.2)
        self.assertEqual(rod.status, 'pending')
        rod.observe(180.3)
        self.assertEqual(rod.status, 'converged')
        rod.observe(182.)
        self.assertEqual(rod.status, 'pending')
        self.assertEqual(rod.measured, 182.)
        self.assertLess(rod.offset, 0.)
        rod.observe(None)
        self.assertIsNone(rod.measured)
        self.assertEqual(rod.confirmations, 0)

    def test_final_sweep_recorrects_drift_and_reports_latest_measurement(self):
        simulation = SimulatedCalibration(bias=(0., 0.))
        original = simulation._servo_exchange
        measurements = 0
        def exchange(command, *args):
            nonlocal measurements
            if command['cmd'] == 'MEASURE_SERVOS':
                measurements += 1
                if measurements == 3:  # after initial two confirmations
                    simulation.bias = (2., -2.)
            return original(command, *args)
        simulation._servo_exchange = exchange
        report = self.run_simulation(simulation)
        result = report['drones']['lb1']
        self.assertEqual(result['verification_pass'], 1)
        self.assertLess(simulation.offsets[0], -1.)
        self.assertGreater(simulation.offsets[1], 1.)
        self.assertEqual([r['measured_deg'] for r in result['rods']],
                         result['observations'][-1]['measured_deg'])
        self.assertTrue(all(r['status'] == 'converged' for r in result['rods']))

    def test_earlier_drone_is_rechecked_after_other_drone_calibration(self):
        simulation = SimulatedCalibration(bias=(0., 0.))
        simulation.manifest['drones'].append(dict(id='lb2', type='V', servo_offsets=[0., 0.]))
        simulation.missions[0]['drones']['lb2'] = copy.deepcopy(simulation.missions[0]['drones']['lb1'])
        offsets = {'lb1': [0., 0.], 'lb2': [0., 0.]}
        active = 'lb1'
        drifted = False
        def exchange(command, *args):
            nonlocal active, drifted
            if command['cmd'] == 'SERVO_CALIBRATE':
                offsets[command['drone_id']] = list(command['offsets'])
                if command.get('selected'):
                    active = command['drone_id']
                    if active == 'lb2':
                        drifted = True
                return dict(offsets=command['offsets'])
            if command['cmd'] == 'MEASURE_SERVOS':
                bias = 2. if active == 'lb1' and drifted else 0.
                return dict(angles=[a+o+bias for a, o in zip(command['reference']['angles'], offsets[active])])
            return {}
        simulation._servo_exchange = exchange
        report = self.run_simulation(simulation)
        first = report['drones']['lb1']
        self.assertEqual(first['verification_pass'], 1)
        self.assertTrue(all(o < -1. for o in first['offsets']))
        self.assertTrue(all(abs(rod['error_deg']) <= .7 for rod in first['rods']))

    def test_reverse_angle_wrapping_and_correction_cap(self):
        rod = RodAdjustment(360., 0., (-15., 15.), DEFAULTS)
        rod.observe(.3)  # optical direction can cross 360 without a 359-degree correction
        rod.observe(.3)
        self.assertEqual(rod.status, 'converged')
        rod = RodAdjustment(180., 15., (-15., 15.), DEFAULTS)
        rod.observe(170.)
        self.assertEqual(rod.status, 'limit_reached')


class TransportTests(unittest.TestCase):
    def launcher(self):
        launcher = ServoCalibrationMixin()
        launcher.running = threading.Event()
        launcher.running.set()
        launcher.emergency = False
        launcher.pub_socket = Mock()
        launcher.pull_socket = Mock()
        launcher.pull_socket.poll.return_value = True
        launcher._handle_marker_grid_event = Mock(return_value=False)
        launcher._handle_camera_runtime_event = Mock()
        return launcher

    def test_stale_nonce_and_old_run_are_ignored(self):
        launcher = self.launcher()
        launcher.pull_socket.recv_json.side_effect = [
            dict(id='CAM', status='SERVO_MEASURED', request_id='old', run_id='run'),
            dict(id='CAM', status='SERVO_MEASURED', request_id='nonce', run_id='old'),
            dict(id='CAM', status='SERVO_MEASURED', request_id='nonce', run_id='run')]
        with patch.object(calibration.uuid, 'uuid4', return_value=SimpleNamespace(hex='nonce')):
            launcher._servo_exchange(dict(cmd='MEASURE_SERVOS', run_id='run'), 'CAM', 'SERVO_MEASURED', float('inf'))
        self.assertEqual(launcher.pull_socket.recv_json.call_count, 3)

    def test_dropped_messages_retry_same_nonce_and_timeout(self):
        launcher = self.launcher()
        launcher.pull_socket.poll.return_value = False
        with patch.object(calibration.time, 'monotonic', side_effect=[1., 1.6, 2.1]):
            with self.assertRaises(TimeoutError):
                launcher._servo_exchange(dict(cmd='SERVO_CALIBRATE'), 'lb1', 'SERVO_APPLIED', 2.)
        self.assertEqual(launcher.pub_socket.send_json.call_count, 2)
        calls = launcher.pub_socket.send_json.call_args_list
        self.assertEqual(calls[0].args[0]['request_id'], calls[1].args[0]['request_id'])


class LaunchOrderingTests(unittest.TestCase):
    def launch(self, enabled=True, fail=False):
        from orchestrator import SwarmOrchestrator
        launcher = SwarmOrchestrator.__new__(SwarmOrchestrator)
        launcher.args = SimpleNamespace(off=None, kill=None, blender=False, record=False,
            skip_dispatcher=True, skip_reboot=True, skip_confirm=True, loadcell=False,
            interaction=False, ground=True, droneless=True)
        launcher.logger = Mock()
        launcher.manifest, launcher.missions, _ = configuration()
        launcher.missions *= 2
        launcher.mission = launcher.missions[0]
        launcher.drones = launcher.manifest['drones']
        launcher.servo_calibration_cfg = dict(enabled=enabled)
        launcher.running = threading.Event()
        launcher.emergency = False
        launcher.imu_validation_plan = None
        launcher.pending_downloads = []
        launcher.common_cfg = {}
        launcher.camera_pose = None
        launcher.loadcell_thread = None
        launcher.pub_socket = Mock()
        events = []
        launcher.pub_socket.send_json.side_effect = lambda message: events.append(message['cmd']) if message['cmd'] != '_' else None
        for method in ('setup_network', '_apply_network_mode', '_configure_camera_pose',
                       '_configure_servo_calibration', '_prepare_camera_run', '_process_pending_downloads',
                       '_boot_marker_grid', '_boot_cameras', '_blender_notify'):
            setattr(launcher, method, Mock())
        launcher._camera_pose_enabled = Mock(return_value=False)
        launcher._measure_mocap_camera_pose = Mock(return_value=None)
        launcher._get_drone_cmd = Mock(return_value='not executed')
        launcher._boot_remote_node = Mock(return_value=True)
        launcher._wait_for_ready = lambda index=None: events.append(f'ready:{index}')
        launcher._start_camera_recording = lambda: events.append('record')

        def calibrate():
            if enabled:
                events.append('calibrate')
                if fail:
                    raise RuntimeError('calibration failed')

        launcher._calibrate_illumination_servos = calibrate
        launcher._monitor_flight = lambda: events.append('monitor')
        launcher.emergency_stop = lambda: events.append('emergency')
        launcher.cleanup = lambda: events.append('cleanup')
        with patch('orchestrator.time.sleep'):
            launcher.run()
        return events

    def test_enabled_calibrates_once_at_target_before_recording_and_illumination(self):
        self.assertEqual(self.launch(), ['ready:None', 'START', 'ready:0', 'calibrate',
            'record', 'START', 'ready:1', 'record', 'START', 'monitor', 'cleanup'])

    def test_failed_calibration_cannot_release_illumination_handshake(self):
        self.assertEqual(self.launch(fail=True), ['ready:None', 'START', 'ready:0',
            'calibrate', 'emergency', 'cleanup'])

    def test_disabled_preserves_initial_recording_and_launch_sequence(self):
        self.assertEqual(self.launch(enabled=False), ['ready:None', 'record', 'START',
            'ready:0', 'record', 'START', 'ready:1', 'record', 'START', 'monitor', 'cleanup'])


if __name__ == '__main__':
    unittest.main()
