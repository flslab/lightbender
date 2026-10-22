"""No-network verification of default adaptive-calibration command forwarding."""
import contextlib
import io
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from interaction_launch import (
    interaction_adaptive_calibration_args,
    interaction_targeted_calibration_args,
    validate_interaction_modes,
)
from test_braking_test_launch import cli_parser, command_builder


class AdaptiveCalibrationLaunchTests(unittest.TestCase):
    @staticmethod
    def translation_mission():
        mission_path = (
            Path(__file__).parents[2]
            / 'Interaction' / 'SFL' / 'translation_inertia.yaml'
        )
        with mission_path.open(encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    def test_translation_inertia_is_flight_and_xyz_calibration_not_training(self):
        mission = self.translation_mission()
        config = mission['Interaction']['config']['wrench_interaction']
        # Removed experiment tuning must not silently return to the flight
        # mission. Plain --calibrate still has its complete XYZ-only protocol.
        for name in ('planar_braking_calibration', 'adaptive_braking_calibration',
                     'online_prediction_calibration', 'mpc_bootstrap_calibration',
                     'predictive_braking'):
            self.assertEqual(config[name], {'enabled': False})
        self.assertEqual(config['learning_velocity_mpc_shadow'],
                         {'enabled': False, 'command_authority': False})
        excitation = config['calibration_excitation']
        self.assertFalse(excitation['enabled'])
        self.assertEqual(excitation['translation_profile'], 'sequential_chirp')
        self.assertEqual(excitation['translation_amplitude_m'], [.12, .12, .07])
        self.assertEqual(excitation['duration_s'], 30)
        self.assertEqual(excitation['yaw_amplitude_deg'], 0)
        self.assertEqual(mission['boundary_limits']['y_min'], -1.5)
        self.assertEqual(mission['boundary_limits']['y_max'], 1.5)

    def test_translation_inertia_preserves_compensated_attitude_baseline(self):
        config = self.translation_mission()['Interaction']['config']
        self.assertEqual(config['duration'], 60)
        self.assertEqual(config['grace_time'], .5)
        self.assertEqual(config['detection_method'], 'momentum_impulse')
        brake = config['wrench_interaction']['firmware_auto_brake']
        self.assertTrue(brake['enabled'])
        self.assertEqual(brake['mode'], 'scurve')
        self.assertEqual(brake['command_mode'], 'attitude')
        self.assertEqual(brake['stop_distance_m'], 0)
        profile = brake['analytic_profile']
        self.assertEqual(profile['shape'], 'velocity_scurve')
        self.assertEqual(profile['handoff'], 'curve_endpoint_forward')
        self.assertEqual(profile['feedback'], 'unified_vicon15')
        self.assertTrue(profile['response_compensation'])
        self.assertTrue(profile['acceleration_residual'])
        self.assertEqual(profile['position_tracking_bandwidth'], 3)
        self.assertFalse(profile['rate_feedforward'])
        self.assertFalse(profile['state_matched_start'])

    def test_translation_inertia_friction_conditions_do_not_implicitly_enable_control(self):
        config = self.translation_mission()['Interaction']['config']
        virtual = config['virtual_object']
        self.assertEqual(virtual['two_afc_friction'], {
            'enabled': True, 'high_mu': .1, 'low_mu': .02, 'random_seed': None,
        })
        self.assertFalse(virtual['force_rendering']['enabled'])
        self.assertFalse(config['wrench_interaction']['firmware_auto_brake'][
            'analytic_profile']['friction_from_interaction'])

    def test_cli_is_tristate_and_both_explicit_forms_parse(self):
        parser = cli_parser()
        self.assertIsNone(parser.parse_args([]).adaptive_braking_calibration)
        self.assertFalse(parser.parse_args([]).targeted_braking_calibration)
        for option, expected in (
                ('--adaptive-braking-calibration', True),
                ('--no-adaptive-braking-calibration', False)):
            with self.subTest(option=option):
                args = parser.parse_args(['--calibrate', option, '--skip-record'])
                validate_interaction_modes(args)
                self.assertTrue(args.calibrate)
                self.assertIs(args.adaptive_braking_calibration, expected)
                self.assertTrue(args.skip_record)

    def test_targeted_cli_requires_exclusive_airborne_calibration(self):
        parser = cli_parser()
        args = parser.parse_args([
            '--calibrate', '--targeted-braking-calibration', '--skip-record',
        ])
        validate_interaction_modes(args)
        self.assertTrue(args.targeted_braking_calibration)
        for options in (
                {}, {'interaction': True}, {'braking_test': True}, {'ground': True},
                {'calibrate': True, 'interaction': True},
                {'calibrate': True, 'braking_test': True},
                {'calibrate': True, 'ground': True}):
            with self.subTest(options=options), self.assertRaisesRegex(
                    ValueError, 'requires --calibrate'):
                validate_interaction_modes(SimpleNamespace(
                    targeted_braking_calibration=True, **options
                ))

    def test_explicit_selection_requires_exclusive_calibration(self):
        for enabled in (True, False):
            for options in ({}, {'interaction': True}, {'braking_test': True},
                            {'calibrate': True, 'interaction': True},
                            {'calibrate': True, 'braking_test': True}):
                with self.subTest(enabled=enabled, options=options), self.assertRaisesRegex(
                        ValueError, 'requires --calibrate'):
                    validate_interaction_modes(SimpleNamespace(
                        adaptive_braking_calibration=enabled, **options))

    def test_default_calibration_explicitly_disables_adaptive_and_other_modes_forward_nothing(self):
        self.assertEqual(interaction_adaptive_calibration_args(
            SimpleNamespace(calibrate=True)), ' --no-adaptive-braking-calibration')
        for mode in ('interaction', 'braking_test'):
            with self.subTest(mode=mode):
                self.assertEqual(interaction_adaptive_calibration_args(SimpleNamespace(**{mode: True})), '')

    def test_explicit_calibration_selection_is_forwarded(self):
        self.assertEqual(interaction_adaptive_calibration_args(SimpleNamespace(
            calibrate=True, adaptive_braking_calibration=True)), ' --adaptive-braking-calibration')
        self.assertEqual(interaction_adaptive_calibration_args(SimpleNamespace(
            calibrate=True, adaptive_braking_calibration=False)), ' --no-adaptive-braking-calibration')

    def test_targeted_selection_is_forwarded_and_forces_fixed_pulses(self):
        args = SimpleNamespace(
            calibrate=True, targeted_braking_calibration=True,
            adaptive_braking_calibration=None,
        )
        self.assertEqual(
            interaction_targeted_calibration_args(args),
            ' --targeted-braking-calibration',
        )
        self.assertEqual(
            interaction_adaptive_calibration_args(args),
            ' --no-adaptive-braking-calibration',
        )
        self.assertEqual(
            interaction_targeted_calibration_args(SimpleNamespace(calibrate=True)),
            '',
        )

    def test_cli_rejects_adaptive_selection_outside_calibration(self):
        parser = cli_parser()
        for option in ('--adaptive-braking-calibration', '--no-adaptive-braking-calibration'):
            with self.subTest(option=option):
                args = parser.parse_args(['--interaction', option])
                with self.assertRaisesRegex(ValueError, 'requires --calibrate'):
                    validate_interaction_modes(args)

    def command(self, enabled, *, targeted=False):
        methods = command_builder()
        launcher = SimpleNamespace(
            args=SimpleNamespace(calibrate=True, adaptive_braking_calibration=enabled,
                                 targeted_braking_calibration=targeted,
                                 interaction=False, braking_test=False, sense=False,
                                 radio=False, droneless=False, ground=False),
            mission={'drones': {'lb11': {'target': [0, 0, 1]}},
                     'Interaction': {'config': {'detection_method': 'momentum_impulse'}}},
            manifest={}, common_cfg={'work_dir': '/home/fls/fls-cf-offboard-controller',
                                     'venv_path': '/home/fls/env'}, tag='adaptive_test')
        launcher._get_drone_cmd_interaction = lambda drone: methods['_get_drone_cmd_interaction'](launcher, drone)
        return shlex.split(methods['_get_drone_cmd'](launcher, {
            'id': 'lb11', 'init_pos': [0, 0, .24], 'servo_count': 0, 'led_count': 0}))

    def test_real_command_builder_defaults_off_and_forwards_explicit_on(self):
        default = self.command(None)
        disabled = self.command(False)
        explicit = self.command(True)
        self.assertEqual(default.count('--no-adaptive-braking-calibration'), 1)
        self.assertNotIn('--adaptive-braking-calibration', default)
        self.assertEqual(disabled.count('--no-adaptive-braking-calibration'), 1)
        self.assertNotIn('--adaptive-braking-calibration', disabled)
        self.assertEqual(explicit.count('--adaptive-braking-calibration'), 1)
        self.assertEqual(default, disabled)
        self.assertEqual(
            [token for token in default if token != '--no-adaptive-braking-calibration'],
            [token for token in explicit if token != '--adaptive-braking-calibration'],
        )
        self.assertIn('--calibrate', default)
        self.assertNotIn('--interaction', default)
        self.assertNotIn('--braking-test', default)
        self.assertIn('--log', default)
        self.assertEqual(default[default.index('--smooth-controller-rate')+1], '100')
        self.assertEqual(default[default.index('--cf-log-period')+1], '10')

    def test_real_command_builder_forwards_targeted_mode_once(self):
        command = self.command(None, targeted=True)
        self.assertEqual(command.count('--targeted-braking-calibration'), 1)
        self.assertEqual(command.count('--no-adaptive-braking-calibration'), 1)
        self.assertNotIn('--adaptive-braking-calibration', command)
        self.assertIn('--calibrate', command)
        self.assertNotIn('--interaction', command)
        self.assertNotIn('--braking-test', command)


if __name__ == '__main__':
    unittest.main()
