"""Offline checks for opt-in rigid-body XYZ-only controller forwarding."""

from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from interaction_launch import (  # noqa: E402
    contact_attitude_skips_pointcloud_dispatcher,
    interaction_mocap_args,
    validate_interaction_modes,
)
from test_braking_test_launch import cli_parser, command_builder  # noqa: E402


class RigidBodyPositionOnlyLaunchTests(unittest.TestCase):
    def test_active_septic_calibration_selects_fixed_mission_fit(self):
        parser = cli_parser()
        args = parser.parse_args(['--active-septic-brake-calibration'])
        validate_interaction_modes(args)
        self.assertTrue(args.calibrate)
        self.assertTrue(args.planar_braking_calibration)
        self.assertTrue(args.vicon_rigidbody_position_only)
        self.assertFalse(args.interaction)
        self.assertFalse(args.sense)
        self.assertTrue(contact_attitude_skips_pointcloud_dispatcher(args))
        validate_interaction_modes(args)

        methods = command_builder()
        launcher = SimpleNamespace(
            args=args,
            mission={
                'drones': {'lb11': {'target': [0, 0, 1]}},
                'Interaction': {'config': {'detection_method': 'momentum_impulse'}},
            },
            manifest={},
            common_cfg={
                'work_dir': '/home/fls/fls-cf-offboard-controller',
                'venv_path': '/home/fls/env',
            },
            tag='active_septic_calibration',
        )
        tokens = shlex.split(methods['_get_drone_cmd_interaction'](
            launcher,
            dict(id='lb11', label='FLS', init_pos=[0, -1, .24],
                 servo_count=0, led_count=0),
        ))
        for expected in ('--calibrate', '--planar-braking-calibration',
                         '--no-adaptive-braking-calibration',
                         '--vicon-rigidbody-position-only'):
            self.assertIn(expected, tokens)
        self.assertNotIn('--sense', tokens)
        self.assertNotIn('--active-septic-brake-test', tokens)

    def test_active_septic_calibration_refuses_other_modes(self):
        parser = cli_parser()
        for extra in ('--interaction', '--calibrate', '--radio',
                      '--skip-confirm', '--adaptive-braking-calibration',
                      '--active-septic-brake-test'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_interaction_modes(parser.parse_args([
                    '--active-septic-brake-calibration', *extra.split(),
                ]))

    def test_active_septic_single_flag_expands_to_safe_interaction_route(self):
        parser = cli_parser()
        args = parser.parse_args(['--active-septic-brake-test'])
        validate_interaction_modes(args)
        self.assertTrue(args.interaction)
        self.assertTrue(args.sense)
        self.assertTrue(args.vicon_rigidbody_position_only)
        self.assertTrue(contact_attitude_skips_pointcloud_dispatcher(args))
        # Helpers may validate more than once while building the command.
        validate_interaction_modes(args)

    def test_active_septic_refuses_combined_modes_and_skipped_confirmation(self):
        parser = cli_parser()
        for extra in ('--interaction', '--calibrate', '--radio',
                      '--skip-confirm', '--contact-attitude-run 2'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_interaction_modes(parser.parse_args([
                    '--active-septic-brake-test', *extra.split(),
                ]))

    def test_active_septic_forwards_controller_prearm_flag(self):
        methods = command_builder()
        args = cli_parser().parse_args(['--active-septic-brake-test'])
        validate_interaction_modes(args)
        launcher = SimpleNamespace(
            args=args,
            mission={
                'drones': {'lb11': {'target': [0, 0, 1]}},
                'Interaction': {'config': {'detection_method': 'momentum_impulse'}},
            },
            manifest={},
            common_cfg={
                'work_dir': '/home/fls/fls-cf-offboard-controller',
                'venv_path': '/home/fls/env',
            },
            tag='active_septic_test',
        )
        tokens = shlex.split(methods['_get_drone_cmd_interaction'](
            launcher,
            dict(id='lb11', label='FLS', init_pos=[0, -1, .24],
                 servo_count=0, led_count=0),
        ))
        for expected in (
                '--active-septic-brake-test', '--interaction', '--sense',
                '--vicon-rigidbody-position-only', '--log',
                '--smooth-controller-rate'):
            self.assertIn(expected, tokens)
        self.assertNotIn('--vicon-full-pose', tokens)
        self.assertNotIn('--contact-attitude-run', tokens)

    def test_option_requires_interaction_or_calibration_and_no_shadow_run(self):
        parser = cli_parser()
        for mode in ('--interaction', '--calibrate'):
            args = parser.parse_args([mode, '--vicon-rigidbody-position-only'])
            validate_interaction_modes(args)
            self.assertTrue(contact_attitude_skips_pointcloud_dispatcher(args))
        for tokens in (
            ['--vicon-rigidbody-position-only'],
            ['--interaction', '--contact-attitude-run', '2', '--sense',
             '--vicon-rigidbody-position-only'],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_interaction_modes(parser.parse_args(tokens))

    def test_route_uses_manifest_name_without_full_pose(self):
        drone = dict(id='lb11', label='FLS', init_pos=[0, -1, .24])
        self.assertEqual(
            interaction_mocap_args(drone, rigidbody_position_only=True),
            '--vicon-rigidbody-position-only FLS ',
        )
        with self.assertRaisesRegex(ValueError, 'requires obj_name or label'):
            interaction_mocap_args(
                dict(id='lb11', init_pos=[0, -1, .24]),
                rigidbody_position_only=True,
            )

    def test_actual_launch_forwards_single_option_to_controller(self):
        methods = command_builder()
        for mode in ('interaction', 'calibrate'):
            with self.subTest(mode=mode):
                launcher = SimpleNamespace(
                    args=SimpleNamespace(
                        interaction=mode == 'interaction',
                        calibrate=mode == 'calibrate',
                        vicon_rigidbody_position_only=True,
                        sense=mode == 'interaction', radio=False,
                        droneless=False, ground=False,
                    ),
                    mission={
                        'drones': {'lb11': {'target': [0, 0, 1]}},
                        'Interaction': {'config': {
                            'detection_method': 'momentum_impulse',
                        }},
                    },
                    manifest={},
                    common_cfg={
                        'work_dir': '/home/fls/fls-cf-offboard-controller',
                        'venv_path': '/home/fls/env',
                    },
                    tag='rigidbody_position_only_test',
                )
                command = methods['_get_drone_cmd_interaction'](
                    launcher,
                    dict(id='lb11', label='FLS', init_pos=[0, -1, .24],
                         servo_count=0, led_count=0),
                )
                tokens = shlex.split(command)
                self.assertIn('--vicon-rigidbody-position-only', tokens)
                self.assertEqual(
                    tokens[tokens.index('--vicon-rigidbody-position-only') + 1],
                    'FLS',
                )
                self.assertNotIn('--vicon-full-pose', tokens)
                self.assertNotIn('--contact-attitude-run', tokens)
                self.assertIn('--log', tokens)


if __name__ == '__main__':
    unittest.main()
