"""Offline launcher checks for the dedicated ``--mpc`` calibration mode."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest

import yaml


sys.path.insert(0, str(Path(__file__).parents[1]))
from interaction_launch import (  # noqa: E402
    interaction_adaptive_calibration_args,
    interaction_application_flag,
    interaction_braking_test_args,
    interaction_contact_attitude_args,
    interaction_mocap_args,
    interaction_targeted_calibration_args,
    validate_interaction_modes,
)


def cli_parser():
    """Load only argument registrations; never construct the orchestrator."""
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare))
    statements = [node for node in main.body if (
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
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=statements, type_ignores=[])
    ), str(source), 'exec'), namespace)
    return namespace['parser']


from launcher_test_support import command_builder


def mpc_command():
    methods = command_builder()
    launcher = SimpleNamespace(
        args=SimpleNamespace(
            interaction=False,
            calibrate=False,
            braking_test=False,
            mpc=True,
            sense=False,
            radio=False,
            droneless=False,
            ground=False,
        ),
        mission={
            'drones': {'lb11': {'target': [0, 0, 1]}},
            'Interaction': {
                'config': {'detection_method': 'momentum_impulse'},
            },
        },
        manifest={},
        common_cfg={
            'work_dir': '/home/fls/fls-cf-offboard-controller',
            'venv_path': '/home/fls/env',
        },
        tag='mpc_bootstrap',
    )
    launcher._get_drone_cmd_interaction = lambda drone: (
        methods['_get_drone_cmd_interaction'](launcher, drone)
    )
    command = methods['_get_drone_cmd'](launcher, {
        'id': 'lb11',
        'init_pos': [0, 0, .24],
        'servo_count': 0,
        'led_count': 0,
    })
    return shlex.split(command)


class MPCCalibrationLaunchTests(unittest.TestCase):
    def test_translation_inertia_keeps_experimental_bootstrap_disabled(self):
        mission_path = (
            Path(__file__).parents[2]
            / 'Interaction' / 'SFL' / 'translation_inertia.yaml'
        )
        mission = yaml.safe_load(mission_path.read_text())
        config = mission['Interaction']['config']['wrench_interaction']
        self.assertEqual(config['mpc_bootstrap_calibration'], {'enabled': False})
        self.assertFalse(config['learning_velocity_mpc_shadow']['enabled'])
        self.assertFalse(config['learning_velocity_mpc_shadow']['command_authority'])

    def test_cli_registers_the_single_flag(self):
        parser = cli_parser()
        args = parser.parse_args(['--mpc', '--skip-record'])
        validate_interaction_modes(args)
        self.assertTrue(args.mpc)
        self.assertTrue(args.skip_record)
        self.assertFalse(args.interaction)
        self.assertFalse(args.calibrate)
        self.assertFalse(args.braking_test)

    def test_mpc_is_a_distinct_application_mode(self):
        args = SimpleNamespace(mpc=True)
        self.assertEqual(interaction_application_flag(args), '--mpc')

    def test_mpc_rejects_every_competing_experiment_mode(self):
        for mode in ('interaction', 'calibrate', 'braking_test'):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                    ValueError, 'mutually exclusive'):
                validate_interaction_modes(
                    SimpleNamespace(mpc=True, **{mode: True})
                )
        for mode in (
                'sense', 'ground', 'illumination',
                'intractable_illumination', 'morphing', 'blender', 'record'):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                    ValueError, 'cannot|does not use'):
                validate_interaction_modes(
                    SimpleNamespace(mpc=True, **{mode: True})
                )

    def test_calibration_only_options_never_leak_into_mpc(self):
        for option in (
                {'targeted_braking_calibration': True},
                {'adaptive_braking_calibration': True},
                {'adaptive_braking_calibration': False}):
            with self.subTest(option=option), self.assertRaisesRegex(
                    ValueError, 'requires --calibrate'):
                validate_interaction_modes(SimpleNamespace(mpc=True, **option))

    def test_command_is_logged_high_rate_mpc_without_other_authority_flags(self):
        tokens = mpc_command()
        self.assertIn('--mpc', tokens)
        self.assertNotIn('--interaction', tokens)
        self.assertNotIn('--calibrate', tokens)
        self.assertNotIn('--braking-test', tokens)
        self.assertNotIn('--sense', tokens)
        self.assertNotIn('--adaptive-braking-calibration', tokens)
        self.assertNotIn('--targeted-braking-calibration', tokens)
        self.assertIn('--orchestrated', tokens)
        self.assertIn('--log', tokens)
        self.assertEqual(
            tokens[tokens.index('--smooth-controller-rate') + 1], '100'
        )
        self.assertEqual(tokens[tokens.index('--cf-log-period') + 1], '10')

    def test_validation_occurs_before_orchestrator_creation(self):
        source = (Path(__file__).parents[1] / 'orchestrator.py').read_text()
        validation = source.rindex('validate_interaction_modes(args, defer_sensing=True)')
        creation = source.rindex('orchestrator = SwarmOrchestrator(args)')
        self.assertLess(validation, creation)

    def test_parser_rejects_combined_mode_before_launch(self):
        parser = cli_parser()
        args = parser.parse_args(['--mpc', '--interaction'])
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            validate_interaction_modes(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit):
            parser.parse_args(['--mpc=online'])


if __name__ == '__main__':
    unittest.main()
