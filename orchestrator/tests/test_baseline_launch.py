"""Offline launcher checks for the isolated ``--baseline`` flight path."""

import argparse
import ast
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest


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


def command_builder():
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    orchestrator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'SwarmOrchestrator'
    )
    methods = [
        node for node in orchestrator.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ('_get_drone_cmd', '_get_drone_cmd_interaction')
    ]
    namespace = {
        'interaction_adaptive_calibration_args': (
            interaction_adaptive_calibration_args
        ),
        'interaction_application_flag': interaction_application_flag,
        'interaction_braking_test_args': interaction_braking_test_args,
        'interaction_contact_attitude_args': interaction_contact_attitude_args,
        'interaction_mocap_args': interaction_mocap_args,
        'interaction_targeted_calibration_args': (
            interaction_targeted_calibration_args
        ),
        'DRONE_SCRIPT': 'controller.py',
    }
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]), str(source), 'exec'
        ),
        namespace,
    )
    return namespace


def cli_parser():
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
    )
    statements = [
        node for node in main.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == 'parser'
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == 'parser'
            and node.value.func.attr == 'add_argument'
        )
    ]
    namespace = {'argparse': argparse}
    exec(
        compile(
            ast.Module(body=statements, type_ignores=[]), str(source), 'exec'
        ),
        namespace,
    )
    return namespace['parser']


class BaselineLaunchTests(unittest.TestCase):
    @staticmethod
    def baseline_args(**overrides):
        values = {
            'baseline': True,
            'interaction': False,
            'calibrate': False,
            'braking_test': False,
            'mpc': False,
            'sense': False,
            'illumination': False,
            'intractable_illumination': False,
            'morphing': False,
            'blender': '',
            'record': False,
            'ground': False,
            'radio': False,
            'droneless': False,
            'targeted_braking_calibration': False,
            'adaptive_braking_calibration': None,
            'braking_test_direction': None,
            'braking_test_repetitions': None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_baseline_is_an_exclusive_application_mode(self):
        args = self.baseline_args()
        validate_interaction_modes(args)
        self.assertEqual(interaction_application_flag(args), '--baseline')
        for mode in ('interaction', 'calibrate', 'braking_test', 'mpc'):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, 'mutually exclusive'
            ):
                validate_interaction_modes(self.baseline_args(**{mode: True}))
        for mode in (
            'sense', 'ground', 'illumination',
            'intractable_illumination', 'morphing', 'record', 'droneless',
        ):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_interaction_modes(self.baseline_args(**{mode: True}))

    def test_orchestrator_cli_registers_baseline(self):
        args = cli_parser().parse_args(['--baseline', '--skip-record'])
        self.assertTrue(args.baseline)
        validate_interaction_modes(args)

    def test_baseline_dispatches_logged_100hz_controller(self):
        methods = command_builder()
        launcher = SimpleNamespace(
            args=self.baseline_args(),
            mission={
                'drones': {'lb11': {'target': [0, -1, 1]}},
                'Interaction': {
                    'config': {'detection_method': 'momentum_impulse'}
                },
            },
            manifest={},
            common_cfg={
                'work_dir': '/home/fls/fls-cf-offboard-controller',
                'venv_path': '/home/fls/env',
            },
            tag='baseline_test',
        )
        launcher._get_drone_cmd_interaction = lambda drone: (
            methods['_get_drone_cmd_interaction'](launcher, drone)
        )
        command = methods['_get_drone_cmd'](launcher, {
            'id': 'lb11',
            'init_pos': [0, -1, .24],
            'servo_count': 0,
            'led_count': 0,
        })
        tokens = shlex.split(command)

        self.assertIn('--baseline', tokens)
        for flag in ('--interaction', '--calibrate', '--braking-test', '--mpc'):
            self.assertNotIn(flag, tokens)
        self.assertNotIn('--sense', tokens)
        self.assertIn('--log', tokens)
        self.assertEqual(
            tokens[tokens.index('--smooth-controller-rate') + 1], '100'
        )
        self.assertEqual(
            tokens[tokens.index('--cf-log-period') + 1], '10'
        )


if __name__ == '__main__':
    unittest.main()
