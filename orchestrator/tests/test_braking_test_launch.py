"""Offline launcher checks: no SSH, hardware imports, or flight startup."""

import ast
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from interaction_launch import (  # noqa: E402
    interaction_application_flag,
    interaction_mocap_args,
    validate_interaction_modes,
)


def command_builder():
    # Exercise the production launch method without importing the module's
    # optional networking/hardware packages or constructing an orchestrator.
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    orchestrator = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'SwarmOrchestrator')
    methods = [node for node in orchestrator.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_get_drone_cmd', '_get_drone_cmd_interaction')]
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {
        'interaction_application_flag': interaction_application_flag,
        'interaction_mocap_args': interaction_mocap_args,
        'DRONE_SCRIPT': 'controller.py',
    }
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace


class BrakingTestLaunchTests(unittest.TestCase):
    def test_modes_remain_distinct(self):
        for name, flag in (
            ('interaction', '--interaction'),
            ('calibrate', '--calibrate'),
            ('braking_test', '--braking-test'),
        ):
            with self.subTest(name=name):
                self.assertEqual(interaction_application_flag(SimpleNamespace(**{name: True})), flag)

    def test_pairwise_mode_conflicts_are_rejected(self):
        for first, second in (
            ('interaction', 'calibrate'),
            ('interaction', 'braking_test'),
            ('calibrate', 'braking_test'),
        ):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    validate_interaction_modes(SimpleNamespace(**{first: True, second: True}))

    def test_sensor_is_not_a_dependency_of_braking_test(self):
        validate_interaction_modes(SimpleNamespace(braking_test=True))
        with self.assertRaisesRegex(ValueError, 'omit --sense'):
            interaction_application_flag(SimpleNamespace(braking_test=True, sense=True))

    def test_other_application_modes_rejected_before_launch(self):
        for mode in ('illumination', 'intractable_illumination', 'morphing', 'blender'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_interaction_modes(SimpleNamespace(braking_test=True, **{mode: True}))

    def test_braking_test_dispatches_logged_100hz_controller_without_sensor(self):
        methods = command_builder()
        launcher = SimpleNamespace(
            args=SimpleNamespace(
                interaction=False, braking_test=True, sense=False,
                radio=False, droneless=False, ground=False,
            ),
            mission={
                'drones': {'lb11': {'target': [0, 0, 1]}},
                'Interaction': {'config': {'detection_method': 'momentum_impulse'}},
            },
            manifest={},
            common_cfg={'work_dir': '/home/fls/fls-cf-offboard-controller',
                        'venv_path': '/home/fls/env'},
            tag='repeat_test',
        )
        launcher._get_drone_cmd_interaction = lambda drone: methods['_get_drone_cmd_interaction'](launcher, drone)
        command = methods['_get_drone_cmd'](launcher, {
            'id': 'lb11', 'init_pos': [0, 0, .24], 'servo_count': 0, 'led_count': 0,
        })
        tokens = shlex.split(command)
        self.assertIn('--braking-test', tokens)
        self.assertNotIn('--calibrate', tokens)
        self.assertNotIn('--interaction', tokens)
        self.assertNotIn('--sense', tokens)
        self.assertIn('--log', tokens)
        self.assertEqual(tokens[tokens.index('--smooth-controller-rate') + 1], '100')
        self.assertEqual(tokens[tokens.index('--cf-log-period') + 1], '10')

    def test_cli_registers_flag_and_validates_before_orchestrator_creation(self):
        source = (Path(__file__).parents[1] / 'orchestrator.py').read_text()
        self.assertIn('"--braking-test", action="store_true"', source)
        validation = source.rindex('validate_interaction_modes(args)')
        creation = source.rindex('orchestrator = SwarmOrchestrator(args)')
        self.assertLess(validation, creation)


if __name__ == '__main__':
    unittest.main()
