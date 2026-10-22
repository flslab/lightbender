"""Offline launcher checks: no SSH, hardware imports, or flight startup."""

import ast
import argparse
import contextlib
import io
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


from launcher_test_support import command_builder


def cli_parser():
    """Run only argparse registration statements, never application startup."""
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
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                 str(source), 'exec'), namespace)
    return namespace['parser']


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

    def test_explicit_selectors_require_braking_test_even_when_default(self):
        for selector in (
            {'braking_test_direction': 'both'},
            {'braking_test_repetitions': 3},
        ):
            for mode in (None, 'interaction', 'calibrate'):
                with self.subTest(selector=selector, mode=mode):
                    args = SimpleNamespace(**selector, **({mode: True} if mode else {}))
                    with self.assertRaisesRegex(ValueError, 'require --braking-test'):
                        interaction_braking_test_args(args)

    def test_selector_validation_rejects_noninteger_and_out_of_range_values(self):
        for value in (True, False, 1.0, 2.5, '1', 0, -1, 4):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'integer from 1 to 3'):
                validate_interaction_modes(SimpleNamespace(braking_test=True, braking_test_repetitions=value))
        with self.assertRaisesRegex(ValueError, 'must be both'):
            validate_interaction_modes(SimpleNamespace(braking_test=True, braking_test_direction='y'))

    def test_omitted_selectors_do_not_change_existing_mode_arguments(self):
        for mode in ('interaction', 'calibrate', 'braking_test'):
            with self.subTest(mode=mode):
                self.assertEqual(interaction_braking_test_args(SimpleNamespace(**{mode: True})), '')
                self.assertEqual(interaction_braking_test_args(SimpleNamespace(
                    **{mode: True}, braking_test_direction=None, braking_test_repetitions=None)), '')

    def test_each_explicit_selector_is_forwarded(self):
        for direction in ('both', 'positive-y', 'negative-y'):
            for count in (1, 2, 3):
                with self.subTest(direction=direction, count=count):
                    self.assertEqual(interaction_braking_test_args(SimpleNamespace(
                        braking_test=True, braking_test_direction=direction, braking_test_repetitions=count)),
                        f' --braking-test-direction {direction} --braking-test-repetitions {count}')
        self.assertEqual(interaction_braking_test_args(SimpleNamespace(
            braking_test=True, braking_test_direction='positive-y')),
            ' --braking-test-direction positive-y')
        self.assertEqual(interaction_braking_test_args(SimpleNamespace(
            braking_test=True, braking_test_repetitions=1)),
            ' --braking-test-repetitions 1')

    def test_actual_cli_parses_single_positive_y_and_preserves_absent_defaults(self):
        parser = cli_parser()
        defaults = parser.parse_args(['--braking-test'])
        self.assertIsNone(defaults.braking_test_direction)
        self.assertIsNone(defaults.braking_test_repetitions)
        args = parser.parse_args([
            '--braking-test', '--braking-test-direction', 'positive-y',
            '--braking-test-repetitions', '1', '--skip-record',
        ])
        validate_interaction_modes(args)
        self.assertEqual(args.braking_test_direction, 'positive-y')
        self.assertEqual(args.braking_test_repetitions, 1)
        self.assertTrue(args.skip_record)
        for option, value in (
            ('--braking-test-direction', 'y'),
            ('--braking-test-repetitions', '0'),
            ('--braking-test-repetitions', '4'),
            ('--braking-test-repetitions', '1.0'),
        ):
            with self.subTest(option=option, value=value):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(['--braking-test', option, value])

    def test_other_application_modes_rejected_before_launch(self):
        for mode in ('illumination', 'intractable_illumination', 'morphing', 'blender'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_interaction_modes(SimpleNamespace(braking_test=True, **{mode: True}))

    def braking_test_command(self, **options):
        methods = command_builder()
        launcher = SimpleNamespace(
            args=SimpleNamespace(
                interaction=False, braking_test=True, sense=False,
                radio=False, droneless=False, ground=False,
                **options,
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
        return shlex.split(command)

    def test_braking_test_dispatches_logged_100hz_controller_without_sensor(self):
        tokens = self.braking_test_command()
        self.assertIn('--braking-test', tokens)
        self.assertNotIn('--calibrate', tokens)
        self.assertNotIn('--interaction', tokens)
        self.assertNotIn('--sense', tokens)
        self.assertIn('--log', tokens)
        self.assertEqual(tokens[tokens.index('--smooth-controller-rate') + 1], '100')
        self.assertEqual(tokens[tokens.index('--cf-log-period') + 1], '10')
        self.assertNotIn('--braking-test-direction', tokens)
        self.assertNotIn('--braking-test-repetitions', tokens)

    def test_single_positive_y_dispatches_explicit_selectors_to_controller(self):
        tokens = self.braking_test_command(braking_test_direction='positive-y', braking_test_repetitions=1)
        self.assertEqual(tokens[tokens.index('--braking-test-direction') + 1], 'positive-y')
        self.assertEqual(tokens[tokens.index('--braking-test-repetitions') + 1], '1')
        self.assertIn('--braking-test', tokens)
        self.assertIn('--log', tokens)
        self.assertNotIn('--sense', tokens)

    def test_cli_registers_flag_and_validates_before_orchestrator_creation(self):
        source = (Path(__file__).parents[1] / 'orchestrator.py').read_text()
        self.assertIn('"--braking-test", action="store_true"', source)
        validation = source.rindex('validate_interaction_modes(args, defer_sensing=True)')
        creation = source.rindex('orchestrator = SwarmOrchestrator(args)')
        self.assertLess(validation, creation)


if __name__ == '__main__':
    unittest.main()
