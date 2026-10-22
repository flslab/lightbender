"""SFL sensor selection through the production command builder; no hardware."""
import copy
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from interaction_launch import resolve_interaction_sensing, validate_interaction_modes
from test_braking_test_launch import cli_parser, command_builder


class SflSensingTests(unittest.TestCase):
    def test_prev_mission_keeps_interaction_entry_without_sensor_or_detector(self):
        import yaml
        path = Path(__file__).parents[2] / 'Interaction/SFL/prev_behavior.yaml'
        mission = yaml.safe_load(path.read_text())
        config = mission['Interaction']['config']
        self.assertEqual(config['behavior'], 'prev')
        self.assertNotIn('detection_method', config)
        self.assertNotIn('wrench_interaction', config)
        for flags in ((), ('--sense',)):
            args = self.args(*flags)
            before = copy.deepcopy(vars(args))
            tokens = self.command(args, config)
            self.assertIn('--interaction', tokens)
            self.assertNotIn('--illumination', tokens)
            self.assertNotIn('--sense', tokens)
            self.assertEqual(vars(args), before)

    def args(self, *extra):
        return cli_parser().parse_args(['--interaction', *extra])

    def config(self, detector='potentiometer', **extra):
        return {'behavior':'level_coast', 'detection_method':detector,
                'level_coast':{}, **extra}

    def command(self, args, config):
        owner = SimpleNamespace(args=args, mission={
            'drones':{'lb11':{'target':[0,0,1]}}, 'Interaction':{'config':config}},
            manifest={}, common_cfg={'work_dir':'/home/fls/fls-cf-offboard-controller',
                                     'venv_path':'/home/fls/env'}, tag='sfl_sensor_test')
        return shlex.split(command_builder()['_get_drone_cmd_interaction'](owner, {
            'id':'lb11', 'label':'FLS', 'init_pos':[0,0,.24], 'servo_count':0, 'led_count':0}))

    def test_potentiometer_automatically_launches_reader_with_existing_logging(self):
        args, config = self.args(), self.config()
        before = copy.deepcopy((vars(args), config))
        tokens = self.command(args, config)
        self.assertEqual(tokens.count('--sense'), 1)
        self.assertIn('--interaction', tokens)
        self.assertIn('--log', tokens)
        self.assertEqual(tokens[tokens.index('--cf-log-period')+1], '10')
        self.assertEqual((vars(args), config), before)

    def test_model_and_velocity_do_not_inherit_unused_sensor_thresholds(self):
        for detector in ('model', 'vel'):
            config = self.config(detector, virtual_object={
                'contact_detection':{'source':'potentiometer'},
                'release_behavior':{'mode':'potentiometer_coast'}})
            with self.subTest(detector=detector):
                self.assertNotIn('--sense', self.command(self.args(), config))

    def test_omitted_detection_method_uses_same_potentiometer_default_as_controller(self):
        self.assertTrue(resolve_interaction_sensing(self.args(), {'behavior':'level_coast'}).sense)

    def test_without_recording_level_coast_ignores_legacy_sensor_overrides(self):
        for detector in ('potentiometer', 'model', 'vel'):
            for legacy in ({}, {'sensing': True}, {'sensing': False}):
                for flags in ((), ('--sense',)):
                    with self.subTest(detector=detector, legacy=legacy, flags=flags):
                        args, config = self.args(*flags), self.config(detector, **legacy)
                        before = copy.deepcopy((vars(args), config))
                        tokens = self.command(args, config)
                        self.assertEqual('--sense' in tokens, detector == 'potentiometer')
                        self.assertEqual((vars(args), config), before)

    def test_recording_starts_sensor_and_logging_without_changing_detector(self):
        for detector in ('potentiometer', 'model', 'vel'):
            for record in (True, False):
                with self.subTest(detector=detector, record=record):
                    args = self.args()
                    config = self.config(detector, record_potentiometer=record, sensing=False)
                    before = copy.deepcopy((vars(args), config))
                    tokens = self.command(args, config)
                    self.assertEqual('--sense' in tokens, record or detector == 'potentiometer')
                    self.assertIn('--log', tokens)
                    self.assertEqual((vars(args), config), before)

    def test_standard_recording_enables_sensor_and_rejects_explicit_conflict(self):
        config = {'behavior': 'standard', 'detection_method': 'momentum_impulse',
                  'record_potentiometer': True}
        self.assertIn('--sense', self.command(self.args(), config))
        with self.assertRaisesRegex(ValueError, 'sensing: false conflicts'):
            self.command(self.args(), {**config, 'sensing': False})

    def test_standard_optional_acquisition_retains_yaml_and_cli_compatibility(self):
        for detector in ('model', 'vel'):
            with self.subTest(detector=detector):
                self.assertIn('--sense', self.command(self.args(), self.config(
                    detector, behavior='standard', sensing=True)))
                self.assertNotIn('--sense', self.command(self.args('--sense'), self.config(
                    detector, behavior='standard', sensing=False)))
                self.assertIn('--sense', self.command(self.args('--sense'), self.config(
                    detector, behavior='standard')))

    def test_sensor_dependent_mission_cannot_disable_reader(self):
        with self.assertRaisesRegex(ValueError, 'sensing: false conflicts'):
            self.command(self.args(), self.config(behavior='standard', sensing=False,
                virtual_object={'contact_detection': {'source': 'potentiometer'}}))

    def test_standard_contact_or_release_selection_also_enables_sensor(self):
        for virtual in (
                {'contact_detection':{'source':'potentiometer'}},
                {'release_behavior':{'mode':'potentiometer_coast'}}):
            with self.subTest(virtual=virtual):
                config = {'detection_method':'momentum_impulse', 'virtual_object':virtual}
                self.assertIn('--sense', self.command(self.args(), config))
        self.assertNotIn('--sense', self.command(self.args(), {}))

    def test_flight_modes_ignore_interaction_sensor_settings(self):
        for mode in ('calibrate','braking-test','mpc','baseline','hover'):
            args = cli_parser().parse_args(['--'+mode])
            with self.subTest(mode=mode):
                self.assertNotIn('--sense', self.command(args, self.config(
                    sensing=True, record_potentiometer=True)))

    def test_sensor_protocol_validates_after_yaml_resolution_before_command_is_built(self):
        args = self.args('--contact-attitude-run','2')
        validate_interaction_modes(args, defer_sensing=True)
        with self.assertRaisesRegex(ValueError, 'requires --sense'):
            validate_interaction_modes(args)
        tokens = self.command(args, self.config())
        self.assertIn('--sense', tokens)
        self.assertIn('--contact-attitude-run', tokens)
        self.assertIn('--sense', self.command(args, self.config('model', sensing=False)))
        args.ground = True
        with self.assertRaisesRegex(ValueError, 'airborne hardware'):
            validate_interaction_modes(args, defer_sensing=True)

    def test_active_brake_protocol_cannot_be_weakened_by_yaml(self):
        args = cli_parser().parse_args(['--active-septic-brake-test'])
        self.assertTrue(resolve_interaction_sensing(args, self.config('model')).sense)
        self.assertTrue(resolve_interaction_sensing(args, self.config('model', sensing=False)).sense)

    def test_each_mission_resolves_independently(self):
        args = self.args()
        self.assertTrue(resolve_interaction_sensing(args, self.config()).sense)
        self.assertTrue(resolve_interaction_sensing(args, self.config('model', record_potentiometer=True)).sense)
        self.assertFalse(resolve_interaction_sensing(args, self.config('model')).sense)
        self.assertFalse(args.sense)

    def test_bad_sensing_and_detection_method_values_fail_before_launch(self):
        for value in ('false', 0, None, {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'record_potentiometer must be boolean'):
                self.command(self.args(), self.config(record_potentiometer=value))
        for value in ('false', 0, None, {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be boolean'):
                self.command(self.args(), self.config(behavior='standard', sensing=value))
        for method in ('typo', 'momentum_impulse', None, False):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'detection_method must be'):
                self.command(self.args(), self.config(method))
        with self.assertRaisesRegex(ValueError, 'detector was removed'):
            self.command(self.args(), self.config(level_coast={'detector': 'model'}))


if __name__ == '__main__':
    unittest.main()
