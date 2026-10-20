import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).parents[1]))
from interaction_launch import (  # noqa: E402
    contact_attitude_skips_pointcloud_dispatcher,
    interaction_contact_attitude_args,
    interaction_mocap_args,
    validate_interaction_modes,
)


class InteractionLaunchTests(unittest.TestCase):
    @staticmethod
    def contact_args(run=2, **overrides):
        values = dict(
            contact_attitude_run=run,
            interaction=True,
            sense=True,
            ground=False,
            droneless=False,
            targeted_braking_calibration=False,
            calibrate=False,
            braking_test=False,
            mpc=False,
            baseline=False,
            hover=False,
            adaptive_braking_calibration=None,
            illumination=False,
            intractable_illumination=False,
            morphing=False,
            blender=False,
            record=False,
            braking_test_direction=None,
            braking_test_repetitions=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_contact_attitude_selector_is_forwarded(self):
        args = self.contact_args(2)
        validate_interaction_modes(args)
        self.assertEqual(
            interaction_contact_attitude_args(args),
            ' --contact-attitude-run 2',
        )

    def test_contact_attitude_selector_requires_sensor_backed_flight(self):
        with self.assertRaisesRegex(ValueError, 'requires --sense'):
            validate_interaction_modes(self.contact_args(2, sense=False))
        with self.assertRaisesRegex(ValueError, 'airborne hardware'):
            validate_interaction_modes(self.contact_args(2, ground=True))

    def test_only_rigidbody_contact_runs_skip_pointcloud_dispatcher(self):
        self.assertFalse(contact_attitude_skips_pointcloud_dispatcher(
            self.contact_args(1)
        ))
        self.assertTrue(contact_attitude_skips_pointcloud_dispatcher(
            self.contact_args(2)
        ))
        self.assertTrue(contact_attitude_skips_pointcloud_dispatcher(
            self.contact_args(3)
        ))
        self.assertFalse(contact_attitude_skips_pointcloud_dispatcher(
            self.contact_args(None)
        ))

    def test_wrench_mode_restores_pointcloud_default_when_init_position_exists(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'label': 'FLS', 'init_pos': [0, 0, 0.24],
        }, wrench_mode=True)
        self.assertEqual(
            args,
            '--init-pos 0 0 0.24 --vicon-mode pointcloud ',
        )

    def test_wrench_mode_accepts_explicit_pointcloud(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'label': 'FLS', 'init_pos': [0, 0, 0.24],
            'interaction_localization': 'pointcloud',
        }, wrench_mode=True)
        self.assertEqual(args, '--init-pos 0 0 0.24 --vicon-mode pointcloud ')

    def test_explicit_rigidbody_remains_available(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'label': 'FLS', 'init_pos': [0, 0, 0.24],
            'interaction_localization': 'rigidbody',
        }, wrench_mode=True)
        self.assertEqual(
            args,
            '--obj-name FLS --vicon-mode rigidbody --vicon-full-pose ',
        )

    def test_legacy_mode_keeps_pointcloud_default(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'label': 'FLS', 'init_pos': [1, 2, 3],
        }, wrench_mode=False)
        self.assertEqual(args, '--init-pos 1 2 3 --vicon-mode pointcloud ')

    def test_contact_attitude_run_one_forces_pointcloud(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'label': 'FLS', 'obj_name': 'FLS',
            'init_pos': [0, 0, 0.24],
            'interaction_localization': 'rigidbody',
        }, wrench_mode=True, contact_attitude_run=1)
        self.assertEqual(args, '--init-pos 0 0 0.24 --vicon-mode pointcloud ')

    def test_contact_attitude_run_two_is_rigidbody_position_only(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'obj_name': 'FLS', 'init_pos': [0, 0, 0.24],
        }, wrench_mode=True, contact_attitude_run=2)
        self.assertEqual(args, '--obj-name FLS --vicon-mode rigidbody ')
        self.assertNotIn('--vicon-full-pose', args)

    def test_contact_attitude_run_three_is_rigidbody_full_pose(self):
        args = interaction_mocap_args({
            'id': 'lb11', 'obj_name': 'FLS', 'init_pos': [0, 0, 0.24],
        }, wrench_mode=True, contact_attitude_run=3)
        self.assertEqual(
            args,
            '--obj-name FLS --vicon-mode rigidbody --vicon-full-pose ',
        )


if __name__ == '__main__':
    unittest.main()
