import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))
from interaction_launch import interaction_mocap_args  # noqa: E402


class InteractionLaunchTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
