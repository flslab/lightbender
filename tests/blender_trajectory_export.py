"""Headless smoke test for the per-frame drone trajectory exporter."""

import json
import math
import os
import shutil
import sys
import tempfile

import bpy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTHORING_DIR = os.path.join(REPO_ROOT, "authoring")
if AUTHORING_DIR not in sys.path:
    sys.path.insert(0, AUTHORING_DIR)

import blender_addon


def main():
    temp_dir = tempfile.mkdtemp(prefix="lightbender-trajectory-test-")
    blender_addon.register()
    try:
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 2
        scene.render.fps = 120
        scene.render.fps_base = 1.0

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        drone = bpy.data.objects.new("lb9900_trajectory_test", None)
        scene.collection.objects.link(drone)
        drone["servo_1"] = 0.0
        drone.rotation_mode = 'QUATERNION'
        drone.location = (1.0, 2.0, 3.0)
        drone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        drone.keyframe_insert(data_path="location", frame=1)
        drone.keyframe_insert(data_path="rotation_quaternion", frame=1)
        drone.location = (2.0, 4.0, 6.0)
        half_angle = math.pi / 4.0
        drone.rotation_quaternion = (
            math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)
        )
        drone.keyframe_insert(data_path="location", frame=2)
        drone.keyframe_insert(data_path="rotation_quaternion", frame=2)

        drone.select_set(True)
        bpy.context.view_layer.objects.active = drone
        scene.frame_set(2)

        output_path = os.path.join(temp_dir, "trajectory.json")
        result = bpy.ops.drone.export_trajectory(filepath=output_path)
        assert result == {'FINISHED'}
        assert scene.frame_current == 2

        with open(output_path, 'r', encoding='utf-8') as trajectory_file:
            trajectory = json.load(trajectory_file)
        assert trajectory["schema"] == "fls-drone-trajectory"
        assert trajectory["coordinate_frame"] == "world_FLU"
        assert trajectory["quaternion_order"] == "xyzw"
        assert trajectory["fps"] == 120.0
        assert trajectory["frame_count"] == 2
        assert trajectory["frames"][0]["video_frame"] == 0
        assert trajectory["frames"][0]["position"] == [1.0, 2.0, 3.0]
        expected = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
        actual = trajectory["frames"][1]["quaternion_xyzw"]
        # Blender's mathutils transforms store float32 components.
        assert all(abs(left - right) < 1e-6
                   for left, right in zip(actual, expected))
        print("PASS: exported every frame in world FLU with xyzw quaternions")
    finally:
        blender_addon.unregister()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
