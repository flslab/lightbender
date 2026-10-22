"""Headless smoke test for the in-session camera video render option."""

import os
import shutil
import sys
import tempfile
from unittest import mock

import bpy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTHORING_DIR = os.path.join(REPO_ROOT, "authoring")
if AUTHORING_DIR not in sys.path:
    sys.path.insert(0, AUTHORING_DIR)

import blender_addon


def main():
    temp_dir = tempfile.mkdtemp(prefix="lightbender-video-test-")
    blender_addon.register()
    try:
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 1

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        drone = bpy.data.objects.new("lb9900_video_test", None)
        scene.collection.objects.link(drone)
        drone["servo_1"] = "video-test-servo"
        camera_data = bpy.data.cameras.new("Drone_Camera_video_test")
        camera = bpy.data.objects.new("Drone_Camera_video_test", camera_data)
        scene.collection.objects.link(camera)
        camera.parent = drone

        drone.select_set(True)
        bpy.context.view_layer.objects.active = drone
        scene.camera = None

        blend_path = os.path.join(temp_dir, "video_test.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

        props = scene.marker_grid_props
        assert props.render_in_background
        props.render_in_background = False
        props.render_output_dir = temp_dir
        assert bpy.ops.drone.render_drone_cameras() == {'FINISHED'}

        output_path = os.path.join(temp_dir, "render_lb9900_video_test.mp4")
        assert os.path.isfile(output_path)
        assert os.path.getsize(output_path) > 0
        assert scene.camera is None
        assert scene.render.ffmpeg.codec == 'H264'
        assert scene.render.ffmpeg.ffmpeg_preset == 'REALTIME'
        assert scene.render.ffmpeg.constant_rate_factor == 'LOW'
        print("PASS: foreground option renders a fast MP4 and restores scene state")

        # Exercise the generated script in a real child Blender process too.
        os.remove(output_path)
        props.render_in_background = True
        children = []
        original_popen = blender_addon.subprocess.Popen

        def capture_child(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            children.append((process, args[0][-1]))
            return process

        try:
            with mock.patch.object(blender_addon.subprocess, "Popen", side_effect=capture_child):
                assert bpy.ops.drone.render_drone_cameras() == {'FINISHED'}
            assert len(children) == 1
            assert children[0][0].wait(timeout=120) == 0
            assert os.path.getsize(output_path) > 0
            print("PASS: background child renders a valid MP4 with the version-appropriate output API")
        finally:
            for child, script_path in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=15)
                if os.path.exists(script_path):
                    os.remove(script_path)

    finally:
        blender_addon.unregister()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
