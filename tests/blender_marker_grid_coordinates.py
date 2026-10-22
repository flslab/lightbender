"""Headless Blender check for marker-grid and camera coordinate conventions."""

import json
import math
import os
import sys
import tempfile

import bpy
from mathutils import Vector


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTHORING_DIR = os.path.join(REPO_ROOT, "authoring")
if AUTHORING_DIR not in sys.path:
    sys.path.insert(0, AUTHORING_DIR)

import blender_addon


def close(actual, expected, tolerance=1e-6):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def main():
    spacing = 0.2
    short_spacing = 0.04
    short_size = 0.03
    markers = []
    expected_locations = {}
    for row in range(2):
        for col in range(3):
            global_x = (0.5 - row) * spacing
            global_y = (1.0 - col) * spacing
            global_z = 0.3
            marker = {
                "i": row,
                "j": col,
                "id": 0,
                "global_x": global_x,
                "global_y": global_y,
                "global_z": global_z,
            }
            markers.append(marker)
            expected_locations[f"GridMarker_0_{row}_{col}"] = (
                global_x, global_y, global_z)

    short_markers = []
    expected_short_locations = {}
    short_center = (0.0, 0.1, 0.3)
    for local_i in range(2):
        for local_j in range(2):
            marker_id = 8 + local_i * 2 + local_j
            location = (
                short_center[0] + (0.5 - local_i) * short_spacing,
                short_center[1] + (0.5 - local_j) * short_spacing,
                short_center[2],
            )
            marker = {
                "local_i": local_i,
                "local_j": local_j,
                "id": marker_id,
                "global_x": location[0],
                "global_y": location[1],
                "global_z": location[2],
            }
            short_markers.append(marker)
            name = f"ShortRangeMarker_0_0_{local_i}_{local_j}_{marker_id}"
            expected_short_locations[name] = (
                location, local_i, local_j, marker_id)

    grid = {
        "rows": 2,
        "cols": 3,
        "num_ids": 1,
        "min_k": 1,
        "cell_spacing": spacing,
        "grid_origin": [0.1, 0.2, 0.3],
        "grid": [[0, 0, 0], [0, 0, 0]],
        "markers": markers,
        "short_range": {
            "window_size": 2,
            "cell_spacing": short_spacing,
            "marker_size": short_size,
            "tiles": [{
                "i": 0,
                "j": 0,
                "signature": [8, 9, 10, 11],
                "markers": short_markers,
            }],
        },
    }

    map_path = None
    blender_addon.register()
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as map_file:
            json.dump(grid, map_file)
            map_path = map_file.name

        props = bpy.context.scene.marker_grid_props
        props.map_filepath = map_path
        props.short_range_marker_size = 0.0
        assert bpy.ops.drone.generate_marker_grid() == {'FINISHED'}

        # The JSON diameter is the default; a positive UI value overrides it.
        marker = bpy.data.objects.get("ShortRangeMarker_0_0_0_0_8")
        assert marker is not None
        assert close(marker.dimensions, (short_size,) * 3)
        assert marker.hide_render

        timing = props.short_range_timings.add()
        timing.tile_i = 0
        timing.tile_j = 0
        timing.off_time = 1.0
        timing.on_time = 3.0

        # IDs outside the configured payload are rejected before replacement.
        short_markers[0]["id"] = 16
        with open(map_path, "w") as map_file:
            json.dump(grid, map_file)
        try:
            result = bpy.ops.drone.generate_marker_grid()
        except RuntimeError as exc:
            assert "invalid coordinates, IDs" in str(exc)
        else:
            assert result == {'CANCELLED'}
        assert bpy.data.objects.get("ShortRangeMarker_0_0_0_0_8") is marker

        short_markers[0]["id"] = 8
        with open(map_path, "w") as map_file:
            json.dump(grid, map_file)
        override_size = 0.02
        props.short_range_marker_size = override_size
        props.marker_shape = 'CYLINDER'
        bpy.context.scene.frame_set(121)
        assert bpy.ops.drone.generate_marker_grid() == {'FINISHED'}

        for name, expected in expected_locations.items():
            marker = bpy.data.objects.get(name)
            assert marker is not None, f"missing {name}"
            assert close(marker.location, expected), (
                f"{name}: got {tuple(marker.location)}, expected {expected}"
            )

        for name, expected_data in expected_short_locations.items():
            expected, local_i, local_j, marker_id = expected_data
            marker = bpy.data.objects.get(name)
            assert marker is not None, f"missing {name}"
            assert close(marker.location, expected), (
                f"{name}: got {tuple(marker.location)}, expected {expected}"
            )
            assert close(
                marker.dimensions,
                (override_size, override_size,
                 blender_addon.MARKER_CYLINDER_HEIGHT_M))
            assert marker["marker_grid"] == "short_range"
            assert (marker["tile_i"], marker["tile_j"]) == (0, 0)
            assert (marker["local_i"], marker["local_j"]) == (
                local_i, local_j)
            assert marker["marker_id"] == marker_id

            curves = blender_addon.get_action_fcurves(marker)
            hide_render = curves.find("hide_render")
            hide_viewport = curves.find("hide_viewport")
            assert hide_render is not None and hide_viewport is not None
            assert hide_render.evaluate(1) == 0
            assert hide_render.evaluate(121) == 1
            assert hide_render.evaluate(361) == 0
            assert all(kp.interpolation == 'CONSTANT'
                       for kp in hide_render.keyframe_points)
            assert len(hide_render.modifiers) == 0

            color_curves = [fc for fc in curves
                            if fc.data_path == "color"]
            assert color_curves
            assert all(any(mod.type == 'CYCLES' for mod in fc.modifiers)
                       for fc in color_curves)

        assert bpy.ops.drone.add_drone(drone_type='TYPE_RING') == {'FINISHED'}
        drone = bpy.context.active_object
        camera = next(child for child in drone.children
                      if child.name.startswith("Drone_Camera"))
        assert abs(camera.rotation_euler.z + math.pi / 2.0) <= 1e-6
        bpy.context.view_layer.update()
        camera_to_drone = camera.matrix_local.to_3x3()
        assert close(camera_to_drone @ Vector((1, 0, 0)), (0, -1, 0))
        assert close(camera_to_drone @ Vector((0, -1, 0)), (-1, 0, 0))
        assert close(camera_to_drone @ Vector((0, 0, -1)), (0, 0, -1))
        print("PASS: main/short marker placement, timing, payload, and camera basis")
    finally:
        if map_path is not None:
            os.unlink(map_path)
        blender_addon.unregister()


if __name__ == "__main__":
    main()
