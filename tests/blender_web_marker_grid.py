"""Headless Blender check for the web-authored fls-marker-grid format."""

import json
import os
import sys
import tempfile

import bpy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTHORING_DIR = os.path.join(REPO_ROOT, "authoring")
if AUTHORING_DIR not in sys.path:
    sys.path.insert(0, AUTHORING_DIR)

import blender_addon


def close(actual, expected, tolerance=1e-6):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def marker(tile_i, tile_j, local_row, local_col, marker_id, center, spacing):
    position = [
        center[0] + (local_col - 0.5) * spacing,
        center[1] + (0.5 - local_row) * spacing,
        center[2],
    ]
    return {
        "local_row": local_row,
        "local_col": local_col,
        "id": marker_id,
        "global_position": position,
        "global_x": position[0],
        "global_y": position[1],
        "global_z": position[2],
    }


def tile(tile_i, tile_j, signature, origin, tile_size, spacing, events=None):
    center = [
        origin[0] + tile_i * tile_size,
        origin[1] + tile_j * tile_size,
        origin[2],
    ]
    local_order = ((0, 0), (0, 1), (1, 1), (1, 0))
    result = {
        "i": tile_i,
        "j": tile_j,
        "center": center,
        "signature": signature,
        "layout": "centered_2x2",
        "marker_spacing": spacing,
        "markers": [
            marker(tile_i, tile_j, row, col, marker_id, center, spacing)
            for (row, col), marker_id in zip(local_order, signature)
        ],
    }
    if events is not None:
        result["blender_animation"] = {
            "initial_state": "off",
            "events": events,
        }
    return result


def main():
    origin = [1.0, -2.0, 0.25]
    hyper_spacing = 0.2
    tile_size = 0.4
    my_spacing = 0.04
    tiles = [
        tile(-1, 2, [0, 0, 0, 1], origin, tile_size, my_spacing),
        tile(1, -1, [0, 0, 1, 1], origin, tile_size, my_spacing, [
            {"time_s": 0.0, "state": "static"},
            {"time_s": 0.5, "state": "blinking"},
            {"time_s": 1.5, "state": "off"},
            {"time_s": 2.0, "state": "on"},
        ]),
    ]
    grid = {
        "schema": "fls-marker-grid",
        "schema_version": 1,
        "units": "metres",
        "coordinate_frame": "world_FLU",
        "grid_origin": origin,
        "hypergrid": {
            "infinite": True,
            "marker_spacing": hyper_spacing,
            "tile_size": tile_size,
            "marker_diameter": 0.02,
            "markers_per_tile": 4,
            "layout": "centered_2x2",
            "markers_always_on": True,
            "markers_have_ids": False,
        },
        "mygrid": {
            "selected_tile_count": len(tiles),
            "markers_per_tile": 4,
            "layout": "centered_2x2",
            "marker_spacing": my_spacing,
            "marker_diameter": 0.01,
            "tiles": tiles,
        },
        "encoding": {
            "max_ids": 8,
            "payload_bits": 3,
            "delimiter_bits": 4,
            "blink_frequency_hz": 5.0,
        },
        "blender": {
            "scene_fps": 20,
            "emission_strength": 5,
            "min_brightness": 0,
            "max_brightness": 1,
        },
    }

    map_path = None
    blender_addon.register()
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as map_file:
            json.dump(grid, map_file)
            map_path = map_file.name

        scene = bpy.context.scene
        props = scene.marker_grid_props
        props.map_filepath = map_path

        # Selecting a file hydrates the editable panel before generation.
        assert props.loaded_grid_format == "fls-marker-grid v1"
        assert props.loaded_tile_count == 2
        assert close(props.grid_origin, origin)
        assert abs(props.hypergrid_spacing - hyper_spacing) <= 1e-6
        assert abs(props.tile_size - tile_size) <= 1e-6
        assert abs(props.mygrid_spacing - my_spacing) <= 1e-6
        assert abs(props.marker_size - 0.02) <= 1e-6
        assert abs(props.short_range_marker_size - 0.01) <= 1e-6
        assert len(props.animation_events) == 4

        # Edited values and the radio-selected shape are honored by generation.
        edited_origin = (1.2, -2.3, 0.3)
        edited_hyper_spacing = 0.22
        edited_tile_size = 0.44
        edited_my_spacing = 0.05
        props.grid_origin = edited_origin
        props.hypergrid_spacing = edited_hyper_spacing
        props.tile_size = edited_tile_size
        props.mygrid_spacing = edited_my_spacing
        props.marker_size = 0.03
        props.short_range_marker_size = 0.015
        props.marker_shape = 'CYLINDER'
        assert bpy.ops.drone.generate_marker_grid() == {'FINISHED'}

        tiles_found = [obj for obj in scene.objects
                       if obj.name.startswith("MarkerTile_")]
        hyper = [obj for obj in scene.objects
                 if obj.name.startswith("HyperGridMarker_")]
        mygrid = [obj for obj in scene.objects
                  if obj.name.startswith("MyGridMarker_")]
        assert len(tiles_found) == 2
        assert len(hyper) == 8
        assert len(mygrid) == 8
        assert {tuple((obj["tile_i"], obj["tile_j"])) for obj in hyper} == {
            (-1, 2), (1, -1)}

        tile_obj = bpy.data.objects["MarkerTile_-1_2"]
        assert close(tile_obj.location[:2], (0.76, -1.42))
        assert close(tile_obj.dimensions[:2],
                     (edited_tile_size, edited_tile_size))
        assert tile_obj.hide_render
        hyper_obj = bpy.data.objects["HyperGridMarker_-1_2_0_0"]
        assert close(hyper_obj.location, (0.65, -1.31, 0.3))
        assert close(hyper_obj.dimensions, (0.03, 0.03, 0.001))
        assert (hyper_obj["grid_x"], hyper_obj["grid_y"]) == (-3, 4)
        assert not hyper_obj.hide_render and not hyper_obj.hide_viewport

        unanimated = [obj for obj in mygrid
                      if (obj["tile_i"], obj["tile_j"]) == (-1, 2)]
        assert all(close(obj.dimensions, (0.015, 0.015, 0.001))
                   for obj in unanimated)
        assert all(obj.hide_render and obj.hide_viewport for obj in unanimated)
        assert all(obj.animation_data is None for obj in unanimated)

        animated = [obj for obj in mygrid
                    if (obj["tile_i"], obj["tile_j"]) == (1, -1)]
        for frame, hidden in ((1, False), (11, False), (31, True), (41, False)):
            scene.frame_set(frame)
            assert all(obj.hide_render == hidden and obj.hide_viewport == hidden
                       for obj in animated)
        assert all(obj.animation_data is not None for obj in animated)
        blinking_marker = next(obj for obj in animated if obj["marker_id"] == 1)
        assert blinking_marker["blink_pattern"] == "00111110"
        scene.frame_set(41)
        assert abs(blinking_marker.color[0] - 0.0) <= 1e-6
        scene.frame_set(49)
        assert abs(blinking_marker.color[0] - 1.0) <= 1e-6
        assert all(
            point.interpolation == 'CONSTANT'
            for obj in animated
            for curve in blender_addon.get_action_fcurves(obj)
            for point in curve.keyframe_points
        )

        assert props.loaded_grid_format == "fls-marker-grid v1"
        assert props.loaded_tile_count == 2
        assert close(props.grid_origin, edited_origin)
        assert abs(props.hypergrid_spacing - edited_hyper_spacing) <= 1e-6
        assert abs(props.tile_size - edited_tile_size) <= 1e-6
        assert abs(props.mygrid_spacing - edited_my_spacing) <= 1e-6
        assert abs(props.marker_size - 0.03) <= 1e-6
        assert abs(props.short_range_marker_size - 0.015) <= 1e-6
        assert props.payload_size == 3
        assert props.delimiter_size == 4
        assert abs(props.fps - 5.0) <= 1e-6
        assert scene.render.fps == 20
        assert len(props.animation_events) == 4
        assert props.render_in_background
        assert [(event.tile_i, event.tile_j, event.time_s, event.state)
                for event in props.animation_events] == [
                    (1, -1, 0.0, "static"),
                    (1, -1, 0.5, "blinking"),
                    (1, -1, 1.5, "off"),
                    (1, -1, 2.0, "on"),
                ]

        assert bpy.ops.drone.reset_marker_map_values() == {'FINISHED'}
        assert close(props.grid_origin, origin)
        assert abs(props.hypergrid_spacing - hyper_spacing) <= 1e-6
        assert abs(props.tile_size - tile_size) <= 1e-6
        assert abs(props.mygrid_spacing - my_spacing) <= 1e-6
        assert abs(props.marker_size - 0.02) <= 1e-6
        assert abs(props.short_range_marker_size - 0.01) <= 1e-6
        assert props.marker_shape == 'CYLINDER'
        print("PASS: web grid hydration, reset, marker shapes, tiles, events, and UI")
    finally:
        if map_path is not None:
            os.unlink(map_path)
        blender_addon.unregister()


if __name__ == "__main__":
    main()
