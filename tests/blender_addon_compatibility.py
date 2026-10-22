"""Run with Blender 3.0+ --background --factory-startup --python-exit-code 1.

Exercises RNA registration, preferences, supported material nodes, and real
animation operators. On Blender 4.4+ it also puts two objects in different
slots of one Action, so accidentally editing its first slot fails this test.
"""

import os
import sys

import addon_utils
import bpy


AUTHORING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "authoring")
sys.path.insert(0, AUTHORING_DIR)


def make_drone(name):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj["servo_1"] = 0.0
    obj["servo_2"] = 180.0
    return obj


def key_motion(obj, offset):
    for frame, x in ((1, 0.0), (2, 0.1), (3, 0.4)):
        obj.location = (offset + x, 0.0, 1.0)
        obj.keyframe_insert(data_path="location", frame=frame)


def main():
    errors = []
    addon = addon_utils.enable("blender_addon", default_set=True, handle_error=errors.append)
    assert addon is not None, errors
    try:
        preferences = bpy.context.preferences.addons["blender_addon"].preferences
        assert preferences.bl_rna.identifier == "LightBenderAddonPreferences"
        assert addon.LightBenderAddonPreferences.bl_idname == "blender_addon"
        assert hasattr(bpy.context.scene, "drone_props")
        assert hasattr(bpy.context.scene, "marker_grid_props")
        assert addon.get_led_material().node_tree.nodes.get("Emission") is not None
        assert addon.get_structure_material().node_tree is not None

        first = make_drone("lb9901")
        key_motion(first, 0.0)
        second = make_drone("lb9902")
        second.animation_data_create()
        if hasattr(first.animation_data.action, "slots"):
            action = first.animation_data.action
            second.animation_data.action = action
            second.animation_data.action_slot = action.slots.new(id_type='OBJECT', name=second.name)
        key_motion(second, 10.0)

        first_x = addon.get_location_fcurves(first)[0]
        second_x = addon.get_location_fcurves(second)[0]
        assert first_x.as_pointer() != second_x.as_pointer()
        assert abs(first_x.evaluate(1) - 0.0) < 1e-6
        assert abs(second_x.evaluate(1) - 10.0) < 1e-6

        # Exercise removal and keyframe insertion in the object's own slot.
        addon.get_action_fcurves(second).remove(second_x)
        assert addon.get_action_fcurves(second).find("location", index=0) is None
        assert addon.get_location_fcurves(first)[0].as_pointer() == first_x.as_pointer()
        key_motion(second, 10.0)

        bpy.ops.object.select_all(action='DESELECT')
        second.select_set(True)
        bpy.context.view_layer.objects.active = second
        result = bpy.ops.drone.apply_flight_attitude()
        assert result == {'FINISHED'}
        for axis in (0, 1):
            assert addon.get_action_fcurves(second).find("rotation_euler", index=axis) is not None
            assert addon.get_action_fcurves(first).find("rotation_euler", index=axis) is None

        # Empty Actions need a slot/channelbag before curve lookup on 4.4+.
        empty = make_drone("lb9903")
        empty.animation_data_create()
        empty.animation_data.action = bpy.data.actions.new("EmptyCompatibilityAction")
        assert len(addon.get_action_fcurves(empty)) == 0
        empty.keyframe_insert(data_path="location", frame=1)
        assert len(addon.get_location_fcurves(empty)) == 3
        print("PASS: preferences, materials, and Action operations on Blender", bpy.app.version_string)
    finally:
        addon_utils.disable("blender_addon", default_set=True)
        assert not hasattr(bpy.types.Scene, "drone_props")
        assert not hasattr(bpy.types.Object, "drone_props")


if __name__ == "__main__":
    main()
