"""Run with Blender --background --factory-startup --python-exit-code 1 --python
tests/blender_body_scale.py.
"""
import math
from pathlib import Path
import sys
import tempfile

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'authoring'))
import blender_addon as addon


def close(actual, expected):
    assert (Vector(actual) - Vector(expected)).length < 1e-6, (actual, expected)


def roots(drone):
    return [c for c in drone.children if c.type == 'MESH'
            and any('led_index' in led for led in c.children)]


def main():
    addon.register()
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        assert scene.drone_props.body_scale == 1.0
        scene.drone_props.model_mode = 'FULL'
        scene.drone_props.body_scale = 1.5
        assert bpy.ops.drone.add_drone() == {'FINISHED'}
        reference, = [c for c in bpy.context.active_object.children
                      if c.get('lightbender_full_model')]
        for kind in ('TYPE_H', 'TYPE_V', 'TYPE_SEMI_H', 'TYPE_SEMI_V', 'TYPE_RING'):
            assert bpy.ops.drone.add_drone(drone_type=kind) == {'FINISHED'}
            drone = bpy.context.active_object
            body, = [c for c in drone.children if c.get('lightbender_full_model')]
            close(body.scale, (1.5,) * 3)
            close(drone.drone_props.light_module_offset, (0.1125, 0, 0.0225))
            assert reference.data is body.data
            drone.location = (1, -2, 3)
            drone.rotation_euler = (0.1, -0.2, math.pi / 2)
            mount = Vector((0.09, -0.03, 0.06))
            drone.drone_props.light_module_offset = mount
            bpy.context.view_layer.update()
            modules = roots(drone)
            components = modules + [led for root in modules for led in root.children]
            sizes = [obj.dimensions.copy() for obj in components]
            mesh_sample = body.data.vertices[0].co.copy()
            for factor in (2.0, 0.25, 1.5):
                drone.drone_props.body_scale = factor
                bpy.context.view_layer.update()
                expected_mount = mount * (factor / 1.5)
                close(drone.drone_props.light_module_offset, expected_mount)
                close(body.scale, (factor,) * 3)
                close(body.location, expected_mount)
                close(body.matrix_world @ mesh_sample,
                      drone.matrix_world @ ((mount / 1.5 + mesh_sample) * factor))
                for root in modules:
                    close(root.matrix_world.translation, body.matrix_world.translation)
                for obj, size in zip(components, sizes):
                    close(obj.dimensions, size)
                close(drone.location, (1, -2, 3))
                camera, = [c for c in drone.children if c.type == 'CAMERA']
                close(camera.location, (0, 0, 0))
                close(reference.scale, (1.5,) * 3)
                close(body.data.vertices[0].co, mesh_sample)

        # Creation scale is ignored in lighting-only mode, including later edits.
        scene.drone_props.model_mode = 'MODULE_ONLY'
        assert bpy.ops.drone.add_drone() == {'FINISHED'}
        module = bpy.context.active_object
        module.drone_props.body_scale = 2.0
        close(module.drone_props.light_module_offset, (0.075, 0, 0.015))
        for root in roots(module):
            close(root.scale, (1, 1, 1))
            close(root.location, (0.075, 0, 0.015))

        # Nonpositive factors clamp; nonfinite values leave the factor intact.
        for value in (0, -1):
            scene.drone_props.body_scale = value
            assert scene.drone_props.body_scale > 0
        scene.drone_props.body_scale = 1.0
        for value in (float('nan'), float('inf')):
            scene.drone_props.body_scale = value
            assert scene.drone_props.body_scale == 1.0

        # Stored factors and geometry survive reopening and subsequent edits.
        drone_name = drone.name
        with tempfile.TemporaryDirectory() as tmp:
            scene.frame_start = scene.frame_end = 1
            scene.drone_props.export_at_keyframes = True
            bpy.ops.object.select_all(action='DESELECT')
            drone.select_set(True)
            bpy.context.view_layer.objects.active = drone
            exported = Path(tmp) / 'scaled.yaml'
            assert bpy.ops.drone.export_yaml(filepath=str(exported)) == {'FINISHED'}
            expected = 'light_module_offset: [0.09, -0.03, 0.06]'
            assert expected in exported.read_text()
            assert expected in addon.build_interaction_yaml_text(scene, [drone])
            path = str(Path(tmp) / 'body-scale.blend')
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path)
            drone = bpy.data.objects[drone_name]
            assert drone.drone_props.body_scale == 1.5
            drone.drone_props.body_scale = 3.0
            close(drone.drone_props.light_module_offset, mount * 2)
            body, = [c for c in drone.children if c.get('lightbender_full_model')]
            close(body.scale, (3,) * 3)
            for root in roots(drone):
                close(root.location, mount * 2)
        print('PASS: body scaling, proportional mounts, unchanged modules, shared mesh isolation, persistence')
    finally:
        addon.unregister()


main()
