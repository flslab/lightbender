"""Headless integration checks for the bundled aircraft and both creation paths."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile

import bpy
from mathutils import Vector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'authoring'))
import blender_addon as addon


def close(a, b):
    assert (Vector(a)-Vector(b)).length < 1e-6, (a, b)


def aircraft(drone):
    return [c for c in drone.children if c.get('lightbender_full_model')]


def rods(drone):
    return [c for c in drone.children if any('led_index' in led for led in c.children)]


def check_full(drone):
    assert drone.drone_props.model_mode == 'FULL'
    body, = aircraft(drone)
    assert not body.hide_render
    assert len(body.data.polygons) > 1000
    for rod in rods(drone):
        assert not rod.hide_render
        close(body.matrix_world.translation, rod.matrix_world.translation)
    assert sum('led_index' in c for rod in rods(drone) for c in rod.children) == 50
    return body


def main():
    addon.register()
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        assert scene.drone_props.model_mode == 'MODULE_ONLY'
        # A nonzero cursor must not change mechanical/LED local coordinates.
        scene.cursor.location = (2, -1, 3)
        assert bpy.ops.drone.add_drone(drone_type='TYPE_H') == {'FINISHED'}
        module = bpy.context.active_object
        assert not aircraft(module)
        for rod in rods(module):
            assert rod.hide_render
            close(rod.location, module.drone_props.light_module_offset)
            assert abs(min(v.co.y for v in rod.data.vertices)) < 1e-6
            assert abs(max(v.co.y for v in rod.data.vertices)-0.15) < 1e-6
        scene.drone_props.model_mode = 'FULL'
        bodies = []
        for kind in ('TYPE_H', 'TYPE_V'):
            assert bpy.ops.drone.add_drone(drone_type=kind) == {'FINISHED'}
            drone = bpy.context.active_object
            drone.location = (1, 2, 3)
            drone.rotation_euler = (0.1, -0.2, math.pi/2)
            drone.drone_props.light_module_offset = (0.08, -0.02, 0.03)
            drone['servo_1'] = 120
            drone['servo_2'] = 300
            bpy.context.view_layer.update()
            bodies.append(check_full(drone))
            close(bodies[-1].matrix_world.translation,
                  drone.matrix_world @ Vector(drone.drone_props.light_module_offset))
            camera, = [c for c in drone.children if c.type == 'CAMERA']
            close(camera.location, (0, 0, 0))
        assert bodies[0].data is bodies[1].data
        mesh = bodies[0].data
        # The exported shaft's distal centre is the asset origin; tube setback 3.5 mm.
        metadata = json.loads((REPO/'authoring/assets/lightbender_full.json').read_text())
        assert metadata['hinge_cm'] == [0, -7.35, 1.25]
        assert not any('+rod1:' in p['path'] or '+rod2:' in p['path'] for p in metadata['retained_bodies'])
        assert metadata['source_body_count'] > len(metadata['retained_bodies'])*10
        assert any(abs(v.co.x) < 1e-7 and abs(v.co.y*v.co.y+v.co.z*v.co.z-0.001**2) < 1e-10 for v in mesh.vertices)

        # Run the real Transform and Place operator with a deterministic placement
        # subprocess fixture, isolating model choice from SVG solver dependencies.
        original = (addon.require_authoring_tools, addon.run_authoring, addon.load_authoring_yaml)
        addon.require_authoring_tools = lambda op: True
        addon.load_authoring_yaml = lambda path: json.loads(Path(path).read_text())
        def run_authoring(args):
            if Path(args[0]).name == 'place.py':
                path = args[args.index('--output')+1]
                Path(path).write_text(json.dumps({'points': [
                    {'x':0,'y':0,'z':1,'yaw':0,'angle_1':0,'angle_2':180,
                     'length_1':0.1,'length_2':0.1}]}))
            return SimpleNamespace(returncode=0, stderr='')
        addon.run_authoring = run_authoring
        try:
            svg = tempfile.NamedTemporaryFile(suffix='.svg')
            svg.write(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
            svg.flush()
            scene.drone_props.import_svg_filepath = svg.name
            scene.drone_props.import_manifest_filepath = ''
            for mode in ('FULL', 'MODULE_ONLY'):
                scene.drone_props.model_mode = mode
                scene.drone_props.body_scale = 0.5
                before = set(bpy.data.objects)
                assert bpy.ops.drone.transform_and_place() == {'FINISHED'}
                new = [o for o in set(bpy.data.objects)-before if addon.is_lightbender(o)]
                assert len(new) == 1
                assert bool(aircraft(new[0])) == (mode == 'FULL')
                if mode == 'FULL':
                    assert aircraft(new[0])[0].data is mesh
                    close(aircraft(new[0])[0].scale, (0.5,) * 3)
                    close(new[0].drone_props.light_module_offset, (0.0375, 0, 0.0075))
        finally:
            svg.close()
            addon.require_authoring_tools, addon.run_authoring, addon.load_authoring_yaml = original
        assert bpy.ops.drone.delete_all() == {'FINISHED'}
        assert not any(o.get('lightbender_full_model') for o in scene.objects)
        print('PASS: both modes, shared mesh, hinge alignment, mount changes, cursor independence, transform/place, deletion')
    finally:
        addon.unregister()


main()
