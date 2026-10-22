"""Headless checks for default dimensions, module geometry, and generated IDs."""
import sys
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'authoring'))
import blender_addon as addon

addon.register()
try:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    props = bpy.context.scene.drone_props
    assert abs(props.import_max_width - .25) < 1e-6
    assert abs(props.import_max_length - .25) < 1e-6
    assert abs(props.import_cy - 1) < 1e-6
    for index, kind in enumerate(('TYPE_H', 'TYPE_V', 'TYPE_SEMI_H', 'TYPE_RING')):
        bpy.ops.drone.add_drone(drone_type=kind)
        obj = bpy.context.active_object
        assert obj.name == f'lb{index}_{kind.removeprefix("TYPE_")}', obj.name
        assert (Vector(obj.drone_props.light_module_offset) - Vector((.075, 0, .015))).length < 1e-6
        modules = [c for c in obj.children if c.type == 'MESH' and any('led_index' in led for led in c.children)]
        assert modules
        assert all((c.location - Vector((.075, 0, .015))).length < 1e-6 for c in modules)
        assert all(c.location.length == 0 for c in obj.children if c.type == 'CAMERA')
        assert addon.resolve_lb_by_message_id(bpy.context.scene, str(index)) == obj
    assert addon.next_lightbender_name('H', extra=True, reserved=['lb12']) == 'lb13_H_extra'
    for name in ('LightBender_Base', 'lb_extra_H_1', 'lb', 'lbfoo', 'lb12oops', 'lb12.001'):
        assert not addon.lightbender_name_match(name), name
    print('PASS: defaults, mounted geometry, sequential names, and numeric ID lookup')
finally:
    addon.unregister()
