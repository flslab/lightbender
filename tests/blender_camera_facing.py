"""Headless end-to-end test of the standalone camera-facing operator."""
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'authoring'))
import blender_addon


def main():
    blender_addon.register()
    try:
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        scene=bpy.context.scene
        data=bpy.data.cameras.new('View');cam=bpy.data.objects.new('View',data);scene.collection.objects.link(cam)
        cam.location=(2.25,0,.94);cam.rotation_euler=(Vector((0,0,1))-cam.location).to_track_quat('-Z','Y').to_euler();scene.camera=cam
        bpy.ops.drone.add_drone(drone_type='TYPE_H');obj=bpy.context.active_object;obj.name='lb4'
        obj.location=(0,0,1);obj.rotation_euler.z=math.radians(150)
        obj['servo_1']=135;obj['servo_2']=225
        obj.drone_props.light_module_offset=(.06,.01,-.02)
        obj.drone_props.led_mode='EXPRESSION';obj.drone_props.led_formula='[227,253,255]'
        bpy.context.view_layer.update()
        def project(point):
            q=cam.matrix_world.inverted()@point
            return Vector((q.x/-q.z,q.y/-q.z))
        rods=[c for c in obj.children if any('led_index' in led for led in c.children)]
        old_lines=[]
        for rod in rods:
            leds=[c for c in rod.children if 'led_index' in c]
            tip=max(leds,key=lambda c:abs(c['led_index']-25))
            hinge=rod.matrix_world@Vector((.0025,0,0))
            old_lines.append((project(hinge),project(tip.matrix_world.translation)))
        assert bpy.ops.drone.camera_facing()=={'FINISHED'}
        bpy.context.view_layer.update()
        normal=obj.matrix_world.to_3x3()@Vector((1,0,0))
        hinge=obj.matrix_world.translation+obj.matrix_world.to_3x3()@Vector(obj.drone_props.light_module_offset)
        assert normal.dot(cam.location-hinge)>0
        assert obj.drone_props.led_mode=='POINTERS'
        assert len(obj.drone_props.led_pointers)==2
        lo,hi=[p.value for p in obj.drone_props.led_pointers]
        for rod in rods:
            for led in rod.children:
                if 'led_index' not in led or not lo <= led['led_index'] < hi:
                    continue
                q=project(led.matrix_world.translation)
                errors=[]
                for a,b in old_lines:
                    direction=b-a
                    t=(q-a).dot(direction)/direction.length_squared
                    errors.append((q-(a+max(0,min(1,t))*direction)).length)
                assert min(errors)<1e-6, (led.name,errors)
        assert all(math.isfinite(x) for x in obj.location)
        loc=obj.location.copy();yaw=obj.rotation_euler.copy()
        assert bpy.ops.drone.camera_facing()=={'FINISHED'}
        assert (loc-obj.location).length<1e-6
        assert abs(yaw.z-obj.rotation_euler.z)<1e-6
        print('PASS: operator subprocess, mount-aware correction, LED pointers, and idempotence')
    finally:blender_addon.unregister()


if __name__=='__main__':main()
