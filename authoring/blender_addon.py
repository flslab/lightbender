bl_info = {
    "name": "LightBender Swarm Animator",
    "author": "HA",
    "version": (1, 14),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > LightBender",
    "description": "Create light-element drones, animate LEDs with pointers, add position errors, and export YAML",
    "category": "Animation",
}

import bpy
import math
import bmesh
import random
import re
from bpy.props import FloatProperty, StringProperty, EnumProperty, BoolProperty, IntProperty, PointerProperty, \
    CollectionProperty
from bpy.app.handlers import persistent
from mathutils import Vector, Matrix


# ------------------------------------------------------------------------
#    Properties & Data Classes
# ------------------------------------------------------------------------

class LEDPointer(bpy.types.PropertyGroup):
    """Defines a split point on the LED strip and the color following it"""
    # Position on the strip (scaled soft_max for larger rings)
    value: FloatProperty(
        name="Position",
        description="Index location of the pointer",
        default=10.0,
        soft_min=0.0,
        soft_max=200.0
    )
    # Color expression for the segment starting at this pointer
    color_expression: StringProperty(
        name="Color",
        description="Color expression for LEDs after this pointer",
        default="[0, 255, 0]"
    )


class DroneProperties(bpy.types.PropertyGroup):
    drone_type: EnumProperty(
        name="Type",
        description="Light Element Type",
        items=[
            ('TYPE_H', "Type H (Rod)", "Segment 1: 0-180, Segment 2: 180-360"),
            ('TYPE_V', "Type V (Rod)", "Segment 1: 90-270, Segment 2: 270-450"),
            ('TYPE_SEMI_H', "Semicircle H", "Arc: 0-180 (One Servo)"),
            ('TYPE_SEMI_V', "Semicircle V", "Arc: 90-270 (One Servo)"),
            ('TYPE_RING', "Ring", "Full Circle (No Servos)"),
        ],
        default='TYPE_H'
    )

    ring_radius: FloatProperty(
        name="Ring Radius",
        description="Radius of the circular lighting element (m)",
        default=0.15,
        min=0.01,
        unit='LENGTH'
    )

    # LED Control Modes
    led_mode: EnumProperty(
        name="LED Mode",
        items=[
            ('EXPRESSION', "Single Expression", "Use one python formula for all LEDs"),
            ('POINTERS', "Pointers", "Use pointers to define colored segments"),
        ],
        default='EXPRESSION'
    )

    # Mode A: Single Expression
    led_formula: StringProperty(
        name="Formula",
        description="Python expression for LEDs. Vars: i (index), t (time), N (total)",
        default="[255, 255, 255] if i < t*10 else [0,0,0]"
    )

    # Mode B: Pointers
    led_base_color: StringProperty(
        name="Base Color",
        description="Color expression for LEDs before the first pointer",
        default="[0, 0, 0]"
    )

    led_pointers: CollectionProperty(type=LEDPointer)
    active_pointer_index: IntProperty(name="Active Pointer", default=0)

    # Position Error Settings
    error_distance: FloatProperty(
        name="Error Distance",
        description="Distance to randomly offset the lb drones in the YZ plane",
        default=0.05,
        min=0.0,
        unit='LENGTH'
    )

    # Dynamic Drift Settings
    drift_xy: FloatProperty(
        name="Max Drift XY",
        description="Maximum distance to drift in the X and Y axes",
        default=0.05,
        min=0.0,
        unit='LENGTH'
    )

    drift_z: FloatProperty(
        name="Max Drift Z",
        description="Maximum distance to drift in the Z axis (usually lower than XY)",
        default=0.01,
        min=0.0,
        unit='LENGTH'
    )

    drift_speed: FloatProperty(
        name="Drift Wave Scale",
        description="Scale of the noise wave. Higher values mean slower, wider drift",
        default=50.0,
        min=1.0
    )

    # Deconfliction Settings
    deconflict_overlap: FloatProperty(
        name="Overlap Threshold",
        default=0.16,
        min=0.01
    )
    deconflict_downwash: FloatProperty(
        name="Downwash Threshold",
        default=0.16,
        min=0.01
    )
    deconflict_selection: EnumProperty(
        name="Selection Method",
        items=[
            ('BRUTE_FORCE', "Brute Force", ""),
            ('GREEDY_MAX_DEGREE', "Greedy Max Degree", ""),
            ('GREEDY_TOP_Z', "Greedy Top Z", ""),
            ('GREEDY_BOTTOM_Z', "Greedy Bottom Z", ""),
            ('RANDOM', "Random", "")
        ],
        default='GREEDY_MAX_DEGREE'
    )
    deconflict_resolution: EnumProperty(
        name="Resolution Order",
        items=[
            ('MAX_DEGREE', "Max Degree", ""),
            ('TOP_Z', "Top Z", ""),
            ('BOTTOM_Z', "Bottom Z", ""),
            ('RANDOM', "Random", "")
        ],
        default='MAX_DEGREE'
    )
    deconflict_trajectory: EnumProperty(
        name="Trajectory Type",
        items=[
            ('POINT_SPECIFIC', "Line of Sight", ""),
            ('GLOBAL_CENTROID', "Global", "")
        ],
        default='POINT_SPECIFIC'
    )
    deconflict_direction: EnumProperty(
        name="Move Direction",
        items=[
            ('AWAY_FROM_CAMERA', "Away", ""),
            ('TOWARDS_CAMERA', "Towards", ""),
            ('HYBRID', "Hybrid", "")
        ],
        default='HYBRID'
    )

    # Export Settings (Scene Level)
    export_mission_name: StringProperty(
        name="Mission Name",
        description="Name of the mission",
        default="blender_mission"
    )
    
    export_dark_room: BoolProperty(
        name="Dark Room",
        description="Run in dark room mode",
        default=False
    )

    export_rate: FloatProperty(
        name="Export Rate (Hz)",
        description="Frequency of waypoints in the output file (1/delta_t)",
        default=2.0,
        min=0.1
    )

    export_at_keyframes: BoolProperty(
        name="Export Keyframes Only",
        description="Export only at keyframe locations (adds dt to waypoints)",
        default=False
    )


# ------------------------------------------------------------------------
#    Material Helper
# ------------------------------------------------------------------------

def get_led_material():
    """Returns a material that emits light based on Object Color"""
    mat_name = "Drone_LED_Material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Create Nodes
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_emit = nodes.new(type='ShaderNodeEmission')
        node_obj_info = nodes.new(type='ShaderNodeObjectInfo')

        # Link Object Color -> Emission Color -> Surface
        links.new(node_obj_info.outputs['Color'], node_emit.inputs['Color'])
        links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])

        # Set default strength high enough to glow
        node_emit.inputs['Strength'].default_value = 5.0
    return mat


# ------------------------------------------------------------------------
#    Geometry & Setup Operator
# ------------------------------------------------------------------------

class OBJECT_OT_add_drone(bpy.types.Operator):
    """Create a new Drone with Light Elements"""
    bl_idname = "drone.add_drone"
    bl_label = "Add Drone"
    bl_options = {'REGISTER', 'UNDO'}

    drone_type: EnumProperty(
        name="Type",
        items=[
            ('TYPE_H', "Type H", "Segment 1: 0-180, Segment 2: 180-360"),
            ('TYPE_V', "Type V", "Segments 1: 90-270, Segment 2: 270-450"),
            ('TYPE_SEMI_H', "Semicircle H", "Arc: 0-180"),
            ('TYPE_SEMI_V', "Semicircle V", "Arc: 90-270"),
            ('TYPE_RING', "Ring", "Full Circle"),
        ],
        default='TYPE_H'
    )

    def create_ring_geometry(self, context, drone_base, R):
        """Creates the full Ring geometry and LEDs using BMesh"""
        W = 0.003

        # Create Mesh for the Ring Structure
        mesh = bpy.data.meshes.new("Ring_Structure")
        bm = bmesh.new()

        # Scale resolution by radius to maintain smooth curves
        segments = max(32, int(R * 300))

        verts_list = []
        for i in range(segments):
            alpha = (i / segments) * 2 * math.pi

            r_in = R - W / 2
            r_out = R + W / 2
            x_half = W / 2

            y_in = -r_in * math.cos(alpha)
            z_in = -r_in * math.sin(alpha)

            y_out = -r_out * math.cos(alpha)
            z_out = -r_out * math.sin(alpha)

            v1 = bm.verts.new((-x_half, y_out, z_out))
            v2 = bm.verts.new((x_half, y_out, z_out))
            v3 = bm.verts.new((x_half, y_in, z_in))
            v4 = bm.verts.new((-x_half, y_in, z_in))

            verts_list.append([v1, v2, v3, v4])

        # Create continuous faces
        for i in range(segments):
            curr = verts_list[i]
            nxt = verts_list[(i + 1) % segments]

            bm.faces.new((curr[0], curr[1], nxt[1], nxt[0]))  # Outer Face
            bm.faces.new((curr[1], curr[2], nxt[2], nxt[1]))  # Side
            bm.faces.new((curr[2], curr[3], nxt[3], nxt[2]))  # Inner Face
            bm.faces.new((curr[3], curr[0], nxt[0], nxt[3]))  # Side

        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new("Ring_Structure", mesh)
        context.collection.objects.link(obj)
        obj.parent = drone_base

        # LEDs
        led_mat = get_led_material()
        pitch = 0.00624
        circumference = 2 * math.pi * R
        num_leds = max(1, round(circumference / pitch))

        # Store LED count on the drone base for accurate expression evaluation
        drone_base["led_count"] = num_leds

        # 2x2x2 mm cube LEDs
        led_scale = (0.002, 0.002, 0.002)
        r_led_center = R + W / 2 + 0.001
        x_pos = 0

        for i in range(num_leds):
            bpy.ops.mesh.primitive_cube_add(size=1)
            led = context.active_object
            led.name = f"Ring_LED_{i}"
            led["led_index"] = i
            led.data.materials.append(led_mat)
            led.scale = led_scale

            alpha = (i / num_leds) * 2 * math.pi

            y = -r_led_center * math.cos(alpha)
            z = -r_led_center * math.sin(alpha)

            led.location = (x_pos, y, z)
            # Local Z outward
            led.rotation_euler = (alpha + math.pi / 2, 0, 0)
            led.parent = obj

        return obj

    def create_arc_geometry(self, context, drone_base):
        """Creates the Semicircle geometry and LEDs using BMesh"""
        # Specs: R_outer=150mm, R_inner=147mm. Thickness=3mm.
        # Mid Radius = 148.5mm = 0.1485m
        R = 0.1485
        W = 0.003
        THICK = 0.003

        # Shift Factor: Move rotation center to the midpoint of the arc.
        Z_SHIFT = R

        # Create Mesh for the Arc Structure
        mesh = bpy.data.meshes.new("Semicircle_Arc")
        bm = bmesh.new()

        segments = 64
        angle_start = 0
        angle_end = math.pi

        # Build the tube
        prev_verts = []

        for i in range(segments + 1):
            t = i / segments
            alpha = angle_start + t * (angle_end - angle_start)

            r_in = 0.147
            r_out = 0.150
            x_half = W / 2

            y_in = -r_in * math.cos(alpha)
            z_in = -r_in * math.sin(alpha) + Z_SHIFT

            y_out = -r_out * math.cos(alpha)
            z_out = -r_out * math.sin(alpha) + Z_SHIFT

            v1 = bm.verts.new((-x_half, y_out, z_out))
            v2 = bm.verts.new((x_half, y_out, z_out))
            v3 = bm.verts.new((x_half, y_in, z_in))
            v4 = bm.verts.new((-x_half, y_in, z_in))

            current_verts = [v1, v2, v3, v4]

            if prev_verts:
                # Make faces
                bm.faces.new((prev_verts[0], prev_verts[1], v2, v1))  # Outer Face
                bm.faces.new((prev_verts[1], prev_verts[2], v3, v2))  # Side
                bm.faces.new((prev_verts[2], prev_verts[3], v4, v3))  # Inner Face
                bm.faces.new((prev_verts[3], prev_verts[0], v1, v4))  # Side

            prev_verts = current_verts

        # Cap ends
        if len(prev_verts) == 4:
            bm.faces.new((prev_verts[0], prev_verts[1], prev_verts[2], prev_verts[3]))

        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new("Arc_Structure", mesh)
        context.collection.objects.link(obj)
        obj.parent = drone_base

        # LEDs
        led_mat = get_led_material()
        led_scale = (0.0015, 0.004, 0.002)
        r_led_center = 0.151

        # Calculate Angular Offset for 2mm padding
        led_angle_padding = 0.002 / r_led_center
        led_angle_start = angle_start + led_angle_padding
        led_angle_end = angle_end - led_angle_padding

        x_pos = 0
        num_leds = 59

        # Store for evaluator
        drone_base["led_count"] = num_leds

        for i in range(num_leds):
            bpy.ops.mesh.primitive_cube_add(size=1)  # Create Unit Cube
            led = context.active_object
            led.name = f"Arc_LED_{i}"
            led["led_index"] = i
            led.data.materials.append(led_mat)
            led.scale = led_scale

            t = i / (num_leds - 1)
            alpha = led_angle_start + t * (led_angle_end - led_angle_start)

            y = -r_led_center * math.cos(alpha)
            z = -r_led_center * math.sin(alpha) + Z_SHIFT

            led.location = (x_pos, y, z)
            led.rotation_euler = (alpha + math.pi / 2, 0, 0)
            led.parent = obj

        return obj

    def execute(self, context):
        # 1. Create Main Drone Body
        bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
        drone_base = context.active_object
        drone_base.name = "Drone_Base"

        # 2. Add Properties
        drone_base.drone_props.drone_type = self.drone_type

        is_semi = 'SEMI' in self.drone_type
        is_ring = 'RING' in self.drone_type

        # Servos properties (Created for all to satisfy exporter schema, but hidden for RING)
        drone_base["servo_1"] = 0.0
        mgr = drone_base.id_properties_ui("servo_1")
        if not is_ring:
            if 'TYPE_H' in self.drone_type or 'TYPE_SEMI_H' in self.drone_type:
                mgr.update(min=0.0, max=180.0)
            else:
                mgr.update(min=90.0, max=270.0)

        drone_base["servo_2"] = 180.0 if 'TYPE_H' in self.drone_type else 270.0
        if is_ring: drone_base["servo_2"] = 0.0

        mgr = drone_base.id_properties_ui("servo_2")
        if not is_ring:
            if 'TYPE_H' in self.drone_type:
                mgr.update(min=180.0, max=360.0)
            elif 'TYPE_V' in self.drone_type:
                mgr.update(min=270.0, max=450.0)

        # 3. Create Geometry
        led_mat = get_led_material()

        if is_ring:
            R = context.scene.drone_props.ring_radius
            self.create_ring_geometry(context, drone_base, R)

        elif is_semi:
            arc_obj = self.create_arc_geometry(context, drone_base)
            d = arc_obj.driver_add("rotation_euler", 0)
            var = d.driver.variables.new()
            var.name = "angle"
            var.type = 'SINGLE_PROP'
            var.targets[0].id = drone_base
            var.targets[0].data_path = '["servo_1"]'
            d.driver.expression = "-radians(angle)"

        else:
            # --- Rod Logic (Legacy) ---
            drone_base["led_count"] = 50
            ROD_LEN = 0.15
            LED_SIZE = 0.002
            LED_SPACING = 0.00624
            ROD_THICKNESS = 0.003

            # Rod 1
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, ROD_LEN / 2, 0))
            rod1 = context.active_object
            rod1.name = "Rod_1"
            rod1.scale = (ROD_THICKNESS, ROD_LEN, ROD_THICKNESS)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
            rod1.parent = drone_base

            for i in range(26):
                bpy.ops.mesh.primitive_cube_add(size=LED_SIZE)
                led = context.active_object
                led.name = f"R1_LED_{i}"
                led["led_index"] = i
                led.data.materials.append(led_mat)
                dist_from_center = (25 - i) * LED_SPACING
                led.location = (ROD_THICKNESS / 2 + LED_SIZE / 2, dist_from_center, 0)
                led.parent = rod1

            # Rod 2
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, ROD_LEN / 2, 0))
            rod2 = context.active_object
            rod2.name = "Rod_2"
            rod2.scale = (ROD_THICKNESS, ROD_LEN, ROD_THICKNESS)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
            rod2.parent = drone_base

            for i in range(24):
                bpy.ops.mesh.primitive_cube_add(size=LED_SIZE)
                led = context.active_object
                led.name = f"R2_LED_{i}"
                led["led_index"] = 26 + i
                led.data.materials.append(led_mat)
                y_pos = ((i + 1) * LED_SPACING)
                led.location = (ROD_THICKNESS / 2 + LED_SIZE / 2, y_pos, 0)
                led.parent = rod2

            # Drivers
            d = rod1.driver_add("rotation_euler", 0)
            var = d.driver.variables.new()
            var.name = "angle"
            var.type = 'SINGLE_PROP'
            var.targets[0].id = drone_base
            var.targets[0].data_path = '["servo_1"]'
            d.driver.expression = "-radians(angle)"

            d = rod2.driver_add("rotation_euler", 0)
            var = d.driver.variables.new()
            var.name = "angle"
            var.type = 'SINGLE_PROP'
            var.targets[0].id = drone_base
            var.targets[0].data_path = '["servo_2"]'
            d.driver.expression = "-radians(angle)"

        # Select Base
        bpy.ops.object.select_all(action='DESELECT')
        drone_base.select_set(True)
        context.view_layer.objects.active = drone_base

        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Pointer List UI Operators
# ------------------------------------------------------------------------

class DRONE_OT_add_pointer(bpy.types.Operator):
    """Add a new LED Pointer"""
    bl_idname = "drone.add_pointer"
    bl_label = "Add Pointer"

    def execute(self, context):
        obj = context.active_object
        if not obj: return {'CANCELLED'}

        # Add new item
        item = obj.drone_props.led_pointers.add()
        # Default value slightly offset to avoid overlap
        item.value = 10.0 + (len(obj.drone_props.led_pointers) * 5)
        item.color_expression = "[255, 0, 0]"

        obj.drone_props.active_pointer_index = len(obj.drone_props.led_pointers) - 1
        return {'FINISHED'}


class DRONE_OT_remove_pointer(bpy.types.Operator):
    """Remove selected LED Pointer"""
    bl_idname = "drone.remove_pointer"
    bl_label = "Remove Pointer"

    def execute(self, context):
        obj = context.active_object
        if not obj: return {'CANCELLED'}

        idx = obj.drone_props.active_pointer_index
        obj.drone_props.led_pointers.remove(idx)

        if idx >= len(obj.drone_props.led_pointers):
            obj.drone_props.active_pointer_index = max(0, len(obj.drone_props.led_pointers) - 1)

        return {'FINISHED'}


class UL_DronePointerList(bpy.types.UIList):
    """List representation of Pointers"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # Draw layout: Position Value | Color Expression
        split = layout.split(factor=0.3)
        split.prop(item, "value", text="", emboss=False)  # Editable number
        split.prop(item, "color_expression", text="", emboss=False)


# ------------------------------------------------------------------------
#    Position Error Feature
# ------------------------------------------------------------------------

def revert_drone_error(obj):
    """Reverts applied positional error from object location and fcurves"""
    if "applied_error_y" in obj and "applied_error_z" in obj:
        dy = obj["applied_error_y"]
        dz = obj["applied_error_z"]

        obj.location.y -= dy
        obj.location.z -= dz

        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                if fc.data_path == "location":
                    if fc.array_index == 1:  # Y axis
                        for kp in fc.keyframe_points:
                            kp.co.y -= dy
                            kp.handle_left.y -= dy
                            kp.handle_right.y -= dy
                        fc.update()
                    elif fc.array_index == 2:  # Z axis
                        for kp in fc.keyframe_points:
                            kp.co.y -= dz
                            kp.handle_left.y -= dz
                            kp.handle_right.y -= dz
                        fc.update()

        del obj["applied_error_y"]
        del obj["applied_error_z"]
        return True
    return False


class DRONE_OT_apply_error(bpy.types.Operator):
    """Apply a random YZ offset to lb[id] drones based on input distance"""
    bl_idname = "drone.apply_error"
    bl_label = "Apply Positional Error"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dist = context.scene.drone_props.error_distance
        applied_count = 0

        for obj in context.scene.objects:
            # Match names like lb1, lb2, lb007, etc.
            if re.match(r"^lb\d+", obj.name):
                # Ensure we reset any previous error before applying a new one
                # so the distance remains exactly as requested from original path
                revert_drone_error(obj)

                theta = random.uniform(0, 2 * math.pi)
                dy = dist * math.cos(theta)
                dz = dist * math.sin(theta)

                # Offset Base Location
                obj.location.y += dy
                obj.location.z += dz

                # Offset Animation F-Curves if they exist
                if obj.animation_data and obj.animation_data.action:
                    for fc in obj.animation_data.action.fcurves:
                        if fc.data_path == "location":
                            if fc.array_index == 1:  # Y
                                for kp in fc.keyframe_points:
                                    kp.co.y += dy
                                    kp.handle_left.y += dy
                                    kp.handle_right.y += dy
                                fc.update()
                            elif fc.array_index == 2:  # Z
                                for kp in fc.keyframe_points:
                                    kp.co.y += dz
                                    kp.handle_left.y += dz
                                    kp.handle_right.y += dz
                                fc.update()

                # Save the offsets applied so they can be reverted
                obj["applied_error_y"] = dy
                obj["applied_error_z"] = dz
                applied_count += 1

        # Force a viewport update
        context.view_layer.update()
        self.report({'INFO'}, f"Applied positional error to {applied_count} lb drones.")
        return {'FINISHED'}


class DRONE_OT_reset_error(bpy.types.Operator):
    """Reset lb[id] drones to their pre-error coordinates"""
    bl_idname = "drone.reset_error"
    bl_label = "Reset Positional Error"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        reset_count = 0
        for obj in context.scene.objects:
            if re.match(r"^lb\d+", obj.name):
                if revert_drone_error(obj):
                    reset_count += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Reset positional error for {reset_count} lb drones.")
        return {'FINISHED'}


class DRONE_OT_apply_drift(bpy.types.Operator):
    """Apply continuous noise-based drift to lb[id] drones"""
    bl_idname = "drone.apply_drift"
    bl_label = "Apply Dynamic Drift"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.drone_props
        applied_count = 0

        for obj in context.scene.objects:
            if re.match(r"^lb\d+", obj.name):
                # Ensure animation data exists
                if not obj.animation_data:
                    obj.animation_data_create()
                if not obj.animation_data.action:
                    obj.animation_data.action = bpy.data.actions.new(name=f"{obj.name}_Action")

                # Apply noise to X, Y, Z location channels
                for i in range(3):
                    fc = obj.animation_data.action.fcurves.find('location', index=i)

                    # We need a keyframe for the fcurve to exist
                    if not fc:
                        obj.keyframe_insert(data_path="location", index=i, frame=context.scene.frame_current)
                        fc = obj.animation_data.action.fcurves.find('location', index=i)

                    # Remove any existing noise modifiers so we don't stack them
                    for mod in list(fc.modifiers):
                        if mod.type == 'NOISE':
                            fc.modifiers.remove(mod)

                    # Add new noise modifier
                    noise = fc.modifiers.new('NOISE')
                    noise.scale = props.drift_speed
                    noise.phase = random.uniform(0, 1000)  # Randomize phase so they drift independently

                    if i == 2:  # Z axis
                        noise.strength = props.drift_z
                    else:  # X and Y axes
                        noise.strength = props.drift_xy

                applied_count += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Applied dynamic drift to {applied_count} lb drones.")
        return {'FINISHED'}


class DRONE_OT_reset_drift(bpy.types.Operator):
    """Remove drift noise modifiers from lb[id] drones"""
    bl_idname = "drone.reset_drift"
    bl_label = "Reset Dynamic Drift"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        reset_count = 0
        for obj in context.scene.objects:
            if re.match(r"^lb\d+", obj.name):
                if obj.animation_data and obj.animation_data.action:
                    cleared = False
                    for i in range(3):
                        fc = obj.animation_data.action.fcurves.find('location', index=i)
                        if fc:
                            for mod in list(fc.modifiers):
                                if mod.type == 'NOISE':
                                    fc.modifiers.remove(mod)
                                    cleared = True
                    if cleared:
                        reset_count += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Removed drift from {reset_count} lb drones.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Deconfliction Feature
# ------------------------------------------------------------------------

def revert_deconflict_stagger(obj):
    restored = False
    if "deconflict_orig_x" in obj and "deconflict_orig_y" in obj and "deconflict_orig_z" in obj:
        obj.location.x = obj["deconflict_orig_x"]
        obj.location.y = obj["deconflict_orig_y"]
        obj.location.z = obj["deconflict_orig_z"]
        del obj["deconflict_orig_x"]
        del obj["deconflict_orig_y"]
        del obj["deconflict_orig_z"]
        restored = True
        
    if "deconflict_scale_factor" in obj:
        scale_factor = obj["deconflict_scale_factor"]
        if scale_factor > 0.0 and scale_factor != 1.0:
            inv_scale = 1.0 / scale_factor
            for ptr in obj.drone_props.led_pointers:
                ptr.value = 25.0 + (ptr.value - 25.0) * inv_scale
                
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    if "led_pointers" in fc.data_path and fc.data_path.endswith(".value"):
                        for kp in fc.keyframe_points:
                            kp.co.y = 25.0 + (kp.co.y - 25.0) * inv_scale
                            kp.handle_left.y = 25.0 + (kp.handle_left.y - 25.0) * inv_scale
                            kp.handle_right.y = 25.0 + (kp.handle_right.y - 25.0) * inv_scale
                        fc.update()
        del obj["deconflict_scale_factor"]
        restored = True
        
    return restored

class DRONE_OT_deconflict_stagger(bpy.types.Operator):
    """Call deconflict.py to stagger drones based on camera perspective"""
    bl_idname = "drone.deconflict_stagger"
    bl_label = "Stagger"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import os
        import subprocess
        import re
        import math
        
        scene = context.scene
        props = scene.drone_props
        
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "No camera found in scene. Necessary for deconfliction.")
            return {'CANCELLED'}
            
        cam_pos = cam.matrix_world.translation
        
        # Gather drones
        points = []
        drones = []
        for obj in scene.objects:
            m = re.match(r"^lb(\d+)", obj.name)
            if m:
                d_id = int(m.group(1))
                drones.append(obj)
                
                # Save original if not already saved
                if "deconflict_orig_x" not in obj:
                    obj["deconflict_orig_x"] = obj.location.x
                    obj["deconflict_orig_y"] = obj.location.y
                    obj["deconflict_orig_z"] = obj.location.z
                
                # Angles
                a1 = obj.get("servo_1", 0.0)
                a2 = obj.get("servo_2", 0.0)
                yaw = math.degrees(obj.matrix_world.to_euler('XYZ').z)
                
                points.append({
                    'id': d_id,
                    'x': round(obj.location.x, 4),
                    'y': round(obj.location.y, 4),
                    'z': round(obj.location.z, 4),
                    'length_1': 0.08,
                    'length_2': 0.08,
                    'angle_1': a1,
                    'angle_2': a2,
                    'yaw': yaw,
                    'max_length_limit': 0.16
                })
        
        if not points:
            self.report({'WARNING'}, "No lb# drones found.")
            return {'CANCELLED'}
            
        import tempfile
        try:
            addon_dir = os.path.dirname(os.path.realpath(__file__))
        except NameError:
            addon_dir = "/Users/hamed/Documents/Holodeck/lightbender/authoring"

        input_yaml = os.path.join(tempfile.gettempdir(), "temp_deconflict_in.yaml")
        output_yaml = os.path.join(tempfile.gettempdir(), "temp_deconflict_out.yaml")
        
        deconflict_script = os.path.join(addon_dir, "deconflict.py")
        if not os.path.exists(deconflict_script):
            alt_path = "/Users/hamed/Documents/Holodeck/lightbender/authoring/deconflict.py"
            if os.path.exists(alt_path):
                deconflict_script = alt_path
            else:
                self.report({'ERROR'}, f"Cannot find deconflict.py at {deconflict_script}")
                return {'CANCELLED'}
        
        # Write input YAML manually
        try:
            with open(input_yaml, 'w') as f:
                f.write("points:\n")
                for p in points:
                    f.write(f"- id: {p['id']}\n")
                    f.write(f"  x: {p['x']}\n")
                    f.write(f"  y: {p['y']}\n")
                    f.write(f"  z: {p['z']}\n")
                    f.write(f"  length_1: {p['length_1']}\n")
                    f.write(f"  length_2: {p['length_2']}\n")
                    f.write(f"  angle_1: {p['angle_1']}\n")
                    f.write(f"  angle_2: {p['angle_2']}\n")
                    f.write(f"  yaw: {p['yaw']}\n")
                    f.write(f"  max_length_limit: {p['max_length_limit']}\n")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to write yaml: {e}")
            return {'CANCELLED'}
            
        cmd_str = (
            f"python3 '{deconflict_script}' "
            f"--input_file '{input_yaml}' "
            f"--output_file '{output_yaml}' "
            f"--threshold_overlap {props.deconflict_overlap} "
            f"--threshold_downwash {props.deconflict_downwash} "
            f"--selection_method {props.deconflict_selection} "
            f"--resolution_order {props.deconflict_resolution} "
            f"--trajectory_type {props.deconflict_trajectory} "
            f"--move_direction {props.deconflict_direction} "
            f"--camera_pos {round(cam_pos.x, 4)} {round(cam_pos.y, 4)} {round(cam_pos.z, 4)} "
            f"--no_viz"
        )
        
        try:
            # Use zsh login shell to ensure user's actual Python environment (with pandas, etc.) is loaded
            res = subprocess.run(["/bin/zsh", "-l", "-c", cmd_str], check=False, capture_output=True, text=True)
            if res.returncode != 0:
                short_err = res.stderr.strip().split('\n')[-1] if res.stderr else "Unknown error"
                self.report({'ERROR'}, f"Deconflict failed: {short_err}")
                print(f"Deconflict error:\n{res.stderr}")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run script: {e}")
            return {'CANCELLED'}
            
        if not os.path.exists(output_yaml):
            self.report({'ERROR'}, "No output yaml generated by deconflict.py")
            return {'CANCELLED'}
            
        # Parse output safely
        out_points = {}
        curr_p = {}
        try:
            with open(output_yaml, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('- id:'):
                        if curr_p and 'id' in curr_p:
                            out_points[curr_p['id']] = curr_p
                        try:
                            curr_p = {'id': int(stripped.split(':')[1].strip())}
                        except ValueError:
                            curr_p = {}
                    elif stripped.startswith('x:') and curr_p:
                        curr_p['x'] = float(stripped.split(':')[1].strip())
                    elif stripped.startswith('y:') and curr_p:
                        curr_p['y'] = float(stripped.split(':')[1].strip())
                    elif stripped.startswith('z:') and curr_p:
                        curr_p['z'] = float(stripped.split(':')[1].strip())
                    elif stripped.startswith('scale_factor:') and curr_p:
                        curr_p['scale_factor'] = float(stripped.split(':')[1].strip())
            if curr_p and 'id' in curr_p:
                out_points[curr_p['id']] = curr_p
        except Exception as e:
            self.report({'ERROR'}, f"Failed to parse output: {e}")
            return {'CANCELLED'}
            
        # Apply positions
        applied = 0
        for obj in drones:
            m = re.match(r"^lb(\d+)", obj.name)
            if m:
                d_id = int(m.group(1))
                if d_id in out_points:
                    p = out_points[d_id]
                    obj.location.x = p.get('x', obj.location.x)
                    obj.location.y = p.get('y', obj.location.y)
                    obj.location.z = p.get('z', obj.location.z)
                    
                    scale_factor = p.get('scale_factor', 1.0)
                    if scale_factor != 1.0:
                        obj["deconflict_scale_factor"] = scale_factor
                        # Scale pointers based on central index 25
                        for ptr in obj.drone_props.led_pointers:
                            ptr.value = 25.0 + (ptr.value - 25.0) * scale_factor
                            
                        # Scale pointer keyframes
                        if obj.animation_data and obj.animation_data.action:
                            for fc in obj.animation_data.action.fcurves:
                                if "led_pointers" in fc.data_path and fc.data_path.endswith(".value"):
                                    for kp in fc.keyframe_points:
                                        kp.co.y = 25.0 + (kp.co.y - 25.0) * scale_factor
                                        kp.handle_left.y = 25.0 + (kp.handle_left.y - 25.0) * scale_factor
                                        kp.handle_right.y = 25.0 + (kp.handle_right.y - 25.0) * scale_factor
                                    fc.update()
                    
                    applied += 1
                    
        context.view_layer.update()
        self.report({'INFO'}, f"Staggered {applied} drones.")
        
        # Cleanup
        try:
            if os.path.exists(input_yaml): os.remove(input_yaml)
            if os.path.exists(output_yaml): os.remove(output_yaml)
        except Exception:
            pass
            
        return {'FINISHED'}


class DRONE_OT_deconflict_reset(bpy.types.Operator):
    """Reset drones to pre-stagger positions"""
    bl_idname = "drone.deconflict_reset"
    bl_label = "Reset Stagger"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import re
        reset_count = 0
        for obj in context.scene.objects:
            if re.match(r"^lb\d+", obj.name):
                if revert_deconflict_stagger(obj):
                    reset_count += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Reset stagger for {reset_count} drones.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Animation Handler (Live Updates)
# ------------------------------------------------------------------------

def evaluate_leds(scene):
    """Iterates all drones and updates LED colors based on formula"""
    fps = scene.render.fps
    # Avoid div by zero
    t = (scene.frame_current - scene.frame_start) / fps if fps else 0
    if t < 0: t = 0

    # 1. Find all Drone Bases
    drones = [obj for obj in scene.objects if "servo_1" in obj and "servo_2" in obj]

    for drone in drones:
        props = drone.drone_props
        mode = props.led_mode

        # Determine total LED count assigned during creation (fallback 50)
        N = drone.get("led_count", 50)

        # Prepare Pointer Data if in Pointer Mode
        sorted_pointers = []
        base_expr = ""

        if mode == 'POINTERS':
            base_expr = props.led_base_color
            raw_pointers = []
            for ptr in props.led_pointers:
                raw_pointers.append({
                    'val': ptr.value,
                    'expr': ptr.color_expression
                })
            # Sort by position value
            sorted_pointers = sorted(raw_pointers, key=lambda x: x['val'])
        else:
            # Expression Mode
            base_expr = props.led_formula

        # Traverse children to find LEDs (Recursively to handle hierarchy depth)
        def recursive_update(obj_list):
            for child in obj_list:
                if "led_index" in child:
                    update_one_led(child, t, mode, base_expr, sorted_pointers, N)

                if child.children:
                    recursive_update(child.children)

        recursive_update(drone.children)


def update_one_led(led, t, mode, base_expr, sorted_pointers, N):
    i = led["led_index"]

    # Determine formula
    active_formula = ""

    if mode == 'EXPRESSION':
        active_formula = base_expr
    elif mode == 'POINTERS':
        active_formula = base_expr
        for ptr in sorted_pointers:
            if i >= ptr['val']:
                active_formula = ptr['expr']
            else:
                break

    # Evaluate
    ctx = {"i": i, "t": t, "N": N, "math": math}

    try:
        raw_color = eval(active_formula, {}, ctx)

        if isinstance(raw_color, (list, tuple)) and len(raw_color) >= 3:
            r, g, b = raw_color[0], raw_color[1], raw_color[2]
            if max(r, g, b) > 1.0:
                r /= 255.0
                g /= 255.0
                b /= 255.0
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            led.color = (r, g, b, 1.0)
    except Exception:
        pass


@persistent
def update_leds_handler(scene):
    evaluate_leds(scene)


class DRONE_OT_force_update(bpy.types.Operator):
    """Force update of LED colors"""
    bl_idname = "drone.force_update_leds"
    bl_label = "Update LEDs"

    def execute(self, context):
        evaluate_leds(context.scene)
        # Trigger redraw of viewports
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Export Operator
# ------------------------------------------------------------------------

class EXPORT_OT_drone_yaml(bpy.types.Operator):
    """Export Drone Animation to YAML"""
    bl_idname = "drone.export_yaml"
    bl_label = "Export YAML"

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        drones_to_export = [obj for obj in context.selected_objects if "servo_1" in obj and "servo_2" in obj]

        if not drones_to_export:
            self.report({'ERROR'}, "No Drones selected")
            return {'CANCELLED'}

        fps = scene.render.fps

        # Export Settings
        use_keyframes = scene.drone_props.export_at_keyframes

        if use_keyframes:
            # Mode A: Keyframes Only
            delta_t = 0.0
        else:
            # Mode B: Fixed Rate
            export_hz = scene.drone_props.export_rate
            if export_hz <= 0: export_hz = 1.0
            step = fps / export_hz
            delta_t = 1.0 / export_hz

        start_frame = scene.frame_start
        end_frame = scene.frame_end

        output_lines = []
        mission_name = scene.drone_props.export_mission_name.strip()
        if not mission_name:
            mission_name = scene.name.replace(' ', '_')
            
        output_lines.append(f"name: {mission_name}")
        output_lines.append("drones:")

        for drone in drones_to_export:
            output_lines.append(f"  {drone.name}:")
            waypoints = []
            servos = []
            pointer_data = []

            # --- Frame Collection ---
            frames_to_export = []
            if use_keyframes:
                keys = set()
                if drone.animation_data and drone.animation_data.action:
                    for fcurve in drone.animation_data.action.fcurves:
                        for kp in fcurve.keyframe_points:
                            if start_frame <= kp.co.x <= end_frame:
                                keys.add(kp.co.x)
                keys.add(start_frame)
                keys.add(end_frame)
                frames_to_export = sorted(list(keys))
            else:
                curr = float(start_frame)
                while curr <= end_frame:
                    frames_to_export.append(curr)
                    curr += step

            # --- Data Collection ---
            prev_time_sec = 0.0

            for i, frame in enumerate(frames_to_export):
                scene.frame_set(int(frame))
                current_time_sec = (frame - start_frame) / fps

                if i == 0:
                    dt_val = 0.0
                else:
                    dt_val = current_time_sec - prev_time_sec
                prev_time_sec = current_time_sec

                loc = drone.matrix_world.translation
                x, y, z = round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)
                rot_euler = drone.matrix_world.to_euler('XYZ')
                yaw = round(rot_euler.z, 4)

                s1 = round(drone["servo_1"], 2)
                s2 = round(drone["servo_2"], 2)

                # Capture Pointers
                ptrs = [round(p.value, 2) for p in drone.drone_props.led_pointers]
                pointer_data.append(f"[{', '.join(map(str, ptrs))}]")

                if use_keyframes:
                    waypoints.append(f"[{x}, {y}, {z}, {yaw}, {round(dt_val, 4)}]")
                    servos.append(f"[{s1}, {s2}]")
                else:
                    waypoints.append(f"[{x}, {y}, {z}, {yaw}]")
                    servos.append(f"[{s1}, {s2}]")

            # --- Write Block ---
            output_lines.append(f"    target: {waypoints[0]}")
            output_lines.append(f"    waypoints: [{', '.join(waypoints)}]")
            output_lines.append(f"    delta_t: {delta_t}")
            output_lines.append(f"    iterations: 1")
            output_lines.append(f"    params:")
            output_lines.append(f"      linear: true")
            output_lines.append(f"      relative: false")
            output_lines.append(f"    servos: [{', '.join(servos)}]")
            output_lines.append(f"    pointers: [{', '.join(pointer_data)}]")

            # --- LED Formula Generation ---
            props = drone.drone_props
            final_formula = ""

            if props.led_mode == 'EXPRESSION':
                final_formula = props.led_formula
            elif props.led_mode == 'POINTERS':
                # Compile pointers
                raw_ptrs = []
                for idx, ptr in enumerate(props.led_pointers):
                    raw_ptrs.append({'id': f"p{idx}", 'v': ptr.value, 'c': ptr.color_expression})
                raw_ptrs.sort(key=lambda x: x['v'])

                if not raw_ptrs:
                    final_formula = props.led_base_color
                else:
                    current_str = f"({raw_ptrs[-1]['c']})"
                    for j in range(len(raw_ptrs) - 2, -1, -1):
                        p_curr = raw_ptrs[j]
                        p_next = raw_ptrs[j + 1]
                        current_str = f"({p_curr['c']}) if i < {p_next['id']} else {current_str}"
                    p0 = raw_ptrs[0]
                    final_formula = f"({props.led_base_color}) if i < {p0['id']} else {current_str}"

            output_lines.append(f"    led:")
            output_lines.append(f"      mode: \"expression\"")
            output_lines.append(f"      rate: 50")
            safe_formula = final_formula.replace('"', '\\"')
            output_lines.append(f"      formula: \"{safe_formula}\"")

        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            self.report({'INFO'}, f"Exported to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"File Write Error: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}


class EXPORT_OT_export_and_illuminate(bpy.types.Operator):
    """Export Drone Animation to YAML and Run Orchestrator"""
    bl_idname = "drone.export_and_illuminate"
    bl_label = "Export and Illuminate"

    def execute(self, context):
        import os
        import subprocess
        import re

        scene = context.scene
        props = scene.drone_props
        
        # Select all lb# drones
        drones_to_export = []
        for obj in scene.objects:
            if re.match(r"^lb\d+", obj.name) and "servo_1" in obj and "servo_2" in obj:
                drones_to_export.append(obj)
                
        if not drones_to_export:
            self.report({'ERROR'}, "No lb# drones found in scene")
            return {'CANCELLED'}

        # Select the drones
        bpy.ops.object.select_all(action='DESELECT')
        for obj in drones_to_export:
            obj.select_set(True)

        target_yaml = "/Users/hamed/Documents/Holodeck/fls-cf-offboard-controller/mission/blender_mission.yaml"
        target_script = "/Users/hamed/Documents/Holodeck/fls-cf-offboard-controller/orchestrator.py"

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_yaml), exist_ok=True)

        res = bpy.ops.drone.export_yaml('EXEC_DEFAULT', filepath=target_yaml)
        if 'FINISHED' not in res:
            self.report({'ERROR'}, "Failed to generate YAML")
            return {'CANCELLED'}

        cmd_str = f"python3 '{target_script}' --illumination --skip-confirm"
        if props.export_dark_room:
            cmd_str += " --dark"

        # Open Mac Terminal to run the orchestrator so the user can monitor progress
        apple_script = f'''
        tell application "Terminal"
            do script "cd \\"{os.path.dirname(target_script)}\\" && {cmd_str}"
            activate
        end tell
        '''
            
        try:
            subprocess.Popen(["osascript", "-e", apple_script])
            self.report({'INFO'}, "Exported and started orchestrator")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run orchestrator: {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    UI Panel
# ------------------------------------------------------------------------

class VIEW3D_PT_drone_swarm(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "LightBender Controls"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        # --- Drone Creation ---
        layout.label(text="Add Drone:")
        row = layout.row()
        row.prop(scene.drone_props, "drone_type", text="")

        # Display Radius property strictly before creation if Ring is selected
        if scene.drone_props.drone_type == 'TYPE_RING':
            layout.prop(scene.drone_props, "ring_radius")

        row.operator("drone.add_drone", text="Create").drone_type = scene.drone_props.drone_type

        layout.separator()

        # --- Selected Drone Controls ---
        if obj and "servo_1" in obj:
            props = obj.drone_props
            is_semi = 'SEMI' in props.drone_type
            is_ring = 'RING' in props.drone_type

            layout.label(text=f"Selected: {obj.name}")
            col = layout.column(align=True)

            # Hide Actuation controls for Ring types
            if is_ring:
                col.label(text="Ring (No Actuation)", icon='MESH_CIRCLE')
            else:
                col.label(text="Servo Angles (deg):")
                # Show Angle 1
                col.prop(obj, '["servo_1"]', text="Angle 1" if not is_semi else "Arc Angle")

                # Hide Angle 2 if Semicircle
                if not is_semi:
                    col.prop(obj, '["servo_2"]', text="Angle 2")

            layout.separator()
            layout.label(text="LED Configuration:")
            layout.prop(props, "led_mode", text="Mode")

            if props.led_mode == 'EXPRESSION':
                layout.prop(props, "led_formula", text="")
            else:
                # Pointer Mode UI
                box = layout.box()
                box.label(text="Base Color (Start):")
                box.prop(props, "led_base_color", text="")

                box.label(text="Pointers (Split Points):")

                row = box.row()
                row.template_list("UL_DronePointerList", "", props, "led_pointers", props, "active_pointer_index")

                col = row.column(align=True)
                col.operator("drone.add_pointer", icon='ADD', text="")
                col.operator("drone.remove_pointer", icon='REMOVE', text="")

                # Hint for usage
                box.label(text="Pos | Right-Side Color", icon='INFO')

            layout.separator()
            layout.operator("drone.force_update_leds", text="Test/Update LEDs", icon='LIGHT')
        else:
            layout.label(text="Select a drone to animate", icon='INFO')

        layout.separator()

        # --- Error & Drift Simulator ---
        layout.label(text="Error & Drift Simulators:", icon='MOD_NOISE')
        box = layout.box()

        box.label(text="Static Error (YZ Plane):")
        box.prop(scene.drone_props, "error_distance")
        row = box.row(align=True)
        row.operator("drone.apply_error", text="Apply Static Error")
        row.operator("drone.reset_error", text="Reset Static")

        box.separator()

        box.label(text="Dynamic Drift (XYZ):")
        box.prop(scene.drone_props, "drift_xy", text="Max Drift XY")
        box.prop(scene.drone_props, "drift_z", text="Max Drift Z")
        box.prop(scene.drone_props, "drift_speed", text="Wave Scale (Slower=Higher)")
        row = box.row(align=True)
        row.operator("drone.apply_drift", text="Apply Drift")
        row.operator("drone.reset_drift", text="Reset Drift")

        layout.separator()

        # --- Deconfliction Simulator ---
        layout.label(text="Stagger:", icon='PARTICLES')
        box = layout.box()
        
        box.prop(scene.drone_props, "deconflict_overlap")
        box.prop(scene.drone_props, "deconflict_downwash")
        box.prop(scene.drone_props, "deconflict_selection")
        box.prop(scene.drone_props, "deconflict_resolution")
        box.prop(scene.drone_props, "deconflict_trajectory")
        box.prop(scene.drone_props, "deconflict_direction")
        
        row = box.row(align=True)
        row.operator("drone.deconflict_stagger", text="Stagger")
        row.operator("drone.deconflict_reset", text="Reset Stagger")
        
        layout.separator()

        # --- Export Config ---
        layout.label(text="Export Config:")
        layout.prop(scene.drone_props, "export_mission_name")
        layout.prop(scene.drone_props, "export_dark_room")
        layout.prop(scene.drone_props, "export_at_keyframes")

        if not scene.drone_props.export_at_keyframes:
            layout.prop(scene.drone_props, "export_rate")

        layout.operator("drone.export_yaml", text="Export to YAML", icon='EXPORT')
        layout.operator("drone.export_and_illuminate", text="Export and Illuminate", icon='PLAY')


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    LEDPointer,
    DroneProperties,
    OBJECT_OT_add_drone,
    DRONE_OT_add_pointer,
    DRONE_OT_remove_pointer,
    UL_DronePointerList,
    DRONE_OT_apply_error,
    DRONE_OT_reset_error,
    DRONE_OT_apply_drift,
    DRONE_OT_reset_drift,
    DRONE_OT_deconflict_stagger,
    DRONE_OT_deconflict_reset,
    EXPORT_OT_drone_yaml,
    EXPORT_OT_export_and_illuminate,
    VIEW3D_PT_drone_swarm,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.drone_props = PointerProperty(type=DroneProperties)
    bpy.types.Object.drone_props = PointerProperty(type=DroneProperties)

    if update_leds_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(update_leds_handler)


def unregister():
    if update_leds_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_leds_handler)

    del bpy.types.Scene.drone_props
    del bpy.types.Object.drone_props

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()