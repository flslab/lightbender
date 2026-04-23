bl_info = {
    "name": "LightBender Swarm Animator",
    "author": "Hamed Alimohammadzadeh",
    "version": (1, 17),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > LightBender",
    "description": "Create light-element drones, animate LEDs with pointers, add position errors, and export YAML",
    "category": "Animation",
}

import bpy
import json
import math
import bmesh
import random
import re
import os
import socket
import subprocess
import time
from bpy.props import FloatProperty, StringProperty, EnumProperty, BoolProperty, IntProperty, PointerProperty, \
    CollectionProperty, FloatVectorProperty 
from bpy.app.handlers import persistent
from mathutils import Vector, Matrix

REPO_DIR = "/path/to/lightbender"
AUTHORING_DIR = os.path.abspath(os.path.join(REPO_DIR, "authoring"))
ORCHESTRATOR_DIR = os.path.abspath(os.path.join(REPO_DIR, "orchestrator"))

# ---------------------------------------------------------------------------
#    Illuminate / Interaction Session State
# ---------------------------------------------------------------------------

_ILLUMINATE_TERMINAL_TAB = {"tab_ref": None}

STATIC_INTERACTION_SESSION = {
    "server_socket": None,
    "sockets": {},
    "buffers": {},
    "timer_registered": False,
    "recording": False,
    "record_deadline": 0.0,
    "record_updated": set(),
    "record_received_any": False,
    "record_failed_sends": [],
}

def get_default_interaction_yaml_path(mission_name):
    return os.path.join(ORCHESTRATOR_DIR, "SFL", f"{mission_name}.yaml")


def get_default_orchestrator_path():
    return os.path.join(ORCHESTRATOR_DIR, "orchestrator.py")


def get_controller_python():
    controller_root = REPO_DIR
    candidates = [
        os.path.join(controller_root, "venv", "bin", "python"),
        os.path.join(controller_root, ".venv", "bin", "python"),
        os.path.join(controller_root, "venv", "Scripts", "python.exe"),
        os.path.join(controller_root, ".venv", "Scripts", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return "python3"


def build_led_formula(props):
    if props.led_mode == 'EXPRESSION':
        return props.led_formula
    raw_ptrs = []
    for idx, ptr in enumerate(props.led_pointers):
        raw_ptrs.append({'id': f"p{idx}", 'v': ptr.value, 'c': ptr.color_expression})
    raw_ptrs.sort(key=lambda x: x['v'])
    if not raw_ptrs:
        return props.led_base_color
    current_str = f"({raw_ptrs[-1]['c']})"
    for j in range(len(raw_ptrs) - 2, -1, -1):
        p_curr = raw_ptrs[j]
        p_next = raw_ptrs[j + 1]
        current_str = f"({p_curr['c']}) if i < {p_next['id']} else {current_str}"
    p0 = raw_ptrs[0]
    return f"({props.led_base_color}) if i < {p0['id']} else {current_str}"


def build_interaction_yaml_text(scene, drones_to_export, include_blender_port=False):
    props = scene.drone_props
    mission_name = props.export_mission_name.strip() or scene.name.replace(' ', '_')
    fps = scene.render.fps
    duration = round((scene.frame_end - scene.frame_start) / fps, 4)
    delta_v = round(props.interaction_sd / 1000.0, 4)

    output_lines = [f"name: {mission_name}", "drones:"]
    original_frame = scene.frame_current
    scene.frame_set(original_frame)

    for drone in drones_to_export:
        loc = drone.matrix_world.translation
        x, y, z = round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)
        yaw = round(drone.matrix_world.to_euler('XYZ').z, 4)
        s1 = round(drone.get("servo_1", 0.0), 2)
        s2 = round(drone.get("servo_2", 0.0), 2)
        safe_formula = build_led_formula(drone.drone_props).replace('"', '\\"')

        output_lines.append(f"  {drone.name}:")
        output_lines.append(f"    target: [{x}, {y}, {z}, {yaw}]")
        output_lines.append("    waypoints: []")
        output_lines.append(f"    delta_t: {duration}")
        output_lines.append("    iterations: 1")
        output_lines.append("    interaction: single")
        output_lines.append("    params:")
        output_lines.append("      linear: true")
        output_lines.append("      relative: false")
        output_lines.append(f"    servos: [[{s1}, {s2}]]")
        output_lines.append("    led:")
        output_lines.append('      mode: "expression"')
        output_lines.append("      rate: 0.5")
        output_lines.append(f'      formula: "{safe_formula}"')

    output_lines.extend([
        "",
        "Interaction:",
        "  action: translation",
        "  config:",
        f"    delta_v: {delta_v}",
        "    z: -1",
        "    friction_coefficient: 0",
        "    base_attitude: -1",
        f"    duration: {duration}",
        "    v_scalar: [10, 10, 3]",
        f"    blender_port: {props.interaction_tcp_port if include_blender_port else 'null'}",
        f"    grace_time: {round(props.interaction_grace_time, 4)}",
    ])
    return "\n".join(output_lines)


def parse_interaction_ips(raw_ips):
    return [ip for ip in re.split(r"[\s,;]+", raw_ips.strip()) if ip]


def get_lb_scene_objects(scene):
    return [obj for obj in scene.objects if re.match(r"^lb\d+", obj.name) and "servo_1" in obj and "servo_2" in obj]


def show_popup(context, title, lines, icon='INFO'):
    def draw(self, _context):
        for line in lines:
            self.layout.label(text=line)
    context.window_manager.popup_menu(draw, title=title, icon=icon)


def read_manifest_ips(id_filter=None):
    """Read drone IP addresses from swarm_manifest.yaml next to the orchestrator.
    
    Args:
        id_filter: optional set/list of drone IDs (e.g. {'lb1','lb2'}) to include.
                   If None, all drones are returned.
    Returns:
        list of IP strings for matching drones.
    """
    entries = []  # list of (id, ip) tuples
    manifest_path = os.path.join(os.path.dirname(get_default_orchestrator_path()), "swarm_manifest.yaml")
    if not os.path.isfile(manifest_path):
        return []
    try:
        with open(manifest_path, 'r') as f:
            in_drones = False
            current_id = None
            current_ip = None
            for line in f:
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                if stripped == 'drones:':
                    in_drones = True
                    continue
                if in_drones:
                    # Detect end of drones section
                    if stripped and not stripped.startswith('-') and not stripped.startswith('id:') and not stripped.startswith('ip:') and ':' in stripped and not stripped.startswith('#'):
                        key = stripped.split(':')[0].strip()
                        if key not in ('id', 'ip', 'user', 'init_pos', 'type', 'uri', 'servo_offsets', 'label', 'obj_name', 'adhoc_ip'):
                            # Save last entry before leaving
                            if current_id and current_ip:
                                entries.append((current_id, current_ip))
                            in_drones = False
                            continue
                    # New drone entry
                    if stripped.startswith('- id:'):
                        # Save previous entry
                        if current_id and current_ip:
                            entries.append((current_id, current_ip))
                        current_id = stripped.split(':', 1)[1].strip().strip('"').strip("'")
                        current_ip = None
                    elif stripped.startswith('id:'):
                        current_id = stripped.split(':', 1)[1].strip().strip('"').strip("'")
                    elif stripped.startswith('ip:') or stripped.startswith('ip :'):
                        current_ip = stripped.split(':', 1)[1].strip().strip('"').strip("'")
            # Don't forget the last entry
            if in_drones and current_id and current_ip:
                entries.append((current_id, current_ip))
    except Exception:
        pass

    if id_filter is not None:
        id_set = set(id_filter)
        return [ip for drone_id, ip in entries if drone_id in id_set]
    return [ip for _, ip in entries]


def launch_orchestrator(target_script, flags, dark_room=False):
    python_exec = get_controller_python()
    cmd_str = f"'{python_exec}' '{target_script}' {' '.join(flags)} --skip-confirm"
    if dark_room:
        cmd_str += " --dark"

    apple_script = f'''
    tell application "Terminal"
        set newTab to do script "cd \\"{os.path.dirname(target_script)}\\" && {cmd_str}"
        activate
    end tell
    '''
    subprocess.Popen(["osascript", "-e", apple_script])


def flush_static_interaction_messages():
    for key, sock_obj in list(STATIC_INTERACTION_SESSION["sockets"].items()):
        STATIC_INTERACTION_SESSION["buffers"][key] = ""
        while True:
            try:
                data = sock_obj.recv(4096)
                if not data:
                    break
            except BlockingIOError:
                break
            except OSError:
                break


def close_static_interaction_session():
    server_socket = STATIC_INTERACTION_SESSION.get("server_socket")
    if server_socket:
        try:
            server_socket.close()
        except OSError:
            pass
    STATIC_INTERACTION_SESSION["server_socket"] = None
    for sock_obj in STATIC_INTERACTION_SESSION["sockets"].values():
        try:
            sock_obj.close()
        except OSError:
            pass
    STATIC_INTERACTION_SESSION["sockets"].clear()
    STATIC_INTERACTION_SESSION["buffers"].clear()
    STATIC_INTERACTION_SESSION["recording"] = False
    STATIC_INTERACTION_SESSION["record_deadline"] = 0.0
    STATIC_INTERACTION_SESSION["record_updated"] = set()
    STATIC_INTERACTION_SESSION["record_received_any"] = False
    STATIC_INTERACTION_SESSION["record_failed_sends"] = []


def update_interaction_editor_state():
    scene = getattr(bpy.context, "scene", None)
    if not scene or not hasattr(scene, "drone_props"):
        return
    props = scene.drone_props
    props.interaction_connected_count = len(STATIC_INTERACTION_SESSION["sockets"])
    props.interaction_recording_active = STATIC_INTERACTION_SESSION["recording"]


def finalize_record_interaction_positions():
    scene = getattr(bpy.context, "scene", None)
    if not scene or not hasattr(scene, "drone_props"):
        STATIC_INTERACTION_SESSION["recording"] = False
        return

    updated = set(STATIC_INTERACTION_SESSION["record_updated"])
    received_any = STATIC_INTERACTION_SESSION["record_received_any"]
    failed_sends = list(STATIC_INTERACTION_SESSION["record_failed_sends"])
    STATIC_INTERACTION_SESSION["recording"] = False
    STATIC_INTERACTION_SESSION["record_deadline"] = 0.0
    STATIC_INTERACTION_SESSION["record_updated"] = set()
    STATIC_INTERACTION_SESSION["record_received_any"] = False
    STATIC_INTERACTION_SESSION["record_failed_sends"] = []

    bpy.context.view_layer.update()
    update_interaction_editor_state()

    if failed_sends:
        show_popup(
            bpy.context,
            "Request Errors",
            [f"Could not request position from {addr}" for addr in failed_sends],
            icon='ERROR'
        )

    all_lbs = {obj.name for obj in get_lb_scene_objects(scene)}
    missing = sorted(all_lbs - updated)

    if not received_any:
        show_popup(bpy.context, "Record Position", ["No new position responses received within 100 ms."], icon='INFO')
    else:
        lines = [f"Updated {len(updated)} LightBenders from TCP responses."]
        if missing:
            lines.extend([f"No update for {name}" for name in missing[:12]])
        show_popup(bpy.context, "Record Position Finished", lines, icon='INFO')


def resolve_lb_by_message_id(scene, raw_id):
    if raw_id is None:
        return None
    raw_str = str(raw_id)
    exact = scene.objects.get(raw_str)
    if exact and "servo_1" in exact:
        return exact
    if raw_str.isdigit():
        return scene.objects.get(f"lb{raw_str}")
    match = re.search(r"(\d+)", raw_str)
    if match:
        return scene.objects.get(f"lb{match.group(1)}")
    return None


def process_static_interaction_socket(key, sock_obj):
    try:
        data = sock_obj.recv(4096)
    except BlockingIOError:
        return
    except OSError:
        data = b""
    if not data:
        try:
            sock_obj.close()
        except OSError:
            pass
        STATIC_INTERACTION_SESSION["sockets"].pop(key, None)
        STATIC_INTERACTION_SESSION["buffers"].pop(key, None)
        return

    buffer_text = STATIC_INTERACTION_SESSION["buffers"].get(key, "") + data.decode('utf-8', errors='ignore')
    lines = buffer_text.splitlines(keepends=False)
    if buffer_text and not buffer_text.endswith("\n"):
        STATIC_INTERACTION_SESSION["buffers"][key] = lines.pop() if lines else buffer_text
    else:
        STATIC_INTERACTION_SESSION["buffers"][key] = ""

    if not STATIC_INTERACTION_SESSION["recording"]:
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        scene = getattr(bpy.context, "scene", None)
        if not scene:
            continue
        STATIC_INTERACTION_SESSION["record_received_any"] = True
        obj = resolve_lb_by_message_id(scene, msg.get("id") or msg.get("drone_id"))
        position = msg.get("position")
        if position is None and all(k in msg for k in ("x", "y", "z")):
            position = [msg["x"], msg["y"], msg["z"]]
        if obj and isinstance(position, (list, tuple)) and len(position) >= 3:
            obj.location = (float(position[0]), float(position[1]), float(position[2]))
            STATIC_INTERACTION_SESSION["record_updated"].add(obj.name)


def static_interaction_timer():
    server_socket = STATIC_INTERACTION_SESSION.get("server_socket")
    if server_socket is None:
        STATIC_INTERACTION_SESSION["timer_registered"] = False
        update_interaction_editor_state()
        return None

    scene = getattr(bpy.context, "scene", None)
    props = getattr(scene, "drone_props", None) if scene else None
    allowed_ips = set(parse_interaction_ips(props.interaction_lightbender_ips)) if props else set()

    while True:
        try:
            client_sock, addr = server_socket.accept()
        except BlockingIOError:
            break
        except OSError:
            break
        client_ip = addr[0]
        if allowed_ips and client_ip not in allowed_ips:
            try:
                client_sock.close()
            except OSError:
                pass
            continue
        client_sock.setblocking(False)
        key = f"{client_ip}:{addr[1]}"
        STATIC_INTERACTION_SESSION["sockets"][key] = client_sock
        STATIC_INTERACTION_SESSION["buffers"][key] = ""

    for key, sock_obj in list(STATIC_INTERACTION_SESSION["sockets"].items()):
        process_static_interaction_socket(key, sock_obj)

    if STATIC_INTERACTION_SESSION["recording"] and time.monotonic() >= STATIC_INTERACTION_SESSION["record_deadline"]:
        finalize_record_interaction_positions()

    update_interaction_editor_state()
    return 0.1

# ------------------------------------------------------------------------
#    Properties & Data Classes
# ------------------------------------------------------------------------

class ColorItem(bpy.types.PropertyGroup):
    color: FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(1.0, 1.0, 1.0),
        min=0.0, max=1.0
    )


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


class DrawEraseGroupItem(bpy.types.PropertyGroup):
    """A single drone assigned to a draw/erase group"""
    obj_ptr: PointerProperty(type=bpy.types.Object)
    direction: EnumProperty(
        name="Direction",
        description="Draw direction for this LightBender (set after generation)",
        items=[
            ('AUTO', "Auto", "Direction auto-determined by pathfinding"),
            ('FORWARD', "Forward", "Draw from start (low index) to end (high index)"),
            ('BACKWARD', "Backward", "Draw from end (high index) to start (low index)"),
        ],
        default='AUTO'
    )


class DrawEraseGroup(bpy.types.PropertyGroup):
    """A collection of drones representing a single stroke or letter"""
    name: StringProperty(name="Group Name", default="Stroke")
    drones: CollectionProperty(type=DrawEraseGroupItem)
    active_drone_index: IntProperty(default=0)

    draw_mode: EnumProperty(
        name="Mode",
        items=[
            ('SEQUENTIAL', "Sequential", "Draw one after the other"),
            ('SIMULTANEOUS', "Simultaneous", "Draw all at once (Branching)")
        ],
        default='SEQUENTIAL'
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
            ('LINE_OF_SIGHT', "Line of Sight", ""),
            ('GLOBAL_CENTROID', "Global", "")
        ],
        default='LINE_OF_SIGHT'
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

    export_mode: EnumProperty(
        name="Mode",
        description="Select export/illuminate mode",
        items=[
            ('ILLUMINATION', "Illumination", "Standard illumination mode"),
            ('INTERACTION', "Interaction", "Interactive editing mode with TCP control"),
        ],
        default='ILLUMINATION'
    )

    export_rate: FloatProperty(
        name="Export Rate (Hz)",
        description="Frequency of waypoints in the output file (1/delta_t)",
        default=2.0,
        min=0.1
    )

    interaction_sd: FloatProperty(
        name="S_D (mm/s)",
        description="Interaction speed threshold exported in m/s",
        default=100.0,
        min=0.0
    )

    interaction_duration: FloatProperty(
        name="Interaction Duration (s)",
        description="Max interaction duration under edit mode in seconds",
        default=300.0,
        min=0.0
    )

    interaction_grace_time: FloatProperty(
        name="Grace Time (s)",
        description="Interaction grace time in seconds",
        default=2.0,
        min=0.0
    )

    interaction_lightbender_ips: StringProperty(
        name="LightBender IPs",
        description="Comma or space separated LightBender IPs for the static interaction editor",
        default=""
    )

    interaction_tcp_port: IntProperty(
        name="TCP Port",
        description="TCP port used by the static interaction editor",
        default=5560,
        min=1,
        max=65535
    )

    interaction_editor_active: BoolProperty(
        name="Interaction Editor Active",
        default=False,
        options={'HIDDEN'}
    )

    interaction_connected_count: IntProperty(
        name="Interaction Connected Count",
        default=0,
        options={'HIDDEN'}
    )

    interaction_recording_active: BoolProperty(
        name="Interaction Recording Active",
        default=False,
        options={'HIDDEN'}
    )

    illuminate_running: BoolProperty(
        name="Illuminate Running",
        default=False,
        options={'HIDDEN'}
    )

    edit_active: BoolProperty(
        name="Edit Active",
        default=False,
        options={'HIDDEN'}
    )

    # SVG to Layout Settings
    import_svg_filepath: StringProperty(
        name="SVG File",
        description="Path to the SVG file",
        subtype="FILE_PATH",
        default=""
    )

    import_manifest_filepath: StringProperty(
        name="Swarm Manifest",
        description="Path to the swarm manifest YAML file",
        subtype="FILE_PATH",
        default=""
    )

    import_max_width: FloatProperty(
        name="Max Width",
        description="Maximum width for the scaled layout",
        default=2.0
    )

    import_max_length: FloatProperty(
        name="Max Height",
        description="Maximum height for the scaled layout",
        default=1.0
    )

    import_cx: FloatProperty(
        name="Center X",
        description="Target X coordinate for the center",
        default=0.0
    )

    import_cy: FloatProperty(
        name="Center Z",
        description="Target Z coordinate for the center",
        default=0.0
    )

    import_policy: EnumProperty(
        name="Placement Policy",
        description="Placement policy",
        items=[
            ('VFG', "VFG", ""),
            ('SC', "Set Cover", "")
        ],
        default='VFG'
    )

    import_color: StringProperty(
        name="LED Color",
        description="Python list color (e.g. [255, 0, 0])",
        default="[255, 255, 255]"
    )

    export_at_keyframes: BoolProperty(
        name="Export Keyframes Only",
        description="Export only at keyframe locations (adds dt to waypoints)",
        default=False
    )

    # Morph Animation Settings
    morph_svg_filepath: StringProperty(
        name="Shape 2 SVG",
        description="Path to the SVG file for the target shape",
        subtype="FILE_PATH",
        default=""
    )
    
    morph_duration: FloatProperty(
        name="Morph Duration",
        description="Time it takes to move between shapes (sec)",
        default=2.0,
        min=0.1
    )
    
    morph_hold_1: FloatProperty(
        name="Hold Shape 1",
        description="Time to hold the first shape before moving (sec)",
        default=2.0,
        min=0.0
    )
    
    morph_hold_2: FloatProperty(
        name="Hold Shape 2",
        description="Time to hold the second shape after moving (sec)",
        default=5.0,
        min=0.0
    )

    # Draw & Erase Sequencer
    de_groups: CollectionProperty(type=DrawEraseGroup)
    de_active_group_index: IntProperty(default=0)
    de_draw_speed: FloatProperty(name="Draw Speed (LEDs/s)", default=50.0, min=1.0)
    de_erase_speed: FloatProperty(name="Erase Speed (LEDs/s)", default=50.0, min=1.0)
    de_hold_time: FloatProperty(name="Hold Time (s)", default=1.0, min=0.0)
    de_overlap: FloatProperty(name="Overlap Next (%)", default=0.0, min=0.0, max=100.0, subtype='PERCENTAGE')
    de_loop: BoolProperty(name="Loop Animation", default=False)
    de_draw_color: StringProperty(name="Draw Color", default="[255, 255, 255]")
    de_bg_color: StringProperty(name="Background", default="[0, 0, 0]")

    # Depth Fly-In Feature
    flyin_ref_distance: FloatProperty(name="Ref Distance", description="Distance of reference point from centroid behind LBs", default=2.0, min=0.1)
    flyin_x_min: FloatProperty(name="X Min", default=-5.0)
    flyin_x_max: FloatProperty(name="X Max", default=5.0)
    flyin_y_min: FloatProperty(name="Y Min", default=-5.0)
    flyin_y_max: FloatProperty(name="Y Max", default=5.0)
    flyin_z_min: FloatProperty(name="Z Min", default=-5.0)
    flyin_z_max: FloatProperty(name="Z Max", default=5.0)
    flyin_outward_duration: FloatProperty(name="Outward Duration(s)", default=2.0, min=0.1)
    flyin_hold_duration: FloatProperty(name="Hold Duration(s)", default=0.5, min=0.0)
    flyin_inward_duration: FloatProperty(name="Inward Duration(s)", default=3.0, min=0.1)

    # Fold Animation Settings
    fold_angle: FloatProperty(
        name="Fold Angle (deg)",
        description="Straight-state angle for segment 1 in degrees (segment 2 = angle + 180). Adjusted per type: TYPE_V adds 180.",
        default=90.0,
        min=0.0,
        max=180.0
    )
    fold_hold_time: FloatProperty(name="Hold Time (s)", description="Time to hold the straight fully-lit state", default=2.0, min=0.0)
    fold_transition_time: FloatProperty(name="Transition Time (s)", description="Time to transition into the current state", default=2.0, min=0.1)
    fold_align_enabled: BoolProperty(
        name="Align",
        description="In the straight state, snap all LBs to the centroid on the chosen axis",
        default=False
    )
    fold_align_axis: EnumProperty(
        name="Align Axis",
        description="Axis on which to align all LBs to the centroid",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='Z'
    )
    fold_distribute_enabled: BoolProperty(
        name="Distribute",
        description="In the straight state, evenly space all LBs along the chosen axis",
        default=False
    )
    fold_distribute_axis: EnumProperty(
        name="Distribute Axis",
        description="Axis along which to evenly distribute LBs",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='X'
    )
    fold_equalize_size: BoolProperty(
        name="Equalize Size",
        description="Scale each LB's LED pointer range by its camera distance so all rods appear the same length from the camera's point of view",
        default=False
    )
    fold_overlap_resolve: BoolProperty(
        name="Resolve Overlaps",
        description="After positioning, detect overlaps and uniformly stretch positions along the chosen axis until the minimum spacing equals the threshold",
        default=False
    )
    fold_overlap_threshold: FloatProperty(
        name="Min Spacing",
        description="Minimum allowed distance between any two LBs along the overlap axis",
        default=0.5,
        min=0.0,
        unit='LENGTH'
    )
    fold_overlap_axis: EnumProperty(
        name="Overlap Axis",
        description="Axis along which to measure and resolve overlaps",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='X'
    )

    # Global LED Expressions
    global_expr_preset: EnumProperty(
        name="Preset",
        description="Select a global animation preset",
        items=[
            ('STATIC', "Static Color", "Single static color for all LEDs"),
            ('BLINK', "Global Blinking", "Cycles colors simultaneously"),
            ('SPARKLE', "Random Sparkle", "Randomly sparkles with the first color"),
            ('RAINBOW', "Rainbow Wheel", "Moving rainbow through all colors"),
            ('CHASE', "Color Chase", "A moving block of color"),
            ('PULSE', "Pulse", "Smoothly pulses brightness"),
            ('COLOR_CYCLE', "Color Cycle", "Smoothly cycles between two colors using sin/cos blend"),
        ],
        default='BLINK'
    )
    global_expr_colors: CollectionProperty(type=ColorItem)
    global_expr_active_color_index: IntProperty(default=0)
    global_expr_speed: FloatProperty(name="Speed Multiplier", default=5.0, min=0.1)
    global_expr_sparkle_threshold: FloatProperty(
        name="Sparkle Threshold",
        description="Probability threshold above which a LED shows the sparkle color (0 = always sparkle, 1 = never)",
        default=0.7,
        min=0.0,
        max=1.0
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


def get_structure_material():
    """Returns a dark-gray diffuse material for drone structure meshes (hidden from render)"""
    mat_name = "Drone_Structure_Material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

        # Almost-black dark gray
        node_bsdf.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)
        node_bsdf.inputs['Roughness'].default_value = 0.8

        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat


# ------------------------------------------------------------------------
#    Geometry & Setup Operator
# ------------------------------------------------------------------------

class OBJECT_OT_add_drone(bpy.types.Operator):
    """Create a new Drone with Light Elements"""
    bl_idname = "drone.add_drone"
    bl_label = "Add LightBender"
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
        obj.data.materials.append(get_structure_material())
        obj.hide_render = True

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
        obj.data.materials.append(get_structure_material())
        obj.hide_render = True

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
        drone_base.empty_display_size = 0.1

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
            rod1.data.materials.append(get_structure_material())
            rod1.hide_render = True

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
            rod2.data.materials.append(get_structure_material())
            rod2.hide_render = True

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
#    Draw & Erase Sequencer Feature
# ------------------------------------------------------------------------

class DRONE_OT_add_de_group(bpy.types.Operator):
    """Add a new Draw/Erase Group"""
    bl_idname = "drone.add_de_group"
    bl_label = "Add Group"

    def execute(self, context):
        props = context.scene.drone_props
        item = props.de_groups.add()
        item.name = f"Stroke {len(props.de_groups)}"
        props.de_active_group_index = len(props.de_groups) - 1
        return {'FINISHED'}


class DRONE_OT_remove_de_group(bpy.types.Operator):
    """Remove active Draw/Erase Group"""
    bl_idname = "drone.remove_de_group"
    bl_label = "Remove Group"

    def execute(self, context):
        props = context.scene.drone_props
        idx = props.de_active_group_index
        if len(props.de_groups) > 0:
            props.de_groups.remove(idx)
            props.de_active_group_index = max(0, idx - 1)
        return {'FINISHED'}


class DRONE_OT_add_drones_to_group(bpy.types.Operator):
    """Add selected LBs to active Draw/Erase Group"""
    bl_idname = "drone.add_drones_to_group"
    bl_label = "Assign Selected"

    def execute(self, context):
        props = context.scene.drone_props
        if not props.de_groups: return {'CANCELLED'}
        group = props.de_groups[props.de_active_group_index]

        added = 0
        for obj in context.selected_objects:
            if "servo_1" in obj and obj.name.startswith("lb"):
                # Avoid duplicates
                exists = any(item.obj_ptr == obj for item in group.drones)
                if not exists:
                    new_item = group.drones.add()
                    new_item.obj_ptr = obj
                    added += 1

        self.report({'INFO'}, f"Added {added} drones to {group.name}")
        return {'FINISHED'}


class DRONE_OT_remove_drone_from_group(bpy.types.Operator):
    """Remove selected drone from the active group list"""
    bl_idname = "drone.remove_drone_from_group"
    bl_label = "Remove Drone"

    def execute(self, context):
        props = context.scene.drone_props
        if not props.de_groups: return {'CANCELLED'}
        group = props.de_groups[props.de_active_group_index]
        idx = group.active_drone_index
        if len(group.drones) > 0:
            group.drones.remove(idx)
            group.active_drone_index = max(0, idx - 1)
        return {'FINISHED'}


class DRONE_OT_move_drone_in_group(bpy.types.Operator):
    """Move drone up or down in the sequence"""
    bl_idname = "drone.move_drone_in_group"
    bl_label = "Move Drone"
    direction: EnumProperty(items=[('UP', "Up", ""), ('DOWN', "Down", "")])

    def execute(self, context):
        props = context.scene.drone_props
        if not props.de_groups: return {'CANCELLED'}
        group = props.de_groups[props.de_active_group_index]
        idx = group.active_drone_index

        if self.direction == 'UP' and idx > 0:
            group.drones.move(idx, idx - 1)
            group.active_drone_index -= 1
        elif self.direction == 'DOWN' and idx < len(group.drones) - 1:
            group.drones.move(idx, idx + 1)
            group.active_drone_index += 1
        return {'FINISHED'}


class DRONE_OT_reverse_de_direction(bpy.types.Operator):
    """Toggle draw direction and reverse existing keyframes in-place if animation is already generated"""
    bl_idname = "drone.reverse_de_direction"
    bl_label = "Reverse Direction"
    bl_options = {'REGISTER', 'UNDO'}
    group_index: IntProperty()
    drone_index: IntProperty()

    def execute(self, context):
        props = context.scene.drone_props
        if self.group_index >= len(props.de_groups):
            return {'CANCELLED'}
        group = props.de_groups[self.group_index]
        if self.drone_index >= len(group.drones):
            return {'CANCELLED'}
        item = group.drones[self.drone_index]
        lb = item.obj_ptr
        if not lb:
            return {'CANCELLED'}

        # Toggle the stored direction flag (AUTO is treated as FORWARD on first click)
        if item.direction == 'BACKWARD':
            item.direction = 'FORWARD'
        else:
            item.direction = 'BACKWARD'

        # If keyframes have already been generated, reverse them in-place via a flip+swap.
        # The lit window is [ptr0, ptr1].  Reversing means:
        #   new_ptr0 = v_min + v_max - old_ptr1
        #   new_ptr1 = v_min + v_max - old_ptr0
        # This keeps all timing intact and only flips which end is swept first.
        if "de_orig_vmin" not in lb:
            return {'FINISHED'}  # No animation yet – direction flag alone is enough

        if not (lb.animation_data and lb.animation_data.action):
            return {'FINISHED'}

        v_min = lb["de_orig_vmin"]
        v_max = lb["de_orig_vmax"]
        action = lb.animation_data.action
        fc0 = action.fcurves.find('drone_props.led_pointers[0].value')
        fc1 = action.fcurves.find('drone_props.led_pointers[1].value')

        if not fc0 or not fc1:
            return {'FINISHED'}

        # Read ALL values before touching anything (avoids read-after-write errors)
        frames0 = [kp.co.x for kp in fc0.keyframe_points]
        frames1 = [kp.co.x for kp in fc1.keyframe_points]
        new_vals0 = [v_min + v_max - fc1.evaluate(f) for f in frames0]
        new_vals1 = [v_min + v_max - fc0.evaluate(f) for f in frames1]

        # Write new values; Blender recalcs LINEAR handles after fc.update()
        for kp, val in zip(fc0.keyframe_points, new_vals0):
            kp.co.y = val
            kp.handle_left.y = val
            kp.handle_right.y = val
        fc0.update()

        for kp, val in zip(fc1.keyframe_points, new_vals1):
            kp.co.y = val
            kp.handle_left.y = val
            kp.handle_right.y = val
        fc1.update()

        return {'FINISHED'}


class DRONE_OT_clear_draw_erase(bpy.types.Operator):
    """Remove generated draw/erase keyframes and restore original pointer values"""
    bl_idname = "drone.clear_draw_erase"
    bl_label = "Clear Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.drone_props

        all_lbs = set()
        for group in props.de_groups:
            for item in group.drones:
                if item.obj_ptr:
                    all_lbs.add(item.obj_ptr)

        cleared = 0
        for lb in all_lbs:
            # Remove pointer keyframe fcurves
            if lb.animation_data and lb.animation_data.action:
                for ptr_idx in [0, 1]:
                    fc = lb.animation_data.action.fcurves.find(
                        f'drone_props.led_pointers[{ptr_idx}].value')
                    if fc:
                        lb.animation_data.action.fcurves.remove(fc)

            # Restore original pointer values saved during generation
            if len(lb.drone_props.led_pointers) >= 2:
                if "de_orig_vmin" in lb:
                    lb.drone_props.led_pointers[0].value = lb["de_orig_vmin"]
                    del lb["de_orig_vmin"]
                if "de_orig_vmax" in lb:
                    lb.drone_props.led_pointers[1].value = lb["de_orig_vmax"]
                    del lb["de_orig_vmax"]
            cleared += 1

        # Reset all item directions to AUTO so next generation starts fresh
        for group in props.de_groups:
            for item in group.drones:
                item.direction = 'AUTO'

        context.view_layer.update()
        self.report({'INFO'}, f"Cleared draw/erase animation from {cleared} LBs.")
        return {'FINISHED'}


class UL_DrawEraseGroups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "name", text="", emboss=False)
        layout.prop(item, "draw_mode", text="")


class UL_DrawEraseDrones(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if item.obj_ptr:
            row = layout.row(align=True)
            row.label(text=item.obj_ptr.name, icon='LIGHT')
            # Show direction as an icon; clicking toggles FORWARD ↔ BACKWARD
            if item.direction == 'FORWARD':
                dir_icon = 'TRIA_RIGHT'
            elif item.direction == 'BACKWARD':
                dir_icon = 'TRIA_LEFT'
            else:  # AUTO — not yet generated
                dir_icon = 'QUESTION'
            op = row.operator("drone.reverse_de_direction", text="", icon=dir_icon, emboss=False)
            op.group_index = context.scene.drone_props.de_active_group_index
            op.drone_index = index
        else:
            layout.label(text="<Missing Object>", icon='ERROR')


class UL_GlobalColorList(bpy.types.UIList):
    """List representation of global colors"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "color", text="")


# ------------------------------------------------------------------------
#    Fold Animation Feature
# ------------------------------------------------------------------------

def _fold_straight_angles(drone_type, fold_angle):
    """Return (s1, s2) straight angles for the given drone type and user fold_angle."""
    if drone_type == 'TYPE_H':
        return fold_angle, fold_angle + 180.0
    elif drone_type == 'TYPE_V':
        if fold_angle > 90:
            return fold_angle, fold_angle + 180.0
        else:
            return fold_angle + 180.0, fold_angle + 360.0
    elif drone_type == 'TYPE_SEMI_H':
        return fold_angle, None
    elif drone_type == 'TYPE_SEMI_V':
        return fold_angle + 180.0, None
    else:  # TYPE_RING or unknown
        return None, None


class DRONE_OT_generate_fold(bpy.types.Operator):
    """Keyframe all LBs from a straight fully-lit state into their current state"""
    bl_idname = "drone.generate_fold"
    bl_label = "Generate Fold"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import re
        scene = context.scene
        props = scene.drone_props
        fps = scene.render.fps
        fold_angle = props.fold_angle
        t_hold = props.fold_hold_time
        t_trans = props.fold_transition_time
        align_enabled = props.fold_align_enabled
        align_axis = props.fold_align_axis
        dist_enabled = props.fold_distribute_enabled
        dist_axis = props.fold_distribute_axis
        equalize_size = props.fold_equalize_size
        overlap_resolve = props.fold_overlap_resolve
        overlap_threshold = props.fold_overlap_threshold
        overlap_axis = props.fold_overlap_axis
        position_override = align_enabled or dist_enabled or overlap_resolve

        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}

        f0 = 0
        f_hold = int(round(t_hold * fps))
        f_end = int(round((t_hold + t_trans) * fps))

        lbs = [obj for obj in scene.objects
               if re.match(r"^lb\d+", obj.name) and "servo_1" in obj]

        if not lbs:
            self.report({'WARNING'}, "No LB drones found in the scene.")
            return {'CANCELLED'}

        # --- Compute per-LB initial positions ---
        # Start from each LB's current location, then apply align and/or distribute.
        computed_positions = {lb.name: list(lb.location) for lb in lbs}
        if position_override:
            n = len(lbs)

            # Align: collapse all LBs to centroid on the align axis
            if align_enabled:
                a_idx = axis_idx[align_axis]
                centroid_a = sum(lb.location[a_idx] for lb in lbs) / n
                for lb in lbs:
                    computed_positions[lb.name][a_idx] = centroid_a

            # Distribute: evenly space LBs along the distribute axis.
            # When a camera is present, distribute so objects appear evenly spaced
            # from the camera's perspective by solving for world positions that
            # produce uniform screen-space intervals (Newton's method).
            if dist_enabled and n > 1:
                import mathutils
                d_idx = axis_idx[dist_axis]
                e_d = mathutils.Vector([1.0 if i == d_idx else 0.0 for i in range(3)])
                camera = scene.camera

                if camera is not None:
                    from bpy_extras.object_utils import world_to_camera_view

                    def _screen(base_pos, t=0.0):
                        return world_to_camera_view(
                            scene, camera, mathutils.Vector(base_pos) + t * e_d
                        )

                    # Pick the screen axis most sensitive to movement along e_d
                    mid_base = computed_positions[lbs[len(lbs) // 2].name]
                    sc_a = _screen(mid_base, 0.0)
                    sc_b = _screen(mid_base, 1.0)
                    s_axis = 0 if abs(sc_b.x - sc_a.x) >= abs(sc_b.y - sc_a.y) else 1

                    # Sort by current screen position on that axis
                    lbs_sorted = sorted(
                        lbs, key=lambda o: _screen(computed_positions[o.name])[s_axis]
                    )
                    s_min = _screen(computed_positions[lbs_sorted[0].name])[s_axis]
                    s_max = _screen(computed_positions[lbs_sorted[-1].name])[s_axis]

                    targets = {
                        lb.name: s_min + (s_max - s_min) * i / (n - 1)
                        for i, lb in enumerate(lbs_sorted)
                    }

                    # Newton's method: find t so that _screen(base, t)[s_axis] == target
                    num_eps = 1e-3
                    for lb in lbs:
                        base = computed_positions[lb.name]
                        s_target = targets[lb.name]
                        t = 0.0
                        for _ in range(30):
                            f = _screen(base, t)[s_axis] - s_target
                            if abs(f) < 1e-7:
                                break
                            df = (_screen(base, t + num_eps)[s_axis] - _screen(base, t)[s_axis]) / num_eps
                            if abs(df) < 1e-12:
                                break
                            t -= f / df
                        computed_positions[lb.name][d_idx] = base[d_idx] + t

                else:
                    # No camera: uniform world-space distribution
                    lbs_sorted = sorted(lbs, key=lambda o: computed_positions[o.name][d_idx])
                    pos_min = computed_positions[lbs_sorted[0].name][d_idx]
                    pos_max = computed_positions[lbs_sorted[-1].name][d_idx]
                    span = pos_max - pos_min
                    for i, lb in enumerate(lbs_sorted):
                        computed_positions[lb.name][d_idx] = pos_min + span * i / (n - 1)

            # Resolve overlaps: check all pairs with Euclidean distance.
            # For each overlapping pair, compute the minimum axis-only scale
            # factor that brings their Euclidean distance to exactly the threshold.
            # Apply the largest required scale so every pair is resolved.
            if overlap_resolve and n > 1:
                import math
                o_idx = axis_idx[overlap_axis]
                thresh_sq = overlap_threshold ** 2
                positions = [computed_positions[lb.name] for lb in lbs]

                required_scale = 1.0
                for i in range(n):
                    for j in range(i + 1, n):
                        pi, pj = positions[i], positions[j]
                        # Squared distance in the axes perpendicular to o_idx
                        d_perp_sq = sum(
                            (pi[k] - pj[k]) ** 2 for k in range(3) if k != o_idx
                        )
                        if d_perp_sq >= thresh_sq:
                            # Perpendicular distance alone already clears threshold
                            continue
                        delta_axis = pi[o_idx] - pj[o_idx]
                        if abs(delta_axis) < 1e-12:
                            # Co-located on axis; axis scaling cannot resolve this
                            continue
                        # Solve: d_perp_sq + (s * delta_axis)^2 = thresh_sq
                        s_needed = math.sqrt((thresh_sq - d_perp_sq) / delta_axis ** 2)
                        if s_needed > required_scale:
                            required_scale = s_needed

                if required_scale > 1.0:
                    centroid_o = sum(computed_positions[lb.name][o_idx] for lb in lbs) / n
                    for lb in lbs:
                        computed_positions[lb.name][o_idx] = (
                            centroid_o
                            + required_scale * (computed_positions[lb.name][o_idx] - centroid_o)
                        )

        # --- Compute per-LB straight-state pointer values ---
        # Default: full range (0 to 50, centred at 25, half-range = 25).
        # With equalize_size, scale each LB's half-range by its distance from the
        # camera relative to the farthest LB so all rods appear the same length.
        straight_pointers = {lb.name: (0.0, 50.0) for lb in lbs}
        if equalize_size and scene.camera is not None:
            import mathutils
            cam_pos = scene.camera.matrix_world.translation
            distances = {
                lb.name: (mathutils.Vector(computed_positions[lb.name]) - cam_pos).length
                for lb in lbs
            }
            d_max = max(distances.values())
            if d_max > 0:
                for lb in lbs:
                    half = 25.0 * distances[lb.name] / d_max
                    straight_pointers[lb.name] = (25.0 - half, 25.0 + half)

        for lb in lbs:
            dp = lb.drone_props
            d_type = dp.drone_type
            s1_straight, s2_straight = _fold_straight_angles(d_type, fold_angle)

            # Save originals
            lb["fold_orig_s1"] = lb.get("servo_1", 0.0)
            if "servo_2" in lb:
                lb["fold_orig_s2"] = lb.get("servo_2", 0.0)

            # Ensure POINTERS mode and at least 2 pointers
            dp.led_mode = 'POINTERS'
            while len(dp.led_pointers) < 2:
                dp.led_pointers.add()

            p0 = dp.led_pointers[0]
            p1 = dp.led_pointers[1]
            lb["fold_orig_p0"] = p0.value
            lb["fold_orig_p1"] = p1.value

            if position_override:
                lb["fold_orig_loc"] = list(lb.location)

            orig_s1 = lb["fold_orig_s1"]
            orig_s2 = lb.get("fold_orig_s2", None)
            orig_p0 = lb["fold_orig_p0"]
            orig_p1 = lb["fold_orig_p1"]

            # --- Set straight + fully-lit state and keyframe at f0 and f_hold ---
            if s1_straight is not None:
                lb["servo_1"] = s1_straight
            if s2_straight is not None and "servo_2" in lb:
                lb["servo_2"] = s2_straight
            p0.value, p1.value = straight_pointers[lb.name]

            if position_override and lb.name in computed_positions:
                cp = computed_positions[lb.name]
                lb.location = (cp[0], cp[1], cp[2])

            for f in (f0, f_hold):
                if s1_straight is not None:
                    lb.keyframe_insert(data_path='["servo_1"]', frame=f)
                if s2_straight is not None and "servo_2" in lb:
                    lb.keyframe_insert(data_path='["servo_2"]', frame=f)
                p0.keyframe_insert(data_path="value", frame=f)
                p1.keyframe_insert(data_path="value", frame=f)
                if position_override:
                    lb.keyframe_insert(data_path="location", frame=f)

            # --- Restore original state and keyframe at f_end ---
            lb["servo_1"] = orig_s1
            if orig_s2 is not None and "servo_2" in lb:
                lb["servo_2"] = orig_s2
            p0.value = orig_p0
            p1.value = orig_p1

            if position_override:
                orig_loc = lb["fold_orig_loc"]
                lb.location = (orig_loc[0], orig_loc[1], orig_loc[2])

            lb.keyframe_insert(data_path='["servo_1"]', frame=f_end)
            if orig_s2 is not None and "servo_2" in lb:
                lb.keyframe_insert(data_path='["servo_2"]', frame=f_end)
            p0.keyframe_insert(data_path="value", frame=f_end)
            p1.keyframe_insert(data_path="value", frame=f_end)
            if position_override:
                lb.keyframe_insert(data_path="location", frame=f_end)

        # Set all fold-inserted keyframes to LINEAR
        for lb in lbs:
            if lb.animation_data and lb.animation_data.action:
                for fc in lb.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'

        context.view_layer.update()
        self.report({'INFO'}, f"Fold animation generated for {len(lbs)} LBs.")
        return {'FINISHED'}


class DRONE_OT_clear_fold(bpy.types.Operator):
    """Remove generated fold keyframes and restore original servo/pointer values"""
    bl_idname = "drone.clear_fold"
    bl_label = "Clear Fold Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import re
        scene = context.scene

        lbs = [obj for obj in scene.objects
               if re.match(r"^lb\d+", obj.name) and "servo_1" in obj]

        cleared = 0
        for lb in lbs:
            if lb.animation_data and lb.animation_data.action:
                action = lb.animation_data.action
                for dp_path in ('["servo_1"]', '["servo_2"]',
                                'drone_props.led_pointers[0].value',
                                'drone_props.led_pointers[1].value'):
                    fc = action.fcurves.find(dp_path)
                    if fc:
                        action.fcurves.remove(fc)
                # Remove location fcurves inserted by fold alignment
                for axis_i in range(3):
                    fc = action.fcurves.find('location', index=axis_i)
                    if fc:
                        action.fcurves.remove(fc)

            dp = lb.drone_props
            if "fold_orig_s1" in lb:
                lb["servo_1"] = lb["fold_orig_s1"]
                del lb["fold_orig_s1"]
            if "fold_orig_s2" in lb and "servo_2" in lb:
                lb["servo_2"] = lb["fold_orig_s2"]
                del lb["fold_orig_s2"]
            if len(dp.led_pointers) >= 2:
                if "fold_orig_p0" in lb:
                    dp.led_pointers[0].value = lb["fold_orig_p0"]
                    del lb["fold_orig_p0"]
                if "fold_orig_p1" in lb:
                    dp.led_pointers[1].value = lb["fold_orig_p1"]
                    del lb["fold_orig_p1"]
            if "fold_orig_loc" in lb:
                orig_loc = lb["fold_orig_loc"]
                lb.location = (orig_loc[0], orig_loc[1], orig_loc[2])
                del lb["fold_orig_loc"]
            cleared += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Cleared fold animation from {cleared} LBs.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Global LED Expressions Feature
# ------------------------------------------------------------------------

class DRONE_OT_add_global_color(bpy.types.Operator):
    """Add a new color to the global expression palette"""
    bl_idname = "drone.add_global_color"
    bl_label = "Add Color"

    def execute(self, context):
        props = context.scene.drone_props
        props.global_expr_colors.add()
        props.global_expr_active_color_index = len(props.global_expr_colors) - 1
        return {'FINISHED'}


class DRONE_OT_remove_global_color(bpy.types.Operator):
    """Remove active color from global expression palette"""
    bl_idname = "drone.remove_global_color"
    bl_label = "Remove Color"

    def execute(self, context):
        props = context.scene.drone_props
        idx = props.global_expr_active_color_index
        if len(props.global_expr_colors) > 0:
            props.global_expr_colors.remove(idx)
            props.global_expr_active_color_index = max(0, idx - 1)
        return {'FINISHED'}


class DRONE_OT_preset_colors(bpy.types.Operator):
    """Apply default colors for the selected preset"""
    bl_idname = "drone.preset_colors"
    bl_label = "Set Default Preset Colors"

    def execute(self, context):
        props = context.scene.drone_props
        props.global_expr_colors.clear()

        preset = props.global_expr_preset
        
        def add_c(r, g, b):
            c = props.global_expr_colors.add()
            c.color = (r, g, b)

        if preset == 'BLINK':
            add_c(1.0, 0.0, 0.0)
            add_c(0.0, 0.0, 1.0)
        elif preset == 'SPARKLE':
            add_c(0.8, 0.9, 1.0)
        elif preset == 'RAINBOW':
            add_c(1.0, 0.0, 0.0)
            add_c(1.0, 0.5, 0.0)
            add_c(1.0, 1.0, 0.0)
            add_c(0.0, 1.0, 0.0)
            add_c(0.0, 0.0, 1.0)
            add_c(0.29, 0.0, 0.51)
            add_c(0.56, 0.0, 1.0)
        elif preset == 'CHASE':
            add_c(0.0, 1.0, 0.0)
            add_c(0.0, 0.1, 0.0)
        elif preset == 'PULSE':
            add_c(1.0, 0.2, 0.5)
        elif preset == 'COLOR_CYCLE':
            add_c(1.0, 0.0, 0.0)
            add_c(0.0, 0.0, 1.0)
        else:
            add_c(1.0, 1.0, 1.0)

        props.global_expr_active_color_index = 0
        return {'FINISHED'}


class DRONE_OT_apply_global_expression(bpy.types.Operator):
    """Apply the global expression to LightBenders"""
    bl_idname = "drone.apply_global_expression"
    bl_label = "Apply Global Expression"
    bl_options = {'REGISTER', 'UNDO'}

    apply_to_all: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        import math
        import re
        scene = context.scene
        props = scene.drone_props
        
        if not props.global_expr_colors:
            self.report({'ERROR'}, "Please add at least one color to the palette.")
            return {'CANCELLED'}

        # Format colors into a valid python list string [ [r,g,b], [r,g,b] ]
        c_list = []
        for c in props.global_expr_colors:
            r = int(c.color[0] * 255)
            g = int(c.color[1] * 255)
            b = int(c.color[2] * 255)
            c_list.append(f"[{r}, {g}, {b}]")
        
        c_str = "[" + ", ".join(c_list) + "]"
        c_len = len(c_list)
        s = props.global_expr_speed
        
        preset = props.global_expr_preset
        formula = ""
        
        if preset == 'STATIC':
            formula = f"{c_str}[0]"
        elif preset == 'BLINK':
            formula = f"{c_str}[int(t * {s}) % {c_len}]"
        elif preset == 'SPARKLE':
            bg = "[0,0,0]" if c_len == 1 else f"{c_str}[1]"
            thr = props.global_expr_sparkle_threshold
            formula = f"{c_str}[0] if random.random() < {thr} else {bg}"
        elif preset == 'RAINBOW':
            formula = f"{c_str}[int((i * 0.5 + t * {s}) % {c_len})]"
        elif preset == 'CHASE':
            bg = "[0,0,0]" if c_len == 1 else f"{c_str}[1]"
            formula = f"{c_str}[0] if int(i*0.5 - t*{s}) % 10 < 3 else {bg}"
        elif preset == 'PULSE':
            r = int(props.global_expr_colors[0].color[0] * 255) if c_len > 0 else 255
            g = int(props.global_expr_colors[0].color[1] * 255) if c_len > 0 else 255
            b = int(props.global_expr_colors[0].color[2] * 255) if c_len > 0 else 255
            m = f"(0.5 + 0.5 * math.sin(t * {s}))"
            formula = f"[{r} * {m}, {g} * {m}, {b} * {m}]"
        elif preset == 'COLOR_CYCLE':
            c1 = props.global_expr_colors[0].color if c_len > 0 else (1.0, 0.0, 0.0)
            c2 = props.global_expr_colors[1].color if c_len > 1 else (0.0, 0.0, 1.0)
            r1, g1, b1 = int(c1[0]*255), int(c1[1]*255), int(c1[2]*255)
            r2, g2, b2 = int(c2[0]*255), int(c2[1]*255), int(c2[2]*255)
            w1 = f"(0.5+0.5*math.sin({s}*t))"
            w2 = f"(0.5+0.5*math.cos({s}*t))"
            formula = (
                f"[int({w1}*{r1}+{w2}*{r2}),"
                f"int({w1}*{g1}+{w2}*{g2}),"
                f"int({w1}*{b1}+{w2}*{b2})]"
            )

        # Apply to drones
        drones_to_affect = []
        if self.apply_to_all:
            for obj in scene.objects:
                if re.match(r"^lb\d+", obj.name) and "servo_1" in obj:
                    drones_to_affect.append(obj)
        else:
            for obj in context.selected_objects:
                if re.match(r"^lb\d+", obj.name) and "servo_1" in obj:
                    drones_to_affect.append(obj)
                    
        if not drones_to_affect:
            self.report({'WARNING'}, "No matching drones found to apply.")
            return {'CANCELLED'}

        for lb in drones_to_affect:
            ob_props = lb.drone_props
            if ob_props.led_mode == 'EXPRESSION':
                ob_props.led_formula = formula
            else:
                target_ptr = None
                if ob_props.led_pointers:
                    target_ptr = ob_props.led_pointers[0]
                    for ptr in ob_props.led_pointers:
                        compact_expr = ptr.color_expression.replace(" ", "")
                        if compact_expr != "[0,0,0]":
                            target_ptr = ptr
                            break
                    target_ptr.color_expression = formula
                else:
                    ob_props.led_base_color = formula
                    
        context.view_layer.update()
        self.report({'INFO'}, f"Applied '{preset}' expression to {len(drones_to_affect)} LightBenders.")
        return {'FINISHED'}


def get_endpoints(lb_obj, v_min, v_max):
    """Helper to find the 3D coordinates of the active ends of a LightBender"""
    leds = []

    def find_leds(obj):
        if "led_index" in obj: leds.append(obj)
        for c in obj.children: find_leds(c)

    find_leds(lb_obj)
    if not leds: return None, None
    leds.sort(key=lambda x: x["led_index"])

    idx_min = max(0, min(len(leds) - 1, int(round(v_min))))
    idx_max = max(0, min(len(leds) - 1, int(round(v_max))))

    return leds[idx_min].matrix_world.translation, leds[idx_max].matrix_world.translation


def get_interpolated_value(time_dict, t):
    """Linear interpolation helper for time folding"""
    times = sorted(time_dict.keys())
    if not times: return 0
    if t <= times[0]: return time_dict[times[0]]
    if t >= times[-1]: return time_dict[times[-1]]
    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i + 1]
        if t1 <= t <= t2:
            v1, v2 = time_dict[t1], time_dict[t2]
            return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
    return 0


class DRONE_OT_generate_draw_erase(bpy.types.Operator):
    """Automatically generate keyframes for drawing and erasing"""
    bl_idname = "drone.generate_draw_erase"
    bl_label = "Generate Draw/Erase Animation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        props = scene.drone_props
        fps = scene.render.fps

        if not props.de_groups:
            self.report({'WARNING'}, "No groups defined.")
            return {'CANCELLED'}

        # 1. Setup Data & Clean Existing Keys
        all_lbs = set()
        for group in props.de_groups:
            for item in group.drones:
                if item.obj_ptr:
                    all_lbs.add(item.obj_ptr)

        lb_ranges = {}

        for lb in all_lbs:
            N = lb.get("led_count", 50)

            # Read existing bounds
            if len(lb.drone_props.led_pointers) >= 2:
                vals = [p.value for p in lb.drone_props.led_pointers]
                v_min = min(vals)
                v_max = max(vals)
                while len(lb.drone_props.led_pointers) > 2:
                    lb.drone_props.led_pointers.remove(len(lb.drone_props.led_pointers) - 1)
            elif len(lb.drone_props.led_pointers) == 1:
                v_min = 0.0
                v_max = lb.drone_props.led_pointers[0].value
                lb.drone_props.led_pointers.add()
            else:
                v_min = 0.0
                v_max = float(N)
                lb.drone_props.led_pointers.add()
                lb.drone_props.led_pointers.add()

            # Setup base and active colors
            lb.drone_props.led_mode = 'POINTERS'
            lb.drone_props.led_base_color = props.de_bg_color
            lb.drone_props.led_pointers[0].color_expression = props.de_draw_color
            lb.drone_props.led_pointers[1].color_expression = props.de_bg_color

            lb_ranges[lb] = (v_min, v_max)

            # Save originals once so Clear can restore them (don't overwrite on re-generate)
            if "de_orig_vmin" not in lb:
                lb["de_orig_vmin"] = v_min
            if "de_orig_vmax" not in lb:
                lb["de_orig_vmax"] = v_max

            # Remove old keyframes for pointers
            if lb.animation_data and lb.animation_data.action:
                fc1 = lb.animation_data.action.fcurves.find('drone_props.led_pointers[0].value')
                if fc1: lb.animation_data.action.fcurves.remove(fc1)
                fc2 = lb.animation_data.action.fcurves.find('drone_props.led_pointers[1].value')
                if fc2: lb.animation_data.action.fcurves.remove(fc2)

        # 2. Time Logic & Pathfinding
        current_time = 0.0
        kf_commands = []  # List of (lb, pointer_index, time, value)

        iterations = len(props.de_groups)
        if props.de_loop and iterations > 0:
            iterations += 1  # Add a wrap iteration

        loop_time_point = 0.0

        for g_idx in range(iterations):
            prev_connected_end = None

            is_wrap_iteration = (g_idx == len(props.de_groups))
            real_g_idx = 0 if is_wrap_iteration else g_idx

            group = props.de_groups[real_g_idx]
            valid_pairs = [(item, item.obj_ptr) for item in group.drones if item.obj_ptr]
            valid_lbs = [pair[1] for pair in valid_pairs]
            valid_items = [pair[0] for pair in valid_pairs]
            if not valid_lbs: continue

            if is_wrap_iteration:
                loop_time_point = current_time

            group_start_time = current_time
            group_draw_end = current_time
            directions = []  # Store to replay in erase phase

            # --- DRAW PHASE ---
            if group.draw_mode == 'SEQUENTIAL':
                for i, (lb_item, lb) in enumerate(valid_pairs):
                    v_min, v_max = lb_ranges[lb]
                    p0, pmax = get_endpoints(lb, v_min, v_max)
                    if not p0:
                        directions.append("FORWARD")
                        continue

                    # Respect manual direction override; otherwise use pathfinding
                    if lb_item.direction != 'AUTO':
                        direction = lb_item.direction
                        connected_end = pmax if direction == 'FORWARD' else p0
                    else:
                        # Heuristic Pathfinding based on active bounds
                        if i == 0:
                            if prev_connected_end:
                                d0 = (p0 - prev_connected_end).length
                                dm = (pmax - prev_connected_end).length
                                direction = "FORWARD" if d0 < dm else "BACKWARD"
                                connected_end = pmax if direction == "FORWARD" else p0
                            else:
                                if len(valid_lbs) > 1:
                                    next_lb = valid_lbs[1]
                                    next_vmin, next_vmax = lb_ranges[next_lb]
                                    p0_next, pmax_next = get_endpoints(next_lb, next_vmin, next_vmax)
                                    if p0_next:
                                        d00 = (p0 - p0_next).length
                                        d0m = (p0 - pmax_next).length
                                        dm0 = (pmax - p0_next).length
                                        dmm = (pmax - pmax_next).length
                                        min_d = min(d00, d0m, dm0, dmm)
                                        if min_d in (d00, d0m):
                                            direction = "BACKWARD"
                                            connected_end = p0
                                        else:
                                            direction = "FORWARD"
                                            connected_end = pmax
                                    else:
                                        direction = "FORWARD"
                                        connected_end = pmax
                                else:
                                    direction = "FORWARD"
                                    connected_end = pmax
                        else:
                            d0 = (p0 - connected_end).length
                            dm = (pmax - connected_end).length
                            direction = "FORWARD" if d0 < dm else "BACKWARD"
                            connected_end = pmax if direction == "FORWARD" else p0
                        # Store auto-determined direction so it can be seen and reversed in the UI
                        lb_item.direction = direction

                    directions.append(direction)

                    # Duration relative to actual range length
                    dur = (v_max - v_min) / props.de_draw_speed
                    if dur <= 0: dur = 0.01

                    self.report({'INFO'}, f"{lb} {direction}.")

                    if direction == "FORWARD":
                        kf_commands.append((lb, 0, current_time, v_min))
                        kf_commands.append((lb, 0, current_time + dur, v_min))
                        kf_commands.append((lb, 1, current_time, v_min))
                        kf_commands.append((lb, 1, current_time + dur, v_max))
                    else:
                        kf_commands.append((lb, 0, current_time, v_max))
                        kf_commands.append((lb, 0, current_time + dur, v_min))
                        kf_commands.append((lb, 1, current_time, v_max))
                        kf_commands.append((lb, 1, current_time + dur, v_max))

                    current_time += dur

                group_draw_end = current_time
                prev_connected_end = connected_end

            else:  # SIMULTANEOUS
                max_dur = 0
                for lb_item, lb in valid_pairs:
                    v_min, v_max = lb_ranges[lb]
                    p0, pmax = get_endpoints(lb, v_min, v_max)
                    if not p0:
                        directions.append("FORWARD")
                        continue

                    if lb_item.direction != 'AUTO':
                        direction = lb_item.direction
                    else:
                        if prev_connected_end:
                            d0 = (p0 - prev_connected_end).length
                            dm = (pmax - prev_connected_end).length
                            direction = "FORWARD" if d0 < dm else "BACKWARD"
                        else:
                            direction = "FORWARD"
                        lb_item.direction = direction

                    directions.append(direction)
                    dur = (v_max - v_min) / props.de_draw_speed
                    if dur <= 0: dur = 0.01
                    max_dur = max(max_dur, dur)

                    if direction == "FORWARD":
                        kf_commands.append((lb, 0, current_time, v_min))
                        kf_commands.append((lb, 0, current_time + dur, v_min))
                        kf_commands.append((lb, 1, current_time, v_min))
                        kf_commands.append((lb, 1, current_time + dur, v_max))
                    else:
                        kf_commands.append((lb, 0, current_time, v_max))
                        kf_commands.append((lb, 0, current_time + dur, v_min))
                        kf_commands.append((lb, 1, current_time, v_max))
                        kf_commands.append((lb, 1, current_time + dur, v_max))

                current_time += max_dur
                group_draw_end = current_time
                if valid_lbs:
                    last_lb = valid_lbs[-1]
                    l_vmin, l_vmax = lb_ranges[last_lb]
                    p0, pmax = get_endpoints(last_lb, l_vmin, l_vmax)
                    if p0: prev_connected_end = pmax

            # --- ERASE PHASE ---
            group_erase_start = group_draw_end + props.de_hold_time
            current_erase_time = group_erase_start

            if group.draw_mode == 'SEQUENTIAL':
                for i, lb in enumerate(valid_lbs):
                    direction = directions[i]
                    v_min, v_max = lb_ranges[lb]
                    dur = (v_max - v_min) / props.de_erase_speed
                    if dur <= 0: dur = 0.01

                    if direction == "FORWARD":
                        kf_commands.append((lb, 0, current_erase_time, v_min))
                        kf_commands.append((lb, 0, current_erase_time + dur, v_max))
                        kf_commands.append((lb, 1, current_erase_time, v_max))
                        kf_commands.append((lb, 1, current_erase_time + dur, v_max))
                    else:
                        kf_commands.append((lb, 0, current_erase_time, v_min))
                        kf_commands.append((lb, 0, current_erase_time + dur, v_min))
                        kf_commands.append((lb, 1, current_erase_time, v_max))
                        kf_commands.append((lb, 1, current_erase_time + dur, v_min))

                    current_erase_time += dur
                group_erase_end = current_erase_time
            else:
                max_erase_dur = 0
                for i, lb in enumerate(valid_lbs):
                    direction = directions[i]
                    v_min, v_max = lb_ranges[lb]
                    dur = (v_max - v_min) / props.de_erase_speed
                    if dur <= 0: dur = 0.01
                    max_erase_dur = max(max_erase_dur, dur)

                    if direction == "FORWARD":
                        kf_commands.append((lb, 0, current_erase_time, v_min))
                        kf_commands.append((lb, 0, current_erase_time + dur, v_max))
                        kf_commands.append((lb, 1, current_erase_time, v_max))
                        kf_commands.append((lb, 1, current_erase_time + dur, v_max))
                    else:
                        kf_commands.append((lb, 0, current_erase_time, v_min))
                        kf_commands.append((lb, 0, current_erase_time + dur, v_min))
                        kf_commands.append((lb, 1, current_erase_time, v_max))
                        kf_commands.append((lb, 1, current_erase_time + dur, v_min))

                current_erase_time += max_erase_dur
                group_erase_end = current_erase_time

            if is_wrap_iteration:
                break

            overlap_sec = (group_erase_end - group_erase_start) * (props.de_overlap / 100.0)
            current_time = group_erase_end - overlap_sec

        # 3. Collapse Tracks and Handle Looping
        tracks = {}
        for lb, ptr_idx, t, val in kf_commands:
            key = (lb, ptr_idx)
            if key not in tracks: tracks[key] = {}
            tracks[key][t] = val

        if props.de_loop and loop_time_point > 0:
            for key, time_dict in tracks.items():
                v_loop = get_interpolated_value(time_dict, loop_time_point)
                new_dict = {}
                for t, val in time_dict.items():
                    if t <= loop_time_point:
                        new_dict[t] = val
                    else:
                        # Time folding wrap
                        wrap_t = t - loop_time_point
                        if wrap_t not in new_dict:
                            new_dict[wrap_t] = val

                new_dict[0.0] = v_loop
                new_dict[loop_time_point] = v_loop
                tracks[key] = new_dict

            scene.frame_end = int(loop_time_point * fps)
            scene.frame_start = 0
        else:
            max_t = 0
            for td in tracks.values():
                if td: max_t = max(max_t, max(td.keys()))
            scene.frame_end = int(max_t * fps)

        # 4. Insert Final Keyframes
        for (lb, ptr_idx), time_dict in tracks.items():
            if not lb.animation_data: lb.animation_data_create()
            if not lb.animation_data.action: lb.animation_data.action = bpy.data.actions.new(name=f"{lb.name}_DE")

            ptr = lb.drone_props.led_pointers[ptr_idx]
            for t, val in sorted(time_dict.items()):
                frame = int(round(t * fps))
                ptr.value = val
                ptr.keyframe_insert(data_path="value", frame=frame)

            fc = lb.animation_data.action.fcurves.find(f'drone_props.led_pointers[{ptr_idx}].value')
            if fc:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

        self.report({'INFO'}, "Successfully generated draw/erase keyframes.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Position Error & Drift Simulator
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
    restored_action = False

    if "deconflict_orig_action" in obj:
        action_name = obj["deconflict_orig_action"]
        if action_name in bpy.data.actions:
            obj.animation_data.action = bpy.data.actions[action_name]
            restored_action = True
            restored = True
        del obj["deconflict_orig_action"]

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
        del obj["deconflict_scale_factor"]
        if scale_factor > 0.0 and scale_factor != 1.0:
            if not restored_action:
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
        drones = []
        for obj in scene.objects:
            m = re.match(r"^lb(\d+)", obj.name)
            if m:
                drones.append(obj)

                # Save original if not already saved
                if "deconflict_orig_x" not in obj:
                    obj["deconflict_orig_x"] = obj.location.x
                    obj["deconflict_orig_y"] = obj.location.y
                    obj["deconflict_orig_z"] = obj.location.z

                # Save original action if not already saved
                if "deconflict_orig_action" not in obj:
                    if obj.animation_data and obj.animation_data.action:
                        orig_act = obj.animation_data.action.copy()
                        orig_act.use_fake_user = True
                        obj["deconflict_orig_action"] = orig_act.name

        if not drones:
            self.report({'WARNING'}, "No lb# drones found.")
            return {'CANCELLED'}

        # Find frames with location keyframes
        frames_to_stagger = set()
        for obj in drones:
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    if fc.data_path == "location":
                        for kp in fc.keyframe_points:
                            frames_to_stagger.add(int(round(kp.co.x)))

        frames_to_stagger = sorted(list(frames_to_stagger))
        if not frames_to_stagger:
            frames_to_stagger = [scene.frame_current]

        import tempfile
        try:
            addon_dir = os.path.dirname(os.path.realpath(__file__))
        except NameError:
            addon_dir = f"{AUTHORING_DIR}"

        input_yaml = os.path.join(tempfile.gettempdir(), "temp_deconflict_in.yaml")
        output_yaml = os.path.join(tempfile.gettempdir(), "temp_deconflict_out.yaml")

        deconflict_script = os.path.join(addon_dir, "deconflict.py")
        if not os.path.exists(deconflict_script):
            alt_path = os.path.join(AUTHORING_DIR, "deconflict.py")
            if os.path.exists(alt_path):
                deconflict_script = alt_path
            else:
                self.report({'ERROR'}, f"Cannot find deconflict.py at {deconflict_script}")
                return {'CANCELLED'}

        original_frame = scene.frame_current
        applied_at_least_once = False

        if len(frames_to_stagger) > 1:
            for frame in frames_to_stagger:
                scene.frame_set(frame)
                for obj in drones:
                    for ptr in obj.drone_props.led_pointers:
                        ptr.keyframe_insert(data_path="value", frame=frame)

        for frame in frames_to_stagger:
            scene.frame_set(frame)

            points = []
            for obj in drones:
                m = re.match(r"^lb(\d+)", obj.name)
                d_id = int(m.group(1))

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
                scene.frame_set(original_frame)
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
                res = subprocess.run(["/bin/zsh", "-l", "-c", cmd_str], check=False, capture_output=True, text=True)
                if res.returncode != 0:
                    short_err = res.stderr.strip().split('\n')[-1] if res.stderr else "Unknown error"
                    self.report({'ERROR'}, f"Deconflict failed on frame {frame}: {short_err}")
                    print(f"Deconflict error on frame {frame}:\n{res.stderr}")
                    continue
            except Exception as e:
                self.report({'ERROR'}, f"Failed to run script: {e}")
                scene.frame_set(original_frame)
                return {'CANCELLED'}

            if not os.path.exists(output_yaml):
                continue

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
                print(f"Failed to parse output on frame {frame}: {e}")
                continue

            for obj in drones:
                m = re.match(r"^lb(\d+)", obj.name)
                d_id = int(m.group(1))
                if d_id in out_points:
                    p = out_points[d_id]
                    obj.location.x = p.get('x', obj.location.x)
                    obj.location.y = p.get('y', obj.location.y)
                    obj.location.z = p.get('z', obj.location.z)
                    
                    if len(frames_to_stagger) > 1:
                        obj.keyframe_insert(data_path="location", frame=frame)

                    scale_factor = p.get('scale_factor', 1.0)
                    if scale_factor != 1.0:
                        obj["deconflict_scale_factor"] = scale_factor
                        for ptr in obj.drone_props.led_pointers:
                            ptr.value = 25.0 + (ptr.value - 25.0) * scale_factor
                            if len(frames_to_stagger) > 1:
                                ptr.keyframe_insert(data_path="value", frame=frame)

                        # Scale pointer keyframes manually only if single frame
                        if len(frames_to_stagger) == 1:
                            if obj.animation_data and obj.animation_data.action:
                                for fc in obj.animation_data.action.fcurves:
                                    if "led_pointers" in fc.data_path and fc.data_path.endswith(".value"):
                                        for kp in fc.keyframe_points:
                                            kp.co.y = 25.0 + (kp.co.y - 25.0) * scale_factor
                                            kp.handle_left.y = 25.0 + (kp.handle_left.y - 25.0) * scale_factor
                                            kp.handle_right.y = 25.0 + (kp.handle_right.y - 25.0) * scale_factor
                                        fc.update()
                    applied_at_least_once = True

        scene.frame_set(original_frame)

        if applied_at_least_once:
            context.view_layer.update()
            if len(frames_to_stagger) > 1:
                self.report({'INFO'}, f"Staggered {len(drones)} drones across {len(frames_to_stagger)} keyframes.")
            else:
                self.report({'INFO'}, f"Staggered {len(drones)} drones.")
        else:
            self.report({'WARNING'}, "Stagger did not apply to any drones.")

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

        context.scene.frame_set(context.scene.frame_current)
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
    ctx = {"i": i, "t": t, "N": N, "math": math, "random": random}

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
        
        if scene.camera:
            cam_loc = scene.camera.matrix_world.translation
            output_lines.append(f"camera: [{round(cam_loc.x, 4)}, {round(cam_loc.y, 4)}, {round(cam_loc.z, 4)}]")

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
    """Export Drone Animation to YAML and Run Orchestrator (Illumination mode)"""
    bl_idname = "drone.export_and_illuminate"
    bl_label = "Illuminate"

    def execute(self, context):
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

        target_yaml = os.path.join(ORCHESTRATOR_DIR, "SFL", "blender_mission.yaml")
        target_script = os.path.join(ORCHESTRATOR_DIR, "orchestrator.py")

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_yaml), exist_ok=True)

        res = bpy.ops.drone.export_yaml('EXEC_DEFAULT', filepath=target_yaml)
        if 'FINISHED' not in res:
            self.report({'ERROR'}, "Failed to generate YAML")
            return {'CANCELLED'}

        python_exec = get_controller_python()
        cmd_str = f"'{python_exec}' '{target_script}' --illumination --skip-confirm"
        if props.export_dark_room:
            cmd_str += " --dark"

        # Open Mac Terminal – capture the tab reference for stop button
        apple_script = f'''
        tell application "Terminal"
            set newTab to do script "cd \\"{os.path.dirname(target_script)}\\" && {cmd_str}"
            activate
            return id of window 1
        end tell
        '''

        try:
            result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
            _ILLUMINATE_TERMINAL_TAB["tab_ref"] = result.stdout.strip()
            props.illuminate_running = True
            self.report({'INFO'}, "Exported and started orchestrator")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run orchestrator: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class DRONE_OT_stop_illuminate(bpy.types.Operator):
    """Send stop command to the orchestrator via TCP"""
    bl_idname = "drone.stop_illuminate"
    bl_label = "Stop Illumination"

    def execute(self, context):
        props = context.scene.drone_props

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(('127.0.0.1', 5599))
            s.sendall(b'Blender commanded stop\n')
            s.close()
            self.report({'INFO'}, "Sent stop signal to orchestrator")
        except Exception as e:
            self.report({'WARNING'}, f"Could not reach orchestrator: {e}")

        props.illuminate_running = False
        _ILLUMINATE_TERMINAL_TAB["tab_ref"] = None
        return {'FINISHED'}


class EXPORT_OT_interaction_yaml(bpy.types.Operator):
    """Export static interaction mission to YAML"""
    bl_idname = "drone.export_interaction_yaml"
    bl_label = "Export Interaction YAML"

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        props = context.scene.drone_props
        self.filepath = get_default_interaction_yaml_path(props.export_mission_name.strip() or "blender_mission")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        drones_to_export = [obj for obj in context.selected_objects if "servo_1" in obj and "servo_2" in obj]

        if not drones_to_export:
            self.report({'ERROR'}, "No Drones selected")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write(build_interaction_yaml_text(scene, drones_to_export, include_blender_port=False))
            self.report({'INFO'}, f"Exported interaction YAML to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"File Write Error: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}


class EXPORT_OT_export_and_interact(bpy.types.Operator):
    """Export interaction YAML and run orchestrator in interaction mode"""
    bl_idname = "drone.export_and_interact"
    bl_label = "Illuminate (Interaction)"

    def execute(self, context):
        scene = context.scene
        props = scene.drone_props
        drones_to_export = get_lb_scene_objects(scene)

        if not drones_to_export:
            self.report({'ERROR'}, "No lb# drones found in scene")
            return {'CANCELLED'}

        target_yaml = os.path.join(ORCHESTRATOR_DIR, "SFL", (props.export_mission_name.strip() or "blender_mission") + ".yaml")
        target_script = os.path.join(ORCHESTRATOR_DIR, "orchestrator.py")

        try:
            os.makedirs(os.path.dirname(target_yaml), exist_ok=True)
            with open(target_yaml, 'w', encoding='utf-8') as f:
                f.write(build_interaction_yaml_text(scene, drones_to_export, include_blender_port=True))
        except Exception as e:
            self.report({'ERROR'}, f"Failed to generate interaction YAML: {e}")
            return {'CANCELLED'}

        try:
            launch_orchestrator(
                target_script,
                ["--interaction", "--intractable-illumination"],
                dark_room=props.export_dark_room
            )
            props.illuminate_running = True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run orchestrator: {e}")
            return {'CANCELLED'}

        # Open TCP listener so LightBenders can connect before Edit is pressed
        close_static_interaction_session()
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(("", props.interaction_tcp_port))
            server_socket.listen()
            server_socket.setblocking(False)
        except OSError as e:
            self.report({'WARNING'}, f"TCP listener failed on port {props.interaction_tcp_port}: {e}")
            return {'FINISHED'}

        STATIC_INTERACTION_SESSION["server_socket"] = server_socket
        props.interaction_editor_active = True
        props.interaction_connected_count = 0
        props.interaction_recording_active = False

        if not STATIC_INTERACTION_SESSION["timer_registered"]:
            bpy.app.timers.register(static_interaction_timer, first_interval=0.1)
            STATIC_INTERACTION_SESSION["timer_registered"] = True

        self.report({'INFO'}, f"Exported interaction YAML, started orchestrator, listening on port {props.interaction_tcp_port}")
        return {'FINISHED'}


class DRONE_OT_start_edit(bpy.types.Operator):
    """Send start_edit messages to LightBenders from swarm manifest"""
    bl_idname = "drone.start_edit"
    bl_label = "Edit"

    def execute(self, context):
        scene = context.scene
        props = scene.drone_props

        if not STATIC_INTERACTION_SESSION["sockets"]:
            self.report({'WARNING'}, "No LightBender connections available.")
            return {'CANCELLED'}

        # Send start_edit over existing inbound connections
        duration = props.interaction_duration
        edit_msg = json.dumps({"cmd": "start_edit", "duration": duration}) + "\n"

        failed_sends = []
        for key, sock_obj in list(STATIC_INTERACTION_SESSION["sockets"].items()):
            try:
                sock_obj.sendall(edit_msg.encode('utf-8'))
            except (BlockingIOError, BrokenPipeError, OSError):
                failed_sends.append(key)

        if failed_sends:
            self.report({'WARNING'}, f"Could not send start_edit to: {', '.join(failed_sends)}")

        props.edit_active = True
        sent_count = len(STATIC_INTERACTION_SESSION["sockets"]) - len(failed_sends)
        self.report({'INFO'}, f"Edit started. Sent start_edit to {sent_count} connection(s).")
        return {'FINISHED'}


class DRONE_OT_stop_edit(bpy.types.Operator):
    """Record positions from TCP responses and stop edit mode"""
    bl_idname = "drone.stop_edit"
    bl_label = "Finish Edit"

    def execute(self, context):
        scene = context.scene
        props = scene.drone_props

        if not props.interaction_editor_active:
            self.report({'ERROR'}, "Edit session is not active.")
            return {'CANCELLED'}

        if props.interaction_recording_active:
            self.report({'WARNING'}, "Already waiting for position responses.")
            return {'CANCELLED'}

        # Request positions (same as record_interaction_positions)
        if STATIC_INTERACTION_SESSION["sockets"]:
            flush_static_interaction_messages()
            request_payload = json.dumps({"cmd": "request_position"}) + "\n"
            failed_sends = []
            for key, sock_obj in list(STATIC_INTERACTION_SESSION["sockets"].items()):
                try:
                    sock_obj.sendall(request_payload.encode('utf-8'))
                except (BlockingIOError, BrokenPipeError, OSError):
                    failed_sends.append(key)

            STATIC_INTERACTION_SESSION["recording"] = True
            STATIC_INTERACTION_SESSION["record_deadline"] = time.monotonic() + 0.1
            STATIC_INTERACTION_SESSION["record_updated"] = set()
            STATIC_INTERACTION_SESSION["record_received_any"] = False
            STATIC_INTERACTION_SESSION["record_failed_sends"] = failed_sends
            props.interaction_recording_active = True

        # End the edit session
        props.edit_active = False
        self.report({'INFO'}, "Edit stopped. Positions recorded.")
        return {'FINISHED'}


class DRONE_OT_stop_illuminate_interaction(bpy.types.Operator):
    """Stop illumination in interaction mode via TCP command"""
    bl_idname = "drone.stop_illuminate_interaction"
    bl_label = "Stop Illumination"

    def execute(self, context):
        props = context.scene.drone_props

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(('127.0.0.1', 5599))
            s.sendall(b'Blender commanded stop\n')
            s.close()
            self.report({'INFO'}, "Sent stop signal to orchestrator")
        except Exception as e:
            self.report({'WARNING'}, f"Could not reach orchestrator: {e}")

        # Clean up interaction session
        close_static_interaction_session()
        props.illuminate_running = False
        props.interaction_editor_active = False
        props.interaction_connected_count = 0
        props.edit_active = False
        _ILLUMINATE_TERMINAL_TAB["tab_ref"] = None
        return {'FINISHED'}


class DRONE_OT_transform_and_place(bpy.types.Operator):
    """Convert SVG to LightBender layout and create drones"""
    bl_idname = "drone.transform_and_place"
    bl_label = "Transform and Place"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import os
        import subprocess
        import tempfile
        import json

        scene = context.scene
        props = scene.drone_props

        svg_path = bpy.path.abspath(props.import_svg_filepath)
        if not os.path.isfile(svg_path):
            self.report({'ERROR'}, f"SVG File not found: {svg_path}")
            return {'CANCELLED'}

        # try:
        #     addon_dir = os.path.dirname(os.path.realpath(__file__))
        # except NameError:
        addon_dir = AUTHORING_DIR
        transform_script = os.path.join(addon_dir, "transform.py")
        place_script = os.path.join(addon_dir, "place.py")

        temp_dir = tempfile.gettempdir()
        target_graph_yaml = os.path.join(temp_dir, "temp_target_graph.yaml")
        target_layout_yaml = os.path.join(temp_dir, "temp_target_layout.yaml")

        # 1. Call transform.py
        cmd_transform = [
            "python3", f"'{transform_script}'",
            "-i", f"'{svg_path}'",
            "-o", f"'{target_graph_yaml}'",
            "-mw", str(props.import_max_width),
            "-ml", str(props.import_max_length),
            "-cy", str(props.import_cx),
            "-cz", str(props.import_cy)
        ]
        res1 = subprocess.run(["/bin/zsh", "-l", "-c", " ".join(cmd_transform)], capture_output=True, text=True)
        if res1.returncode != 0:
            self.report({'ERROR'}, f"Transform failed: {res1.stderr}")
            return {'CANCELLED'}

        # 2. Call place.py
        cmd_place = [
            "python3", f"'{place_script}'",
            "--input", f"'{target_graph_yaml}'",
            "--output", f"'{target_layout_yaml}'",
            "--policy", props.import_policy,
            "--max_len", "0.16",
            "--no_viz"
        ]
        res2 = subprocess.run(["/bin/zsh", "-l", "-c", " ".join(cmd_place)], capture_output=True, text=True)
        if res2.returncode != 0:
            self.report({'ERROR'}, f"Place failed: {res2.stderr}")
            return {'CANCELLED'}

        if not os.path.exists(target_layout_yaml):
            self.report({'ERROR'}, "Placement layout file not generated")
            return {'CANCELLED'}

        def load_yaml_safe(filepath):
            try:
                # Use external Python to parse YAML and dump as JSON since Blender might lack PyYAML
                cmd_yaml2json = f"python3 -c \"import yaml, json, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))\" '{filepath}'"
                json_str = subprocess.check_output(["/bin/zsh", "-l", "-c", cmd_yaml2json], text=True)
                return json.loads(json_str)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to parse YAML file {filepath}: {e}")
                return {}

        layout_data = load_yaml_safe(target_layout_yaml)
        points = layout_data.get('points', [])

        # 3. Process Manifest
        inventory = {'H': [], 'V': []}
        manifest_path = bpy.path.abspath(props.import_manifest_filepath)
        if os.path.isfile(manifest_path):
            manifest_data = load_yaml_safe(manifest_path)
            for d in manifest_data.get('drones', []):
                t = d.get('type')
                if t in ('H', 'V'):
                    inventory[t].append(d['id'])

        # 4. Angle Logic & Assignment
        def get_type_matches(a1, a2, l1, l2):
            def fits_H(a, b):
                am = a % 360
                bm = b % 360

                # type H: 0 <= angle_1 <= 180, 180 <= angle_2 <= 360
                ok_a = (l1 == 0) or (0 <= am <= 180)
                bs = 360 if bm == 0 else bm
                ok_b = (l2 == 0) or (180 <= bs <= 360)

                if ok_a and ok_b: return (am, bs)
                return None

            def fits_V(a, b):
                am = a % 360
                bm = b % 360

                # type V: 90 <= angle_1 <= 270, 270 <= angle_2 <= 450
                ok_a = (l1 == 0) or (90 <= am <= 270)
                bs = bm if bm > 270 else bm + 360
                ok_b = (l2 == 0) or (270 <= bs <= 450)

                if ok_a and ok_b: return (am, bs)
                return None

            matches = {}
            res_h = fits_H(a1, a2)
            if res_h: matches['H'] = (res_h[0], res_h[1], l1, l2, False)
            res_v = fits_V(a1, a2)
            if res_v: matches['V'] = (res_v[0], res_v[1], l1, l2, False)

            # Allow swapping parameters
            res_h_s = fits_H(a2, a1)
            if res_h_s and 'H' not in matches: matches['H'] = (res_h_s[0], res_h_s[1], l2, l1, True)
            res_v_s = fits_V(a2, a1)
            if res_v_s and 'V' not in matches: matches['V'] = (res_v_s[0], res_v_s[1], l2, l1, True)

            return matches

        assignments = [None] * len(points)
        created_counts = {'H': 0, 'V': 0}

        ambiguous = []
        for i, pt in enumerate(points):
            matches = get_type_matches(pt['angle_1'], pt['angle_2'], pt['length_1'], pt['length_2'])
            if len(matches) == 1:
                t = list(matches.keys())[0]
                assignments[i] = (t, matches[t])
            elif len(matches) == 0:
                t = 'H' if len(inventory['H']) >= len(inventory['V']) else 'V'
                assignments[i] = (t, (pt['angle_1'] % 360, pt['angle_2'] % 360, pt['length_1'], pt['length_2'], False))
            else:
                ambiguous.append((i, matches))

        # Handle ambiguous by assigning them to whichever type has most remaining inventory
        for i, matches in ambiguous:
            t = 'H' if len(inventory['H']) >= len(inventory['V']) else 'V'
            assignments[i] = (t, matches[t])
            if len(inventory[t]) > 0:
                inventory[t].pop(0)  # pop from front
            else:
                created_counts[t] += 1

        # Refill inventory arrays for actual generation since we mutated it
        inventory = {'H': [], 'V': []}
        if os.path.isfile(manifest_path):
            manifest_data = load_yaml_safe(manifest_path)
            for d in manifest_data.get('drones', []):
                t = d.get('type')
                if t in ('H', 'V'):
                    inventory[t].append(d['id'])

        # 5. Spawn Drones
        warnings = []
        created_counts = {'H': 0, 'V': 0}
        for i, pt in enumerate(points):
            t, (a1, a2, l1, l2, swapped) = assignments[i]

            if len(inventory[t]) > 0:
                lb_id = inventory[t].pop(0)
            else:
                created_counts[t] += 1
                lb_id = f"lb_extra_{t}_{created_counts[t]}"
                warnings.append(lb_id)

            bpy.ops.drone.add_drone(drone_type=f'TYPE_{t}')
            obj = context.active_object
            obj.name = lb_id

            obj.location = (pt['x'], pt['y'], pt['z'])

            import math
            obj.rotation_euler.z = math.radians(pt['yaw'])

            obj["servo_1"] = a1
            obj["servo_2"] = a2

            obj.drone_props.led_mode = 'POINTERS'
            obj.drone_props.led_base_color = "[0, 0, 0]"

            obj.drone_props.led_pointers.clear()

            import_color = props.import_color
            max_len = pt.get('max_length_limit', 0.16)

            # The length from middle to the tip of rods
            active_1 = round(25 * (l1 / max_len)) if max_len > 0 else 0
            active_2 = round(25 * (l2 / max_len)) if max_len > 0 else 0

            p1 = obj.drone_props.led_pointers.add()
            p1.value = 25.0 - active_1
            p1.color_expression = import_color

            p2 = obj.drone_props.led_pointers.add()
            p2.value = 25.0 + active_2
            p2.color_expression = "[0, 0, 0]"

        if warnings:
            self.report({'WARNING'},
                        f"Warning: {len(warnings)} LightBenders created with incremental IDs because manifest lacked enough.")
        else:
            self.report({'INFO'}, f"Successfully placed {len(points)} LightBenders from SVG.")

        context.view_layer.update()

        # Cleanup
        try:
            if os.path.exists(target_graph_yaml): os.remove(target_graph_yaml)
            if os.path.exists(target_layout_yaml): os.remove(target_layout_yaml)
        except Exception:
            pass

        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Depth Fly-In Feature
# ------------------------------------------------------------------------

class DRONE_OT_generate_flyin(bpy.types.Operator):
    """Generate fly-in exploded assembly animation"""
    bl_idname = "drone.generate_flyin"
    bl_label = "Generate Fly-In"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import re
        from mathutils import Vector
        from bpy_extras.object_utils import world_to_camera_view
        
        scene = context.scene
        props = scene.drone_props
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "A camera is required for Depth Fly-In.")
            return {'CANCELLED'}

        # Get target drones (lb*)
        lbs = [obj for obj in scene.objects if re.match(r"^lb\d+", obj.name) and "servo_1" in obj]
        if not lbs:
            self.report({'WARNING'}, "No LightBenders found.")
            return {'CANCELLED'}

        # Calculate Centroid
        centroid = Vector((0.0, 0.0, 0.0))
        for lb in lbs:
            centroid += lb.matrix_world.translation
        centroid /= len(lbs)

        # Camera Look Vector (+Z in local space)
        cam_z = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
        cam_z.normalize()

        # Reference Point
        ref_point = centroid - cam_z * props.flyin_ref_distance

        orig_positions = {}
        closest_lb = None
        min_depth = float('inf')

        for lb in lbs:
            pos = lb.matrix_world.translation
            orig_positions[lb] = pos.copy()
            cam_to_lb = pos - cam.matrix_world.translation
            depth = cam_to_lb.dot(cam_z)
            if depth < min_depth:
                min_depth = depth
                closest_lb = lb

        if not closest_lb:
            return {'CANCELLED'}

        # Bounding box constraints
        b_min = Vector((props.flyin_x_min, props.flyin_y_min, props.flyin_z_min))
        b_max = Vector((props.flyin_x_max, props.flyin_y_max, props.flyin_z_max))

        # We need a uniform scale K that keeps everything in bounds.
        # Find maximum K such that all LBs remain in bounding box
        k_target = 1000.0  # start with a large upper bound
        for lb in lbs:
            v_i = orig_positions[lb] - ref_point
            # We check intersection with all 6 planes for this particular ray
            t_candidates = []
            for i in range(3):
                if abs(v_i[i]) > 1e-6:
                    t1 = (b_min[i] - ref_point[i]) / v_i[i]
                    t2 = (b_max[i] - ref_point[i]) / v_i[i]
                    if t1 > 1.0: t_candidates.append(t1)
                    if t2 > 1.0: t_candidates.append(t2)
            if t_candidates:
                k_target = min(k_target, min(t_candidates))

        if k_target < 1.0:
            k_target = 1.0

        # Now we apply this k_target to all LBs and check/adjust for visibility.
        outward_positions = {}
        for lb in lbs:
            v_dir = orig_positions[lb] - ref_point
            curr_pos = ref_point + k_target * v_dir
            
            # Check visibility and adjust "ray" to push outward
            is_visible = True
            adjust_iters = 0
            
            while is_visible and adjust_iters < 50:
                co_ndc = world_to_camera_view(scene, cam, curr_pos)
                if co_ndc.z < 0 or co_ndc.x < -0.1 or co_ndc.x > 1.1 or co_ndc.y < -0.1 or co_ndc.y > 1.1:
                    is_visible = False
                    break
                
                # Push outwards from camera center (NDC 0.5, 0.5)
                push_vector = Vector((co_ndc.x - 0.5, co_ndc.y - 0.5, 0.0))
                if push_vector.length < 1e-4:
                    push_vector = Vector((1.0, 0.0, 0.0))
                push_vector.normalize()
                push_vector *= 0.1  # Step size in NDC
                
                # Move curr_pos locally along camera X/Y
                cam_x = cam.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
                cam_y = cam.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
                
                curr_pos += cam_x * push_vector.x * max(0.1, co_ndc.z)
                curr_pos += cam_y * push_vector.y * max(0.1, co_ndc.z)
                
                adjust_iters += 1
            
            # Clamp to bounds strictly after adjustments
            curr_pos.x = max(b_min.x, min(b_max.x, curr_pos.x))
            curr_pos.y = max(b_min.y, min(b_max.y, curr_pos.y))
            curr_pos.z = max(b_min.z, min(b_max.z, curr_pos.z))
            
            outward_positions[lb] = curr_pos

        # Generate Animation
        fps = scene.render.fps
        t_start = 0.0
        t_out = props.flyin_outward_duration
        t_hold = t_out + props.flyin_hold_duration
        t_in = t_hold + props.flyin_inward_duration

        for lb in lbs:
            if not lb.animation_data: lb.animation_data_create()
            if not lb.animation_data.action: lb.animation_data.action = bpy.data.actions.new(name=f"{lb.name}_FlyIn")
            
            # Save original position to property so we can clear
            if "flyin_orig_x" not in lb:
                lb["flyin_orig_x"] = orig_positions[lb].x
                lb["flyin_orig_y"] = orig_positions[lb].y
                lb["flyin_orig_z"] = orig_positions[lb].z

            out_pos = outward_positions[lb]
            orig_pos = orig_positions[lb]

            # Clear existing location fcurves
            action = lb.animation_data.action
            for i in range(3):
                fc = action.fcurves.find('location', index=i)
                if fc: action.fcurves.remove(fc)

            # Insert keys
            lb.location = orig_pos
            lb.keyframe_insert(data_path="location", frame=max(1, int(t_start * fps)))
            
            lb.location = out_pos
            lb.keyframe_insert(data_path="location", frame=max(1, int(t_out * fps)))
            lb.keyframe_insert(data_path="location", frame=max(1, int(t_hold * fps)))
            
            lb.location = orig_pos
            lb.keyframe_insert(data_path="location", frame=max(1, int(t_in * fps)))

            # Edit pointer expressions
            if lb.drone_props.led_mode == 'POINTERS':
                # Base Color
                base_expr = lb.drone_props.led_base_color
                if "flyin_orig_base" not in lb:
                    lb["flyin_orig_base"] = base_expr
                match_base = re.search(r"\((.*?)\) if t >= \d+\.?\d* else \[0, \s*0, \s*0\]", base_expr)
                if match_base: base_expr = match_base.group(1)
                lb.drone_props.led_base_color = f"({base_expr}) if t >= {t_hold} else [0, 0, 0]"

                for ptr in lb.drone_props.led_pointers:
                    orig_expr = ptr.color_expression
                    if "flyin_orig_expr" not in ptr:
                        ptr["flyin_orig_expr"] = orig_expr
                    match = re.search(r"\((.*?)\) if t >= \d+\.?\d* else \[0, \s*0, \s*0\]", orig_expr)
                    if match: orig_expr = match.group(1)
                    ptr.color_expression = f"({orig_expr}) if t >= {t_hold} else [0, 0, 0]"

        scene.frame_start = 1
        scene.frame_end = int(t_in * fps)
        
        self.report({'INFO'}, "Depth Fly-In generated.")
        return {'FINISHED'}


class DRONE_OT_clear_flyin(bpy.types.Operator):
    """Clear fly-in animation and restore LEDs"""
    bl_idname = "drone.clear_flyin"
    bl_label = "Clear Fly-In"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import re
        lbs = [obj for obj in context.scene.objects if re.match(r"^lb\d+", obj.name)]
        for lb in lbs:
            # Restore location
            if "flyin_orig_x" in lb:
                lb.location.x = lb["flyin_orig_x"]
                lb.location.y = lb["flyin_orig_y"]
                lb.location.z = lb["flyin_orig_z"]
                del lb["flyin_orig_x"]
                del lb["flyin_orig_y"]
                del lb["flyin_orig_z"]
                
                if lb.animation_data and lb.animation_data.action:
                    for i in range(3):
                        fc = lb.animation_data.action.fcurves.find('location', index=i)
                        if fc: lb.animation_data.action.fcurves.remove(fc)

            # Restore explicit property backup
            if "flyin_orig_base" in lb:
                lb.drone_props.led_base_color = lb["flyin_orig_base"]
                del lb["flyin_orig_base"]
            else:
                match_base = re.search(r"\((.*?)\) if t >= \d+\.?\d* else \[0, \s*0, \s*0\]", lb.drone_props.led_base_color)
                if match_base: lb.drone_props.led_base_color = match_base.group(1)

            for ptr in lb.drone_props.led_pointers:
                if "flyin_orig_expr" in ptr:
                    ptr.color_expression = ptr["flyin_orig_expr"]
                    del ptr["flyin_orig_expr"]
                else:
                    match = re.search(r"\((.*?)\) if t >= \d+\.?\d* else \[0, \s*0, \s*0\]", ptr.color_expression)
                    if match: ptr.color_expression = match.group(1)

        self.report({'INFO'}, "Fly-In animation cleared.")
        return {'FINISHED'}


# ------------------------------------------------------------------------
#    Morph Animation
# ------------------------------------------------------------------------

class DRONE_OT_generate_morph(bpy.types.Operator):
    """Morph from current scene layout to a new SVG"""
    bl_idname = "drone.generate_morph"
    bl_label = "Generate Morph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import os
        import math
        import subprocess
        import tempfile
        import json
        import re

        scene = context.scene
        props = scene.drone_props

        svg_path = bpy.path.abspath(props.morph_svg_filepath)
        if not os.path.isfile(svg_path):
            self.report({'ERROR'}, f"Morph SVG not found: {svg_path}")
            return {'CANCELLED'}

        # Ensure we have camera
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "Camera required for deconflict stagger of morph target.")
            return {'CANCELLED'}

        lbs = []
        for obj in scene.objects:
            m = re.match(r"^lb(\d+)", obj.name)
            if m and "servo_1" in obj:
                d_id = int(m.group(1))
                lbs.append((obj, d_id))

        if not lbs:
            self.report({'WARNING'}, "No existing LightBenders found in scene.")
            return {'CANCELLED'}

        addon_dir = AUTHORING_DIR
        temp_dir = tempfile.gettempdir()
        target_graph_yaml = os.path.join(temp_dir, "temp_morph_graph.yaml")
        target_layout_yaml = os.path.join(temp_dir, "temp_morph_layout.yaml")
        p1_json = os.path.join(temp_dir, "morph_p1.json")
        p2_json = os.path.join(temp_dir, "morph_p2.json")
        match_out_json = os.path.join(temp_dir, "morph_match_out.json")

        cmd_transform = [
            "python3", f"'{os.path.join(addon_dir, 'transform.py')}'",
            "-i", f"'{svg_path}'", "-o", f"'{target_graph_yaml}'",
            "-mw", str(props.import_max_width), "-ml", str(props.import_max_length),
            "-cy", str(props.import_cx), "-cz", str(props.import_cy)
        ]
        res1 = subprocess.run(["/bin/zsh", "-l", "-c", " ".join(cmd_transform)], capture_output=True, text=True)
        if res1.returncode != 0:
            self.report({'ERROR'}, f"Transform failed: {res1.stderr}")
            return {'CANCELLED'}

        cmd_place = [
            "python3", f"'{os.path.join(addon_dir, 'place.py')}'",
            "--input", f"'{target_graph_yaml}'", "--output", f"'{target_layout_yaml}'",
            "--policy", props.import_policy, "--max_len", "0.16", "--no_viz"
        ]
        res2 = subprocess.run(["/bin/zsh", "-l", "-c", " ".join(cmd_place)], capture_output=True, text=True)
        if res2.returncode != 0 or not os.path.exists(target_layout_yaml):
            self.report({'ERROR'}, f"Place failed: {res2.stderr}")
            return {'CANCELLED'}

        def load_yaml_safe(filepath):
            cmd = f"python3 -c \"import yaml, json, sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))\" '{filepath}'"
            return json.loads(subprocess.check_output(["/bin/zsh", "-l", "-c", cmd], text=True))

        try:
            layout_data = load_yaml_safe(target_layout_yaml)
            target_points = layout_data.get('points', [])
        except Exception as e:
            self.report({'ERROR'}, f"Failed parsing target layout: {e}")
            return {'CANCELLED'}

        p1_data = []
        for obj, d_id in lbs:
            d_type = 'H'
            if 'TYPE_V' in obj.drone_props.drone_type: d_type = 'V'
            elif 'SEMI_V' in obj.drone_props.drone_type: d_type = 'V'
            
            p1_data.append({
                'id': d_id, 'name': obj.name,
                'x': obj.location.x, 'y': obj.location.y, 'z': obj.location.z,
                'type': d_type
            })

        def get_valid_types(l1, l2, a1, a2):
            types = []
            a1m, a2m = a1 % 360, a2 % 360
            b2s = 360 if a2m == 0 else a2m
            ok_h = ((l1 == 0) or (0 <= a1m <= 180)) and ((l2 == 0) or (180 <= b2s <= 360))
            if ok_h: types.append('H')
            
            b2v = a2m if a2m > 270 else a2m + 360
            ok_v = ((l1 == 0) or (90 <= a1m <= 270)) and ((l2 == 0) or (270 <= b2v <= 450))
            if ok_v: types.append('V')

            # Swapped
            b1s = 360 if a1m == 0 else a1m
            ok_hs = ((l2 == 0) or (0 <= a2m <= 180)) and ((l1 == 0) or (180 <= b1s <= 360))
            if ok_hs and 'H' not in types: types.append('H')

            b1v = a1m if a1m > 270 else a1m + 360
            ok_vs = ((l2 == 0) or (90 <= a2m <= 270)) and ((l1 == 0) or (270 <= b1v <= 450))
            if ok_vs and 'V' not in types: types.append('V')
            
            if not types: return ['H', 'V'] # fallback
            return types

        p2_data = []
        for pt in target_points:
            p2_data.append({
                'x': pt['x'], 'y': pt['y'], 'z': pt['z'],
                'length_1': pt.get('length_1', 0), 'length_2': pt.get('length_2', 0),
                'angle_1': pt.get('angle_1', 0), 'angle_2': pt.get('angle_2', 0),
                'yaw': pt.get('yaw', 0),
                'max_length_limit': pt.get('max_length_limit', 0.16),
                'types': get_valid_types(pt.get('length_1',0), pt.get('length_2',0), pt.get('angle_1',0), pt.get('angle_2',0))
            })

        with open(p1_json, 'w') as f: json.dump(p1_data, f)
        with open(p2_json, 'w') as f: json.dump(p2_data, f)

        cmd_match = ["python3", f"'{os.path.join(addon_dir, 'morph_match.py')}'", f"'{p1_json}'", f"'{p2_json}'", f"'{match_out_json}'"]
        res_match = subprocess.run(["/bin/zsh", "-l", "-c", " ".join(cmd_match)], capture_output=True, text=True)
        if res_match.returncode != 0:
            self.report({'ERROR'}, f"Match failed: {res_match.stderr}")
            return {'CANCELLED'}

        with open(match_out_json, 'r') as f:
            assignments = json.load(f)

        fps = scene.render.fps
        f_start = 1
        f_h1 = max(1, int(f_start + props.morph_hold_1 * fps))
        f_end = max(1, int(f_h1 + props.morph_duration * fps))
        scene.frame_end = int(f_end + props.morph_hold_2 * fps)

        # Helper for angle mapping considering H/V constraints
        def map_angles(d_type, a1, a2, l1, l2):
            a1m, a2m = a1 % 360, a2 % 360
            if d_type == 'H':
                b2s = 360 if a2m == 0 else a2m
                ok = ((l1 == 0) or (0 <= a1m <= 180)) and ((l2 == 0) or (180 <= b2s <= 360))
                if ok: return (a1m, b2s, l1, l2)
                b1s = 360 if a1m == 0 else a1m
                ok_s = ((l2 == 0) or (0 <= a2m <= 180)) and ((l1 == 0) or (180 <= b1s <= 360))
                if ok_s: return (a2m, b1s, l2, l1)
            elif d_type == 'V':
                b2v = a2m if a2m > 270 else a2m + 360
                ok = ((l1 == 0) or (90 <= a1m <= 270)) and ((l2 == 0) or (270 <= b2v <= 450))
                if ok: return (a1m, b2v, l1, l2)
                b1v = a1m if a1m > 270 else a1m + 360
                ok_s = ((l2 == 0) or (90 <= a2m <= 270)) and ((l1 == 0) or (270 <= b1v <= 450))
                if ok_s: return (a2m, b1v, l2, l1)
            return (a1, a2, l1, l2) # fallback

        # Apply Keyframes
        matched_ids = set()
        for asn in assignments:
            d_id = asn['drone_id']
            matched_ids.add(d_id)
            obj = next((o for o, idx in lbs if idx == d_id), None)
            if not obj: continue
            
            d_type = next((d['type'] for d in p1_data if d['id'] == d_id), 'H')
            t = asn['target']
            
            tz = t['z']
            t_scale = 1.0
                
            ta1, ta2, tl1, tl2 = map_angles(d_type, t['angle_1'], t['angle_2'], t['length_1'], t['length_2'])
            
            # Start keys
            obj.keyframe_insert(data_path="location", frame=f_start)
            obj.keyframe_insert(data_path="location", frame=f_h1)
            obj.keyframe_insert(data_path='rotation_euler', frame=f_start)
            obj.keyframe_insert(data_path='rotation_euler', frame=f_h1)
            obj.keyframe_insert(data_path='["servo_1"]', frame=f_start)
            obj.keyframe_insert(data_path='["servo_1"]', frame=f_h1)
            if "servo_2" in obj:
                obj.keyframe_insert(data_path='["servo_2"]', frame=f_start)
                obj.keyframe_insert(data_path='["servo_2"]', frame=f_h1)
                
            for ptr in obj.drone_props.led_pointers:
                ptr.keyframe_insert(data_path="value", frame=f_start)
                ptr.keyframe_insert(data_path="value", frame=f_h1)

            # End keys
            obj.location = (t['x'], t['y'], tz)
            obj.rotation_euler.z = math.radians(t['yaw'])
            obj["servo_1"] = ta1
            if "servo_2" in obj: obj["servo_2"] = ta2
            
            ml = t['max_length_limit']
            ac1 = round(25 * (tl1 / ml)) if ml > 0 else 0
            ac2 = round(25 * (tl2 / ml)) if ml > 0 else 0
            
            # Scale active portions based on stagger's scale_factor if any
            if len(obj.drone_props.led_pointers) >= 2:
                p1 = obj.drone_props.led_pointers[0]
                p2 = obj.drone_props.led_pointers[1]
                
                v1 = 25.0 - ac1
                v2 = 25.0 + ac2
                
                v1_scaled = 25.0 + (v1 - 25.0) * t_scale
                v2_scaled = 25.0 + (v2 - 25.0) * t_scale
                
                p1.value = v1_scaled
                p2.value = v2_scaled

            obj.keyframe_insert(data_path="location", frame=f_end)
            obj.keyframe_insert(data_path='rotation_euler', frame=f_end)
            obj.keyframe_insert(data_path='["servo_1"]', frame=f_end)
            if "servo_2" in obj: obj.keyframe_insert(data_path='["servo_2"]', frame=f_end)
            for ptr in obj.drone_props.led_pointers:
                ptr.keyframe_insert(data_path="value", frame=f_end)

            # Change LED colors across morph (simple: just set them to import_color if needed, or keep existing rules. 
            # Often handled globally by the active led mode which checks t)
            # Drones map directly, so base expression should remain. We don't overwrite the expression here unless needed.

        # Retract non-matched drones
        for obj, d_id in lbs:
            if d_id not in matched_ids:
                for ptr in obj.drone_props.led_pointers:
                    ptr.keyframe_insert(data_path="value", frame=f_start)
                    ptr.keyframe_insert(data_path="value", frame=f_h1)
                    
                    # Retract to 25.0 so they go dark
                    ptr.value = 25.0
                    ptr.keyframe_insert(data_path="value", frame=f_end)
                    
        self.report({'INFO'}, f"Morph Generated using {len(matched_ids)} drones.")
        
        # Make keyframes linear
        for obj, _ in lbs:
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'

        context.view_layer.update()
        scene.frame_set(f_start)

        # Cleanup
        for p in [p1_json, p2_json, match_out_json]:
            try:
                if os.path.exists(p): os.remove(p)
            except: pass

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
        props = scene.drone_props
        obj = context.active_object

        # --- Drone Creation ---
        layout.label(text="Add LightBender:")
        row = layout.row()
        row.prop(props, "drone_type", text="")

        # Display Radius property strictly before creation if Ring is selected
        if props.drone_type == 'TYPE_RING':
            layout.prop(props, "ring_radius")

        row.operator("drone.add_drone", text="Create").drone_type = props.drone_type

        layout.separator()

        # --- Selected Drone Controls ---
        if obj and "servo_1" in obj:
            obj_props = obj.drone_props
            is_semi = 'SEMI' in obj_props.drone_type
            is_ring = 'RING' in obj_props.drone_type

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
            layout.prop(obj_props, "led_mode", text="Mode")

            if obj_props.led_mode == 'EXPRESSION':
                layout.prop(obj_props, "led_formula", text="")
            else:
                # Pointer Mode UI
                box = layout.box()
                box.label(text="Base Color (Start):")
                box.prop(obj_props, "led_base_color", text="")
                box.label(text="Pointers (Split Points):")
                row = box.row()
                row.template_list("UL_DronePointerList", "", obj_props, "led_pointers", obj_props,
                                  "active_pointer_index")
                col = row.column(align=True)
                col.operator("drone.add_pointer", icon='ADD', text="")
                col.operator("drone.remove_pointer", icon='REMOVE', text="")
                box.label(text="Pos | Right-Side Color", icon='INFO')

            layout.separator()
            layout.operator("drone.force_update_leds", text="Test/Update LEDs", icon='LIGHT')
        else:
            layout.label(text="Select a drone to animate", icon='INFO')


class VIEW3D_PT_lb_generative_layouts(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "Generative Layouts"
    bl_parent_id = "VIEW3D_PT_drone_swarm"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.drone_props

        # --- SVG to Layout ---
        layout.label(text="SVG Transform and Place:", icon='GRAPH')
        box = layout.box()

        box.prop(props, "import_svg_filepath")
        box.prop(props, "import_manifest_filepath")

        row = box.row()
        row.prop(props, "import_max_width")
        row.prop(props, "import_max_length")

        row = box.row()
        row.prop(props, "import_cx")
        row.prop(props, "import_cy")

        row = box.row()
        row.prop(props, "import_policy")
        row.prop(props, "import_color")

        box.operator("drone.transform_and_place", text="Transform and Place", icon='OUTLINER_OB_LIGHT')


class VIEW3D_PT_lb_automated_animations(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "Automated Animations"
    bl_parent_id = "VIEW3D_PT_drone_swarm"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.drone_props

        # --- Morph Animation ---
        layout.label(text="Morph:", icon='MOD_ARMATURE')
        box = layout.box()

        box.prop(props, "morph_svg_filepath")
        
        row = box.row(align=True)
        row.prop(props, "morph_hold_1", text="Hold 1")
        row.prop(props, "morph_duration", text="Morph")
        row.prop(props, "morph_hold_2", text="Hold 2")
        
        box.operator("drone.generate_morph", text="Generate Morph", icon='ANIM')

        layout.separator()

        # --- Fold Animation ---
        layout.label(text="Fold:", icon='CON_ROTLIKE')
        box = layout.box()
        box.prop(props, "fold_angle")
        row = box.row(align=True)
        row.prop(props, "fold_hold_time", text="Hold")
        row.prop(props, "fold_transition_time", text="Transition")
        row = box.row(align=True)
        row.prop(props, "fold_align_enabled", text="Align")
        if props.fold_align_enabled:
            row.prop(props, "fold_align_axis", expand=True)
        row = box.row(align=True)
        row.prop(props, "fold_distribute_enabled", text="Distribute")
        if props.fold_distribute_enabled:
            row.prop(props, "fold_distribute_axis", expand=True)
        box.prop(props, "fold_equalize_size")
        row = box.row(align=True)
        row.prop(props, "fold_overlap_resolve", text="Resolve Overlaps")
        if props.fold_overlap_resolve:
            row.prop(props, "fold_overlap_axis", expand=True)
        if props.fold_overlap_resolve:
            box.prop(props, "fold_overlap_threshold", text="Min Spacing")
        row = box.row(align=True)
        row.operator("drone.generate_fold", text="Generate Fold", icon='KEYINGSET')
        row.operator("drone.clear_fold", text="", icon='TRASH')

        layout.separator()

        # --- Draw & Erase Sequencer ---
        layout.label(text="Draw/Erase:", icon='GREASEPENCIL')
        box = layout.box()

        row = box.row()
        row.template_list("UL_DrawEraseGroups", "", props, "de_groups", props, "de_active_group_index", rows=3)
        col = row.column(align=True)
        col.operator("drone.add_de_group", icon='ADD', text="")
        col.operator("drone.remove_de_group", icon='REMOVE', text="")

        if props.de_groups and props.de_active_group_index < len(props.de_groups):
            act_grp = props.de_groups[props.de_active_group_index]
            box.label(text=f"LightBenders in {act_grp.name}:")

            row = box.row()
            row.template_list("UL_DrawEraseDrones", "", act_grp, "drones", act_grp, "active_drone_index", rows=4)
            col = row.column(align=True)
            col.operator("drone.add_drones_to_group", icon='ADD', text="")
            col.operator("drone.remove_drone_from_group", icon='REMOVE', text="")
            col.separator()
            col.operator("drone.move_drone_in_group", icon='TRIA_UP', text="").direction = 'UP'
            col.operator("drone.move_drone_in_group", icon='TRIA_DOWN', text="").direction = 'DOWN'

        box.separator()
        box.label(text="Settings:")
        box.prop(props, "de_draw_speed")
        box.prop(props, "de_erase_speed")
        box.prop(props, "de_hold_time")
        box.prop(props, "de_overlap")
        box.prop(props, "de_loop")

        row = box.row(align=True)
        row.prop(props, "de_draw_color", text="Draw RGB")
        row.prop(props, "de_bg_color", text="Erase RGB")

        row = box.row(align=True)
        row.operator("drone.generate_draw_erase", text="Generate Draw/Erase", icon='KEYINGSET')
        row.operator("drone.clear_draw_erase", text="", icon='TRASH')

        layout.separator()

        # --- Depth Fly-In Feature ---
        layout.label(text="Fly-In/Fly-Out:", icon='OUTLINER_OB_CAMERA')
        box = layout.box()
        
        box.prop(props, "flyin_ref_distance")

        row = box.row(align=True)
        row.prop(props, "flyin_x_min", text="Min X")
        row.prop(props, "flyin_x_max", text="Max X")
        row = box.row(align=True)
        row.prop(props, "flyin_y_min", text="Min Y")
        row.prop(props, "flyin_y_max", text="Max Y")
        row = box.row(align=True)
        row.prop(props, "flyin_z_min", text="Min Z")
        row.prop(props, "flyin_z_max", text="Max Z")

        box.prop(props, "flyin_outward_duration")
        box.prop(props, "flyin_hold_duration")
        box.prop(props, "flyin_inward_duration")

        row = box.row(align=True)
        row.operator("drone.generate_flyin", text="Generate Fly-In/Fly-Out", icon='KEYINGSET')
        row.operator("drone.clear_flyin", text="", icon='TRASH')


class VIEW3D_PT_lb_global_leds(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "Global LED Effects"
    bl_parent_id = "VIEW3D_PT_drone_swarm"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.drone_props

        # --- Global LED Expressions ---
        layout.label(text="Global LED Expressions:", icon='SHADING_RENDERED')
        box = layout.box()

        box.prop(props, "global_expr_preset")

        row = box.row()
        row.template_list("UL_GlobalColorList", "", props, "global_expr_colors", props, "global_expr_active_color_index", rows=3)
        col = row.column(align=True)
        col.operator("drone.add_global_color", icon='ADD', text="")
        col.operator("drone.remove_global_color", icon='REMOVE', text="")

        box.operator("drone.preset_colors", text="Load Preset Colors", icon='COLOR')
        box.prop(props, "global_expr_speed")
        if props.global_expr_preset == 'SPARKLE':
            box.prop(props, "global_expr_sparkle_threshold")

        row = box.row(align=True)
        row.operator("drone.apply_global_expression", text="Apply to Selected", icon='VIEW_PAN').apply_to_all = False
        row.operator("drone.apply_global_expression", text="Apply to All", icon='SCENE_DATA').apply_to_all = True


class VIEW3D_PT_lb_simulators(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "Physics & Simulation"
    bl_parent_id = "VIEW3D_PT_drone_swarm"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.drone_props

        # --- Error & Drift Simulator ---
        layout.label(text="Error & Drift Simulators:", icon='MOD_NOISE')
        box = layout.box()

        box.label(text="Static Error (YZ Plane):")
        box.prop(props, "error_distance")
        row = box.row(align=True)
        row.operator("drone.apply_error", text="Apply Static Error")
        row.operator("drone.reset_error", text="Reset Static")

        box.separator()

        box.label(text="Dynamic Drift (XYZ):")
        box.prop(props, "drift_xy", text="Max Drift XY")
        box.prop(props, "drift_z", text="Max Drift Z")
        box.prop(props, "drift_speed", text="Wave Scale (Slower=Higher)")
        row = box.row(align=True)
        row.operator("drone.apply_drift", text="Apply Drift")
        row.operator("drone.reset_drift", text="Reset Drift")

        layout.separator()

        # --- Deconfliction Simulator ---
        layout.label(text="Stagger:", icon='PARTICLES')
        box = layout.box()

        box.prop(props, "deconflict_overlap")
        box.prop(props, "deconflict_downwash")
        box.prop(props, "deconflict_selection")
        box.prop(props, "deconflict_resolution")
        box.prop(props, "deconflict_trajectory")
        box.prop(props, "deconflict_direction")

        row = box.row(align=True)
        row.operator("drone.deconflict_stagger", text="Stagger")
        row.operator("drone.deconflict_reset", text="Reset Stagger")


class VIEW3D_PT_lb_export(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LightBender"
    bl_label = "Mission Export"
    bl_parent_id = "VIEW3D_PT_drone_swarm"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.drone_props

        # --- Export Config ---
        layout.label(text="Export Config:")
        layout.prop(props, "export_mission_name")
        layout.prop(props, "export_dark_room")

        # --- Mode Radio Buttons ---
        layout.label(text="Mode")
        row = layout.row(align=True)
        row.prop_enum(props, "export_mode", 'ILLUMINATION')
        row.prop_enum(props, "export_mode", 'INTERACTION')

        if props.export_mode == 'ILLUMINATION':
            # --- Illumination Mode ---
            layout.prop(props, "export_at_keyframes")
            if not props.export_at_keyframes:
                layout.prop(props, "export_rate")

            layout.operator("drone.export_yaml", text="Export YAML", icon='EXPORT')

            row = layout.row(align=True)
            # Illuminate button
            illuminate_row = row.row(align=True)
            illuminate_row.enabled = not props.illuminate_running
            illuminate_row.operator("drone.export_and_illuminate", text="Illuminate", icon='PLAY')
            # Stop Illumination button
            stop_row = row.row(align=True)
            stop_row.enabled = props.illuminate_running
            stop_row.operator("drone.stop_illuminate", text="Stop Illumination", icon='SNAP_FACE')

        else:
            # --- Interaction Mode (inline, always visible) ---
            layout.separator()
            layout.label(text="Interaction Config", icon='SETTINGS')

            # S_D and Grace Time – disabled when illuminate is running
            config_col = layout.column()
            config_col.enabled = not props.illuminate_running
            config_col.prop(props, "interaction_sd")
            config_col.prop(props, "interaction_grace_time")
            layout.prop(props, "interaction_duration")

            layout.separator()

            # Export YAML
            layout.operator("drone.export_interaction_yaml", text="Export YAML", icon='EXPORT')

            # Illuminate + Stop Illumination row
            row = layout.row(align=True)
            illuminate_row = row.row(align=True)
            illuminate_row.enabled = not props.illuminate_running
            illuminate_row.operator("drone.export_and_interact", text="Illuminate", icon='PLAY')
            stop_row = row.row(align=True)
            stop_row.enabled = props.illuminate_running and not props.edit_active
            stop_row.operator("drone.stop_illuminate_interaction", text="Stop Illumination", icon='SNAP_FACE')

            layout.separator()

            # Edit + Stop Edit row
            row = layout.row(align=True)
            edit_row = row.row(align=True)
            # Edit available only when illuminate is running AND not already editing
            # AND we have TCP connections (interaction_connected_count > 0)
            edit_row.enabled = props.illuminate_running and not props.edit_active and props.interaction_connected_count > 0
            edit_row.operator("drone.start_edit", text="Edit", icon='GREASEPENCIL')
            stop_edit_row = row.row(align=True)
            stop_edit_row.enabled = props.edit_active
            stop_edit_row.operator("drone.stop_edit", text="Finish Edit", icon='CHECKMARK')

            # Status info
            if props.illuminate_running and not props.interaction_editor_active and not props.edit_active:
                if props.interaction_connected_count > 0:
                    layout.label(text=f"{props.interaction_connected_count} connection(s)", icon='LINKED')
                else:
                    layout.label(text="Waiting for LB connections...", icon='TIME')


# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    LEDPointer,
    ColorItem,
    DrawEraseGroupItem,
    DrawEraseGroup,
    DroneProperties,
    OBJECT_OT_add_drone,
    DRONE_OT_add_pointer,
    DRONE_OT_remove_pointer,
    UL_DronePointerList,
    UL_GlobalColorList,
    UL_DrawEraseGroups,
    UL_DrawEraseDrones,
    DRONE_OT_add_de_group,
    DRONE_OT_remove_de_group,
    DRONE_OT_add_drones_to_group,
    DRONE_OT_remove_drone_from_group,
    DRONE_OT_move_drone_in_group,
    DRONE_OT_reverse_de_direction,
    DRONE_OT_clear_draw_erase,
    DRONE_OT_generate_draw_erase,
    DRONE_OT_generate_flyin,
    DRONE_OT_clear_flyin,
    DRONE_OT_generate_fold,
    DRONE_OT_clear_fold,
    DRONE_OT_apply_error,
    DRONE_OT_reset_error,
    DRONE_OT_apply_drift,
    DRONE_OT_reset_drift,
    DRONE_OT_deconflict_stagger,
    DRONE_OT_deconflict_reset,
    DRONE_OT_transform_and_place,
    DRONE_OT_generate_morph,
    DRONE_OT_add_global_color,
    DRONE_OT_remove_global_color,
    DRONE_OT_preset_colors,
    DRONE_OT_apply_global_expression,
    EXPORT_OT_drone_yaml,
    EXPORT_OT_export_and_illuminate,
    DRONE_OT_stop_illuminate,
    EXPORT_OT_interaction_yaml,
    EXPORT_OT_export_and_interact,
    DRONE_OT_start_edit,
    DRONE_OT_stop_edit,
    DRONE_OT_stop_illuminate_interaction,
    VIEW3D_PT_drone_swarm,
    VIEW3D_PT_lb_generative_layouts,
    VIEW3D_PT_lb_automated_animations,
    VIEW3D_PT_lb_global_leds,
    VIEW3D_PT_lb_simulators,
    VIEW3D_PT_lb_export,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.drone_props = PointerProperty(type=DroneProperties)
    bpy.types.Object.drone_props = PointerProperty(type=DroneProperties)

    if update_leds_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(update_leds_handler)


def unregister():
    close_static_interaction_session()

    if update_leds_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(update_leds_handler)

    del bpy.types.Scene.drone_props
    del bpy.types.Object.drone_props

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
