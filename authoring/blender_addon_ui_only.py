bl_info = {
    "name": "LightBender Swarm Animator (UI Only Edit)",
    "author": "Hamed Alimohammadzadeh",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > LightBender",
    "description": "Variant of the LightBender add-on where Edit / Finish Edit only drive UI state",
    "category": "Animation",
}

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).with_name("blender_addon.py")
_SPEC = importlib.util.spec_from_file_location("lightbender_blender_addon_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


def _ui_only_start_edit_execute(self, context):
    scene = context.scene
    props = scene.drone_props

    if not props.interaction_editor_active:
        self.report({'ERROR'}, "Interaction editor is not active.")
        return {'CANCELLED'}

    props.edit_active = True
    props.interaction_recording_active = False

    self.report({'INFO'}, "Edit started in UI-only mode.")
    return {'FINISHED'}


def _ui_only_stop_edit_execute(self, context):
    scene = context.scene
    props = scene.drone_props

    if not props.interaction_editor_active:
        self.report({'ERROR'}, "Interaction editor is not active.")
        return {'CANCELLED'}

    props.edit_active = False
    props.interaction_recording_active = False

    # Keep the shared interaction session inert in the UI-only variant.
    _BASE.STATIC_INTERACTION_SESSION["recording"] = False
    _BASE.STATIC_INTERACTION_SESSION["record_updated"] = set()
    _BASE.STATIC_INTERACTION_SESSION["record_received_any"] = False
    _BASE.STATIC_INTERACTION_SESSION["record_failed_sends"] = []

    self.report({'INFO'}, "Edit finished in UI-only mode.")
    return {'FINISHED'}


def _ui_only_export_panel_draw(self, context):
    layout = self.layout
    scene = context.scene
    props = scene.drone_props

    layout.label(text="Export Config:")
    layout.prop(props, "export_mission_name")
    layout.prop(props, "export_dark_room")

    layout.label(text="Mode")
    row = layout.row(align=True)
    row.prop_enum(props, "export_mode", 'ILLUMINATION')
    row.prop_enum(props, "export_mode", 'INTERACTION')

    if props.export_mode == 'ILLUMINATION':
        layout.prop(props, "export_at_keyframes")
        if not props.export_at_keyframes:
            layout.prop(props, "export_rate")

        layout.operator("drone.export_yaml", text="Export YAML", icon='EXPORT')

        illuminate_row = layout.row()
        illuminate_row.enabled = not props.illuminate_running and props.swarm_logs_fetched
        illuminate_row.scale_y = 1.2
        illuminate_row.operator("drone.export_and_illuminate", text="Illuminate", icon='PLAY')

        all_ready = bool(props.swarm_drones) and all(d.status == "ready" for d in props.swarm_drones)
        row = layout.row(align=True)
        confirm_col = row.row(align=True)
        confirm_col.enabled = all_ready and props.illuminate_running and not props.swarm_launch_confirmed and not props.swarm_stopping
        op_text = "Launch Confirmed" if props.swarm_launch_confirmed else "Confirm Launch"
        op_icon = 'CHECKMARK' if props.swarm_launch_confirmed else 'PLAY'
        confirm_col.operator("drone.confirm_launch", text=op_text, icon=op_icon)
        stop_col = row.row(align=True)
        stop_col.enabled = props.illuminate_running and props.swarm_connected and not props.swarm_stopping
        stop_col.operator("drone.stop_illuminate", text="Stop", icon='SNAP_FACE')

        if props.swarm_stopping:
            layout.label(text="Stopping - waiting for LightBenders to land...", icon='INFO')
        return

    layout.separator()
    layout.label(text="Interaction Config", icon='SETTINGS')

    config_col = layout.column()
    config_col.enabled = not props.illuminate_running
    config_col.prop(props, "interaction_sd")
    config_col.prop(props, "interaction_grace_time")
    layout.prop(props, "interaction_duration")

    layout.separator()
    layout.operator("drone.export_interaction_yaml", text="Export YAML", icon='EXPORT')

    illuminate_row = layout.row()
    illuminate_row.enabled = not props.illuminate_running and props.swarm_logs_fetched
    illuminate_row.scale_y = 1.2
    illuminate_row.operator("drone.export_and_interact", text="Illuminate", icon='PLAY')

    all_ready = bool(props.swarm_drones) and all(d.status == "ready" for d in props.swarm_drones)
    row = layout.row(align=True)
    confirm_col = row.row(align=True)
    confirm_col.enabled = all_ready and props.illuminate_running and not props.swarm_launch_confirmed and not props.swarm_stopping
    confirm_col.operator("drone.confirm_launch", text="Confirm Launch", icon='PLAY')
    stop_col = row.row(align=True)
    stop_col.enabled = props.illuminate_running and props.swarm_connected and not props.swarm_stopping and not props.edit_active
    stop_col.operator("drone.stop_illuminate_interaction", text="Stop", icon='SNAP_FACE')

    if props.swarm_stopping:
        layout.label(text="Stopping - waiting for LightBenders to land...", icon='INFO')

    layout.separator()

    row = layout.row(align=True)
    edit_row = row.row(align=True)
    edit_row.enabled = props.illuminate_running and not props.edit_active
    edit_row.operator("drone.start_edit", text="Edit", icon='GREASEPENCIL')

    stop_edit_row = row.row(align=True)
    stop_edit_row.enabled = props.edit_active
    stop_edit_row.operator("drone.stop_edit", text="Finish Edit", icon='CHECKMARK')

    if props.illuminate_running and props.interaction_editor_active:
        if props.edit_active:
            layout.label(text="UI-only edit mode active", icon='INFO')
        else:
            layout.label(text="UI-only interaction controls ready", icon='INFO')


_BASE.DRONE_OT_start_edit.execute = _ui_only_start_edit_execute
_BASE.DRONE_OT_stop_edit.execute = _ui_only_stop_edit_execute
_BASE.VIEW3D_PT_lb_export.draw = _ui_only_export_panel_draw


def register():
    _BASE.register()


def unregister():
    _BASE.unregister()


if __name__ == "__main__":
    register()
