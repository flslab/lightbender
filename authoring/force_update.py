import bpy

def refresh_depsgraph(scene):
    bpy.context.view_layer.update()

bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(refresh_depsgraph)
