"""Installed-ZIP integration test; runs only in a disposable Blender process.

Run Blender with --background --factory-startup --python-exit-code 1 --python
tests/blender_addon_standalone.py. Builds and extracts the release into a path
with spaces and an apostrophe, enables it without saving user preferences,
sets up its real private environment, and exercises the bundled CLI tools.
The reusable test environment and setup log live under /tmp, never in the
user's Blender configuration. First setup can download dependencies.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

import addon_utils
import bpy
from mathutils import Vector


SOURCE = Path(__file__).resolve().parents[1] / "authoring"
STORAGE = Path("/tmp/lightbender-addon-verification/managed")


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def scene_drones():
    return [obj for obj in bpy.context.scene.objects if "servo_1" in obj]


def make_camera():
    scene = bpy.context.scene
    data = bpy.data.cameras.new("StandaloneView")
    camera = bpy.data.objects.new("StandaloneView", data)
    scene.collection.objects.link(camera)
    camera.location = (2.25, 0.0, 0.94)
    camera.rotation_euler = (Vector((0.0, 0.0, 1.0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def assert_preferences_draw(addon, preferences):
    class Layout:
        def __init__(self):
            self.properties = []

        def prop(self, data, name, **kwargs):
            getattr(data, name)
            self.properties.append(name)

        def label(self, **kwargs):
            pass

        def separator(self):
            pass

        def operator(self, identifier, **kwargs):
            assert identifier == "drone.setup_python_tools"
            return SimpleNamespace()

    class DrawPreferences:
        def __init__(self):
            self.layout = Layout()

        def __getattr__(self, name):
            return getattr(preferences, name)

    fake = DrawPreferences()
    addon.LightBenderAddonPreferences.draw(fake, bpy.context)
    assert set(fake.layout.properties) == {
        "repo_dir", "python_executable", "install_gurobi", "orchestrator_python",
    }


def prepare_runtime(addon):
    runtime = addon.addon_runtime.RuntimeManager(STORAGE)
    addon._AUTHORING_RUNTIME = runtime
    ready = runtime.start(allow_install=True)
    deadline = time.monotonic() + 180.0
    while not ready and runtime.process is not None and time.monotonic() < deadline:
        # No Blender event loop, user preferences, or Blender API access during
        # setup. Poll the asynchronous installer as an ordinary child process.
        time.sleep(0.2)
        ready = runtime.poll()
    if not ready:
        tail = runtime.log_path.read_text(encoding="utf-8")[-6000:] if runtime.log_path.exists() else ""
        raise AssertionError(runtime.message + "\n" + tail)
    probe = runtime.run(["-c", "import json,sys,scipy,yaml; from svg.path import parse_path; "
                         "print(json.dumps({'prefix':sys.prefix,'scipy':scipy.__version__}))"])
    assert probe.returncode == 0, probe.stderr
    assert Path(json.loads(probe.stdout)["prefix"]).resolve() == runtime.environment.resolve()
    assert str(runtime.environment) not in "\n".join(sys.path)
    history = []
    actual_run = runtime.run

    def checked_run(arguments, **kwargs):
        result = actual_run(arguments, **kwargs)
        assert result.returncode == 0, (arguments, result.stdout, result.stderr)
        history.append(Path(str(arguments[0])).name)
        return result

    runtime.run = checked_run
    print("PASS: installed archive bootstrapped its isolated Python dependencies", flush=True)
    return history


def test_transform_place_and_morph(addon, directory, policy='VFG'):
    clear_scene()
    make_camera()
    scene = bpy.context.scene
    props = scene.drone_props
    svg = directory / "user's small shape.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10">'
                   '<path d="M 0 10 L 10 0 L 20 10"/></svg>', encoding="utf-8")
    props.import_svg_filepath = str(svg)
    props.import_manifest_filepath = ""
    props.import_policy = policy
    props.import_set_cover_solver = 'bnb'
    props.import_max_width = 0.2
    props.import_max_length = 0.08
    props.import_cy = 1.0
    assert bpy.ops.drone.transform_and_place() == {'FINISHED'}
    assert props.export_mission_name == svg.stem
    export = SimpleNamespace(filepath="")
    selected = []
    context = SimpleNamespace(scene=scene, window_manager=SimpleNamespace(fileselect_add=selected.append))
    assert addon.EXPORT_OT_drone_yaml.invoke(export, context, None) == {'RUNNING_MODAL'}
    assert export.filepath == svg.stem + ".yaml"
    assert selected == [export]
    props.export_mission_name = "custom mission"
    addon.EXPORT_OT_drone_yaml.invoke(export, context, None)
    assert export.filepath == "custom mission.yaml"

    drones = scene_drones()
    assert drones, "SVG placement must create at least one LightBender"
    assert all(addon.lightbender_name_match(d.name) and "_extra" in d.name for d in drones)
    for index, drone in enumerate(drones, 9100):
        drone.name = "lb" + str(index)
        assert all(math.isfinite(value) for value in drone.location)
    print("PASS: installed transform/place scripts create Blender objects with", policy, flush=True)

    props.morph_svg_filepath = str(svg)
    props.import_cx = 0.03
    props.morph_hold_1 = 0.0
    props.morph_duration = 0.1
    props.morph_hold_2 = 0.0
    assert bpy.ops.drone.generate_morph() == {'FINISHED'}
    assert any(addon.get_location_fcurves(drone) for drone in drones)
    assert any(len(curve.keyframe_points) >= 2 for drone in drones
               for curve in addon.get_location_fcurves(drone).values())
    print("PASS: installed transform/place/morph_match scripts generate morph keyframes with", policy, flush=True)


def test_read_only_set_cover(addon, directory):
    package = directory / "lightbender_swarm"
    working = directory / "read only working directory"
    working.mkdir()
    original = Path.cwd()
    try:
        package.chmod(0o555)
        working.chmod(0o555)
        os.chdir(working)
        test_transform_place_and_morph(addon, directory, policy='SC')
        assert not list(working.iterdir()), "Set Cover must keep generated logs out of the working directory"
    finally:
        os.chdir(original)
        package.chmod(0o755)
        working.chmod(0o755)
    print("PASS: Set Cover placement and morph work from a read-only directory", flush=True)


def test_yaml_import(addon, directory):
    clear_scene()
    mission = directory / "user's imported mission.yaml"
    mission.write_text("""name: standalone_import
drones:
  lb9200:
    target: [0, 0, 1, 0]
    waypoints: [[0, 0, 1, 0, 0], [0, 0.1, 1, 0, 1]]
    delta_t: 0
    iterations: 1
    servos: [[0, 180], [0, 180]]
    led:
      mode: expression
      formula: '[255, 255, 255]'
""", encoding="utf-8")
    bpy.context.scene.drone_props.import_mission_filepath = str(mission)
    assert bpy.ops.drone.import_mission_yaml() == {'FINISHED'}
    drone = bpy.data.objects.get("lb9200")
    assert drone is not None
    assert addon.get_location_fcurves(drone)
    print("PASS: isolated PyYAML import recreates an animated mission", flush=True)


def test_camera_facing_and_stagger(addon):
    clear_scene()
    camera = make_camera()
    assert bpy.ops.drone.add_drone(drone_type='TYPE_H') == {'FINISHED'}
    drone = bpy.context.active_object
    drone.name = "lb9300"
    drone.location = (0.0, 0.0, 1.0)
    drone.rotation_euler.z = math.radians(150.0)
    drone["servo_1"] = 135.0
    drone["servo_2"] = 225.0
    drone.drone_props.light_module_offset = (0.06, 0.01, -0.02)
    drone.drone_props.led_mode = 'EXPRESSION'
    drone.drone_props.led_formula = "[227,253,255]"
    bpy.context.view_layer.update()
    assert bpy.ops.drone.camera_facing() == {'FINISHED'}
    bpy.context.view_layer.update()
    normal = drone.matrix_world.to_3x3() @ Vector((1, 0, 0))
    hinge = drone.matrix_world.translation + drone.matrix_world.to_3x3() @ Vector(drone.drone_props.light_module_offset)
    assert normal.dot(camera.location - hinge) > 0
    assert len(drone.drone_props.led_pointers) == 2
    assert all(math.isfinite(value) for value in drone.location)
    print("PASS: installed camera-facing script corrects a mounted light module", flush=True)

    assert bpy.ops.drone.deconflict_stagger() == {'FINISHED'}
    assert "deconflict_orig_x" in drone
    assert all(math.isfinite(value) for value in drone.location)
    assert bpy.ops.drone.deconflict_reset() == {'FINISHED'}
    assert "deconflict_orig_x" not in drone
    print("PASS: installed stagger script and Blender reset complete", flush=True)


def test_disable_and_reenable(addon):
    callbacks = (addon.static_interaction_timer, addon.swarm_monitor_timer)
    for callback in callbacks:
        bpy.app.timers.register(callback, first_interval=60.0)
    addon.STATIC_INTERACTION_SESSION["timer_registered"] = True
    addon.SWARM_MONITOR_SESSION["timer_registered"] = True
    addon_utils.disable("lightbender_swarm", default_set=True)
    assert not addon.STATIC_INTERACTION_SESSION["timer_registered"]
    assert not addon.SWARM_MONITOR_SESSION["timer_registered"]
    assert all(not bpy.app.timers.is_registered(callback) for callback in callbacks)
    assert not hasattr(bpy.types.Scene, "drone_props")
    assert addon_utils.enable("lightbender_swarm", default_set=True) is addon
    assert hasattr(bpy.types.Scene, "drone_props")
    assert not addon.STATIC_INTERACTION_SESSION["timer_registered"]
    assert not addon.SWARM_MONITOR_SESSION["timer_registered"]
    assert addon.get_addon_preferences().repo_dir == ""
    print("PASS: disable clears callbacks and flags; re-enable restores registration", flush=True)


def main():
    spec = importlib.util.spec_from_file_location("standalone_release_builder", SOURCE / "build_addon.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with tempfile.TemporaryDirectory(prefix="LightBender user's install ") as temporary:
        directory = Path(temporary)
        archive = builder.build_release(SOURCE, directory / "release package.zip")
        with zipfile.ZipFile(archive) as release:
            release.extractall(directory)
        sys.path.insert(0, str(directory))
        addon = addon_utils.enable("lightbender_swarm", default_set=True)
        assert addon is not None
        try:
            # register() schedules first setup through a Blender timer. Inject
            # /tmp storage before returning control to Blender's event loop.
            history = prepare_runtime(addon)
            preferences = bpy.context.preferences.addons["lightbender_swarm"].preferences
            assert addon.get_addon_preferences().as_pointer() == preferences.as_pointer()
            assert preferences.repo_dir == ""
            assert addon.get_repo_dir() == ""
            assert Path(addon.get_authoring_dir()).parent == (directory / "lightbender_swarm").resolve()
            assert_preferences_draw(addon, preferences)
            test_transform_place_and_morph(addon, directory)
            test_read_only_set_cover(addon, directory)
            test_yaml_import(addon, directory)
            test_camera_facing_and_stagger(addon)
            assert {"transform.py", "place.py", "morph_match.py", "camera_facing.py", "deconflict.py"} <= set(history)
            assert preferences.repo_dir == "", "Standalone authoring must never need a checkout"
            test_disable_and_reenable(addon)
            print("PASS: standalone installed ZIP integration on Blender", bpy.app.version_string, flush=True)
        finally:
            addon_utils.disable("lightbender_swarm", default_set=True)
            sys.path.remove(str(directory))


if __name__ == "__main__":
    main()
