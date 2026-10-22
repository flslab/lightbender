"""Load real command builders without importing networking or hardware packages."""

import ast
from functools import wraps
from pathlib import Path
import shlex

import interaction_launch


def command_builder():
    source = Path(__file__).parents[1] / 'orchestrator.py'
    tree = ast.parse(source.read_text())
    orchestrator = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == 'SwarmOrchestrator')
    dependencies = ('_get_viewpoint_args', '_mocap_cli_args',
                    '_get_marker_grid_tracker_arg', '_get_drone_by_id')
    builders = ('_get_drone_cmd', '_get_drone_cmd_interaction',
                '_get_drone_cmd_illumination', '_get_drone_cmd_morphing')
    methods = [node for node in orchestrator.body
               if isinstance(node, ast.FunctionDef)
               and node.name in dependencies + builders]
    resolver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'resolve_mocap_config')
    namespace = {name: getattr(interaction_launch, name)
                 for name in dir(interaction_launch) if not name.startswith('_')}
    namespace['shlex'] = shlex
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        'TIME_SYNC_CMD', 'DEFAULT_MOCAP', 'DRONE_SCRIPT'):
                    namespace[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=[resolver, *methods], type_ignores=[]),
                 str(source), 'exec'), namespace)
    helpers = {name: namespace[name] for name in dependencies}

    def bind_helpers(method):
        @wraps(method)
        def call(owner, *args, **kwargs):
            for name, helper in helpers.items():
                setattr(owner, name, helper.__get__(owner))
            return method(owner, *args, **kwargs)
        return call

    for name in builders:
        namespace[name] = bind_helpers(namespace[name])
    return namespace
