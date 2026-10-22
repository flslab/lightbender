"""Multiple recording hosts using the existing camera-node run-ID protocol."""

import math
import re
import shlex
import time
import uuid
from pathlib import Path

import zmq
from fabric import Connection

from .pose_estimation import CAMERA_POSE_FAILURE_STATUSES

FAILURE_STATUSES = {'FAILED', 'CAMERA_RECORDING_FAILED', *CAMERA_POSE_FAILURE_STATUSES}


def resolve_camera_nodes(manifest):
    """Resolve enabled cameras; older camera_node manifests remain supported."""
    if 'camera_nodes' in manifest and 'camera_node' in manifest:
        raise ValueError('use camera_nodes or camera_node, not both')
    configs = manifest.get('camera_nodes')
    if 'camera_nodes' not in manifest:
        legacy = manifest.get('camera_node')
        configs = [legacy] if legacy is not None else []
    if not isinstance(configs, list):
        raise ValueError('camera_nodes must be a list')
    enabled = []
    ids = set()
    hosts = set()
    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            raise ValueError('each camera node must be a mapping')
        if not isinstance(config.get('enabled', True), bool):
            raise ValueError('camera enabled must be boolean')
        camera_id = config.setdefault('id', 'CAM' if index == 0 else f'CAM_{index + 1}')
        if not isinstance(camera_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', camera_id):
            raise ValueError('camera id must contain only letters, digits, underscores or hyphens')
        if camera_id in ids:
            raise ValueError(f'duplicate camera id: {camera_id}')
        ids.add(camera_id)
        if not config.get('enabled', True):
            continue
        for field in ('ip', 'user'):
            if not isinstance(config.get(field), str) or not config[field].strip():
                raise ValueError(f'camera {camera_id}.{field} must be a nonempty string')
        if config['ip'] in hosts:
            raise ValueError(f'duplicate enabled camera host: {config["ip"]}')
        hosts.add(config['ip'])
        for field, default in (('ready_timeout_s', 15), ('recording_start_timeout_s', 8),
                               ('stop_timeout_s', 15)):
            _timeout(config, field, default)
        pose = config.get('pose_estimation', {})
        if not isinstance(pose, dict):
            raise ValueError(f'camera {camera_id}.pose_estimation must be a mapping')
        if not isinstance(pose.get('enabled', False), bool):
            raise ValueError(f'camera {camera_id}.pose_estimation.enabled must be boolean')
        enabled.append(config)
    pose_cameras = [c for c in enabled if c.get('pose_estimation', {}).get('enabled', False)]
    if len(pose_cameras) > 1:
        raise ValueError('only one enabled camera may provide the swarm pose transform')
    return enabled


def _timeout(config, field, default):
    value = float(config.get(field, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'camera {config.get("id", "CAM")}.{field} must be positive')
    return value


class CameraNodesMixin:
    def _configure_camera_nodes(self):
        self.camera_nodes = resolve_camera_nodes(self.manifest)
        # Pose correction still comes from one authored viewpoint.
        self.camera_cfg = next((c for c in self.camera_nodes
                                if c.get('pose_estimation', {}).get('enabled', False)),
                               self.camera_nodes[0] if self.camera_nodes else None)

    def _active_cameras(self):
        # Also supports callers constructing an orchestrator with legacy state.
        configs = getattr(self, 'camera_nodes', None)
        if configs is None:
            config = getattr(self, 'camera_cfg', None)
            configs = [config] if config else []
        return [c for c in configs if c.get('enabled', True)]

    @staticmethod
    def _camera_id(config):
        return config.get('id', 'CAM')

    def _prepare_camera_run(self):
        super()._prepare_camera_run()
        self.camera_ready_ids = set()
        self.requested_camera_ids = set()
        for config in self._active_cameras():
            if config is not self.camera_cfg:
                config['run_id'] = uuid.uuid4().hex
            if getattr(self.args, 'record', False):
                config.get('pose_estimation', {})['enabled'] = False

    def _boot_cameras(self):
        if getattr(self.args, 'skip_record', False):
            return
        # Complete pose preflight before other cameras can send READY.
        ordered = sorted(self._active_cameras(), key=lambda c: c is not self.camera_cfg)
        for config in ordered:
            camera_id = self._camera_id(config)
            if not self._boot_remote_node(config, self._get_camera_cmd(config), 'Camera'):
                raise RuntimeError(f'failed to launch camera node {camera_id}')
            if config is self.camera_cfg and self._camera_pose_enabled():
                self._wait_for_camera_pose()
                self._confirm_camera_pose()
        if not ordered:
            self.logger.info('No enabled camera nodes. Skipping recording.')

    def _camera_for_message(self, msg):
        configs = self._active_cameras()
        for config in configs:
            expected = config.get('run_id', getattr(self, 'camera_run_id', None))
            if (msg.get('id') in {'CAM', self._camera_id(config)}
                    and (expected is None or msg.get('run_id') == expected)):
                return config
        return None

    def _pending_camera_ids(self):
        if getattr(self.args, 'skip_record', False):
            return set()
        ready = set(getattr(self, 'camera_ready_ids', set()))
        if getattr(self, 'camera_ready', False) and self.camera_cfg:
            ready.add(self._camera_id(self.camera_cfg))
        return {self._camera_id(c) for c in self._active_cameras()} - ready

    def _camera_ready_deadlines(self):
        pending = self._pending_camera_ids()
        return {self._camera_id(c): time.monotonic() + _timeout(c, 'ready_timeout_s', 15)
                for c in self._active_cameras() if self._camera_id(c) in pending}

    def _check_camera_deadlines(self, deadlines, action, pending=None):
        if pending is None:
            pending = self._pending_camera_ids()
        for camera_id in pending:
            if time.monotonic() >= deadlines[camera_id]:
                raise RuntimeError(f'camera {camera_id} did not {action} before its timeout')

    def _handle_camera_runtime_event(self, msg):
        config = self._camera_for_message(msg)
        if config is None:
            return False
        camera_id = self._camera_id(config)
        status = msg.get('status')
        if status in FAILURE_STATUSES:
            raise RuntimeError(f'camera {camera_id} runtime failure: {msg.get("error", status)}')
        if status == 'READY':
            if not hasattr(self, 'camera_ready_ids'):
                self.camera_ready_ids = set()
            self.camera_ready_ids.add(camera_id)
            if config is self.camera_cfg:
                self.camera_ready = True
            self.logger.info('Camera %s ready.', camera_id)
        return True

    def _start_camera_recording(self):
        """Require a fresh recording acknowledgement from every enabled host."""
        if getattr(self.args, 'skip_record', False):
            return
        configs = self._active_cameras()
        if not configs:
            return
        request_id = uuid.uuid4().hex
        pending = {self._camera_id(c) for c in configs}
        deadlines = {}
        self.requested_camera_ids = set(pending)
        for config in configs:
            camera_id = self._camera_id(config)
            deadlines[camera_id] = time.monotonic() + _timeout(config, 'recording_start_timeout_s', 8)
            self.pub_socket.send_json({
                'cmd': 'START_CAMERA',
                'run_id': config.get('run_id', getattr(self, 'camera_run_id', None)),
                'request_id': request_id,
            })
        while self.running.is_set() and pending:
            self._check_camera_deadlines(deadlines, 'confirm recording', pending)
            try:
                msg = self.pull_socket.recv_json()
            except zmq.Again:
                continue
            if self._handle_marker_grid_event(msg):
                continue
            config = self._camera_for_message(msg)
            if config is None:
                continue
            self._handle_camera_runtime_event(msg)
            if msg.get('status') == 'RECORDING' and msg.get('request_id') == request_id:
                camera_id = self._camera_id(config)
                pending.discard(camera_id)
                self.logger.info('Camera %s recording confirmed before launch.', camera_id)
        if pending:
            raise RuntimeError('camera recording start was interrupted')

    def _stop_cameras_and_download(self):
        configs = self._active_cameras()
        if not configs or not self.pub_socket or getattr(self.args, 'skip_record', False):
            return
        self.pub_socket.send_json({'cmd': 'STOP_CAMERA'})
        wanted = getattr(self, 'requested_camera_ids', set())
        for config in configs:
            camera_id = self._camera_id(config)
            if not self.took_off and camera_id not in wanted:
                continue
            try:
                # The recorder exits after SIGINT closes the video. Wait for
                # that exit instead of copying a file still being finalized.
                timeout = _timeout(config, 'stop_timeout_s', 15)
                check = 'while pgrep -f "^python3 .*camera_node[.]py( |$)" >/dev/null; do sleep 0.25; done'
                with Connection(host=config['ip'], user=config['user'], connect_timeout=10) as conn:
                    conn.run(f'timeout {timeout:g}s bash -c {shlex.quote(check)}',
                             timeout=timeout + 2, hide=True)
                work = config.get('work_dir', self.common_cfg['work_dir'])
                local_dir = Path('./logs') / self.tag
                local_dir.mkdir(parents=True, exist_ok=True)
                local = local_dir / f'{camera_id}_{self.tag}.mp4'
                self._download_file(config, f'{work}/mission_footage.mp4', str(local),
                                    f'Mission Video ({camera_id})', required=True)
            except Exception as error:
                self.logger.error('Could not finalize/download camera %s: %s', camera_id, error)
