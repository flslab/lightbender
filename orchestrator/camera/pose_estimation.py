"""RGB camera pose preflight and persistence for the swarm orchestrator.

The mixin operates on the orchestrator's camera, mission, socket, and logging
state so launch sequencing can remain in ``orchestrator.py``.
"""

import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import zmq
from fabric import Connection

BASE_DIR = Path(__file__).resolve().parents[1]
CAMERA_POSE_CACHE_FILE = BASE_DIR / 'logs' / 'camera_pose_latest.json'
CAMERA_POSE_STATUSES = frozenset({'CAMERA_POSE', 'POSE'})
CAMERA_POSE_FAILURE_STATUSES = frozenset({'CAMERA_POSE_FAILED', 'POSE_FAILED'})
CAMERA_POSE_OVERLAY_FILENAME = 'camera_pose_overlay.jpeg'


def _finite_vector(value, length, field):
    """Return a numeric JSON vector or raise a configuration/protocol error."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    result = []
    for item in value:
        if (isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(item)):
            raise ValueError(f"{field} must contain only finite numbers")
        result.append(float(item))
    return result


def _finite_scalar(value, field):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _wrap_radians(angle):
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class CameraPoseMixin:
    def _configure_camera_pose(self):
        """Validate and complete the optional RGB-camera pose configuration.

        The camera receives this configuration through the in-memory manifest
        uploaded by ``_boot_remote_node``.  Paths are therefore paths on the
        camera computer, not paths on the orchestrator laptop.
        """
        if not self.camera_cfg:
            return

        config = self.camera_cfg.get('pose_estimation')
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError("camera_node.pose_estimation must be a mapping")

        enabled = config.get('enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError("camera_node.pose_estimation.enabled must be boolean")
        if not enabled:
            return
        if (
            getattr(self.args, 'record', False)
            or getattr(self.args, 'skip_record', False)
        ):
            # _prepare_camera_run disables pose estimation in the runtime
            # manifest for --record; --skip-record explicitly omits the camera.
            return

        # Command-line values are convenient for one-off experiments, while
        # the manifest remains the reproducible source of configuration.
        rough_position = getattr(self.args, 'camera_rough_position', None)
        if rough_position is not None:
            config['rough_position_xyz'] = rough_position
        rough_yaw = getattr(self.args, 'camera_rough_yaw', None)
        if rough_yaw is not None:
            config['rough_yaw_rad'] = rough_yaw
        grid_file = getattr(self.args, 'camera_grid_file', None)
        if grid_file is not None:
            config['grid_file'] = grid_file
        calibration_file = getattr(self.args, 'camera_calibration_file', None)
        if calibration_file is not None:
            config['calibration_file'] = calibration_file

        for field in ('grid_file', 'calibration_file'):
            value = config.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"camera_node.pose_estimation.{field} must be a non-empty "
                    "path on the camera computer"
                )
        config['rough_position_xyz'] = _finite_vector(
            config.get('rough_position_xyz'), 3,
            'camera_node.pose_estimation.rough_position_xyz',
        )
        if 'rough_yaw_rad' in config:
            config['rough_yaw_rad'] = _finite_scalar(
                config['rough_yaw_rad'],
                'camera_node.pose_estimation.rough_yaw_rad',
            )

        mission_camera = (self.mission or {}).get('camera')
        desired_position = config.get('desired_position_xyz', mission_camera)
        config['desired_position_xyz'] = _finite_vector(
            desired_position, 3,
            'camera_node.pose_estimation.desired_position_xyz',
        )

        # SFL currently records the authored camera position but not a pitch.
        # The physical recording camera is kept level, so default to a
        # horizontal optical axis aimed toward world XY origin at the authored
        # camera height. Missions that deliberately use pitch can provide
        # camera_look_at, and the manifest can still override look_at_xyz.
        horizontal_look_at = [0, 0, config['desired_position_xyz'][2]]
        look_at = config.get(
            'look_at_xyz',
            (self.mission or {}).get('camera_look_at', horizontal_look_at),
        )
        config['look_at_xyz'] = _finite_vector(
            look_at, 3, 'camera_node.pose_estimation.look_at_xyz'
        )
        desired_yaw_was_explicit = config.get('desired_yaw_rad') is not None
        desired_yaw = config.get('desired_yaw_rad')
        if desired_yaw is None:
            dx = config['look_at_xyz'][0] - config['desired_position_xyz'][0]
            dy = config['look_at_xyz'][1] - config['desired_position_xyz'][1]
            if math.hypot(dx, dy) < 1e-9:
                raise ValueError(
                    "camera pose look_at_xyz must not share XY with the authored "
                    "camera; set desired_yaw_rad explicitly"
                )
            desired_yaw = math.atan2(dy, dx)
        config['desired_yaw_rad'] = _wrap_radians(_finite_scalar(
            desired_yaw, 'camera_node.pose_estimation.desired_yaw_rad'
        ))

        # One measured pose is passed to every SFL handled by this process.
        # Reject a mixed authored-camera batch instead of applying the first
        # file's delta to a different camera pivot later in the sequence.
        for index, mission in enumerate(getattr(self, 'missions', [])):
            authored_camera = _finite_vector(
                mission.get('camera'), 3, f'mission[{index}].camera'
            )
            if any(abs(a - b) > 1e-9 for a, b in zip(
                    authored_camera, config['desired_position_xyz'])):
                raise ValueError(
                    "all SFL files must use the same authored camera position "
                    "when RGB camera pose estimation is enabled"
                )
            if not desired_yaw_was_explicit:
                authored_look_at = _finite_vector(
                    mission.get('camera_look_at', config['look_at_xyz']),
                    3,
                    f'mission[{index}].camera_look_at',
                )
                dx = authored_look_at[0] - authored_camera[0]
                dy = authored_look_at[1] - authored_camera[1]
                if math.hypot(dx, dy) < 1e-9:
                    raise ValueError(
                        f"mission[{index}] camera and camera_look_at share XY"
                    )
                authored_yaw = math.atan2(dy, dx)
                if abs(_wrap_radians(
                        authored_yaw - config['desired_yaw_rad'])) > 1e-9:
                    raise ValueError(
                        "all SFL files must use the same authored camera yaw "
                        "when RGB camera pose estimation is enabled"
                    )

        timeout = config.get('timeout_s', 20.0)
        config['timeout_s'] = _finite_scalar(
            timeout, 'camera_node.pose_estimation.timeout_s'
        )
        if config['timeout_s'] <= 0:
            raise ValueError(
                "camera_node.pose_estimation.timeout_s must be positive"
            )

    def _camera_pose_enabled(self):
        config = (self.camera_cfg or {}).get('pose_estimation', {})
        return bool(
            config.get('enabled', False) and
            not getattr(self.args, 'skip_record', False) and
            not getattr(self.args, 'record', False)
        )

    def _prepare_camera_run(self):
        """Tag this camera process and derive its runtime-only configuration."""
        self.camera_ready = False
        self.camera_run_id = uuid.uuid4().hex
        if not self.camera_cfg:
            return
        self.camera_cfg['run_id'] = self.camera_run_id

        config = self.camera_cfg.get('pose_estimation')
        if (
            isinstance(config, dict)
            and config.get('enabled', False)
            and getattr(self.args, 'record', False)
        ):
            # Camera-only recording has no swarm to transform and deliberately
            # does not boot the marker grid. Do not let an enabled preflight
            # setting inherited from the normal manifest deadlock this mode.
            self.logger.info(
                "Skipping RGB pose estimation in --record camera-only mode"
            )
            config['enabled'] = False

    def _camera_message_matches_run(self, msg):
        expected = getattr(self, 'camera_run_id', None)
        return expected is None or msg.get('run_id') == expected

    def _parse_camera_pose(self, msg):
        """Normalize a version-1 camera pose message from ``camera_node``."""
        if not isinstance(msg, dict):
            raise ValueError("camera pose message must be a JSON object")
        if msg.get('id') != 'CAM':
            raise ValueError("camera pose message id must be CAM")
        if not self._camera_message_matches_run(msg):
            raise ValueError("camera pose message belongs to a different run")
        if msg.get('status') not in CAMERA_POSE_STATUSES:
            raise ValueError("camera pose message has an unsupported status")

        pose = msg.get('pose', {})
        offset = msg.get('offset', {})
        if not isinstance(pose, dict) or not isinstance(offset, dict):
            raise ValueError("camera pose and offset fields must be mappings")

        position_value = pose.get(
            'position_xyz', pose.get(
                'position', msg.get('position_xyz', msg.get('position'))
            )
        )
        yaw_value = pose.get(
            'yaw_rad', pose.get('yaw', msg.get('yaw_rad', msg.get('yaw')))
        )
        position = _finite_vector(
            position_value, 3, 'camera pose position_xyz'
        )
        yaw = _wrap_radians(_finite_scalar(yaw_value, 'camera pose yaw_rad'))

        config = self.camera_cfg['pose_estimation']
        desired_position = config['desired_position_xyz']
        desired_yaw = config['desired_yaw_rad']
        reported_position_offset = offset.get('position_offset_xyz', offset.get(
            'position_xyz', offset.get(
                'position',
                msg.get('position_offset_xyz', msg.get('position_offset')),
            )
        ))
        if reported_position_offset is None:
            position_offset = [
                position[index] - desired_position[index] for index in range(3)
            ]
        else:
            position_offset = _finite_vector(
                reported_position_offset, 3, 'camera pose position_offset_xyz'
            )

        reported_yaw_offset = offset.get('yaw_offset_rad', offset.get(
            'yaw_rad', offset.get(
                'yaw', msg.get('yaw_offset_rad', msg.get('yaw_offset'))
            )
        ))
        if reported_yaw_offset is None:
            yaw_offset = _wrap_radians(yaw - desired_yaw)
        else:
            yaw_offset = _wrap_radians(_finite_scalar(
                reported_yaw_offset, 'camera pose yaw_offset_rad'
            ))

        # Do not silently accept a solver result whose redundant offsets use
        # a different sign or authored camera than the orchestrator.
        expected_position_offset = [
            position[index] - desired_position[index] for index in range(3)
        ]
        if any(abs(a - b) > 1e-4 for a, b in zip(
                position_offset, expected_position_offset)):
            raise ValueError(
                "camera pose position offset is inconsistent with actual minus "
                "authored position"
            )
        expected_yaw_offset = _wrap_radians(yaw - desired_yaw)
        if abs(_wrap_radians(yaw_offset - expected_yaw_offset)) > 1e-4:
            raise ValueError(
                "camera pose yaw offset is inconsistent with actual minus "
                "authored yaw"
            )

        quality = msg.get('quality', {})
        if not isinstance(quality, dict):
            raise ValueError("camera pose quality must be a mapping")
        solver_config = config.get('solver', {})
        minimum_markers = int(solver_config.get('min_matched_markers', 12))
        matched_markers = quality.get('matched_markers')
        if (
            isinstance(matched_markers, bool)
            or not isinstance(matched_markers, int)
            or matched_markers < minimum_markers
        ):
            raise ValueError(
                "camera pose quality has too few matched HyperGrid markers"
            )
        expected_consensus = int(config.get('consensus_frames', 3))
        if quality.get('consensus_frames') != expected_consensus:
            raise ValueError(
                "camera pose did not report the configured temporal consensus"
            )
        reprojection_rms = quality.get('reprojection_rms_px')
        max_reprojection_rms = float(
            solver_config.get('max_reprojection_rms_px', 2.5)
        )
        if (
            isinstance(reprojection_rms, bool)
            or not isinstance(reprojection_rms, (int, float))
            or not math.isfinite(reprojection_rms)
            or reprojection_rms > max_reprojection_rms
        ):
            raise ValueError(
                "camera pose reprojection RMS is missing or exceeds its gate"
            )
        provenance = msg.get('provenance', {})
        if not isinstance(provenance, dict):
            raise ValueError("camera pose provenance must be a mapping")
        expected_grid_sha256 = getattr(self, 'marker_grid_sha256', None)
        if expected_grid_sha256 is not None:
            camera_grid_sha256 = provenance.get('grid_sha256')
            if (
                not isinstance(camera_grid_sha256, str)
                or len(camera_grid_sha256) != 64
                or camera_grid_sha256.lower() != expected_grid_sha256
            ):
                raise ValueError(
                    "camera pose grid SHA-256 does not match the active "
                    "marker-grid controller"
                )
        matches = msg.get('matches', [])
        if not isinstance(matches, list):
            raise ValueError("camera pose matches must be a list")
        rotation = msg.get('world_from_camera_rotation')
        if rotation is not None:
            if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
                raise ValueError(
                    "world_from_camera_rotation must be a 3x3 matrix"
                )
            rotation = [
                _finite_vector(
                    row, 3, f'world_from_camera_rotation[{index}]'
                )
                for index, row in enumerate(rotation)
            ]
        protocol_version = msg.get('protocol_version', 1)
        if (isinstance(protocol_version, bool) or
                not isinstance(protocol_version, int) or
                protocol_version != 1):
            raise ValueError(
                f"unsupported camera pose protocol version {protocol_version!r}"
            )
        return {
            'protocol_version': protocol_version,
            'run_id': msg.get('run_id'),
            'position_xyz': position,
            'yaw_rad': yaw,
            'position_offset_xyz': position_offset,
            'yaw_offset_rad': yaw_offset,
            'desired_position_xyz': list(desired_position),
            'desired_yaw_rad': desired_yaw,
            'quality': quality,
            'provenance': provenance,
            'world_from_camera_rotation': rotation,
            'matches': matches,
        }

    def _save_current_camera_pose(self, pose):
        """Save the normalized pose alongside this experiment's outputs."""
        if not getattr(self, 'tag', None):
            return
        pose_log_dir = BASE_DIR / 'logs' / self.tag
        pose_log_dir.mkdir(parents=True, exist_ok=True)
        with (pose_log_dir / 'camera_pose_estimate.json').open('w') as stream:
            json.dump(pose, stream, indent=2)

    def _save_camera_pose_overlay(self, message):
        """Keep the current frame's projected tile outlines with run logs."""
        if not getattr(self, 'tag', None):
            return
        if message.get('overlay_file') != CAMERA_POSE_OVERLAY_FILENAME:
            self.logger.warning('RGB camera pose did not include an overlay')
            return
        log_dir = BASE_DIR / 'logs' / self.tag
        log_dir.mkdir(parents=True, exist_ok=True)
        remote_work_dir = self.camera_cfg.get(
            'work_dir', self.common_cfg['work_dir']
        )
        remote = f"{remote_work_dir}/{CAMERA_POSE_OVERLAY_FILENAME}"
        local = log_dir / CAMERA_POSE_OVERLAY_FILENAME
        try:
            conn = Connection(
                host=self.camera_cfg['ip'], user=self.camera_cfg['user'],
                connect_timeout=10,
            )
            conn.get(remote, str(local))
            self.logger.info('Saved RGB camera pose overlay: %s', local)
        except Exception as error:
            self.logger.warning(
                'Could not save RGB camera pose overlay to experiment logs: %s',
                error,
            )

    def _save_latest_camera_pose(
            self, pose, *, saved_at=None, source_tag=None):
        """Atomically persist a validated physical-camera measurement."""
        if not getattr(self, 'tag', None):
            return
        CAMERA_POSE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        document = {
            'schema': 'fls-rgb-camera-pose-cache',
            'schema_version': 1,
            'saved_at': saved_at or datetime.now().astimezone().isoformat(),
            'source_tag': (
                source_tag if source_tag is not None else self.tag
            ),
            'pose': pose,
        }
        temporary = CAMERA_POSE_CACHE_FILE.with_suffix('.json.tmp')
        with temporary.open('w') as stream:
            json.dump(document, stream, indent=2)
        temporary.replace(CAMERA_POSE_CACHE_FILE)

    def _load_saved_camera_pose(self):
        """Load the latest cache, falling back to the newest per-run result."""
        if CAMERA_POSE_CACHE_FILE.exists():
            with CAMERA_POSE_CACHE_FILE.open() as stream:
                document = json.load(stream)
            if (
                not isinstance(document, dict)
                or document.get('schema') != 'fls-rgb-camera-pose-cache'
                or document.get('schema_version') != 1
                or not isinstance(document.get('pose'), dict)
            ):
                raise ValueError('saved RGB camera pose cache is malformed')
            return document

        candidates = sorted(
            CAMERA_POSE_CACHE_FILE.parent.glob(
                '*/camera_pose_estimate.json'
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        source = candidates[0]
        with source.open() as stream:
            pose = json.load(stream)
        if not isinstance(pose, dict):
            raise ValueError('saved per-run RGB camera pose is malformed')
        saved_at = datetime.fromtimestamp(
            source.stat().st_mtime
        ).astimezone().isoformat()
        return {
            'schema': 'fls-rgb-camera-pose-cache',
            'schema_version': 1,
            'saved_at': saved_at,
            'source_tag': source.parent.name,
            'pose': pose,
        }

    def _reuse_or_recompute_camera_pose(self):
        """Offer a saved measurement and return whether it was reused."""
        try:
            saved = self._load_saved_camera_pose()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.logger.warning(
                'Cannot reuse saved RGB camera pose; recomputing: %s', error
            )
            return False
        if saved is None:
            self.logger.info('No saved RGB camera pose found; recomputing.')
            return False

        cached_pose = saved['pose']
        position = cached_pose.get('position_xyz')
        yaw = cached_pose.get('yaw_rad')
        if getattr(self.args, 'skip_confirm', False):
            choice = 'reuse'
        else:
            while True:
                response = input(
                    "\n>>> Saved RGB camera pose\n"
                    f"    saved at: {saved.get('saved_at', 'unknown')}\n"
                    f"    source run: {saved.get('source_tag', 'unknown')}\n"
                    f"    measured XYZ: {position}\n"
                    f"    measured yaw: {yaw}\n"
                    ">>> Reuse this pose or recompute it? "
                    "[R/c] (default: reuse): "
                ).strip().lower()
                if response in {'', 'r', 'reuse', 'y', 'yes'}:
                    choice = 'reuse'
                    break
                if response in {'c', 'recompute', 'n', 'no'}:
                    choice = 'recompute'
                    break
                print("Please enter 'r' to reuse or 'c' to recompute.")
        if choice == 'recompute':
            self.logger.info('Operator requested a fresh RGB camera pose.')
            return False

        # Rebuild a protocol message from only the measured values. This
        # deliberately recomputes offsets against the current SFL camera
        # instead of reusing offsets derived for an older authored camera.
        message = {
            'protocol_version': cached_pose.get('protocol_version', 1),
            'run_id': self.camera_run_id,
            'id': 'CAM',
            'status': 'CAMERA_POSE',
            'position_xyz': position,
            'yaw_rad': yaw,
            'quality': cached_pose.get('quality', {}),
            'provenance': cached_pose.get('provenance', {}),
            'matches': cached_pose.get('matches', []),
            'world_from_camera_rotation': cached_pose.get(
                'world_from_camera_rotation'
            ),
        }
        try:
            pose = self._parse_camera_pose(message)
        except (TypeError, ValueError) as error:
            self.logger.warning(
                'Saved RGB camera pose is incompatible with this run; '
                'recomputing: %s', error
            )
            return False

        pose['reused'] = True
        pose['reused_from'] = {
            'saved_at': saved.get('saved_at'),
            'source_tag': saved.get('source_tag'),
        }
        self.camera_pose = pose
        self.manifest['camera_pose_estimate'] = pose
        for drone in self.drones:
            drone.pop('viewpoint', None)
        self._save_current_camera_pose(pose)
        # Promote a legacy per-run result to the dedicated latest cache.
        if not CAMERA_POSE_CACHE_FILE.exists():
            self._save_latest_camera_pose(
                pose,
                saved_at=saved.get('saved_at'),
                source_tag=saved.get('source_tag'),
            )
        self.logger.info(
            'Reusing RGB camera pose xyz=%s yaw=%.6f rad; recomputed swarm '
            'offset=%s yaw_offset=%.6f rad',
            pose['position_xyz'], pose['yaw_rad'],
            pose['position_offset_xyz'], pose['yaw_offset_rad'],
        )
        return True

    def _wait_for_camera_pose(self):
        """Wait for a validated PnP result before allowing any drone to boot."""
        timeout_s = self.camera_cfg['pose_estimation']['timeout_s']
        deadline = time.monotonic() + timeout_s
        self.logger.info(
            f"Waiting up to {timeout_s:g}s for RGB camera pose..."
        )
        while self.running.is_set():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"camera pose estimation timed out after {timeout_s:g}s"
                )
            try:
                msg = self.pull_socket.recv_json()
            except zmq.Again:
                continue

            if self._handle_marker_grid_event(msg):
                continue
            if msg.get('id') != 'CAM':
                self.logger.warning(
                    f"Ignoring preflight status from {msg.get('id')}: "
                    f"{msg.get('status')}"
                )
                continue
            if not self._camera_message_matches_run(msg):
                self.logger.warning(
                    "Ignoring stale camera status for run %s",
                    msg.get('run_id'),
                )
                continue
            status = msg.get('status')
            if status in CAMERA_POSE_FAILURE_STATUSES:
                detail = msg.get('error', 'camera node reported pose failure')
                raise RuntimeError(f"camera pose estimation failed: {detail}")
            if status == 'READY':
                self.camera_ready = True
                continue
            if status not in CAMERA_POSE_STATUSES:
                continue

            self.camera_pose = self._parse_camera_pose(msg)
            self.manifest['camera_pose_estimate'] = self.camera_pose
            for drone in self.drones:
                drone.pop('viewpoint', None)
            self._save_current_camera_pose(self.camera_pose)
            self._save_camera_pose_overlay(msg)
            self.logger.info(
                "RGB camera pose: xyz=%s yaw=%.6f rad; swarm offset=%s "
                "yaw_offset=%.6f rad; quality=%s",
                self.camera_pose['position_xyz'], self.camera_pose['yaw_rad'],
                self.camera_pose['position_offset_xyz'],
                self.camera_pose['yaw_offset_rad'],
                self.camera_pose['quality'],
            )
            return self.camera_pose

    def _confirm_camera_pose(self):
        """Require operator approval of the pose before any drone boots."""
        pose = getattr(self, 'camera_pose', None)
        if pose is None:
            return
        if getattr(self.args, 'skip_confirm', False):
            self._save_latest_camera_pose(pose)
            return

        position = pose['position_xyz']
        position_offset = pose['position_offset_xyz']
        yaw_degrees = math.degrees(pose['yaw_rad'])
        yaw_offset_degrees = math.degrees(pose['yaw_offset_rad'])
        prompt = (
            "\n>>> RGB camera pose preflight\n"
            f"    measured XYZ: {position}\n"
            f"    measured yaw: {pose['yaw_rad']:.6f} rad "
            f"({yaw_degrees:.3f} deg)\n"
            f"    swarm XYZ offset: {position_offset}\n"
            f"    swarm yaw offset: {pose['yaw_offset_rad']:.6f} rad "
            f"({yaw_offset_degrees:.3f} deg)\n"
            ">>> Apply this transform and boot the swarm? [y/N]: "
        )
        response = input(prompt).strip().lower()
        if response not in {'y', 'yes'}:
            raise RuntimeError("RGB camera pose was not confirmed by operator")
        self._save_latest_camera_pose(pose)
        self.logger.info("RGB camera pose confirmed by operator.")

