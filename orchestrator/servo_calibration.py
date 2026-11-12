"""Bounded front-camera servo calibration at the first illumination handshake."""
import json
import math
import time
import uuid
import copy
import os
import stat
import tempfile
from pathlib import Path

import zmq
import yaml


DEFAULTS = dict(enabled=False, camera_id='CAM', timeout_s=60.,
                command_timeout_s=3., measurement_timeout_s=5., settle_s=.4,
                max_iterations=10, tolerance_deg=.75, gain=.7, max_step_deg=3.,
                max_correction_deg=15., stall_iterations=3,
                min_improvement_deg=.25, consensus_frames=5, max_spread_deg=1.,
                confirmation_measurements=2, verification_passes=1,
                min_color_pixels=6, min_span_px=8., association_radius_px=80.,
                led_pitch_m=.00624, failure_policy='abort')


def save_manifest_results(path, entries, report, config):
    """Patch only calibration fields, preserving comments and unrelated edits."""
    path = Path(path)
    original = path.read_text()
    document = yaml.safe_load(original)
    tree = yaml.compose(original)
    drones_node = next(value for key, value in tree.value if key.value == 'drones')
    expected = copy.deepcopy(document)
    by_id = {drone['id']: drone for drone in expected['drones']}
    nodes = {}
    for node in drones_node.value:
        fields = {key.value: (key, value) for key, value in node.value}
        name = yaml.safe_load(original[fields['id'][1].start_mark.index:fields['id'][1].end_mark.index])
        if name in nodes:
            raise ValueError(f'duplicate manifest drone {name}')
        nodes[name] = fields
    edits = []
    for entry in entries:
        name = entry['id']
        drone, fields = by_id[name], nodes[name]
        existing = drone.get('servo_offsets', [0., 0.])
        if existing != entry['initial_offsets']:
            raise RuntimeError(f'{name}: servo offsets were edited during calibration; refusing to overwrite them')
        result = report['drones'].get(name, {})
        rods = result.get('rods', [])
        eligible = (report['completed'] and not report.get('error') and not report.get('cleanup_error')
                    and result.get('verification_pass') == config['verification_passes'])
        offsets = list(existing)
        for index, rod in enumerate(rods):
            if eligible and rod['status'] == 'converged':
                offsets[index] = entry['offsets'][index]
        metadata = dict(tag=report['tag'], verified=eligible and len(rods) == 2 and
                        all(rod['status'] == 'converged' for rod in rods),
                        rods=rods, saved_offsets=offsets)
        if report.get('error') or report.get('cleanup_error'):
            metadata['error'] = report.get('cleanup_error') or report['error']
        additions = []
        for field, value in (('servo_offsets', offsets), ('servo_calibration_result', metadata)):
            drone[field] = value
            text = yaml.safe_dump(value, default_flow_style=True, width=100000, sort_keys=False).strip()
            if field in fields:
                node = fields[field][1]
                if isinstance(node, (yaml.SequenceNode, yaml.MappingNode)) and not node.flow_style:
                    text = ' '*max(0, fields[field][0].start_mark.column+2-node.start_mark.column)+text
                start = node.start_mark.index
                end = start + len(original[start:node.end_mark.index].rstrip())
                edits.append((start, end, text))
            else:
                indent = fields['id'][0].start_mark.column
                additions.append(' '*indent+field+': '+text+'\n')
        if additions:
            anchor = original.find('\n', fields['id'][1].end_mark.index)
            if anchor < 0:
                anchor = len(original)
                insertion = '\n'+''.join(additions)
            else:
                anchor += 1
                insertion = ''.join(additions)
            edits.append((anchor, anchor, insertion))
    updated = original
    for start, end, text in sorted(edits, reverse=True):
        updated = updated[:start]+text+updated[end:]
    if yaml.safe_load(updated) != expected:
        raise ValueError('manifest layout cannot be safely patched; original retained')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        if path.read_text() != original:
            raise RuntimeError('manifest changed while saving; original retained')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {entry['id']: by_id[entry['id']]['servo_offsets'] for entry in entries}


def configure(manifest, args, cameras, missions):
    raw = manifest.get('servo_calibration', {})
    if not isinstance(raw, dict):
        raise ValueError('servo_calibration must be a mapping')
    config = dict(DEFAULTS, **raw)
    if (getattr(args, 'off', None) is not None or getattr(args, 'kill', None) is not None
            or getattr(args, 'test_marker_grid', False)):
        return dict(config, enabled=False)
    if getattr(args, 'calibrate_servos', False):
        config['enabled'] = True
    if not isinstance(config['enabled'], bool):
        raise ValueError('servo_calibration.enabled must be boolean')
    if not config['enabled']:
        return config
    unknown = set(config) - set(DEFAULTS) - {'calibration_file'}
    if unknown:
        raise ValueError(f'unknown servo_calibration options: {sorted(unknown)}')
    if (not getattr(args, 'illumination', False) or
            any(getattr(args, flag, False) for flag in ('record', 'skip_record', 'morphing', 'radio', 'interaction'))):
        raise ValueError('servo calibration requires illumination with camera recording enabled')
    if not manifest.get('mission', {}).get('require_handshake'):
        raise ValueError('servo calibration requires mission.require_handshake')
    camera = next((c for c in cameras if c.get('id', 'CAM') == config['camera_id']), None)
    if camera is None:
        raise ValueError('servo calibration camera_id must select an enabled front camera')
    config.setdefault('calibration_file', camera.get('pose_estimation', {}).get('calibration_file'))
    if not isinstance(config['calibration_file'], str) or not config['calibration_file'].strip():
        raise ValueError('servo_calibration.calibration_file must be a path on the front camera')
    if config['failure_policy'] not in ('abort', 'continue'):
        raise ValueError('servo calibration failure_policy must be abort or continue')
    for field in set(DEFAULTS) - {'enabled', 'camera_id', 'failure_policy'}:
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'servo_calibration.{field} must be positive and finite')
    for field in ('max_iterations', 'stall_iterations', 'consensus_frames', 'min_color_pixels', 'confirmation_measurements', 'verification_passes'):
        if not isinstance(config[field], int):
            raise ValueError(f'servo_calibration.{field} must be an integer')
    if config['gain'] > 1 or config['max_correction_deg'] > 45:
        raise ValueError('servo calibration gain must be <= 1 and max_correction_deg <= 45')
    if not missions:
        raise ValueError('servo calibration requires an SFL mission')
    for mission in missions:
        camera_xyz = mission.get('camera')
        if (not isinstance(camera_xyz, (list, tuple)) or len(camera_xyz) != 3 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in camera_xyz)):
            raise ValueError('servo calibration requires a finite SFL camera position')
        look_at = mission.get('camera_look_at', [0., 0., camera_xyz[2]])
        if (len(look_at) != 3 or any(not math.isfinite(v) for v in look_at) or
                math.hypot(look_at[0]-camera_xyz[0], look_at[1]-camera_xyz[1]) < 1e-6):
            raise ValueError('servo calibration requires a nonvertical SFL camera look-at')
    # Only launched mission drones participate. Validate their entire batch
    # before network setup/takeoff.
    participants(manifest, missions, config)
    manifest['servo_calibration'] = config
    return config


def participants(manifest, missions, config):
    entries = []
    for drone in manifest.get('drones', []):
        name = drone['id']
        setting = missions[0].get('drones', {}).get(name)
        if setting is None:
            continue
        if (drone.get('servo_count', 2) != 2 or drone.get('led_count', 50) < 8 or
                drone.get('led_count', 50) % 2 or drone.get('type') not in ('H', 'V')):
            raise ValueError(f'{name}: calibration requires two H/V servos and an even LED count >= 8')
        if setting.get('relative_anchor') or setting.get('follow'):
            raise ValueError(f'{name}: calibration requires a stationary absolute SFL target')
        rows = []
        for mission in missions:
            track = mission.get('drones', {}).get(name, {}).get('servos', [])
            if not track:
                raise ValueError(f'{name}: calibration requires explicit SFL servo samples')
            rows.extend(track)
        offsets = drone.get('servo_offsets', [0., 0.])
        for pair in [offsets, *rows]:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pair)):
                raise ValueError(f'{name}: invalid SFL servo samples/offsets')
        base = [0., 180.] if drone['type'] == 'H' else [90., 270.]
        limits = [(max(lo-min(row[i] for row in rows), offsets[i]-config['max_correction_deg']),
                   min(lo+180-max(row[i] for row in rows), offsets[i]+config['max_correction_deg']))
                  for i, lo in enumerate(base)]
        if any(not low <= offset <= high for offset, (low, high) in zip(offsets, limits)):
            raise ValueError(f'{name}: existing servo offsets exceed SFL hardware limits')
        target = setting.get('target')
        if (not isinstance(target, (list, tuple)) or len(target) < 3 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in target)):
            raise ValueError(f'{name}: invalid SFL target')
        entries.append(dict(id=name, angles=list(setting['servos'][0]), offsets=list(offsets),
                            initial_offsets=list(offsets), limits=limits, setting=setting, drone=drone))
    if not entries:
        raise ValueError('servo calibration has no participating drones')
    return entries


class RodAdjustment:
    def __init__(self, target, offset, limits, config):
        self.target, self.offset, self.limits, self.config = target, offset, limits, config
        self.status = 'pending'
        self.best_error = float('inf')
        self.stalls = 0
        self.measured = None
        self.confirmations = 0

    def observe(self, measured):
        if measured is None:
            self.measured = None
            self.confirmations = 0
            if self.status == 'converged':
                self.status = 'pending'
            return
        if isinstance(measured, bool) or not isinstance(measured, (float, int)) or not math.isfinite(measured):
            raise ValueError('invalid camera servo angle')
        self.measured = measured
        if self.status in ('stalled', 'limit_reached'):
            return
        if self.status == 'converged':
            self.status = 'pending'
        error = (self.target-measured+180.) % 360.-180.
        magnitude = abs(error)
        if magnitude <= self.config['tolerance_deg']:
            self.confirmations += 1
            self.stalls = 0
            self.best_error = float('inf')
            if self.confirmations >= self.config['confirmation_measurements']:
                self.status = 'converged'
            return
        self.confirmations = 0
        if magnitude < self.best_error-self.config['min_improvement_deg']:
            self.best_error, self.stalls = magnitude, 0
        else:
            self.stalls += 1
        if self.stalls >= self.config['stall_iterations']:
            self.status = 'stalled'
            return
        step = max(-self.config['max_step_deg'], min(self.config['max_step_deg'], self.config['gain']*error))
        proposed = max(self.limits[0], min(self.limits[1], self.offset+step))
        if abs(proposed-self.offset) < 1e-6:
            self.status = 'limit_reached'
        else:
            self.offset = proposed


class ServoCalibrationMixin:
    def _configure_servo_calibration(self):
        self.servo_calibration_cfg = configure(self.manifest, self.args, self._active_cameras(), self.missions)

    def _servo_calibration_enabled(self):
        return getattr(self, 'servo_calibration_cfg', {}).get('enabled', False)

    def _servo_exchange(self, command, expected_id, status, deadline):
        request_id = uuid.uuid4().hex
        command = dict(command, request_id=request_id)
        next_send = 0.
        while self.running.is_set() and not self.emergency:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f'servo calibration timed out waiting for {expected_id}: {status}')
            if now >= next_send:
                self.pub_socket.send_json(command)
                next_send = now+.5
            if not self.pull_socket.poll(100):
                continue
            try:
                message = self.pull_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            if self._handle_marker_grid_event(message):
                continue
            # Preserve normal camera failure handling throughout calibration.
            self._handle_camera_runtime_event(message)
            if message.get('id') != expected_id or message.get('request_id') != request_id:
                continue
            if 'run_id' in command and message.get('run_id') != command['run_id']:
                continue
            if 'tag' in command and message.get('tag') != command['tag']:
                continue
            if message.get('status') in ('SERVO_APPLY_FAILED', 'SERVO_CAMERA_FAILED'):
                raise RuntimeError(message.get('error', 'servo calibration device failure'))
            if message.get('status') == status:
                return message
        raise RuntimeError('servo calibration interrupted')

    def _apply_servo_calibration(self, entry, deadline, *, selected=False, finish=False):
        message = self._servo_exchange(dict(cmd='SERVO_CALIBRATE', tag=self.tag, mission_index=0,
            drone_id=entry['id'], angles=entry['angles'], offsets=entry['offsets'],
            selected=selected, finish=finish), entry['id'], 'SERVO_APPLIED',
            min(deadline, time.monotonic()+self.servo_calibration_cfg['command_timeout_s']))
        if message.get('offsets') != entry['offsets']:
            raise RuntimeError(f"{entry['id']}: calibration applied unexpected offsets")

    def _calibrate_illumination_servos(self):
        if not self._servo_calibration_enabled():
            return
        config = self.servo_calibration_cfg
        entries = participants(self.manifest, self.missions, config)
        camera = next(c for c in self._active_cameras() if self._camera_id(c) == config['camera_id'])
        camera_command = dict(run_id=camera['run_id'])
        deadline = time.monotonic()+config['timeout_s']
        report = dict(tag=self.tag, mission_index=0, config=config, drones={}, completed=False)
        error = None
        self.logger.info('Calibrating rod servos from front camera before illumination START')
        try:
            for entry in entries:
                self._apply_servo_calibration(entry, deadline)
            for pass_index in range(config['verification_passes']+1):
                for entry in entries:
                    rods = [RodAdjustment(a, o, limits, config)
                            for a, o, limits in zip(entry['angles'], entry['offsets'], entry['limits'])]
                    result = report['drones'].setdefault(entry['id'], dict(initial_offsets=entry['initial_offsets'], observations=[]))
                    setting = entry['setting']
                    mission = self.missions[0]
                    reference = dict(camera=mission['camera'],
                        look_at=mission.get('camera_look_at', [0., 0., mission['camera'][2]]),
                        target=setting['target'], yaw=setting['target'][3] if len(setting['target']) > 3 else 0.,
                        light_module_offset=setting.get('light_module_offset', [0., 0., 0.]),
                        angles=entry['angles'], led_count=entry['drone'].get('led_count', 50))
                    for iteration in range(config['max_iterations']):
                        self._apply_servo_calibration(entry, deadline, selected=True)
                        settle_deadline = min(deadline, time.monotonic()+config['settle_s'])
                        while time.monotonic() < settle_deadline:
                            if not self.running.is_set() or self.emergency:
                                raise RuntimeError('servo calibration interrupted')
                            time.sleep(.02)
                        self.servo_calibration_recording_camera_id = config['camera_id']
                        report['footage'] = dict(camera_id=config['camera_id'],
                            video_file=f"{config['camera_id']}_{self.tag}_servo_calibration.mp4",
                            frame_index_file=f"{config['camera_id']}_{self.tag}_servo_calibration.frames.jsonl")
                        reply = self._servo_exchange(dict(camera_command, cmd='MEASURE_SERVOS', reference=reference),
                            config['camera_id'], 'SERVO_MEASURED',
                            min(deadline, time.monotonic()+config['measurement_timeout_s']+.75))
                        measured = reply.get('angles')
                        if not isinstance(measured, list) or len(measured) != 2:
                            raise ValueError('camera did not report both servo observations')
                        for rod, angle in zip(rods, measured):
                            rod.observe(angle)
                        result['observations'].append(dict(iteration=iteration+1, pass_index=pass_index,
                            phase='initial' if pass_index == 0 else 'verification',
                            commanded_offsets=list(entry['offsets']), measured_deg=measured,
                            status=[rod.status for rod in rods], diagnostics=reply.get('diagnostics', {})))
                        entry['offsets'] = [rod.offset for rod in rods]
                        if all(rod.status != 'pending' for rod in rods):
                            break
                    for i, rod in enumerate(rods):
                        if rod.status == 'pending':
                            rod.status = 'not_detected' if rod.measured is None else 'iteration_limit'
                        if rod.status != 'converged':
                            entry['offsets'][i] = entry['initial_offsets'][i]
                    result.update(offsets=list(entry['offsets']), iterations=result.get('iterations', 0)+iteration+1, verification_pass=pass_index,
                                  rods=[dict(status=rod.status, measured_deg=rod.measured, target_deg=rod.target,
                                       error_deg=None if rod.measured is None else (rod.target-rod.measured+180.)%360.-180.,
                                       confirmations=rod.confirmations) for rod in rods])
                    self.logger.info('Servo calibration %s: %s', entry['id'], result)
                    self._apply_servo_calibration(entry, deadline)
                if any(rod['status'] != 'converged' for result in report['drones'].values()
                       for rod in result.get('rods', [])):
                    break
            report['completed'] = True
        except (Exception, KeyboardInterrupt) as caught:
            error = caught
            report['error'] = str(caught)
        finally:
            # Cleanup has its own finite budget. A missing device/cleanup ACK
            # always aborts: never START with a possibly active test pattern.
            cleanup_error = None
            cleanup_deadline = time.monotonic()+config['command_timeout_s']*(len(entries)+1)
            if self.running.is_set() and not self.emergency:
                for entry in entries:
                    if error is not None:
                        entry['offsets'] = list(entry['initial_offsets'])
                    try:
                        self._apply_servo_calibration(entry, cleanup_deadline, finish=True)
                    except Exception as caught:
                        cleanup_error = caught
                try:
                    self._servo_exchange(dict(camera_command, cmd='STOP_SERVO_CAMERA'), config['camera_id'],
                        'SERVO_CAMERA_STOPPED', min(cleanup_deadline, time.monotonic()+config['command_timeout_s']))
                except Exception as caught:
                    cleanup_error = caught
            directory = Path(__file__).resolve().parent / 'logs' / self.tag
            directory.mkdir(parents=True, exist_ok=True)
            report['applied_offsets'] = {entry['id']: entry['offsets'] for entry in entries}
            if cleanup_error:
                report['cleanup_error'] = str(cleanup_error)
            if not self.running.is_set() or self.emergency:
                report.setdefault('error', 'calibration interrupted before cleanup completed')
            try:
                saved = save_manifest_results(getattr(self, 'manifest_path',
                    Path(__file__).resolve().parent/'swarm_manifest.yaml'), entries, report, config)
                report['manifest_saved_offsets'] = saved
                for drone in self.manifest.get('drones', []):
                    if drone['id'] in saved:
                        drone['servo_offsets'] = list(saved[drone['id']])
                self.logger.info('Saved calibration results to swarm manifest: %s', saved)
            except Exception as caught:
                report['manifest_save_error'] = str(caught)
                self.logger.error('Could not save servo calibration results to manifest: %s', caught)
            (directory/'servo_calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            if cleanup_error:
                raise RuntimeError('servo calibration cleanup failed; illumination will not start') from cleanup_error
        if isinstance(error, KeyboardInterrupt):
            raise error
        if not self.running.is_set() or self.emergency:
            raise RuntimeError('servo calibration interrupted; illumination will not start') from error
        failed = error is not None or any(rod['status'] != 'converged'
            for result in report['drones'].values() for rod in result.get('rods', []))
        if failed:
            if config['failure_policy'] == 'abort':
                raise RuntimeError('servo calibration failed; see servo_calibration.json') from error
            self.logger.warning('Continuing after incomplete servo calibration; failed offsets were restored')
