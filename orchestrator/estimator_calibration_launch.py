"""Interactive motors-off IMU calibration, separate from swarm flight startup."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import uuid

import yaml


OPTIONS = ('drone_id', 'imu_firmware_id', 'imu_fixture_id', 'imu_reference_note',
           'imu_duration_s', 'imu_settle_s')
ARTIFACTS = ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md',
             'fit/calibration.json')


def validate_estimator_calibration_options(args):
    enabled = getattr(args, 'calibrate_estimator_imu', False)
    if not enabled:
        if any(getattr(args, field, None) is not None for field in OPTIONS):
            raise ValueError('--drone-id and --imu-* options require --calibrate-estimator-imu')
        return
    conflicts = (
        'interaction', 'calibrate', 'braking_test', 'mpc', 'baseline', 'hover',
        'active_septic_brake_test', 'active_septic_brake_calibration',
        'illumination', 'intractable_illumination', 'morphing', 'sense',
        'ground', 'droneless', 'record', 'loadcell', 'blender', 'skip_confirm',
        'targeted_braking_calibration', 'vicon_rigidbody_position_only',
        'vicon_source_metadata', 'test_marker_grid',
    )
    if (any(getattr(args, field, False) for field in conflicts)
            or any(getattr(args, field, None) is not None for field in (
                'adaptive_braking_calibration', 'braking_test_direction',
                'braking_test_repetitions', 'contact_attitude_run', 'kill', 'off'))):
        raise ValueError('--calibrate-estimator-imu is a standalone motors-off mode; '
                         'do not combine it with flight, sensor, shutdown or skip-confirm options')
    for field, low in (('imu_duration_s', 3), ('imu_settle_s', 1)):
        value = getattr(args, field, None)
        if value is not None and (not math.isfinite(value) or not low <= value <= 30):
            raise ValueError(f'--{field.replace("_", "-")} must be between {low} and 30 seconds')
    drone_id = getattr(args, 'drone_id', None)
    if drone_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', drone_id):
        raise ValueError('--drone-id must be a simple manifest ID, not a path')


def select_drone(manifest, args, base_dir):
    drones = manifest.get('drones', [])
    requested = getattr(args, 'drone_id', None)
    if requested is None and len(drones) > 1:
        # Read identity only; never create SwarmOrchestrator or invoke Dispatcher.
        controller = manifest.get('controller', {})
        files = controller.get('mission_files') or [controller.get('mission_file')]
        identities = set()
        for filename in files:
            if not filename:
                continue
            mission_path = Path(controller.get('mission_path', '.')) / filename
            if not mission_path.is_absolute():
                mission_path = Path(base_dir) / mission_path
            with mission_path.open() as stream:
                mission = yaml.safe_load(stream)
            identities.update((mission or {}).get('drones', {}))
        if len(identities) == 1:
            requested = next(iter(identities))
    selected = [d for d in drones if requested is None or d['id'] == requested]
    if len(selected) != 1:
        raise ValueError('IMU calibration requires one aircraft; select --drone-id from the manifest')
    drone = selected[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', str(drone['id'])):
        raise ValueError('unsafe manifest drone ID')
    for field in ('ip', 'user'):
        if not isinstance(drone.get(field), str) or not drone[field].strip():
            raise ValueError(f'drone {drone["id"]} needs manifest {field}')
    return drone


def calibration_command(manifest, drone, args, session, provenance):
    common = manifest['common']
    work = PurePosixPath(common['work_dir'])
    python = PurePosixPath(common['venv_path']) / 'bin' / 'python'
    if not work.is_absolute() or not python.is_absolute():
        raise ValueError('manifest work_dir and venv_path must be absolute remote paths')
    uri = 'usb://0'
    if getattr(args, 'radio', False):
        uri = drone.get('uri', '')
        if not isinstance(uri, str) or not uri.startswith('radio://'):
            raise ValueError('--radio requires the selected aircraft radio:// URI in the manifest')
    relative = PurePosixPath('Interaction/estimator_calibrations') / drone['id'] / session
    tokens = [str(python), '-u', '-m', 'Interaction.calibrate_estimator_imu', 'collect',
              '--uri', uri, '--drone-id', drone['id'],
              '--firmware-id', provenance['firmware_id'],
              '--fixture-id', provenance['fixture_id'],
              '--reference-note', provenance['reference_note'],
              '--output', str(relative),
              '--duration-s', str(getattr(args, 'imu_duration_s', None) or 4.),
              '--settle-s', str(getattr(args, 'imu_settle_s', None) or 2.)]
    return f'cd {shlex.quote(str(work))} && {shlex.join(tokens)}', work / relative


class Tee:
    def __init__(self, console, transcript):
        self.console, self.transcript = console, transcript

    def write(self, text):
        self.console.write(text)
        self.transcript.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.transcript.flush()


def run_estimator_calibration(args, manifest_file, base_dir, *, connection_factory=None,
                              prompt=None, session=None):
    """Forward the live terminal; retrieve results even when fitting fails."""
    validate_estimator_calibration_options(args)
    with Path(manifest_file).open() as stream:
        manifest = yaml.safe_load(stream)
    drone = select_drone(manifest, args, base_dir)
    prompt = input if prompt is None else prompt
    provenance = {}
    labels = {
        'firmware_id': 'Actual flashed firmware build/tag (not offboard revision)',
        'fixture_id': 'Identity of the independently checked body-aligned fixture',
        'reference_note': 'How the fixture/body-axis alignment was independently checked',
    }
    for key, label in labels.items():
        value = getattr(args, 'imu_' + key, None)
        if value is None:
            if prompt is input and not sys.stdin.isatty():
                raise ValueError('interactive terminal required; use --imu-' + key.replace('_', '-'))
            value = prompt(label + ': ')
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} cannot be empty')
        provenance[key] = value.strip()
    session = session or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                          + '_' + uuid.uuid4().hex[:8])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session):
        raise ValueError('invalid calibration session identity')
    command, remote_dir = calibration_command(manifest, drone, args, session, provenance)
    local_dir = Path(base_dir) / 'logs' / f'estimator_imu_{drone["id"]}_{session}'
    local_dir.mkdir(parents=True, exist_ok=False)
    result = {'mode': 'calibrate_estimator_imu', 'drone_id': drone['id'],
              'provenance': provenance, 'remote_dir': str(remote_dir),
              'exit_code': None, 'downloaded': [], 'missing': [], 'download_errors': {},
              'firmware_applied': False, 'flight_started': False}
    print(f'IMU fixture calibration: {drone["id"]} at {drone["ip"]}. Remove props.')
    print(f'Artifacts: {local_dir}', flush=True)
    if connection_factory is None:
        from fabric import Connection
        connection_factory = Connection
    try:
        with connection_factory(host=drone['ip'], user=drone['user'], connect_timeout=5) as connection:
            with (local_dir / 'terminal.txt').open('w') as transcript:
                try:
                    remote_result = connection.run(command, pty=True, hide=False, warn=True,
                                                   in_stream=sys.stdin,
                                                   out_stream=Tee(sys.stdout, transcript))
                    result['exit_code'] = remote_result.exited
                except (KeyboardInterrupt, Exception) as error:
                    result['error'] = str(error) or type(error).__name__
                    result['exit_code'] = 130 if isinstance(error, KeyboardInterrupt) else 1
                finally:
                    # This dedicated session never joins the flight upload queue.
                    for name in ARTIFACTS:
                        local = local_dir / name
                        local.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            connection.get(str(remote_dir / name), local=str(local))
                            result['downloaded'].append(name)
                        except FileNotFoundError:
                            result['missing'].append(name)
                        except Exception as error:
                            result['download_errors'][name] = str(error)
    except (KeyboardInterrupt, Exception) as error:
        result['error'] = str(error) or type(error).__name__
        result['exit_code'] = 130 if isinstance(error, KeyboardInterrupt) else 1
    code = result['exit_code'] if type(result['exit_code']) is int else 1
    if code == 0:
        if result['missing'] or result['download_errors']:
            code = 1
            result['error'] = 'calibration returned success but some artifacts were not retrieved'
        else:
            try:
                fit = json.loads((local_dir / 'fit/calibration.json').read_text())
                if (fit.get('schema') != 'estimator_processed_imu_calibration_v1'
                        or fit.get('accepted') is not True or fit.get('drone_id') != drone['id']
                        or fit.get('firmware_id') != provenance['firmware_id']):
                    raise ValueError('downloaded calibration not accepted or belongs to another aircraft')
                checksum = hashlib.sha256((local_dir / 'dataset.json').read_bytes()).hexdigest()
                if fit.get('dataset_sha256') != checksum:
                    raise ValueError('downloaded dataset does not match the calibration fingerprint')
            except (ValueError, OSError) as error:
                code = 1
                result['error'] = str(error)
    result['launcher_exit_code'] = code
    (local_dir / 'launch_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Calibration {"PASSED" if code == 0 else "STOPPED/REJECTED"}. Results: {local_dir}')
    if result.get('error'):
        print(result['error'], file=sys.stderr)
    if result['download_errors']:
        print('Some downloads failed; original session remains on the Pi:', remote_dir, file=sys.stderr)
    print('Saved only; no firmware calibration has been applied.')
    return code
